import json
import os
import threading
from datetime import datetime, timedelta

import pytest

from app.optimize_jobs import (
    STATUS_CANCELLED, STATUS_COMPLETED, STATUS_FAILED, STATUS_RUNNING, TAIPEI_TIMEZONE,
    OptimizeJobConflictError, OptimizeJobManager,
)


class FakeClock:
    """每次呼叫前進 10 秒，讓每個工作的完成時間（結果編號）不同。"""

    def __init__(self):
        self.now = datetime(2026, 11, 9, 14, 0, 0, tzinfo=TAIPEI_TIMEZONE)

    def __call__(self):
        self.now += timedelta(seconds=10)
        return self.now


def fake_result(train_return=12.5, mode="random"):
    return {"mode": mode, "seed": 42, "trials": 3, "totalCombinations": 3, "startDate": "2020-06-01", "endDate": "2026-11-06",
            "elapsedSeconds": 1.2, "top": [{"trainReturnPercent": train_return, "validationReturnPercent": 3.0, "fullReturnPercent": 15.9}]}


def three_step_runner(progress):
    for done in (1, 2, 3):
        progress(done, 3, {"trainReturnPercent": float(done), "validationReturnPercent": 0.0})
    return fake_result()


@pytest.fixture
def manager(tmp_path):
    return OptimizeJobManager(str(tmp_path / "kd_optimize"), results_kept=2, clock=FakeClock())


class TestOptimizeJobManager:
    def test_completed_job_is_saved_and_listed(self, manager, tmp_path):
        job = manager.submit("random", 3, {"startDate": "2020-06-01"}, three_step_runner, background=False)

        assert job["status"] == STATUS_COMPLETED
        assert (job["processedCount"], job["totalCount"], job["best"]["trainReturnPercent"]) == (3, 3, 3.0)
        assert job["result"]["top"][0]["trainReturnPercent"] == 12.5
        assert job["resultId"] == "20261109-140030-random"

        saved = manager.load_result(job["resultId"])
        assert saved["settings"] == {"startDate": "2020-06-01"}
        assert saved["result"] == job["result"]
        assert os.listdir(tmp_path / "kd_optimize") == ["20261109-140030-random.json"]   # 暫存檔已改名
        assert manager.list_results() == [{
            "resultId": "20261109-140030-random", "savedAt": "2026-11-09T14:00:30+08:00", "mode": "random", "seed": 42,
            "trials": 3, "totalCombinations": 3, "startDate": "2020-06-01", "endDate": "2026-11-06", "elapsedSeconds": 1.2,
            "best": {"trainReturnPercent": 12.5, "validationReturnPercent": 3.0, "fullReturnPercent": 15.9},
        }]

    def test_keeps_only_most_recent_results(self, manager):
        ids = [manager.submit("random", 1, {}, lambda progress, value=value: fake_result(value), background=False)["resultId"]
               for value in (1.0, 2.0, 3.0)]

        assert [item["resultId"] for item in manager.list_results()] == ids[:0:-1]   # 新到舊，最舊的一筆已刪除
        assert manager.load_result(ids[0]) is None

    def test_result_id_does_not_overwrite_existing_file(self, tmp_path):
        fixed = datetime(2026, 11, 9, 14, 0, 0, tzinfo=TAIPEI_TIMEZONE)
        manager = OptimizeJobManager(str(tmp_path), clock=lambda: fixed)

        first = manager.submit("random", 1, {}, lambda progress: fake_result(1.0), background=False)
        second = manager.submit("random", 1, {}, lambda progress: fake_result(2.0), background=False)

        assert (first["resultId"], second["resultId"]) == ("20261109-140000-random", "20261109-140001-random")
        assert manager.load_result(first["resultId"])["result"]["top"][0]["trainReturnPercent"] == 1.0

    def test_only_one_job_runs_at_a_time_and_can_be_cancelled(self, manager):
        started, release = threading.Event(), threading.Event()

        def blocking_runner(progress):
            started.set()
            release.wait(5)
            progress(1, 10, None)   # 已要求停止時，回報進度會中止搜尋
            return fake_result()

        job = manager.submit("exhaustive", 10, {}, blocking_runner)
        assert started.wait(5)
        assert manager.current()["jobId"] == job["jobId"]
        assert manager.current()["status"] == STATUS_RUNNING
        with pytest.raises(OptimizeJobConflictError):
            manager.submit("random", 1, {}, three_step_runner, background=False)

        manager.cancel(job["jobId"])
        release.set()
        for _ in range(100):
            if manager.get(job["jobId"])["status"] != STATUS_RUNNING:
                break
            threading.Event().wait(0.05)

        assert manager.get(job["jobId"])["status"] == STATUS_CANCELLED
        assert manager.current() is None
        assert manager.list_results() == []   # 停止的工作不存檔
        # 停止後可以再開始新的工作
        assert manager.submit("random", 3, {}, three_step_runner, background=False)["status"] == STATUS_COMPLETED

    def test_runner_error_marks_job_failed(self, manager):
        def broken_runner(progress):
            raise RuntimeError("無法下載股價資料")

        job = manager.submit("random", 3, {}, broken_runner, background=False)

        assert (job["status"], job["errorMessage"]) == (STATUS_FAILED, "無法下載股價資料")
        assert job["finishedAt"] is not None
        assert manager.list_results() == []

    def test_unknown_job_returns_none(self, manager):
        assert manager.get("0" * 32) is None
        assert manager.cancel("0" * 32) is None

    @pytest.mark.parametrize("result_id", ["../secrets", "20261109-140030-random/../x", "20261109-140030-genetic", ""])
    def test_load_result_rejects_unexpected_ids(self, manager, result_id):
        assert manager.load_result(result_id) is None

    def test_ignores_unrelated_and_broken_files(self, manager, tmp_path):
        directory = tmp_path / "kd_optimize"
        directory.mkdir()
        (directory / "notes.json").write_text("{}", encoding="utf-8")
        (directory / "20261101-090000-random.json").write_text("{broken", encoding="utf-8")
        (directory / "20261102-090000-random.json").write_text(json.dumps({"savedAt": "x", "result": fake_result()}), encoding="utf-8")

        assert [item["resultId"] for item in manager.list_results()] == ["20261102-090000-random"]

    def test_works_without_result_directory(self):
        manager = OptimizeJobManager(None, clock=FakeClock())

        job = manager.submit("random", 3, {}, three_step_runner, background=False)

        assert job["status"] == STATUS_COMPLETED
        assert job["resultId"] is None and job["result"] is not None
        assert manager.list_results() == []
