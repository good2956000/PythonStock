"""KD 波段策略參數最佳化的背景工作（Application 層）。

完整搜尋可能需要數小時，無法在單一 HTTP 請求內完成，因此採「建立工作 → 背景執行 → 前端輪詢進度」：
- 同一時間只執行一個工作，避免多人同時搜尋拖垮主機
- 關閉網頁不影響執行；重新開啟頁面可查詢目前工作的進度
- 完成的結果存成 JSON 檔（預設 cache/kd_optimize/），服務重新啟動後仍可查詢；只保留最近幾筆
"""
import json
import logging
import os
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_CANCELLED = "cancelled"
STATUS_FAILED = "failed"
ACTIVE_STATUSES = (STATUS_QUEUED, STATUS_RUNNING)

# 結果編號 = 完成時間（台北）+ 搜尋模式，同時也是檔名
RESULT_ID_PATTERN = r"^\d{8}-\d{6}-(random|exhaustive)$"
TAIPEI_TIMEZONE = timezone(timedelta(hours=8))
DEFAULT_RESULTS_KEPT = 20
JOBS_KEPT_IN_MEMORY = 10

# 執行函式：接收進度回呼 (已完成組數, 總組數, 目前最佳或 None)，回傳最佳化結果
ProgressCallback = Callable[[int, int, Optional[Dict]], None]
OptimizeRunner = Callable[[ProgressCallback], Dict]


class OptimizeJobConflictError(Exception):
    """已有最佳化工作在執行中。"""


class OptimizeJobCancelledError(Exception):
    """使用者停止了最佳化工作（由進度回呼丟出以中止搜尋）。"""


@dataclass
class OptimizeJob:
    job_id: str
    mode: str
    total_count: int
    settings: Dict
    status: str = STATUS_QUEUED
    processed_count: int = 0
    best: Optional[Dict] = None
    created_at: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error_message: Optional[str] = None
    result_id: Optional[str] = None
    result: Optional[Dict] = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self, include_result: bool = True) -> Dict:
        return {
            "jobId": self.job_id,
            "mode": self.mode,
            "status": self.status,
            "processedCount": self.processed_count,
            "totalCount": self.total_count,
            "best": self.best,
            "settings": self.settings,
            "createdAt": self.created_at,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "errorMessage": self.error_message,
            "resultId": self.result_id,
            "result": self.result if include_result else None,
        }


class OptimizeJobManager:
    def __init__(
        self,
        result_directory: Optional[str],
        results_kept: int = DEFAULT_RESULTS_KEPT,
        clock: Callable[[], datetime] = lambda: datetime.now(TAIPEI_TIMEZONE),
    ):
        self._result_directory = result_directory
        self._results_kept = max(1, results_kept)
        self._clock = clock
        self._jobs: "OrderedDict[str, OptimizeJob]" = OrderedDict()
        self._lock = threading.Lock()

    def submit(self, mode: str, total_count: int, settings: Dict, runner: OptimizeRunner, background: bool = True) -> Dict:
        """建立並開始執行工作；已有工作執行中時丟出 OptimizeJobConflictError。background=False 供測試同步執行。"""
        with self._lock:
            if any(job.status in ACTIVE_STATUSES for job in self._jobs.values()):
                raise OptimizeJobConflictError()
            job = OptimizeJob(job_id=uuid.uuid4().hex, mode=mode, total_count=total_count, settings=settings,
                              created_at=self._clock().isoformat())
            self._jobs[job.job_id] = job
            while len(self._jobs) > JOBS_KEPT_IN_MEMORY:
                self._jobs.popitem(last=False)
        if background:
            threading.Thread(target=self._run, args=(job, runner), name=f"kd-optimize-{job.job_id[:8]}", daemon=True).start()
        else:
            self._run(job, runner)
        return self.get(job.job_id)

    def get(self, job_id: str) -> Optional[Dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def current(self) -> Optional[Dict]:
        """執行中（或排隊中）的工作；沒有則回傳 None。"""
        with self._lock:
            active = [job for job in self._jobs.values() if job.status in ACTIVE_STATUSES]
            return active[-1].to_dict(include_result=False) if active else None

    def cancel(self, job_id: str) -> Optional[Dict]:
        """要求停止工作（搜尋會在下一次回報進度時中止）；工作不存在時回傳 None。"""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status in ACTIVE_STATUSES:
                job.cancel_event.set()
            return job.to_dict(include_result=False)

    def shutdown(self) -> None:
        with self._lock:
            for job in self._jobs.values():
                job.cancel_event.set()

    def list_results(self) -> List[Dict]:
        """已存檔的結果摘要（新到舊）。"""
        summaries = []
        for result_id in self._saved_result_ids():
            saved = self.load_result(result_id)
            if saved is None:
                continue
            result = saved.get("result") or {}
            top = result.get("top") or []
            summaries.append({
                "resultId": result_id,
                "savedAt": saved.get("savedAt"),
                "mode": result.get("mode"),
                "seed": result.get("seed"),
                "trials": result.get("trials"),
                "totalCombinations": result.get("totalCombinations"),
                "startDate": result.get("startDate"),
                "endDate": result.get("endDate"),
                "elapsedSeconds": result.get("elapsedSeconds"),
                "best": {key: top[0][key] for key in ("trainReturnPercent", "validationReturnPercent", "fullReturnPercent")}
                if top else None,
            })
        return summaries

    def load_result(self, result_id: str) -> Optional[Dict]:
        if not self._result_directory or not re.match(RESULT_ID_PATTERN, result_id):
            return None
        path = os.path.join(self._result_directory, result_id + ".json")
        try:
            with open(path, encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            logger.exception("無法讀取最佳化結果 %s", path)
            return None

    def _run(self, job: OptimizeJob, runner: OptimizeRunner) -> None:
        self._update(job, status=STATUS_RUNNING, started_at=self._clock().isoformat())

        def report(processed: int, total: int, best: Optional[Dict]) -> None:
            if job.cancel_event.is_set():
                raise OptimizeJobCancelledError()
            with self._lock:
                job.processed_count, job.total_count, job.best = processed, total, best

        try:
            result = runner(report)
        except OptimizeJobCancelledError:
            self._update(job, status=STATUS_CANCELLED, finished_at=self._clock().isoformat())
            return
        except Exception as error:  # 背景執行緒的例外必須記錄下來，否則前端只會一直等待
            logger.exception("最佳化工作 %s 失敗", job.job_id)
            self._update(job, status=STATUS_FAILED, finished_at=self._clock().isoformat(), error_message=str(error))
            return

        finished = self._clock()
        result_id = f"{finished:%Y%m%d-%H%M%S}-{job.mode}"
        while self.load_result(result_id) is not None:   # 同一秒完成兩個工作時往後順延，避免覆蓋
            finished += timedelta(seconds=1)
            result_id = f"{finished:%Y%m%d-%H%M%S}-{job.mode}"
        saved = self._save(result_id, {"resultId": result_id, "savedAt": finished.isoformat(), "settings": job.settings,
                                       "result": result})
        with self._lock:
            job.result = result
            job.result_id = result_id if saved else None
            job.status = STATUS_COMPLETED
            job.finished_at = finished.isoformat()

    def _update(self, job: OptimizeJob, **changes) -> None:
        with self._lock:
            for name, value in changes.items():
                setattr(job, name, value)

    def _save(self, result_id: str, content: Dict) -> bool:
        """寫入結果檔（先寫暫存檔再改名，避免寫到一半被讀取）並刪除較舊的結果；失敗時只記錄，不影響工作結果。"""
        if not self._result_directory:
            return False
        try:
            os.makedirs(self._result_directory, exist_ok=True)
            path = os.path.join(self._result_directory, result_id + ".json")
            temporary_path = path + ".tmp"
            with open(temporary_path, "w", encoding="utf-8") as file:
                json.dump(content, file, ensure_ascii=False)
            os.replace(temporary_path, path)
            for old_id in self._saved_result_ids()[self._results_kept:]:
                os.remove(os.path.join(self._result_directory, old_id + ".json"))
            return True
        except OSError:
            logger.exception("無法儲存最佳化結果 %s", result_id)
            return False

    def _saved_result_ids(self) -> List[str]:
        """已存檔的結果編號（新到舊；編號以完成時間開頭，依字串排序即依時間排序）。"""
        if not self._result_directory or not os.path.isdir(self._result_directory):
            return []
        names = [name[:-5] for name in os.listdir(self._result_directory) if name.endswith(".json")]
        return sorted((name for name in names if re.match(RESULT_ID_PATTERN, name)), reverse=True)
