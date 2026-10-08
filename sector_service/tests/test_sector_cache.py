import json
import threading
import time

import pytest
import requests

from app import market_data
from app.market_data import Sector
from app.sector_cache import SectorCache


def make_sectors(label="v1"):
    return [Sector("1", f"水泥工業-{label}", ["1101.TW"], {"1101.TW": "台泥"})]


class CountingLoader:
    """記錄被呼叫次數的假下載函式，可設定回傳值或延遲。"""

    def __init__(self, results, delay_seconds=0.0):
        self._results = list(results)
        self.calls = 0
        self.delay_seconds = delay_seconds

    def __call__(self):
        self.calls += 1
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        return self._results.pop(0) if self._results else []


def wait_until(predicate, timeout_seconds=3.0):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class TestSectorCache:
    def test_first_call_without_any_data_downloads_synchronously(self):
        loader = CountingLoader([make_sectors()])
        cache = SectorCache(ttl_seconds=3600, loader=loader)

        assert cache.get()[0].name == "水泥工業-v1"
        assert cache.get()[0].name == "水泥工業-v1"
        assert loader.calls == 1   # 第二次直接使用快取

    def test_expired_cache_returns_stale_data_immediately_and_refreshes_in_background(self):
        # 這是本次修正的核心：證交所很慢時，使用者不需等待下載
        loader = CountingLoader([make_sectors("v1"), make_sectors("v2")], delay_seconds=0)
        cache = SectorCache(ttl_seconds=0, loader=loader)
        cache.get()
        loader.delay_seconds = 0.5

        started = time.monotonic()
        stale = cache.get()

        assert time.monotonic() - started < 0.2
        assert stale[0].name == "水泥工業-v1"
        assert wait_until(lambda: cache.get()[0].name == "水泥工業-v2")

    def test_failed_refresh_keeps_previous_data(self):
        loader = CountingLoader([make_sectors("v1"), []])
        cache = SectorCache(ttl_seconds=3600, loader=loader)
        cache.get()

        assert cache.refresh()[0].name == "水泥工業-v1"

    def test_concurrent_background_refresh_runs_only_once(self):
        loader = CountingLoader([make_sectors("v1"), make_sectors("v2"), make_sectors("v3")])
        cache = SectorCache(ttl_seconds=0, loader=loader)
        cache.get()
        loader.delay_seconds = 0.3

        threads = [threading.Thread(target=cache.get) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert wait_until(lambda: loader.calls == 2)
        time.sleep(0.4)

        assert loader.calls == 2

    def test_disk_cache_is_used_after_restart_without_downloading(self, tmp_path):
        cache_file = str(tmp_path / "cache" / "sectors.json")
        SectorCache(3600, CountingLoader([make_sectors("disk")]), cache_file).get()

        # 模擬服務重新啟動：新的快取物件、下載函式若被呼叫會回傳空清單
        restarted_loader = CountingLoader([])
        restarted = SectorCache(3600, restarted_loader, cache_file)

        sectors = restarted.get()
        assert sectors[0].name == "水泥工業-disk"
        assert sectors[0].stock_names == {"1101.TW": "台泥"}
        assert restarted_loader.calls == 0

    def test_warm_up_loads_disk_and_refreshes_expired_data_in_background(self, tmp_path):
        cache_file = tmp_path / "sectors.json"
        cache_file.write_text(
            json.dumps({"loaded_at": time.time() - 99999, "sectors": [
                {"sector_id": "1", "name": "舊資料", "tickers": [], "stock_names": {}}
            ]}, ensure_ascii=False),
            encoding="utf-8",
        )
        loader = CountingLoader([make_sectors("new")])
        cache = SectorCache(ttl_seconds=3600, loader=loader, cache_file=str(cache_file))

        cache.warm_up()

        assert wait_until(lambda: cache.get()[0].name == "水泥工業-new")

    def test_corrupted_disk_cache_falls_back_to_download(self, tmp_path):
        cache_file = tmp_path / "sectors.json"
        cache_file.write_text("{not json", encoding="utf-8")
        cache = SectorCache(3600, CountingLoader([make_sectors()]), str(cache_file))

        assert cache.get()[0].name == "水泥工業-v1"


class FakeStreamingResponse:
    """模擬伺服器以極慢速度持續傳送資料（每段之間不超過 read timeout）。"""

    def __init__(self, chunk_count, delay_seconds):
        self._chunk_count = chunk_count
        self._delay_seconds = delay_seconds

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for _ in range(self._chunk_count):
            time.sleep(self._delay_seconds)
            yield "台".encode("big5")


class TestDownloadDeadline:
    def test_slow_trickle_download_is_aborted_by_total_deadline(self, monkeypatch):
        monkeypatch.setattr(requests, "get", lambda *a, **k: FakeStreamingResponse(chunk_count=100, delay_seconds=0.05))

        with pytest.raises(TimeoutError):
            market_data.download_text_with_deadline("https://example", "big5", read_timeout_seconds=1, total_timeout_seconds=0.2)

    def test_normal_download_is_decoded_with_big5(self, monkeypatch):
        monkeypatch.setattr(requests, "get", lambda *a, **k: FakeStreamingResponse(chunk_count=2, delay_seconds=0))

        assert market_data.download_text_with_deadline("https://example", "big5", 1, 5) == "台台"

    def test_one_market_failure_returns_empty_list_instead_of_partial(self, monkeypatch):
        def fake_download(url, *args):
            if "strMode=2" in url:
                raise TimeoutError("listed market too slow")
            return "<table></table>"

        monkeypatch.setattr(market_data, "download_text_with_deadline", fake_download)

        assert market_data.fetch_sectors() == []
