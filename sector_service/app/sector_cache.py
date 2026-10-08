"""產業族群清單快取（Application 層）。

證交所 ISIN 清單頁約 9 MB、回應速度波動很大（曾觀察到超過 30 秒），
若在使用者請求當下才同步下載，容易讓 ASP.NET 端逾時而顯示錯誤。因此：

1. 快取未過期：直接回傳
2. 快取已過期：先回傳舊資料，同時在背景更新（stale-while-revalidate）
3. 完全沒有資料：先讀硬碟快取檔；沒有檔案才同步下載
4. 每次成功更新都寫入硬碟，服務重新啟動後可立即使用
"""
import json
import logging
import os
import threading
import time
from dataclasses import asdict
from typing import Callable, List, Optional

from .market_data import Sector

logger = logging.getLogger(__name__)

SectorLoader = Callable[[], List[Sector]]


class SectorCache:
    def __init__(self, ttl_seconds: int, loader: SectorLoader, cache_file: Optional[str] = None):
        self._ttl_seconds = ttl_seconds
        self._loader = loader
        self._cache_file = cache_file
        self._sectors: List[Sector] = []
        self._loaded_at = 0.0             # time.time()，以便與硬碟檔案時間比較
        self._lock = threading.Lock()     # 保護 _sectors / _loaded_at
        self._refresh_lock = threading.Lock()   # 同一時間只允許一個下載
        self._refresh_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ 公開介面
    def get(self) -> List[Sector]:
        """取得族群清單；只有在「記憶體與硬碟都沒有資料」時才會同步等待下載。"""
        with self._lock:
            sectors, loaded_at = self._sectors, self._loaded_at
        if not sectors:
            self._load_from_disk()
            with self._lock:
                sectors, loaded_at = self._sectors, self._loaded_at
        if not sectors:
            return self.refresh()
        if self._is_expired(loaded_at):
            self.refresh_in_background()
        return sectors

    def refresh(self) -> List[Sector]:
        """同步下載最新清單；失敗時保留原本的資料。"""
        with self._refresh_lock:
            started = time.monotonic()
            sectors = self._loader()
            if not sectors:
                logger.warning("族群清單更新失敗，沿用既有資料（%d 個族群）", len(self._sectors))
                with self._lock:
                    return self._sectors
            with self._lock:
                self._sectors = sectors
                self._loaded_at = time.time()
            logger.info("族群清單已更新：%d 個族群，耗時 %.1f 秒", len(sectors), time.monotonic() - started)
            self._save_to_disk(sectors)
            return sectors

    def refresh_in_background(self) -> None:
        """在背景執行緒更新；已有更新進行中時不重複啟動。"""
        with self._lock:
            if self._refresh_thread is not None and self._refresh_thread.is_alive():
                return
            self._refresh_thread = threading.Thread(target=self._safe_refresh, name="sector-refresh", daemon=True)
            self._refresh_thread.start()

    def warm_up(self) -> None:
        """服務啟動時呼叫：載入硬碟快取，並視需要在背景更新。"""
        self._load_from_disk()
        with self._lock:
            needs_refresh = not self._sectors or self._is_expired(self._loaded_at)
        if needs_refresh:
            self.refresh_in_background()

    # ------------------------------------------------------------------ 內部
    def _is_expired(self, loaded_at: float) -> bool:
        return time.time() - loaded_at > self._ttl_seconds

    def _safe_refresh(self) -> None:
        try:
            self.refresh()
        except Exception:
            logger.exception("背景更新族群清單失敗")

    def _load_from_disk(self) -> None:
        if not self._cache_file or not os.path.exists(self._cache_file):
            return
        try:
            with open(self._cache_file, encoding="utf-8") as file:
                payload = json.load(file)
            sectors = [Sector(**item) for item in payload["sectors"]]
            loaded_at = float(payload["loaded_at"])
        except Exception:
            logger.exception("讀取族群快取檔失敗：%s", self._cache_file)
            return
        with self._lock:
            # 記憶體已有較新的資料時不覆蓋
            if sectors and loaded_at > self._loaded_at:
                self._sectors, self._loaded_at = sectors, loaded_at

    def _save_to_disk(self, sectors: List[Sector]) -> None:
        if not self._cache_file:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._cache_file)), exist_ok=True)
            temporary_file = self._cache_file + ".tmp"
            with open(temporary_file, "w", encoding="utf-8") as file:
                json.dump({"loaded_at": time.time(), "sectors": [asdict(s) for s in sectors]}, file, ensure_ascii=False)
            # 先寫暫存檔再取代，避免寫到一半時被讀取
            os.replace(temporary_file, self._cache_file)
        except Exception:
            logger.exception("寫入族群快取檔失敗：%s", self._cache_file)
