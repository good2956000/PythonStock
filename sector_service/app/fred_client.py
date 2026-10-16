"""FRED（Federal Reserve Economic Data）API 用戶端（Infrastructure 層）。

API 文件：https://fred.stlouisfed.org/docs/api/fred/
金鑰一律由環境變數 FRED_API_KEY 提供，不得寫入程式碼或設定檔。

注意：requests 的例外訊息會包含完整網址（含 api_key 參數），
因此所有錯誤都轉成不含網址的 FredApiError，避免金鑰出現在 log 或 API 回應中。
"""
import logging
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred/"
PAGE_LIMIT = 1000                 # release/dates 單次最多 1000 筆
DEFAULT_CACHE_SECONDS = 60 * 60   # 發布日程與最新數值一小時內變動極少
STALE_FALLBACK_SECONDS = 24 * 60 * 60   # FRED 暫時無法連線時，可沿用 24 小時內的舊資料
REQUEST_TIMEOUT_SECONDS = 20
MAX_ATTEMPTS = 2                  # FRED 偶爾回應逾時（實測曾發生），失敗時重試一次
RETRY_DELAY_SECONDS = 1.0
MAX_PAGES = 20                    # 防止異常回應造成無限分頁


class FredApiError(Exception):
    """FRED 呼叫失敗；訊息不含網址與金鑰。"""


class FredClient:
    def __init__(
        self,
        api_key: str,
        cache_seconds: int = DEFAULT_CACHE_SECONDS,
        http_get: Optional[Callable[..., requests.Response]] = None,
    ):
        if not api_key:
            raise ValueError("FRED API key is required")
        self._api_key = api_key
        self._cache_seconds = cache_seconds
        self._http_get = http_get or requests.get
        self._cache: Dict[Tuple, Tuple[float, object]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ 公開介面
    def get_release_dates(self, start_date: str, end_date: str) -> List[Dict]:
        """取得期間內所有 release 的發布日期（含尚未發布的排程），自動分頁。"""
        return self._cached_call(("release_dates", start_date, end_date),
                                 lambda: self._fetch_release_dates(start_date, end_date))

    def get_observations(self, series_id: str, units: str = "lin", limit: int = 2) -> List[Dict]:
        """取得數列最新的 limit 筆觀測值（由新到舊）；units 為 FRED 的換算方式（lin / chg / pch / pc1）。"""
        return self._cached_call(("observations", series_id, units, limit),
                                 lambda: self._fetch_observations(series_id, units, limit))

    # ------------------------------------------------------------------ 內部
    def _cached_call(self, cache_key: Tuple, fetch: Callable[[], List[Dict]]) -> List[Dict]:
        """一小時內使用快取；重新抓取失敗時，若有 24 小時內的舊資料則沿用並記錄警告。"""
        cached, age = self._get_cached(cache_key)
        if cached is not None and age < self._cache_seconds:
            return cached
        try:
            value = fetch()
        except FredApiError:
            if cached is not None and age < STALE_FALLBACK_SECONDS:
                logger.warning("FRED 暫時無法取得 %s，沿用 %.0f 分鐘前的資料", cache_key[:2], age / 60)
                return cached
            raise
        self._set_cached(cache_key, value)
        return value

    def _fetch_release_dates(self, start_date: str, end_date: str) -> List[Dict]:
        dates: List[Dict] = []
        for page in range(MAX_PAGES):
            payload = self._request("releases/dates", {
                "realtime_start": start_date,
                "realtime_end": end_date,
                "include_release_dates_with_no_data": "true",
                "order_by": "release_date",
                "sort_order": "asc",
                "limit": PAGE_LIMIT,
                "offset": page * PAGE_LIMIT,
            })
            batch = payload.get("release_dates", [])
            dates.extend(batch)
            if len(dates) >= int(payload.get("count", 0)) or len(batch) < PAGE_LIMIT:
                break
        return dates

    def _fetch_observations(self, series_id: str, units: str, limit: int) -> List[Dict]:
        payload = self._request("series/observations", {
            "series_id": series_id,
            "units": units,
            "sort_order": "desc",
            "limit": limit,
        })
        return payload.get("observations", [])

    def _request(self, path: str, params: Dict) -> Dict:
        query = dict(params, api_key=self._api_key, file_type="json")
        response = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._http_get(FRED_BASE_URL + path, params=query, timeout=REQUEST_TIMEOUT_SECONDS)
            except requests.RequestException as error:
                # 只記錄例外類型，不記錄訊息（訊息含完整網址與金鑰）
                logger.warning("FRED %s 連線失敗（第 %d 次）：%s", path, attempt, type(error).__name__)
                if attempt == MAX_ATTEMPTS:
                    raise FredApiError(f"Unable to reach FRED ({type(error).__name__})") from None
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            # 伺服器端錯誤可重試；4xx（例如數列不存在、金鑰無效）重試也不會成功
            if response.status_code >= 500 and attempt < MAX_ATTEMPTS:
                logger.warning("FRED %s 回應 %s（第 %d 次），稍後重試", path, response.status_code, attempt)
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            break
        if response.status_code != 200:
            message = ""
            try:
                message = response.json().get("error_message", "")
            except ValueError:
                pass
            logger.warning("FRED %s 回應 %s：%s", path, response.status_code, message)
            raise FredApiError(f"FRED returned {response.status_code}: {message}".strip(": "))
        try:
            return response.json()
        except ValueError:
            raise FredApiError("FRED returned invalid JSON") from None

    def _get_cached(self, key: Tuple) -> Tuple[Optional[object], float]:
        """回傳（快取值, 已存在秒數）；沒有快取時回傳 (None, inf)。"""
        with self._lock:
            entry = self._cache.get(key)
        if entry is None:
            return None, float("inf")
        return entry[1], time.monotonic() - entry[0]

    def _set_cached(self, key: Tuple, value) -> None:
        with self._lock:
            self._cache[key] = (time.monotonic(), value)
            # 保留到可作為備援的期限為止，之後才移除
            expired = [k for k, (t, _) in self._cache.items() if time.monotonic() - t >= STALE_FALLBACK_SECONDS]
            for expired_key in expired:
                del self._cache[expired_key]
