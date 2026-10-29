"""公開資訊觀測站（新版 MOPS）「近期發布之重大訊息」。

新版 MOPS（https://mops.twse.com.tw/mops/#/...）為單頁式應用程式，資料由 JSON API 提供（已實測確認）：
- 公司摘要：POST https://mops.twse.com.tw/mops/api/t146sb05      {"companyId": "3073"}
  result.recent_important_news.data = [[民國日期, 主旨, {"apiName": "t05st01_detail", "parameters": {...}}], ...]（最近 10 筆）
- 重大訊息內容：POST https://mops.twse.com.tw/mops/api/t05st01_detail
  {"marketKind": "otc", "companyId": "3073", "enterDate": "1151013", "serialNumber": "2"}

依最小揭露原則，不回傳發言人姓名與電話。
"""
import logging
import re
import threading
import time
from datetime import date
from typing import Callable, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

MOPS_API_BASE = "https://mops.twse.com.tw/mops/api/"
MOPS_PAGE_BASE = "https://mops.twse.com.tw/mops/#/web/"
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Content-Type": "application/json",
    "Referer": "https://mops.twse.com.tw/mops/",
}
REQUEST_TIMEOUT_SECONDS = 20
MAX_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 1.0
NEWS_CACHE_SECONDS = 10 * 60          # 重大訊息清單 10 分鐘內使用快取
DETAIL_CACHE_SECONDS = 24 * 60 * 60   # 已發布的訊息內容不會變動
REQUEST_INTERVAL_SECONDS = 0.3        # 查詢多家公司時，每次請求間隔，避免對 MOPS 造成負擔

COMPANY_ID_PATTERN = re.compile(r"^[0-9A-Z]{4,6}$")
MARKET_KIND_PATTERN = re.compile(r"^[a-z]{2,5}$")
ENTER_DATE_PATTERN = re.compile(r"^\d{7}$")         # 民國 yyyMMdd，例如 1151013
SERIAL_NUMBER_PATTERN = re.compile(r"^\d{1,3}$")

# 重大訊息內容（t05st01_detail）各欄位的位置；發言人姓名 / 職稱 / 電話（3~5）不回傳
DETAIL_INDEX = {"serial": 0, "speak_date": 1, "speak_time": 2, "subject": 6, "clause": 7, "fact_date": 8, "description": 9}
MARKET_NAMES = {"sii": "上市", "otc": "上櫃", "rotc": "興櫃", "pub": "公開發行"}


class MopsApiError(Exception):
    """MOPS 呼叫失敗；is_client_error 表示查詢條件有誤（例如公司代號不存在）。"""

    def __init__(self, message: str, is_client_error: bool = False):
        super().__init__(message)
        self.is_client_error = is_client_error


def roc_to_iso(roc_date: str) -> Optional[str]:
    """民國日期轉西元：'115/10/14' 或 '1151014' → '2026-10-14'；格式不符時回傳 None。"""
    text = (roc_date or "").strip()
    match = re.fullmatch(r"(\d{2,3})/(\d{1,2})/(\d{1,2})", text) or re.fullmatch(r"(\d{3})(\d{2})(\d{2})", text)
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    try:
        return date(year + 1911, month, day).isoformat()
    except ValueError:
        return None


def one_month_before(today: date) -> date:
    """往前推一個月的同一天；該月沒有這一天時取月底（例如 3/31 → 2/28）。"""
    year, month = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    for day in (today.day, 30, 29, 28):
        try:
            return date(year, month, min(today.day, day))
        except ValueError:
            continue
    return date(year, month, 28)


def filter_recent_items(news: Dict, since: date) -> Dict:
    """只保留發言日期在 since（含）之後的重大訊息；回傳新的 dict，不修改快取中的原始資料。"""
    since_text = since.isoformat()
    items = [item for item in news["items"] if item["date"] and item["date"] >= since_text]
    return dict(news, items=items)


def _clean_text(value) -> str:
    """去除 MOPS 欄位常見的尾端空白與 \\r\\n 斷行（主旨會被硬斷行）。"""
    return re.sub(r"[ \t]+\n", "\n", str(value or "").replace("\r\n", "\n")).strip()


def _one_line(value) -> str:
    return re.sub(r"\s*\n\s*", "", _clean_text(value))


class MopsClient:
    def __init__(self, http_post: Optional[Callable[..., requests.Response]] = None):
        self._http_post = http_post or requests.post
        self._cache: Dict[Tuple, Tuple[float, Dict]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ 公開介面
    def get_recent_news(self, company_id: str) -> Dict:
        """單一公司的公司名稱與最近 10 筆重大訊息。"""
        if not COMPANY_ID_PATTERN.match(company_id):
            raise MopsApiError("公司代號格式錯誤", is_client_error=True)
        cache_key = ("news", company_id)
        cached = self._get_cached(cache_key, NEWS_CACHE_SECONDS)
        if cached is not None:
            return cached

        result = self._post("t146sb05", {"companyId": company_id})
        basic = result.get("basic_info") or {}
        news_block = result.get("recent_important_news") or {}
        items = []
        for row in news_block.get("data") or []:
            if len(row) < 3 or not isinstance(row[2], dict):
                continue
            parameters = row[2].get("parameters") or {}
            items.append({
                "date": roc_to_iso(row[0]),
                "rocDate": str(row[0]).strip(),
                "subject": _one_line(row[1]),
                "detail": {
                    "marketKind": parameters.get("marketKind", ""),
                    "enterDate": parameters.get("enterDate", ""),
                    "serialNumber": str(parameters.get("serialNumber", "")),
                },
            })
        market_kind = items[0]["detail"]["marketKind"] if items else ""
        news = {
            "companyId": company_id,
            "name": basic.get("companyAbbreviation") or "",
            "industry": basic.get("industryCategory") or "",
            "market": MARKET_NAMES.get(market_kind, ""),
            "items": items,
            "sourceUrl": f"{MOPS_PAGE_BASE}t146sb05?companyId={company_id}",
        }
        self._set_cached(cache_key, news)
        return news

    def get_news_for_companies(self, company_ids: List[str], since: Optional[date] = None) -> Dict:
        """多家公司：各自查詢，單一公司失敗不影響其他公司；指定 since 時只保留該日（含）之後的訊息。"""
        companies, errors = [], []
        for index, company_id in enumerate(dict.fromkeys(company_ids)):
            cached = self._get_cached(("news", company_id), NEWS_CACHE_SECONDS)
            if cached is None and index > 0:
                time.sleep(REQUEST_INTERVAL_SECONDS)
            try:
                news = cached if cached is not None else self.get_recent_news(company_id)
            except MopsApiError as error:
                errors.append({"companyId": company_id, "message": str(error)})
                continue
            companies.append(filter_recent_items(news, since) if since else news)
        return {"companies": companies, "errors": errors, "sinceDate": since.isoformat() if since else None}

    def get_news_detail(self, company_id: str, market_kind: str, enter_date: str, serial_number: str) -> Dict:
        """單則重大訊息的完整內容。"""
        if not (COMPANY_ID_PATTERN.match(company_id) and MARKET_KIND_PATTERN.match(market_kind)
                and ENTER_DATE_PATTERN.match(enter_date) and SERIAL_NUMBER_PATTERN.match(serial_number)):
            raise MopsApiError("查詢參數格式錯誤", is_client_error=True)
        cache_key = ("detail", company_id, market_kind, enter_date, serial_number)
        cached = self._get_cached(cache_key, DETAIL_CACHE_SECONDS)
        if cached is not None:
            return cached

        result = self._post("t05st01_detail", {
            "marketKind": market_kind, "companyId": company_id,
            "enterDate": enter_date, "serialNumber": serial_number,
        })
        rows = result.get("data") or []
        if not rows or len(rows[0]) <= DETAIL_INDEX["description"]:
            raise MopsApiError("查無此重大訊息內容", is_client_error=True)
        row = rows[0]
        detail = {
            "companyId": company_id,
            "name": result.get("companyAbbreviation") or "",
            "market": result.get("marketName") or MARKET_NAMES.get(market_kind, ""),
            "speakDate": roc_to_iso(row[DETAIL_INDEX["speak_date"]]),
            "speakTime": _clean_text(row[DETAIL_INDEX["speak_time"]]),
            "subject": _one_line(row[DETAIL_INDEX["subject"]]),
            "clause": _clean_text(row[DETAIL_INDEX["clause"]]),
            "factDate": roc_to_iso(row[DETAIL_INDEX["fact_date"]]),
            "description": _clean_text(row[DETAIL_INDEX["description"]]),
        }
        self._set_cached(cache_key, detail)
        return detail

    # ------------------------------------------------------------------ 內部
    def _post(self, api_name: str, payload: Dict) -> Dict:
        response = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._http_post(MOPS_API_BASE + api_name, json=payload,
                                           headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
            except requests.RequestException as error:
                logger.warning("MOPS %s 連線失敗（第 %d 次）：%s", api_name, attempt, type(error).__name__)
                if attempt == MAX_ATTEMPTS:
                    raise MopsApiError(f"無法連線公開資訊觀測站（{type(error).__name__}）") from None
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            if response.status_code >= 500 and attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            break

        if response.status_code != 200:
            raise MopsApiError(f"公開資訊觀測站回應 HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError:
            raise MopsApiError("公開資訊觀測站回傳格式錯誤") from None
        # MOPS 以 body.code 表示查詢結果；例如公司代號不存在時為 500「公司代號格式錯誤」
        if body.get("code") != 200:
            raise MopsApiError(str(body.get("message") or "查詢失敗"), is_client_error=True)
        return body.get("result") or {}

    def _get_cached(self, key: Tuple, ttl_seconds: int) -> Optional[Dict]:
        with self._lock:
            entry = self._cache.get(key)
            if entry and time.monotonic() - entry[0] < ttl_seconds:
                return entry[1]
        return None

    def _set_cached(self, key: Tuple, value: Dict) -> None:
        with self._lock:
            self._cache[key] = (time.monotonic(), value)
            now = time.monotonic()
            expired = [k for k, (t, _) in self._cache.items() if now - t >= DETAIL_CACHE_SECONDS]
            for expired_key in expired:
                del self._cache[expired_key]
