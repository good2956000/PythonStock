"""大盤席勒本益比的資料來源與背景更新（Infrastructure 層）。

- 個股收盤價與本益比：證交所每日收盤行情 MI_INDEX（2005-09 起提供本益比）
- 個股發行股數：證交所外資及陸資投資持股統計 MI_QFIIS
- 消費者物價指數：主計總處「消費者物價指數銜接表」（cpispl.xls，110 年 = 100）

首次需回補約 250 個月（每月 2 次請求，為避免被證交所封鎖，每次間隔數秒，約 25 分鐘），
因此改由背景執行緒更新，每月彙整結果存成 JSON；之後只需補抓新的月份。
"""
import io
import json
import logging
import os
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd
import requests

from .market_valuation import aggregate_market_earnings, parse_cpi_table, parse_number, select_target_dates

logger = logging.getLogger(__name__)

REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
QUOTES_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
SHARES_URL = "https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS"
CPI_URL = "https://ws.dgbas.gov.tw/001/Upload/463/relfile/10315/2726/cpispl.xls"
# 證交所自 2005-09 起提供個股本益比
DATA_START_DATE = pd.Timestamp("2005-09-01")
# 同一月份的最後幾個交易日都抓不到時才放棄（例如資料延後公布）
FALLBACK_TRADING_DAYS = 3
# 抓取失敗的日期在此時間內不再重試，避免每次開頁都重新呼叫證交所
RETRY_AFTER_SECONDS = 60 * 60
CPI_REFRESH_SECONDS = 24 * 60 * 60
REQUEST_TIMEOUT_SECONDS = 30

STATE_IDLE, STATE_RUNNING = "idle", "running"

Snapshot = Dict
SnapshotFetcher = Callable[[pd.Timestamp], Optional[Snapshot]]
CpiFetcher = Callable[[], pd.Series]


class TwseClient:
    """抓取證交所報表並彙整成大盤市值與盈餘；每次請求之間固定間隔，避免觸發證交所的流量限制。"""

    def __init__(self, request_delay_seconds: float, sleep: Callable[[float], None] = time.sleep):
        self._request_delay_seconds = request_delay_seconds
        self._sleep = sleep
        self._last_request_at = 0.0

    def fetch_snapshot(self, trade_date: pd.Timestamp) -> Optional[Snapshot]:
        """回傳 {marketCap, earnings, stockCount, profitableCount}；當日無資料或格式不符時回傳 None。"""
        date_key = trade_date.strftime("%Y%m%d")
        quotes_report = self._get_json(QUOTES_URL, {"date": date_key, "type": "ALLBUT0999", "response": "json"})
        if quotes_report is None:
            return None
        quotes = parse_quotes(quotes_report)
        if not quotes:
            return None
        shares_report = self._get_json(SHARES_URL, {"date": date_key, "selectType": "ALLBUT0999", "response": "json"})
        if shares_report is None:
            return None
        return aggregate_market_earnings(quotes, parse_shares(shares_report))

    def _get_json(self, url: str, params: Dict[str, str]) -> Optional[Dict]:
        wait = self._request_delay_seconds - (time.monotonic() - self._last_request_at)
        if wait > 0:
            self._sleep(wait)
        self._last_request_at = time.monotonic()
        try:
            response = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            report = response.json()
        except Exception:
            logger.warning("抓取證交所報表失敗：%s %s", url, params.get("date"), exc_info=True)
            return None
        return report if report.get("stat") == "OK" else None


def parse_quotes(report: Dict) -> Dict[str, tuple]:
    """MI_INDEX：找出含「收盤價」與「本益比」欄位的每日收盤行情表，回傳 代號 → (收盤價, 本益比)。"""
    for table in report.get("tables", []):
        fields = table.get("fields") or []
        if {"證券代號", "收盤價", "本益比"}.issubset(fields):
            code_at, close_at, pe_at = fields.index("證券代號"), fields.index("收盤價"), fields.index("本益比")
            return {
                str(row[code_at]).strip(): (parse_number(row[close_at]), parse_number(row[pe_at]))
                for row in table.get("data", [])
            }
    return {}


def parse_shares(report: Dict) -> Dict[str, Optional[float]]:
    """MI_QFIIS：回傳 代號 → 發行股數。"""
    fields = report.get("fields") or []
    if "證券代號" not in fields or "發行股數" not in fields:
        return {}
    code_at, shares_at = fields.index("證券代號"), fields.index("發行股數")
    return {str(row[code_at]).strip(): parse_number(row[shares_at]) for row in report.get("data", [])}


def fetch_cpi() -> pd.Series:
    """下載主計總處 CPI 銜接表（.xls），回傳月 CPI。"""
    response = requests.get(CPI_URL, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    table = pd.read_excel(io.BytesIO(response.content), sheet_name=0, header=None)
    return parse_cpi_table(table)


class MarketValuationStore:
    """每月快照與 CPI 的硬碟快取（JSON，寫入時先寫暫存檔再取代，避免中斷時檔案毀損）。"""

    def __init__(self, directory: str):
        self._snapshot_file = os.path.join(directory, "snapshots.json")
        self._cpi_file = os.path.join(directory, "cpi.json")
        self._directory = directory

    def load_snapshots(self) -> Dict[str, Snapshot]:
        return self._read(self._snapshot_file, {})

    def save_snapshots(self, snapshots: Dict[str, Snapshot]) -> None:
        self._write(self._snapshot_file, dict(sorted(snapshots.items())))

    def load_cpi(self) -> Optional[Dict]:
        """回傳 {"fetchedAt": epoch 秒, "values": {"YYYY-MM": 指數}}；沒有快取時回傳 None。"""
        return self._read(self._cpi_file, None)

    def save_cpi(self, cpi: pd.Series, fetched_at: float) -> None:
        self._write(self._cpi_file, {"fetchedAt": fetched_at, "values": {str(month): value for month, value in cpi.items()}})

    def _read(self, path: str, default):
        try:
            with open(path, encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            return default
        except (OSError, ValueError):
            logger.warning("讀取快取失敗，忽略：%s", path, exc_info=True)
            return default

    def _write(self, path: str, content) -> None:
        os.makedirs(self._directory, exist_ok=True)
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(content, file, ensure_ascii=False)
        os.replace(temporary, path)


def cpi_from_cache(cached: Dict) -> pd.Series:
    values = {pd.Period(month, freq="M"): float(value) for month, value in cached["values"].items()}
    return pd.Series(values).sort_index()


class MarketValuationService:
    """管理每月快照的背景更新；同一時間只有一個更新執行緒。"""

    def __init__(
        self,
        store: MarketValuationStore,
        snapshot_fetcher: SnapshotFetcher,
        cpi_fetcher: CpiFetcher = fetch_cpi,
        clock: Callable[[], float] = time.time,
    ):
        self._store = store
        self._snapshot_fetcher = snapshot_fetcher
        self._cpi_fetcher = cpi_fetcher
        self._clock = clock
        self._lock = threading.Lock()
        self._snapshots: Dict[str, Snapshot] = store.load_snapshots()
        self._cpi: Optional[pd.Series] = None
        self._cpi_fetched_at = 0.0
        cached_cpi = store.load_cpi()
        if cached_cpi:
            self._cpi, self._cpi_fetched_at = cpi_from_cache(cached_cpi), float(cached_cpi.get("fetchedAt", 0))
        self._failed_at: Dict[str, float] = {}
        self._state = STATE_IDLE
        self._processed = self._total = 0
        self._last_error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def snapshots(self) -> Dict[str, Snapshot]:
        with self._lock:
            return dict(self._snapshots)

    def cpi(self) -> Optional[pd.Series]:
        with self._lock:
            return self._cpi

    def progress(self) -> Dict:
        with self._lock:
            return {
                "state": self._state,
                "processedMonths": self._processed,
                "totalMonths": self._total,
                "storedMonths": len(self._snapshots),
                "lastError": self._last_error,
            }

    def pending_months(self, trade_dates: Sequence[pd.Timestamp]) -> List[List[pd.Timestamp]]:
        """尚未取得的月份；每個月份為依序嘗試的候選交易日（最後一個交易日優先）。"""
        dates = pd.DatetimeIndex(trade_dates)
        dates = dates[dates >= DATA_START_DATE].sort_values()
        now = self._clock()
        with self._lock:
            # 每個月只存一筆：月份 → 已存的交易日
            stored_by_month = {key[:7]: key for key in self._snapshots}
            failed_at = dict(self._failed_at)
        pending = []
        for target in select_target_dates(dates):
            target_key = target.strftime("%Y-%m-%d")
            # 已有該月最後交易日的資料即完成；最新月份若只有較早的交易日資料，需改抓新的最新交易日
            if stored_by_month.get(target_key[:7], "") >= target_key:
                continue
            if now - failed_at.get(target_key, float("-inf")) < RETRY_AFTER_SECONDS:
                continue
            in_month = dates[(dates.year == target.year) & (dates.month == target.month)][::-1][:FALLBACK_TRADING_DAYS]
            pending.append(list(in_month))
        return pending

    def update_in_background(self, trade_dates: Sequence[pd.Timestamp]) -> bool:
        """有缺漏的月份或 CPI 需要更新時啟動背景更新；已在執行或無需更新時回傳 False。"""
        pending = self.pending_months(trade_dates)
        needs_cpi = self._cpi is None or self._clock() - self._cpi_fetched_at >= CPI_REFRESH_SECONDS
        with self._lock:
            if self._state == STATE_RUNNING or (not pending and not needs_cpi):
                return False
            self._state, self._processed, self._total, self._last_error = STATE_RUNNING, 0, len(pending), None
            self._thread = threading.Thread(target=self._run, args=(pending, needs_cpi), name="market-valuation", daemon=True)
            self._thread.start()
        return True

    def wait(self, timeout: Optional[float] = None) -> None:
        """測試用：等待背景更新結束。"""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def shutdown(self) -> None:
        self._stop.set()

    def _run(self, pending: List[List[pd.Timestamp]], needs_cpi: bool) -> None:
        try:
            if needs_cpi:
                self._refresh_cpi()
            for candidates in pending:
                if self._stop.is_set():
                    return
                self._fetch_month(candidates)
                with self._lock:
                    self._processed += 1
        except Exception as error:   # 背景執行緒不可讓例外中斷服務，記錄後結束
            logger.exception("大盤席勒本益比資料更新失敗")
            with self._lock:
                self._last_error = str(error)
        finally:
            with self._lock:
                self._state = STATE_IDLE

    def _refresh_cpi(self) -> None:
        try:
            cpi = self._cpi_fetcher()
        except Exception as error:
            # 沿用舊的 CPI 快取；完全沒有快取時才顯示錯誤
            logger.warning("下載 CPI 失敗", exc_info=True)
            with self._lock:
                if self._cpi is None:
                    self._last_error = f"無法下載主計總處消費者物價指數：{error}"
            return
        fetched_at = self._clock()
        self._store.save_cpi(cpi, fetched_at)
        with self._lock:
            self._cpi, self._cpi_fetched_at = cpi, fetched_at

    def _fetch_month(self, candidates: List[pd.Timestamp]) -> None:
        for trade_date in candidates:
            snapshot = self._snapshot_fetcher(trade_date)
            if snapshot is not None:
                self._save_snapshot(trade_date, snapshot)
                return
        if candidates:
            with self._lock:
                self._failed_at[candidates[0].strftime("%Y-%m-%d")] = self._clock()

    def _save_snapshot(self, trade_date: pd.Timestamp, snapshot: Snapshot) -> None:
        key = trade_date.strftime("%Y-%m-%d")
        with self._lock:
            # 每個月只保留最後一個交易日的資料（最新月份更新到新的交易日時，移除同月較早的資料）
            for old_key in [k for k in self._snapshots if k[:7] == key[:7] and k != key]:
                del self._snapshots[old_key]
            self._snapshots[key] = snapshot
            snapshots = dict(self._snapshots)
        self._store.save_snapshots(snapshots)


def build_snapshot_frame(snapshots: Dict[str, Snapshot], index_close: pd.Series) -> pd.DataFrame:
    """快照加上當日加權指數收盤；找不到指數收盤的日期略過。"""
    frame = pd.DataFrame.from_dict(snapshots, orient="index")
    if frame.empty:
        return frame
    frame.index = pd.to_datetime(frame.index)
    closes = index_close.copy()
    closes.index = pd.to_datetime(closes.index).normalize()
    frame["indexClose"] = closes.reindex(frame.index)
    return frame.dropna(subset=["indexClose"]).sort_index()
