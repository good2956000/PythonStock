"""外部資料來源（Infrastructure 層）：證交所產業分類清單、Yahoo Finance 日 K。"""
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from io import StringIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
ISIN_URL_TEMPLATE = "https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"

# strMode=2 為上市（Yahoo 代號後綴 .TW），strMode=4 為上櫃（後綴 .TWO）
MARKET_TARGETS = [("TW", "2"), ("TWO", "4")]

PRICE_REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass
class Sector:
    sector_id: str
    name: str
    tickers: List[str] = field(default_factory=list)
    # ticker → 股票中文名稱（例如 "2330.TW" → "台積電"）；固定清單為空，名稱由產業族群查得
    stock_names: Dict[str, str] = field(default_factory=dict)


def split_code_and_name(value: str) -> Tuple[str, str]:
    """把 ISIN 頁面的「2330　台積電」拆成 ("2330", "台積電")；沒有名稱時名稱為空字串。"""
    parts = re.split(r"[\u3000 ]+", str(value).strip(), maxsplit=1)
    code = parts[0].strip()
    name = parts[1].strip() if len(parts) > 1 else ""
    return code, name


def build_stock_name_map(sectors: List[Sector]) -> Dict[str, str]:
    """合併所有族群的 ticker → 名稱對照表。"""
    names: Dict[str, str] = {}
    for sector in sectors:
        names.update(sector.stock_names)
    return names


def download_text_with_deadline(url: str, encoding: str, read_timeout_seconds: int, total_timeout_seconds: int) -> str:
    """下載網頁文字，並限制「總下載時間」。

    requests 的 timeout 只限制連線與「兩段資料之間」的等待時間，
    伺服器以極慢速度持續傳送時不會逾時（證交所 ISIN 頁面約 9 MB，曾觀察到超過 30 秒），
    因此以串流方式讀取並自行檢查總時間，超過即中止。
    """
    deadline = time.monotonic() + total_timeout_seconds
    chunks = []
    with requests.get(url, headers=REQUEST_HEADERS, timeout=read_timeout_seconds, stream=True) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=256 * 1024):
            chunks.append(chunk)
            if time.monotonic() > deadline:
                raise TimeoutError(f"Download exceeded {total_timeout_seconds}s: {url}")
    # 與 requests 的 response.text 相同，無法解碼的位元組以替代字元處理
    return b"".join(chunks).decode(encoding, errors="replace")


def fetch_sectors(read_timeout_seconds: int = 15, total_timeout_seconds: int = 90) -> List[Sector]:
    """從證交所 ISIN 頁面抓取上市、上櫃普通股並依產業別分組。

    上市與上櫃中同名的產業會合併為同一族群（與原腳本行為一致）。
    任一市場抓取失敗即回傳空清單，避免把不完整的清單當成有效資料快取起來。
    """
    sectors_by_name: Dict[str, Sector] = {}

    for suffix, mode in MARKET_TARGETS:
        url = ISIN_URL_TEMPLATE.format(mode=mode)
        try:
            html = download_text_with_deadline(url, "big5", read_timeout_seconds, total_timeout_seconds)
            frame = pd.read_html(StringIO(html))[0]
        except Exception:
            logger.exception("抓取 %s 股票清單失敗，本次不更新族群清單", suffix)
            return []

        frame.columns = frame.iloc[0]
        frame = frame.iloc[2:]
        frame = frame.dropna(subset=["產業別"])
        code_and_name = frame["有價證券代號及名稱"].apply(split_code_and_name)
        frame["代號"] = code_and_name.str[0]
        frame["名稱"] = code_and_name.str[1]
        # 只保留 4 碼數字的普通股（排除特別股、ETF、權證等）
        frame = frame[frame["代號"].str.match(r"^\d{4}$")]

        for _, row in frame.iterrows():
            industry = str(row["產業別"]).strip()
            sector = sectors_by_name.get(industry)
            if sector is None:
                sector = Sector(sector_id=str(len(sectors_by_name) + 1), name=industry)
                sectors_by_name[industry] = sector
            ticker = f"{row['代號']}.{suffix}"
            sector.tickers.append(ticker)
            if row["名稱"]:
                sector.stock_names[ticker] = row["名稱"]

    return list(sectors_by_name.values())


# 固定股票池（不從證交所動態抓取），id 以 P 開頭以便與產業族群區分
PRESET_SECTORS = [
    Sector(
        sector_id="P1",
        name="營建精選（stock.py 清單）",
        tickers=[
            "2501.TW", "2504.TW", "2505.TW", "2511.TW", "2515.TW",
            "2520.TW", "2524.TW", "2527.TW", "2528.TW", "2534.TW",
            "2535.TW", "2538.TW", "2539.TW", "2542.TW", "2543.TW",
            "2545.TW", "2546.TW", "2547.TW", "2548.TW", "3703.TW",
            "5515.TW", "5519.TW", "5522.TW", "5534.TW",
        ],
    ),
]


def download_price_history(
    tickers: List[str],
    period: str = "6mo",
    chunk_size: int = 50,
    chunk_delay_seconds: float = 1.0,
) -> Dict[str, pd.DataFrame]:
    """分批下載多檔股票日 K，回傳 {ticker: DataFrame}；下載失敗或無資料的股票不會出現在結果中。

    原腳本逐檔下載並每檔 sleep 1 秒，全市場約需 30 分鐘以上；
    這裡改為每批多檔一次下載，批次之間仍保留延遲以降低被 Yahoo 限流的機會。
    """
    result: Dict[str, pd.DataFrame] = {}

    for start in range(0, len(tickers), chunk_size):
        chunk = tickers[start:start + chunk_size]
        if start > 0 and chunk_delay_seconds > 0:
            time.sleep(chunk_delay_seconds)
        try:
            raw = yf.download(
                chunk, period=period, group_by="ticker", progress=False, threads=True, auto_adjust=True
            )
        except Exception:
            logger.exception("下載股價失敗：%s", chunk)
            continue
        result.update(split_download_frame(raw, chunk))

    return result


MARGIN_API_URL = "https://www.twse.com.tw/exchangeReport/MI_MARGN"
MARGIN_REQUEST_DELAY_SECONDS = 0.3

# 已公布的歷史融資資料不會再變動，成功取得後即快取（key: YYYYMMDD）
_margin_cache: Dict[str, Dict] = {}
_margin_cache_lock = threading.Lock()


def fetch_margin_day(trade_date: pd.Timestamp, timeout_seconds: int = 10) -> Optional[Dict]:
    """抓取單一交易日的整體市場融資融券餘額；尚未公布或失敗時回傳 None。

    回傳欄位：margin_lots（融資張數）、short_lots（融券張數）、margin_amount_thousand（融資金額，仟元）
    """
    date_key = trade_date.strftime("%Y%m%d")
    with _margin_cache_lock:
        if date_key in _margin_cache:
            return _margin_cache[date_key]

    params = {"date": date_key, "selectType": "MS", "response": "json"}
    try:
        response = requests.get(MARGIN_API_URL, params=params, headers=REQUEST_HEADERS, timeout=timeout_seconds)
        data = response.json()
    except Exception:
        logger.warning("抓取 %s 融資資料失敗", date_key, exc_info=True)
        return None
    if data.get("stat") != "OK" or not data.get("tables"):
        return None

    record: Dict = {"date": trade_date}
    for row in data["tables"][0].get("data", []):
        name = row[0]
        today_balance = int(str(row[5]).replace(",", ""))   # index 5 = 今日餘額
        if "融資(交易單位)" in name:
            record["margin_lots"] = today_balance
        elif "融券(交易單位)" in name:
            record["short_lots"] = today_balance
        elif "融資金額" in name:
            record["margin_amount_thousand"] = today_balance

    if not {"margin_lots", "short_lots", "margin_amount_thousand"}.issubset(record):
        return None
    with _margin_cache_lock:
        _margin_cache[date_key] = record
    return record


def fetch_margin_history(trade_dates: List[pd.Timestamp]) -> List[Dict]:
    """逐日抓取融資資料；只有尚未快取的日期才會實際呼叫證交所並延遲。"""
    records = []
    for trade_date in trade_dates:
        with _margin_cache_lock:
            cached = _margin_cache.get(trade_date.strftime("%Y%m%d"))
        if cached is not None:
            records.append(cached)
            continue
        record = fetch_margin_day(trade_date)
        if record is not None:
            records.append(record)
        time.sleep(MARGIN_REQUEST_DELAY_SECONDS)
    return records


# 單一標的下載結果快取，避免多人重複開啟頁面時觸發 Yahoo 限流（YFRateLimitError）
SINGLE_HISTORY_CACHE_SECONDS = 10 * 60
_history_cache: Dict[tuple, tuple] = {}
_history_cache_lock = threading.Lock()


def download_single_history(
    ticker: str, period: Optional[str] = None, start: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """下載單一標的日 K（Open/High/Low/Close/Volume），查無資料時回傳 None。

    period 與 start 擇一使用（start 格式 YYYY-MM-DD）。成功結果快取 10 分鐘。
    """
    cache_key = (ticker, period, start)
    with _history_cache_lock:
        cached = _history_cache.get(cache_key)
        if cached is not None and time.monotonic() - cached[0] < SINGLE_HISTORY_CACHE_SECONDS:
            return cached[1].copy()

    frame = _download_single_history_uncached(ticker, period, start)
    if frame is not None:
        with _history_cache_lock:
            _history_cache[cache_key] = (time.monotonic(), frame)
            # 清掉過期項目，避免快取無限成長
            expired = [k for k, (t, _) in _history_cache.items() if time.monotonic() - t >= SINGLE_HISTORY_CACHE_SECONDS]
            for key in expired:
                del _history_cache[key]
        return frame.copy()
    return None


def _download_single_history_uncached(
    ticker: str, period: Optional[str], start: Optional[str]
) -> Optional[pd.DataFrame]:
    try:
        if start:
            raw = yf.download(ticker, start=start, group_by="ticker", progress=False, auto_adjust=True)
        else:
            raw = yf.download(ticker, period=period or "6mo", group_by="ticker", progress=False, auto_adjust=True)
    except Exception:
        logger.exception("下載股價失敗：%s", ticker)
        return None
    return split_download_frame(raw, [ticker]).get(ticker)


def split_download_frame(raw: pd.DataFrame, tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """把 yf.download(group_by='ticker') 的結果拆成每檔一個 DataFrame。"""
    result: Dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return result

    for ticker in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker not in raw.columns.get_level_values(0):
                continue
            frame = raw[ticker]
        elif len(tickers) == 1:
            frame = raw
        else:
            continue

        if not set(PRICE_REQUIRED_COLUMNS).issubset(frame.columns):
            continue
        frame = frame[PRICE_REQUIRED_COLUMNS].dropna(subset=["Close"])
        if not frame.empty:
            result[ticker] = frame

    return result
