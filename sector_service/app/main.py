"""Python 選股策略 API（Presentation 層）。

提供 ASP.NET 網站「Python 選股」選單下各頁面使用的資料，
邏輯移植自 PythonStock/ 資料夾內的各支腳本。

啟動方式（需先設定環境變數 SECTOR_API_KEY）：
    uvicorn app.main:app --host 127.0.0.1 --port 8001
"""
import hmac
import logging
import os
import re
import threading
from contextlib import asynccontextmanager
from datetime import date, timedelta
from functools import partial
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from . import economic_calendar, market_analysis, market_data, signal_analysis
from .fred_client import FredApiError, FredClient
from .indicators import BOTTOM_BREAKOUT_CONDITIONS, STRATEGY_CLASSIFIERS, STRATEGY_STANDARD, add_indicators
from .scan_jobs import ScanJobManager
from .sector_cache import SectorCache

# 讓本服務各模組的 INFO 以上訊息輸出到 stderr（部署時導向 logs/service.log）
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

logger = logging.getLogger(__name__)

API_KEY_ENV_NAME = "SECTOR_API_KEY"
# FRED API 金鑰（全球日曆使用）；未設定時日曆 API 回傳 503
FRED_API_KEY_ENV_NAME = "FRED_API_KEY"
CALENDAR_MAX_DAYS = 92
# 背景預熱 FRED 快取的間隔（需小於 FredClient 的 1 小時快取，讓使用者幾乎都命中快取）
CALENDAR_WARM_UP_INTERVAL_SECONDS = int(os.getenv("CALENDAR_WARM_UP_INTERVAL_SECONDS", str(50 * 60)))
_calendar_warm_up_stop = threading.Event()
SECTOR_CACHE_SECONDS = int(os.getenv("SECTOR_CACHE_SECONDS", str(12 * 60 * 60)))
# 族群清單硬碟快取（服務重新啟動後立即可用）；預設放在 app 資料夾旁的 cache/
SECTOR_CACHE_FILE = os.getenv(
    "SECTOR_CACHE_FILE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "sectors.json")
)
DOWNLOAD_CHUNK_DELAY_SECONDS = float(os.getenv("DOWNLOAD_CHUNK_DELAY_SECONDS", "1"))
TICKER_PATTERN = re.compile(r"^\d{4}\.(TW|TWO)$")
# 個股分析接受純數字台股代號（自動判斷上市 / 上櫃）、含後綴的台股代號或美股代號
STOCK_CODE_PATTERN = re.compile(r"^(\d{4,6}(\.(TW|TWO))?|[A-Z]{1,5})$")
JOB_ID_PATTERN = r"^[0-9a-f]{32}$"
SECTOR_ID_PATTERN = r"^[A-Z]?\d{1,4}$"

SIGNAL_ANALYSIS_PERIODS = {"3mo", "6mo", "1y", "2y", "5y"}
REBOUND_PERIODS = {"2y", "5y", "10y", "max"}
INDEX_TICKER = "^TWII"
ETF_TICKER = "0050.TW"


class CamelModel(BaseModel):
    """對外 JSON 一律使用 camelCase。"""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class SectorSummary(CamelModel):
    sector_id: str
    name: str
    ticker_count: int
    is_preset: bool = False


class ConditionDefinition(CamelModel):
    # key 與掃描結果 conditions 內的鍵相同（camelCase），供前端自選條件篩選
    key: str
    label: str


class StrategySummary(CamelModel):
    strategy_id: str
    name: str
    source_script: str
    strong_label: str
    watch_label: Optional[str] = None
    description: str
    # 非空時，前端以「自選條件」取代伺服器端的觀察名單
    condition_definitions: List[ConditionDefinition] = Field(default_factory=list)
    default_custom_conditions: List[str] = Field(default_factory=list)


class StartScanRequest(CamelModel):
    # 空陣列或含 "0" 代表全部產業族群（與原腳本輸入 0 的意義相同，不含固定清單）
    sector_ids: List[str] = Field(default_factory=list, max_length=200)
    strategy: str = STRATEGY_STANDARD


class StartScanResponse(CamelModel):
    job_id: str
    total_count: int


STRATEGIES = [
    StrategySummary(
        strategy_id="standard", name="爆量強勢", source_script="stock_sector.py",
        strong_label="⭐ 爆量強勢", watch_label="🔔 候選觀察（待爆量確認）",
        description="① 收盤站上月線（MA20） ② KD 黃金交叉 ③ 5 日均量 > 500 張 ④ 今日量 > 5 日均量 × 1.2。"
                    "符合 ①②③④ 列為爆量強勢，符合 ①②③ 列為候選觀察。",
    ),
    StrategySummary(
        strategy_id="bottom_breakout", name="底部起漲", source_script="stock_sectorX.py",
        strong_label="⭐ 底部剛突破", watch_label="🎯 自選條件",
        description="① MA5/MA20/MA60 糾結（差距 < 5%） ② 收盤突破三條均線 ③ KD 低檔黃金交叉（昨日 K < 30） "
                    "④ MACD 柱狀翻紅 ⑤ 5 日均量 > 500 張 ⑥ 今日量 > 5 日均量 × 1.2 ⑦ 月線大於季線（MA20 > MA60）。"
                    "7 個條件全部符合列為底部剛突破；自選條件可任意勾選其中幾項即時篩選。"
                    "自選條件另提供進階輔助：⑧ 月線乖離率 < 6% ⑨ 布林通道壓縮 ⑩ 帶量突破布林上軌（不影響底部剛突破名單）。",
        condition_definitions=[ConditionDefinition(key=to_camel(key), label=label) for key, label in BOTTOM_BREAKOUT_CONDITIONS],
        # 預設勾選：⑤ 流動性（5 日均量 > 500 張） ⑦ 月線大於季線 ⑧ 月線乖離率 < 6%
        default_custom_conditions=[to_camel(key) for key in ("liquid", "ma20_above_ma60", "low_bias_ma20")],
    ),
    StrategySummary(
        strategy_id="basic", name="月線 + KD 金叉", source_script="stock.py",
        strong_label="⭐ 強勢觀察名單", watch_label=None,
        description="① 收盤站上月線（MA20） ② KD 黃金交叉。兩者同時符合即列入強勢觀察名單（不看量能）。",
    ),
]
# 策略清單與 Domain 層的分類器必須一致
assert {s.strategy_id for s in STRATEGIES} == set(STRATEGY_CLASSIFIERS)


sector_cache = SectorCache(SECTOR_CACHE_SECONDS, market_data.fetch_sectors, SECTOR_CACHE_FILE)
job_manager = ScanJobManager(
    price_downloader=partial(market_data.download_price_history, chunk_delay_seconds=DOWNLOAD_CHUNK_DELAY_SECONDS)
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not os.getenv(API_KEY_ENV_NAME):
        logger.warning("未設定環境變數 %s，所有 API 請求都會被拒絕", API_KEY_ENV_NAME)
    # 先載入硬碟快取並在背景更新，避免第一位使用者等待證交所下載
    sector_cache.warm_up()
    threading.Thread(target=_warm_up_calendar_loop, name="calendar-warm-up", daemon=True).start()
    yield
    _calendar_warm_up_stop.set()
    job_manager.shutdown()


app = FastAPI(title="Taiwan Stock Strategy API", version="2.0.0", lifespan=lifespan)


def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """驗證呼叫端（ASP.NET 網站）帶入的 X-Api-Key。"""
    expected_key = os.getenv(API_KEY_ENV_NAME)
    if not expected_key:
        raise HTTPException(status_code=503, detail="Service API key is not configured")
    if not x_api_key or not hmac.compare_digest(x_api_key, expected_key):
        raise HTTPException(status_code=401, detail="Invalid API key")


def load_sectors_or_fail() -> List[market_data.Sector]:
    sectors = sector_cache.get()
    if not sectors:
        raise HTTPException(status_code=502, detail="Unable to load sector list from TWSE")
    return sectors


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 族群掃描（stock_sector.py / stock_sectorX.py / stock.py）
# ---------------------------------------------------------------------------
@app.get("/api/strategies", response_model=List[StrategySummary], dependencies=[Depends(verify_api_key)])
def list_strategies() -> List[StrategySummary]:
    """列出可用的族群掃描策略。"""
    return STRATEGIES


@app.get("/api/sectors", response_model=List[SectorSummary], dependencies=[Depends(verify_api_key)])
def list_sectors() -> List[SectorSummary]:
    """列出固定清單與上市＋上櫃產業族群。"""
    presets = [
        SectorSummary(sector_id=s.sector_id, name=s.name, ticker_count=len(s.tickers), is_preset=True)
        for s in market_data.PRESET_SECTORS
    ]
    sectors = [
        SectorSummary(sector_id=s.sector_id, name=s.name, ticker_count=len(s.tickers))
        for s in load_sectors_or_fail()
    ]
    return presets + sectors


@app.post("/api/scans", response_model=StartScanResponse, status_code=202, dependencies=[Depends(verify_api_key)])
def start_scan(request: StartScanRequest) -> StartScanResponse:
    """建立背景掃描工作，回傳 jobId 供之後查詢進度。"""
    if request.strategy not in STRATEGY_CLASSIFIERS:
        raise HTTPException(status_code=400, detail="Unknown strategy")
    for sector_id in request.sector_ids:
        if not re.match(SECTOR_ID_PATTERN, sector_id):
            raise HTTPException(status_code=400, detail="Invalid sector id")

    wanted_ids = set(request.sector_ids)
    presets = [s for s in market_data.PRESET_SECTORS if s.sector_id in wanted_ids]
    needs_twse = not request.sector_ids or "0" in wanted_ids or any(not i.startswith("P") for i in wanted_ids)
    # 只掃描固定清單時，證交所清單僅用來查股票名稱，抓取失敗也不影響掃描
    twse_sectors = load_sectors_or_fail() if needs_twse else sector_cache.get()

    if not request.sector_ids or "0" in wanted_ids:
        selected = presets + twse_sectors
    else:
        selected = presets + [s for s in twse_sectors if s.sector_id in wanted_ids]
    if not selected:
        raise HTTPException(status_code=400, detail="No valid sector selected")

    tickers = [ticker for sector in selected for ticker in sector.tickers]
    stock_names = market_data.build_stock_name_map(twse_sectors)
    job = job_manager.submit([s.name for s in selected], tickers, request.strategy, stock_names)
    return StartScanResponse(job_id=job.job_id, total_count=len(job.tickers))


@app.get("/api/scans/{job_id}", dependencies=[Depends(verify_api_key)])
def get_scan(job_id: str = Path(pattern=JOB_ID_PATTERN)) -> dict:
    """查詢掃描工作狀態與結果。"""
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Scan job not found")
    return to_camel_keys(job)


@app.get("/api/stocks/{ticker}/indicators", dependencies=[Depends(verify_api_key)])
def get_indicators(ticker: str) -> dict:
    """回傳單一股票近 6 個月的技術指標序列，供前端繪製圖表。"""
    ticker = ticker.upper()
    if not TICKER_PATTERN.match(ticker):
        raise HTTPException(status_code=400, detail="Ticker must look like 2330.TW or 6488.TWO")

    frames = market_data.download_price_history([ticker], chunk_delay_seconds=0)
    if ticker not in frames:
        raise HTTPException(status_code=404, detail="No price data")

    frame = add_indicators(frames[ticker]).round(4)
    columns = ["Close", "MA5", "MA20", "MA60", "MACD", "Signal", "OSC", "RSI14", "K", "D"]
    series = {column.lower(): frame[column].tolist() for column in columns}
    return {"ticker": ticker, "dates": [d.strftime("%Y-%m-%d") for d in frame.index], **series}


# ---------------------------------------------------------------------------
# 個股訊號分析（stock_bias_bollinger.py）
# ---------------------------------------------------------------------------
@app.get("/api/stocks/{code}/signal-analysis", dependencies=[Depends(verify_api_key)])
def get_signal_analysis(code: str, period: str = Query(default="1y")) -> dict:
    """RSI / MACD / KD 買賣訊號、動態出場回測與繪圖資料。"""
    code = code.upper()
    if not STOCK_CODE_PATTERN.match(code):
        raise HTTPException(status_code=400, detail="Invalid stock code")
    if period not in SIGNAL_ANALYSIS_PERIODS:
        raise HTTPException(status_code=400, detail="Invalid period")

    # 純數字代號先試上市（.TW），查無資料再試上櫃（.TWO），與原腳本相同
    candidates = [code + ".TW", code + ".TWO"] if code.isdigit() else [code]
    for ticker in candidates:
        frame = market_data.download_single_history(ticker, period=period)
        if frame is not None and not frame.empty:
            result = signal_analysis.analyze_price_history(ticker, frame)
            if result is None:
                raise HTTPException(status_code=422, detail="Not enough data to calculate indicators")
            return result
    raise HTTPException(status_code=404, detail=f"No price data for {code}")


# ---------------------------------------------------------------------------
# 大盤分析（stock_all.py / market_rebound.py / market_margin.py）
# ---------------------------------------------------------------------------
@app.get("/api/market/kd-backtest", dependencies=[Depends(verify_api_key)])
def get_kd_backtest(start_date: date = Query(default=date(2020, 6, 1), alias="startDate")) -> dict:
    """大盤 KD 波段策略回測 0050。"""
    if start_date < date(2005, 1, 1) or start_date >= date.today():
        raise HTTPException(status_code=400, detail="startDate must be between 2005-01-01 and yesterday")
    start = start_date.isoformat()
    twii = market_data.download_single_history(INDEX_TICKER, start=start)
    etf = market_data.download_single_history(ETF_TICKER, start=start)
    if twii is None or etf is None:
        raise HTTPException(status_code=502, detail="Unable to download price data")
    return market_analysis.run_kd_band_backtest(twii, etf)


@app.get("/api/market/rebound", dependencies=[Depends(verify_api_key)])
def get_rebound(period: str = Query(default="5y")) -> dict:
    """大盤單日回檔後隔月反彈統計。"""
    if period not in REBOUND_PERIODS:
        raise HTTPException(status_code=400, detail="Invalid period")
    index_frame = market_data.download_single_history(INDEX_TICKER, period=period)
    if index_frame is None:
        raise HTTPException(status_code=502, detail="Unable to download index data")
    return {"period": period, **market_analysis.analyze_rebound(index_frame)}


@app.get("/api/market/margin", dependencies=[Depends(verify_api_key)])
def get_margin(days: int = Query(default=60, ge=10, le=120)) -> dict:
    """近 N 個交易日的加權指數、融資融券餘額與近似融資維持率。

    首次查詢需逐日呼叫證交所（每日約 0.5 秒），之後已公布的日期會使用快取。
    """
    index_frame = market_data.download_single_history(INDEX_TICKER, period="1y")
    if index_frame is None:
        raise HTTPException(status_code=502, detail="Unable to download index data")
    trade_dates = list(index_frame.index[-days:])
    records = market_data.fetch_margin_history(trade_dates)
    if not records:
        raise HTTPException(status_code=502, detail="Unable to download margin data from TWSE")
    return market_analysis.analyze_margin(index_frame, records)


# ---------------------------------------------------------------------------
# 全球經濟日曆（FRED）
# ---------------------------------------------------------------------------
_fred_client: Optional[FredClient] = None
_fred_client_key: Optional[str] = None


def get_fred_client() -> FredClient:
    """依環境變數建立 FRED 用戶端；金鑰變更時重建（快取隨之清空）。"""
    global _fred_client, _fred_client_key
    api_key = os.getenv(FRED_API_KEY_ENV_NAME, "")
    if not api_key:
        raise HTTPException(status_code=503, detail="FRED API key is not configured")
    if _fred_client is None or _fred_client_key != api_key:
        _fred_client, _fred_client_key = FredClient(api_key), api_key
    return _fred_client


def _warm_up_calendar_once() -> None:
    """預先載入本月起三個月的日曆、重要指標最新值與政策利率（與頁面預設查詢的範圍相同）。"""
    if not os.getenv(FRED_API_KEY_ENV_NAME):
        return
    client = get_fred_client()
    today = date.today()
    month_start = today.replace(day=1)
    for offset in range(3):
        start = (month_start + timedelta(days=32 * offset)).replace(day=1)
        end = (start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        calendar = economic_calendar.build_calendar(client, start, end)
        release_ids = [event["releaseId"] for event in calendar["events"] if event["hasIndicators"]]
        economic_calendar.get_release_indicators(client, release_ids)
    economic_calendar.get_policy_rates(client)


def _warm_up_calendar_loop() -> None:
    while not _calendar_warm_up_stop.is_set():
        try:
            _warm_up_calendar_once()
        except Exception as error:   # 背景預熱失敗不影響服務，下一輪再試；使用者請求時仍會即時查詢
            # 只記錄類型與已去除金鑰的 FredApiError 訊息
            detail = str(error) if isinstance(error, FredApiError) else type(error).__name__
            logger.warning("全球日曆背景預熱失敗：%s", detail)
        _calendar_warm_up_stop.wait(CALENDAR_WARM_UP_INTERVAL_SECONDS)


@app.get("/api/calendar", dependencies=[Depends(verify_api_key)])
def get_calendar(
    start_date: date = Query(alias="startDate"),
    end_date: date = Query(alias="endDate"),
    scope: str = Query(default=economic_calendar.SCOPE_IMPORTANT),
) -> dict:
    """經濟數據發布日程（含未來排程）與重要指標的最新值。"""
    if scope not in (economic_calendar.SCOPE_IMPORTANT, economic_calendar.SCOPE_ALL):
        raise HTTPException(status_code=400, detail="scope must be important or all")
    if end_date < start_date or (end_date - start_date).days > CALENDAR_MAX_DAYS:
        raise HTTPException(status_code=400, detail=f"Date range must be 0-{CALENDAR_MAX_DAYS} days")
    try:
        return economic_calendar.build_calendar(get_fred_client(), start_date, end_date, scope)
    except FredApiError as error:
        raise HTTPException(status_code=502, detail=str(error))


@app.get("/api/calendar/indicators", dependencies=[Depends(verify_api_key)])
def get_calendar_indicators(release_ids: str = Query(alias="releaseIds", max_length=200)) -> dict:
    """重要指標 release 的最新值 / 前值，releaseIds 以逗號分隔（例如 10,50,53）。"""
    if not re.fullmatch(r"\d{1,5}(,\d{1,5}){0,49}", release_ids):
        raise HTTPException(status_code=400, detail="releaseIds must be comma separated numbers")
    ids = [int(value) for value in release_ids.split(",")]
    return {"indicators": economic_calendar.get_release_indicators(get_fred_client(), ids)}


@app.get("/api/calendar/policy-rates", dependencies=[Depends(verify_api_key)])
def get_policy_rates() -> dict:
    """美國聯準會與歐洲央行目前政策利率（含上次調整日期）。"""
    try:
        return {"rates": economic_calendar.get_policy_rates(get_fred_client())}
    except FredApiError as error:
        raise HTTPException(status_code=502, detail=str(error))


def to_camel_keys(value):
    """遞迴把 dict 的 snake_case 鍵轉成 camelCase。"""
    if isinstance(value, dict):
        return {to_camel(key): to_camel_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_camel_keys(item) for item in value]
    return value
