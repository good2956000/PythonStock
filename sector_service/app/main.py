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
from datetime import date, datetime, timedelta, timezone
from functools import partial
from typing import Annotated, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from . import economic_calendar, market_analysis, market_data, signal_analysis
from .fred_client import FredApiError, FredClient
from .mops_news import MopsApiError, MopsClient, one_month_before
from .optimize_jobs import RESULT_ID_PATTERN, OptimizeJobConflictError, OptimizeJobManager
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
# KD 波段策略參數最佳化結果存檔位置與保留筆數；預設放在 app 資料夾旁的 cache/kd_optimize/（不受部署覆蓋）
KD_OPTIMIZE_RESULT_DIR = os.getenv(
    "KD_OPTIMIZE_RESULT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "kd_optimize")
)
KD_OPTIMIZE_RESULTS_KEPT = int(os.getenv("KD_OPTIMIZE_RESULTS_KEPT", "20"))
DOWNLOAD_CHUNK_DELAY_SECONDS = float(os.getenv("DOWNLOAD_CHUNK_DELAY_SECONDS", "1"))
TICKER_PATTERN = re.compile(r"^\d{4}\.(TW|TWO)$")
# 個股分析接受純數字台股代號（自動判斷上市 / 上櫃）、含後綴的台股代號或美股代號
STOCK_CODE_PATTERN = re.compile(r"^(\d{4,6}[A-Z]?(\.(TW|TWO))?|[A-Z]{1,5})$")
# 台股代號（純數字或數字加一個英文字尾，例如 00631L），不含 .TW / .TWO 後綴
TAIWAN_CODE_PATTERN = re.compile(r"^\d{4,6}[A-Z]?$")
JOB_ID_PATTERN = r"^[0-9a-f]{32}$"
SECTOR_ID_PATTERN = r"^[A-Z]?\d{1,4}$"

SIGNAL_ANALYSIS_PERIODS = {"3mo", "6mo", "1y", "2y", "5y"}
REBOUND_PERIODS = {"2y", "5y", "10y", "max"}
TAIPEI_TIMEZONE = timezone(timedelta(hours=8))   # 台灣無日光節約時間，固定 UTC+8
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
optimize_job_manager = OptimizeJobManager(KD_OPTIMIZE_RESULT_DIR, KD_OPTIMIZE_RESULTS_KEPT)


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
    optimize_job_manager.shutdown()


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
    candidates = [code + ".TW", code + ".TWO"] if TAIWAN_CODE_PATTERN.match(code) else [code]
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
def kd_band_parameters(
    kd_period: int = Query(default=market_analysis.KD_PERIOD, alias="kdPeriod", ge=2, le=60),
    k_smoothing: int = Query(default=market_analysis.K_SMOOTHING, alias="kSmoothing", ge=1, le=20),
    d_smoothing: int = Query(default=market_analysis.D_SMOOTHING, alias="dSmoothing", ge=1, le=20),
    ma_period: int = Query(default=market_analysis.MA_PERIOD, alias="maPeriod", ge=2, le=240),
    buy_k_threshold: float = Query(default=market_analysis.BUY_K_THRESHOLD, alias="buyKThreshold", ge=0, le=100),
    take_profit_k_threshold: float = Query(
        default=market_analysis.TAKE_PROFIT_K_THRESHOLD, alias="takeProfitKThreshold", ge=0, le=100),
    take_profit_bias_percent: float = Query(
        default=market_analysis.TAKE_PROFIT_BIAS_PERCENT, alias="takeProfitBiasPercent", ge=-30, le=50),
    trend_break_k_threshold: float = Query(
        default=market_analysis.TREND_BREAK_K_THRESHOLD, alias="trendBreakKThreshold", ge=0, le=100),
    trailing_stop_method: str = Query(
        default=market_analysis.TRAILING_STOP_METHOD, alias="trailingStopMethod",
        pattern=f"^({'|'.join(market_analysis.TRAILING_STOP_METHODS)})$"),
    trailing_stop_source: str = Query(
        default=market_analysis.TRAILING_STOP_SOURCE, alias="trailingStopSource",
        pattern=f"^({'|'.join(market_analysis.TRAILING_SOURCES)})$"),
    trailing_ma_period: int = Query(default=market_analysis.TRAILING_MA_PERIOD, alias="trailingMaPeriod", ge=2, le=240),
    trailing_ma_filter: str = Query(
        default=market_analysis.TRAILING_MA_FILTER, alias="trailingMaFilter",
        pattern=f"^({'|'.join(market_analysis.MA_FILTERS)})$"),
    trailing_ma_confirm_days: int = Query(
        default=market_analysis.TRAILING_MA_CONFIRM_DAYS, alias="trailingMaConfirmDays", ge=1, le=20),
    trailing_ma_break_percent: float = Query(
        default=market_analysis.TRAILING_MA_BREAK_PERCENT, alias="trailingMaBreakPercent", ge=0, le=30),
    trailing_drawdown_percent: float = Query(
        default=market_analysis.TRAILING_DRAWDOWN_PERCENT, alias="trailingDrawdownPercent", gt=0, le=50),
    atr_period: int = Query(default=market_analysis.ATR_PERIOD, alias="atrPeriod", ge=2, le=60),
    atr_multiplier: float = Query(default=market_analysis.ATR_MULTIPLIER, alias="atrMultiplier", gt=0, le=10),
    swing_low_lookback: int = Query(default=market_analysis.SWING_LOW_LOOKBACK, alias="swingLowLookback", ge=2, le=120),
    transaction_costs_enabled: bool = Query(
        default=market_analysis.TRANSACTION_COSTS_ENABLED, alias="transactionCostsEnabled"),
    fee_rate_percent: float = Query(default=market_analysis.FEE_RATE_PERCENT, alias="feeRatePercent", ge=0, le=1),
    fee_discount: float = Query(default=market_analysis.FEE_DISCOUNT, alias="feeDiscount", ge=0, le=1),
    sell_tax_percent: float = Query(default=market_analysis.SELL_TAX_PERCENT, alias="sellTaxPercent", ge=0, le=1),
) -> market_analysis.KdBandParameters:
    """KD 波段策略參數（回測與最佳化共用）；未指定的參數使用 market_analysis.py 最上方的預設值。"""
    return market_analysis.KdBandParameters(
        kd_period=kd_period, k_smoothing=k_smoothing, d_smoothing=d_smoothing, ma_period=ma_period,
        buy_k_threshold=buy_k_threshold, take_profit_k_threshold=take_profit_k_threshold,
        take_profit_bias_percent=take_profit_bias_percent, trend_break_k_threshold=trend_break_k_threshold,
        trailing_stop_method=trailing_stop_method, trailing_stop_source=trailing_stop_source,
        trailing_ma_period=trailing_ma_period, trailing_ma_filter=trailing_ma_filter,
        trailing_ma_confirm_days=trailing_ma_confirm_days, trailing_ma_break_percent=trailing_ma_break_percent,
        trailing_drawdown_percent=trailing_drawdown_percent, atr_period=atr_period, atr_multiplier=atr_multiplier,
        swing_low_lookback=swing_low_lookback, transaction_costs_enabled=transaction_costs_enabled,
        fee_rate_percent=fee_rate_percent, fee_discount=fee_discount, sell_tax_percent=sell_tax_percent,
    )


def _validate_kd_start_date(start_date: date) -> None:
    if start_date < date(2005, 1, 1) or start_date >= date.today():
        raise HTTPException(status_code=400, detail="startDate must be between 2005-01-01 and yesterday")


def _download_kd_prices(start_date: date):
    """下載加權指數與 0050 日 K；盤中只算到前一個收盤日，避免以即時價產生訊號。下載失敗回傳 None。"""
    start = start_date.isoformat()
    twii = market_data.download_single_history(INDEX_TICKER, start=start)
    etf = market_data.download_single_history(ETF_TICKER, start=start)
    if twii is None or etf is None:
        return None
    taipei_now = datetime.now(TAIPEI_TIMEZONE)
    return market_analysis.drop_unclosed_session(twii, taipei_now), market_analysis.drop_unclosed_session(etf, taipei_now)


def _load_kd_prices(start_date: date):
    _validate_kd_start_date(start_date)
    prices = _download_kd_prices(start_date)
    if prices is None:
        raise HTTPException(status_code=502, detail="Unable to download price data")
    return prices


@app.get("/api/market/kd-backtest", dependencies=[Depends(verify_api_key)])
def get_kd_backtest(
    start_date: date = Query(default=date(2020, 6, 1), alias="startDate"),
    parameters: market_analysis.KdBandParameters = Depends(kd_band_parameters),
) -> dict:
    """大盤 KD 波段策略回測 0050；KD 天數、月線天數、各門檻、移動停利（擇一）與交易成本可由查詢參數調整。"""
    twii, etf = _load_kd_prices(start_date)
    return market_analysis.run_kd_band_backtest(twii, etf, parameters)


MAX_SEARCH_VALUES = market_analysis.OPTIMIZE_MAX_VALUES_PER_PARAMETER


def _search_values(item_type, **limits):
    """搜尋範圍內單一參數的候選值清單型別（未提供或空清單 = 不搜尋，固定為目前設定）。"""
    return Optional[Annotated[List[Annotated[item_type, Field(**limits)]], Field(max_length=MAX_SEARCH_VALUES)]]


class OptimizeSpaceBody(CamelModel):
    """各參數的候選值；範圍與回測參數相同。"""

    kd_period: _search_values(int, ge=2, le=60) = None
    k_smoothing: _search_values(int, ge=1, le=20) = None
    d_smoothing: _search_values(int, ge=1, le=20) = None
    ma_period: _search_values(int, ge=2, le=240) = None
    buy_k_threshold: _search_values(float, ge=0, le=100) = None
    take_profit_k_threshold: _search_values(float, ge=0, le=100) = None
    take_profit_bias_percent: _search_values(float, ge=-30, le=50) = None
    trend_break_k_threshold: _search_values(float, ge=0, le=100) = None
    trailing_stop_method: _search_values(str, pattern=f"^({'|'.join(market_analysis.TRAILING_STOP_METHODS)})$") = None
    trailing_ma_period: _search_values(int, ge=2, le=240) = None
    trailing_ma_filter: _search_values(str, pattern=f"^({'|'.join(market_analysis.MA_FILTERS)})$") = None
    trailing_ma_confirm_days: _search_values(int, ge=1, le=20) = None
    trailing_ma_break_percent: _search_values(float, ge=0, le=30) = None
    trailing_drawdown_percent: _search_values(float, gt=0, le=50) = None
    atr_period: _search_values(int, ge=2, le=60) = None
    atr_multiplier: _search_values(float, gt=0, le=10) = None
    swing_low_lookback: _search_values(int, ge=2, le=120) = None


class StartOptimizeBody(CamelModel):
    mode: str = Field(default=market_analysis.OPTIMIZE_MODE_RANDOM, pattern=f"^({'|'.join(market_analysis.OPTIMIZE_MODES)})$")
    trials: int = Field(default=market_analysis.OPTIMIZE_TRIALS, ge=10, le=market_analysis.OPTIMIZE_MAX_TRIALS)
    # 隨機抽樣的亂數種子；未指定時每次重新抽樣（實際使用的種子會回傳，可填回重現結果）
    seed: Optional[int] = Field(default=None, ge=1, le=market_analysis.OPTIMIZE_MAX_SEED)
    train_ratio: float = Field(default=market_analysis.OPTIMIZE_TRAIN_RATIO, ge=0.5, le=0.9)
    min_trades: int = Field(default=market_analysis.OPTIMIZE_MIN_TRADES, ge=0, le=50)
    # 未提供時使用預設搜尋範圍
    space: Optional[OptimizeSpaceBody] = None


@app.get("/api/market/kd-backtest/optimize-options", dependencies=[Depends(verify_api_key)])
def get_kd_optimize_options() -> dict:
    """參數最佳化的預設搜尋範圍與上限，供網頁產生搜尋範圍表單。"""
    return {
        "space": market_analysis.build_search_space(market_analysis.KdBandParameters()).searched_camel(),
        "defaultTrials": market_analysis.OPTIMIZE_TRIALS,
        "maxTrials": market_analysis.OPTIMIZE_MAX_TRIALS,
        "maxCombinations": market_analysis.OPTIMIZE_MAX_COMBINATIONS,
        "maxValuesPerParameter": market_analysis.OPTIMIZE_MAX_VALUES_PER_PARAMETER,
    }


@app.post("/api/market/kd-backtest/optimize-jobs", status_code=202, dependencies=[Depends(verify_api_key)])
def start_kd_optimize_job(
    body: StartOptimizeBody,
    start_date: date = Query(default=date(2020, 6, 1), alias="startDate"),
    parameters: market_analysis.KdBandParameters = Depends(kd_band_parameters),
) -> dict:
    """建立 KD 波段策略參數最佳化背景工作（隨機抽樣或完整搜尋），回傳工作狀態供之後查詢進度。

    目前的參數會列為比較基準；交易成本與停利判斷價格沿用目前設定，不參與搜尋。同一時間只執行一個工作。
    """
    _validate_kd_start_date(start_date)
    space = {key: values for key, values in body.space.model_dump().items() if values} if body.space else None
    try:
        search = market_analysis.build_search_space(parameters, space)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    exhaustive = body.mode == market_analysis.OPTIMIZE_MODE_EXHAUSTIVE
    if exhaustive and search.total > market_analysis.OPTIMIZE_MAX_COMBINATIONS:
        raise HTTPException(status_code=400, detail=f"完整搜尋共 {search.total:,} 組，超過上限 "
                                                    f"{market_analysis.OPTIMIZE_MAX_COMBINATIONS:,} 組，請縮小搜尋範圍")
    total_count = search.total if exhaustive else min(body.trials, search.total)
    settings = {
        "startDate": start_date.isoformat(), "mode": body.mode, "trials": body.trials, "seed": body.seed,
        "trainRatio": body.train_ratio, "minTrades": body.min_trades, "parameters": parameters.to_camel_dict(),
        "space": search.searched_camel(),
    }

    def run(progress):
        prices = _download_kd_prices(start_date)
        if prices is None:
            raise RuntimeError("無法下載加權指數或 0050 股價資料，請稍後再試")
        twii, etf = prices
        return market_analysis.optimize_kd_band(
            twii, etf, parameters, space=space, mode=body.mode, trials=body.trials, seed=body.seed,
            train_ratio=body.train_ratio, min_trades=body.min_trades, progress=progress)

    try:
        return optimize_job_manager.submit(body.mode, total_count, settings, run)
    except OptimizeJobConflictError:
        raise HTTPException(status_code=409, detail="目前已有參數最佳化在執行中，請等它完成或先停止")


@app.get("/api/market/kd-backtest/optimize-jobs/current", dependencies=[Depends(verify_api_key)])
def get_current_kd_optimize_job() -> dict:
    """執行中的最佳化工作（沒有時 job 為 null），重新開啟頁面時用來接續顯示進度。"""
    return {"job": optimize_job_manager.current()}


@app.get("/api/market/kd-backtest/optimize-jobs/{job_id}", dependencies=[Depends(verify_api_key)])
def get_kd_optimize_job(job_id: str = Path(pattern=JOB_ID_PATTERN)) -> dict:
    """查詢最佳化工作進度；完成後 result 為最佳化結果。"""
    job = optimize_job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="找不到最佳化工作（服務可能已重新啟動）")
    return job


@app.post("/api/market/kd-backtest/optimize-jobs/{job_id}/cancel", dependencies=[Depends(verify_api_key)])
def cancel_kd_optimize_job(job_id: str = Path(pattern=JOB_ID_PATTERN)) -> dict:
    """停止最佳化工作（停止的工作不存檔）。"""
    job = optimize_job_manager.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="找不到最佳化工作（服務可能已重新啟動）")
    return job


@app.get("/api/market/kd-backtest/optimize-results", dependencies=[Depends(verify_api_key)])
def list_kd_optimize_results() -> dict:
    """已存檔的最佳化結果摘要（新到舊）。"""
    return {"results": optimize_job_manager.list_results()}


@app.get("/api/market/kd-backtest/optimize-results/{result_id}", dependencies=[Depends(verify_api_key)])
def get_kd_optimize_result(result_id: str = Path(pattern=RESULT_ID_PATTERN)) -> dict:
    """單筆已存檔的最佳化結果（含當時的設定）。"""
    saved = optimize_job_manager.load_result(result_id)
    if saved is None:
        raise HTTPException(status_code=404, detail="找不到最佳化結果")
    return saved


@app.get("/api/market/rebound", dependencies=[Depends(verify_api_key)])
def get_rebound(period: str = Query(default="5y")) -> dict:
    """大盤單日回檔後隔月反彈統計。"""
    if period not in REBOUND_PERIODS:
        raise HTTPException(status_code=400, detail="Invalid period")
    index_frame = market_data.download_single_history(INDEX_TICKER, period=period)
    if index_frame is None:
        raise HTTPException(status_code=502, detail="Unable to download index data")
    index_frame = market_analysis.drop_unclosed_session(index_frame, datetime.now(TAIPEI_TIMEZONE))
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


# ---------------------------------------------------------------------------
# 近期發布之重大訊息（公開資訊觀測站）
# ---------------------------------------------------------------------------
MOPS_MAX_COMPANIES = 30  # 與 .NET MaterialNewsOptions.MaxCompanies 一致；逐家查詢、間隔 0.3 秒
mops_client = MopsClient()


def _raise_for_mops_error(error: MopsApiError):
    # 查詢條件錯誤（例如公司代號不存在）回 404，MOPS 本身無法連線回 502
    raise HTTPException(status_code=404 if error.is_client_error else 502, detail=str(error))


@app.get("/api/mops/news", dependencies=[Depends(verify_api_key)])
def get_material_news(company_ids: str = Query(alias="companyIds", max_length=250)) -> dict:
    """多家公司近一個月內的重大訊息（MOPS 每家最多提供最近 10 筆），companyIds 以逗號分隔（最多 MOPS_MAX_COMPANIES 家）；
    個別公司查詢失敗列於 errors。"""
    ids = [value.strip().upper() for value in company_ids.split(",") if value.strip()]
    if not ids or len(ids) > MOPS_MAX_COMPANIES or not all(re.fullmatch(r"[0-9A-Z]{4,6}", i) for i in ids):
        raise HTTPException(status_code=400, detail=f"companyIds must be 1-{MOPS_MAX_COMPANIES} stock codes")
    return mops_client.get_news_for_companies(ids, since=one_month_before(date.today()))


@app.get("/api/mops/news/detail", dependencies=[Depends(verify_api_key)])
def get_material_news_detail(
    company_id: str = Query(alias="companyId", pattern=r"^[0-9A-Z]{4,6}$"),
    market_kind: str = Query(alias="marketKind", pattern=r"^[a-z]{2,5}$"),
    enter_date: str = Query(alias="enterDate", pattern=r"^\d{7}$"),
    serial_number: str = Query(alias="serialNumber", pattern=r"^\d{1,3}$"),
) -> dict:
    """單則重大訊息完整內容（不含發言人姓名與電話）。"""
    try:
        return mops_client.get_news_detail(company_id, market_kind, enter_date, serial_number)
    except MopsApiError as error:
        _raise_for_mops_error(error)


def to_camel_keys(value):
    """遞迴把 dict 的 snake_case 鍵轉成 camelCase。"""
    if isinstance(value, dict):
        return {to_camel(key): to_camel_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_camel_keys(item) for item in value]
    return value
