"""大盤層級的分析（Domain 層，輸入為已下載的 DataFrame，不直接連網）。

- run_kd_band_backtest : 移植自 PythonStock/stock_all.py      （大盤 KD 波段策略回測 0050）
- analyze_rebound      : 移植自 PythonStock/market_rebound.py （大盤回檔後隔月反彈統計）
- analyze_margin       : 移植自 PythonStock/market_margin.py  （融資餘額 + 近似融資維持率）
"""
import heapq
import math
import secrets
import time as timer
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .serialization import dates_to_list, series_to_list, to_float

# ---------------------------------------------------------------------------
# 大盤 KD 波段策略（stock_all.py）可調參數預設值；網頁未指定時使用
# ---------------------------------------------------------------------------
KD_PERIOD = 9                       # RSV 計算天數
K_SMOOTHING = 3                     # K 值平滑：K = 昨日K × (1 - 1/3) + 今日RSV × 1/3
D_SMOOTHING = 3                     # D 值平滑：D = 昨日D × (1 - 1/3) + 今日K × 1/3
MA_PERIOD = 20                      # 月線（移動平均）天數
BUY_K_THRESHOLD = 20.0              # 買進：昨日 K < 20 的低檔黃金交叉
TAKE_PROFIT_K_THRESHOLD = 80.0      # 賣出 A：昨日 K > 80 的高檔死亡交叉
TAKE_PROFIT_BIAS_PERCENT = 3.5      # 賣出 A：且今日收盤月線正乖離 > 3.5%
TREND_BREAK_K_THRESHOLD = 50.0      # 賣出 B：收盤剛跌破月線且今日 K < 50

# 賣出 C：移動停利（每次回測只選一種，與賣出 A / B 任一成立即出場）
TRAILING_STOP_NONE = "none"                 # 不使用
TRAILING_STOP_MOVING_AVERAGE = "ma"         # 1. 均線型移動停利
TRAILING_STOP_PERCENT = "percent"           # 2a. 固定百分比回撤
TRAILING_STOP_ATR = "atr"                   # 2b. ATR 波動率回撤
TRAILING_STOP_SWING_LOW = "swing_low"       # 3. 前波低點防守
TRAILING_STOP_METHODS = (TRAILING_STOP_NONE, TRAILING_STOP_MOVING_AVERAGE, TRAILING_STOP_PERCENT,
                         TRAILING_STOP_ATR, TRAILING_STOP_SWING_LOW)
TRAILING_SOURCES = ("etf", "index")         # 停利判斷使用 0050 或加權指數的價格
MA_FILTERS = ("consecutive", "percent", "either", "both")   # 均線濾網：連續 N 日 / 跌破 X% / 任一 / 同時

TRAILING_STOP_METHOD = TRAILING_STOP_NONE   # 預設不使用，維持原策略
TRAILING_STOP_SOURCE = "etf"                # 預設以 0050 價格判斷
TRAILING_MA_PERIOD = 60                     # 1. 均線天數（20 = 月線積極型、60 = 季線穩健型）
TRAILING_MA_FILTER = "consecutive"          # 1. 濾網方式
TRAILING_MA_CONFIRM_DAYS = 3                # 1. 連續 N 個交易日收盤低於均線
TRAILING_MA_BREAK_PERCENT = 3.0             # 1. 收盤跌破均線超過 X%
TRAILING_DRAWDOWN_PERCENT = 10.0            # 2a. 收盤自進場後最高收盤回落 X%（建議 10 ~ 15）
ATR_PERIOD = 14                             # 2b. ATR 天數
ATR_MULTIPLIER = 3.0                        # 2b. 收盤跌破「進場後最高收盤 − N × ATR」（建議 2.5 ~ 3）
SWING_LOW_LOOKBACK = 20                     # 3. 前波低點 = 過去 N 個交易日（不含當日）的最低價

# 交易成本（買進：手續費；賣出：手續費 + 證交稅）；不計每筆最低手續費 20 元（回測不設定投入金額）
TRANSACTION_COSTS_ENABLED = True
FEE_RATE_PERCENT = 0.1425                   # 券商手續費率（%）
FEE_DISCOUNT = 1.0                          # 手續費折扣（1 = 不打折、0.28 = 2.8 折）
SELL_TAX_PERCENT = 0.1                      # 證券交易稅（%）；ETF 為 0.1%

# 參數允許範圍（與 API 查詢參數驗證相同）：(最小值, 最大值, 最小值是否不含)
PARAMETER_RANGES = {
    "kd_period": (2, 60, False),
    "k_smoothing": (1, 20, False),
    "d_smoothing": (1, 20, False),
    "ma_period": (2, 240, False),
    "buy_k_threshold": (0, 100, False),
    "take_profit_k_threshold": (0, 100, False),
    "take_profit_bias_percent": (-30, 50, False),
    "trend_break_k_threshold": (0, 100, False),
    "trailing_ma_period": (2, 240, False),
    "trailing_ma_confirm_days": (1, 20, False),
    "trailing_ma_break_percent": (0, 30, False),
    "trailing_drawdown_percent": (0, 50, True),
    "atr_period": (2, 60, False),
    "atr_multiplier": (0, 10, True),
    "swing_low_lookback": (2, 120, False),
    "fee_rate_percent": (0, 1, False),
    "fee_discount": (0, 1, False),
    "sell_tax_percent": (0, 1, False),
}
INTEGER_PARAMETERS = ("kd_period", "k_smoothing", "d_smoothing", "ma_period", "trailing_ma_period",
                      "trailing_ma_confirm_days", "atr_period", "swing_low_lookback")

# 參數最佳化：前段（訓練期）找累積報酬最高的參數，後段（驗證期）檢查是否依然有效
OPTIMIZE_MODE_RANDOM = "random"             # 隨機抽樣：從搜尋範圍中不重複抽出指定組數
OPTIMIZE_MODE_EXHAUSTIVE = "exhaustive"     # 完整搜尋：搜尋範圍內每一組都回測
OPTIMIZE_MODES = (OPTIMIZE_MODE_RANDOM, OPTIMIZE_MODE_EXHAUSTIVE)
OPTIMIZE_TRIALS = 20000                     # 預設隨機抽樣組數
OPTIMIZE_MAX_TRIALS = 500_000               # 隨機抽樣組數上限
OPTIMIZE_MAX_COMBINATIONS = 200_000_000     # 完整搜尋組合數上限（預設搜尋範圍約 1.14 億組）
OPTIMIZE_MAX_VALUES_PER_PARAMETER = 30      # 每個參數最多幾個候選值
OPTIMIZE_MAX_SEED = 2_147_483_647           # 亂數種子上限（網站端以 32 位元整數處理）
OPTIMIZE_TRAIN_RATIO = 0.7                  # 訓練期占回測交易日比例
OPTIMIZE_MIN_TRADES = 5                     # 訓練期至少完成幾輪交易（避免只靠一兩筆運氣好的交易）
OPTIMIZE_TOP_N = 10                         # 回傳前幾名
OPTIMIZE_PROGRESS_INTERVAL = 200            # 每回測幾組回報一次進度
OPTIMIZE_CACHE_SIZE = 1024                  # 指標快取最多保留幾組（搜尋範圍很大時避免記憶體無限增加）
# 預設搜尋範圍（網頁可自行調整；交易成本與停利判斷價格沿用頁面設定，不參與搜尋）
OPTIMIZE_SPACE = {
    "kd_period": [5, 7, 9, 12, 14, 18],
    "k_smoothing": [2, 3, 4, 5],
    "d_smoothing": [2, 3, 4, 5],
    "ma_period": [10, 20, 40, 60],
    "buy_k_threshold": [10, 15, 20, 25, 30, 40],
    "take_profit_k_threshold": [70, 75, 80, 85, 90],
    "take_profit_bias_percent": [2.0, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0],
    "trend_break_k_threshold": [30, 40, 50, 60, 70],
    "trailing_stop_method": list(TRAILING_STOP_METHODS),
    "trailing_ma_period": [20, 40, 60, 120],
    "trailing_ma_filter": list(MA_FILTERS),
    "trailing_ma_confirm_days": [1, 2, 3, 5],
    "trailing_ma_break_percent": [1.0, 2.0, 3.0, 5.0],
    "trailing_drawdown_percent": [5.0, 8.0, 10.0, 12.0, 15.0, 20.0],
    "atr_period": [10, 14, 20],
    "atr_multiplier": [1.5, 2.0, 2.5, 3.0, 4.0],
    "swing_low_lookback": [10, 20, 40, 60],
}
# 各停利方式實際用到的參數
TRAILING_METHOD_KEYS = {
    TRAILING_STOP_NONE: (),
    TRAILING_STOP_MOVING_AVERAGE: ("trailing_ma_period", "trailing_ma_filter", "trailing_ma_confirm_days", "trailing_ma_break_percent"),
    TRAILING_STOP_PERCENT: ("trailing_drawdown_percent",),
    TRAILING_STOP_ATR: ("atr_period", "atr_multiplier"),
    TRAILING_STOP_SWING_LOW: ("swing_low_lookback",),
}
ALL_TRAILING_KEYS = tuple(key for keys in TRAILING_METHOD_KEYS.values() for key in keys)
# 買賣條件參數（依序搜尋：指標天數放最前面，同一組指標的組合會連續回測，快取命中率高）
MAIN_SEARCH_KEYS = ("kd_period", "k_smoothing", "d_smoothing", "ma_period", "buy_k_threshold",
                    "take_profit_k_threshold", "take_profit_bias_percent", "trend_break_k_threshold")
CHOICE_OPTIONS = {"trailing_stop_method": TRAILING_STOP_METHODS, "trailing_ma_filter": MA_FILTERS}


@dataclass(frozen=True)
class KdBandParameters:
    """KD 波段策略參數（預設值為上方全域變數）。"""
    kd_period: int = KD_PERIOD
    k_smoothing: int = K_SMOOTHING
    d_smoothing: int = D_SMOOTHING
    ma_period: int = MA_PERIOD
    buy_k_threshold: float = BUY_K_THRESHOLD
    take_profit_k_threshold: float = TAKE_PROFIT_K_THRESHOLD
    take_profit_bias_percent: float = TAKE_PROFIT_BIAS_PERCENT
    trend_break_k_threshold: float = TREND_BREAK_K_THRESHOLD
    trailing_stop_method: str = TRAILING_STOP_METHOD
    trailing_stop_source: str = TRAILING_STOP_SOURCE
    trailing_ma_period: int = TRAILING_MA_PERIOD
    trailing_ma_filter: str = TRAILING_MA_FILTER
    trailing_ma_confirm_days: int = TRAILING_MA_CONFIRM_DAYS
    trailing_ma_break_percent: float = TRAILING_MA_BREAK_PERCENT
    trailing_drawdown_percent: float = TRAILING_DRAWDOWN_PERCENT
    atr_period: int = ATR_PERIOD
    atr_multiplier: float = ATR_MULTIPLIER
    swing_low_lookback: int = SWING_LOW_LOOKBACK
    transaction_costs_enabled: bool = TRANSACTION_COSTS_ENABLED
    fee_rate_percent: float = FEE_RATE_PERCENT
    fee_discount: float = FEE_DISCOUNT
    sell_tax_percent: float = SELL_TAX_PERCENT

    @property
    def buy_cost_rate(self) -> float:
        """買進成本比例（手續費 × 折扣）。"""
        return self.fee_rate_percent * self.fee_discount / 100 if self.transaction_costs_enabled else 0.0

    @property
    def sell_cost_rate(self) -> float:
        """賣出成本比例（手續費 × 折扣 + 證交稅）。"""
        return (self.fee_rate_percent * self.fee_discount + self.sell_tax_percent) / 100 if self.transaction_costs_enabled else 0.0

    def __post_init__(self):
        if self.trailing_stop_method not in TRAILING_STOP_METHODS:
            raise ValueError(f"Unknown trailing stop method: {self.trailing_stop_method}")
        if self.trailing_stop_source not in TRAILING_SOURCES:
            raise ValueError(f"Unknown trailing stop source: {self.trailing_stop_source}")
        if self.trailing_ma_filter not in MA_FILTERS:
            raise ValueError(f"Unknown moving average filter: {self.trailing_ma_filter}")
        for key, (minimum, maximum, exclusive_minimum) in PARAMETER_RANGES.items():
            value = getattr(self, key)
            too_small = value <= minimum if exclusive_minimum else value < minimum
            if too_small or value > maximum:
                raise ValueError(f"{key} must be between {minimum} and {maximum}: {value}")

    def to_camel_dict(self) -> Dict:
        """回傳給網頁的參數（camelCase）。"""
        return {_camel(key): value for key, value in asdict(self).items()}


def run_kd_band_backtest(twii: pd.DataFrame, etf: pd.DataFrame, parameters: Optional[KdBandParameters] = None) -> Dict:
    """以大盤 KD 產生訊號、用 0050 計算報酬。twii 需 High/Low/Close，etf 需 Open/High/Low/Close。

    指標只用大盤計算；所有判斷只使用今日（T）與昨日（T-1）的資料，訊號隔一個交易日才進出場，避免未來函數。
    """
    p = parameters or KdBandParameters()
    frame = _prepare_frame(twii, etf)
    frame["Signal"], sell_reasons = _generate_signals(frame, p)
    frame["Position"], _, frame["Cumulative_Return"] = _equity_curve(frame, frame["Signal"].to_numpy(), p)

    # 對照組：第一個交易日收盤買進 0050 後一路持有（同樣扣買進成本）
    frame["Buy_And_Hold"] = frame["ETF_0050"] / frame["ETF_0050"].iloc[0] * (1 - p.buy_cost_rate)

    signals: List[Dict] = []
    executions: List[Dict] = []
    for signal_date, row in frame[frame["Signal"] != 0].iterrows():
        location = frame.index.get_loc(signal_date)
        direction = "buy" if row["Signal"] == 1 else "sell"
        # 賣出訊號由哪些條件觸發（A：takeProfit、B：trendBreak、C：trailingStop）
        reasons = [name for name, hits in sell_reasons.items() if hits[location]] if direction == "sell" else []
        if location + 1 >= len(frame):
            signals.append({"direction": direction, "signalDate": signal_date.strftime("%Y-%m-%d"),
                            "executionDate": None, "price": None, "reasons": reasons})
            continue
        execution_date = frame.index[location + 1]
        price = float(frame.loc[execution_date, "ETF_0050"])
        signals.append({"direction": direction, "signalDate": signal_date.strftime("%Y-%m-%d"),
                        "executionDate": execution_date.strftime("%Y-%m-%d"), "price": to_float(price, 2),
                        "reasons": reasons})
        executions.append({"date": execution_date, "direction": direction, "price": price})

    # 配對進場 / 出場
    round_trips: List[Dict] = []
    entry = None
    for execution in executions:
        if execution["direction"] == "buy" and entry is None:
            entry = execution
        elif execution["direction"] == "sell" and entry is not None:
            price_ratio = execution["price"] / entry["price"]
            round_trips.append({
                "entryDate": entry["date"].strftime("%Y-%m-%d"),
                "exitDate": execution["date"].strftime("%Y-%m-%d"),
                "holdingDays": (execution["date"] - entry["date"]).days,
                "returnPercent": to_float((price_ratio - 1) * 100, 2),
                "netReturnPercent": to_float((price_ratio * (1 - p.buy_cost_rate) * (1 - p.sell_cost_rate) - 1) * 100, 2),
                "isOpen": False,
            })
            entry = None
    if entry is not None:
        # 尚未出場：出場日為 None，持有天數與報酬（未實現，只扣買進成本）算到最新收盤
        last_date = frame.index[-1]
        price_ratio = float(frame["ETF_0050"].iloc[-1]) / entry["price"]
        round_trips.append({
            "entryDate": entry["date"].strftime("%Y-%m-%d"),
            "exitDate": None,
            "holdingDays": (last_date - entry["date"]).days,
            "returnPercent": to_float((price_ratio - 1) * 100, 2),
            "netReturnPercent": to_float((price_ratio * (1 - p.buy_cost_rate) - 1) * 100, 2),
            "isOpen": True,
        })

    candles = etf.reindex(frame.index)
    return {
        "startDate": frame.index[0].strftime("%Y-%m-%d"),
        "endDate": frame.index[-1].strftime("%Y-%m-%d"),
        "parameters": p.to_camel_dict(),
        "cumulativeReturnPercent": to_float((frame["Cumulative_Return"].iloc[-1] - 1) * 100, 2),
        "buyAndHoldReturnPercent": to_float((frame["Buy_And_Hold"].iloc[-1] - 1) * 100, 2),
        "exposurePercent": to_float(frame["Position"].mean() * 100, 1),     # 有部位的交易日比例
        "buyCostPercent": to_float(p.buy_cost_rate * 100, 4),
        "sellCostPercent": to_float(p.sell_cost_rate * 100, 4),
        "signalCount": len(signals),
        "signals": signals,
        "roundTrips": round_trips,
        "openPositionSince": entry["date"].strftime("%Y-%m-%d") if entry is not None else None,
        "candles": {
            "dates": dates_to_list(candles.index),
            "open": series_to_list(candles["Open"], 2),
            "high": series_to_list(candles["High"], 2),
            "low": series_to_list(candles["Low"], 2),
            "close": series_to_list(candles["Close"], 2),
        },
        "equityCurve": series_to_list(frame["Cumulative_Return"]),
        "buyAndHoldCurve": series_to_list(frame["Buy_And_Hold"]),
        "buyMarkers": [e["date"].strftime("%Y-%m-%d") for e in executions if e["direction"] == "buy"],
        "sellMarkers": [e["date"].strftime("%Y-%m-%d") for e in executions if e["direction"] == "sell"],
    }


def _prepare_frame(twii: pd.DataFrame, etf: pd.DataFrame) -> pd.DataFrame:
    """整理回測用資料：以大盤交易日為準，0050 偶有缺漏的日子（Yahoo 資料缺漏）沿用前一日價格，
    避免整列被刪掉而讓大盤 KD 少算一天。"""
    frame = pd.DataFrame({
        "TWII_H": twii["High"],
        "TWII_L": twii["Low"],
        "TWII_C": twii["Close"],
    }).dropna()
    frame["ETF_0050"] = etf["Close"].reindex(frame.index).ffill()
    frame["ETF_H"] = etf["High"].reindex(frame.index).ffill()
    frame["ETF_L"] = etf["Low"].reindex(frame.index).ffill()
    return frame.dropna(subset=["ETF_0050"])


class IndicatorCache:
    """最佳化用的指標快取：同一組指標天數只計算一次，只保留最近使用的 max_size 組，避免記憶體無限增加。"""

    def __init__(self, max_size: int = OPTIMIZE_CACHE_SIZE):
        self._items: "OrderedDict[tuple, object]" = OrderedDict()
        self._max_size = max_size

    def get_or_compute(self, key: tuple, compute):
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        value = compute()
        self._items[key] = value
        if len(self._items) > self._max_size:
            self._items.popitem(last=False)
        return value


def _cached(cache: Optional[IndicatorCache], key: tuple, compute):
    """有快取時同一組指標天數只計算一次。"""
    return compute() if cache is None else cache.get_or_compute(key, compute)


def _signal_inputs(frame: pd.DataFrame, p: KdBandParameters, cache: Optional[IndicatorCache]) -> Dict[str, np.ndarray]:
    """只與 KD 天數、平滑天數、月線天數有關的逐日陣列（交叉、乖離、跌破月線），與門檻無關。"""
    def compute_kd():
        # 大盤 KD：先算 RSV，再以 1/平滑天數 為平滑係數計算 K、D（預設 (9, 3, 3) 即傳統 1/3）
        low_min = frame["TWII_L"].rolling(window=p.kd_period).min()
        high_max = frame["TWII_H"].rolling(window=p.kd_period).max()
        rsv = 100 * (frame["TWII_C"] - low_min) / (high_max - low_min)
        k = rsv.ewm(alpha=1 / p.k_smoothing, adjust=False).mean()
        return k, k.ewm(alpha=1 / p.d_smoothing, adjust=False).mean()

    def compute():
        k, d = _cached(cache, ("kd", p.kd_period, p.k_smoothing, p.d_smoothing), compute_kd)
        # 月線與正乖離率 (Close - MA) / MA
        sma = _cached(cache, ("sma", p.ma_period), lambda: frame["TWII_C"].rolling(window=p.ma_period).mean())
        close = frame["TWII_C"]
        previous_k, previous_d = k.shift(1), d.shift(1)
        return {
            "k": k.to_numpy(),
            "previousK": previous_k.to_numpy(),
            "goldenCross": ((k > d) & (previous_k <= previous_d)).to_numpy(),   # 今日 K > D 且 昨日 K <= D
            "deathCross": ((k < d) & (previous_k >= previous_d)).to_numpy(),    # 今日 K < D 且 昨日 K >= D
            "bias": ((close - sma) / sma * 100).to_numpy(),
            "maBreak": ((close < sma) & (close.shift(1) >= sma.shift(1))).to_numpy(),   # 今日收盤 < 月線 且 昨日收盤 >= 昨日月線
        }

    return _cached(cache, ("signalInputs", p.kd_period, p.k_smoothing, p.d_smoothing, p.ma_period), compute)


def _generate_signals(frame: pd.DataFrame, p: KdBandParameters, cache: Optional[IndicatorCache] = None):
    """產生每日訊號（1 買進、-1 賣出、0 無）與各賣出條件觸發的日子；只使用今日（T）與昨日（T-1）的資料。"""
    inputs = _signal_inputs(frame, p, cache)
    with np.errstate(invalid="ignore"):   # 指標暖身期為 NaN，比較結果為 False（不產生訊號）
        # 買進：KD 黃金交叉 且 昨日 K < 低檔門檻
        buy_signal = inputs["goldenCross"] & (inputs["previousK"] < p.buy_k_threshold)
        # 賣出 A（高檔過熱死叉）：死亡交叉 且 昨日 K > 高檔門檻 且 今日月線正乖離 > 門檻
        sell_take_profit = (inputs["deathCross"] & (inputs["previousK"] > p.take_profit_k_threshold)
                            & (inputs["bias"] > p.take_profit_bias_percent))
        # 賣出 B（趨勢轉弱）：今日收盤 < 月線 且 昨日收盤 >= 昨日月線 且 今日 K < 門檻
        sell_trend_break = inputs["maBreak"] & (inputs["k"] < p.trend_break_k_threshold)

    # 賣出 C（移動停利）需要知道是否持有與進場後的最高價，逐日模擬持倉
    # 買進 1、賣出 -1（同日同時成立時以賣出為準，與原腳本相同）
    trailing_series = None
    if p.trailing_stop_method != TRAILING_STOP_NONE:
        trailing_key = ("trailing", p.trailing_stop_source, p.trailing_ma_period, p.trailing_ma_filter,
                        p.trailing_ma_confirm_days, p.trailing_ma_break_percent, p.atr_period, p.swing_low_lookback)
        trailing_series = _cached(cache, trailing_key, lambda: _trailing_stop_series(frame, p))
    signals, trailing_hit = _simulate_signals(buy_signal, sell_take_profit | sell_trend_break, trailing_series, p)
    return signals, {
        "takeProfit": sell_take_profit,
        "trendBreak": sell_trend_break,
        "trailingStop": trailing_hit,
    }


def _equity_curve(frame: pd.DataFrame, signals: np.ndarray, p: KdBandParameters, cache: Optional[IndicatorCache] = None):
    """依訊號計算每日部位與累積報酬（淨值，起始為 1）。

    T 日收盤出現訊號，T+1 日以收盤價成交（與每輪交易表相同），避免用訊號當天的收盤價成交；
    成交日扣交易成本：買進扣手續費，賣出扣手續費 + 證交稅。
    """
    signals = np.asarray(signals)
    # 每天沿用最近一次訊號：最近一次為買進則持有，賣出或尚無訊號則空手
    last_signal_index = np.maximum.accumulate(np.where(signals != 0, np.arange(len(signals)), 0))
    signal_state = np.where(signals[last_signal_index] == 1, 1.0, 0.0)
    held_after_close = np.concatenate(([0.0], signal_state[:-1]))          # 第 d 日收盤成交後是否持有
    position_change = np.diff(held_after_close, prepend=0.0)               # +1 買進成交日、-1 賣出成交日
    position = np.concatenate(([0.0], held_after_close[:-1]))              # 第 d 日的漲跌是否計入（前一日收盤已持有）
    trade_cost = np.where(position_change > 0, p.buy_cost_rate, np.where(position_change < 0, p.sell_cost_rate, 0.0))
    daily_return = _cached(cache, ("dailyReturn",), lambda: frame["ETF_0050"].pct_change().fillna(0).to_numpy())
    cumulative = np.cumprod((1 + daily_return * position) * (1 - trade_cost))
    return position, position_change, cumulative


def _camel(key: str) -> str:
    return "".join(part.capitalize() if i else part for i, part in enumerate(key.split("_")))


@dataclass(frozen=True)
class SearchSpace:
    """參數搜尋範圍：每個參數的候選值，未列入搜尋的參數固定為目前設定。

    每組參數對應一個編號（0 ~ total-1）：編號 = 買賣條件參數的組合 × 停利方式（含該方式自己的參數）的組合；
    沒用到的停利參數固定為目前設定，避免實質相同的組合重複計算。
    """
    base: KdBandParameters
    main_options: Tuple[Tuple[str, Tuple], ...]
    trailing_options: Tuple[Tuple[str, Tuple[Tuple[str, Tuple], ...], int], ...]   # (停利方式, 該方式的參數與候選值, 組合數)
    searched: Tuple[Tuple[str, Tuple], ...]                                        # 使用者指定要搜尋的參數

    @property
    def trailing_count(self) -> int:
        return sum(count for _, _, count in self.trailing_options)

    @property
    def total(self) -> int:
        return math.prod(len(options) for _, options in self.main_options) * self.trailing_count

    def parameters_at(self, index: int) -> KdBandParameters:
        """第 index 組參數；KD 天數變化最慢，同一組指標天數的組合編號相鄰（快取命中率高）。"""
        main_index, trailing_index = divmod(index, self.trailing_count)
        values = {}
        for key, options in reversed(self.main_options):
            main_index, position = divmod(main_index, len(options))
            values[key] = options[position]
        for method, method_options, count in self.trailing_options:
            if trailing_index < count:
                values["trailing_stop_method"] = method
                for key, options in reversed(method_options):
                    trailing_index, position = divmod(trailing_index, len(options))
                    values[key] = options[position]
                break
            trailing_index -= count
        return replace(self.base, **values)

    def searched_camel(self) -> Dict[str, List]:
        """回傳給網頁的搜尋範圍（camelCase）。"""
        return {_camel(key): list(values) for key, values in self.searched}


def _normalize_search_value(key: str, value):
    if key in CHOICE_OPTIONS:
        if value not in CHOICE_OPTIONS[key]:
            raise ValueError(f"Invalid {key}: {value!r}")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a number: {value!r}")
    if key in INTEGER_PARAMETERS:
        if value != int(value):
            raise ValueError(f"{key} must be an integer: {value!r}")
        return int(value)
    return float(value)


def build_search_space(base: KdBandParameters, space: Optional[Dict[str, Sequence]] = None) -> SearchSpace:
    """依目前設定與各參數候選值建立搜尋範圍；space 為 None 時使用 OPTIMIZE_SPACE，空清單代表不搜尋該參數。"""
    space = OPTIMIZE_SPACE if space is None else space
    unknown = sorted(set(space) - set(OPTIMIZE_SPACE))
    if unknown:
        raise ValueError(f"Unknown search parameters: {unknown}")
    searched: Dict[str, Tuple] = {}
    for key in OPTIMIZE_SPACE:   # 依固定順序，讓同樣的範圍產生同樣的編號
        values = tuple(dict.fromkeys(_normalize_search_value(key, value) for value in space.get(key) or ()))
        if not values:
            continue
        if len(values) > OPTIMIZE_MAX_VALUES_PER_PARAMETER:
            raise ValueError(f"{key} has more than {OPTIMIZE_MAX_VALUES_PER_PARAMETER} values")
        for value in values:
            replace(base, **{key: value})   # 超出允許範圍時丟出 ValueError
        searched[key] = values

    options_of = lambda key: searched.get(key) or (getattr(base, key),)
    trailing_options = []
    for method in options_of("trailing_stop_method"):
        method_options = tuple((key, options_of(key)) for key in TRAILING_METHOD_KEYS[method])
        trailing_options.append((method, method_options, math.prod(len(options) for _, options in method_options)))
    return SearchSpace(
        base=base,
        main_options=tuple((key, options_of(key)) for key in MAIN_SEARCH_KEYS),
        trailing_options=tuple(trailing_options),
        searched=tuple(searched.items()),
    )


# 進度回報：(已回測組數, 總組數, 目前訓練期最佳的一組或 None)；回呼丟出例外即中止搜尋
ProgressCallback = Callable[[int, int, Optional[Dict]], None]


def optimize_kd_band(
    twii: pd.DataFrame,
    etf: pd.DataFrame,
    base: Optional[KdBandParameters] = None,
    space: Optional[Dict[str, Sequence]] = None,
    mode: str = OPTIMIZE_MODE_RANDOM,
    trials: int = OPTIMIZE_TRIALS,
    seed: Optional[int] = None,
    train_ratio: float = OPTIMIZE_TRAIN_RATIO,
    min_trades: int = OPTIMIZE_MIN_TRADES,
    top_n: int = OPTIMIZE_TOP_N,
    progress: Optional[ProgressCallback] = None,
) -> Dict:
    """搜尋參數組合，依「訓練期」累積報酬排序，並列出同一組參數在「驗證期」與全期間的報酬。

    - 隨機抽樣（random）：從搜尋範圍不重複抽出 trials 組；同一個亂數種子每次抽到同一批，未指定種子時隨機產生。
      trials 大於等於總組合數時，等同完整搜尋。
    - 完整搜尋（exhaustive）：搜尋範圍內每一組都回測。
    訓練期 = 回測期間前 train_ratio 的交易日，驗證期 = 其餘交易日；排序只看訓練期，驗證期的資料不參與挑選。
    每組參數都在完整資料上回測（指標暖身與持倉可跨越分界日），再切出兩段的淨值變化。
    """
    if mode not in OPTIMIZE_MODES:
        raise ValueError(f"Unknown optimize mode: {mode}")
    if trials < 1:
        raise ValueError("trials must be at least 1")
    base = base or KdBandParameters()
    search = build_search_space(base, space)
    frame = _prepare_frame(twii, etf)
    split = int(len(frame) * train_ratio)
    if split < 60 or len(frame) - split < 20:
        raise ValueError("回測期間太短，無法切分訓練期與驗證期，請提早回測起始日")

    started = timer.perf_counter()
    if mode == OPTIMIZE_MODE_RANDOM and seed is None:
        seed = secrets.randbelow(OPTIMIZE_MAX_SEED) + 1
    indices = _candidate_indices(search.total, mode, trials, seed)
    cache = IndicatorCache()
    current = _evaluate_parameters(frame, base, split, cache)

    # 只保留訓練期報酬最高的 top_n 組（報酬相同時編號小的優先）；完整搜尋組數很大時不必保存每一組
    best: List[Tuple[float, int, Dict]] = []
    qualified = 0
    for order, index in enumerate(indices, start=1):
        parameters = search.parameters_at(index)
        row = _evaluate_parameters(frame, parameters, split, cache)
        if row["trainTrades"] >= min_trades:
            qualified += 1
            item = (row["trainReturnPercent"], -order, row)
            if len(best) < top_n:
                heapq.heappush(best, item)
            elif item[:2] > best[0][:2]:
                heapq.heapreplace(best, item)
        if progress is not None and order % OPTIMIZE_PROGRESS_INTERVAL == 0:
            progress(order, len(indices), _progress_best(best))
    if progress is not None:
        progress(len(indices), len(indices), _progress_best(best))

    top = [row for _, _, row in sorted(best, key=lambda item: item[:2], reverse=True)]
    for row in [current] + top:
        row["isCurrent"] = row["parameters"] == base
        row["parameters"] = row["parameters"].to_camel_dict()

    closes = frame["ETF_0050"].to_numpy()
    buy_hold = lambda start, end: to_float((closes[end] / closes[start] - 1) * 100, 2)
    return {
        "mode": mode,
        "seed": seed if mode == OPTIMIZE_MODE_RANDOM else None,
        "space": search.searched_camel(),
        "totalCombinations": search.total,
        "trials": len(indices),
        "evaluatedAll": len(indices) == search.total,
        "elapsedSeconds": to_float(timer.perf_counter() - started, 1),
        "startDate": frame.index[0].strftime("%Y-%m-%d"),
        "splitDate": frame.index[split].strftime("%Y-%m-%d"),
        "endDate": frame.index[-1].strftime("%Y-%m-%d"),
        "trainDays": split,
        "validationDays": len(frame) - 1 - split,
        "qualifiedTrials": qualified,
        "minTrades": min_trades,
        "buyAndHold": {
            "trainReturnPercent": buy_hold(0, split),
            "validationReturnPercent": buy_hold(split, len(frame) - 1),
            "fullReturnPercent": buy_hold(0, len(frame) - 1),
        },
        "current": current,
        "top": top,
    }


def _candidate_indices(total: int, mode: str, trials: int, seed: Optional[int]) -> Sequence[int]:
    """要回測的組合編號（由小到大）：完整搜尋為全部；隨機抽樣為不重複抽出 trials 組。"""
    if mode == OPTIMIZE_MODE_EXHAUSTIVE or trials >= total:
        return range(total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=trials, replace=False)).tolist()


def _evaluate_parameters(frame: pd.DataFrame, parameters: KdBandParameters, split: int, cache: IndicatorCache) -> Dict:
    """回測一組參數，切出訓練期、驗證期與全期間報酬（parameters 先保留物件，排名確定後才轉成 camelCase）。"""
    signals, _ = _generate_signals(frame, parameters, cache)
    _, position_change, curve = _equity_curve(frame, signals, parameters, cache)
    entry_count = int(np.count_nonzero(position_change > 0))
    exits = np.flatnonzero(position_change < 0)
    return {
        "parameters": parameters,
        "trainReturnPercent": to_float((curve[split] / curve[0] - 1) * 100, 2),
        "validationReturnPercent": to_float((curve[-1] / curve[split] - 1) * 100, 2),
        "fullReturnPercent": to_float((curve[-1] - 1) * 100, 2),
        "trainTrades": int(np.count_nonzero(exits <= split)),
        "totalTrades": int(len(exits)),
        "openPosition": entry_count > len(exits),
        "exposurePercent": to_float(np.mean(np.concatenate(([0.0], (np.cumsum(position_change) > 0)[:-1]))) * 100, 1),
    }


def _progress_best(best: List[Tuple[float, int, Dict]]) -> Optional[Dict]:
    if not best:
        return None
    row = max(best, key=lambda item: item[:2])[2]
    return {"trainReturnPercent": row["trainReturnPercent"], "validationReturnPercent": row["validationReturnPercent"]}


def _trailing_stop_series(frame: pd.DataFrame, p: KdBandParameters) -> Dict[str, np.ndarray]:
    """移動停利需要的序列（只用當日與過去資料）：判斷用收盤、均線跌破觸發、ATR、前波低點。"""
    if p.trailing_stop_source == "index":
        close, high, low = frame["TWII_C"], frame["TWII_H"], frame["TWII_L"]
    else:
        close, high, low = frame["ETF_0050"], frame["ETF_H"], frame["ETF_L"]

    # 1. 均線：連續 N 日收盤低於均線、或收盤低於均線超過 X%；條件「剛成立」的那天觸發（由上往下跌破）
    moving_average = close.rolling(window=p.trailing_ma_period).mean()
    below = close < moving_average
    consecutive_below = below.astype(int).groupby((~below).cumsum()).cumsum()
    by_days = consecutive_below >= p.trailing_ma_confirm_days
    by_percent = close < moving_average * (1 - p.trailing_ma_break_percent / 100)
    condition = {
        "consecutive": by_days,
        "percent": by_percent,
        "either": by_days | by_percent,
        "both": by_days & by_percent,
    }[p.trailing_ma_filter]
    ma_trigger = condition & ~condition.shift(1, fill_value=False)

    # 2b. ATR（Wilder 平滑）
    previous_close = close.shift(1)
    true_range = pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / p.atr_period, adjust=False).mean()

    # 3. 前波低點：過去 N 個交易日（不含當日）的最低價
    prior_low = low.rolling(window=p.swing_low_lookback).min().shift(1)

    return {
        "close": close.to_numpy(dtype=float),
        "maTrigger": ma_trigger.to_numpy(dtype=bool),
        "atr": atr.to_numpy(dtype=float),
        "priorLow": prior_low.to_numpy(dtype=float),
    }


def _simulate_signals(buy: np.ndarray, base_sell: np.ndarray, series: Optional[Dict[str, np.ndarray]], p: KdBandParameters):
    """逐日產生訊號（1 買進、-1 賣出、0 無）；回傳訊號與移動停利觸發的日子。

    進場執行日為買進訊號的隔一個交易日，從該日起記錄進場後最高收盤並檢查移動停利；
    不使用移動停利時與原本「賣出優先、否則買進」的訊號完全相同（不需逐日模擬）。
    """
    method = p.trailing_stop_method
    trailing_hit = np.zeros(len(buy), dtype=bool)
    if method == TRAILING_STOP_NONE or series is None:
        return np.where(base_sell, -1, np.where(buy, 1, 0)), trailing_hit
    # 逐日迴圈改用 Python list 存取（比逐一讀取 numpy 元素快數倍，數值完全相同）
    closes, atr, prior_low, ma_trigger = (np.asarray(series[key]).tolist() for key in ("close", "atr", "priorLow", "maTrigger"))
    buy_days, sell_days = np.asarray(buy).tolist(), np.asarray(base_sell).tolist()
    signals = np.zeros(len(buy), dtype=int)
    holding, entry_signal_index = False, -1
    highest, stop_level = math.nan, math.nan
    drawdown_ratio = 1 - p.trailing_drawdown_percent / 100

    for i in range(len(buy_days)):
        hit = False
        if holding and i > entry_signal_index:
            close = closes[i]
            is_new_high = math.isnan(highest) or close > highest
            if is_new_high:
                highest = close
            if method == TRAILING_STOP_MOVING_AVERAGE:
                hit = bool(ma_trigger[i])
            elif method == TRAILING_STOP_PERCENT:
                # 收盤自進場後最高收盤回落 X% 以上
                hit = close <= highest * drawdown_ratio
            elif method == TRAILING_STOP_ATR:
                # 收盤跌破「進場後最高收盤 − N × ATR」
                hit = not math.isnan(atr[i]) and close < highest - p.atr_multiplier * atr[i]
            elif method == TRAILING_STOP_SWING_LOW:
                # 停利點 = 前波低點；每次創新高時上移到最新的前波低點（只上移不下移）
                if is_new_high and not math.isnan(prior_low[i]):
                    stop_level = prior_low[i] if math.isnan(stop_level) else max(stop_level, prior_low[i])
                hit = not math.isnan(stop_level) and close < stop_level

        if sell_days[i] or hit:
            signals[i] = -1
            trailing_hit[i] = hit
            holding = False
        elif buy_days[i]:
            signals[i] = 1
            if not holding:
                holding, entry_signal_index = True, i
                highest, stop_level = math.nan, math.nan
    return signals, trailing_hit


# ---------------------------------------------------------------------------
# 大盤回檔後隔月反彈統計（market_rebound.py）
# ---------------------------------------------------------------------------
MONTH_TRADING_DAYS = 21                                    # 約一個月交易日
REBOUND_BINS = [-np.inf, -9, -8, -7, -6, -5, -4, -3]       # 只看單日跌 3% 以上
STRONG_UP_RATE = 55                                        # 隔月上漲率 > 55% 特別標示
MARKET_CLOSE_TIME = time(13, 30)                           # 台股收盤時間（台北時間）


def drop_unclosed_session(frame: pd.DataFrame, taipei_now: datetime) -> pd.DataFrame:
    """盤中（台北時間 13:30 前）Yahoo 會回傳當日即時價，尚非收盤價，先排除，只統計到最新收盤日。"""
    if frame.empty or taipei_now.time() >= MARKET_CLOSE_TIME:
        return frame
    if frame.index[-1].date() == taipei_now.date():
        return frame.iloc[:-1]
    return frame


def _bin_labels() -> List[str]:
    labels = []
    for low, high in zip(REBOUND_BINS[:-1], REBOUND_BINS[1:]):
        labels.append(f"< {high:.1f}%" if low == -np.inf else f"{low:.1f}% ~ {high:.1f}%")
    return labels


def analyze_rebound(index_frame: pd.DataFrame) -> Dict:
    """統計大盤單日下跌各區間之後，約一個月（21 個交易日）的表現。

    資料保留到最新收盤日；最近未滿 21 個交易日的回檔日，隔月漲跌尚未產生（回傳 None），
    列在明細與 pendingCount，但不納入上漲率與平均值的計算。
    """
    frame = index_frame[["Close"]].copy()
    frame["Return_T"] = frame["Close"].pct_change() * 100
    # 隔月 = 往後第 21 個交易日的收盤；隔月漲跌 = (隔月收盤 / 當日收盤 - 1) × 100
    frame["Close_T1"] = frame["Close"].shift(-MONTH_TRADING_DAYS)
    frame["Date_T1"] = pd.Series(frame.index, index=frame.index).shift(-MONTH_TRADING_DAYS)
    frame["Return_T1"] = (frame["Close_T1"] / frame["Close"] - 1) * 100
    # 只去掉第一列（沒有前一日可比較）；隔月報酬為 NaN 的最近一個月要保留
    frame = frame.dropna(subset=["Return_T"])
    settled_frame = frame.dropna(subset=["Return_T1"])

    labels = _bin_labels()
    declines = frame[frame["Return_T"] < 0].copy()
    declines["Bin"] = pd.cut(declines["Return_T"], bins=REBOUND_BINS, labels=labels)
    declines = declines.dropna(subset=["Bin"])

    buckets = []
    for label in labels:
        group = declines[declines["Bin"] == label]
        if group.empty:
            continue
        settled = group.dropna(subset=["Return_T1"])
        up_count = int((settled["Return_T1"] > 0).sum())
        up_rate = up_count / len(settled) * 100 if len(settled) else None
        buckets.append({
            "range": label,
            "sampleCount": len(settled),               # 已有隔月結果的樣本數（上漲率的分母）
            "pendingCount": len(group) - len(settled), # 未滿 21 個交易日、尚無隔月結果
            "upCount": up_count,
            "upRate": to_float(up_rate, 1),
            "averageNextMonth": to_float(settled["Return_T1"].mean(), 2),
            "medianNextMonth": to_float(settled["Return_T1"].median(), 2),
            "averageDecline": to_float(group["Return_T"].mean(), 2),
            "isStrong": up_rate is not None and bool(round(up_rate, 1) > STRONG_UP_RATE),
        })

    details = declines[declines["Return_T"] <= -3].sort_index()
    detail_rows = [{
        "date": date.strftime("%Y-%m-%d"),
        "close": to_float(row["Close"], 2),
        "dailyReturn": to_float(row["Return_T"], 2),
        # 隔月（第 21 個交易日）的日期與收盤；未滿 21 個交易日時為 None
        "nextMonthDate": row["Date_T1"].strftime("%Y-%m-%d") if pd.notna(row["Date_T1"]) else None,
        "nextMonthClose": to_float(row["Close_T1"], 2),
        "nextMonthReturn": to_float(row["Return_T1"], 2),
    } for date, row in details.iterrows()]
    settled_details = details.dropna(subset=["Return_T1"])
    detail_up_count = int((settled_details["Return_T1"] > 0).sum())

    return {
        "startDate": frame.index[0].strftime("%Y-%m-%d"),
        "endDate": frame.index[-1].strftime("%Y-%m-%d"),
        "latestClose": to_float(frame["Close"].iloc[-1], 2),
        # 最後一個已有隔月結果的交易日；之後的日期隔月漲跌尚未產生
        "settledEndDate": settled_frame.index[-1].strftime("%Y-%m-%d") if len(settled_frame) else None,
        "tradingDays": len(frame),
        "monthTradingDays": MONTH_TRADING_DAYS,
        "buckets": buckets,
        "details": detail_rows,
        "detailUpCount": detail_up_count,
        "detailDownCount": len(settled_details) - detail_up_count,
        "detailPendingCount": len(details) - len(settled_details),
    }


# ---------------------------------------------------------------------------
# 融資餘額 + 近似融資維持率（market_margin.py）
# ---------------------------------------------------------------------------
MARGIN_LOAN_RATE = 0.60   # 融資成數假設（台股多數股票適用）


def analyze_margin(index_frame: pd.DataFrame, margin_records: List[Dict]) -> Dict:
    """合併加權指數與每日融資融券資料，並估算融資維持率。

    近似維持率 = 加權指數 / (融資成數 0.6 × 加權指數 60 日均) × 100，僅供趨勢參考。
    """
    index_close = index_frame[["Close"]].copy()
    ma60 = index_close["Close"].rolling(60).mean()

    margin = pd.DataFrame(margin_records).set_index("date")
    margin.index = pd.to_datetime(margin.index)

    combined = index_close.join(margin, how="inner").dropna()
    combined["MA60"] = ma60.reindex(combined.index)
    combined["ApproximateRatio"] = (combined["Close"] / (MARGIN_LOAN_RATE * combined["MA60"]) * 100).round(1)
    combined = combined.dropna(subset=["ApproximateRatio"])

    return {
        "dates": dates_to_list(combined.index),
        "indexClose": series_to_list(combined["Close"], 2),
        # 融資金額單位為仟元，/ 100,000 換算為億元
        "marginBalance100Million": series_to_list(combined["margin_amount_thousand"] / 100_000, 1),
        "marginLots": series_to_list(combined["margin_lots"], 0),
        # 融券單位為張，/ 10,000 換算為萬張
        "shortBalance10kLots": series_to_list(combined["short_lots"] / 10_000, 2),
        "approximateRatio": series_to_list(combined["ApproximateRatio"], 1),
        "loanRateAssumption": MARGIN_LOAN_RATE,
    }
