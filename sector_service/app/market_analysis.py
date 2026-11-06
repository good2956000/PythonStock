"""大盤層級的分析（Domain 層，輸入為已下載的 DataFrame，不直接連網）。

- run_kd_band_backtest : 移植自 PythonStock/stock_all.py      （大盤 KD 波段策略回測 0050）
- analyze_rebound      : 移植自 PythonStock/market_rebound.py （大盤回檔後隔月反彈統計）
- analyze_margin       : 移植自 PythonStock/market_margin.py  （融資餘額 + 近似融資維持率）
"""
from dataclasses import asdict, dataclass, replace
from datetime import datetime, time
from typing import Dict, List, Optional

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

# 參數最佳化：前段（訓練期）找累積報酬最高的參數，後段（驗證期）檢查是否依然有效
OPTIMIZE_TRIALS = 2000                      # 預設隨機搜尋組數（每 1000 組約 3 秒）
OPTIMIZE_TRAIN_RATIO = 0.7                  # 訓練期占回測交易日比例
OPTIMIZE_MIN_TRADES = 5                     # 訓練期至少完成幾輪交易（避免只靠一兩筆運氣好的交易）
OPTIMIZE_TOP_N = 10                         # 回傳前幾名
OPTIMIZE_RANDOM_SEED = 20260101             # 固定亂數種子：同樣條件每次搜尋結果相同
# 搜尋範圍（每組參數從各清單隨機挑一個值；交易成本與停利判斷價格沿用頁面設定，不參與搜尋）
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

    def to_camel_dict(self) -> Dict:
        """回傳給網頁的參數（camelCase）。"""
        return {
            "".join(part.capitalize() if i else part for i, part in enumerate(key.split("_"))): value
            for key, value in asdict(self).items()
        }


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


def _cached(cache: Optional[Dict], key: tuple, compute):
    """最佳化時同一組指標天數只計算一次。"""
    if cache is None:
        return compute()
    if key not in cache:
        cache[key] = compute()
    return cache[key]


def _generate_signals(frame: pd.DataFrame, p: KdBandParameters, cache: Optional[Dict] = None):
    """產生每日訊號（1 買進、-1 賣出、0 無）與各賣出條件觸發的日子；只使用今日（T）與昨日（T-1）的資料。"""
    def compute_kd():
        # 大盤 KD：先算 RSV，再以 1/平滑天數 為平滑係數計算 K、D（預設 (9, 3, 3) 即傳統 1/3）
        low_min = frame["TWII_L"].rolling(window=p.kd_period).min()
        high_max = frame["TWII_H"].rolling(window=p.kd_period).max()
        rsv = 100 * (frame["TWII_C"] - low_min) / (high_max - low_min)
        k = rsv.ewm(alpha=1 / p.k_smoothing, adjust=False).mean()
        return k, k.ewm(alpha=1 / p.d_smoothing, adjust=False).mean()

    k, d = _cached(cache, ("kd", p.kd_period, p.k_smoothing, p.d_smoothing), compute_kd)
    # 月線與正乖離率 (Close - MA) / MA
    sma = _cached(cache, ("sma", p.ma_period), lambda: frame["TWII_C"].rolling(window=p.ma_period).mean())
    bias = (frame["TWII_C"] - sma) / sma * 100

    previous_k, previous_d = k.shift(1), d.shift(1)
    golden_cross = (k > d) & (previous_k <= previous_d)   # 今日 K > D 且 昨日 K <= D
    death_cross = (k < d) & (previous_k >= previous_d)    # 今日 K < D 且 昨日 K >= D

    # 買進：KD 黃金交叉 且 昨日 K < 低檔門檻
    buy_signal = golden_cross & (previous_k < p.buy_k_threshold)
    # 賣出 A（高檔過熱死叉）：死亡交叉 且 昨日 K > 高檔門檻 且 今日月線正乖離 > 門檻
    sell_take_profit = death_cross & (previous_k > p.take_profit_k_threshold) & (bias > p.take_profit_bias_percent)
    # 賣出 B（趨勢轉弱）：今日收盤 < 月線 且 昨日收盤 >= 昨日月線 且 今日 K < 門檻
    sell_trend_break = (frame["TWII_C"] < sma) & (frame["TWII_C"].shift(1) >= sma.shift(1)) & (k < p.trend_break_k_threshold)

    # 賣出 C（移動停利）需要知道是否持有與進場後的最高價，逐日模擬持倉
    # 買進 1、賣出 -1（同日同時成立時以賣出為準，與原腳本相同）
    trailing_series = None
    if p.trailing_stop_method != TRAILING_STOP_NONE:
        trailing_key = ("trailing", p.trailing_stop_source, p.trailing_ma_period, p.trailing_ma_filter,
                        p.trailing_ma_confirm_days, p.trailing_ma_break_percent, p.atr_period, p.swing_low_lookback)
        trailing_series = _cached(cache, trailing_key, lambda: _trailing_stop_series(frame, p))
    signals, trailing_hit = _simulate_signals(
        buy_signal.to_numpy(), (sell_take_profit | sell_trend_break).to_numpy(), trailing_series, p)
    return signals, {
        "takeProfit": sell_take_profit.to_numpy(),
        "trendBreak": sell_trend_break.to_numpy(),
        "trailingStop": trailing_hit,
    }


def _equity_curve(frame: pd.DataFrame, signals: np.ndarray, p: KdBandParameters):
    """依訊號計算每日部位與累積報酬（淨值，起始為 1）。

    T 日收盤出現訊號，T+1 日以收盤價成交（與每輪交易表相同），避免用訊號當天的收盤價成交；
    成交日扣交易成本：買進扣手續費，賣出扣手續費 + 證交稅。
    """
    signal_state = pd.Series(signals, dtype=float).replace(0, np.nan).ffill().replace(-1, 0).fillna(0).to_numpy()
    held_after_close = np.concatenate(([0.0], signal_state[:-1]))          # 第 d 日收盤成交後是否持有
    position_change = np.diff(held_after_close, prepend=0.0)               # +1 買進成交日、-1 賣出成交日
    position = np.concatenate(([0.0], held_after_close[:-1]))              # 第 d 日的漲跌是否計入（前一日收盤已持有）
    trade_cost = np.where(position_change > 0, p.buy_cost_rate, np.where(position_change < 0, p.sell_cost_rate, 0.0))
    daily_return = frame["ETF_0050"].pct_change().fillna(0).to_numpy()
    cumulative = np.cumprod((1 + daily_return * position) * (1 - trade_cost))
    return position, position_change, cumulative


def optimize_kd_band(
    twii: pd.DataFrame,
    etf: pd.DataFrame,
    base: Optional[KdBandParameters] = None,
    trials: int = OPTIMIZE_TRIALS,
    train_ratio: float = OPTIMIZE_TRAIN_RATIO,
    min_trades: int = OPTIMIZE_MIN_TRADES,
    top_n: int = OPTIMIZE_TOP_N,
    seed: int = OPTIMIZE_RANDOM_SEED,
) -> Dict:
    """隨機搜尋參數組合，依「訓練期」累積報酬排序，並列出同一組參數在「驗證期」與全期間的報酬。

    訓練期 = 回測期間前 train_ratio 的交易日，驗證期 = 其餘交易日；排序只看訓練期，驗證期的資料不參與挑選。
    每組參數都在完整資料上回測（指標暖身與持倉可跨越分界日），再切出兩段的淨值變化。
    """
    base = base or KdBandParameters()
    frame = _prepare_frame(twii, etf)
    split = int(len(frame) * train_ratio)
    if split < 60 or len(frame) - split < 20:
        raise ValueError("Backtest period is too short to split into training and validation periods")

    closes = frame["ETF_0050"].to_numpy()
    rng = np.random.default_rng(seed)
    keys = list(OPTIMIZE_SPACE)
    # 第一組固定為目前頁面上的設定，方便比較
    candidates = [base]
    seen = {tuple(getattr(base, key) for key in keys)}
    attempts = 0
    while len(candidates) < trials and attempts < trials * 20:
        attempts += 1
        values = {key: OPTIMIZE_SPACE[key][rng.integers(len(OPTIMIZE_SPACE[key]))] for key in keys}
        values = {key: (value.item() if hasattr(value, "item") else value) for key, value in values.items()}
        # 沒用到的停利參數統一為目前設定，避免實質相同的組合重複計算
        used = TRAILING_METHOD_KEYS[values["trailing_stop_method"]]
        values.update({key: getattr(base, key) for key in ALL_TRAILING_KEYS if key not in used})
        identity = tuple(values[key] for key in keys)
        if identity in seen:
            continue
        seen.add(identity)
        candidates.append(replace(base, **values))

    cache: Dict = {}
    evaluated = []
    for index, parameters in enumerate(candidates):
        signals, _ = _generate_signals(frame, parameters, cache)
        _, position_change, curve = _equity_curve(frame, signals, parameters)
        entries = np.flatnonzero(position_change > 0)
        exits = np.flatnonzero(position_change < 0)
        train_trades = int(np.sum(exits <= split))
        evaluated.append({
            "isCurrent": index == 0,
            "parameters": parameters.to_camel_dict(),
            "trainReturnPercent": to_float((curve[split] / curve[0] - 1) * 100, 2),
            "validationReturnPercent": to_float((curve[-1] / curve[split] - 1) * 100, 2),
            "fullReturnPercent": to_float((curve[-1] - 1) * 100, 2),
            "trainTrades": train_trades,
            "totalTrades": int(len(exits)),
            "openPosition": bool(len(entries) > len(exits)),
            "exposurePercent": to_float(np.mean(np.concatenate(([0.0], (np.cumsum(position_change) > 0)[:-1]))) * 100, 1),
        })

    qualified = [row for row in evaluated if row["trainTrades"] >= min_trades]
    ranked = sorted(qualified, key=lambda row: row["trainReturnPercent"], reverse=True)
    buy_hold = lambda start, end: to_float((closes[end] / closes[start] - 1) * 100, 2)
    return {
        "startDate": frame.index[0].strftime("%Y-%m-%d"),
        "splitDate": frame.index[split].strftime("%Y-%m-%d"),
        "endDate": frame.index[-1].strftime("%Y-%m-%d"),
        "trainDays": split,
        "validationDays": len(frame) - 1 - split,
        "trials": len(candidates),
        "qualifiedTrials": len(qualified),
        "minTrades": min_trades,
        "buyAndHold": {
            "trainReturnPercent": buy_hold(0, split),
            "validationReturnPercent": buy_hold(split, len(frame) - 1),
            "fullReturnPercent": buy_hold(0, len(frame) - 1),
        },
        "current": evaluated[0],
        "top": ranked[:top_n],
    }


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
    closes, atr, prior_low, ma_trigger = series["close"], series["atr"], series["priorLow"], series["maTrigger"]
    signals = np.zeros(len(buy), dtype=int)
    holding, entry_signal_index = False, -1
    highest, stop_level = np.nan, np.nan

    for i in range(len(buy)):
        hit = False
        if holding and method != TRAILING_STOP_NONE and i > entry_signal_index:
            close = closes[i]
            is_new_high = np.isnan(highest) or close > highest
            if is_new_high:
                highest = close
            if method == TRAILING_STOP_MOVING_AVERAGE:
                hit = bool(ma_trigger[i])
            elif method == TRAILING_STOP_PERCENT:
                # 收盤自進場後最高收盤回落 X% 以上
                hit = close <= highest * (1 - p.trailing_drawdown_percent / 100)
            elif method == TRAILING_STOP_ATR:
                # 收盤跌破「進場後最高收盤 − N × ATR」
                hit = not np.isnan(atr[i]) and close < highest - p.atr_multiplier * atr[i]
            elif method == TRAILING_STOP_SWING_LOW:
                # 停利點 = 前波低點；每次創新高時上移到最新的前波低點（只上移不下移）
                if is_new_high and not np.isnan(prior_low[i]):
                    stop_level = prior_low[i] if np.isnan(stop_level) else max(stop_level, prior_low[i])
                hit = not np.isnan(stop_level) and close < stop_level

        if base_sell[i] or hit:
            signals[i] = -1
            trailing_hit[i] = hit
            holding = False
        elif buy[i]:
            signals[i] = 1
            if not holding:
                holding, entry_signal_index = True, i
                highest, stop_level = np.nan, np.nan
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
