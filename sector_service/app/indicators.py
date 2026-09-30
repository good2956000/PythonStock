"""技術指標計算與選股條件判斷（Domain 層，純函式，不依賴網路）。

邏輯移植自下列腳本，條件與參數維持原腳本的實際行為：
- standard        : PythonStock/stock_sector.py   （站上月線 + KD 金叉 + 流動性 + 爆量）
- basic           : PythonStock/stock.py          （站上月線 + KD 金叉）
- bottom_breakout : PythonStock/stock_sectorX.py  （均線糾結突破 + KD 低檔金叉 + MACD 翻紅 + 量能 + 月線大於季線）
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# 流動性門檻：5 日均量需大於 500 張（1 張 = 1000 股，yfinance 的 Volume 單位為股）
MIN_AVERAGE_VOLUME_SHARES = 500 * 1000

# 爆量門檻：今日量 > 5 日均量 * 此倍數
# 註：stock_sector.py 註解寫「* 2」、stock_sectorX.py 註解寫「1.5 倍」，
#     但兩支腳本的實際程式碼皆為 1.2，此處沿用程式碼的實際值
VOLUME_SURGE_RATIO = 1.2

# K 值高於此數視為短線過熱
OVERHEATED_K_THRESHOLD = 80

# bottom_breakout：MA5 / MA20 / MA60 最大與最小值差距 < 5% 視為均線糾結
MA_TANGLE_RATIO = 0.05
# bottom_breakout：昨日 K 值低於此數才算低檔金叉
LOW_GOLDEN_CROSS_K = 30

CATEGORY_STRONG = "strong"      # 強勢名單：策略的完整條件全部符合
CATEGORY_WATCH = "watch"        # 候選觀察：符合核心條件，尚未完全確認

TREND_OVERHEATED = "overheated"
TREND_WEAK = "weak"
TREND_BULLISH = "bullish"

STRATEGY_STANDARD = "standard"
STRATEGY_BASIC = "basic"
STRATEGY_BOTTOM_BREAKOUT = "bottom_breakout"


@dataclass(frozen=True)
class SignalResult:
    """單一股票最新交易日的指標與條件判斷結果。"""

    trade_date: str
    close: float
    ma20: float
    k: float
    d: float
    volume: float
    volume_ma5: float
    category: Optional[str]
    trend: str
    conditions: Dict[str, bool] = field(default_factory=dict)


def add_indicators(price_frame: pd.DataFrame) -> pd.DataFrame:
    """在含有 Open/High/Low/Close/Volume 欄位的日 K 資料上計算技術指標，回傳新的 DataFrame。"""
    frame = price_frame.copy()
    close = frame["Close"]

    frame["MA5"] = close.rolling(window=5).mean()
    frame["MA20"] = close.rolling(window=20).mean()
    frame["MA60"] = close.rolling(window=60).mean()

    # MACD (12, 26, 9)
    fast_ema = close.ewm(span=12, adjust=False).mean()
    slow_ema = close.ewm(span=26, adjust=False).mean()
    frame["MACD"] = fast_ema - slow_ema
    frame["Signal"] = frame["MACD"].ewm(span=9, adjust=False).mean()
    frame["OSC"] = frame["MACD"] - frame["Signal"]

    # RSI (14)，採 Wilder 平滑（com=13）
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    average_gain = gain.ewm(com=13, adjust=False).mean()
    average_loss = loss.ewm(com=13, adjust=False).mean()
    relative_strength = average_gain / average_loss
    frame["RSI14"] = 100 - (100 / (1 + relative_strength))

    # KD (9, 3, 3)
    lowest_low = frame["Low"].rolling(window=9, min_periods=1).min()
    highest_high = frame["High"].rolling(window=9, min_periods=1).max()
    price_range = (highest_high - lowest_low).replace(0, np.nan)
    # 9 日內最高價等於最低價（例如停牌、一字盤）時無法計算 RSV，以 50（中性）代替，
    # 避免原腳本中 0/0 產生 NaN 後被 dropna 刪掉最新交易日
    frame["RSV"] = (100 * (close - lowest_low) / price_range).fillna(50)
    frame["K"] = frame["RSV"].ewm(alpha=1 / 3, adjust=False).mean()
    frame["D"] = frame["K"].ewm(alpha=1 / 3, adjust=False).mean()

    frame["Volume_MA5"] = frame["Volume"].rolling(window=5).mean()

    return frame.dropna()


def _is_kd_golden_cross(latest: pd.Series, previous: pd.Series) -> bool:
    """KD 黃金交叉：今日 K > D 且昨日 K <= D。"""
    return bool(latest["K"] > latest["D"] and previous["K"] <= previous["D"])


def _classify_standard(latest: pd.Series, previous: pd.Series) -> Tuple[Optional[str], Dict[str, bool]]:
    """stock_sector.py：站上月線 + KD 金叉 + 流動性，再依是否爆量區分強勢 / 觀察。"""
    conditions = {
        "above_ma20": bool(latest["Close"] > latest["MA20"]),
        "kd_golden_cross": _is_kd_golden_cross(latest, previous),
        "liquid": bool(latest["Volume_MA5"] > MIN_AVERAGE_VOLUME_SHARES),
        "volume_surge": bool(latest["Volume"] > latest["Volume_MA5"] * VOLUME_SURGE_RATIO),
    }
    category = None
    if conditions["above_ma20"] and conditions["kd_golden_cross"] and conditions["liquid"]:
        category = CATEGORY_STRONG if conditions["volume_surge"] else CATEGORY_WATCH
    return category, conditions


def _classify_basic(latest: pd.Series, previous: pd.Series) -> Tuple[Optional[str], Dict[str, bool]]:
    """stock.py：站上月線 + KD 金叉即列為強勢，沒有觀察名單與量能條件。"""
    conditions = {
        "above_ma20": bool(latest["Close"] > latest["MA20"]),
        "kd_golden_cross": _is_kd_golden_cross(latest, previous),
    }
    category = CATEGORY_STRONG if conditions["above_ma20"] and conditions["kd_golden_cross"] else None
    return category, conditions


# bottom_breakout 的 7 個條件（鍵值與前端「自選條件」共用，順序即畫面顯示順序）
BOTTOM_BREAKOUT_CONDITIONS = [
    ("ma_tangled", "均線糾結（MA5/MA20/MA60 差距 < 5%）"),
    ("breaking_out", "收盤突破三條均線"),
    ("kd_low_golden_cross", "KD 低檔黃金交叉（昨日 K < 30）"),
    ("macd_red", "MACD 柱狀翻紅（OSC > 0）"),
    ("liquid", "5 日均量 > 500 張"),
    ("volume_surge", "今日量 > 5 日均量 × 1.2"),
    ("ma20_above_ma60", "月線大於季線（MA20 > MA60）"),
]


def _classify_bottom_breakout(latest: pd.Series, previous: pd.Series) -> Tuple[Optional[str], Dict[str, bool]]:
    """stock_sectorX.py：抓底部起漲點。

    7 個條件全部符合才列為「底部剛突破」（strong）。
    原腳本的「均線糾結待突破」觀察名單已改為前端自選條件，依回傳的 conditions 即時篩選，
    因此這裡不再產生 watch 類別。
    """
    ma_values = [latest["MA5"], latest["MA20"], latest["MA60"]]
    ma_max, ma_min = max(ma_values), min(ma_values)
    conditions = {
        "ma_tangled": bool((ma_max - ma_min) / ma_min < MA_TANGLE_RATIO),
        "breaking_out": bool(latest["Close"] > ma_max),
        "kd_low_golden_cross": _is_kd_golden_cross(latest, previous) and bool(previous["K"] < LOW_GOLDEN_CROSS_K),
        "macd_red": bool(latest["OSC"] > 0),
        "liquid": bool(latest["Volume_MA5"] > MIN_AVERAGE_VOLUME_SHARES),
        "volume_surge": bool(latest["Volume"] > latest["Volume_MA5"] * VOLUME_SURGE_RATIO),
        # 第 7 個條件：月線在季線之上，確認中期趨勢已轉多
        "ma20_above_ma60": bool(latest["MA20"] > latest["MA60"]),
    }
    category = CATEGORY_STRONG if all(conditions.values()) else None
    return category, conditions


STRATEGY_CLASSIFIERS: Dict[str, Callable[[pd.Series, pd.Series], Tuple[Optional[str], Dict[str, bool]]]] = {
    STRATEGY_STANDARD: _classify_standard,
    STRATEGY_BASIC: _classify_basic,
    STRATEGY_BOTTOM_BREAKOUT: _classify_bottom_breakout,
}


def evaluate_signals(indicator_frame: pd.DataFrame, strategy: str = STRATEGY_STANDARD) -> Optional[SignalResult]:
    """依最新兩個交易日判斷選股條件；資料不足兩筆時回傳 None。"""
    if len(indicator_frame) < 2:
        return None
    classifier = STRATEGY_CLASSIFIERS.get(strategy)
    if classifier is None:
        raise ValueError(f"Unknown strategy: {strategy}")

    latest = indicator_frame.iloc[-1]
    previous = indicator_frame.iloc[-2]
    category, conditions = classifier(latest, previous)

    if latest["K"] > OVERHEATED_K_THRESHOLD:
        trend = TREND_OVERHEATED
    elif latest["K"] < latest["D"]:
        trend = TREND_WEAK
    else:
        trend = TREND_BULLISH

    trade_date = indicator_frame.index[-1]
    return SignalResult(
        trade_date=trade_date.strftime("%Y-%m-%d") if hasattr(trade_date, "strftime") else str(trade_date),
        close=float(latest["Close"]),
        ma20=float(latest["MA20"]),
        k=float(latest["K"]),
        d=float(latest["D"]),
        volume=float(latest["Volume"]),
        volume_ma5=float(latest["Volume_MA5"]),
        category=category,
        trend=trend,
        conditions=conditions,
    )
