"""個股 RSI / MACD / KD 買賣訊號分析與動態出場回測（Domain 層）。

移植自 PythonStock/stock_bias_bollinger.py，指標、訊號與出場規則維持原腳本行為。
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .serialization import dates_to_list, series_to_list, to_float

STOP_LOSS_PCT = 0.07          # A 初始停損：帳面虧損達 7%
PROFIT_PROTECT_PCT = 0.05     # 獲利超過 5% 後啟動停利保護
ENTRY_SCORE_THRESHOLD = 80    # 右側順勢計分 >= 80 分才進場
PULLBACK_MAX_BIAS_60 = 0.05   # 回踩季線：距離季線不超過 5%
RECENT_SIGNAL_COUNT = 15


def add_signal_indicators(price_frame: pd.DataFrame) -> pd.DataFrame:
    """計算指標與各種買賣訊號欄位。"""
    frame = price_frame[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    close = frame["Close"]

    # RSI (14 日)
    delta = close.diff()
    average_gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    average_loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    frame["RSI"] = 100 - (100 / (1 + average_gain / average_loss))

    # MACD (12/26/9)
    frame["MACD"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    frame["SIGNAL"] = frame["MACD"].ewm(span=9, adjust=False).mean()
    frame["OSC"] = frame["MACD"] - frame["SIGNAL"]

    # KD (9 日)
    lowest_low = frame["Low"].rolling(window=9, min_periods=1).min()
    highest_high = frame["High"].rolling(window=9, min_periods=1).max()
    price_range = (highest_high - lowest_low).replace(0, np.nan)
    rsv = 100 * (close - lowest_low) / price_range
    frame["K"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    frame["D"] = frame["K"].ewm(alpha=1 / 3, adjust=False).mean()

    for window in (5, 10, 20, 60):
        frame[f"MA{window}"] = close.rolling(window=window).mean()
    frame["Volume_MA5"] = frame["Volume"].rolling(window=5).mean()

    frame = frame.dropna()

    # RSI 訊號：跌破 70（前一天 > 70）為賣、<= 30 為買
    frame["RSI_SELL"] = (frame["RSI"].shift(1) > 70) & (frame["RSI"] <= 70)
    frame["RSI_BUY"] = frame["RSI"] <= 30

    # MACD 柱狀由負轉正 / 由正轉負
    previous_osc = frame["OSC"].shift(1)
    frame["MACD_BUY"] = (frame["OSC"] > 0) & (previous_osc <= 0)
    frame["MACD_SELL"] = (frame["OSC"] < 0) & (previous_osc >= 0)
    # OSC 仍在水下但較昨日上升：空方動能收斂
    frame["OSC_SHRINK"] = (frame["OSC"] < 0) & (frame["OSC"] > previous_osc)

    # KD 金叉（K < 50）/ 死叉（K > 50）
    previous_k = frame["K"].shift(1)
    previous_d = frame["D"].shift(1)
    frame["KD_BUY"] = (frame["K"] > frame["D"]) & (previous_k <= previous_d) & (frame["K"] < 50)
    frame["KD_SELL"] = (frame["K"] < frame["D"]) & (previous_k >= previous_d) & (frame["K"] > 50)
    frame["RSI_KD_SELL"] = frame["RSI_SELL"] & frame["KD_SELL"]

    # OSC 動態 0 軸：近 60 日 OSC 波動幅度的 5% 內視為貼近 0 軸，且動能向上
    osc_range = frame["OSC"].rolling(60).max() - frame["OSC"].rolling(60).min()
    near_zero = frame["OSC"].abs() <= osc_range * 0.05
    frame["OSC_READY"] = near_zero & (frame["OSC"] > previous_osc)

    # 右側順勢計分：站上季線 40、剛突破月線 40、OSC 貼近 0 軸 10、量增 10
    long_trend_ok = frame["Close"] > frame["MA60"]
    cross_over_ma20 = (frame["Close"] > frame["MA20"]) & (frame["Close"].shift(1) <= frame["MA20"].shift(1))
    volume_ok = frame["Volume"] > frame["Volume_MA5"]
    frame["Bias_20"] = (frame["Close"] - frame["MA20"]) / frame["MA20"]
    frame["ENTRY_SCORE"] = (
        long_trend_ok.astype(int) * 40
        + cross_over_ma20.astype(int) * 40
        + frame["OSC_READY"].astype(int) * 10
        + volume_ok.astype(int) * 10
    )
    frame["COMBO_BUY"] = frame["ENTRY_SCORE"] >= ENTRY_SCORE_THRESHOLD

    # 回踩季線買進：站上季線、OSC 貼近 0 軸、季線正乖離 0~5%、量增
    frame["Bias_60"] = (frame["Close"] - frame["MA60"]) / frame["MA60"]
    frame["PULLBACK_BUY"] = (
        long_trend_ok
        & frame["OSC_READY"]
        & (frame["Bias_60"] > 0)
        & (frame["Bias_60"] <= PULLBACK_MAX_BIAS_60)
        & volume_ok
    )
    frame["ENTRY_SIGNAL"] = frame["COMBO_BUY"] | frame["PULLBACK_BUY"]
    return frame


def simulate_exits(frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[float, float]]]:
    """逐日模擬進場與動態出場，回傳（含 EXIT_SIGNAL / EXIT_REASON 的 DataFrame, [(進場價, 出場價)]）。"""
    frame = frame.copy()
    frame["EXIT_SIGNAL"] = False
    frame["EXIT_REASON"] = ""
    trades: List[Tuple[float, float]] = []

    entry_price: Optional[float] = None
    entry_low: Optional[float] = None

    for position in range(2, len(frame)):
        current = frame.iloc[position]
        previous = frame.iloc[position - 1]
        two_days_ago = frame.iloc[position - 2]
        close = float(current["Close"])

        if entry_price is None:
            if bool(current["ENTRY_SIGNAL"]):
                entry_price, entry_low = close, float(current["Low"])
            continue

        reason = ""
        change = (close - entry_price) / entry_price
        if close < entry_low:
            reason = "A 初始停損：跌破進場K線低點"
        elif change <= -STOP_LOSS_PCT:
            reason = f"A 初始停損：帳面虧損達 {STOP_LOSS_PCT * 100:.0f}%"
        elif change > PROFIT_PROTECT_PCT:
            three_day_low = close < min(float(previous["Close"]), float(two_days_ago["Close"]))
            if close < float(current["MA10"]) or three_day_low:
                reason = "停利保護：跌破MA10或創近3日新低"

        if reason:
            frame.iat[position, frame.columns.get_loc("EXIT_SIGNAL")] = True
            frame.iat[position, frame.columns.get_loc("EXIT_REASON")] = reason
            trades.append((entry_price, close))
            entry_price = entry_low = None

    return frame, trades


def _current_signals(frame: pd.DataFrame) -> List[Dict[str, str]]:
    """依最新兩日判斷目前的買 / 賣 / 觀察訊號。"""
    latest = frame.iloc[-1]
    rsi, previous_rsi = float(latest["RSI"]), float(frame["RSI"].iloc[-2])
    k = float(latest["K"])
    osc, previous_osc = float(latest["OSC"]), float(frame["OSC"].iloc[-2])

    signals = []
    if previous_rsi > 70 and rsi <= 70 and bool(frame["KD_SELL"].iloc[-1]):
        signals.append({"type": "sell", "message": f"RSI 跌破70（{previous_rsi:.1f}→{rsi:.1f}）且 KD 高檔死叉"})
    if rsi <= 30:
        signals.append({"type": "buy", "message": f"RSI {rsi:.1f} ≤ 30（超賣）"})
    if osc > 0 and previous_osc <= 0:
        signals.append({"type": "buy", "message": "MACD 柱狀由負轉正（黃金交叉）"})
    if osc < 0 and previous_osc >= 0:
        signals.append({"type": "sell", "message": "MACD 柱狀由正轉負（死亡交叉）"})
    if osc < 0 and osc > previous_osc:
        signals.append({"type": "watch", "message": f"MACD OSC 水下收斂 ({previous_osc:+.4f} → {osc:+.4f})，空方動能衰退"})
    if k <= 20:
        signals.append({"type": "buy", "message": f"KD K值 {k:.1f} 進入超賣區（≤20）"})
    return signals


def _recent_signal_rows(frame: pd.DataFrame) -> List[Dict]:
    mask = (
        frame["RSI_SELL"] | frame["RSI_BUY"] | frame["MACD_SELL"] | frame["MACD_BUY"]
        | frame["KD_SELL"] | frame["KD_BUY"] | frame["ENTRY_SIGNAL"] | frame["EXIT_SIGNAL"]
        | frame["OSC_SHRINK"] | frame["RSI_KD_SELL"]
    )
    rows = []
    for date, row in frame[mask].tail(RECENT_SIGNAL_COUNT).iterrows():
        tags = []
        if row["COMBO_BUY"]:
            tags.append("🔥強勢突破買進")
        elif row["PULLBACK_BUY"]:
            tags.append("📈回踩季線買進")
        elif row["EXIT_SIGNAL"]:
            tags.append(f"🛑出場({row['EXIT_REASON']})")
        else:
            if row["RSI_BUY"]:
                tags.append("RSI買")
            if row["MACD_BUY"]:
                tags.append("MACD金叉")
            if row["KD_BUY"]:
                tags.append("KD金叉")
            if row["MACD_SELL"]:
                tags.append("MACD死叉")
            if row["RSI_KD_SELL"]:
                tags.append("🚨RSI破70+KD死叉")
            if row["OSC_SHRINK"]:
                tags.append("🔍OSC水下收斂")
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "close": to_float(row["Close"], 2),
            "rsi": to_float(row["RSI"], 1),
            "k": to_float(row["K"], 1),
            "d": to_float(row["D"], 1),
            "tags": tags,
        })
    return rows


def _backtest_statistics(trades: List[Tuple[float, float]]) -> Dict:
    if not trades:
        return {"tradeCount": 0}
    returns = [(exit_price - entry_price) / entry_price * 100 for entry_price, exit_price in trades]
    wins = [value for value in returns if value > 0]
    losses = [value for value in returns if value <= 0]
    average_win = sum(wins) / len(wins) if wins else 0.0
    average_loss = sum(losses) / len(losses) if losses else 0.0
    return {
        "tradeCount": len(trades),
        "winCount": len(wins),
        "lossCount": len(losses),
        "winRate": to_float(len(wins) / len(trades) * 100, 1),
        "averageWin": to_float(average_win, 2),
        "averageLoss": to_float(average_loss, 2),
        # 沒有虧損交易時賺賠比為無限大，以 None 表示
        "rewardRiskRatio": to_float(abs(average_win / average_loss), 2) if average_loss != 0 else None,
    }


def _dates_where(frame: pd.DataFrame, mask: pd.Series) -> List[str]:
    return dates_to_list(frame.index[mask.to_numpy()])


def analyze_price_history(ticker: str, price_frame: pd.DataFrame) -> Optional[Dict]:
    """完整分析單一股票；資料不足以計算指標時回傳 None。"""
    frame = add_signal_indicators(price_frame)
    if len(frame) < 3:
        return None
    frame, trades = simulate_exits(frame)
    latest = frame.iloc[-1]

    return {
        "ticker": ticker,
        "startDate": frame.index[0].strftime("%Y-%m-%d"),
        "endDate": frame.index[-1].strftime("%Y-%m-%d"),
        "tradingDays": len(frame),
        "latest": {
            "close": to_float(latest["Close"], 2),
            "rsi": to_float(latest["RSI"], 1),
            "macd": to_float(latest["MACD"]),
            "signal": to_float(latest["SIGNAL"]),
            "osc": to_float(latest["OSC"]),
            "k": to_float(latest["K"], 1),
            "d": to_float(latest["D"], 1),
            "entryScore": int(latest["ENTRY_SCORE"]),
        },
        "currentSignals": _current_signals(frame),
        "recentSignals": _recent_signal_rows(frame),
        "backtest": _backtest_statistics(trades),
        "series": {
            "dates": dates_to_list(frame.index),
            "open": series_to_list(frame["Open"]),
            "close": series_to_list(frame["Close"]),
            "ma20": series_to_list(frame["MA20"]),
            "ma60": series_to_list(frame["MA60"]),
            "rsi": series_to_list(frame["RSI"], 2),
            "macd": series_to_list(frame["MACD"]),
            "signal": series_to_list(frame["SIGNAL"]),
            "osc": series_to_list(frame["OSC"]),
            "k": series_to_list(frame["K"], 2),
            "d": series_to_list(frame["D"], 2),
            "volume": series_to_list(frame["Volume"], 0),
        },
        "markers": {
            "comboBuy": _dates_where(frame, frame["COMBO_BUY"]),
            "pullbackBuy": _dates_where(frame, frame["PULLBACK_BUY"] & ~frame["COMBO_BUY"]),
            "exit": _dates_where(frame, frame["EXIT_SIGNAL"]),
            "rsiKdSell": _dates_where(frame, frame["RSI_KD_SELL"]),
            "macdBuy": _dates_where(frame, frame["MACD_BUY"]),
            "macdSell": _dates_where(frame, frame["MACD_SELL"]),
            "oscShrink": _dates_where(frame, frame["OSC_SHRINK"]),
            "kdBuy": _dates_where(frame, frame["KD_BUY"]),
            "kdSell": _dates_where(frame, frame["KD_SELL"]),
        },
    }
