"""大盤層級的分析（Domain 層，輸入為已下載的 DataFrame，不直接連網）。

- run_kd_band_backtest : 移植自 PythonStock/stock_all.py      （大盤 KD 波段策略回測 0050）
- analyze_rebound      : 移植自 PythonStock/market_rebound.py （大盤回檔後隔月反彈統計）
- analyze_margin       : 移植自 PythonStock/market_margin.py  （融資餘額 + 近似融資維持率）
"""
from typing import Dict, List

import numpy as np
import pandas as pd

from .serialization import dates_to_list, series_to_list, to_float

# ---------------------------------------------------------------------------
# 大盤 KD 波段策略（stock_all.py）
# ---------------------------------------------------------------------------
TAKE_PROFIT_BIAS_20 = 3.5   # KD 高檔死叉且大盤正乖離 > 3.5% 時停利


def run_kd_band_backtest(twii: pd.DataFrame, etf: pd.DataFrame) -> Dict:
    """以大盤 KD 產生訊號、用 0050 計算報酬。twii 需 High/Low/Close，etf 需 Open/High/Low/Close。"""
    frame = pd.DataFrame({
        "TWII_H": twii["High"],
        "TWII_L": twii["Low"],
        "TWII_C": twii["Close"],
        "ETF_0050": etf["Close"],
    }).dropna()

    # 大盤 KD (9, 3, 3)；com=2 等同平滑係數 1/3
    low_min = frame["TWII_L"].rolling(window=9).min()
    high_max = frame["TWII_H"].rolling(window=9).max()
    frame["RSV"] = 100 * (frame["TWII_C"] - low_min) / (high_max - low_min)
    frame["K"] = frame["RSV"].ewm(com=2, adjust=False).mean()
    frame["D"] = frame["K"].ewm(com=2, adjust=False).mean()

    frame["SMA_20"] = frame["TWII_C"].rolling(window=20).mean()
    frame["Bias_20"] = (frame["TWII_C"] - frame["SMA_20"]) / frame["SMA_20"] * 100

    # 買進：KD 低檔（昨日 K < 20）黃金交叉
    previous_k, previous_d = frame["K"].shift(1), frame["D"].shift(1)
    frame["Buy_Signal"] = (previous_k < 20) & (previous_k <= previous_d) & (frame["K"] > frame["D"])

    # 賣出一：KD 高檔死叉且正乖離 > 3.5%（極度樂觀時停利）
    kd_death_cross = (previous_k > 80) & (previous_k >= previous_d) & (frame["K"] < frame["D"])
    sell_take_profit = kd_death_cross & (frame["Bias_20"] > TAKE_PROFIT_BIAS_20)
    # 賣出二：收盤剛跌破月線且 K < 50（趨勢轉弱時防守）
    sell_trend_break = (
        (frame["TWII_C"] < frame["SMA_20"])
        & (frame["TWII_C"].shift(1) >= frame["SMA_20"].shift(1))
        & (frame["K"] < 50)
    )
    frame["Sell_Signal"] = sell_take_profit | sell_trend_break

    # 買進 1、賣出 -1（同日同時成立時以賣出為準，與原腳本相同）
    frame["Signal"] = 0
    frame.loc[frame["Buy_Signal"], "Signal"] = 1
    frame.loc[frame["Sell_Signal"], "Signal"] = -1

    # 持倉狀態：遇到 1 持有、遇到 -1 空手；訊號隔天才反映在報酬上
    frame["Position"] = frame["Signal"].replace(0, np.nan).ffill().replace(-1, 0).fillna(0)
    frame["Position"] = frame["Position"].shift(1).fillna(0)

    frame["Daily_Return"] = frame["ETF_0050"].pct_change()
    frame["Strategy_Return"] = frame["Daily_Return"] * frame["Position"]
    frame["Cumulative_Return"] = (1 + frame["Strategy_Return"]).cumprod()

    signals: List[Dict] = []
    executions: List[Dict] = []
    for signal_date, row in frame[frame["Signal"] != 0].iterrows():
        location = frame.index.get_loc(signal_date)
        direction = "buy" if row["Signal"] == 1 else "sell"
        if location + 1 >= len(frame):
            signals.append({"direction": direction, "signalDate": signal_date.strftime("%Y-%m-%d"),
                            "executionDate": None, "price": None})
            continue
        execution_date = frame.index[location + 1]
        price = float(frame.loc[execution_date, "ETF_0050"])
        signals.append({"direction": direction, "signalDate": signal_date.strftime("%Y-%m-%d"),
                        "executionDate": execution_date.strftime("%Y-%m-%d"), "price": to_float(price, 2)})
        executions.append({"date": execution_date, "direction": direction, "price": price})

    # 配對進場 / 出場
    round_trips: List[Dict] = []
    entry = None
    for execution in executions:
        if execution["direction"] == "buy" and entry is None:
            entry = execution
        elif execution["direction"] == "sell" and entry is not None:
            round_trips.append({
                "entryDate": entry["date"].strftime("%Y-%m-%d"),
                "exitDate": execution["date"].strftime("%Y-%m-%d"),
                "holdingDays": (execution["date"] - entry["date"]).days,
                "returnPercent": to_float((execution["price"] / entry["price"] - 1) * 100, 2),
            })
            entry = None

    candles = etf.loc[frame.index]
    return {
        "startDate": frame.index[0].strftime("%Y-%m-%d"),
        "endDate": frame.index[-1].strftime("%Y-%m-%d"),
        "cumulativeReturnPercent": to_float((frame["Cumulative_Return"].iloc[-1] - 1) * 100, 2),
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
        "buyMarkers": [e["date"].strftime("%Y-%m-%d") for e in executions if e["direction"] == "buy"],
        "sellMarkers": [e["date"].strftime("%Y-%m-%d") for e in executions if e["direction"] == "sell"],
    }


# ---------------------------------------------------------------------------
# 大盤回檔後隔月反彈統計（market_rebound.py）
# ---------------------------------------------------------------------------
MONTH_TRADING_DAYS = 21                                    # 約一個月交易日
REBOUND_BINS = [-np.inf, -9, -8, -7, -6, -5, -4, -3]       # 只看單日跌 3% 以上
STRONG_UP_RATE = 55                                        # 隔月上漲率 > 55% 特別標示


def _bin_labels() -> List[str]:
    labels = []
    for low, high in zip(REBOUND_BINS[:-1], REBOUND_BINS[1:]):
        labels.append(f"< {high:.1f}%" if low == -np.inf else f"{low:.1f}% ~ {high:.1f}%")
    return labels


def analyze_rebound(index_frame: pd.DataFrame) -> Dict:
    """統計大盤單日下跌各區間之後，約一個月（21 個交易日）的表現。"""
    frame = index_frame[["Close"]].copy()
    frame["Return_T"] = frame["Close"].pct_change() * 100
    frame["Return_T1"] = frame["Close"].pct_change(periods=MONTH_TRADING_DAYS).shift(-MONTH_TRADING_DAYS) * 100
    frame = frame.dropna()

    labels = _bin_labels()
    declines = frame[frame["Return_T"] < 0].copy()
    declines["Bin"] = pd.cut(declines["Return_T"], bins=REBOUND_BINS, labels=labels)
    declines = declines.dropna(subset=["Bin"])

    buckets = []
    for label in labels:
        group = declines[declines["Bin"] == label]
        if group.empty:
            continue
        up_count = int((group["Return_T1"] > 0).sum())
        up_rate = up_count / len(group) * 100
        buckets.append({
            "range": label,
            "sampleCount": len(group),
            "upCount": up_count,
            "upRate": to_float(up_rate, 1),
            "averageNextMonth": to_float(group["Return_T1"].mean(), 2),
            "medianNextMonth": to_float(group["Return_T1"].median(), 2),
            "averageDecline": to_float(group["Return_T"].mean(), 2),
            "isStrong": bool(round(up_rate, 1) > STRONG_UP_RATE),
        })

    details = declines[declines["Return_T"] <= -3].sort_index()
    detail_rows = [{
        "date": date.strftime("%Y-%m-%d"),
        "close": to_float(row["Close"], 2),
        "dailyReturn": to_float(row["Return_T"], 2),
        "nextMonthReturn": to_float(row["Return_T1"], 2),
    } for date, row in details.iterrows()]
    detail_up_count = int((details["Return_T1"] > 0).sum())

    return {
        "startDate": frame.index[0].strftime("%Y-%m-%d"),
        "endDate": frame.index[-1].strftime("%Y-%m-%d"),
        "tradingDays": len(frame),
        "monthTradingDays": MONTH_TRADING_DAYS,
        "buckets": buckets,
        "details": detail_rows,
        "detailUpCount": detail_up_count,
        "detailDownCount": len(detail_rows) - detail_up_count,
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
