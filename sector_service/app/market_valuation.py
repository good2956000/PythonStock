"""大盤席勒本益比（CAPE）計算（Domain 層，純函式，不依賴網路）。

證交所沒有提供大盤本益比的歷史資料，因此以個股資料自行彙整（估算值，非官方數字）：
1. 每月最後一個交易日：個股市值 = 收盤價 × 發行股數；個股盈餘 = 市值 ÷ 本益比（證交所本益比為近四季 EPS）
2. 大盤本益比 = 普通股總市值 ÷ 盈餘合計；虧損或未申報財報的公司本益比為「-」，盈餘以 0 計
3. 加權指數每點盈餘 = 加權指數 ÷ 大盤本益比
4. 席勒本益比 = 實質加權指數 ÷ 近 10 年（120 個月）實質每點盈餘平均；以消費者物價指數（CPI）平減
"""
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .serialization import dates_to_list, series_to_list, to_float

CAPE_WINDOW_MONTHS = 120
# 10 年內允許少數月份缺資料（例如證交所當月報表無法取得），不足此月數不計算
CAPE_MIN_MONTHS = 114
RECENT_MONTH_COUNT = 12


def is_common_stock(code: str) -> bool:
    """上市普通股代號為 4 碼數字且不以 0 開頭（排除 ETF、受益憑證等）。"""
    return len(code) == 4 and code.isdigit() and code[0] != "0"


def parse_number(text) -> Optional[float]:
    """證交所數字欄位：去除千分位；「-」「--」或空白等無法解析時回傳 None。"""
    try:
        value = float(str(text).replace(",", "").strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def aggregate_market_earnings(
    quotes: Mapping[str, Tuple[Optional[float], Optional[float]]], shares: Mapping[str, Optional[float]]
) -> Optional[Dict]:
    """彙整單一交易日的大盤市值與盈餘。

    quotes：代號 → (收盤價, 本益比)；shares：代號 → 發行股數。只計入有收盤價與發行股數的普通股。
    """
    market_cap = earnings = 0.0
    stock_count = profitable_count = 0
    for code, (close, pe) in quotes.items():
        share_count = shares.get(code)
        if not is_common_stock(code) or not close or not share_count:
            continue
        stock_market_cap = close * share_count
        market_cap += stock_market_cap
        stock_count += 1
        if pe and pe > 0:
            earnings += stock_market_cap / pe
            profitable_count += 1
    if stock_count == 0 or earnings <= 0:
        return None
    return {
        "marketCap": market_cap,
        "earnings": earnings,
        "stockCount": stock_count,
        "profitableCount": profitable_count,
    }


def parse_cpi_table(table: pd.DataFrame) -> pd.Series:
    """解析主計總處「消費者物價指數銜接表」（列：民國年；欄：1~12 月），回傳月 CPI（index 為月份 Period）。"""
    header_row = next(
        (i for i in range(len(table)) if str(table.iat[i, 0]).strip() == "民國年"), None
    )
    if header_row is None:
        raise ValueError("CPI 表格格式不符：找不到「民國年」標題列")
    month_columns = {}
    for column in range(1, table.shape[1]):
        label = str(table.iat[header_row, column]).strip()
        if label.endswith("月") and label[:-1].isdigit():
            month_columns[column] = int(label[:-1])
    if len(month_columns) != 12:
        raise ValueError("CPI 表格格式不符：月份欄位不是 12 個")

    values: Dict[pd.Period, float] = {}
    for row in range(header_row + 1, len(table)):
        year_text = str(table.iat[row, 0]).strip()
        if not year_text.isdigit():
            continue
        year = int(year_text) + 1911
        for column, month in month_columns.items():
            value = parse_number(table.iat[row, column])
            if value is not None and value > 0:
                values[pd.Period(year=year, month=month, freq="M")] = value
    if not values:
        raise ValueError("CPI 表格沒有資料")
    return pd.Series(values).sort_index()


def select_target_dates(trade_dates: Sequence[pd.Timestamp]) -> List[pd.Timestamp]:
    """每個月最後一個交易日（最新月份為目前最新交易日）。"""
    if not len(trade_dates):
        return []
    dates = pd.Series(pd.DatetimeIndex(trade_dates).sort_values())
    return list(dates.groupby(dates.dt.to_period("M")).max())


def calculate_cape(snapshots: pd.DataFrame, cpi: pd.Series) -> pd.DataFrame:
    """依每月快照計算本益比與席勒本益比。

    snapshots：index 為日期，欄位 indexClose、marketCap、earnings（每月一筆）。
    cpi：index 為月份 Period；最新幾個月 CPI 尚未公布時沿用最近一期。
    回傳欄位：indexClose、pe、eps（每點盈餘）、cpi、cape（不足 10 年為 NaN）。
    """
    frame = snapshots.sort_index().copy()
    frame["pe"] = frame["marketCap"] / frame["earnings"]
    frame["eps"] = frame["indexClose"] / frame["pe"]

    months = frame.index.to_period("M")
    full_cpi = cpi.sort_index().reindex(pd.period_range(min(cpi.index.min(), months.min()), months.max(), freq="M")).ffill()
    frame["cpi"] = full_cpi.reindex(months).to_numpy()

    # 實質每點盈餘：以 CPI 平減；滾動視窗以「月份」計算，缺漏月份不會讓視窗跨越超過 10 年
    real_eps = pd.Series((frame["eps"] / frame["cpi"]).to_numpy(), index=months)
    monthly = real_eps.reindex(pd.period_range(months.min(), months.max(), freq="M"))
    average_real_eps = monthly.rolling(CAPE_WINDOW_MONTHS, min_periods=CAPE_MIN_MONTHS).mean()
    frame["cape"] = (frame["indexClose"] / frame["cpi"]).to_numpy() / average_real_eps.reindex(months).to_numpy()
    return frame


def _percentile_rank(values: pd.Series, value: float) -> float:
    """value 在歷史資料中的百分位（≤ 該值的比例 × 100）。"""
    return float((values <= value).mean() * 100)


def analyze_cape(frame: pd.DataFrame) -> Dict:
    """整理成前端使用的 JSON：時間序列、最新值、歷史統計與近 12 個月明細。"""
    cape = frame["cape"].dropna()
    if cape.empty:
        raise ValueError("資料不足 10 年，無法計算席勒本益比")
    latest_date = frame.index[-1]
    latest = frame.iloc[-1]
    mean, std = float(cape.mean()), float(cape.std(ddof=0))

    recent = frame.iloc[-RECENT_MONTH_COUNT:].iloc[::-1]
    return {
        "dates": dates_to_list(frame.index),
        "indexClose": series_to_list(frame["indexClose"], 2),
        "pe": series_to_list(frame["pe"], 2),
        "cape": series_to_list(frame["cape"], 2),
        "dataStartDate": frame.index[0].strftime("%Y-%m-%d"),
        "capeStartDate": cape.index[0].strftime("%Y-%m-%d"),
        "latest": {
            "date": latest_date.strftime("%Y-%m-%d"),
            "indexClose": to_float(latest["indexClose"], 2),
            "pe": to_float(latest["pe"], 2),
            "cape": to_float(latest["cape"], 2),
            "capePercentile": to_float(_percentile_rank(cape, latest["cape"]), 1) if not np.isnan(latest["cape"]) else None,
            "stockCount": int(latest["stockCount"]),
            "profitableCount": int(latest["profitableCount"]),
        },
        "statistics": {
            "count": int(cape.size),
            "mean": to_float(mean, 2),
            "std": to_float(std, 2),
            "min": to_float(cape.min(), 2),
            "minDate": cape.idxmin().strftime("%Y-%m-%d"),
            "max": to_float(cape.max(), 2),
            "maxDate": cape.idxmax().strftime("%Y-%m-%d"),
        },
        "recent": [
            {
                "date": date.strftime("%Y-%m-%d"),
                "indexClose": to_float(row["indexClose"], 2),
                "pe": to_float(row["pe"], 2),
                "eps": to_float(row["eps"], 2),
                "cpi": to_float(row["cpi"], 2),
                "cape": to_float(row["cape"], 2),
            }
            for date, row in recent.iterrows()
        ],
    }
