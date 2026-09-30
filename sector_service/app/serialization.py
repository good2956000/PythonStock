"""把 pandas / numpy 結果轉成可 JSON 序列化的 Python 原生型別。"""
import math
from typing import Any, List, Optional

import numpy as np
import pandas as pd


def to_float(value: Any, digits: int = 4) -> Optional[float]:
    """NaN / inf 轉為 None，其餘四捨五入為 float。"""
    if value is None:
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, digits)


def series_to_list(series: pd.Series, digits: int = 4) -> List[Optional[float]]:
    return [to_float(value, digits) for value in series.tolist()]


def dates_to_list(index: pd.Index) -> List[str]:
    return [pd.Timestamp(value).strftime("%Y-%m-%d") for value in index]


def to_native(value: Any) -> Any:
    """遞迴把 numpy 純量、NaN 轉成 JSON 可接受的型別。"""
    if isinstance(value, dict):
        return {key: to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return to_float(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value
