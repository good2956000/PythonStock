import numpy as np
import pandas as pd
import pytest


def make_price_frame(closes, volumes=None, spread=1.0):
    """依收盤價序列產生測試用日 K（High/Low 為收盤價上下 spread）。"""
    closes = np.asarray(closes, dtype=float)
    if volumes is None:
        volumes = np.full(len(closes), 1_000_000.0)
    index = pd.bdate_range("2026-01-01", periods=len(closes))
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes + spread,
            "Low": closes - spread,
            "Close": closes,
            "Volume": np.asarray(volumes, dtype=float),
        },
        index=index,
    )


@pytest.fixture
def price_frame_factory():
    return make_price_frame
