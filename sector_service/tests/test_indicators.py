import pandas as pd
import pytest

from app.indicators import (
    CATEGORY_STRONG,
    CATEGORY_WATCH,
    STRATEGY_BASIC,
    STRATEGY_BOTTOM_BREAKOUT,
    TREND_BULLISH,
    TREND_OVERHEATED,
    TREND_WEAK,
    add_indicators,
    evaluate_signals,
)


def make_signal_frame(previous, latest):
    """直接建構已含指標欄位的兩日資料，專注測試條件判斷。"""
    defaults = {
        "Close": 100.0, "MA5": 96.0, "MA20": 95.0, "MA60": 94.0, "OSC": 0.5,
        "K": 50.0, "D": 50.0, "Volume": 2_000_000.0, "Volume_MA5": 1_000_000.0,
    }
    rows = [{**defaults, **previous}, {**defaults, **latest}]
    return pd.DataFrame(rows, index=pd.to_datetime(["2026-09-28", "2026-09-29"]))


class TestAddIndicators:
    def test_drops_warm_up_rows_so_every_indicator_has_a_value(self, price_frame_factory):
        frame = add_indicators(price_frame_factory(range(100, 200)))

        # MA60 需要 60 筆資料才有值，前 59 筆會被移除
        assert len(frame) == 100 - 59
        assert not frame.isna().any().any()

    def test_moving_average_matches_manual_calculation(self, price_frame_factory):
        frame = add_indicators(price_frame_factory(range(1, 81)))

        # 最後一天收盤 80，前 20 天收盤為 61..80
        assert frame["MA20"].iloc[-1] == pytest.approx(sum(range(61, 81)) / 20)
        assert frame["MA5"].iloc[-1] == pytest.approx(sum(range(76, 81)) / 5)

    def test_flat_price_range_keeps_latest_day_with_neutral_rsv(self, price_frame_factory):
        # 最近 9 日價格完全不動（最高 = 最低）時，原腳本的 RSV 為 0/0，會把最新交易日刪掉
        prices = price_frame_factory([50.0 + (i % 3) for i in range(71)] + [50.0] * 9)
        prices.loc[prices.index[-9:], ["High", "Low"]] = 50.0

        frame = add_indicators(prices)

        assert frame.index[-1] == prices.index[-1]
        assert frame["RSV"].iloc[-1] == pytest.approx(50)


class TestEvaluateSignals:
    def test_returns_none_when_less_than_two_rows(self):
        frame = make_signal_frame({}, {}).iloc[-1:]

        assert evaluate_signals(frame) is None

    def test_all_four_conditions_met_is_strong(self):
        frame = make_signal_frame(previous={"K": 40, "D": 45}, latest={"K": 55, "D": 50})

        result = evaluate_signals(frame)

        assert result.category == CATEGORY_STRONG
        assert result.conditions["volume_surge"]
        assert result.trade_date == "2026-09-29"

    def test_without_volume_surge_is_watch(self):
        frame = make_signal_frame(
            previous={"K": 40, "D": 45},
            latest={"K": 55, "D": 50, "Volume": 1_100_000.0},  # 1.1 倍，未達 1.2 倍門檻
        )

        assert evaluate_signals(frame).category == CATEGORY_WATCH

    def test_kd_already_crossed_yesterday_is_not_golden_cross(self):
        frame = make_signal_frame(previous={"K": 50, "D": 45}, latest={"K": 55, "D": 50})

        result = evaluate_signals(frame)

        assert not result.conditions["kd_golden_cross"]
        assert result.category is None

    def test_below_ma20_is_excluded(self):
        frame = make_signal_frame(previous={"K": 40, "D": 45}, latest={"K": 55, "D": 50, "Close": 90.0})

        assert evaluate_signals(frame).category is None

    def test_illiquid_stock_is_excluded(self):
        frame = make_signal_frame(
            previous={"K": 40, "D": 45},
            latest={"K": 55, "D": 50, "Volume": 900_000.0, "Volume_MA5": 400_000.0},  # 均量 400 張
        )

        assert evaluate_signals(frame).category is None

    @pytest.mark.parametrize(
        "k, d, expected_trend",
        [(85, 70, TREND_OVERHEATED), (30, 40, TREND_WEAK), (60, 50, TREND_BULLISH)],
    )
    def test_trend_label(self, k, d, expected_trend):
        frame = make_signal_frame(previous={}, latest={"K": k, "D": d})

        assert evaluate_signals(frame).trend == expected_trend


class TestBasicStrategy:
    """stock.py：站上月線 + KD 金叉，不看量能。"""

    def test_illiquid_stock_still_strong_because_volume_is_ignored(self):
        frame = make_signal_frame(
            previous={"K": 40, "D": 45},
            latest={"K": 55, "D": 50, "Volume": 10_000.0, "Volume_MA5": 10_000.0},
        )

        assert evaluate_signals(frame, STRATEGY_BASIC).category == CATEGORY_STRONG

    def test_never_produces_watch_category(self):
        frame = make_signal_frame(previous={"K": 40, "D": 45}, latest={"K": 55, "D": 50, "Close": 90.0})

        assert evaluate_signals(frame, STRATEGY_BASIC).category is None


class TestBottomBreakoutStrategy:
    """stock_sectorX.py：均線糾結突破 + KD 低檔金叉 + MACD 翻紅 + 量能。"""

    def test_all_conditions_met_is_strong(self):
        frame = make_signal_frame(previous={"K": 20, "D": 25}, latest={"K": 32, "D": 28})

        result = evaluate_signals(frame, STRATEGY_BOTTOM_BREAKOUT)

        assert result.category == CATEGORY_STRONG
        assert all(result.conditions.values())

    def test_golden_cross_above_30_is_not_low_golden_cross(self):
        frame = make_signal_frame(previous={"K": 40, "D": 45}, latest={"K": 55, "D": 50})

        result = evaluate_signals(frame, STRATEGY_BOTTOM_BREAKOUT)

        assert not result.conditions["kd_low_golden_cross"]
        assert result.category is None

    def test_tangled_but_not_broken_out_is_watch(self):
        frame = make_signal_frame(
            previous={"K": 20, "D": 25},
            latest={"K": 32, "D": 28, "Close": 95.5, "OSC": -0.1},  # 未突破 MA5、MACD 未翻紅
        )

        assert evaluate_signals(frame, STRATEGY_BOTTOM_BREAKOUT).category == CATEGORY_WATCH

    def test_moving_averages_far_apart_is_not_tangled(self):
        frame = make_signal_frame(previous={"K": 20, "D": 25}, latest={"K": 32, "D": 28, "MA60": 80.0})

        result = evaluate_signals(frame, STRATEGY_BOTTOM_BREAKOUT)

        assert not result.conditions["ma_tangled"]
        assert result.category is None

    def test_unknown_strategy_raises(self):
        frame = make_signal_frame(previous={}, latest={})

        with pytest.raises(ValueError):
            evaluate_signals(frame, "no_such_strategy")
