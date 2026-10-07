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
        "Bias_20": 0.05, "BB_Upper": 99.0, "BB_Squeeze_Recent": True,
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


class TestBollingerAndBias:
    def test_bollinger_bands_match_manual_calculation(self, price_frame_factory):
        closes = [100.0 + (i % 7) * 1.5 for i in range(80)]
        frame = add_indicators(price_frame_factory(closes))

        window = pd.Series(closes[-20:])
        middle = window.mean()
        std = window.std(ddof=0)
        assert frame["BB_Upper"].iloc[-1] == pytest.approx(middle + 2 * std)
        assert frame["BB_Lower"].iloc[-1] == pytest.approx(middle - 2 * std)
        assert frame["BB_Width"].iloc[-1] == pytest.approx(4 * std / middle)

    def test_bias_20_is_distance_from_ma20(self, price_frame_factory):
        frame = add_indicators(price_frame_factory(range(1, 81)))

        ma20 = sum(range(61, 81)) / 20
        assert frame["Bias_20"].iloc[-1] == pytest.approx((80 - ma20) / ma20)

    def test_narrowing_band_after_volatile_period_is_squeeze(self, price_frame_factory):
        # 前段大幅震盪、最後 25 日幾乎不動 → 帶寬落在近 60 日最低區間
        closes = [100.0 + (8 if i % 2 else -8) for i in range(75)] + [100.0 + (i % 2) * 0.2 for i in range(25)]
        frame = add_indicators(price_frame_factory(closes))

        assert bool(frame["BB_Squeeze_Recent"].iloc[-1])

    def test_widening_band_is_not_squeeze(self, price_frame_factory):
        # 前段平穩、最後 25 日大幅震盪 → 帶寬為近 60 日最寬，不屬於壓縮
        closes = [100.0 + (i % 2) * 0.2 for i in range(75)] + [100.0 + (8 if i % 2 else -8) for i in range(25)]
        frame = add_indicators(price_frame_factory(closes))

        assert not bool(frame["BB_Squeeze_Recent"].iloc[-1])


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
    """stock_sectorX.py：7 個核心條件（均線糾結、突破、KD 低檔金叉、MACD 翻紅、流動性、量能、月線 > 季線）
    加上 3 個進階輔助條件（乖離率控管、布林通道壓縮、帶量突破布林上軌）。"""

    def test_all_ten_conditions_met_is_strong(self):
        frame = make_signal_frame(previous={"K": 20, "D": 25}, latest={"K": 32, "D": 28})

        result = evaluate_signals(frame, STRATEGY_BOTTOM_BREAKOUT)

        assert result.category == CATEGORY_STRONG
        assert len(result.conditions) == 10
        assert all(result.conditions.values())

    def test_ma20_below_ma60_is_not_strong(self):
        # 其餘 6 個條件都符合，只有月線低於季線
        frame = make_signal_frame(previous={"K": 20, "D": 25}, latest={"K": 32, "D": 28, "MA20": 94.0, "MA60": 95.0})

        result = evaluate_signals(frame, STRATEGY_BOTTOM_BREAKOUT)

        assert not result.conditions["ma20_above_ma60"]
        assert sum(result.conditions.values()) == 9
        assert result.category is None

    def test_golden_cross_above_30_is_not_low_golden_cross(self):
        frame = make_signal_frame(previous={"K": 40, "D": 45}, latest={"K": 55, "D": 50})

        result = evaluate_signals(frame, STRATEGY_BOTTOM_BREAKOUT)

        assert not result.conditions["kd_low_golden_cross"]
        assert result.category is None

    def test_partial_match_has_no_server_side_watch_category(self):
        # 原腳本的「均線糾結待突破」改由前端自選條件篩選，伺服器只回傳各條件結果
        frame = make_signal_frame(
            previous={"K": 20, "D": 25},
            latest={"K": 32, "D": 28, "Close": 95.5, "OSC": -0.1},  # 未突破 MA5、MACD 未翻紅
        )

        result = evaluate_signals(frame, STRATEGY_BOTTOM_BREAKOUT)

        assert result.category is None
        assert result.conditions["ma_tangled"] and result.conditions["kd_low_golden_cross"]
        assert not result.conditions["breaking_out"] and not result.conditions["macd_red"]

    def test_moving_averages_far_apart_is_not_tangled(self):
        frame = make_signal_frame(previous={"K": 20, "D": 25}, latest={"K": 32, "D": 28, "MA60": 80.0})

        result = evaluate_signals(frame, STRATEGY_BOTTOM_BREAKOUT)

        assert not result.conditions["ma_tangled"]
        assert result.category is None

    def test_auxiliary_conditions_do_not_affect_strong_category(self):
        # 進階輔助條件全部不符合，但 7 個核心條件都符合，仍列為底部剛突破
        frame = make_signal_frame(
            previous={"K": 20, "D": 25},
            latest={"K": 32, "D": 28, "Bias_20": 0.15, "BB_Upper": 105.0, "BB_Squeeze_Recent": False},
        )

        result = evaluate_signals(frame, STRATEGY_BOTTOM_BREAKOUT)

        assert result.category == CATEGORY_STRONG
        assert not result.conditions["low_bias_ma20"]
        assert not result.conditions["bollinger_squeeze"]
        assert not result.conditions["bollinger_volume_breakout"]

    @pytest.mark.parametrize("bias, expected", [(0.0599, True), (0.06, False), (0.10, False)])
    def test_low_bias_ma20_requires_bias_below_six_percent(self, bias, expected):
        frame = make_signal_frame(previous={"K": 20, "D": 25}, latest={"K": 32, "D": 28, "Bias_20": bias})

        result = evaluate_signals(frame, STRATEGY_BOTTOM_BREAKOUT)

        assert result.conditions["low_bias_ma20"] is expected

    def test_close_above_upper_band_without_volume_is_not_breakout(self):
        frame = make_signal_frame(
            previous={"K": 20, "D": 25},
            latest={"K": 32, "D": 28, "Volume": 1_100_000.0},  # 量 < 5 日均量 × 1.2
        )

        result = evaluate_signals(frame, STRATEGY_BOTTOM_BREAKOUT)

        assert not result.conditions["bollinger_volume_breakout"]

    def test_close_below_upper_band_is_not_breakout(self):
        frame = make_signal_frame(previous={"K": 20, "D": 25}, latest={"K": 32, "D": 28, "BB_Upper": 100.5})

        result = evaluate_signals(frame, STRATEGY_BOTTOM_BREAKOUT)

        assert not result.conditions["bollinger_volume_breakout"]

    def test_unknown_strategy_raises(self):
        frame = make_signal_frame(previous={}, latest={})

        with pytest.raises(ValueError):
            evaluate_signals(frame, "no_such_strategy")
