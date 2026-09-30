import json

import numpy as np
import pandas as pd
import pytest

from app import market_analysis, signal_analysis


def make_wave_prices(days=260, seed=7):
    """產生有漲有跌的測試股價，確保各種交叉訊號都有機會出現。"""
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.normal(0, 1.5, days)) + 10 * np.sin(np.arange(days) / 12)
    index = pd.bdate_range("2025-01-01", periods=days)
    return pd.DataFrame(
        {
            "Open": closes - 0.5,
            "High": closes + 1.5,
            "Low": closes - 1.5,
            "Close": closes,
            "Volume": rng.integers(500_000, 3_000_000, days).astype(float),
        },
        index=index,
    )


class TestSignalAnalysis:
    def test_result_is_json_serializable_and_series_lengths_match(self):
        result = signal_analysis.analyze_price_history("2330.TW", make_wave_prices())

        json.dumps(result)  # NaN / numpy 型別都已轉換，不會丟出例外
        series = result["series"]
        assert len(series["dates"]) == result["tradingDays"]
        assert all(len(values) == len(series["dates"]) for values in series.values())

    def test_marker_dates_are_within_series(self):
        result = signal_analysis.analyze_price_history("2330.TW", make_wave_prices())

        all_dates = set(result["series"]["dates"])
        for dates in result["markers"].values():
            assert set(dates) <= all_dates

    def test_exit_rules_stop_loss_below_entry_low(self):
        frame = pd.DataFrame(
            {
                "Close": [100.0, 100.0, 100.0, 99.0],
                "Low": [99.0, 99.0, 99.5, 98.0],
                "MA10": [95.0] * 4,
                "ENTRY_SIGNAL": [False, False, True, False],
            },
            index=pd.bdate_range("2026-01-01", periods=4),
        )

        simulated, trades = signal_analysis.simulate_exits(frame)

        # 第 3 天收 100 進場（低點 99.5），第 4 天收 99 跌破進場 K 線低點 → 停損
        assert trades == [(100.0, 99.0)]
        assert simulated["EXIT_REASON"].iloc[-1] == "A 初始停損：跌破進場K線低點"

    def test_backtest_without_losses_has_no_reward_risk_ratio(self):
        stats = signal_analysis._backtest_statistics([(100.0, 110.0)])

        assert stats["winRate"] == 100.0
        assert stats["rewardRiskRatio"] is None

    def test_returns_none_when_not_enough_data(self):
        assert signal_analysis.analyze_price_history("2330.TW", make_wave_prices(days=40)) is None


class TestKdBandBacktest:
    def test_positions_are_delayed_one_day_and_output_is_serializable(self):
        prices = make_wave_prices(days=400)

        result = market_analysis.run_kd_band_backtest(prices, prices)

        json.dumps(result)
        assert len(result["candles"]["dates"]) == len(result["equityCurve"])
        for signal in result["signals"]:
            if signal["executionDate"]:
                assert signal["executionDate"] > signal["signalDate"]
        # 每一輪交易都是先進場後出場
        for trip in result["roundTrips"]:
            assert trip["exitDate"] > trip["entryDate"]


class TestRebound:
    def test_bins_and_next_month_return(self):
        closes = [100.0] * 30
        closes[5] = 95.0            # 單日 -5%（落在 -6% ~ -5% 區間）
        for i in range(6, 30):
            closes[i] = 96.0
        index = pd.bdate_range("2026-01-01", periods=len(closes))
        frame = pd.DataFrame({"Close": closes}, index=index)

        result = market_analysis.analyze_rebound(frame)

        assert len(result["buckets"]) == 1
        bucket = result["buckets"][0]
        assert bucket["range"] == "-6.0% ~ -5.0%"
        assert bucket["sampleCount"] == 1
        assert bucket["upCount"] == 1        # 95 → 21 個交易日後 96，上漲
        assert result["details"][0]["dailyReturn"] == pytest.approx(-5.0)


class TestMargin:
    def test_approximate_ratio_formula(self):
        index = pd.bdate_range("2026-01-01", periods=70)
        index_frame = pd.DataFrame({"Close": [20000.0] * 70}, index=index)
        records = [
            {"date": day, "margin_lots": 9_000_000, "short_lots": 200_000, "margin_amount_thousand": 300_000_000}
            for day in index[-5:]
        ]

        result = market_analysis.analyze_margin(index_frame, records)

        assert len(result["dates"]) == 5
        # 指數等於 60 日均時，維持率 = 1 / 0.6 × 100 ≈ 166.7%
        assert result["approximateRatio"][0] == pytest.approx(166.7)
        assert result["marginBalance100Million"][0] == pytest.approx(3000.0)
        assert result["shortBalance10kLots"][0] == pytest.approx(20.0)
