import json
import re
from datetime import datetime

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

    def test_bollinger_bands_match_manual_calculation(self):
        prices = make_wave_prices()
        result = signal_analysis.analyze_price_history("2330.TW", prices)

        last_20 = prices["Close"].iloc[-20:]
        middle, std = last_20.mean(), last_20.std(ddof=0)
        upper, lower = middle + 2 * std, middle - 2 * std
        latest = result["latest"]
        assert latest["bollingerMiddle"] == pytest.approx(middle, abs=0.01)
        assert latest["bollingerUpper"] == pytest.approx(upper, abs=0.01)
        assert latest["bollingerLower"] == pytest.approx(lower, abs=0.01)
        assert latest["bollingerWidth"] == pytest.approx((upper - lower) / middle * 100, abs=0.01)
        assert latest["bollingerPercentB"] == pytest.approx((prices["Close"].iloc[-1] - lower) / (upper - lower) * 100, abs=0.1)
        assert result["series"]["bollingerUpper"][-1] == pytest.approx(upper, abs=0.0001)

    def test_bollinger_does_not_drop_trading_days(self):
        # 布林通道只需 20 日資料，暖機期由 MA60 決定，加入後交易日數不變
        result = signal_analysis.analyze_price_history("2330.TW", make_wave_prices())

        assert result["tradingDays"] == 260 - 59

    @pytest.mark.parametrize("calm_tail, expected", [(True, True), (False, False)])
    def test_bollinger_squeeze_when_recent_band_is_narrowest(self, calm_tail, expected):
        prices = make_wave_prices(days=200)
        if calm_tail:
            # 最後 30 天幾乎不動，帶寬收斂到近 60 日最低
            calm = prices["Close"].iloc[-31] + np.tile([0.0, 0.05], 15)
            for column, offset in (("Open", -0.02), ("High", 0.1), ("Low", -0.1), ("Close", 0.0)):
                prices.iloc[-30:, prices.columns.get_loc(column)] = calm + offset

        result = signal_analysis.analyze_price_history("2330.TW", prices)

        assert result["latest"]["bollingerSqueeze"] is expected


class TestKdBandBacktest:
    def test_positions_are_delayed_one_day_and_output_is_serializable(self):
        prices = make_wave_prices(days=400)

        result = market_analysis.run_kd_band_backtest(prices, prices)

        json.dumps(result)
        assert len(result["candles"]["dates"]) == len(result["equityCurve"])
        for signal in result["signals"]:
            if signal["executionDate"]:
                assert signal["executionDate"] > signal["signalDate"]
        # 每一輪交易都是先進場後出場（尚未出場的一輪出場日為 None）
        for trip in result["roundTrips"]:
            assert trip["isOpen"] == (trip["exitDate"] is None)
            if trip["exitDate"]:
                assert trip["exitDate"] > trip["entryDate"]

    def test_default_parameters_match_traditional_kd(self):
        prices = make_wave_prices(days=400)

        default = market_analysis.run_kd_band_backtest(prices, prices)
        explicit = market_analysis.run_kd_band_backtest(prices, prices, market_analysis.KdBandParameters(
            kd_period=9, k_smoothing=3, d_smoothing=3, ma_period=20, buy_k_threshold=20,
            take_profit_k_threshold=80, take_profit_bias_percent=3.5, trend_break_k_threshold=50))

        assert default["signals"] == explicit["signals"]
        assert default["parameters"] == {
            "kdPeriod": 9, "kSmoothing": 3, "dSmoothing": 3, "maPeriod": 20, "buyKThreshold": 20.0,
            "takeProfitKThreshold": 80.0, "takeProfitBiasPercent": 3.5, "trendBreakKThreshold": 50.0,
            "trailingStopMethod": "none", "trailingStopSource": "etf", "trailingMaPeriod": 60,
            "trailingMaFilter": "consecutive", "trailingMaConfirmDays": 3, "trailingMaBreakPercent": 3.0,
            "trailingDrawdownPercent": 10.0, "atrPeriod": 14, "atrMultiplier": 3.0, "swingLowLookback": 20,
            "transactionCostsEnabled": True, "feeRatePercent": 0.1425, "feeDiscount": 1.0, "sellTaxPercent": 0.1,
        }
        assert default["signalCount"] > 0
        # 不使用移動停利時，賣出原因只會是 A / B
        assert all("trailingStop" not in s["reasons"] for s in default["signals"])

    def test_thresholds_change_buy_and_sell_conditions(self):
        prices = make_wave_prices(days=400)
        run = lambda **kwargs: market_analysis.run_kd_band_backtest(prices, prices, market_analysis.KdBandParameters(**kwargs))

        # 昨日 K 不可能 < 0 → 不會有買進訊號
        assert all(s["direction"] == "sell" for s in run(buy_k_threshold=0)["signals"])
        # 放寬低檔門檻 → 買進訊號只會變多
        count_buys = lambda result: sum(s["direction"] == "buy" for s in result["signals"])
        assert count_buys(run(buy_k_threshold=50)) >= count_buys(run())
        # 昨日 K 不可能 > 100、今日 K 不可能 < 0 → 兩種賣出情境都不成立
        no_sell = run(take_profit_k_threshold=100, trend_break_k_threshold=0)
        assert all(s["direction"] == "buy" for s in no_sell["signals"])
        # 改變 KD 天數與平滑會得到不同的訊號
        assert run(kd_period=14, k_smoothing=5, d_smoothing=5)["signals"] != run()["signals"]


class TestKdBandReturns:
    """累積報酬以「訊號隔日收盤成交」計算，與每輪交易表一致；可扣交易成本；附 0050 買進持有對照。"""

    @pytest.mark.parametrize("costs_enabled", [True, False])
    @pytest.mark.parametrize("method", ["none", "swing_low"])
    def test_cumulative_return_equals_compounded_round_trips(self, costs_enabled, method):
        prices = make_wave_prices(days=400)
        parameters = market_analysis.KdBandParameters(trailing_stop_method=method, transaction_costs_enabled=costs_enabled)

        result = market_analysis.run_kd_band_backtest(prices, prices, parameters)

        compounded = np.prod([1 + trip["netReturnPercent"] / 100 for trip in result["roundTrips"]])
        assert len(result["roundTrips"]) > 3
        assert 1 + result["cumulativeReturnPercent"] / 100 == pytest.approx(compounded, rel=2e-3)

    def test_position_starts_after_execution_day_close(self):
        prices = make_wave_prices(days=400)
        result = market_analysis.run_kd_band_backtest(prices, prices, market_analysis.KdBandParameters(transaction_costs_enabled=False))
        dates = result["candles"]["dates"]
        curve = result["equityCurve"]
        first_entry = dates.index(result["roundTrips"][0]["entryDate"])

        # 訊號日與成交日當天淨值都不變（成交日以收盤價買進），成交日隔天才開始反映 0050 漲跌
        assert curve[first_entry] == pytest.approx(curve[first_entry - 1], abs=1e-4)   # 輸出四捨五入到小數 4 位
        closes = prices["Close"].to_numpy()
        expected = curve[first_entry] * closes[first_entry + 1] / closes[first_entry]
        assert curve[first_entry + 1] == pytest.approx(expected, abs=1e-4)

    def test_costs_and_buy_and_hold_benchmark(self):
        prices = make_wave_prices(days=400)
        discounted = market_analysis.KdBandParameters(fee_discount=0.28)

        result = market_analysis.run_kd_band_backtest(prices, prices, discounted)
        free = market_analysis.run_kd_band_backtest(prices, prices, market_analysis.KdBandParameters(transaction_costs_enabled=False))

        # 2.8 折：買進 0.1425% × 0.28 = 0.0399%；賣出再加證交稅 0.1%
        assert result["buyCostPercent"] == pytest.approx(0.0399)
        assert result["sellCostPercent"] == pytest.approx(0.1399)
        assert all(trip["netReturnPercent"] < trip["returnPercent"] for trip in result["roundTrips"])
        assert all(trip["netReturnPercent"] == trip["returnPercent"] for trip in free["roundTrips"])
        assert result["cumulativeReturnPercent"] < free["cumulativeReturnPercent"]

        closes = prices["Close"]
        expected_hold = (closes.iloc[-1] / closes.iloc[0] * (1 - 0.000399) - 1) * 100
        assert result["buyAndHoldReturnPercent"] == pytest.approx(expected_hold, abs=0.01)
        assert len(result["buyAndHoldCurve"]) == len(result["equityCurve"])
        assert 0 < result["exposurePercent"] < 100


def parameters_from_camel(camel: dict) -> market_analysis.KdBandParameters:
    snake = {re.sub(r"([A-Z])", lambda m: "_" + m.group(1).lower(), key): value for key, value in camel.items()}
    return market_analysis.KdBandParameters(**snake)


class TestOptimize:
    """參數最佳化：訓練期挑參數、驗證期檢查。"""

    def test_ranks_by_training_return_and_reports_validation(self):
        prices = make_wave_prices(days=500)
        base = market_analysis.KdBandParameters(fee_discount=0.5)

        result = market_analysis.optimize_kd_band(prices, prices, base, trials=150, seed=7, train_ratio=0.7, min_trades=2)

        json.dumps(result)
        assert (result["mode"], result["trials"], result["seed"], result["evaluatedAll"]) == ("random", 150, 7, False)
        assert result["totalCombinations"] == 113_702_400   # 預設搜尋範圍
        assert result["splitDate"] == prices.index[int(500 * 0.7)].strftime("%Y-%m-%d")
        # 目前設定另外列出作為比較基準
        assert parameters_from_camel(result["current"]["parameters"]) == base
        assert result["current"]["isCurrent"] is True
        top = result["top"]
        assert 0 < len(top) <= market_analysis.OPTIMIZE_TOP_N
        assert [row["trainReturnPercent"] for row in top] == sorted((row["trainReturnPercent"] for row in top), reverse=True)
        assert all(row["trainTrades"] >= 2 for row in top)
        # 交易成本沿用目前設定、不參與搜尋
        assert all(row["parameters"]["feeDiscount"] == 0.5 for row in top)
        # 搜尋出的組合不重複
        identities = [json.dumps(row["parameters"], sort_keys=True) for row in top]
        assert len(identities) == len(set(identities))

    def test_segments_match_full_backtest(self):
        prices = make_wave_prices(days=500)
        result = market_analysis.optimize_kd_band(prices, prices, trials=60, min_trades=1)
        best = result["top"][0]

        full = market_analysis.run_kd_band_backtest(prices, prices, parameters_from_camel(best["parameters"]))

        assert best["fullReturnPercent"] == pytest.approx(full["cumulativeReturnPercent"], abs=0.01)
        # 訓練期與驗證期報酬連乘 = 全期間報酬
        compounded = (1 + best["trainReturnPercent"] / 100) * (1 + best["validationReturnPercent"] / 100)
        assert compounded == pytest.approx(1 + best["fullReturnPercent"] / 100, rel=1e-3)

    def test_same_seed_gives_same_result(self):
        prices = make_wave_prices(days=400)

        first = market_analysis.optimize_kd_band(prices, prices, trials=50, seed=123)
        second = market_analysis.optimize_kd_band(prices, prices, trials=50, seed=123)

        without_time = lambda result: {key: value for key, value in result.items() if key != "elapsedSeconds"}
        assert without_time(first) == without_time(second)

    def test_without_seed_draws_new_sample_and_reports_seed(self):
        prices = make_wave_prices(days=400)

        first = market_analysis.optimize_kd_band(prices, prices, trials=50, min_trades=0)
        second = market_analysis.optimize_kd_band(prices, prices, trials=50, min_trades=0)
        replay = market_analysis.optimize_kd_band(prices, prices, trials=50, min_trades=0, seed=first["seed"])

        assert 1 <= first["seed"] <= market_analysis.OPTIMIZE_MAX_SEED
        assert first["seed"] != second["seed"]
        # 填回回傳的種子即可重現同一份排名
        assert replay["top"] == first["top"]

    def test_exhaustive_evaluates_every_combination(self):
        prices = make_wave_prices(days=400)
        space = {"kd_period": [5, 9], "buy_k_threshold": [20, 30, 40], "trailing_stop_method": ["none", "percent"],
                 "trailing_drawdown_percent": [5, 10]}

        result = market_analysis.optimize_kd_band(prices, prices, space=space, mode="exhaustive", min_trades=0, top_n=100)

        # 2 × 3 ×（不使用 1 種 + 百分比回撤 2 種）= 18 組
        assert (result["totalCombinations"], result["trials"], result["evaluatedAll"], result["seed"]) == (18, 18, True, None)
        assert len(result["top"]) == 18
        identities = {json.dumps(row["parameters"], sort_keys=True) for row in result["top"]}
        assert len(identities) == 18
        assert result["space"]["kdPeriod"] == [5, 9] and result["space"]["trailingDrawdownPercent"] == [5.0, 10.0]
        # 每一組都與單獨回測的結果相同
        for row in result["top"][:3]:
            full = market_analysis.run_kd_band_backtest(prices, prices, parameters_from_camel(row["parameters"]))
            assert row["fullReturnPercent"] == pytest.approx(full["cumulativeReturnPercent"], abs=0.01)

    def test_random_trials_beyond_space_size_evaluate_everything(self):
        prices = make_wave_prices(days=400)

        result = market_analysis.optimize_kd_band(prices, prices, space={"kd_period": [5, 9, 14]}, trials=100, min_trades=0)

        assert (result["totalCombinations"], result["trials"], result["evaluatedAll"]) == (3, 3, True)

    def test_marks_current_settings_in_ranking(self):
        prices = make_wave_prices(days=400)
        base = market_analysis.KdBandParameters()

        result = market_analysis.optimize_kd_band(prices, prices, base, space={"kd_period": [9]}, mode="exhaustive", min_trades=0)

        assert [row["isCurrent"] for row in result["top"]] == [True]

    def test_reports_progress_and_stops_when_callback_raises(self):
        prices = make_wave_prices(days=400)
        reports = []

        market_analysis.optimize_kd_band(prices, prices, trials=450, seed=1, min_trades=0,
                                         progress=lambda done, total, best: reports.append((done, total, best)))

        assert [done for done, _, _ in reports] == [200, 400, 450]
        assert all(total == 450 for _, total, _ in reports)
        assert set(reports[-1][2]) == {"trainReturnPercent", "validationReturnPercent"}

        def stop(done, total, best):
            raise RuntimeError("stopped")

        with pytest.raises(RuntimeError, match="stopped"):
            market_analysis.optimize_kd_band(prices, prices, trials=450, seed=1, progress=stop)

    def test_min_trades_filters_lucky_combinations(self):
        prices = make_wave_prices(days=400)

        result = market_analysis.optimize_kd_band(prices, prices, trials=80, min_trades=1000)

        assert result["top"] == []
        assert result["qualifiedTrials"] == 0

    def test_rejects_too_short_period(self):
        prices = make_wave_prices(days=60)

        with pytest.raises(ValueError):
            market_analysis.optimize_kd_band(prices, prices, trials=10)

    def test_rejects_unknown_mode(self):
        prices = make_wave_prices(days=400)

        with pytest.raises(ValueError):
            market_analysis.optimize_kd_band(prices, prices, mode="genetic")


class TestSearchSpace:
    """搜尋範圍：組合編號、沒用到的停利參數固定、範圍驗證。"""

    def test_default_space_size(self):
        space = market_analysis.build_search_space(market_analysis.KdBandParameters())

        # 買賣條件 6×4×4×4×6×5×7×5 = 403,200；停利 不使用 1 + 均線 256 + 百分比 6 + ATR 15 + 前波低點 4 = 282
        assert space.total == 403_200 * 282

    def test_every_index_maps_to_a_distinct_combination(self):
        base = market_analysis.KdBandParameters(atr_period=14, fee_discount=0.28)
        space = market_analysis.build_search_space(base, {
            "kd_period": [5, 9], "ma_period": [10, 20, 60], "trailing_stop_method": ["none", "atr", "swing_low"],
            "atr_period": [10, 20], "atr_multiplier": [2, 3], "swing_low_lookback": [10, 20, 40], "trailing_drawdown_percent": [5],
        })

        combinations = [space.parameters_at(index) for index in range(space.total)]

        assert space.total == 2 * 3 * (1 + 4 + 3)
        assert len(set(combinations)) == space.total
        # 第一個參數（KD 天數）變化最慢，同一組指標天數的編號相鄰
        assert [p.kd_period for p in combinations] == [5] * 24 + [9] * 24
        # 沒用到的停利參數固定為目前設定；未搜尋的參數（含交易成本）沿用目前設定
        assert {p.atr_period for p in combinations if p.trailing_stop_method != "atr"} == {14}
        # 指定了候選值、但停利方式沒有選用的參數也固定為目前設定
        assert {p.trailing_drawdown_percent for p in combinations} == {10.0}
        assert {p.fee_discount for p in combinations} == {0.28}

    def test_unsearched_trailing_method_uses_current_setting(self):
        base = market_analysis.KdBandParameters(trailing_stop_method="percent")
        space = market_analysis.build_search_space(base, {"trailing_drawdown_percent": [8, 10, 12]})

        assert space.total == 3
        assert [p.trailing_drawdown_percent for p in map(space.parameters_at, range(3))] == [8.0, 10.0, 12.0]

    def test_duplicate_values_are_removed(self):
        space = market_analysis.build_search_space(market_analysis.KdBandParameters(), {"kd_period": [9, 9.0, 14]})

        assert space.searched_camel() == {"kdPeriod": [9, 14]}

    @pytest.mark.parametrize("space", [
        {"kd_period": [1]}, {"buy_k_threshold": [101]}, {"trailing_drawdown_percent": [0]}, {"kd_period": [9.5]},
        {"kd_period": ["9"]}, {"atr_multiplier": [float("nan")]}, {"trailing_stop_method": ["magic"]},
        {"fee_discount": [0.5]}, {"kd_period": list(range(2, 40))},
    ])
    def test_rejects_invalid_space(self, space):
        with pytest.raises(ValueError):
            market_analysis.build_search_space(market_analysis.KdBandParameters(), space)

    def test_parameters_validate_ranges(self):
        with pytest.raises(ValueError):
            market_analysis.KdBandParameters(kd_period=1)
        with pytest.raises(ValueError):
            market_analysis.KdBandParameters(atr_multiplier=0)

    def test_indicator_cache_keeps_most_recent_entries(self):
        cache = market_analysis.IndicatorCache(max_size=2)
        calls = []
        compute = lambda key: cache.get_or_compute(key, lambda: calls.append(key) or key)

        for key in ["a", "b", "a", "c", "b"]:
            compute(key)

        # a 最近用過所以保留；加入 c 時淘汰 b，之後再用 b 需要重新計算
        assert calls == ["a", "b", "c", "b"]


class TestTrailingStops:
    """賣出 C：移動停利（每次回測只選一種）。"""

    @staticmethod
    def simulate(closes, method, atr=None, prior_low=None, ma_trigger=None, **kwargs):
        days = len(closes)
        buy = np.zeros(days, dtype=bool)
        buy[0] = True                                   # 第 0 天買進訊號，第 1 天進場開始追蹤
        series = {
            "close": np.array(closes, dtype=float),
            "atr": np.array(atr if atr is not None else [np.nan] * days, dtype=float),
            "priorLow": np.array(prior_low if prior_low is not None else [np.nan] * days, dtype=float),
            "maTrigger": np.array(ma_trigger if ma_trigger is not None else [False] * days, dtype=bool),
        }
        parameters = market_analysis.KdBandParameters(trailing_stop_method=method, **kwargs)
        signals, hits = market_analysis._simulate_signals(buy, np.zeros(days, dtype=bool), series, parameters)
        return signals.tolist(), hits.tolist()

    def test_none_matches_sell_first_then_buy(self):
        rng = np.random.default_rng(3)
        buy, sell = rng.random(300) < 0.1, rng.random(300) < 0.1
        series = {"close": np.ones(300), "atr": np.ones(300), "priorLow": np.ones(300), "maTrigger": np.zeros(300, dtype=bool)}

        signals, hits = market_analysis._simulate_signals(buy, sell, series, market_analysis.KdBandParameters())

        assert signals.tolist() == np.where(sell, -1, np.where(buy, 1, 0)).tolist()
        assert not hits.any()

    def test_percent_drawdown_from_highest_close(self):
        # 進場後最高收盤 120，回落 10% → 108 出場
        signals, hits = self.simulate([100, 100, 110, 120, 108, 107], "percent", trailing_drawdown_percent=10)

        assert signals == [1, 0, 0, 0, -1, 0]
        assert hits[4]

    def test_atr_trailing_stop(self):
        # 最高收盤 120 − 3 × ATR(3) = 111，收盤 111 不出場、110.9 出場
        signals, _ = self.simulate([100, 100, 110, 120, 113, 111, 110.9], "atr", atr=[3.0] * 7, atr_multiplier=3)

        assert signals == [1, 0, 0, 0, 0, 0, -1]

    def test_swing_low_moves_up_only_on_new_highs(self):
        prior_low = [np.nan, 95, 96, 99, 100, 101]
        # 新高日停利點依序上移到 95 → 96 → 99；第 4 天沒有新高，停利點維持 99
        stays, _ = self.simulate([100, 101, 103, 105, 104, 100], "swing_low", prior_low=prior_low)
        breaks, _ = self.simulate([100, 101, 103, 105, 104, 98], "swing_low", prior_low=prior_low)

        assert stays == [1, 0, 0, 0, 0, 0]
        assert breaks == [1, 0, 0, 0, 0, -1]

    def test_moving_average_trigger_only_while_holding(self):
        # 進場前的跌破不算；持有中的跌破觸發才出場
        signals, _ = self.simulate([100] * 5, "ma", ma_trigger=[False, False, False, True, False])
        before_entry, _ = self.simulate([100] * 3, "ma", ma_trigger=[True, False, False])

        assert signals == [-1 if i == 3 else (1 if i == 0 else 0) for i in range(5)]
        assert before_entry == [1, 0, 0]    # 尚未持有時的跌破不產生賣出，照常買進

    @pytest.mark.parametrize("ma_filter, expected_index", [("consecutive", 7), ("percent", 6), ("both", 7), ("either", 6)])
    def test_moving_average_filters_trigger_on_first_day(self, ma_filter, expected_index):
        closes = [10.0] * 6 + [8.0, 7.0, 6.0, 5.0]
        frame = pd.DataFrame({"TWII_C": closes, "TWII_H": closes, "TWII_L": closes, "ETF_0050": closes},
                             index=pd.bdate_range("2026-01-01", periods=len(closes)))
        parameters = market_analysis.KdBandParameters(
            trailing_stop_source="index", trailing_ma_period=3, trailing_ma_filter=ma_filter,
            trailing_ma_confirm_days=2, trailing_ma_break_percent=10)

        trigger = market_analysis._trailing_stop_series(frame, parameters)["maTrigger"]

        # 3 日均線：第 6 天收盤 8 < 均線 9.33（跌破 14%），第 7 天連續 2 日低於均線；之後不再重複觸發
        assert [i for i, value in enumerate(trigger) if value] == [expected_index]

    @pytest.mark.parametrize("method", ["ma", "percent", "atr", "swing_low"])
    def test_each_method_runs_end_to_end(self, method):
        prices = make_wave_prices(days=400)
        parameters = market_analysis.KdBandParameters(
            trailing_stop_method=method, trailing_ma_period=20, trailing_ma_confirm_days=1,
            trailing_drawdown_percent=2, atr_multiplier=1, swing_low_lookback=5)

        result = market_analysis.run_kd_band_backtest(prices, prices, parameters)

        json.dumps(result)
        assert result["parameters"]["trailingStopMethod"] == method
        assert any("trailingStop" in s["reasons"] for s in result["signals"])
        assert all(s["reasons"] == [] for s in result["signals"] if s["direction"] == "buy")

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError):
            market_analysis.KdBandParameters(trailing_stop_method="unknown")

    def test_open_position_is_listed_with_unrealized_return(self):
        prices = make_wave_prices(days=400)
        full = market_analysis.run_kd_band_backtest(prices, prices)
        dates = [d.strftime("%Y-%m-%d") for d in prices.index]
        # 找一次進場後至少 3 個交易日內沒有出場的進場日，把資料截到進場後第 2 個交易日
        entry_date = next(
            buy for buy in full["buyMarkers"]
            if all(not (buy < sell <= dates[dates.index(buy) + 3]) for sell in full["sellMarkers"])
        )
        last_index = dates.index(entry_date) + 2
        truncated = prices.iloc[: last_index + 1]

        result = market_analysis.run_kd_band_backtest(truncated, truncated)

        open_trip = result["roundTrips"][-1]
        entry_close = float(truncated.loc[entry_date, "Close"])
        last_close = float(truncated["Close"].iloc[-1])
        assert open_trip["isOpen"] is True
        assert open_trip["entryDate"] == entry_date
        assert open_trip["exitDate"] is None
        assert open_trip["holdingDays"] == (truncated.index[-1] - pd.Timestamp(entry_date)).days
        assert open_trip["returnPercent"] == pytest.approx(round((last_close / entry_close - 1) * 100, 2))
        assert result["openPositionSince"] == entry_date
        assert sum(not trip["isOpen"] for trip in result["roundTrips"]) == len(result["roundTrips"]) - 1

    def test_missing_etf_day_does_not_drop_index_day(self):
        prices = make_wave_prices(days=400)
        etf_with_gap = prices.drop(prices.index[200])

        full = market_analysis.run_kd_band_backtest(prices, prices)
        result = market_analysis.run_kd_band_backtest(prices, etf_with_gap)

        # KD 以大盤計算，0050 缺一天不影響訊號日與日期序列
        assert result["candles"]["dates"] == full["candles"]["dates"]
        assert [s["signalDate"] for s in result["signals"]] == [s["signalDate"] for s in full["signals"]]
        assert result["candles"]["close"][200] is None
        json.dumps(result)


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

    def test_detail_includes_next_month_date_and_close(self):
        closes = [100.0] * 30
        closes[5] = 95.0            # 單日 -5%
        for i in range(6, 30):
            closes[i] = 96.0 + i    # 第 26 個交易日（5 + 21）收盤 122
        index = pd.bdate_range("2026-01-01", periods=len(closes))
        frame = pd.DataFrame({"Close": closes}, index=index)

        detail = market_analysis.analyze_rebound(frame)["details"][0]

        # 隔月 = 往後第 21 個交易日；隔月漲跌 = 隔月收盤 / 當日收盤 - 1
        assert detail["nextMonthDate"] == index[5 + 21].strftime("%Y-%m-%d")
        assert detail["nextMonthClose"] == pytest.approx(122.0)
        assert detail["nextMonthReturn"] == pytest.approx(round((122.0 / 95.0 - 1) * 100, 2))

    def test_keeps_latest_close_and_marks_pending_next_month(self):
        closes = [100.0] * 40
        closes[5] = 95.0            # 單日 -5%，21 個交易日後已有結果
        for i in range(6, 40):
            closes[i] = 96.0
        closes[35] = 90.0           # 單日 -6.25%，距最新收盤不足 21 個交易日
        for i in range(36, 40):
            closes[i] = 91.0
        index = pd.bdate_range("2026-01-01", periods=len(closes))
        frame = pd.DataFrame({"Close": closes}, index=index)

        result = market_analysis.analyze_rebound(frame)

        # 結束日期為最新收盤日，不再少掉最近 21 個交易日
        assert result["endDate"] == index[-1].strftime("%Y-%m-%d")
        assert result["latestClose"] == pytest.approx(91.0)
        assert result["settledEndDate"] == index[-1 - 21].strftime("%Y-%m-%d")
        assert result["tradingDays"] == len(closes) - 1

        pending_bucket = next(b for b in result["buckets"] if b["range"] == "-7.0% ~ -6.0%")
        assert (pending_bucket["sampleCount"], pending_bucket["pendingCount"]) == (0, 1)
        assert pending_bucket["upRate"] is None
        assert pending_bucket["averageNextMonth"] is None
        assert pending_bucket["isStrong"] is False
        assert pending_bucket["averageDecline"] == pytest.approx(-6.25)

        settled_bucket = next(b for b in result["buckets"] if b["range"] == "-6.0% ~ -5.0%")
        assert (settled_bucket["sampleCount"], settled_bucket["pendingCount"], settled_bucket["upRate"]) == (1, 0, 100.0)

        assert [row["nextMonthReturn"] is None for row in result["details"]] == [False, True]
        assert (result["details"][1]["nextMonthDate"], result["details"][1]["nextMonthClose"]) == (None, None)
        assert (result["detailUpCount"], result["detailDownCount"], result["detailPendingCount"]) == (1, 0, 1)

    @pytest.mark.parametrize("now, expected_last", [
        (datetime(2026, 11, 5, 10, 0), "2026-11-04"),    # 盤中：當日即時價排除
        (datetime(2026, 11, 5, 13, 30), "2026-11-05"),   # 已收盤：保留當日
        (datetime(2026, 11, 6, 9, 30), "2026-11-05"),    # 隔日盤中、尚無當日資料：不影響前一日收盤
    ])
    def test_drop_unclosed_session(self, now, expected_last):
        index = pd.bdate_range("2026-11-02", "2026-11-05")
        frame = pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0]}, index=index)

        result = market_analysis.drop_unclosed_session(frame, now)

        assert result.index[-1].strftime("%Y-%m-%d") == expected_last


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
