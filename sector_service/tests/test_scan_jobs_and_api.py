import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main
from app.market_data import Sector, build_stock_name_map, split_code_and_name, split_download_frame
from app.optimize_jobs import OptimizeJobManager
from app.scan_jobs import STATUS_COMPLETED, STATUS_FAILED, ScanJobManager

API_KEY = "test-key-for-unit-tests"


class TestScanJobManager:
    def test_scans_every_ticker_and_skips_missing_data(self, price_frame_factory):
        requested_batches = []

        def fake_downloader(tickers):
            requested_batches.append(list(tickers))
            return {t: price_frame_factory(range(100, 200)) for t in tickers if t != "9999.TW"}

        manager = ScanJobManager(price_downloader=fake_downloader, batch_size=2)

        job = manager.run_synchronously(["半導體業"], ["2330.TW", "2303.TW", "9999.TW", "2330.TW"])

        assert job.status == STATUS_COMPLETED
        assert requested_batches == [["2330.TW", "2303.TW"], ["9999.TW"]]  # 重複代號已去除
        assert job.processed_count == 3
        assert [r.ticker for r in job.results] == ["2330.TW", "2303.TW"]
        assert job.results[0].volume_ma5_lots == 1000

    def test_results_include_stock_name_when_known(self, price_frame_factory):
        manager = ScanJobManager(price_downloader=lambda t: {x: price_frame_factory(range(100, 200)) for x in t})

        job = manager.run_synchronously(["x"], ["2330.TW", "6666.TW"], stock_names={"2330.TW": "台積電"})

        assert [(r.ticker, r.name) for r in job.results] == [("2330.TW", "台積電"), ("6666.TW", None)]
        assert "stockNames" not in job.to_dict() and "stock_names" not in job.to_dict()

    def test_downloader_error_marks_job_failed(self):
        def broken_downloader(_):
            raise RuntimeError("Yahoo unavailable")

        job = ScanJobManager(price_downloader=broken_downloader).run_synchronously(["x"], ["2330.TW"])

        assert job.status == STATUS_FAILED
        assert job.error_message == "Yahoo unavailable"
        assert job.finished_at is not None


class TestStockNames:
    @pytest.mark.parametrize(
        "raw, expected",
        [("2330　台積電", ("2330", "台積電")), ("1240 茂生農經", ("1240", "茂生農經")), ("股票", ("股票", ""))],
    )
    def test_split_code_and_name(self, raw, expected):
        assert split_code_and_name(raw) == expected

    def test_build_stock_name_map_merges_sectors(self):
        sectors = [Sector("1", "a", ["2330.TW"], {"2330.TW": "台積電"}), Sector("2", "b", ["6488.TWO"], {"6488.TWO": "環球晶"})]

        assert build_stock_name_map(sectors) == {"2330.TW": "台積電", "6488.TWO": "環球晶"}


class TestSplitDownloadFrame:
    def test_splits_multi_index_columns_per_ticker(self, price_frame_factory):
        single = price_frame_factory([10, 11, 12])
        raw = pd.concat({"2330.TW": single, "2303.TW": single}, axis=1)

        result = split_download_frame(raw, ["2330.TW", "2303.TW", "9999.TW"])

        assert set(result) == {"2330.TW", "2303.TW"}
        assert list(result["2330.TW"].columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_empty_download_returns_empty_dict(self):
        assert split_download_frame(pd.DataFrame(), ["2330.TW"]) == {}


@pytest.fixture
def optimize_client(monkeypatch, tmp_path):
    """最佳化 API 測試用：結果存到暫存資料夾、工作同步執行、股價下載改為假資料。"""
    monkeypatch.setenv(main.API_KEY_ENV_NAME, API_KEY)
    manager = OptimizeJobManager(str(tmp_path / "kd_optimize"))
    synchronous_submit = manager.submit
    monkeypatch.setattr(manager, "submit", lambda *args, **kwargs: synchronous_submit(*args, background=False, **kwargs))
    monkeypatch.setattr(main, "optimize_job_manager", manager)
    monkeypatch.setattr(main.market_data, "download_single_history", lambda ticker, **kwargs: pd.DataFrame({"Close": [1.0]}))
    monkeypatch.setattr(main.market_analysis, "drop_unclosed_session", lambda frame, now: frame)
    return TestClient(main.app), manager


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv(main.API_KEY_ENV_NAME, API_KEY)
    sectors = [
        Sector("1", "半導體業", ["2330.TW", "2303.TW"], {"2330.TW": "台積電", "2303.TW": "聯電"}),
        Sector("2", "航運業", ["2603.TW"], {"2603.TW": "長榮"}),
    ]
    monkeypatch.setattr(main.sector_cache, "get", lambda: sectors)

    submitted = {}

    class FakeJob:
        job_id = "a" * 32
        tickers = []

    def fake_submit(sector_names, tickers, strategy="standard", stock_names=None):
        submitted["sector_names"] = sector_names
        submitted["tickers"] = tickers
        submitted["strategy"] = strategy
        submitted["stock_names"] = stock_names
        FakeJob.tickers = tickers
        return FakeJob

    monkeypatch.setattr(main.job_manager, "submit", fake_submit)
    client = TestClient(main.app)
    client.submitted = submitted
    return client


class TestApi:
    def test_rejects_request_without_api_key(self, api_client):
        assert api_client.get("/api/sectors").status_code == 401

    def test_rejects_when_service_key_not_configured(self, api_client, monkeypatch):
        monkeypatch.delenv(main.API_KEY_ENV_NAME)

        response = api_client.get("/api/sectors", headers={"X-Api-Key": API_KEY})

        assert response.status_code == 503

    def test_lists_sectors_in_camel_case(self, api_client):
        response = api_client.get("/api/sectors", headers={"X-Api-Key": API_KEY})

        assert response.status_code == 200
        body = response.json()
        # 固定清單排在最前面，接著是證交所產業族群
        assert body[0]["sectorId"] == "P1" and body[0]["isPreset"] is True
        assert body[1] == {"sectorId": "1", "name": "半導體業", "tickerCount": 2, "isPreset": False}

    def test_start_scan_with_preset_and_strategy(self, api_client):
        response = api_client.post(
            "/api/scans", json={"sectorIds": ["P1"], "strategy": "basic"}, headers={"X-Api-Key": API_KEY}
        )

        assert response.status_code == 202
        assert api_client.submitted["strategy"] == "basic"
        assert len(api_client.submitted["tickers"]) == 24

    def test_start_scan_with_unknown_strategy_returns_400(self, api_client):
        response = api_client.post(
            "/api/scans", json={"sectorIds": ["1"], "strategy": "magic"}, headers={"X-Api-Key": API_KEY}
        )

        assert response.status_code == 400

    def test_lists_three_strategies(self, api_client):
        response = api_client.get("/api/strategies", headers={"X-Api-Key": API_KEY})

        assert [s["strategyId"] for s in response.json()] == ["standard", "bottom_breakout", "basic"]

    def test_bottom_breakout_exposes_ten_custom_conditions(self, api_client):
        strategies = {s["strategyId"]: s for s in api_client.get("/api/strategies", headers={"X-Api-Key": API_KEY}).json()}
        bottom = strategies["bottom_breakout"]

        keys = [c["key"] for c in bottom["conditionDefinitions"]]
        assert len(keys) == 10 and "ma20AboveMa60" in keys
        # 進階輔助條件排在 7 個核心條件之後
        assert keys[7:] == ["lowBiasMa20", "bollingerSqueeze", "bollingerVolumeBreakout"]
        # 預設勾選 ⑤ 流動性 ⑦ 月線大於季線 ⑧ 月線乖離率 < 6%，且都必須是有效的條件鍵
        assert bottom["defaultCustomConditions"] == ["liquid", "ma20AboveMa60", "lowBiasMa20"]
        assert set(bottom["defaultCustomConditions"]) <= set(keys)
        assert strategies["standard"]["conditionDefinitions"] == []

    def test_start_scan_with_zero_selects_all_sectors(self, api_client):
        response = api_client.post("/api/scans", json={"sectorIds": ["0"]}, headers={"X-Api-Key": API_KEY})

        assert response.status_code == 202
        assert response.json() == {"jobId": "a" * 32, "totalCount": 3}
        assert api_client.submitted["sector_names"] == ["半導體業", "航運業"]
        assert api_client.submitted["strategy"] == "standard"
        assert api_client.submitted["stock_names"]["2330.TW"] == "台積電"

    def test_start_scan_with_unknown_sector_returns_400(self, api_client):
        response = api_client.post("/api/scans", json={"sectorIds": ["999"]}, headers={"X-Api-Key": API_KEY})

        assert response.status_code == 400

    def test_invalid_job_id_format_is_rejected(self, api_client):
        response = api_client.get("/api/scans/not-a-job-id", headers={"X-Api-Key": API_KEY})

        assert response.status_code == 422

    def test_invalid_ticker_is_rejected(self, api_client):
        response = api_client.get("/api/stocks/2330;DROP/indicators", headers={"X-Api-Key": API_KEY})

        assert response.status_code == 400


class TestAnalysisEndpoints:
    def test_signal_analysis_falls_back_to_otc_suffix(self, api_client, monkeypatch, price_frame_factory):
        requested = []

        def fake_download(ticker, period=None, start=None):
            requested.append(ticker)
            return price_frame_factory([100 + (i % 7) for i in range(200)]) if ticker.endswith(".TWO") else None

        monkeypatch.setattr(main.market_data, "download_single_history", fake_download)

        response = api_client.get("/api/stocks/6488/signal-analysis?period=1y", headers={"X-Api-Key": API_KEY})

        assert response.status_code == 200
        assert requested == ["6488.TW", "6488.TWO"]
        assert response.json()["ticker"] == "6488.TWO"

    def test_signal_analysis_treats_letter_suffix_code_as_taiwan_stock(self, api_client, monkeypatch, price_frame_factory):
        requested = []

        def fake_download(ticker, period=None, start=None):
            requested.append(ticker)
            return price_frame_factory([100 + (i % 7) for i in range(200)])

        monkeypatch.setattr(main.market_data, "download_single_history", fake_download)

        response = api_client.get("/api/stocks/00631L/signal-analysis?period=1y", headers={"X-Api-Key": API_KEY})

        assert response.status_code == 200
        assert requested == ["00631L.TW"]

    def test_signal_analysis_rejects_invalid_period(self, api_client):
        response = api_client.get("/api/stocks/2330/signal-analysis?period=99y", headers={"X-Api-Key": API_KEY})

        assert response.status_code == 400

    def test_kd_backtest_rejects_future_start_date(self, api_client):
        response = api_client.get("/api/market/kd-backtest?startDate=2999-01-01", headers={"X-Api-Key": API_KEY})

        assert response.status_code == 400

    def test_kd_backtest_passes_custom_parameters(self, api_client, monkeypatch):
        captured = {}
        monkeypatch.setattr(main.market_data, "download_single_history", lambda ticker, **kwargs: pd.DataFrame({"Close": [1.0]}))
        monkeypatch.setattr(main.market_analysis, "drop_unclosed_session", lambda frame, now: frame)

        def fake_backtest(twii, etf, parameters):
            captured["parameters"] = parameters
            return {"parameters": parameters.to_camel_dict()}

        monkeypatch.setattr(main.market_analysis, "run_kd_band_backtest", fake_backtest)
        query = ("startDate=2020-06-01&kdPeriod=14&kSmoothing=2&dSmoothing=4&maPeriod=60"
                 "&buyKThreshold=25&takeProfitKThreshold=85&takeProfitBiasPercent=5&trendBreakKThreshold=40")

        response = api_client.get(f"/api/market/kd-backtest?{query}", headers={"X-Api-Key": API_KEY})

        assert response.status_code == 200
        assert captured["parameters"] == main.market_analysis.KdBandParameters(
            kd_period=14, k_smoothing=2, d_smoothing=4, ma_period=60, buy_k_threshold=25,
            take_profit_k_threshold=85, take_profit_bias_percent=5, trend_break_k_threshold=40)
        assert response.json()["parameters"]["takeProfitBiasPercent"] == 5

    def test_kd_backtest_passes_trailing_stop_parameters(self, api_client, monkeypatch):
        captured = {}
        monkeypatch.setattr(main.market_data, "download_single_history", lambda ticker, **kwargs: pd.DataFrame({"Close": [1.0]}))
        monkeypatch.setattr(main.market_analysis, "drop_unclosed_session", lambda frame, now: frame)
        monkeypatch.setattr(main.market_analysis, "run_kd_band_backtest",
                            lambda twii, etf, parameters: captured.setdefault("parameters", parameters) and {})
        query = ("trailingStopMethod=swing_low&trailingStopSource=index&trailingMaPeriod=20&trailingMaFilter=either"
                 "&trailingMaConfirmDays=2&trailingMaBreakPercent=5&trailingDrawdownPercent=15&atrPeriod=10"
                 "&atrMultiplier=2.5&swingLowLookback=30&transactionCostsEnabled=false&feeRatePercent=0.1"
                 "&feeDiscount=0.28&sellTaxPercent=0.15")

        response = api_client.get(f"/api/market/kd-backtest?{query}", headers={"X-Api-Key": API_KEY})

        assert response.status_code == 200
        p = captured["parameters"]
        assert (p.trailing_stop_method, p.trailing_stop_source, p.trailing_ma_period, p.trailing_ma_filter) == ("swing_low", "index", 20, "either")
        assert (p.trailing_ma_confirm_days, p.trailing_ma_break_percent, p.trailing_drawdown_percent) == (2, 5, 15)
        assert (p.atr_period, p.atr_multiplier, p.swing_low_lookback) == (10, 2.5, 30)
        assert (p.transaction_costs_enabled, p.fee_rate_percent, p.fee_discount, p.sell_tax_percent) == (False, 0.1, 0.28, 0.15)

    def test_kd_optimize_options_return_default_space(self, api_client):
        response = api_client.get("/api/market/kd-backtest/optimize-options", headers={"X-Api-Key": API_KEY})

        assert response.status_code == 200
        options = response.json()
        assert options["space"]["kdPeriod"] == [5, 7, 9, 12, 14, 18]
        assert options["space"]["trailingStopMethod"] == ["none", "ma", "percent", "atr", "swing_low"]
        assert (options["maxTrials"], options["maxValuesPerParameter"]) == (500_000, 30)

    def test_kd_optimize_job_passes_settings_and_saves_result(self, optimize_client, monkeypatch):
        client, manager = optimize_client
        captured = {}

        def fake_optimize(twii, etf, parameters, **kwargs):
            captured.update(parameters=parameters, **kwargs)
            kwargs["progress"](6, 6, None)
            return {"mode": kwargs["mode"], "seed": kwargs["seed"], "top": []}

        monkeypatch.setattr(main.market_analysis, "optimize_kd_band", fake_optimize)
        body = {"mode": "random", "trials": 500, "seed": 48213, "trainRatio": 0.8, "minTrades": 3,
                "space": {"kdPeriod": [9, 14], "trailingStopMethod": ["none", "atr"], "atrMultiplier": [2, 3], "maPeriod": []}}

        response = client.post("/api/market/kd-backtest/optimize-jobs?startDate=2021-01-04&feeDiscount=0.28",
                               json=body, headers={"X-Api-Key": API_KEY})

        assert response.status_code == 202
        job = response.json()
        # KD 天數 2 種 ×（不使用 1 種 + ATR 倍數 2 種）= 6 組，抽 500 組即全部
        assert (job["status"], job["totalCount"], job["result"]["seed"]) == ("completed", 6, 48213)
        assert (captured["mode"], captured["trials"], captured["seed"], captured["train_ratio"], captured["min_trades"]) == (
            "random", 500, 48213, 0.8, 3)
        assert captured["parameters"].fee_discount == 0.28
        # 空清單不搜尋（月線固定為目前設定）
        assert captured["space"] == {"kd_period": [9, 14], "trailing_stop_method": ["none", "atr"], "atr_multiplier": [2.0, 3.0]}
        saved = client.get(f"/api/market/kd-backtest/optimize-results/{job['resultId']}", headers={"X-Api-Key": API_KEY}).json()
        assert saved["settings"]["startDate"] == "2021-01-04"
        assert saved["settings"]["space"] == {"kdPeriod": [9, 14], "trailingStopMethod": ["none", "atr"], "atrMultiplier": [2.0, 3.0]}
        listed = client.get("/api/market/kd-backtest/optimize-results", headers={"X-Api-Key": API_KEY}).json()
        assert [item["resultId"] for item in listed["results"]] == [job["resultId"]]
        assert client.get(f"/api/market/kd-backtest/optimize-jobs/{job['jobId']}", headers={"X-Api-Key": API_KEY}).json()["status"] == "completed"
        assert client.get("/api/market/kd-backtest/optimize-jobs/current", headers={"X-Api-Key": API_KEY}).json() == {"job": None}

    def test_kd_optimize_job_reports_download_failure(self, optimize_client, monkeypatch):
        client, _ = optimize_client
        monkeypatch.setattr(main.market_data, "download_single_history", lambda ticker, **kwargs: None)

        job = client.post("/api/market/kd-backtest/optimize-jobs", json={}, headers={"X-Api-Key": API_KEY}).json()

        assert job["status"] == "failed"
        assert "無法下載" in job["errorMessage"]

    def test_kd_optimize_rejects_exhaustive_search_over_limit(self, optimize_client, monkeypatch):
        client, _ = optimize_client
        monkeypatch.setattr(main.market_analysis, "OPTIMIZE_MAX_COMBINATIONS", 100)

        response = client.post("/api/market/kd-backtest/optimize-jobs", json={"mode": "exhaustive"}, headers={"X-Api-Key": API_KEY})

        assert response.status_code == 400
        assert "113,702,400" in response.json()["detail"]

    def test_kd_optimize_rejects_second_job_while_running(self, optimize_client, monkeypatch):
        client, manager = optimize_client

        def busy(*args, **kwargs):
            raise main.OptimizeJobConflictError()

        monkeypatch.setattr(manager, "submit", busy)

        response = client.post("/api/market/kd-backtest/optimize-jobs", json={}, headers={"X-Api-Key": API_KEY})

        assert response.status_code == 409

    @pytest.mark.parametrize("body", [
        {"trials": 5}, {"trials": 600000}, {"trainRatio": 0.3}, {"minTrades": -1}, {"mode": "genetic"}, {"seed": 0},
        {"space": {"kdPeriod": [1]}}, {"space": {"kdPeriod": [9.5]}}, {"space": {"trailingStopMethod": ["magic"]}},
        {"space": {"kdPeriod": list(range(2, 40))}}, {"space": {"atrMultiplier": [0]}},
    ])
    def test_kd_optimize_rejects_invalid_settings(self, optimize_client, body):
        client, _ = optimize_client

        response = client.post("/api/market/kd-backtest/optimize-jobs", json=body, headers={"X-Api-Key": API_KEY})

        assert response.status_code == 422

    def test_kd_optimize_rejects_invalid_parameters_and_dates(self, optimize_client):
        client, _ = optimize_client

        assert client.post("/api/market/kd-backtest/optimize-jobs?kdPeriod=1", json={}, headers={"X-Api-Key": API_KEY}).status_code == 422
        assert client.post("/api/market/kd-backtest/optimize-jobs?startDate=2000-01-01", json={},
                           headers={"X-Api-Key": API_KEY}).status_code == 400

    @pytest.mark.parametrize("path", ["optimize-jobs/not-a-job-id", "optimize-results/..%2Fsecrets", "optimize-results/20261109-genetic"])
    def test_kd_optimize_rejects_malformed_ids(self, optimize_client, path):
        client, _ = optimize_client

        response = client.get(f"/api/market/kd-backtest/{path}", headers={"X-Api-Key": API_KEY})

        assert response.status_code in (404, 422)

    def test_kd_optimize_unknown_job_and_result_return_404(self, optimize_client):
        client, _ = optimize_client

        assert client.get(f"/api/market/kd-backtest/optimize-jobs/{'0' * 32}", headers={"X-Api-Key": API_KEY}).status_code == 404
        assert client.post(f"/api/market/kd-backtest/optimize-jobs/{'0' * 32}/cancel", headers={"X-Api-Key": API_KEY}).status_code == 404
        assert client.get("/api/market/kd-backtest/optimize-results/20261109-140000-random",
                          headers={"X-Api-Key": API_KEY}).status_code == 404

    def test_kd_optimize_requires_api_key(self, optimize_client):
        client, _ = optimize_client

        assert client.post("/api/market/kd-backtest/optimize-jobs", json={}).status_code == 401
        assert client.get("/api/market/kd-backtest/optimize-results").status_code == 401

    @pytest.mark.parametrize("query", ["kdPeriod=1", "kSmoothing=0", "maPeriod=500", "buyKThreshold=101", "trendBreakKThreshold=-1",
                                       "trailingStopMethod=magic", "trailingStopSource=tsmc", "trailingMaFilter=x",
                                       "trailingDrawdownPercent=0", "atrMultiplier=11", "swingLowLookback=1",
                                       "feeDiscount=2", "sellTaxPercent=-0.1", "transactionCostsEnabled=maybe"])
    def test_kd_backtest_rejects_out_of_range_parameters(self, api_client, query):
        response = api_client.get(f"/api/market/kd-backtest?{query}", headers={"X-Api-Key": API_KEY})

        assert response.status_code == 422

    def test_margin_days_out_of_range_is_rejected(self, api_client):
        response = api_client.get("/api/market/margin?days=5000", headers={"X-Api-Key": API_KEY})

        assert response.status_code == 422
