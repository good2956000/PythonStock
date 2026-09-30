import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main
from app.market_data import Sector, split_download_frame
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

    def test_downloader_error_marks_job_failed(self):
        def broken_downloader(_):
            raise RuntimeError("Yahoo unavailable")

        job = ScanJobManager(price_downloader=broken_downloader).run_synchronously(["x"], ["2330.TW"])

        assert job.status == STATUS_FAILED
        assert job.error_message == "Yahoo unavailable"
        assert job.finished_at is not None


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
def api_client(monkeypatch):
    monkeypatch.setenv(main.API_KEY_ENV_NAME, API_KEY)
    sectors = [Sector("1", "半導體業", ["2330.TW", "2303.TW"]), Sector("2", "航運業", ["2603.TW"])]
    monkeypatch.setattr(main.sector_cache, "get", lambda: sectors)

    submitted = {}

    class FakeJob:
        job_id = "a" * 32
        tickers = []

    def fake_submit(sector_names, tickers, strategy="standard"):
        submitted["sector_names"] = sector_names
        submitted["tickers"] = tickers
        submitted["strategy"] = strategy
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

    def test_start_scan_with_zero_selects_all_sectors(self, api_client):
        response = api_client.post("/api/scans", json={"sectorIds": ["0"]}, headers={"X-Api-Key": API_KEY})

        assert response.status_code == 202
        assert response.json() == {"jobId": "a" * 32, "totalCount": 3}
        assert api_client.submitted["sector_names"] == ["半導體業", "航運業"]
        assert api_client.submitted["strategy"] == "standard"

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

    def test_signal_analysis_rejects_invalid_period(self, api_client):
        response = api_client.get("/api/stocks/2330/signal-analysis?period=99y", headers={"X-Api-Key": API_KEY})

        assert response.status_code == 400

    def test_kd_backtest_rejects_future_start_date(self, api_client):
        response = api_client.get("/api/market/kd-backtest?startDate=2999-01-01", headers={"X-Api-Key": API_KEY})

        assert response.status_code == 400

    def test_margin_days_out_of_range_is_rejected(self, api_client):
        response = api_client.get("/api/market/margin?days=5000", headers={"X-Api-Key": API_KEY})

        assert response.status_code == 422
