import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main, market_valuation, valuation_data
from app.valuation_data import MarketValuationService, MarketValuationStore

API_KEY = "test-key-for-unit-tests"


def month_end_frame(months, index_close, pe, cpi_growth=0.0, start="2010-01"):
    """每月一筆快照：盈餘 = 市值 ÷ 本益比；CPI 每月以 cpi_growth 成長。"""
    dates = pd.period_range(start, periods=months, freq="M").to_timestamp(how="end").normalize()
    frame = pd.DataFrame(
        {
            "indexClose": index_close if np.ndim(index_close) else [index_close] * months,
            "marketCap": [1_000.0] * months,
            "stockCount": 900,
            "profitableCount": 700,
        },
        index=dates,
    )
    frame["earnings"] = frame["marketCap"] / (pe if np.ndim(pe) else np.full(months, pe))
    cpi = pd.Series(100 * (1 + cpi_growth) ** np.arange(months), index=dates.to_period("M"))
    return frame, cpi


class TestAggregateMarketEarnings:
    def test_market_cap_weighted_earnings_of_common_stocks_only(self):
        quotes = {
            "2330": (100.0, 20.0),   # 市值 1,000、盈餘 50
            "2603": (50.0, 10.0),    # 市值 1,000、盈餘 100
            "2498": (10.0, None),    # 虧損（本益比「-」）：市值計入、盈餘以 0 計
            "0050": (150.0, None),   # ETF 不計
            "1101": (None, 15.0),    # 當日無收盤價不計
            "9999": (20.0, 10.0),    # 查無發行股數不計
        }
        shares = {"2330": 10, "2603": 20, "2498": 100, "0050": 1_000, "1101": 10}

        result = market_valuation.aggregate_market_earnings(quotes, shares)

        assert result == {"marketCap": 3_000.0, "earnings": 150.0, "stockCount": 3, "profitableCount": 2}

    def test_returns_none_without_any_earnings(self):
        assert market_valuation.aggregate_market_earnings({"2498": (10.0, None)}, {"2498": 100}) is None


class TestParseTwseReports:
    def test_quotes_and_shares_are_read_by_field_name(self):
        quotes_report = {"stat": "OK", "tables": [
            {"title": "大盤統計資訊", "fields": ["成交統計", "成交金額(元)"], "data": [["1.一般股票", "1"]]},
            {"fields": ["證券代號", "證券名稱", "收盤價", "本益比"],
             "data": [["2330", "台積電", "1,460.00", "25.31"], ["2498", "宏達電", "40.10", "-"], ["1234", "某股", "--", "0.00"]]},
        ]}
        shares_report = {"stat": "OK", "fields": ["證券代號", "證券名稱", "國際證券編碼", "發行股數"],
                         "data": [["2330", "台積電", "TW0002330008", "25,932,733,242"]]}

        assert valuation_data.parse_quotes(quotes_report) == {
            "2330": (1460.0, 25.31), "2498": (40.1, None), "1234": (None, 0.0)}
        assert valuation_data.parse_shares(shares_report) == {"2330": 25_932_733_242.0}

    def test_missing_tables_give_empty_results(self):
        assert valuation_data.parse_quotes({"stat": "OK", "tables": []}) == {}
        assert valuation_data.parse_shares({"stat": "OK", "fields": ["證券代號"], "data": []}) == {}


class TestParseCpiTable:
    def test_reads_roc_years_and_monthly_columns(self):
        header = ["民國年"] + [f"{month}月" for month in range(1, 13)] + ["累計平均"]
        table = pd.DataFrame([
            [None, "消費者物價指數銜接表"] + [None] * 12,
            header,
            ["114"] + [100 + month for month in range(12)] + [105.5],
            ["115", 120.5, 121.0] + [np.nan] * 10 + [120.75],
            ["說明：最近3個月資料均可能修正。"] + [None] * 13,
        ])

        cpi = market_valuation.parse_cpi_table(table)

        assert cpi[pd.Period("2025-01", freq="M")] == 100
        assert cpi[pd.Period("2025-12", freq="M")] == 111
        assert cpi.index[-1] == pd.Period("2026-02", freq="M")   # 尚未公布的月份不列入
        assert len(cpi) == 14

    def test_unexpected_layout_raises(self):
        with pytest.raises(ValueError):
            market_valuation.parse_cpi_table(pd.DataFrame([["年份", "1月"]]))


class TestSelectTargetDates:
    def test_last_trading_day_of_each_month(self):
        dates = pd.to_datetime(["2026-09-29", "2026-09-30", "2026-10-30", "2026-10-31", "2026-11-03", "2026-11-20"])

        assert market_valuation.select_target_dates(dates) == list(pd.to_datetime(["2026-09-30", "2026-10-31", "2026-11-20"]))


class TestCalculateCape:
    def test_constant_earnings_without_inflation_equals_pe(self):
        frame, cpi = month_end_frame(130, index_close=10_000.0, pe=20.0)

        result = market_valuation.calculate_cape(frame, cpi)

        assert result["cape"].iloc[:113].isna().all()          # 不足 114 個月不計算
        assert result["cape"].iloc[-1] == pytest.approx(20.0)
        assert result["eps"].iloc[-1] == pytest.approx(500.0)   # 每點盈餘 = 指數 ÷ 本益比

    def test_inflation_adjusts_past_earnings_to_today(self):
        # 名目每點盈餘固定，但物價每月上漲 0.5%：換算成今日幣值後，過去的盈餘較高，席勒本益比低於一般本益比
        frame, cpi = month_end_frame(120, index_close=10_000.0, pe=20.0, cpi_growth=0.005)

        result = market_valuation.calculate_cape(frame, cpi)

        real_average = np.mean(500.0 * cpi.iloc[-1] / cpi.to_numpy())
        assert result["cape"].iloc[-1] == pytest.approx(10_000.0 / real_average)
        assert result["cape"].iloc[-1] < result["pe"].iloc[-1]

    def test_latest_months_without_cpi_use_latest_published_cpi(self):
        frame, cpi = month_end_frame(120, index_close=10_000.0, pe=20.0)

        result = market_valuation.calculate_cape(frame, cpi.iloc[:-2])

        assert result["cpi"].iloc[-1] == cpi.iloc[-3]
        assert result["cape"].iloc[-1] == pytest.approx(20.0)

    def test_missing_months_do_not_stretch_window_beyond_ten_years(self):
        frame, cpi = month_end_frame(126, index_close=10_000.0, pe=np.r_[np.full(6, 5.0), np.full(120, 20.0)])

        # 拿掉 10 年視窗內的 3 個月仍可計算，且視窗不會因此往前納入 10 年前（本益比 5 倍）的月份
        result = market_valuation.calculate_cape(frame.drop(frame.index[[50, 60, 70]]), cpi)

        assert result["cape"].iloc[-1] == pytest.approx(20.0)

    def test_analysis_summary(self):
        index_close = np.r_[np.full(125, 10_000.0), [12_000.0]]
        frame, cpi = month_end_frame(126, index_close=index_close, pe=20.0)

        analysis = market_valuation.analyze_cape(market_valuation.calculate_cape(frame, cpi))

        json.dumps(analysis)
        assert analysis["capeStartDate"] == analysis["dates"][113]
        assert analysis["latest"]["cape"] > analysis["statistics"]["mean"]
        assert analysis["latest"]["capePercentile"] == 100.0
        assert analysis["statistics"]["maxDate"] == analysis["latest"]["date"]
        assert [row["date"] for row in analysis["recent"]][:2] == analysis["dates"][-1:-3:-1]
        assert len(analysis["recent"]) == 12

    def test_analysis_requires_ten_years(self):
        frame, cpi = month_end_frame(60, index_close=10_000.0, pe=20.0)

        with pytest.raises(ValueError):
            market_valuation.analyze_cape(market_valuation.calculate_cape(frame, cpi))


SNAPSHOT = {"marketCap": 1_000.0, "earnings": 50.0, "stockCount": 3, "profitableCount": 2}


class FakeClock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


def make_service(tmp_path, fetcher, clock=None, cpi=None):
    cpi_series = cpi if cpi is not None else pd.Series({pd.Period("2026-10", freq="M"): 112.9})
    return MarketValuationService(MarketValuationStore(str(tmp_path)), fetcher, lambda: cpi_series, clock or FakeClock())


class TestMarketValuationService:
    TRADE_DATES = pd.to_datetime(["2005-08-31", "2026-09-29", "2026-09-30", "2026-10-30", "2026-11-19", "2026-11-20"])

    def test_fetches_last_trading_day_of_each_month_and_saves_to_disk(self, tmp_path):
        requested = []
        service = make_service(tmp_path, lambda day: requested.append(day) or SNAPSHOT)

        assert service.update_in_background(self.TRADE_DATES)
        service.wait(5)

        # 2005-08 早於證交所提供本益比的月份，不抓
        assert requested == list(pd.to_datetime(["2026-09-30", "2026-10-30", "2026-11-20"]))
        assert sorted(service.snapshots()) == ["2026-09-30", "2026-10-30", "2026-11-20"]
        assert service.progress() == {"state": "idle", "processedMonths": 3, "totalMonths": 3, "storedMonths": 3, "lastError": None}
        # 服務重新啟動後從硬碟載入，不需重新抓取
        reloaded = make_service(tmp_path, lambda day: pytest.fail("不應重新抓取"))
        assert sorted(reloaded.snapshots()) == ["2026-09-30", "2026-10-30", "2026-11-20"]
        assert reloaded.cpi().iloc[-1] == 112.9
        assert not reloaded.update_in_background(self.TRADE_DATES)

    def test_falls_back_to_earlier_trading_day_in_same_month(self, tmp_path):
        service = make_service(tmp_path, lambda day: None if day == pd.Timestamp("2026-09-30") else SNAPSHOT)

        service.update_in_background(self.TRADE_DATES[:3])
        service.wait(5)

        assert list(service.snapshots()) == ["2026-09-29"]

    def test_latest_month_is_replaced_when_a_newer_trading_day_appears(self, tmp_path):
        service = make_service(tmp_path, lambda day: SNAPSHOT)
        service.update_in_background(self.TRADE_DATES[:5])
        service.wait(5)
        assert "2026-11-19" in service.snapshots()

        service.update_in_background(self.TRADE_DATES)
        service.wait(5)

        assert [key for key in service.snapshots() if key.startswith("2026-11")] == ["2026-11-20"]

    def test_failed_month_is_not_retried_within_an_hour(self, tmp_path):
        clock = FakeClock()
        calls = []
        service = make_service(tmp_path, lambda day: calls.append(day), clock=clock)
        service.update_in_background(self.TRADE_DATES[-2:])
        service.wait(5)
        assert len(calls) == 2   # 同月份兩個候選交易日都試過

        assert service.pending_months(self.TRADE_DATES[-2:]) == []
        clock.now += valuation_data.RETRY_AFTER_SECONDS
        assert len(service.pending_months(self.TRADE_DATES[-2:])) == 1

    def test_cpi_failure_without_cache_is_reported(self, tmp_path):
        def failing_cpi():
            raise RuntimeError("連線逾時")

        service = MarketValuationService(MarketValuationStore(str(tmp_path)), lambda day: SNAPSHOT, failing_cpi, FakeClock())
        service.update_in_background(self.TRADE_DATES[-1:])
        service.wait(5)

        assert service.cpi() is None
        assert "連線逾時" in service.progress()["lastError"]


class TestTwseClient:
    def test_waits_between_requests(self, monkeypatch):
        sleeps = []
        client = valuation_data.TwseClient(request_delay_seconds=3, sleep=sleeps.append)
        responses = iter([
            {"stat": "OK", "tables": [{"fields": ["證券代號", "收盤價", "本益比"], "data": [["2330", "100", "20"]]}]},
            {"stat": "OK", "fields": ["證券代號", "發行股數"], "data": [["2330", "10"]]},
        ])

        class FakeResponse:
            def __init__(self, body):
                self.body = body

            def raise_for_status(self):
                pass

            def json(self):
                return self.body

        monkeypatch.setattr(valuation_data.requests, "get", lambda *args, **kwargs: FakeResponse(next(responses)))

        snapshot = client.fetch_snapshot(pd.Timestamp("2026-10-30"))

        assert snapshot == {"marketCap": 1_000.0, "earnings": 50.0, "stockCount": 1, "profitableCount": 1}
        assert len(sleeps) == 1 and 0 < sleeps[0] <= 3

    def test_holiday_or_error_returns_none(self, monkeypatch):
        client = valuation_data.TwseClient(request_delay_seconds=0)

        class Holiday:
            def raise_for_status(self):
                pass

            def json(self):
                return {"stat": "很抱歉，沒有符合條件的資料!"}

        monkeypatch.setattr(valuation_data.requests, "get", lambda *args, **kwargs: Holiday())

        assert client.fetch_snapshot(pd.Timestamp("2026-10-10")) is None


class TestCapeApi:
    @pytest.fixture
    def cape_client(self, monkeypatch, tmp_path):
        monkeypatch.setenv(main.API_KEY_ENV_NAME, API_KEY)
        frame, cpi = month_end_frame(126, index_close=10_000.0, pe=20.0)
        snapshots = {date.strftime("%Y-%m-%d"): {k: row[k] for k in ("marketCap", "earnings", "stockCount", "profitableCount")}
                     for date, row in frame.iterrows()}
        service = make_service(tmp_path, lambda day: pytest.fail("快照已齊全，不應抓取"), cpi=cpi)
        # 模擬已有硬碟快取：快照齊全、CPI 剛更新過
        service._snapshots = snapshots
        service._cpi, service._cpi_fetched_at = cpi, service._clock()
        monkeypatch.setattr(main, "valuation_service", service)
        prices = pd.DataFrame({"Close": frame["indexClose"]}, index=frame.index)
        monkeypatch.setattr(main.market_data, "download_single_history", lambda ticker, **kwargs: prices)
        monkeypatch.setattr(main.market_analysis, "drop_unclosed_session", lambda frame, now: frame)
        return TestClient(main.app), service

    def test_returns_analysis_when_ten_years_available(self, cape_client):
        client, _ = cape_client

        body = client.get("/api/market/cape", headers={"X-Api-Key": API_KEY}).json()

        assert body["ready"] is True
        assert body["analysis"]["latest"]["cape"] == 20.0
        assert body["progress"]["state"] == "idle"

    def test_reports_progress_while_history_is_incomplete(self, cape_client):
        client, service = cape_client
        service._snapshots = dict(list(service._snapshots.items())[-60:])
        service._snapshot_fetcher = lambda day: None

        body = client.get("/api/market/cape", headers={"X-Api-Key": API_KEY}).json()
        service.wait(5)

        assert body["ready"] is False
        assert body["analysis"] is None
        assert body["progress"]["totalMonths"] == 66

    def test_requires_api_key(self, cape_client):
        client, _ = cape_client

        assert client.get("/api/market/cape").status_code == 401
