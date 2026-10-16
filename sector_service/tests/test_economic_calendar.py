from datetime import date, timedelta

import pytest
import requests
from fastapi.testclient import TestClient

from app import economic_calendar, main
from app.fred_client import FredApiError, FredClient

FAKE_FRED_KEY = "fake-fred-key-0123456789abcdef"


class FakeFredClient:
    """以記憶體資料模擬 FRED，記錄被查詢的數列。"""

    def __init__(self, release_dates, observations=None):
        self._release_dates = release_dates
        self._observations = observations or {}
        self.observation_calls = []

    def get_release_dates(self, start_date, end_date):
        return [d for d in self._release_dates if start_date <= d["date"] <= end_date]

    def get_observations(self, series_id, units="lin", limit=2):
        self.observation_calls.append((series_id, units, limit))
        return self._observations.get(series_id, [])[:limit]


def release(release_id, day, name="Some Release"):
    return {"release_id": release_id, "release_name": name, "date": day}


class TestBuildCalendar:
    def test_important_scope_lists_only_curated_releases_with_chinese_names(self):
        client = FakeFredClient(
            [release(10, "2026-11-10", "Consumer Price Index"), release(322, "2026-11-10", "Business Leaders Survey")],
            {"CPIAUCSL": [{"date": "2026-09-01", "value": "3.71725"}, {"date": "2026-08-01", "value": "3.35302"}]},
        )

        result = economic_calendar.build_calendar(client, date(2026, 11, 1), date(2026, 11, 30))

        assert [e["releaseId"] for e in result["events"]] == [10]
        event = result["events"][0]
        assert event["name"] == "美國消費者物價指數（CPI）"
        assert event["importance"] == "high" and event["country"] == "US"
        assert event["hasIndicators"] is True
        # 日曆本身不查詢數列，指標值由 get_release_indicators 另外取得
        assert client.observation_calls == []

        cpi = economic_calendar.get_release_indicators(client, [10])["10"][0]
        assert (cpi["label"], cpi["latestValue"], cpi["previousValue"], cpi["error"]) == ("CPI 年增率", 3.72, 3.35, None)

    def test_all_scope_includes_other_releases_but_skips_daily_ones(self):
        start = date(2026, 11, 2)
        daily = [release(197, (start + timedelta(days=i)).isoformat(), "Dow Jones Averages") for i in range(14)]
        client = FakeFredClient(daily + [release(322, "2026-11-10", "Business Leaders Survey")])

        result = economic_calendar.build_calendar(client, start, start + timedelta(days=13), economic_calendar.SCOPE_ALL)

        assert [(e["releaseId"], e["name"], e["importance"]) for e in result["events"]] == [
            (322, "Business Leaders Survey", None)
        ]

    def test_fomc_and_ecb_daily_rate_releases_are_never_listed(self):
        client = FakeFredClient([release(101, "2026-11-04", "FOMC Press Release"), release(484, "2026-11-04")])

        for scope in (economic_calendar.SCOPE_IMPORTANT, economic_calendar.SCOPE_ALL):
            assert economic_calendar.build_calendar(client, date(2026, 11, 1), date(2026, 11, 30), scope)["events"] == []

    def test_indicator_values_are_fetched_once_per_release_and_scaled(self):
        client = FakeFredClient([], {"ICSA": [{"date": "2026-10-31", "value": "199000"}, {"date": "2026-10-24", "value": "."}]})

        result = economic_calendar.get_release_indicators(client, [180, 180, 99999])   # 重複與未收錄的 id

        assert list(result) == ["180"]
        assert client.observation_calls == [("ICSA", "lin", 2)]
        indicator = result["180"][0]
        assert indicator["latestValue"] == 199          # 人數 → 千人
        assert indicator["previousValue"] is None       # FRED 以 "." 表示缺值

    def test_one_failing_series_does_not_break_other_indicators(self):
        class PartlyFailingClient(FakeFredClient):
            def get_observations(self, series_id, units="lin", limit=2):
                if series_id == "UNRATE":
                    raise FredApiError("FRED returned 502")
                return super().get_observations(series_id, units, limit)

        client = PartlyFailingClient([], {"PAYEMS": [{"date": "2026-09-01", "value": "29"}]})

        payrolls, unemployment = economic_calendar.get_release_indicators(client, [50])["50"]

        assert payrolls["latestValue"] == 29 and payrolls["error"] is None
        assert unemployment["latestValue"] is None and unemployment["error"] == "暫時無法取得"

    def test_past_and_today_flags(self):
        today = date.today()
        client = FakeFredClient([release(322, (today - timedelta(days=1)).isoformat()), release(322, today.isoformat())])

        events = economic_calendar.build_calendar(client, today - timedelta(days=1), today, economic_calendar.SCOPE_ALL)["events"]

        assert [(e["isPast"], e["isToday"]) for e in events] == [(True, False), (False, True)]


class TestPolicyRates:
    def test_last_change_is_detected_from_history(self):
        client = FakeFredClient([], {
            "DFEDTARU": [
                {"date": "2026-10-15", "value": "4.00"},
                {"date": "2026-09-18", "value": "4.00"},
                {"date": "2026-09-17", "value": "4.25"},
            ],
            "DFEDTARL": [{"date": "2026-10-15", "value": "3.75"}],
            "ECBDFR": [{"date": "2026-10-15", "value": "2.5"}],
        })

        rates = {r["code"]: r for r in economic_calendar.get_policy_rates(client)}

        assert rates["FED"]["value"] == 4.0 and rates["FED"]["lowerValue"] == 3.75
        assert rates["FED"]["lastChange"] == {"date": "2026-09-18", "from": 4.25, "to": 4.0}
        assert rates["ECB"]["lastChange"] is None    # 歷史資料內沒有變動


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class TestFredClient:
    def test_release_dates_are_paginated(self):
        calls = []

        def fake_get(url, params, timeout):
            calls.append(params["offset"])
            total = 1500
            start = params["offset"]
            batch = [{"release_id": 1, "date": "2026-11-01"}] * min(params["limit"], total - start)
            return FakeResponse(200, {"count": total, "release_dates": batch})

        dates = FredClient(FAKE_FRED_KEY, http_get=fake_get).get_release_dates("2026-11-01", "2026-11-30")

        assert len(dates) == 1500
        assert calls == [0, 1000]

    def test_results_are_cached(self):
        calls = []

        def fake_get(url, params, timeout):
            calls.append(url)
            return FakeResponse(200, {"observations": [{"date": "2026-10-01", "value": "1"}]})

        client = FredClient(FAKE_FRED_KEY, http_get=fake_get)
        client.get_observations("UNRATE")
        client.get_observations("UNRATE")

        assert len(calls) == 1

    def test_connection_error_message_does_not_contain_api_key(self, monkeypatch):
        monkeypatch.setattr("app.fred_client.RETRY_DELAY_SECONDS", 0)

        def fake_get(url, params, timeout):
            # requests 的例外訊息通常包含完整網址（含 api_key）
            raise requests.ConnectionError(f"Max retries exceeded with url: {url}?api_key={params['api_key']}")

        with pytest.raises(FredApiError) as error:
            FredClient(FAKE_FRED_KEY, http_get=fake_get).get_observations("UNRATE")

        assert FAKE_FRED_KEY not in str(error.value)
        assert error.value.__cause__ is None and error.value.__suppress_context__

    def test_timeout_is_retried_once(self, monkeypatch):
        monkeypatch.setattr("app.fred_client.RETRY_DELAY_SECONDS", 0)
        attempts = []

        def flaky_get(url, params, timeout):
            attempts.append(1)
            if len(attempts) == 1:
                raise requests.ReadTimeout("slow")
            return FakeResponse(200, {"observations": [{"date": "2026-10-01", "value": "4.2"}]})

        assert FredClient(FAKE_FRED_KEY, http_get=flaky_get).get_observations("UNRATE")[0]["value"] == "4.2"
        assert len(attempts) == 2

    def test_client_error_is_not_retried(self, monkeypatch):
        monkeypatch.setattr("app.fred_client.RETRY_DELAY_SECONDS", 0)
        attempts = []

        def bad_request(url, params, timeout):
            attempts.append(1)
            return FakeResponse(400, {"error_message": "Bad Request."})

        with pytest.raises(FredApiError):
            FredClient(FAKE_FRED_KEY, http_get=bad_request).get_observations("NOPE")
        assert len(attempts) == 1

    def test_stale_data_is_used_when_refresh_fails(self, monkeypatch):
        monkeypatch.setattr("app.fred_client.RETRY_DELAY_SECONDS", 0)
        state = {"fail": False}

        def fake_get(url, params, timeout):
            if state["fail"]:
                raise requests.ReadTimeout("FRED is slow")
            return FakeResponse(200, {"observations": [{"date": "2026-10-01", "value": "4.2"}]})

        client = FredClient(FAKE_FRED_KEY, cache_seconds=0, http_get=fake_get)   # 快取立即過期
        client.get_observations("UNRATE")
        state["fail"] = True

        assert client.get_observations("UNRATE")[0]["value"] == "4.2"

    def test_error_status_raises_with_fred_message(self):
        fake_get = lambda url, params, timeout: FakeResponse(400, {"error_message": "Bad Request. Series does not exist."})

        with pytest.raises(FredApiError, match="400: Bad Request. Series does not exist."):
            FredClient(FAKE_FRED_KEY, http_get=fake_get).get_observations("NOPE")

    def test_missing_api_key_is_rejected(self):
        with pytest.raises(ValueError):
            FredClient("")


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv(main.API_KEY_ENV_NAME, "service-key")
    monkeypatch.setenv(main.FRED_API_KEY_ENV_NAME, FAKE_FRED_KEY)
    return TestClient(main.app)


HEADERS = {"X-Api-Key": "service-key"}


class TestCalendarApi:
    def test_returns_503_when_fred_key_not_configured(self, api_client, monkeypatch):
        monkeypatch.delenv(main.FRED_API_KEY_ENV_NAME)

        response = api_client.get("/api/calendar?startDate=2026-11-01&endDate=2026-11-30", headers=HEADERS)

        assert response.status_code == 503

    @pytest.mark.parametrize("query", [
        "startDate=2026-11-30&endDate=2026-11-01",     # 結束早於開始
        "startDate=2026-01-01&endDate=2026-12-31",     # 超過 92 天
        "startDate=2026-11-01&endDate=2026-11-30&scope=everything",
    ])
    def test_invalid_query_returns_400(self, api_client, query):
        assert api_client.get(f"/api/calendar?{query}", headers=HEADERS).status_code == 400

    def test_fred_error_becomes_502_without_leaking_key(self, api_client, monkeypatch):
        def failing(*args, **kwargs):
            raise FredApiError("Unable to reach FRED (ConnectionError)")

        monkeypatch.setattr(main.economic_calendar, "build_calendar", failing)

        response = api_client.get("/api/calendar?startDate=2026-11-01&endDate=2026-11-30", headers=HEADERS)

        assert response.status_code == 502
        assert FAKE_FRED_KEY not in response.text

    @pytest.mark.parametrize("ids", ["10;50", "abc", "", ",".join(["1"] * 51)])
    def test_invalid_release_ids_are_rejected(self, api_client, ids):
        response = api_client.get(f"/api/calendar/indicators?releaseIds={ids}", headers=HEADERS)

        assert response.status_code in (400, 422)

    def test_indicators_endpoint(self, api_client, monkeypatch):
        monkeypatch.setattr(main.economic_calendar, "get_release_indicators", lambda client, ids: {str(i): [] for i in ids})

        response = api_client.get("/api/calendar/indicators?releaseIds=10,50", headers=HEADERS)

        assert response.status_code == 200 and response.json() == {"indicators": {"10": [], "50": []}}

    def test_calendar_success(self, api_client, monkeypatch):
        monkeypatch.setattr(main.economic_calendar, "build_calendar", lambda client, s, e, scope: {"events": [], "scope": scope})

        response = api_client.get("/api/calendar?startDate=2026-11-01&endDate=2026-11-30&scope=all", headers=HEADERS)

        assert response.status_code == 200 and response.json()["scope"] == "all"
