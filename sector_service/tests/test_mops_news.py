import json

import pytest
import requests
from fastapi.testclient import TestClient

from app import main, mops_news
from datetime import date

from app.mops_news import MopsApiError, MopsClient, filter_recent_items, one_month_before, roc_to_iso

SPOKESPERSON_NAME = "王小明"
SPOKESPERSON_PHONE = "02-1234-5678#100"


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def summary_payload(company_id="3073", name="天方能源", market_kind="otc"):
    """依實測的 t146sb05 回應格式建構（主旨含 \\r\\n 硬斷行）。"""
    return {"code": 200, "message": "查詢成功", "result": {
        "basic_info": {"companyId": company_id, "companyAbbreviation": name, "industryCategory": "上櫃綠能環保"},
        "recent_important_news": {"data": [
            ["115/10/14", "公告本公司115年9月份自結合併財務報告之\r\n負債比率、流動比率及速動比率",
             {"apiName": "t05st01_detail", "parameters": {
                 "marketKind": market_kind, "companyId": company_id, "enterDate": "1151013", "serialNumber": "2"}}],
            ["115/09/23", "公告本公司向關係人取得使用權資產",
             {"apiName": "t05st01_detail", "parameters": {
                 "marketKind": market_kind, "companyId": company_id, "enterDate": "1150923", "serialNumber": "2"}}],
        ]},
    }}


def detail_payload():
    return {"code": 200, "message": "查詢成功", "result": {
        "marketName": "上櫃公司", "companyId": "3073", "companyAbbreviation": "天方能源",
        "data": [["2", "115/10/14", "16:28:14", SPOKESPERSON_NAME + "      ", "協理     ", SPOKESPERSON_PHONE,
                  "公告本公司115年9月份自結合併財務報告之\r\n負債比率、流動比率及速動比率     ",
                  "第53款", "115/10/14", "1.事實發生日:115/10/14\n2.發生緣由:依規定辦理公告。\n"]],
    }}


class RecordingPost:
    """記錄呼叫並依 API 名稱回傳預設回應的假 requests.post。"""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, url, json, headers, timeout):
        api_name = url.rsplit("/", 1)[-1]
        self.calls.append((api_name, json))
        response = self.responses[api_name]
        if callable(response):
            return response(json)
        return response


@pytest.fixture(autouse=True)
def no_delay(monkeypatch):
    monkeypatch.setattr(mops_news, "RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(mops_news, "REQUEST_INTERVAL_SECONDS", 0)


class TestRocDate:
    @pytest.mark.parametrize("raw, expected", [
        ("115/10/14", "2026-10-14"), ("1151013", "2026-10-13"), (" 99/01/05 ", "2010-01-05"),
        ("115/02/30", None), ("abc", None), ("", None),
    ])
    def test_roc_to_iso(self, raw, expected):
        assert roc_to_iso(raw) == expected


class TestOneMonthWindow:
    @pytest.mark.parametrize("today, expected", [
        (date(2026, 10, 28), date(2026, 9, 28)),
        (date(2026, 3, 31), date(2026, 2, 28)),    # 2 月沒有 31 日 → 月底
        (date(2028, 3, 30), date(2028, 2, 29)),    # 閏年
        (date(2026, 1, 15), date(2025, 12, 15)),   # 跨年
    ])
    def test_one_month_before(self, today, expected):
        assert one_month_before(today) == expected

    def test_filter_keeps_items_on_or_after_cutoff_without_mutating_cache(self):
        news = {"companyId": "3073", "items": [
            {"date": "2026-10-14"}, {"date": "2026-09-28"}, {"date": "2026-09-27"}, {"date": None},
        ]}

        filtered = filter_recent_items(news, date(2026, 9, 28))

        assert [i["date"] for i in filtered["items"]] == ["2026-10-14", "2026-09-28"]
        assert len(news["items"]) == 4   # 快取中的原始資料不變

    def test_get_news_for_companies_applies_since(self):
        client = MopsClient(http_post=RecordingPost({"t146sb05": FakeResponse(summary_payload())}))

        result = client.get_news_for_companies(["3073"], since=date(2026, 10, 1))

        assert [i["date"] for i in result["companies"][0]["items"]] == ["2026-10-14"]
        assert result["sinceDate"] == "2026-10-01"
        # 未指定 since 時保留全部（同一份快取）
        assert len(client.get_news_for_companies(["3073"])["companies"][0]["items"]) == 2


class TestRecentNews:
    def test_parses_summary_and_joins_hard_wrapped_subject(self):
        client = MopsClient(http_post=RecordingPost({"t146sb05": FakeResponse(summary_payload())}))

        news = client.get_recent_news("3073")

        assert (news["name"], news["market"], len(news["items"])) == ("天方能源", "上櫃", 2)
        first = news["items"][0]
        assert first["date"] == "2026-10-14"
        assert first["subject"] == "公告本公司115年9月份自結合併財務報告之負債比率、流動比率及速動比率"
        assert first["detail"] == {"marketKind": "otc", "enterDate": "1151013", "serialNumber": "2"}
        assert news["sourceUrl"].endswith("t146sb05?companyId=3073")

    def test_results_are_cached(self):
        post = RecordingPost({"t146sb05": FakeResponse(summary_payload())})
        client = MopsClient(http_post=post)

        client.get_recent_news("3073")
        client.get_recent_news("3073")

        assert len(post.calls) == 1

    def test_unknown_company_is_client_error(self):
        post = RecordingPost({"t146sb05": FakeResponse({"code": 500, "message": "公司代號格式錯誤", "result": None})})

        with pytest.raises(MopsApiError) as error:
            MopsClient(http_post=post).get_recent_news("9999")

        assert error.value.is_client_error and str(error.value) == "公司代號格式錯誤"

    def test_invalid_company_id_is_rejected_without_calling_mops(self):
        post = RecordingPost({})

        with pytest.raises(MopsApiError):
            MopsClient(http_post=post).get_recent_news("2330;DROP")
        assert post.calls == []

    def test_one_failing_company_does_not_break_others(self):
        def by_company(payload):
            if payload["companyId"] == "9999":
                return FakeResponse({"code": 500, "message": "公司代號格式錯誤"})
            return FakeResponse(summary_payload(payload["companyId"], "台積電", "sii"))

        client = MopsClient(http_post=RecordingPost({"t146sb05": by_company}))

        result = client.get_news_for_companies(["2330", "9999", "2330"])   # 重複代號只查一次

        assert [c["companyId"] for c in result["companies"]] == ["2330"]
        assert result["companies"][0]["market"] == "上市"
        assert result["errors"] == [{"companyId": "9999", "message": "公司代號格式錯誤"}]

    def test_timeout_is_retried_then_reported_as_server_error(self):
        def always_timeout(payload):
            raise requests.ReadTimeout("slow")

        with pytest.raises(MopsApiError) as error:
            MopsClient(http_post=RecordingPost({"t146sb05": always_timeout})).get_recent_news("3073")

        assert not error.value.is_client_error


class TestNewsDetail:
    def test_detail_excludes_spokesperson_personal_data(self):
        client = MopsClient(http_post=RecordingPost({"t05st01_detail": FakeResponse(detail_payload())}))

        detail = client.get_news_detail("3073", "otc", "1151013", "2")

        serialized = json.dumps(detail, ensure_ascii=False)
        assert SPOKESPERSON_NAME not in serialized and SPOKESPERSON_PHONE not in serialized
        assert detail["speakDate"] == "2026-10-14" and detail["speakTime"] == "16:28:14"
        assert detail["clause"] == "第53款" and detail["factDate"] == "2026-10-14"
        assert detail["subject"] == "公告本公司115年9月份自結合併財務報告之負債比率、流動比率及速動比率"
        assert detail["description"].startswith("1.事實發生日:115/10/14\n2.發生緣由")

    @pytest.mark.parametrize("args", [
        ("3073", "OTC", "1151013", "2"), ("3073", "otc", "2026-10-13", "2"), ("3073", "otc", "1151013", "x"),
    ])
    def test_invalid_parameters_are_rejected(self, args):
        post = RecordingPost({})

        with pytest.raises(MopsApiError):
            MopsClient(http_post=post).get_news_detail(*args)
        assert post.calls == []


HEADERS = {"X-Api-Key": "service-key"}


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv(main.API_KEY_ENV_NAME, "service-key")
    return TestClient(main.app)


class TestMopsApi:
    @pytest.mark.parametrize("ids", ["", "2330;1", ",".join(["2330"] * 11), "../etc"])
    def test_invalid_company_ids_are_rejected(self, api_client, ids):
        assert api_client.get(f"/api/mops/news?companyIds={ids}", headers=HEADERS).status_code in (400, 422)

    def test_news_endpoint_normalizes_codes(self, api_client, monkeypatch):
        captured = {}
        def fake_get_news(ids, since=None):
            captured.update(ids=ids, since=since)
            return {"companies": [], "errors": [], "sinceDate": since.isoformat()}

        monkeypatch.setattr(main.mops_client, "get_news_for_companies", fake_get_news)

        response = api_client.get("/api/mops/news?companyIds=3073, 2330 ,00878", headers=HEADERS)

        assert response.status_code == 200
        assert captured["ids"] == ["3073", "2330", "00878"]
        assert captured["since"] == one_month_before(date.today())   # 只取近一個月

    def test_detail_not_found_maps_to_404_and_outage_to_502(self, api_client, monkeypatch):
        query = "/api/mops/news/detail?companyId=3073&marketKind=otc&enterDate=1151013&serialNumber=2"

        def not_found(*args):
            raise MopsApiError("查無此重大訊息內容", is_client_error=True)

        monkeypatch.setattr(main.mops_client, "get_news_detail", not_found)
        assert api_client.get(query, headers=HEADERS).status_code == 404

        def outage(*args):
            raise MopsApiError("無法連線公開資訊觀測站（ConnectionError）")

        monkeypatch.setattr(main.mops_client, "get_news_detail", outage)
        assert api_client.get(query, headers=HEADERS).status_code == 502

    def test_detail_parameter_validation(self, api_client):
        response = api_client.get("/api/mops/news/detail?companyId=3073&marketKind=otc&enterDate=abc&serialNumber=2", headers=HEADERS)

        assert response.status_code == 422
