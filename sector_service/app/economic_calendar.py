"""全球經濟日曆（Domain / Application 層）：以 FRED 的發布日程為來源。

資料限制（已實測確認）：
- FRED 以美國經濟數據為主，國際部分僅有歐元區、日本、OECD 等少數項目，沒有台灣、中國的官方發布日程
- FRED 只提供「日期」，不含發布時間
- FOMC Press Release（101）與 Key ECB Interest Rates（484）在 FRED 中每天都有日期（底下是每日利率數列），
  並非會議日程，因此不列入日曆，改以「目前政策利率」呈現（含上次調整日期）
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, List, Optional, Tuple

from .fred_client import FredApiError, FredClient

logger = logging.getLogger(__name__)

# 同時向 FRED 查詢的數列數量（FRED 限制每分鐘 120 次請求）
INDICATOR_FETCH_WORKERS = 6

IMPORTANCE_HIGH = "high"
IMPORTANCE_MEDIUM = "medium"

COUNTRY_US = "US"
COUNTRY_EU = "EU"
COUNTRY_JP = "JP"
COUNTRY_GLOBAL = "OECD"


@dataclass(frozen=True)
class KeyIndicator:
    """發布內容中要顯示「最新值 / 前值」的數列。"""

    series_id: str
    label: str
    units: str = "lin"          # FRED 換算：lin 原值、chg 變動量、pch 月(季)增率、pc1 年增率
    unit_label: str = ""
    scale: float = 1.0          # 顯示前乘上的倍數（例如人數 → 千人）
    decimals: int = 1


@dataclass(frozen=True)
class CalendarRelease:
    release_id: int
    name_zh: str
    country: str
    importance: str
    indicators: Tuple[KeyIndicator, ...] = field(default_factory=tuple)


# 重要經濟指標（release_id 與數列 ID 皆已用 FRED API 實測確認存在）
CALENDAR_RELEASES: Dict[int, CalendarRelease] = {r.release_id: r for r in [
    # ---------------- 美國：高重要性
    CalendarRelease(10, "美國消費者物價指數（CPI）", COUNTRY_US, IMPORTANCE_HIGH, (
        KeyIndicator("CPIAUCSL", "CPI 年增率", "pc1", "%", decimals=2),
        KeyIndicator("CPILFESL", "核心 CPI 年增率", "pc1", "%", decimals=2),
    )),
    CalendarRelease(50, "美國非農就業報告", COUNTRY_US, IMPORTANCE_HIGH, (
        KeyIndicator("PAYEMS", "非農新增就業", "chg", "千人", decimals=0),
        KeyIndicator("UNRATE", "失業率", "lin", "%", decimals=1),
    )),
    CalendarRelease(53, "美國國內生產毛額（GDP）", COUNTRY_US, IMPORTANCE_HIGH, (
        KeyIndicator("A191RL1Q225SBEA", "實質 GDP 季增年率", "lin", "%", decimals=1),
    )),
    CalendarRelease(54, "美國個人所得與支出（PCE）", COUNTRY_US, IMPORTANCE_HIGH, (
        KeyIndicator("PCEPI", "PCE 物價年增率", "pc1", "%", decimals=2),
        KeyIndicator("PCEPILFE", "核心 PCE 年增率", "pc1", "%", decimals=2),
    )),
    CalendarRelease(46, "美國生產者物價指數（PPI）", COUNTRY_US, IMPORTANCE_HIGH, (
        KeyIndicator("PPIFIS", "PPI 最終需求年增率", "pc1", "%", decimals=2),
    )),
    CalendarRelease(9, "美國零售銷售", COUNTRY_US, IMPORTANCE_HIGH, (
        KeyIndicator("RSAFS", "零售銷售月增率", "pch", "%", decimals=2),
    )),
    # ---------------- 美國：中重要性
    CalendarRelease(180, "美國初領失業救濟金人數（每週）", COUNTRY_US, IMPORTANCE_MEDIUM, (
        KeyIndicator("ICSA", "初領失業金人數", "lin", "千人", scale=0.001, decimals=0),
    )),
    CalendarRelease(194, "美國 ADP 民間就業報告", COUNTRY_US, IMPORTANCE_MEDIUM, (
        KeyIndicator("ADPMNUSNERSA", "ADP 新增就業", "chg", "千人", scale=0.001, decimals=0),
    )),
    CalendarRelease(192, "美國職缺與勞動流動調查（JOLTS）", COUNTRY_US, IMPORTANCE_MEDIUM, (
        KeyIndicator("JTSJOL", "職缺數", "lin", "千個", decimals=0),
    )),
    CalendarRelease(13, "美國工業生產與產能利用率", COUNTRY_US, IMPORTANCE_MEDIUM, (
        KeyIndicator("INDPRO", "工業生產月增率", "pch", "%", decimals=2),
    )),
    CalendarRelease(91, "美國密西根大學消費者信心", COUNTRY_US, IMPORTANCE_MEDIUM, (
        KeyIndicator("UMCSENT", "消費者信心指數", "lin", "", decimals=1),
    )),
    CalendarRelease(11, "美國就業成本指數（ECI）", COUNTRY_US, IMPORTANCE_MEDIUM, (
        KeyIndicator("ECIALLCIV", "ECI 年增率", "pc1", "%", decimals=2),
    )),
    CalendarRelease(27, "美國新屋開工", COUNTRY_US, IMPORTANCE_MEDIUM, (
        KeyIndicator("HOUST", "新屋開工（年化）", "lin", "千戶", decimals=0),
    )),
    CalendarRelease(97, "美國新屋銷售", COUNTRY_US, IMPORTANCE_MEDIUM, (
        KeyIndicator("HSN1F", "新屋銷售（年化）", "lin", "千戶", decimals=0),
    )),
    CalendarRelease(291, "美國成屋銷售", COUNTRY_US, IMPORTANCE_MEDIUM, (
        KeyIndicator("EXHOSLUSM495S", "成屋銷售（年化）", "lin", "萬戶", scale=0.0001, decimals=1),
    )),
    CalendarRelease(51, "美國國際貿易收支", COUNTRY_US, IMPORTANCE_MEDIUM, (
        KeyIndicator("BOPGSTB", "貿易收支", "lin", "億美元", scale=0.01, decimals=1),
    )),
    # ---------------- 國際
    CalendarRelease(251, "歐元區調和消費者物價指數（HICP）", COUNTRY_EU, IMPORTANCE_HIGH, (
        KeyIndicator("CP0000EZ19M086NEST", "歐元區 HICP 年增率", "pc1", "%", decimals=2),
    )),
    CalendarRelease(267, "歐元區國內生產毛額（Eurostat）", COUNTRY_EU, IMPORTANCE_MEDIUM, (
        KeyIndicator("CLVMNACSCAB1GQEA19", "歐元區實質 GDP 季增率", "pch", "%", decimals=2),
    )),
    CalendarRelease(269, "日本國民經濟計算（GDP）", COUNTRY_JP, IMPORTANCE_MEDIUM, (
        KeyIndicator("JPNRGDPEXP", "日本實質 GDP 季增率", "pch", "%", decimals=2),
    )),
    CalendarRelease(205, "OECD 主要經濟指標", COUNTRY_GLOBAL, IMPORTANCE_MEDIUM),
]}

# 每日更新、不代表會議日程的 release（見模組說明），任何範圍都不列入日曆
DAILY_POLICY_RATE_RELEASES = {101, 484}

# 「全部」模式下，期間內出現天數超過「週數 × 此值」的 release 視為每日更新資料而略過
DAILY_RELEASE_OCCURRENCES_PER_WEEK = 2

POLICY_RATES = [
    # (代碼, 名稱, 數列, 下限數列)
    ("FED", "美國聯邦資金利率目標區間", "DFEDTARU", "DFEDTARL"),
    ("ECB", "歐洲央行存款機制利率", "ECBDFR", None),
]
POLICY_RATE_HISTORY_LIMIT = 1000   # 用來找出上次調整日期的歷史筆數（日資料約 4 年）

SCOPE_IMPORTANT = "important"
SCOPE_ALL = "all"


def _format_indicator(client: FredClient, indicator: KeyIndicator) -> Dict:
    """取得指標最新兩期數值；FRED 以 "." 表示缺值。單一數列失敗時回傳 error，不影響其他指標。"""
    try:
        observations = client.get_observations(indicator.series_id, indicator.units, limit=2)
    except FredApiError as error:
        logger.warning("無法取得 %s：%s", indicator.series_id, error)
        return {"seriesId": indicator.series_id, "label": indicator.label, "unit": indicator.unit_label,
                "decimals": indicator.decimals, "latestPeriod": None, "latestValue": None,
                "previousPeriod": None, "previousValue": None, "error": "暫時無法取得"}
    values = []
    for observation in observations:
        raw = observation.get("value", ".")
        value = None if raw in (".", "") else round(float(raw) * indicator.scale, indicator.decimals)
        values.append((observation.get("date"), value))
    latest = values[0] if values else (None, None)
    previous = values[1] if len(values) > 1 else (None, None)
    return {
        "seriesId": indicator.series_id,
        "label": indicator.label,
        "unit": indicator.unit_label,
        "decimals": indicator.decimals,
        "latestPeriod": latest[0],
        "latestValue": latest[1],
        "previousPeriod": previous[0],
        "previousValue": previous[1],
        "error": None,
    }


def _is_daily_release(occurrences: int, start: date, end: date) -> bool:
    weeks = max(1.0, ((end - start).days + 1) / 7)
    return occurrences > weeks * DAILY_RELEASE_OCCURRENCES_PER_WEEK


def build_calendar(client: FredClient, start: date, end: date, scope: str = SCOPE_IMPORTANT) -> Dict:
    """建立期間內的經濟日曆。scope=important 只列重要指標；all 另外列出其他非每日更新的 release。

    只需一次（分頁）FRED 請求；各指標最新值由 get_release_indicators 另外查詢，
    讓頁面先顯示日曆，不必等待十多個數列逐一下載。
    """
    release_dates = client.get_release_dates(start.isoformat(), end.isoformat())

    occurrences: Dict[int, int] = {}
    for item in release_dates:
        occurrences[item["release_id"]] = occurrences.get(item["release_id"], 0) + 1

    today = date.today()
    events = []
    seen = set()
    for item in release_dates:
        release_id = item["release_id"]
        key = (release_id, item["date"])
        if key in seen or release_id in DAILY_POLICY_RATE_RELEASES:
            continue
        seen.add(key)

        definition = CALENDAR_RELEASES.get(release_id)
        if definition is None:
            if scope != SCOPE_ALL or _is_daily_release(occurrences[release_id], start, end):
                continue
        event_date = date.fromisoformat(item["date"])
        events.append({
            "date": item["date"],
            "releaseId": release_id,
            "name": definition.name_zh if definition else item.get("release_name", ""),
            "originalName": item.get("release_name", ""),
            "country": definition.country if definition else None,
            "importance": definition.importance if definition else None,
            "isPast": event_date < today,
            "isToday": event_date == today,
            "hasIndicators": bool(definition and definition.indicators),
            "sourceUrl": f"https://fred.stlouisfed.org/release?rid={release_id}",
        })

    return {
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "scope": scope,
        "events": events,
    }


def get_release_indicators(client: FredClient, release_ids: Iterable[int]) -> Dict[str, List[Dict]]:
    """平行查詢多個重要指標 release 的最新值，回傳 {release_id 字串: [指標...]}；未收錄的 id 會略過。"""
    targets = [(release_id, indicator)
               for release_id in dict.fromkeys(release_ids)
               if release_id in CALENDAR_RELEASES
               for indicator in CALENDAR_RELEASES[release_id].indicators]
    with ThreadPoolExecutor(max_workers=INDICATOR_FETCH_WORKERS) as executor:
        formatted = list(executor.map(lambda target: _format_indicator(client, target[1]), targets))

    result: Dict[str, List[Dict]] = {}
    for (release_id, _), indicator in zip(targets, formatted):
        result.setdefault(str(release_id), []).append(indicator)
    return result


def get_policy_rates(client: FredClient) -> List[Dict]:
    """目前政策利率與上次調整日期（由每日歷史數列中找出最近一次數值變動）。"""
    rates = []
    for code, name, series_id, lower_series_id in POLICY_RATES:
        history = client.get_observations(series_id, "lin", limit=POLICY_RATE_HISTORY_LIMIT)
        points = [(o["date"], float(o["value"])) for o in history if o.get("value") not in (".", "", None)]
        if not points:
            continue
        latest_date, latest_value = points[0]
        last_change: Optional[Dict] = None
        for (newer_date, newer_value), (_, older_value) in zip(points, points[1:]):
            if newer_value != older_value:
                last_change = {"date": newer_date, "from": older_value, "to": newer_value}
                break

        lower_value = None
        if lower_series_id:
            lower = client.get_observations(lower_series_id, "lin", limit=1)
            if lower and lower[0].get("value") not in (".", ""):
                lower_value = float(lower[0]["value"])

        rates.append({
            "code": code,
            "name": name,
            "seriesId": series_id,
            "asOf": latest_date,
            "value": latest_value,
            "lowerValue": lower_value,
            "lastChange": last_change,
            "sourceUrl": f"https://fred.stlouisfed.org/series/{series_id}",
        })
    return rates
