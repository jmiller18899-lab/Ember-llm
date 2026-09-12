"""Offline service-contract and fault tests; real calls are measured separately."""
from copy import deepcopy
from datetime import datetime, timezone
import io
import json
from urllib.error import HTTPError, URLError

import pytest

from tool_assistant.live_services import JsonClient, LiveServices, ServiceError
from tool_assistant.live_smoke import CASES, check
from tool_assistant.runtime_v5 import ParserV4Control

NOW = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
PLACE = {"id": 3133895, "name": "Tromsø", "latitude": 69.6489, "longitude": 18.95508,
         "timezone": "Europe/Oslo", "feature_code": "PPLA", "country": "Norway", "admin1": "Troms"}
WEATHER = {"latitude": 69.65, "longitude": 18.95,
    "current_units": {"time": "unixtime", "temperature_2m": "°C", "relative_humidity_2m": "%", "precipitation": "mm"},
    "current": {"time": NOW.timestamp()-300, "temperature_2m": 10.5, "relative_humidity_2m": 80,
                "precipitation": 0.2, "weather_code": 61}}
CLOCK = {"timezone": "Asia/Kathmandu", "date_time": "2026-09-12T17:45:00+05:45",
         "utc_offset_seconds": 20700, "dst_active": False}


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.events = []

    def get(self, service, params, **kwargs):
        self.events.append({"service": service, "http_status": 200})
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return deepcopy(value)


def services(*responses, search_key=""):
    return LiveServices(client=FakeClient(*responses), search_key=search_key, now=lambda: NOW)


def routed(tool):
    runtime = ParserV4Control.__new__(ParserV4Control)
    runtime.route = lambda _: (tool, 1.0)
    return runtime


def test_real_runtime_dispatches_weather_once_and_returns_units_and_provenance():
    live = services({"results": [PLACE, {**PLACE, "id": 2, "name": "Tromsø Airport", "feature_code": "AIRP"}]}, WEATHER)
    result = live.run(routed("weather"), 'What is the weather in "Tromsø, Norway" now?')
    assert result["status"] == "tool_result"
    assert len(result["service_calls"]) == 1
    assert result["result"]["location"]["id"] == 3133895
    assert result["result"]["current"]["temperature_2m"] == 10.5
    assert result["result"]["units"]["temperature_2m"] == "°C"
    assert result["result"]["data_kind"] == "current_weather_model_conditions"


def test_geocoding_never_picks_first_match_between_different_places():
    first = {**PLACE, "name": "London", "country": "United Kingdom", "admin1": "England"}
    second = {**PLACE, "id": 2, "name": "London", "country": "Canada", "admin1": "Ontario"}
    live = services({"results": [first, second]})
    result = live.run(routed("weather"), "What is the weather in London now?")
    assert result["status"] == "needs_clarification"
    assert result["service_error"]["code"] == "ambiguous_location"
    assert len(result["service_error"]["choices"]) == 2
    assert [event["service"] for event in live.client.events] == ["geocoding"]


def test_country_qualifier_filters_exact_city_and_decorated_names_do_not_match():
    rows = [{**PLACE, "id": 2, "country": "Canada"}, {**PLACE, "id": 3, "name": "Tromsøya"}, PLACE]
    assert services({"results": rows}).place("Tromsø, Norway")["id"] == 3133895


@pytest.mark.parametrize("payload", [{}, {"results": []}, {"results": [{**PLACE, "name": "Tromsø Airport", "feature_code": "AIRP"}]}])
def test_unknown_place_does_not_reach_weather(payload):
    live = services(payload)
    result = live.run(routed("weather"), "What is the weather in Qzxvplon now?")
    assert result["status"] == "needs_clarification"
    assert result["service_error"]["code"] == "location_not_found"
    assert len(live.client.events) == 1


@pytest.mark.parametrize("field,value", [("time", NOW.timestamp()-7200), ("time", NOW.timestamp()+7200),
    ("temperature_2m", None), ("temperature_2m", float("nan")), ("relative_humidity_2m", 101),
    ("precipitation", -1), ("weather_code", "rain")])
def test_invalid_or_stale_weather_cannot_be_a_success(field, value):
    payload = deepcopy(WEATHER)
    payload["current"][field] = value
    result = services({"results": [PLACE]}, payload).run(routed("weather"), "Weather in Tromsø now?")
    assert result["status"] == "tool_error"
    assert "result" not in result
    assert result["service_error"]["code"] in ("stale_response", "invalid_response")


def test_wrong_weather_units_are_rejected():
    payload = deepcopy(WEATHER)
    payload["current_units"]["temperature_2m"] = "°F"
    with pytest.raises(ServiceError):
        services({"results": [PLACE]}, payload).weather("Tromsø")


def test_clock_keeps_fractional_offset_and_fresh_external_timestamp():
    value = services(CLOCK).get_time("Asia/Kathmandu")
    assert value["datetime"].endswith("+05:45")
    assert value["clock_difference_seconds"] == 0
    assert value["provider"] == "TimeAPI.io"


@pytest.mark.parametrize("change", [
    {"date_time": "2026-09-11T17:45:00+05:45"}, {"date_time": "2026-09-12T17:45:00"},
    {"date_time": "2026-09-12T17:45:00+05:30"}, {"timezone": "Europe/Oslo"},
    {"utc_offset_seconds": 18000}, {"dst_active": True}, {"date_time": "bad timestamp"}])
def test_stale_or_inconsistent_clock_data_is_rejected(change):
    with pytest.raises(ServiceError):
        services({**CLOCK, **change}).get_time("Asia/Kathmandu")


def test_city_to_timezone_then_external_clock():
    clock = {"timezone": "Europe/Oslo", "date_time": "2026-09-12T14:00:00+02:00",
             "utc_offset_seconds": 7200, "dst_active": True}
    live = services({"results": [PLACE]}, clock)
    value = live.get_time("Tromsø, Norway")
    assert value["timezone"] == "Europe/Oslo"
    assert [e["service"] for e in live.client.events] == ["geocoding", "clock"]


def test_invalid_iana_zone_never_calls_a_provider():
    live = services()
    with pytest.raises(ServiceError, match="valid IANA"):
        live.get_time("Mars/Olympus")
    assert not live.client.events


def test_calculator_executes_bounded_arithmetic_locally():
    live = services()
    assert live.calculator("(144-24)/8")["value"] == 15
    assert live.calculator("(12.5/100)*640")["value"] == 80
    assert not live.client.events


@pytest.mark.parametrize("expression", ["1/0", "2**10000", "__import__('os')", "1e999"])
def test_calculator_errors_are_explicit(expression):
    with pytest.raises(ServiceError, match="supported limits"):
        services().calculator(expression)


def test_missing_search_key_is_unavailable_and_makes_no_request():
    live = services()
    result = live.run(routed("web_search"), "Search the web for Python documentation.")
    assert result["status"] == "tool_unavailable"
    assert result["service_error"]["code"] == "missing_credentials"
    assert not live.client.events


def test_search_returns_provider_urls_without_fetching_untrusted_pages():
    live = services({"web": {"results": [{"title": "Python", "url": "https://docs.python.org/3/"}]}}, search_key="fixture-key")
    assert live.web_search("Python documentation")["results"] == [{"title": "Python", "url": "https://docs.python.org/3/"}]
    assert len(live.client.events) == 1


@pytest.mark.parametrize("user,tool,status", [
    ("Weather in Oslo tomorrow?", "weather", "needs_clarification"),
    ("Explain rainbows.", "direct", "direct_answer_unavailable")])
def test_no_service_calls_for_clarification_or_direct(user, tool, status):
    live = services()
    result = live.run(routed(tool), user)
    assert result["status"] == status
    assert result["service_calls"] == []
    assert live.client.events == []


class Response(io.BytesIO):
    status = 200


class Opener:
    def __init__(self, value):
        self.value = value
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        if isinstance(self.value, Exception):
            raise self.value
        return Response(self.value)


@pytest.mark.parametrize("failure,code", [
    (TimeoutError("private transport details"), "timeout"),
    (URLError(TimeoutError()), "timeout"), (URLError("private transport details"), "network_error"),
    *[(HTTPError("https://example.invalid/", n, "private body", {}, None), f"http_{n}") for n in (401, 429, 503)]])
def test_network_faults_have_sanitized_messages_and_one_attempt(failure, code):
    opener = Opener(failure)
    client = JsonClient(opener=opener)
    with pytest.raises(ServiceError) as error:
        client.get("search", {"q": "test"}, headers={"X-Subscription-Token": "fixture-secret"})
    assert error.value.code == code
    assert "private" not in str(error.value)
    assert "fixture-secret" not in json.dumps(client.events)
    assert len(opener.requests) == 1


@pytest.mark.parametrize("raw", [b"<html>not JSON</html>", b"[]", b'{"error":true}', b"x"*1_048_577])
def test_invalid_or_oversize_http_responses_are_not_successes(raw):
    with pytest.raises(ServiceError):
        JsonClient(opener=Opener(raw)).get("weather", {})


def test_search_secret_only_goes_in_auth_header():
    opener = Opener(b'{"web":{"results":[]}}')
    client = JsonClient(opener=opener)
    live = LiveServices(client=client, search_key="fixture-secret")
    live.web_search("Python documentation")
    req = opener.requests[0]
    assert req.get_header("X-subscription-token") == "fixture-secret"
    assert "fixture-secret" not in req.full_url
    assert "fixture-secret" not in json.dumps(client.events)


def test_live_smoke_does_not_count_missing_search_as_passed():
    case = next(c for c in CASES if c["id"] == "web_search")
    output = {"route": "web_search", "status": "tool_unavailable", "service_calls": [],
              "service_error": {"code": "missing_credentials"}}
    assert check(case, output, [])[0] == "BLOCKED"


def test_live_smoke_rejects_fixture_output_without_real_http_evidence():
    case = next(c for c in CASES if c["id"] == "weather_unicode")
    output = {"route": "weather", "status": "tool_result", "service_calls": [{}],
              "result": {"provider": "Open-Meteo", "location": {"id": 3133895}}}
    assert check(case, output, [])[0] == "FAIL"
