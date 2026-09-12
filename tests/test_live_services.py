"""Offline service-contract and fault tests; real calls are measured separately."""
from copy import deepcopy
from datetime import datetime, timezone
import io
import json
from urllib.error import HTTPError, URLError

import pytest

from tool_assistant.live_services import JsonClient, LiveServices, ServiceError
from tool_assistant.live_smoke import CASES, check, completed_http_requests
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


class SequenceOpener:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests, self.timeouts = [], []

    def open(self, request, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        if isinstance(value, dict):
            value = json.dumps(value).encode()
        return Response(value) if isinstance(value, bytes) else value


def retry_services(*responses):
    opener, sleeps = SequenceOpener(*responses), []
    client = JsonClient(opener=opener, sleep=sleeps.append)
    return LiveServices(client=client, search_key="", now=lambda: NOW), opener, sleeps


@pytest.mark.parametrize("service", ["weather", "geocoding"])
@pytest.mark.parametrize("failure", [TimeoutError("private details"), URLError(TimeoutError("private details"))])
def test_open_meteo_timeout_recovers_and_keeps_the_failed_attempt(service, failure):
    live, opener, sleeps = retry_services(failure, {"value": 1})
    assert live.client.get(service, {"name": "Tromsø, Norway"}) == {"value": 1}
    assert sleeps == [0.5]
    assert opener.timeouts == [12, 12]
    assert opener.requests[0].full_url == opener.requests[1].full_url
    first, second = live.client.events
    assert first["outcome"] == "error" and first["error_code"] == "timeout"
    assert first["phase"] == "open" and "http_status" not in first
    assert (first["request_id"], first["attempt"], second["request_id"], second["attempt"]) == (1, 1, 1, 2)
    assert second["outcome"] == "success" and second["http_status"] == 200
    assert "private" not in json.dumps(live.client.events)
    assert completed_http_requests(live.client.events) == [service]


def test_forecast_recovery_does_not_repeat_geocoding_or_tool_dispatch():
    live, opener, sleeps = retry_services({"results": [PLACE]}, TimeoutError(), TimeoutError(), WEATHER)
    case = next(c for c in CASES if c["id"] == "weather_unicode")
    output = live.run(routed("weather"), case["user"])
    assert output["status"] == "tool_result"
    assert output["result"]["current"]["temperature_2m"] == 10.5
    assert len(output["service_calls"]) == 1
    assert [e["service"] for e in live.client.events] == ["geocoding", "weather", "weather", "weather"]
    assert [e["attempt"] for e in live.client.events] == [1, 1, 2, 3]
    assert [e["request_id"] for e in live.client.events] == [1, 2, 2, 2]
    assert sleeps == [0.5, 1.0] and len(opener.requests) == 4
    assert check(case, output, live.client.events)[0] == "PASS"


def test_exhausted_forecast_timeouts_are_an_explicit_failure_after_three_attempts():
    live, opener, sleeps = retry_services({"results": [PLACE]}, TimeoutError(), TimeoutError(), TimeoutError())
    case = next(c for c in CASES if c["id"] == "weather_unicode")
    output = live.run(routed("weather"), case["user"])
    assert output["status"] == "tool_error" and output["service_error"]["code"] == "timeout"
    assert "result" not in output and len(output["service_calls"]) == 1
    assert len(opener.requests) == 4 and sleeps == [0.5, 1.0]
    assert all(e["outcome"] == "error" for e in live.client.events[1:])
    assert "retry_delay_seconds" not in live.client.events[-1]
    assert check(case, output, live.client.events)[0] == "FAIL"


def test_exhausted_geocoding_timeouts_never_reach_the_forecast():
    live, opener, sleeps = retry_services(TimeoutError(), TimeoutError(), TimeoutError())
    output = live.run(routed("weather"), "What is the weather in Tromsø now?")
    assert output["status"] == "tool_error" and "result" not in output
    assert [e["service"] for e in live.client.events] == ["geocoding"] * 3
    assert len(opener.requests) == 3 and sleeps == [0.5, 1.0]


class ReadTimeoutResponse(Response):
    def read(self, size=-1):
        raise TimeoutError("private read details")


def test_http_200_with_an_incomplete_body_is_a_failed_attempt_and_is_closed():
    interrupted = ReadTimeoutResponse()
    live, _, _ = retry_services(interrupted, {"results": []})
    live.client.get("geocoding", {"name": "Tromsø"})
    assert interrupted.closed
    first = live.client.events[0]
    assert first["http_status"] == 200 and first["outcome"] == "error"
    assert first["phase"] == "body" and first["error_code"] == "timeout"
    assert "body_sha256" not in first
    assert completed_http_requests(live.client.events) == ["geocoding"]


@pytest.mark.parametrize("service", ["geocoding", "weather"])
@pytest.mark.parametrize("failure,code", [
    *[(HTTPError("https://example.invalid/", n, "private body", {"Retry-After": "60"}, None), f"http_{n}") for n in (401, 429, 503)],
    (URLError("private transport details"), "network_error"),
    (b"invalid JSON", "invalid_response"),
])
def test_weather_retries_do_not_repeat_http_errors_or_invalid_responses(service, failure, code):
    live, opener, sleeps = retry_services(failure)
    with pytest.raises(ServiceError) as error:
        live.client.get(service, {})
    assert error.value.code == code
    assert len(opener.requests) == 1 and not sleeps
    assert live.client.events[0]["error_code"] == code
    assert "private" not in json.dumps(live.client.events)


def test_clock_timeout_keeps_its_original_single_attempt():
    live, opener, sleeps = retry_services(TimeoutError())
    with pytest.raises(ServiceError) as error:
        live.get_time("UTC")
    assert error.value.code == "timeout"
    assert len(opener.requests) == 1 and not sleeps


def test_recovered_geocoding_still_rejects_stale_weather():
    stale = deepcopy(WEATHER)
    stale["current"]["time"] = NOW.timestamp() - 7200
    live, opener, sleeps = retry_services(TimeoutError(), {"results": [PLACE]}, stale)
    case = next(c for c in CASES if c["id"] == "weather_unicode")
    output = live.run(routed("weather"), case["user"])
    assert output["status"] == "tool_error" and output["service_error"]["code"] == "stale_response"
    assert "result" not in output and len(opener.requests) == 3 and sleeps == [0.5]
    assert check(case, output, live.client.events)[0] == "FAIL"


def test_recovered_geocoding_still_clarifies_ambiguous_places_without_a_forecast():
    london = {**PLACE, "name": "London", "country": "United Kingdom"}
    other = {**london, "id": 2, "country": "Canada"}
    live, opener, _ = retry_services(TimeoutError(), {"results": [london, other]})
    case = next(c for c in CASES if c["id"] == "ambiguous_place")
    output = live.run(routed("weather"), case["user"])
    assert output["status"] == "needs_clarification"
    assert len(opener.requests) == 2
    assert check(case, output, live.client.events)[0] == "PASS"


@pytest.mark.parametrize("damage", ["missing_attempt", "permanent_error", "different_url", "reordered_request",
                                    "extra_request", "error_after_200", "missing_body_hash"])
def test_smoke_cannot_hide_failed_or_unrelated_attempts_behind_a_final_success(damage):
    live, _, _ = retry_services({"results": [PLACE]}, TimeoutError(), WEATHER)
    case = next(c for c in CASES if c["id"] == "weather_unicode")
    output = live.run(routed("weather"), case["user"])
    events = deepcopy(live.client.events)
    assert check(case, output, events)[0] == "PASS"
    if damage == "missing_attempt":
        del events[1]
    elif damage == "permanent_error":
        events[1]["error_code"] = "http_401"
    elif damage == "different_url":
        events[-1]["url"] += "&latitude=0"
    elif damage == "reordered_request":
        events[-1]["request_id"] = events[0]["request_id"]
    elif damage == "extra_request":
        events.append({**events[-1], "request_id": 3, "attempt": 1})
    elif damage == "error_after_200":
        events[-1]["error_code"] = "timeout"
    elif damage == "missing_body_hash":
        del events[-1]["body_sha256"]
    assert check(case, output, events)[0] == "FAIL"
