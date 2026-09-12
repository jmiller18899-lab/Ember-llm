"""Opt-in service implementations for the existing frozen tool helper."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .evaluate import arithmetic

ENDPOINTS = {
    "geocoding": "https://geocoding-api.open-meteo.com/v1/search",
    "weather": "https://api.open-meteo.com/v1/forecast",
    "clock": "https://timeapi.io/api/v1/time/current/zone",
    "search": "https://api.search.brave.com/res/v1/web/search",
}


class ServiceError(Exception):
    def __init__(self, code, message, *, choices=None):
        super().__init__(message)
        self.code, self.message, self.choices = code, message, choices

    def as_dict(self):
        result = {"code": self.code, "message": self.message}
        if self.choices is not None:
            result["choices"] = self.choices
        return result


def require(condition, message="The service returned incomplete or invalid data."):
    if not condition:
        raise ServiceError("invalid_response", message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class JsonClient:
    """Bounded HTTPS GETs; the evidence log never includes authentication headers."""
    def __init__(self, *, timeout=12, opener=None):
        self.timeout = timeout
        self.opener = opener or build_opener(NoRedirect())
        self.events = []

    def get(self, service, params, *, headers=None):
        url = ENDPOINTS[service] + "?" + urlencode(params)
        request = Request(url, headers={"Accept": "application/json",
            "User-Agent": "Ember-llm-live-smoke/1.0", "Cache-Control": "no-cache", **(headers or {})})
        event = {"service": service, "url": url,
                 "started_at": datetime.now(timezone.utc).isoformat()}
        started = time.monotonic()
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                event["http_status"] = response.status
                require(response.status == 200)
                raw = response.read(1_048_577)
            require(len(raw) <= 1_048_576, "The service response exceeded the size limit.")
            event["body_sha256"] = hashlib.sha256(raw).hexdigest()
            try:
                data = json.loads(raw)
            except (ValueError, UnicodeError) as exc:
                raise ServiceError("invalid_response", "The service returned invalid JSON.") from exc
            require(isinstance(data, dict) and not data.get("error"))
            return data
        except HTTPError as exc:
            event["http_status"] = exc.code
            raise ServiceError(f"http_{exc.code}", f"The {service} service returned HTTP {exc.code}.") from None
        except TimeoutError:
            raise ServiceError("timeout", f"The {service} service timed out.") from None
        except URLError as exc:
            code = "timeout" if isinstance(exc.reason, TimeoutError) else "network_error"
            raise ServiceError(code, f"The {service} service could not be reached.") from None
        finally:
            event["elapsed_ms"] = round((time.monotonic() - started) * 1000, 2)
            self.events.append(event)


def name_key(text):
    return " ".join("".join(c for c in unicodedata.normalize("NFKD", text.casefold())
                            if not unicodedata.combining(c)).split())


class LiveServices:
    def __init__(self, *, client=None, search_key=None, now=None):
        self.client = client or JsonClient()
        self.search_key = os.environ.get("BRAVE_SEARCH_API_KEY", "") if search_key is None else search_key
        self.now = now or (lambda: datetime.now(timezone.utc))

    def place(self, location):
        if not isinstance(location, str) or not location.strip() or location.count(",") > 1:
            raise ServiceError("location_not_found", "Please give a city, optionally followed by its country or full region name.")
        parts = [p.strip() for p in location.split(",")]
        require(all(parts), "The location contains an empty name or qualifier.")
        payload = self.client.get("geocoding", {"name": location, "count": 100,
                                               "language": "en", "format": "json"})
        rows = payload.get("results", [])
        require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows))
        matches = {}
        for row in rows:
            if not isinstance(row.get("name"), str) or not isinstance(row.get("feature_code"), str):
                continue
            if name_key(row["name"]) != name_key(parts[0]) or not row["feature_code"].startswith("PPL"):
                continue
            if len(parts) == 2 and name_key(parts[1]) not in {
                name_key(str(row.get(k, ""))) for k in ("country", "country_code", "admin1")
            }:
                continue
            require(type(row.get("id")) is int)
            matches[row["id"]] = row
        if not matches:
            raise ServiceError("location_not_found", "I could not resolve that city exactly. Please give its full name and country.")
        if len(matches) != 1 or len(rows) >= 100:
            choices = [", ".join(dict.fromkeys(str(row[k]) for k in ("name", "admin1", "country") if row.get(k)))
                       for row in list(matches.values())[:5]]
            raise ServiceError("ambiguous_location", "Several places match. Please include the country or full region name.", choices=choices)
        row = next(iter(matches.values()))
        require(finite(row.get("latitude")) and -90 <= row["latitude"] <= 90)
        require(finite(row.get("longitude")) and -180 <= row["longitude"] <= 180)
        require(isinstance(row.get("timezone"), str))
        try:
            ZoneInfo(row["timezone"])
        except (ValueError, ZoneInfoNotFoundError):
            raise ServiceError("invalid_response", "The location service returned an unknown time zone.") from None
        return {key: row[key] for key in ("id", "name", "latitude", "longitude", "timezone", "country", "admin1") if key in row}

    def weather(self, location):
        place = self.place(location)
        data = self.client.get("weather", {"latitude": place["latitude"], "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
            "timezone": "UTC", "timeformat": "unixtime", "forecast_days": 1})
        current, units = data.get("current"), data.get("current_units")
        require(isinstance(current, dict) and isinstance(units, dict))
        require(finite(data.get("latitude")) and abs(data["latitude"]-place["latitude"]) < 1)
        require(finite(data.get("longitude")) and abs(data["longitude"]-place["longitude"]) < 1)
        require(units.get("time") == "unixtime" and units.get("temperature_2m") == "°C"
                and units.get("relative_humidity_2m") == "%" and units.get("precipitation") == "mm")
        require(finite(current.get("time")))
        age = self.now().timestamp() - current["time"]
        if not -300 <= age <= 3600:
            raise ServiceError("stale_response", "The weather service did not return current conditions.")
        require(finite(current.get("temperature_2m")) and -100 <= current["temperature_2m"] <= 70)
        require(finite(current.get("relative_humidity_2m")) and 0 <= current["relative_humidity_2m"] <= 100)
        require(finite(current.get("precipitation")) and current["precipitation"] >= 0)
        require(type(current.get("weather_code")) is int and 0 <= current["weather_code"] <= 99)
        return {"provider": "Open-Meteo", "attribution_url": "https://open-meteo.com/",
            "data_kind": "current_weather_model_conditions", "location": place,
            "valid_at": datetime.fromtimestamp(current["time"], timezone.utc).isoformat(),
            "fetched_at": self.now().isoformat(), "age_seconds": round(age, 2),
            "current": current, "units": units}

    def get_time(self, timezone):
        location = None
        try:
            zone = ZoneInfo(timezone)
        except (ValueError, ZoneInfoNotFoundError):
            if "/" in timezone:
                raise ServiceError("invalid_timezone", "Please give a valid IANA time zone or a city and country.") from None
            location = self.place(timezone)
            zone = ZoneInfo(location["timezone"])
        data = self.client.get("clock", {"timezone": zone.key})
        require(data.get("timezone") == zone.key and isinstance(data.get("date_time"), str))
        try:
            local = datetime.fromisoformat(data["date_time"])
            require(local.tzinfo is not None)
            expected = local.astimezone(zone)
            require(local.utcoffset() == expected.utcoffset())
            require(data.get("utc_offset_seconds") == expected.utcoffset().total_seconds())
            require(type(data.get("dst_active")) is bool and data["dst_active"] == bool(expected.dst()))
        except (ValueError, OverflowError):
            raise ServiceError("invalid_response", "The clock service returned an invalid timestamp.") from None
        age = self.now().timestamp() - local.timestamp()
        if abs(age) > 120:
            raise ServiceError("stale_response", "The clock service returned a stale or incorrect time.")
        return {"provider": "TimeAPI.io", "attribution_url": "https://timeapi.io/",
            "timezone": zone.key, "datetime": local.isoformat(), "location": location,
            "utc_offset_seconds": data["utc_offset_seconds"], "dst_active": data["dst_active"],
            "fetched_at": self.now().isoformat(), "clock_difference_seconds": round(age, 3)}

    def calculator(self, expression):
        try:
            value = arithmetic(expression)
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError):
            raise ServiceError("invalid_expression", "The expression could not be calculated within the supported limits.") from None
        return {"provider": "local_bounded_arithmetic", "expression": expression, "value": value}

    def web_search(self, query):
        if not self.search_key:
            raise ServiceError("missing_credentials", "Web search needs BRAVE_SEARCH_API_KEY to be configured.")
        require(isinstance(query, str) and 0 < len(query) <= 4000)
        data = self.client.get("search", {"q": query, "count": 3},
                               headers={"X-Subscription-Token": self.search_key})
        web = data.get("web", {})
        require(isinstance(web, dict))
        rows = web.get("results", [])
        require(isinstance(rows, list))
        results = []
        for row in rows[:3]:
            require(isinstance(row, dict) and isinstance(row.get("title"), str) and isinstance(row.get("url"), str))
            require(urlsplit(row["url"]).scheme in ("http", "https") and bool(urlsplit(row["url"]).netloc))
            results.append({"title": row["title"], "url": row["url"]})
        return {"provider": "Brave Search", "attribution_url": "https://search.brave.com/",
                "query": query, "results": results, "fetched_at": self.now().isoformat()}

    def run(self, runtime, user):
        """Use the frozen runtime's routing, parsing, and one-call dispatch unchanged."""
        errors, calls = [], []
        def wrapped(name):
            def handler(**kwargs):
                calls.append({"name": name, "arguments": kwargs})
                try:
                    return getattr(self, name)(**kwargs)
                except ServiceError as exc:
                    errors.append(exc)
                    raise
            return handler
        output = runtime.run(user, {name: wrapped(name) for name in ("weather", "get_time", "calculator", "web_search")})
        output = {**output, "service_calls": calls}
        if errors:
            error = errors[-1]
            status = "tool_error"
            if error.code == "missing_credentials":
                status = "tool_unavailable"
            elif error.code in ("ambiguous_location", "location_not_found", "invalid_timezone"):
                status = "needs_clarification"
            output.update(status=status, service_error=error.as_dict(), message=error.message)
        return output
