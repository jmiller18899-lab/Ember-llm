"""Measure real service calls through the unchanged frozen v5 helper."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

import torch

from .evaluate_v5 import SOURCE_FILES
from .evidence_v5 import emit_json
from .live_services import ENDPOINTS, OPEN_METEO_RETRY_DELAYS, OPEN_METEO_SERVICES, LiveServices
from .runtime import sha256
from .runtime_v5 import RoutingV5Runtime, TextHelperRuntime

CASES = [
    {"id": "weather_unicode", "kind": "service", "route": "weather",
     "user": 'What is the current weather in "Tromsø, Norway"?', "place_id": 3133895},
    {"id": "weather_context", "kind": "service", "route": "weather",
     "user": 'I am stepping outside; what is the current temperature in "Reykjavík, Iceland"?', "place_id": 3413829},
    {"id": "time_utc", "kind": "service", "route": "get_time",
     "user": "What time is it in UTC now?", "timezone": "UTC"},
    {"id": "time_fractional_offset", "kind": "service", "route": "get_time",
     "user": "What time is it in Asia/Kathmandu right now?", "timezone": "Asia/Kathmandu"},
    {"id": "time_geocoded", "kind": "service", "route": "get_time",
     "user": 'What time is it in "Reykjavík, Iceland" now?', "timezone": "Atlantic/Reykjavik",
     "network_services": ["geocoding", "clock"]},
    {"id": "calculator_parentheses", "kind": "service", "route": "calculator",
     "user": "Calculate (144 - 24) / 8.", "value": 15},
    {"id": "calculator_percentage", "kind": "service", "route": "calculator",
     "user": "What is 12.5 percent of 640?", "value": 80},
    {"id": "web_search", "kind": "service", "route": "web_search",
     "user": "Search the web for the official Python documentation."},
    {"id": "ambiguous_place", "kind": "guard", "route": "weather",
     "user": "What is the weather in London now?", "status": "needs_clarification",
     "error": "ambiguous_location", "network_services": ["geocoding"], "dispatches": 1},
    {"id": "unknown_place", "kind": "guard", "route": "weather",
     "user": "What is the weather in Qzxvplon now?", "status": "needs_clarification",
     "error": "location_not_found", "network_services": ["geocoding"], "dispatches": 1},
    {"id": "future_weather", "kind": "guard", "route": "weather",
     "user": "What will the weather in Oslo be tomorrow?", "status": "needs_clarification",
     "error": "unsupported_time", "network_services": [], "dispatches": 0},
    {"id": "direct_no_dispatch", "kind": "guard", "route": "direct",
     "user": "Explain why rainbows form.", "status": "direct_answer_unavailable",
     "network_services": [], "dispatches": 0},
]
KNOWN = {"id": "preserved_v5_routing_failure", "kind": "known_routing_failure", "route": "weather", "place_id": 3133895,
         "user": "I am contacting a friend in Tromsø; is it cold there currently?"}


def verify_freeze(root, *, bundle=None, head=None):
    lock_path = root / "tool_assistant/data/routing-v5-source-lock.json"
    lock = json.loads(lock_path.read_text())
    if set(lock["files"]) != set(SOURCE_FILES):
        raise ValueError("Incomplete routing source freeze")
    for name, digest in lock["files"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Frozen routing source changed: {name}")
    result = {"source_lock_sha256": sha256(lock_path), "source_files_verified": len(lock["files"])}
    if bundle is not None:
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["files"] != lock["candidate_files"]:
            raise ValueError("The candidate differs from the v5 freeze")
        for name, digest in lock["candidate_files"].items():
            path = (bundle / name).resolve()
            if not path.is_relative_to(bundle.resolve()) or sha256(path) != digest:
                raise ValueError(f"Frozen candidate changed: {name}")
        result.update(candidate_manifest_sha256=sha256(manifest_path),
                      candidate_files_verified=len(manifest["files"]))
    else:
        if sha256(head) != lock["candidate_files"]["routing-v5.pt"]:
            raise ValueError("The text helper does not match the frozen fitted file")
        result.update(routing_head_sha256=sha256(head), candidate_files_verified=1)
    return result


def completed_http_requests(events):
    """Validate every attempt before treating a recovered GET as complete."""
    groups = []
    for event in events:
        request_id = event.get("request_id")
        if type(request_id) is not int or request_id < 1:
            return None
        if not groups or request_id != groups[-1][0]["request_id"]:
            if groups and request_id <= groups[-1][0]["request_id"]:
                return None
            groups.append([])
        groups[-1].append(event)
    completed = []
    for group in groups:
        first, last = group[0], group[-1]
        service, url = first.get("service"), first.get("url")
        if service not in ENDPOINTS or not isinstance(url, str) or not url.startswith(ENDPOINTS[service] + "?"):
            return None
        limit = len(OPEN_METEO_RETRY_DELAYS) + 1 if service in OPEN_METEO_SERVICES else 1
        if len(group) > limit:
            return None
        for attempt, event in enumerate(group, 1):
            if (type(event.get("attempt")) is not int or event["attempt"] != attempt
                    or event.get("service") != service or event.get("url") != url):
                return None
            if event is not last and (event.get("outcome") != "error" or event.get("error_code") != "timeout"
                    or event.get("http_status") not in (None, 200)
                    or event.get("phase") not in ("open", "body")
                    or event.get("retry_delay_seconds") != OPEN_METEO_RETRY_DELAYS[attempt - 1]):
                return None
        if (last.get("outcome") != "success" or last.get("http_status") != 200 or "error_code" in last
                or last.get("phase") != "complete" or "retry_delay_seconds" in last
                or not re.fullmatch(r"[0-9a-f]{64}", str(last.get("body_sha256", "")))):
            return None
        completed.append(service)
    return completed


def check(case, output, events):
    checks = {"expected_route": output.get("route") == case["route"]}
    calls = output.get("service_calls", [])
    completed = completed_http_requests(events)
    checks["http_trace_valid"] = completed is not None
    if case["kind"] == "guard":
        checks.update(expected_status=output.get("status") == case["status"],
                      expected_dispatch_count=len(calls) == case["dispatches"],
                      expected_network_calls=completed == case["network_services"])
        if "error" in case:
            checks["expected_reason"] = output.get("service_error", {}).get("code", output.get("reason")) == case["error"]
        return "PASS" if all(checks.values()) else "FAIL", checks
    if checks["expected_route"] and output.get("service_error", {}).get("code") == "missing_credentials":
        return "BLOCKED", {**checks, "live_search_executed": False}
    checks.update(tool_result=output.get("status") == "tool_result", exactly_one_dispatch=len(calls) == 1)
    result = output.get("result", {})
    if case["route"] == "weather":
        checks.update(provider=result.get("provider") == "Open-Meteo",
            requested_place=result.get("location", {}).get("id") == case.get("place_id"),
            live_http=completed == ["geocoding", "weather"])
    elif case["route"] == "get_time":
        checks.update(provider=result.get("provider") == "TimeAPI.io", requested_timezone=result.get("timezone") == case["timezone"],
            live_http=completed == case.get("network_services", ["clock"]))
    elif case["route"] == "calculator":
        value = result.get("value")
        checks.update(correct_value=type(value) in (int, float) and math.isclose(value, case["value"], abs_tol=1e-9),
                      executes_locally=not events and result.get("provider") == "local_bounded_arithmetic")
    elif case["route"] == "web_search":
        checks.update(provider=result.get("provider") == "Brave Search", nonempty_results=bool(result.get("results")),
            live_http=completed == ["search"])
    return "PASS" if all(checks.values()) else "FAIL", checks


def measure(runtime, services):
    rows = []
    for case in CASES + [KNOWN]:
        start = len(services.client.events)
        try:
            output = services.run(runtime, case["user"])
        except Exception as exc:
            output = {"status": "runtime_error", "error_type": type(exc).__name__}
        events = services.client.events[start:]
        status, checks = check(case, output, events)
        retries = sum(event.get("attempt", 1) > 1 for event in events)
        row = {**case, "outcome": status, "checks": checks, "output": output, "http": events,
               "additional_http_attempts": retries, "first_attempt_outcome": "FAIL" if retries else status,
               "recovered_after_retry": status == "PASS" and retries > 0}
        rows.append(row)
        print(json.dumps({"event": "live_smoke_case", "id": case["id"], "outcome": status,
            "route": output.get("route"), "status": output.get("status"),
            "service_error": output.get("service_error", {}).get("code"),
            "additional_http_attempts": retries, "recovered_after_retry": row["recovered_after_retry"]}), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bundle", type=Path)
    mode.add_argument("--head", type=Path, help="Local preflight only; excludes Ember tokenizer/checkpoint integration")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("Refusing to overwrite live evidence")
    root = Path(__file__).resolve().parent.parent
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    freeze = verify_freeze(root, bundle=args.bundle, head=args.head)
    results = {}
    precisions = ("full", "int4") if args.bundle else ("text_helper_preflight",)
    search_configured = False
    for precision in precisions:
        runtime = (RoutingV5Runtime(args.bundle, precision) if args.bundle else
                   TextHelperRuntime(torch.load(args.head, map_location="cpu", weights_only=True)))
        services = LiveServices()
        search_configured = bool(services.search_key)
        results[precision] = measure(runtime, services)
        del runtime
    if freeze != verify_freeze(root, bundle=args.bundle, head=args.head):
        raise RuntimeError("The frozen candidate changed during live testing")
    measured = [r for rows in results.values() for r in rows if r["kind"] != "known_routing_failure"]
    counts = {status: sum(r["outcome"] == status for r in measured) for status in ("PASS", "FAIL", "BLOCKED")}
    status = "FAIL" if counts["FAIL"] else "PARTIAL" if counts["BLOCKED"] else "PASS"
    report = {"schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "frozen_bundle_live_service_smoke" if args.bundle else "verified_text_helper_live_preflight",
        "status": status, "counts": counts, "results": results, "freeze": freeze,
        "first_attempt_counts": {status: sum(r["first_attempt_outcome"] == status for r in measured)
                                 for status in ("PASS", "FAIL", "BLOCKED")},
        "retry_summary": {"cases_with_retries": sum(r["additional_http_attempts"] > 0 for r in measured),
                          "recovered_cases": sum(r["recovered_after_retry"] for r in measured),
                          "additional_http_attempts": sum(r["additional_http_attempts"] for r in measured)},
        "retry_policy": {"services": sorted(OPEN_METEO_SERVICES), "error_codes": ["timeout"],
                         "max_attempts": len(OPEN_METEO_RETRY_DELAYS) + 1,
                         "delays_seconds": list(OPEN_METEO_RETRY_DELAYS),
                         "socket_timeout_seconds": 12, "hard_wall_clock_deadline": False},
        "search_key_configured": search_configured,
        "all_services_live_verified": status == "PASS", "production_ready": False,
        "base_model_trained": False, "direct_answer_quality_tested": False,
        "routing_confirmation_changed": False, "routing_confirmation_status": "FAIL",
        "known_routing_failures": sum(r["outcome"] != "PASS" for rows in results.values() for r in rows if r["kind"] == "known_routing_failure"),
        "source_sha256": {name: sha256(root / name) for name in (
            "tool_assistant/live_services.py", "tool_assistant/live_smoke.py", "tests/test_live_services.py",
            "tool_assistant/restore_live_candidate.py", "tests/test_restore_live_candidate.py",
            ".github/workflows/ember-live-services.yml", ".github/workflows/ember-live-services-run.yml")},
        "interpretation": "Live HTTP and local calculator execution through the frozen helper. Outcomes allow bounded, fully recorded Open-Meteo timeout retries; first-attempt outcomes remain separate. Deliberate error injection is covered by offline tests. This smoke suite is not a new routing confirmation or uptime guarantee."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    emit_json(args.report, "live-services-smoke.json")
    print(json.dumps({"event": "live_smoke_summary", "status": status, "counts": counts,
                     "first_attempt_counts": report["first_attempt_counts"], "retry_summary": report["retry_summary"],
                     "known_routing_failures": report["known_routing_failures"]}), flush=True)
    if status != "PASS":
        raise SystemExit(1 if status == "FAIL" else 2)


if __name__ == "__main__":
    main()
