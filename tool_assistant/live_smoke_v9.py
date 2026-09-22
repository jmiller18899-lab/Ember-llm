"""Retain live services and verify the contact repairs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from .evidence_v5 import emit_json
from .freeze_v9 import verify_freeze
from .live_services import OPEN_METEO_RETRY_DELAYS, OPEN_METEO_SERVICES, LiveServices
from .live_smoke import check
from .live_smoke_v8 import V8_CASES
from .runtime_v8 import RoutingV8Runtime
from .runtime_v9 import RoutingV9Runtime

REPAIRS = [
    {"id": "repaired_cousin_weather", "kind": "service", "route": "weather",
     "user": "I am contacting my cousin in Akureyri; is it cold there now?",
     "place_id": 2633274},
    {"id": "repaired_client_time", "kind": "service", "route": "get_time",
     "user": "We are phoning a client in Fes; what time is it there currently?",
     "timezone": "Africa/Casablanca", "network_services": ["geocoding", "clock"]},
]

V9_CASES = [*V8_CASES, *REPAIRS]


def measure(runtime, services):
    rows = []
    for case in V9_CASES:
        start = len(services.client.events)
        try:
            output = services.run(runtime, case["user"])
        except Exception as exc:
            output = {"status": "runtime_error", "error_type": type(exc).__name__}
        events = services.client.events[start:]
        outcome, checks = check(case, output, events)
        retries = sum(event.get("attempt", 1) > 1 for event in events)
        rows.append({**case, "outcome": outcome, "checks": checks, "output": output, "http": events,
                     "additional_http_attempts": retries, "first_attempt_outcome": "FAIL" if retries else outcome,
                     "recovered_after_retry": outcome == "PASS" and retries > 0})
        print(json.dumps({"event": "live_v9_case", "id": case["id"], "outcome": outcome,
                          "route": output.get("route"), "status": output.get("status"),
                          "service_error": output.get("service_error", {}).get("code"),
                          "additional_http_attempts": retries}), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("Refusing to overwrite live evidence")
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    freeze = verify_freeze(root, args.bundle)
    results, controls = {}, {}
    for precision in ("full", "int4"):
        original = RoutingV8Runtime(args.bundle, precision)
        controls[precision] = [{"id": c["id"], "user": c["user"], "output": original.plan(c["user"]),
                                "live_tools_tested": False} for c in REPAIRS]
        del original
        runtime = RoutingV9Runtime(args.bundle, precision)
        results[precision] = measure(runtime, LiveServices())
        del runtime
    if freeze != verify_freeze(root, args.bundle):
        raise RuntimeError("Frozen source or candidate changed during live measurement")
    rows = [r for precision in results.values() for r in precision]
    counts = {s: sum(r["outcome"] == s for r in rows) for s in ("PASS", "FAIL", "BLOCKED")}
    status = "FAIL" if counts["FAIL"] else "PARTIAL" if counts["BLOCKED"] else "PASS"
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "status": status, "counts": counts, "results": results, "freeze": freeze,
              "preserved_v8_control_plans": controls,
              "first_attempt_counts": {s: sum(r["first_attempt_outcome"] == s for r in rows) for s in counts},
              "retry_summary": {"cases_with_retries": sum(r["additional_http_attempts"] > 0 for r in rows),
                                "recovered_cases": sum(r["recovered_after_retry"] for r in rows),
                                "additional_http_attempts": sum(r["additional_http_attempts"] for r in rows)},
              "retry_policy": {"services": sorted(OPEN_METEO_SERVICES), "error_codes": ["timeout"],
                               "max_attempts": len(OPEN_METEO_RETRY_DELAYS) + 1,
                               "delays_seconds": list(OPEN_METEO_RETRY_DELAYS),
                               "socket_timeout_seconds": 12, "hard_wall_clock_deadline": False},
              "all_services_live_verified": status == "PASS", "production_ready": False,
              "base_model_trained": False, "direct_answer_quality_tested": False,
              "interpretation": "All 25 cases per checkpoint count, including every previous live case, the contact repairs. Direct questions verify no dispatch, not generated-answer quality. The unchanged v8 plans are recorded separately. Live service evidence is separate from routing confirmation and is not an uptime guarantee."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    emit_json(args.report, "live-services-v9.json")
    print(json.dumps({"event": "live_v9_summary", "status": status, "counts": counts,
                      "first_attempt_counts": report["first_attempt_counts"],
                      "retry_summary": report["retry_summary"]}), flush=True)
    if status != "PASS":
        raise SystemExit(1 if status == "FAIL" else 2)


if __name__ == "__main__":
    main()
