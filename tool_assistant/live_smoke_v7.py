"""Real services and no-dispatch checks for all six repaired v6 failures."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from .evidence_v5 import emit_json
from .freeze_v7 import verify_freeze
from .live_services import OPEN_METEO_RETRY_DELAYS, OPEN_METEO_SERVICES, LiveServices
from .live_smoke import check
from .live_smoke_v6 import V6_CASES
from .runtime_v6 import RoutingV6Runtime
from .runtime_v7 import RoutingV7Runtime

REPAIRS = [
    {"id": "repaired_word_powers", "kind": "service", "route": "calculator",
     "user": "Calculate 3 squared plus 4 squared.", "value": 25},
    {"id": "repaired_explicit_search", "kind": "service", "route": "web_search",
     "user": "Search the web for this week’s events at the Barbican Centre."},
    *({"id": "repaired_explanation_" + str(index), "kind": "guard", "route": "direct",
       "user": user, "status": "direct_answer_unavailable", "network_services": [], "dispatches": 0}
      for index, user in enumerate(("Why do some leaves change colour in autumn?",
                                   "How do roots help a plant absorb water?",
                                   "How does insulation reduce heat loss?",
                                   "What is the role of yeast in bread dough?"), 1)),
]
V7_CASES = [*V6_CASES, *REPAIRS]


def measure(runtime, services):
    rows = []
    for case in V7_CASES:
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
        print(json.dumps({"event": "live_v7_case", "id": case["id"], "outcome": outcome,
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
        original = RoutingV6Runtime(args.bundle, precision)
        controls[precision] = [{"id": c["id"], "user": c["user"], "output": original.plan(c["user"]),
                                "live_tools_tested": False} for c in REPAIRS]
        del original
        runtime = RoutingV7Runtime(args.bundle, precision)
        results[precision] = measure(runtime, LiveServices())
        del runtime
    if freeze != verify_freeze(root, args.bundle):
        raise RuntimeError("Frozen source or candidate changed during live measurement")
    rows = [r for precision in results.values() for r in precision]
    counts = {s: sum(r["outcome"] == s for r in rows) for s in ("PASS", "FAIL", "BLOCKED")}
    status = "FAIL" if counts["FAIL"] else "PARTIAL" if counts["BLOCKED"] else "PASS"
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "status": status, "counts": counts, "results": results, "freeze": freeze,
              "preserved_v6_control_plans": controls,
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
              "interpretation": "All 20 cases per checkpoint count, including every repaired v6 failure. Direct questions verify no dispatch, not generated-answer quality. The unchanged v6 plans are recorded separately. Live service evidence is separate from routing confirmation and is not an uptime guarantee."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    emit_json(args.report, "live-services-v7.json")
    print(json.dumps({"event": "live_v7_summary", "status": status, "counts": counts,
                      "first_attempt_counts": report["first_attempt_counts"],
                      "retry_summary": report["retry_summary"]}), flush=True)
    if status != "PASS":
        raise SystemExit(1 if status == "FAIL" else 2)


if __name__ == "__main__":
    main()
