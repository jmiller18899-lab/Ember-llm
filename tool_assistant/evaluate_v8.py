"""Compare definition routing with the frozen v7 control; never retune while scoring."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from .evaluate import evaluate, validate_cases
from .evaluate_v7 import parser_regressions
from .evidence_v5 import emit_json
from .freeze_v8 import verify_confirmation, verify_freeze
from .runtime import sha256
from .runtime_v7 import RoutingV7Runtime
from .runtime_v8 import RoutingV8Runtime

DEVELOPMENT_SUITES = ("confirmation-100", "routing-v5-confirmation", "routing-v6-confirmation", "routing-v7-confirmation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--kind", choices=("development", "confirmation"), required=True)
    parser.add_argument("--suite", choices=DEVELOPMENT_SUITES)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("Refusing to overwrite measured evidence")
    if (args.kind == "development") != bool(args.suite):
        raise ValueError("Choose a consumed suite for development only")
    root = Path(__file__).resolve().parents[1]
    freeze = verify_freeze(root, args.bundle)
    name = args.suite or "routing-v8-confirmation"
    path = root / "tool_assistant/data" / (name + ".json")
    cases_digest = sha256(path)
    suite = json.loads(path.read_text())
    if args.kind == "confirmation":
        verify_confirmation(suite, root)
    else:
        validate_cases(suite["cases"], [])
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    results = {}
    for precision in ("full", "int4"):
        results[precision] = {}
        for arm, cls in (("frozen_v7", RoutingV7Runtime), ("definition_v8", RoutingV8Runtime)):
            runtime = cls(args.bundle, precision)
            result = evaluate(runtime, suite["cases"])
            results[precision][arm] = result
            del runtime
            print(json.dumps({"event": "routing_v8_measurement", "precision": precision, "arm": arm,
                              "suite": name, "scores": {k: v for k, v in result.items() if k != "cases"}}), flush=True)
    regressions = parser_regressions(root)
    if freeze != verify_freeze(root, args.bundle):
        raise RuntimeError("Source or candidate changed during scoring")
    if sha256(path) != cases_digest:
        raise RuntimeError("Suite changed during scoring")
    passed = all(r["definition_v8"]["combined_correct"] == 100 for r in results.values())
    passed = passed and regressions["passed"] == regressions["total"] == 60
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "kind": args.kind, "suite": name, "cases_sha256": cases_digest, "freeze": freeze,
              "status": "PASS" if passed else "FAIL", "strict_pass": passed, "results": results,
              "parser_regressions": regressions, "confirmation_consumed": args.kind == "confirmation",
              "base_model_trained": False, "routing_head_refitted": False, "uses_ember_features": False,
              "live_tools_tested": False, "direct_answer_quality_tested": False, "production_ready": False,
              "interpretation": "Paired helper routing, exact arguments, and one fixture dispatch. Direct cases score routing only. Both checkpoint integrations share one archived head and source overlay; equal scores are not independent LLM confirmations. New requests are assistant-authored finite contract coverage, not blind human testing."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    emit_json(args.report, args.report.name)
    print(json.dumps({"event": "routing_v8_summary", "suite": name, "kind": args.kind,
                      "status": report["status"], "strict_pass": passed}), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
