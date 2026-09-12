"""Paired v5/v6 scoring against frozen bytes, without fitting or live calls."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from .evaluate import evaluate, validate_cases
from .evaluate_v5 import parser_regressions
from .evidence_v5 import emit_json
from .freeze_v6 import verify_confirmation, verify_freeze
from .runtime import sha256
from .runtime_v5 import RoutingV5Runtime
from .runtime_v6 import RoutingV6Runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--kind", choices=("development", "confirmation"), required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("Refusing to overwrite measured evidence")
    root = Path(__file__).resolve().parents[1]
    freeze = verify_freeze(root, args.bundle)
    names = (("confirmation-100", "routing-v5-confirmation") if args.kind == "development"
             else ("routing-v6-confirmation",))
    suites, digests = {}, {}
    for name in names:
        path = root / "tool_assistant/data" / (name + ".json")
        suite = json.loads(path.read_text())
        if args.kind == "confirmation":
            verify_confirmation(suite, root)
        else:
            validate_cases(suite["cases"], [])
        suites[name], digests[name] = suite["cases"], sha256(path)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    results = {}
    for precision in ("full", "int4"):
        results[precision] = {}
        for arm, cls in (("frozen_v5", RoutingV5Runtime), ("context_v6", RoutingV6Runtime)):
            runtime = cls(args.bundle, precision)
            results[precision][arm] = {}
            for name, cases in suites.items():
                result = evaluate(runtime, cases)
                results[precision][arm][name] = result
                print(json.dumps({"event": "routing_v6_measurement", "precision": precision,
                                  "arm": arm, "suite": name,
                                  "scores": {k: v for k, v in result.items() if k != "cases"}}), flush=True)
            del runtime
    regressions = parser_regressions(root)
    if freeze != verify_freeze(root, args.bundle):
        raise RuntimeError("Candidate or overlay changed during scoring")
    passed = all(r["combined_correct"] == 100 for p in results.values() for r in p["context_v6"].values())
    passed = passed and regressions["passed"] == regressions["total"] == 60
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "kind": args.kind, "status": "PASS" if passed else "FAIL", "strict_pass": passed,
              "results": results, "parser_regressions": regressions, "freeze": freeze,
              "cases_sha256": digests, "confirmation_consumed": args.kind == "confirmation",
              "base_model_trained": False, "routing_head_refitted": False, "uses_ember_features": False,
              "live_tools_tested": False, "direct_answer_quality_tested": False, "production_ready": False,
              "interpretation": "Paired helper routing, exact arguments, and one fixture dispatch. Direct cases score routing only. Both checkpoints share one frozen text head and deterministic overlay; equal scores are not independent LLM confirmations. New requests are assistant-authored finite contract coverage, not blind human evaluation."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    emit_json(args.report, f"routing-v6-{args.kind}.json")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
