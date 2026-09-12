"""Measure routing v4 with the unchanged exact-argument and dispatch scoring."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from .evaluate import case_hash, evaluate, validate_cases
from .routing_data_v4 import load_training
from .runtime import Runtime, sha256
from .runtime_v3 import ParserRuntime
from .runtime_v4 import RoutingRuntime

SOURCE_FILES = (
    "ember-v0.0.7-hf-ready.zip",
    *(f"tool_assistant/{name}.py" for name in (
        "build", "build_v4", "binary", "family", "runtime", "runtime_v3", "runtime_v4",
        "resolver", "resolver_v3", "routing_data_v4", "routing_v4", "evaluate", "evaluate_v4")),
    *(f"tool_assistant/data/{name}.json" for name in (
        "router-training", "prior-string-hashes", "confirmation-100", "parser-v3-confirmation")),
    "tests/test_routing_v4.py",
)
REQUIRED_CANDIDATE_FILES = {
    "model-full.pt", "model-int4.pt", "router-full.pt", "router-int4.pt", "family.pt",
    "routing-v4-full.pt", "routing-v4-int4.pt", "routing-development.json",
}


def verify_confirmation(suite, root, lock_path, bundle):
    lock = json.loads(lock_path.read_text())
    if suite.get("source_lock_sha256") != sha256(lock_path):
        raise ValueError("Confirmation was not bound to the frozen routing source")
    if set(lock.get("files", {})) != set(SOURCE_FILES):
        raise ValueError("Incomplete frozen routing source lock")
    for name, digest in lock["files"].items():
        target = (root / name).resolve()
        if not target.is_relative_to(root.resolve()) or sha256(target) != digest:
            raise ValueError(f"Frozen source changed: {name}")
    candidate_files = lock.get("candidate_files", {})
    if not REQUIRED_CANDIDATE_FILES.issubset(candidate_files):
        raise ValueError("The fitted candidate must be frozen before confirmation")
    manifest = json.loads((bundle / "manifest.json").read_text())
    if candidate_files != manifest["files"]:
        raise ValueError("Candidate differs from the pre-confirmation freeze")
    for name, digest in candidate_files.items():
        target = (bundle / name).resolve()
        if not target.is_relative_to(bundle.resolve()) or sha256(target) != digest:
            raise ValueError(f"Frozen candidate changed: {name}")
    historical, development = load_training(root)
    seen = set(json.loads((root / "tool_assistant/data/prior-string-hashes.json").read_text()))
    for filename in ("confirmation-100.json", "parser-v3-confirmation.json"):
        prior = json.loads((root / "tool_assistant/data" / filename).read_text())
        seen.update(case_hash(c["user"]) for c in prior["cases"])
    seen.update(case_hash(c["user"]) for c in historical + development)
    validate_cases(suite["cases"], seen)
    return lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--kind", choices=("development", "confirmation"), required=True)
    parser.add_argument("--source-lock", type=Path,
                        default=Path("tool_assistant/data/routing-v4-source-lock.json"))
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("Refusing to overwrite measured evidence")
    root = Path(__file__).resolve().parent.parent
    suite = json.loads(args.cases.read_text())
    cases = suite["cases"]
    if args.kind == "confirmation":
        verify_confirmation(suite, root, args.source_lock, args.bundle)
    else:
        validate_cases(cases, [])
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    before = sha256(args.bundle / "manifest.json")
    results = {}
    for precision in ("full", "int4"):
        # Paired controls keep parser gains separate from routing gains.
        results[precision] = {}
        for name, cls in (("original_v2", Runtime), ("parser_v3", ParserRuntime), ("routing_v4", RoutingRuntime)):
            runtime = cls(args.bundle, precision)
            results[precision][name] = evaluate(runtime, cases)
            del runtime
            print(json.dumps({"event": "routing_measurement", "precision": precision,
                              "arm": name, "kind": args.kind,
                              "scores": {k: v for k, v in results[precision][name].items() if k != "cases"}}), flush=True)
    if before != sha256(args.bundle / "manifest.json"):
        raise RuntimeError("Candidate changed during evaluation")
    passed = all(results[p]["routing_v4"]["combined_correct"] == 100 for p in results)
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "kind": args.kind, "status": "PASS" if passed else "FAIL",
              "strict_pass": passed, "production_ready": False,
              "candidate_manifest_sha256": before, "cases_sha256": sha256(args.cases),
              "confirmation_consumed": args.kind == "confirmation",
              "live_tools_tested": False, "direct_answer_quality_tested": False,
              "base_model_trained": False, "results": results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.kind == "confirmation" and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
