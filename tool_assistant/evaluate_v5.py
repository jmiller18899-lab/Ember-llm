"""Paired v5 measurements with frozen-candidate verification for new requests."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from .evidence_v5 import emit_json
from .evaluate import case_hash, evaluate, validate_cases
from .evaluate_parser import check_case
from .evaluate_v4 import SOURCE_FILES as V4_SOURCE_FILES, REQUIRED_CANDIDATE_FILES as V4_REQUIRED_FILES
from .routing_data_v5 import load_training
from .runtime import sha256
from .runtime_v4 import RoutingRuntime
from .runtime_v5 import ParserV4Control, RoutingV5Runtime

SOURCE_FILES = tuple(sorted(set(V4_SOURCE_FILES) | {
    *(f"tool_assistant/{name}.py" for name in ("resolver_v4", "routing_data_v5", "routing_v5",
        "runtime_v5", "build_v5", "evaluate_v5", "evidence_v5", "evaluate_parser")),
    "tests/test_resolver_v3.py", "tests/test_resolver_v4.py", "tests/test_routing_v5.py",
}))
REQUIRED_CANDIDATE_FILES = V4_REQUIRED_FILES | {"routing-v5.pt", "routing-v5-training.json"}


def prior_hashes(root):
    historical, development = load_training(root)
    seen = set(json.loads((root / "tool_assistant/data/prior-string-hashes.json").read_text()))
    seen.update(case_hash(c["user"]) for c in historical + development)
    for name in ("confirmation-100.json", "parser-v3-confirmation.json"):
        suite = json.loads((root / "tool_assistant/data" / name).read_text())
        seen.update(case_hash(c["user"]) for c in suite["cases"])
    for name in ("tests/test_resolver_v3.py", "tests/test_resolver_v4.py", "tests/test_routing_v4.py", "tests/test_routing_v5.py"):
        tree = ast.parse((root / name).read_text())
        seen.update(case_hash(n.value) for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str))
    return seen


def verify_confirmation(suite, root, lock_path, bundle):
    lock = json.loads(lock_path.read_text())
    if suite.get("source_lock_sha256") != sha256(lock_path):
        raise ValueError("Confirmation is not bound to the source freeze")
    if set(lock.get("files", {})) != set(SOURCE_FILES):
        raise ValueError("Incomplete source lock")
    for name, digest in lock["files"].items():
        target = (root / name).resolve()
        if not target.is_relative_to(root.resolve()) or sha256(target) != digest:
            raise ValueError(f"Frozen source changed: {name}")
    candidate_files = lock.get("candidate_files", {})
    if not REQUIRED_CANDIDATE_FILES.issubset(candidate_files):
        raise ValueError("The fitted candidate was not frozen")
    manifest = json.loads((bundle / "manifest.json").read_text())
    if candidate_files != manifest["files"]:
        raise ValueError("Candidate differs from the pre-confirmation freeze")
    for name, digest in candidate_files.items():
        target = (bundle / name).resolve()
        if not target.is_relative_to(bundle.resolve()) or sha256(target) != digest:
            raise ValueError(f"Frozen candidate changed: {name}")
    validate_cases(suite["cases"], prior_hashes(root))
    return lock


def parser_regressions(root):
    cases = json.loads((root / "tool_assistant/data/parser-v3-confirmation.json").read_text())["cases"]
    rows = []
    for case in cases:
        runtime = ParserV4Control.__new__(ParserV4Control)
        runtime.route = lambda _, tool=case["tool"]: (tool, 0.0)
        rows.append(check_case(case, runtime))
    return {"passed": sum(r["passed"] for r in rows), "total": len(rows),
            "scope": "consumed_parser_regressions_with_supplied_routes", "cases": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--kind", choices=("development", "confirmation"), required=True)
    parser.add_argument("--source-lock", type=Path, default=Path("tool_assistant/data/routing-v5-source-lock.json"))
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("Refusing to overwrite measured evidence")
    root = Path(__file__).resolve().parent.parent
    suite = json.loads(args.cases.read_text())
    if args.kind == "confirmation":
        verify_confirmation(suite, root, args.source_lock, args.bundle)
    else:
        validate_cases(suite["cases"], [])
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    before = sha256(args.bundle / "manifest.json")
    results = {}
    for precision in ("full", "int4"):
        results[precision] = {}
        for name, cls in (("routing_v4", RoutingRuntime), ("parser_v4", ParserV4Control), ("routing_v5", RoutingV5Runtime)):
            runtime = cls(args.bundle, precision)
            result = results[precision][name] = evaluate(runtime, suite["cases"])
            del runtime
            print(json.dumps({"event": "routing_v5_measurement", "precision": precision,
                              "arm": name, "kind": args.kind,
                              "scores": {k: v for k, v in result.items() if k != "cases"}}), flush=True)
    regressions = parser_regressions(root)
    if before != sha256(args.bundle / "manifest.json"):
        raise RuntimeError("The measured candidate changed")
    if args.kind == "confirmation":
        verify_confirmation(suite, root, args.source_lock, args.bundle)
    passed = all(results[p]["routing_v5"]["combined_correct"] == 100 for p in results) and regressions["passed"] == regressions["total"] == 60
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "kind": args.kind, "status": "PASS" if passed else "FAIL", "strict_pass": passed,
              "candidate_manifest_sha256": before, "cases_sha256": sha256(args.cases),
              "source_lock_sha256": sha256(args.source_lock) if args.kind == "confirmation" else None,
              "confirmation_consumed": args.kind == "confirmation", "production_ready": False,
              "base_model_trained": False, "uses_ember_features": False,
              "live_tools_tested": False, "direct_answer_quality_tested": False,
              "interpretation": "Text helper routing, exact tool arguments, and fixture dispatch. Direct requests score routing only.",
              "results": results, "parser_regressions": regressions}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    emit_json(args.report, f"routing-v5-{args.kind}.json")
    if args.kind == "confirmation" and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
