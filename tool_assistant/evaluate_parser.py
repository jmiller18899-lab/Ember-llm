"""Reproduce the v3 parser-only check with explicitly supplied tool choices.

This does not load a model, score routing, call a live service, or authorize a
prototype release. The original frozen combined confirmation is separate.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from .evaluate import arithmetic, case_hash
from .resolver_v3 import REVISION
from .runtime_v3 import ParserRuntime

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tool_assistant/data"
SCOPE = "parser_only_supplied_route"
NOVELTY_INPUTS = (
    "tool_assistant/data/prior-string-hashes.json",
    "tool_assistant/data/router-training.json",
    "tool_assistant/data/confirmation-100.json",
    "tests/test_resolver_v3.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(lock_path):
    lock = json.loads(lock_path.read_text())
    for path, expected in lock["files"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"Parser source changed after freezing: {path}")
    # A parser revision must not silently alter the evaluated v2 candidate.
    original = json.loads((DATA / "candidate-source-lock.json").read_text())
    for name, expected in original["files"].items():
        if digest(ROOT / "tool_assistant" / name) != expected:
            raise ValueError(f"Original frozen source changed: {name}")
    return lock


def prior_hashes():
    seen = set(json.loads((ROOT / NOVELTY_INPUTS[0]).read_text()))
    for name in NOVELTY_INPUTS[1:3]:
        cases = json.loads((ROOT / name).read_text())["cases"]
        seen.update(case_hash(c["user"]) for c in cases)
    # Literal development requests count as already seen, including regression
    # examples. This is exact normalized-text disjointness, not semantic novelty.
    tree = ast.parse((ROOT / NOVELTY_INPUTS[3]).read_text())
    seen.update(case_hash(n.value) for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str))
    return seen


def validate_cases(suite, lock_path):
    if suite["scope"] != SCOPE or suite["parser_revision"] != REVISION:
        raise ValueError("Wrong parser evaluation scope or revision")
    if suite["source_lock_sha256"] != digest(lock_path):
        raise ValueError("Requests were not authored against this source lock")
    cases = suite["cases"]
    ids = [c["id"] for c in cases]
    hashes = [case_hash(c["user"]) for c in cases]
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate parser requests")
    if set(hashes) & prior_hashes():
        raise ValueError("Parser requests overlap prior or development requests")
    counts = {"calculator_call": 0, "location_call": 0, "clarification": 0}
    for case in cases:
        tool = case["tool"]
        expected = case["expected"]
        if tool not in {"calculator", "weather", "get_time"}:
            raise ValueError("Unsupported tool in parser confirmation")
        if expected == {"needs_clarification": True}:
            counts["clarification"] += 1
        else:
            key = {"calculator": "expression", "weather": "location", "get_time": "timezone"}[tool]
            if set(expected["arguments"]) != {key}:
                raise ValueError("Wrong argument shape in expected result")
            if tool == "calculator":
                if not math.isfinite(expected["result"]):
                    raise ValueError("Expected arithmetic result must be finite")
                counts["calculator_call"] += 1
            else:
                counts["location_call"] += 1
    if counts != {"calculator_call": 20, "location_call": 20, "clarification": 20}:
        raise ValueError(f"Unexpected parser case balance: {counts}")
    return counts


def check_case(case, runtime):
    calls = []

    def handler(**arguments):
        calls.append(arguments)
        return arithmetic(arguments["expression"]) if case["tool"] == "calculator" else {"fixture_id": case["id"]}

    try:
        output = runtime.run(case["user"], {case["tool"]: handler})
        expected = case["expected"]
        if expected.get("needs_clarification"):
            passed = (output["status"] == "needs_clarification" and output["call"] is None
                      and not calls and bool(output.get("reason")) and bool(output.get("message")))
        else:
            passed = (output["status"] == "tool_result"
                      and output["call"] == {"name": case["tool"], "arguments": expected["arguments"]}
                      and calls == [expected["arguments"]])
            if passed and case["tool"] == "calculator":
                passed = math.isclose(output["result"], expected["result"], rel_tol=1e-9, abs_tol=1e-9)
    except Exception as exc:
        output, passed = {"error_type": type(exc).__name__, "message": str(exc)}, False
    return {**case, "output": output, "fixture_calls": calls, "passed": passed}


def evaluate(cases):
    rows = []
    for case in cases:
        runtime = ParserRuntime.__new__(ParserRuntime)
        # Deliberately bypass model loading and routing. Only parsing and the
        # inherited caller-supplied-handler boundary are under test here.
        runtime.route = lambda _user, tool=case["tool"]: (tool, 0.0)
        rows.append(check_case(case, runtime))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DATA / "parser-v3-confirmation.json")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("Refusing to overwrite consumed parser confirmation evidence")
    lock_path = DATA / "parser-v3-source-lock.json"
    lock = verify_sources(lock_path)
    suite = json.loads(args.cases.read_text())
    counts = validate_cases(suite, lock_path)
    rows = evaluate(suite["cases"])
    verify_sources(lock_path)
    passed = sum(r["passed"] for r in rows)
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed == len(rows) else "FAIL", "scope": SCOPE,
        "parser_revision": REVISION, "passed": passed, "total": len(rows),
        "case_counts": counts, "source_lock_sha256": digest(lock_path),
        "source_files": lock["files"], "cases_sha256": digest(args.cases),
        "novelty_inputs_sha256": {name: digest(ROOT / name) for name in NOVELTY_INPUTS},
        "by_tool": {tool: {"passed": sum(r["passed"] for r in rows if r["tool"] == tool),
                           "total": sum(r["tool"] == tool for r in rows)}
                    for tool in ("calculator", "weather", "get_time")},
        "model_routing_tested": False, "live_tools_tested": False,
        "direct_answers_tested": False, "production_ready": False,
        "confirmation_consumed": True,
        "interpretation": "Newly authored finite parser check with supplied routes and fixture handlers. "
                          "Exact-text disjoint from recorded prior requests. Reuse is regression evidence, "
                          "not another independent confirmation. Does not qualify a combined prototype for release.",
        "cases": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "scope", "passed", "total", "case_counts", "by_tool")}))
    raise SystemExit(0 if passed == len(rows) else 1)


if __name__ == "__main__":
    main()
