"""Score a frozen candidate once on new requests; never fit or tune here."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import operator
from pathlib import Path
import sys

import torch

from .runtime import Runtime, sha256


def arithmetic(expression):
    if len(expression) > 200:
        raise ValueError("Expression too long")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 60:
        raise ValueError("Expression too complex")
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in ops:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("Exponent outside prototype limit")
            value = ops[type(node.op)](left, right)
        else:
            raise ValueError("Unsupported arithmetic")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e15:
            raise ValueError("Arithmetic result outside prototype limit")
        return value

    return visit(tree.body)


def case_hash(user):
    return hashlib.sha256(" ".join(user.casefold().split()).encode()).hexdigest()


def validate_cases(cases, old_hashes):
    ids = [c["id"] for c in cases]
    hashes = [case_hash(c["user"]) for c in cases]
    if len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)):
        raise ValueError("Duplicate fresh requests")
    if set(hashes) & set(old_hashes):
        raise ValueError("Fresh requests overlap old experiments or training")
    counts = {k: sum(c["route"] == k for c in cases) for k in ("direct", "weather", "calculator", "web_search", "get_time")}
    if counts != {k: 20 for k in counts}:
        raise ValueError(f"Expected 20 new requests per route: {counts}")
    return counts


def score(case, output):
    route_ok = output["route"] == case["route"]
    if case["route"] == "direct":
        return {"route_ok": route_ok, "arguments_ok": None, "passed": route_ok and output["call"] is None}
    call = output.get("call")
    arguments_ok = False
    if route_ok and call and call.get("name") == case["route"]:
        args = call.get("arguments", {})
        if case["route"] == "calculator":
            if set(args) == {"expression"}:
                try:
                    arguments_ok = math.isclose(arithmetic(args["expression"]), case["result"], rel_tol=1e-9, abs_tol=1e-9)
                except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError):
                    pass
        else:
            arguments_ok = args == case["arguments"]
    return {"route_ok": route_ok, "arguments_ok": arguments_ok, "passed": route_ok and arguments_ok}


def evaluate(runtime, cases):
    rows = []
    for case in cases:
        try:
            output = runtime.plan(case["user"])
            result = score(case, output)
        except (ValueError, RuntimeError) as exc:
            output = {"error": str(exc)}
            result = {"route_ok": False, "arguments_ok": False, "passed": False}
        # A deterministic fixture checks dispatch without making live service calls.
        dispatch_ok = None
        if result["passed"] and case["route"] != "direct":
            calls = []

            def handler(**arguments):
                calls.append(arguments)
                return arithmetic(arguments["expression"]) if case["route"] == "calculator" else {"fixture_id": case["id"]}

            dispatched = runtime.run(case["user"], {case["route"]: handler})
            dispatch_ok = dispatched["status"] == "tool_result" and len(calls) == 1 and calls[0] == output["call"]["arguments"]
            result["passed"] = result["passed"] and dispatch_ok
        rows.append({**case, "output": output, **result, "fixture_dispatch_ok": dispatch_ok})
    return {
        "routing_correct": sum(r["route_ok"] for r in rows), "routing_total": len(rows),
        "exact_arguments_correct": sum(r["arguments_ok"] is True for r in rows), "arguments_total": 80,
        "combined_correct": sum(r["passed"] for r in rows), "combined_total": len(rows),
        "by_route": {route: {"passed": sum(r["passed"] for r in rows if r["route"] == route), "total": 20} for route in ("direct", "weather", "calculator", "web_search", "get_time")},
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise RuntimeError("Refusing to overwrite consumed confirmation evidence")
    torch.set_num_threads(2)
    torch.manual_seed(20260911)
    torch.use_deterministic_algorithms(True)
    cases = json.loads(args.cases.read_text())["cases"]
    index = Path(__file__).parent / "data/prior-string-hashes.json"
    counts = validate_cases(cases, json.loads(index.read_text()))
    before = sha256(args.bundle / "manifest.json")
    results = {}
    for precision in ("full", "int4"):
        runtime = Runtime(args.bundle, precision)
        results[precision] = evaluate(runtime, cases)
        del runtime
    if sha256(args.bundle / "manifest.json") != before:
        raise RuntimeError("Frozen candidate changed during evaluation")
    passed = all(r["combined_correct"] == r["combined_total"] for r in results.values())
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL", "strict_pass": passed,
        "candidate_manifest_sha256": before, "cases_sha256": sha256(args.cases),
        "fresh_case_counts": counts, "results": results,
        "live_tools_tested": False, "direct_answer_quality_tested": False,
        "production_ready": False,
        "interpretation": "Tool planning and fixture dispatch only. Direct answers and live service adapters require separate gates.",
        "confirmation_consumed": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = ["# Frozen Ember combined confirmation", "", f"Strict gate: **{report['status']}**", "",
               "| Precision | Routing | Exact arguments | Combined |", "| --- | ---: | ---: | ---: |"]
    for key, r in results.items():
        summary.append(f"| {key} | {r['routing_correct']}/100 | {r['exact_arguments_correct']}/80 | {r['combined_correct']}/100 |")
    summary += ["", report["interpretation"], "", "No tuning or refitting on this consumed confirmation set."]
    args.report.with_suffix(".md").write_text("\n".join(summary) + "\n")
    print(json.dumps({"event": "combined_confirmation", "strict_pass": passed, "results": {k: {n: v for n, v in r.items() if n != "cases"} for k, r in results.items()}}), flush=True)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
