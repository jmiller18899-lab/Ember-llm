from __future__ import annotations
from collections import defaultdict
from jobs import ember_v048_data as data

base = data.base
control = data.control
v045 = data.v045


def familiar_90(model, tokenizer, torch, cfg: dict) -> dict:
    rows = []
    for case in v045.frozen_v044_cases():
        generation = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"]))
        score = control.score_case(case, generation["completion"])
        rows.append({"id": case["id"], "kind": case["kind"], "subtype": case["subtype"], "score": score})
    by_kind, by_subtype = defaultdict(int), defaultdict(int)
    for row in rows:
        if row["score"]["envelope_json_valid"] and row["score"]["tool_name_correct"]:
            by_kind[row["kind"]] += 1
            by_subtype[row["subtype"]] += 1
    return {
        "cases": 90,
        "envelope_json_valid": sum(r["score"]["envelope_json_valid"] for r in rows),
        "correct_tool": sum(r["score"]["tool_name_correct"] for r in rows),
        "by_kind": dict(by_kind),
        "by_subtype": dict(by_subtype),
        "rows": rows,
    }


def references(model, tokenizer, torch, cfg: dict) -> dict:
    rows = []
    for case in control.reference_cases(v045.V044_CONFIG):
        generation = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"]))
        score = control.score_reference(case, generation["completion"])
        rows.append({"id": case["id"], "score": score})
    passed = sum(r["score"]["envelope_json_valid"] and r["score"]["tool_name_correct"] for r in rows)
    return {"passed": passed == 4, "passed_cases": passed, "cases": 4, "rows": rows}


def floor_checks(result: dict, cfg: dict) -> dict:
    checks = {
        "json_floor": result["envelope_json_valid"] >= int(cfg["gate"]["minimum_familiar_envelope_json"]),
        "tool_floor": result["correct_tool"] >= int(cfg["gate"]["minimum_familiar_correct_tool"]),
    }
    for kind, floor in cfg["historical_kind_floor"].items():
        checks[f"kind_{kind}"] = int(result["by_kind"].get(kind, 0)) >= int(floor)
    for subtype, floor in cfg["historical_subtype_floor"].items():
        checks[f"subtype_{subtype.replace('/', '_')}"] = int(result["by_subtype"].get(subtype, 0)) >= int(floor)
    return {"passed": all(checks.values()), "checks": checks}
