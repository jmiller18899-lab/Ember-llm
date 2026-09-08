"""CPU-only cross-seed structural-subtype calibration for Ember v0.0.42.

Starts from the v0.0.40 safe stack, explicitly carries forward the two
non-regressing v0.0.41 gains, restores short_code/len5 to the v0.0.40 prompt,
and calibrates only six remaining failure-bearing subtypes. Challengers must
beat the mandatory baseline on two independent synthetic folds.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import ember_envelope_preflight_v040 as prior

base = prior.base
copy_data = prior.copy_data
control = prior.control
schema_prompt = prior.schema_prompt
DEFAULT_CONFIG = ROOT / "config/ember_envelope_preflight_v0.0.42.json"
TOOL_BY_KIND = dict(prior.TOOL_BY_KIND)
WEAK_KINDS = set(prior.CALIBRATION_KINDS)

TARGET_SUBTYPES = {
    "short_code/len4", "short_code/len5", "long_code/4x4",
    "long_code/3x5", "url/two_segment", "path/plain_leaf",
}
SUBTYPE_VARIANT = {
    "short_code/len4": ("short_code", 0),
    "short_code/len5": ("short_code", 1),
    "long_code/4x4": ("long_code", 0),
    "long_code/3x5": ("long_code", 1),
    "url/two_segment": ("url", 0),
    "path/plain_leaf": ("path", 1),
}
LABEL_BY_KIND = {
    "short_code": "identifier", "long_code": "identifier",
    "url": "URL", "path": "file path",
}
EXPECTED_KIND_FLOOR = {"short_code": 8, "long_code": 7, "url": 9, "path": 9, "mixed": 10}
EXPECTED_SUBTYPE_FLOOR = {
    "short_code/len4": 5, "short_code/len5": 3,
    "long_code/4x4": 5, "long_code/3x5": 2,
    "url/one_mixed": 4, "url/two_segment": 2,
    "path/plain_leaf": 3, "mixed/upper": 3,
}

V041_QUOTED_TEXT = 'Use web_search to search for the exact quoted text "{value}".'
V041_LITERAL_QUERY = 'Use web_search. Use the literal query string "{value}".'


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if path.resolve() != DEFAULT_CONFIG.resolve() or cfg.get("version") != "0.0.42":
        raise ValueError("unsupported v0.0.42 configuration")
    if any(cfg.get(k) is not False for k in ("training_authorized", "gpu_training_authorized", "production_authorized")):
        raise ValueError("v0.0.42 is CPU preflight-only")
    if int(cfg.get("expected_cases", 0)) != 90 or int(cfg.get("expected_reference_cases", 0)) != 4:
        raise ValueError("v0.0.42 requires 90 held-out cases and four references")
    if set(cfg.get("calibration_subtypes", [])) != TARGET_SUBTYPES:
        raise ValueError("v0.0.42 target subtypes changed")
    if cfg.get("calibration_folds") != ["select", "confirm"]:
        raise ValueError("v0.0.42 requires select and confirm folds")
    if int(cfg.get("calibration_values_per_fold", 0)) != 6:
        raise ValueError("v0.0.42 requires six values per fold")
    if int(cfg.get("replacement_margin_per_fold", 0)) < 1 or int(cfg.get("replacement_margin_combined", 0)) < 2:
        raise ValueError("v0.0.42 replacement margins weakened")
    if set(cfg.get("challenger_templates", {})) != {"typed_exact", "tool_query", "query_exact"}:
        raise ValueError("v0.0.42 challenger set changed")
    if cfg.get("historical_floor") != EXPECTED_KIND_FLOOR:
        raise ValueError("v0.0.42 kind floors changed")
    if cfg.get("historical_subtype_floor") != EXPECTED_SUBTYPE_FLOOR:
        raise ValueError("v0.0.42 subtype floors changed")
    return cfg


def subtype_for(kind: str, value: str) -> str:
    if kind == "short_code":
        return f"short_code/len{len(value)}"
    if kind == "long_code":
        left, right = value.split("-", 1)
        return f"long_code/{len(left)}x{len(right)}"
    if kind == "url":
        parts = [p for p in urlsplit(value).path.split("/") if p]
        if len(parts) == 2:
            return "url/two_segment"
        if len(parts) == 1:
            return "url/one_mixed" if parts[0] and parts[0][0].islower() else "url/one_upper"
        raise ValueError(f"unsupported URL structure: {value}")
    if kind == "path":
        leaf = Path(value).name
        stem = leaf[:-5] if leaf.endswith(".json") else leaf
        if re.fullmatch(r"result-[A-Za-z0-9]+", stem):
            return "path/result_code"
        if re.fullmatch(r"[A-Za-z]+-\d{2}", stem):
            return "path/leaf_numeric"
        return "path/plain_leaf"
    if kind == "mixed":
        body = value.split("_", 1)[1].rsplit("-", 1)[0]
        letters = [ch for ch in body if ch.isalpha()]
        if letters and all(ch.islower() for ch in letters):
            return "mixed/lower"
        if letters and all(ch.isupper() for ch in letters):
            return "mixed/upper"
        return "mixed/mixedcase"
    return f"{kind}/default"


def assembled_safe_user(kind: str, value: str, cfg: dict) -> tuple[str, str]:
    subtype = subtype_for(kind, value)
    if subtype == "url/one_mixed":
        return "v041:quoted_text|url/one_mixed", V041_QUOTED_TEXT.format(value=value)
    if subtype == "mixed/upper":
        return "v041:literal_query|mixed/upper", V041_LITERAL_QUERY.format(value=value)
    if kind in WEAK_KINDS:
        return f"v040:{prior.BASELINE_BY_KIND[kind]}", prior.baseline_user(kind, value)
    return "v037_schema_natural", schema_prompt._user_request(TOOL_BY_KIND[kind][0], value)


def _prompt(kind: str, user_request: str) -> str:
    tool, field = TOOL_BY_KIND[kind]
    return f"<|system|>\n{schema_prompt.schema_system(tool, field)}\n<|user|>\n{user_request}\n<|assistant|>\n"


def calibration_values(cfg: dict) -> dict[str, dict[str, list[str]]]:
    count = int(cfg["calibration_values_per_fold"])
    used = set(copy_data.HELD_OUT_VALUES)
    out: dict[str, dict[str, list[str]]] = {}
    for subtype in sorted(TARGET_SUBTYPES):
        kind, variant = SUBTYPE_VARIANT[subtype]
        out[subtype] = {}
        for fold in cfg["calibration_folds"]:
            values = []
            i = 0
            while len(values) < count:
                seed = copy_data._digest("v042-calibration", fold, subtype, i)
                value = copy_data._render(kind, variant, seed)
                i += 1
                if value in used or subtype_for(kind, value) != subtype:
                    continue
                used.add(value)
                values.append(value)
            out[subtype][fold] = values
    return out


def _challenger_user(kind: str, value: str, template: str) -> str:
    return template.format(label=LABEL_BY_KIND[kind], value=value)


def build_calibration_cases(cfg: dict) -> list[dict]:
    cases = []
    for subtype, folds in calibration_values(cfg).items():
        kind, _ = SUBTYPE_VARIANT[subtype]
        tool, field = TOOL_BY_KIND[kind]
        for fold, values in folds.items():
            for value_index, value in enumerate(values):
                _, baseline_request = assembled_safe_user(kind, value, cfg)
                candidates = {"baseline": baseline_request}
                candidates.update({name: _challenger_user(kind, value, template) for name, template in cfg["challenger_templates"].items()})
                for candidate, user_request in candidates.items():
                    cases.append({
                        "id": f"cal_{subtype.replace('/', '_')}_{fold}_{value_index:02d}_{candidate}",
                        "kind": kind, "subtype": subtype, "fold": fold, "target": value,
                        "candidate": candidate, "expected_tool": tool, "argument_key": field,
                        "prompt": _prompt(kind, user_request),
                    })
    expected = len(TARGET_SUBTYPES) * 2 * int(cfg["calibration_values_per_fold"]) * 4
    if len(cases) != expected:
        raise ValueError(f"expected {expected} calibration cases, got {len(cases)}")
    return cases


def summarize_matrix(rows: list[dict], cfg: dict) -> tuple[dict, dict[str, str]]:
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        grouped[row["subtype"]][row["fold"]][row["candidate"]].append(row)
    candidates = ["typed_exact", "tool_query", "query_exact"]
    expected = int(cfg["calibration_values_per_fold"])
    per_fold = int(cfg["replacement_margin_per_fold"])
    combined = int(cfg["replacement_margin_combined"])
    summary, selected = {}, {}
    for subtype in sorted(TARGET_SUBTYPES):
        metrics = {}
        for fold in cfg["calibration_folds"]:
            metrics[fold] = {}
            for name in ["baseline", *candidates]:
                items = grouped[subtype][fold][name]
                if len(items) != expected:
                    raise ValueError(f"incomplete matrix for {subtype}/{fold}/{name}")
                valid = sum(bool(x["score"]["envelope_json_valid"]) for x in items)
                tool = sum(bool(x["score"]["tool_name_correct"]) for x in items)
                metrics[fold][name] = {"cases": len(items), "envelope_json_valid": valid, "tool_name_correct": tool}
        eligible = []
        for name in candidates:
            fold_wins = all(
                metrics[fold][name]["envelope_json_valid"] >= metrics[fold]["baseline"]["envelope_json_valid"] + per_fold
                and metrics[fold][name]["tool_name_correct"] >= metrics[fold]["baseline"]["tool_name_correct"] + per_fold
                for fold in cfg["calibration_folds"]
            )
            base_valid = sum(metrics[f]["baseline"]["envelope_json_valid"] for f in cfg["calibration_folds"])
            base_tool = sum(metrics[f]["baseline"]["tool_name_correct"] for f in cfg["calibration_folds"])
            cand_valid = sum(metrics[f][name]["envelope_json_valid"] for f in cfg["calibration_folds"])
            cand_tool = sum(metrics[f][name]["tool_name_correct"] for f in cfg["calibration_folds"])
            if fold_wins and cand_valid >= base_valid + combined and cand_tool >= base_tool + combined:
                eligible.append((cand_tool, cand_valid, -candidates.index(name), name))
        winner = max(eligible)[3] if eligible else "baseline"
        selected[subtype] = winner
        summary[subtype] = {"folds": metrics, "selected": winner}
    return summary, selected


def final_user(kind: str, value: str, selected: dict[str, str], cfg: dict) -> tuple[str, str]:
    subtype = subtype_for(kind, value)
    baseline_id, baseline_request = assembled_safe_user(kind, value, cfg)
    if subtype not in TARGET_SUBTYPES or selected[subtype] == "baseline":
        return f"{baseline_id}|{subtype}", baseline_request
    choice = selected[subtype]
    return f"{choice}|{subtype}", _challenger_user(kind, value, cfg["challenger_templates"][choice])


def build_final_cases(cfg: dict, selected: dict[str, str]) -> list[dict]:
    diagnostics = list(copy_data.DIAGNOSTICS)
    counts = defaultdict(int)
    cases = []
    for kind, value, _ in diagnostics:
        index = counts[kind]; counts[kind] += 1
        tool, field = TOOL_BY_KIND[kind]
        subtype = subtype_for(kind, value)
        variant, request = final_user(kind, value, selected, cfg)
        cases.append({
            "id": f"crossseed_target_{kind}_{index:02d}", "kind": kind, "subtype": subtype,
            "target": value, "variant": variant, "expected_tool": tool, "argument_key": field,
            "prompt": _prompt(kind, request),
        })
    if len(cases) != 90 or any(counts[k] != 10 for k in copy_data.KINDS):
        raise ValueError("held-out battery changed")
    return cases


def cohort(rows: list[dict]) -> dict:
    total = len(rows)
    valid = sum(bool(r["score"]["envelope_json_valid"]) for r in rows)
    tool = sum(bool(r["score"]["tool_name_correct"]) for r in rows)
    slot = sum(bool(r["score"]["slot_exact"]) for r in rows)
    return {"cases": total, "envelope_json_valid": valid, "envelope_json_valid_rate": valid/total if total else None, "tool_name_correct": tool, "tool_name_rate": tool/total if total else None, "slot_exact": slot, "slot_exact_rate_on_evaluable": slot/tool if tool else None}


def per_kind(rows: list[dict]) -> dict:
    return {kind: cohort([r for r in rows if r["kind"] == kind]) for kind in copy_data.KINDS}


def per_subtype(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows: grouped[row["subtype"]].append(row)
    return {key: cohort(items) for key, items in sorted(grouped.items())}


def regression_gate(metrics: dict, floors: dict) -> dict:
    checks = {key: {"floor": int(floor), "actual": int(metrics[key]["envelope_json_valid"])} for key, floor in floors.items()}
    for check in checks.values(): check["passed"] = check["actual"] >= check["floor"]
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks}


def summary_markdown(report: dict) -> str:
    ref = report["reference_gate"]["metrics"]
    m = report["baseline_gate"]["metrics"]
    lines = [
        f"# Ember v0.0.42 cross-seed subtype calibration: {report['status']}", "",
        "CPU-only. No optimizer, training, GPU submission, promotion, deployment, or integration occurred.", "",
        "| Measurement | Result |", "| --- | ---: |",
        f"| Exact v0.0.8 reference JSON | {ref['envelope_json_valid']}/{ref['cases']} ({ref['envelope_json_valid_rate']:.1%}) |",
        f"| Exact v0.0.8 reference tool | {ref['envelope_tool_name_correct']}/{ref['cases']} ({ref['envelope_tool_name_rate']:.1%}) |",
        f"| 90-case JSON envelope | {m['envelope_json_valid']}/{m['cases']} ({m['envelope_json_valid_rate']:.1%}) |",
        f"| 90-case correct tool | {m['envelope_tool_name_correct']}/{m['cases']} ({m['envelope_tool_name_rate']:.1%}) |",
        f"| kind no-regression gate | {'PASS' if report['kind_regression_gate']['passed'] else 'FAIL'} |",
        f"| subtype no-regression gate | {'PASS' if report['subtype_regression_gate']['passed'] else 'FAIL'} |", "",
        "Frozen selection:", "",
    ]
    lines += [f"- `{k}`: `{v}`" for k, v in sorted(report["selected_candidates"].items())]
    lines += ["", "A PASS validates only the baseline for a separate placement-learning design; it does not authorize training."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("v042-results"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not os.environ.get("HF_TOKEN", "").strip(): raise ValueError("HF_TOKEN is required")
    import torch
    torch.set_num_threads(2); torch.manual_seed(int(cfg["seed"])); torch.use_deterministic_algorithms(True)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version":1,"version":"0.0.42","phase":cfg["phase"],"created_at":datetime.now(timezone.utc).isoformat(),"status":"ERROR","code_commit":os.environ.get("GITHUB_SHA"),"training_authorized":False,"gpu_training_authorized":False,"production_authorized":False,"historical_floor":cfg["historical_floor"],"historical_subtype_floor":cfg["historical_subtype_floor"]}
    try:
        source_cfg = json.loads(base.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="ember-v042-") as td:
            model, tokenizer, source, _splits, source_ref = base.load_inputs(source_cfg, Path(td), torch)
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != cfg["source_version"]: raise ValueError("wrong source checkpoint")
            reference_rows = []
            for case in control.reference_cases(cfg):
                g = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"]))
                s = control.score_reference(case, g["completion"]); reference_rows.append({**case, **g, "score": s})
            reference_gate = control.reference_summary(reference_rows, cfg)
            calibration_rows = []
            for case in build_calibration_cases(cfg):
                g = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"]))
                s = control.score_case(case, g["completion"]); calibration_rows.append({**case, **g, "score": s})
                print(json.dumps({"event":"calibration_case","subtype":case["subtype"],"fold":case["fold"],"candidate":case["candidate"],"json_valid":s["envelope_json_valid"],"tool_name_correct":s["tool_name_correct"]}), flush=True)
            calibration_summary, selected = summarize_matrix(calibration_rows, cfg)
            print(json.dumps({"event":"matrix_selected","selected_candidates":selected}), flush=True)
            rows = []
            for case in build_final_cases(cfg, selected):
                g = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"]))
                s = control.score_case(case, g["completion"]); rows.append({**case, **g, "score": s})
                print(json.dumps({"event":"heldout_case","id":case["id"],"kind":case["kind"],"subtype":case["subtype"],"variant":case["variant"],"json_valid":s["envelope_json_valid"],"tool_name_correct":s["tool_name_correct"],"slot_exact":s["slot_exact"]}), flush=True)
            baseline_gate = control.baseline_summary(rows, cfg)
            kind_metrics = per_kind(rows); subtype_metrics = per_subtype(rows)
            kind_gate = regression_gate(kind_metrics, cfg["historical_floor"]); subtype_gate = regression_gate(subtype_metrics, cfg["historical_subtype_floor"])
            passed = reference_gate["passed"] and baseline_gate["passed"] and kind_gate["passed"] and subtype_gate["passed"]
            report.update(source={"repo_id":source_ref["repo_id"],"checkpoint_path":source_ref["checkpoint_path"],"revision":source_ref["revision"],"checkpoint_sha256":source_ref["checkpoint_sha256"],"step":source["step"],"version":source["train_config"]["version"]},reference_cases=reference_rows,reference_gate=reference_gate,calibration_cases=calibration_rows,calibration_summary=calibration_summary,selected_candidates=selected,cases=rows,baseline_gate=baseline_gate,per_kind=kind_metrics,per_subtype=subtype_metrics,kind_regression_gate=kind_gate,subtype_regression_gate=subtype_gate,status="PASS" if passed else "FAIL")
    except Exception as exc:
        report["status"] = "ERROR"; report["error"] = {"type":type(exc).__name__,"message":str(exc)}; raise
    finally:
        (output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
        if "reference_gate" in report and "baseline_gate" in report: (output/"summary.md").write_text(summary_markdown(report),encoding="utf-8")
    print(json.dumps({"event":"complete","status":report["status"],"baseline":report["baseline_gate"]["metrics"],"selected_candidates":report["selected_candidates"],"kind_regression_gate":report["kind_regression_gate"],"subtype_regression_gate":report["subtype_regression_gate"]}), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__": raise SystemExit(main())
