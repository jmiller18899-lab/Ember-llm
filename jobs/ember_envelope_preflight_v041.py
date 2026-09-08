"""CPU-only structural-subtype prompt calibration for Ember v0.0.41.

The v0.0.40 best-safe prompt is the mandatory fallback for every structural
subtype. Prompt challengers are selected only on deterministic synthetic values
of the same format variant; the 90 held-out values are measured once afterward.
No optimizer, model write, GPU submission, promotion, or deployment exists here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import ember_envelope_preflight_v040 as prior

base = prior.base
copy_data = prior.copy_data
control = prior.control
schema_prompt = prior.schema_prompt
DEFAULT_CONFIG = ROOT / "config/ember_envelope_preflight_v0.0.41.json"
TOOL_BY_KIND = dict(prior.TOOL_BY_KIND)
CALIBRATION_KINDS = set(prior.CALIBRATION_KINDS)
PERFECT_KINDS = set(prior.PERFECT_KINDS)
LABEL_BY_KIND = dict(prior.LABEL_BY_KIND)
BASELINE_SPEC = dict(prior.BASELINE_SPEC)


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if path.resolve() != DEFAULT_CONFIG.resolve() or cfg.get("version") != "0.0.41":
        raise ValueError("unsupported v0.0.41 structural calibration configuration")
    if any(cfg.get(k) is not False for k in ("training_authorized", "gpu_training_authorized", "production_authorized")):
        raise ValueError("v0.0.41 is CPU preflight-only")
    if int(cfg.get("expected_cases", 0)) != 90 or int(cfg.get("expected_reference_cases", 0)) != 4:
        raise ValueError("v0.0.41 requires 90 held-out and four reference cases")
    if set(cfg.get("calibration_kinds", [])) != CALIBRATION_KINDS:
        raise ValueError("v0.0.41 calibration kinds changed")
    if int(cfg.get("calibration_values_per_subtype", 0)) != 6:
        raise ValueError("v0.0.41 requires six synthetic values per structural subtype")
    if set(cfg.get("challenger_templates", {})) != {"typed_exact", "query_literal"}:
        raise ValueError("v0.0.41 challenger matrix changed")
    if int(cfg.get("replacement_margin", 0)) < 1:
        raise ValueError("replacement margin must be positive")
    if cfg.get("historical_floor") != {"short_code": 8, "long_code": 7, "url": 7, "path": 9, "mixed": 9}:
        raise ValueError("historical floors changed")
    return cfg


def subtype(kind: str, value: str) -> int:
    """Resolve the exact v0.0.26 structural variant from the rendered value."""
    if kind == "short_code":
        return 0 if len(value) == 4 else 1
    if kind == "long_code":
        left, right = value.split("-", 1)
        return 0 if (len(left), len(right)) == (4, 4) else 1
    if kind == "url":
        tail = value.removeprefix("https://example.test/")
        if "/" in tail:
            return 0
        return 1 if any(ch.islower() for ch in tail) and any(ch.isupper() for ch in tail) else 2
    if kind == "path":
        leaf = value.rsplit("/", 1)[1]
        if leaf.startswith("result-"):
            return 0
        stem = leaf.removesuffix(".json")
        return 2 if stem[-3:-2] == "-" and stem[-2:].isdigit() else 1
    if kind == "mixed":
        body = value.split("_", 1)[1].split("-", 1)[0]
        if body.islower():
            return 0
        if any(ch.islower() for ch in body) and any(ch.isupper() for ch in body):
            return 1
        return 2
    raise ValueError(kind)


def baseline_user(kind: str, value: str) -> str:
    mode, template = BASELINE_SPEC[kind]
    if mode == "template":
        return template.format(label=LABEL_BY_KIND[kind], value=value)
    if mode == "natural":
        return schema_prompt._user_request("web_search", value)
    raise ValueError(mode)


def prompt(kind: str, value: str, user: str) -> str:
    tool, field = TOOL_BY_KIND[kind]
    return f"<|system|>\n{schema_prompt.schema_system(tool, field)}\n<|user|>\n{user}\n<|assistant|>\n"


def calibration_values(cfg: dict) -> dict[tuple[str, int], list[str]]:
    count = int(cfg["calibration_values_per_subtype"])
    used = set(copy_data.HELD_OUT_VALUES)
    out = {}
    for kind in sorted(CALIBRATION_KINDS):
        for variant in range(copy_data.VARIANTS[kind]):
            values = []
            i = 0
            while len(values) < count:
                seed = copy_data._digest("v041-subtype", kind, variant, i)
                value = copy_data._render(kind, variant, seed)
                i += 1
                if value in used:
                    continue
                if subtype(kind, value) != variant:
                    raise ValueError(f"subtype resolver mismatch for {kind}/{variant}: {value}")
                used.add(value)
                values.append(value)
            out[(kind, variant)] = values
    return out


def candidate_user(cfg: dict, kind: str, value: str, candidate: str) -> str:
    if candidate == "baseline":
        return baseline_user(kind, value)
    return cfg["challenger_templates"][candidate].format(label=LABEL_BY_KIND[kind], value=value)


def build_calibration_cases(cfg: dict) -> list[dict]:
    cases = []
    candidates = ["baseline", *cfg["challenger_templates"].keys()]
    for (kind, variant), values in calibration_values(cfg).items():
        tool, field = TOOL_BY_KIND[kind]
        for i, value in enumerate(values):
            for candidate in candidates:
                user = candidate_user(cfg, kind, value, candidate)
                cases.append({
                    "id": f"subcal_{kind}_v{variant}_{i:02d}_{candidate}", "kind": kind, "subtype": variant,
                    "target": value, "candidate": candidate, "expected_tool": tool, "argument_key": field,
                    "prompt": prompt(kind, value, user),
                })
    expected = sum(copy_data.VARIANTS[k] for k in CALIBRATION_KINDS) * int(cfg["calibration_values_per_subtype"]) * len(candidates)
    if len(cases) != expected:
        raise ValueError(f"expected {expected} calibration cases, got {len(cases)}")
    return cases


def select(rows: list[dict], cfg: dict) -> tuple[dict, dict[tuple[str, int], str]]:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[(row["kind"], row["subtype"])][row["candidate"]].append(row)
    selected = {}
    summary = {}
    margin = int(cfg["replacement_margin"])
    candidates = ["baseline", *cfg["challenger_templates"].keys()]
    n = int(cfg["calibration_values_per_subtype"])
    for kind in sorted(CALIBRATION_KINDS):
        for variant in range(copy_data.VARIANTS[kind]):
            key = (kind, variant)
            scores = {}
            for candidate in candidates:
                items = grouped[key][candidate]
                if len(items) != n:
                    raise ValueError(f"incomplete subtype matrix {key}/{candidate}")
                scores[candidate] = {
                    "json": sum(bool(x["score"]["envelope_json_valid"]) for x in items),
                    "tool": sum(bool(x["score"]["tool_name_correct"]) for x in items),
                }
            baseline = scores["baseline"]
            winner = "baseline"
            eligible = [c for c in candidates[1:] if scores[c]["json"] >= baseline["json"] + margin and scores[c]["tool"] >= baseline["tool"] + margin]
            if eligible:
                winner = max(eligible, key=lambda c: (scores[c]["tool"], scores[c]["json"], -candidates.index(c)))
            selected[key] = winner
            summary[f"{kind}:v{variant}"] = {"scores": scores, "selected": winner}
    return summary, selected


def build_final_cases(cfg: dict, selected: dict[tuple[str, int], str]) -> list[dict]:
    counts = defaultdict(int)
    cases = []
    for kind, value, _corrupt in copy_data.DIAGNOSTICS:
        i = counts[kind]
        counts[kind] += 1
        tool, field = TOOL_BY_KIND[kind]
        if kind in CALIBRATION_KINDS:
            variant = subtype(kind, value)
            candidate = selected[(kind, variant)]
            user = candidate_user(cfg, kind, value, candidate)
            label = f"v{variant}:{candidate}"
        else:
            variant = None
            candidate = "v037_schema_natural"
            user = schema_prompt._user_request(tool, value)
            label = candidate
        cases.append({
            "id": f"subtype_target_{kind}_{i:02d}", "kind": kind, "subtype": variant, "target": value,
            "variant": label, "expected_tool": tool, "argument_key": field, "prompt": prompt(kind, value, user),
        })
    if len(cases) != 90:
        raise ValueError("held-out battery changed")
    return cases


def regression_gate(per_kind: dict, cfg: dict) -> dict:
    checks = {kind: {"floor": floor, "actual": int(per_kind[kind]["envelope_json_valid"]), "passed": int(per_kind[kind]["envelope_json_valid"]) >= floor} for kind, floor in cfg["historical_floor"].items()}
    return {"passed": all(x["passed"] for x in checks.values()), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("v041-results"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not os.environ.get("HF_TOKEN", "").strip():
        raise ValueError("HF_TOKEN is required")
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(int(cfg["seed"]))
    torch.use_deterministic_algorithms(True)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "version": "0.0.41", "phase": cfg["phase"], "created_at": datetime.now(timezone.utc).isoformat(), "status": "ERROR", "code_commit": os.environ.get("GITHUB_SHA"), "training_authorized": False, "gpu_training_authorized": False, "production_authorized": False}
    try:
        source_cfg = json.loads(base.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="ember-v041-") as td:
            model, tokenizer, source, _splits, source_ref = base.load_inputs(source_cfg, Path(td), torch)
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != cfg["source_version"]:
                raise ValueError("must measure pinned v0.0.31 step-479 baseline")
            ref_rows = []
            for case in control.reference_cases(cfg):
                gen = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"]))
                score = control.score_reference(case, gen["completion"]); ref_rows.append({**case, **gen, "score": score})
                print(json.dumps({"event":"reference_case","id":case["id"],"json_valid":score["envelope_json_valid"],"tool_name_correct":score["tool_name_correct"]}), flush=True)
            ref_gate = control.reference_summary(ref_rows, cfg)
            cal_rows = []
            for case in build_calibration_cases(cfg):
                gen = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"]))
                score = control.score_case(case, gen["completion"]); cal_rows.append({**case, **gen, "score": score})
            matrix, selected = select(cal_rows, cfg)
            print(json.dumps({"event":"subtype_selected","selected":{f"{k[0]}:v{k[1]}":v for k,v in selected.items()}}), flush=True)
            rows = []
            for case in build_final_cases(cfg, selected):
                gen = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"]))
                score = control.score_case(case, gen["completion"]); rows.append({**case, **gen, "score": score})
                print(json.dumps({"event":"heldout_case","id":case["id"],"kind":case["kind"],"subtype":case["subtype"],"variant":case["variant"],"json_valid":score["envelope_json_valid"],"tool_name_correct":score["tool_name_correct"],"slot_exact":score["slot_exact"]}), flush=True)
            baseline = control.baseline_summary(rows, cfg)
            pk = prior.matrix.per_kind(rows)
            reg = regression_gate(pk, cfg)
            passed = ref_gate["passed"] and baseline["passed"] and reg["passed"]
            report.update(source={"repo_id":source_ref["repo_id"],"checkpoint_path":source_ref["checkpoint_path"],"revision":source_ref["revision"],"checkpoint_sha256":source_ref["checkpoint_sha256"],"step":source["step"],"version":source["train_config"]["version"]}, reference_gate=ref_gate, calibration_matrix=matrix, selected_subtypes={f"{k[0]}:v{k[1]}":v for k,v in selected.items()}, baseline_gate=baseline, per_kind=pk, regression_gate=reg, cases=rows, status="PASS" if passed else "FAIL")
    except Exception as exc:
        report["status"]="ERROR"; report["error"]={"type":type(exc).__name__,"message":str(exc)}; raise
    finally:
        (output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
        if "baseline_gate" in report:
            m=report["baseline_gate"]["metrics"]
            text=f"# Ember v0.0.41 structural-subtype calibration: {report['status']}\n\nCPU only; no optimizer, training, GPU, promotion, or deployment.\n\n- JSON envelope: {m['envelope_json_valid']}/{m['cases']} ({m['envelope_json_valid_rate']:.1%})\n- Correct tool: {m['envelope_tool_name_correct']}/{m['cases']} ({m['envelope_tool_name_rate']:.1%})\n- Slot exact: {m['slot_exact']}/{m['slot_evaluable']} ({m['slot_exact_rate']:.1%})\n- No-regression gate: {'PASS' if report['regression_gate']['passed'] else 'FAIL'}\n"
            (output/"summary.md").write_text(text,encoding="utf-8")
    print(json.dumps({"event":"complete","status":report["status"],"baseline":report.get("baseline_gate",{}).get("metrics"),"selected_subtypes":report.get("selected_subtypes