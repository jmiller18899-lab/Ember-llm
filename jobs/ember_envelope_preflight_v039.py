"""CPU-only targeted prompt matrix for Ember v0.0.39.

The pinned v0.0.31 step-479 checkpoint is measured only. Five weak web-search
value kinds are calibrated on synthetic values that are disjoint from the
90-case held-out battery. The best prompt variant per kind is then frozen and
measured once on the same 90-case battery used by v0.0.36-v0.0.38.

No optimizer, model write, GPU submission, promotion, deployment, or production
integration is possible in this runner.
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

from jobs import ember_envelope_preflight_v038 as prior

base = prior.base
copy_data = prior.copy_data
control = prior.prior.prior  # v0.0.36 scoring/reference helpers
schema_prompt = prior.prior   # v0.0.37 schema-bearing prompt helpers
DEFAULT_CONFIG = ROOT / "config/ember_envelope_preflight_v0.0.39.json"
TOOL_BY_KIND = dict(prior.TOOL_BY_KIND)
CALIBRATION_KINDS = {"short_code", "long_code", "url", "path", "mixed"}
PERFECT_KINDS = set(copy_data.KINDS) - CALIBRATION_KINDS
LABEL_BY_KIND = {
    "short_code": "identifier",
    "long_code": "identifier",
    "url": "URL",
    "path": "file path",
    "mixed": "identifier",
}


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if path.resolve() != DEFAULT_CONFIG.resolve() or cfg.get("version") != "0.0.39":
        raise ValueError("unsupported v0.0.39 targeted calibration configuration")
    if any(cfg.get(key) is not False for key in (
        "training_authorized", "gpu_training_authorized", "production_authorized"
    )):
        raise ValueError("v0.0.39 is CPU preflight-only; training and production authorization must be false")
    if int(cfg.get("expected_cases", 0)) != 90 or int(cfg.get("expected_reference_cases", 0)) != 4:
        raise ValueError("v0.0.39 requires 90 held-out cases and four reference cases")
    if set(cfg.get("calibration_kinds", [])) != CALIBRATION_KINDS:
        raise ValueError("v0.0.39 calibration kinds changed")
    if int(cfg.get("calibration_values_per_kind", 0)) != 4:
        raise ValueError("v0.0.39 requires four synthetic calibration values per weak kind")
    variants = cfg.get("prompt_variants", {})
    priority = cfg.get("variant_priority", [])
    if set(variants) != {"current_exact", "tool_named", "lookup_exact"} or set(priority) != set(variants):
        raise ValueError("v0.0.39 prompt matrix changed")
    if len(priority) != len(set(priority)):
        raise ValueError("variant priority must be unique")
    for template in variants.values():
        if "{value}" not in template or "{label}" not in template:
            raise ValueError("every v0.0.39 prompt variant must contain {label} and {value}")
    for group in ("reference_gate", "baseline_gate"):
        for key in ("minimum_envelope_json_valid_rate", "minimum_envelope_tool_name_rate"):
            threshold = float(cfg.get(group, {}).get(key, -1))
            if not 0.0 < threshold <= 1.0:
                raise ValueError(f"invalid {group} threshold: {key}")
    return cfg


def calibration_values(cfg: dict) -> dict[str, list[str]]:
    """Create deterministic values that never overlap held-out targets/corruptions."""
    count = int(cfg["calibration_values_per_kind"])
    used = set(copy_data.HELD_OUT_VALUES)
    out: dict[str, list[str]] = {}
    for kind in sorted(CALIBRATION_KINDS):
        values: list[str] = []
        i = 0
        while len(values) < count:
            variant = i % copy_data.VARIANTS[kind]
            seed = copy_data._digest("v039-calibration", kind, variant, i)
            value = copy_data._render(kind, variant, seed)
            i += 1
            if value in used:
                continue
            used.add(value)
            values.append(value)
        out[kind] = values
    return out


def _user_from_template(kind: str, value: str, template: str) -> str:
    return template.format(label=LABEL_BY_KIND[kind], value=value)


def _prompt(kind: str, value: str, user_request: str) -> str:
    tool, field = TOOL_BY_KIND[kind]
    return (
        f"<|system|>\n{schema_prompt.schema_system(tool, field)}\n"
        f"<|user|>\n{user_request}\n"
        "<|assistant|>\n"
    )


def build_calibration_cases(cfg: dict) -> list[dict]:
    cases = []
    for kind, values in calibration_values(cfg).items():
        tool, field = TOOL_BY_KIND[kind]
        if (tool, field) != ("web_search", "query"):
            raise ValueError(f"targeted weak kind unexpectedly changed tool mapping: {kind}")
        for value_index, value in enumerate(values):
            for variant_name in cfg["variant_priority"]:
                user = _user_from_template(kind, value, cfg["prompt_variants"][variant_name])
                cases.append({
                    "id": f"cal_{kind}_{value_index:02d}_{variant_name}",
                    "kind": kind,
                    "target": value,
                    "variant": variant_name,
                    "expected_tool": tool,
                    "argument_key": field,
                    "prompt": _prompt(kind, value, user),
                })
    expected = len(CALIBRATION_KINDS) * int(cfg["calibration_values_per_kind"]) * len(cfg["prompt_variants"])
    if len(cases) != expected:
        raise ValueError(f"expected {expected} calibration cases, got {len(cases)}")
    return cases


def summarize_matrix(rows: list[dict], cfg: dict) -> tuple[dict, dict[str, str]]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["kind"]][row["variant"]].append(row)

    priority_index = {name: i for i, name in enumerate(cfg["variant_priority"])}
    summary = {}
    selected = {}
    expected_per_variant = int(cfg["calibration_values_per_kind"])
    for kind in sorted(CALIBRATION_KINDS):
        variants = {}
        for name in cfg["variant_priority"]:
            items = grouped[kind][name]
            if len(items) != expected_per_variant:
                raise ValueError(f"incomplete calibration matrix for {kind}/{name}")
            json_valid = sum(bool(x["score"]["envelope_json_valid"]) for x in items)
            tool_correct = sum(bool(x["score"]["tool_name_correct"]) for x in items)
            variants[name] = {
                "cases": len(items),
                "envelope_json_valid": json_valid,
                "tool_name_correct": tool_correct,
            }
        winner = max(
            cfg["variant_priority"],
            key=lambda name: (
                variants[name]["tool_name_correct"],
                variants[name]["envelope_json_valid"],
                -priority_index[name],
            ),
        )
        selected[kind] = winner
        summary[kind] = {"variants": variants, "selected": winner}
    return summary, selected


def build_final_cases(cfg: dict, selected: dict[str, str]) -> list[dict]:
    diagnostics = list(copy_data.DIAGNOSTICS)
    if len(diagnostics) != 90:
        raise ValueError(f"expected 90 held-out values, got {len(diagnostics)}")
    counts = defaultdict(int)
    cases = []
    for kind, value, _corrupt in diagnostics:
        index = counts[kind]
        counts[kind] += 1
        tool, field = TOOL_BY_KIND[kind]
        if kind in CALIBRATION_KINDS:
            variant_name = selected[kind]
            user = _user_from_template(kind, value, cfg["prompt_variants"][variant_name])
        else:
            variant_name = "v037_schema_natural"
            user = schema_prompt._user_request(tool, value)
        cases.append({
            "id": f"matrix_target_{kind}_{index:02d}",
            "kind": kind,
            "target": value,
            "variant": variant_name,
            "expected_tool": tool,
            "argument_key": field,
            "prompt": _prompt(kind, value, user),
        })
    if set(counts) != set(copy_data.KINDS) or any(counts[kind] != 10 for kind in copy_data.KINDS):
        raise ValueError("held-out battery must remain balanced at ten cases per kind")
    return cases


def cohort(rows: list[dict], kinds: set[str]) -> dict:
    selected = [row for row in rows if row["kind"] in kinds]
    total = len(selected)
    valid = sum(bool(row["score"]["envelope_json_valid"]) for row in selected)
    tool = sum(bool(row["score"]["tool_name_correct"]) for row in selected)
    slot = sum(bool(row["score"]["slot_exact"]) for row in selected)
    return {
        "cases": total,
        "envelope_json_valid": valid,
        "envelope_json_valid_rate": valid / total if total else None,
        "tool_name_correct": tool,
        "tool_name_rate": tool / total if total else None,
        "slot_exact": slot,
        "slot_exact_rate_on_evaluable": slot / tool if tool else None,
    }


def per_kind(rows: list[dict]) -> dict:
    return {kind: cohort(rows, {kind}) for kind in copy_data.KINDS}


def summary_markdown(report: dict) -> str:
    ref = report["reference_gate"]["metrics"]
    metrics = report["baseline_gate"]["metrics"]
    weak = report["targeted_kind_summary"]
    stable = report["perfect_kind_summary"]
    lines = [
        f"# Ember v0.0.39 targeted envelope matrix: {report['status']}", "",
        "CPU calibration and held-out baseline only. No optimizer, training, GPU submission, promotion, deployment, or production integration occurred.", "",
        "| Measurement | Result |", "| --- | ---: |",
        f"| Exact v0.0.8 reference JSON | {ref['envelope_json_valid']}/{ref['cases']} ({ref['envelope_json_valid_rate']:.1%}) |",
        f"| Exact v0.0.8 reference tool name | {ref['envelope_tool_name_correct']}/{ref['cases']} ({ref['envelope_tool_name_rate']:.1%}) |",
        f"| 90-case JSON envelope | {metrics['envelope_json_valid']}/{metrics['cases']} ({metrics['envelope_json_valid_rate']:.1%}) |",
        f"| 90-case correct tool | {metrics['envelope_tool_name_correct']}/{metrics['cases']} ({metrics['envelope_tool_name_rate']:.1%}) |",
        f"| 90-case slot exact | {metrics['slot_exact']}/{metrics['slot_evaluable']} ({metrics['slot_exact_rate']:.1%}) |" if metrics["slot_exact_rate"] is not None else "| 90-case slot exact | undefined |",
        f"| targeted 50-case JSON envelope | {weak['envelope_json_valid']}/{weak['cases']} ({weak['envelope_json_valid_rate']:.1%}) |",
        f"| untouched perfect 40-case JSON envelope | {stable['envelope_json_valid']}/{stable['cases']} ({stable['envelope_json_valid_rate']:.1%}) |",
        "", "Selected prompt variant by weak kind:", "",
    ]
    for kind in sorted(report["selected_variants"]):
        lines.append(f"- `{kind}`: `{report['selected_variants'][kind]}`")
    lines += ["", f"Reference gate: {'PASS' if report['reference_gate']['passed'] else 'FAIL'}. 90-case baseline gate: {'PASS' if report['baseline_gate']['passed'] else 'FAIL'}." ]
    if report["status"] == "PASS":
        lines += ["", "The targeted matrix clears the 95% envelope baseline. This still authorizes no optimizer or GPU work; placement learning must be a separate phase."]
    else:
        lines += ["", "The envelope baseline remains below 95%. Stop here; do not start placement learning or spend GPU from this result."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("v039-results"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not os.environ.get("HF_TOKEN", "").strip():
        raise ValueError("HF_TOKEN is required to read the pinned private checkpoint")

    import torch

    torch.set_num_threads(2)
    torch.manual_seed(int(cfg["seed"]))
    torch.use_deterministic_algorithms(True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "version": "0.0.39",
        "phase": cfg["phase"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ERROR",
        "code_commit": os.environ.get("GITHUB_SHA"),
        "training_authorized": False,
        "gpu_training_authorized": False,
        "production_authorized": False,
        "prompt_reference": cfg["prompt_reference"],
        "calibration_kinds": sorted(CALIBRATION_KINDS),
        "prompt_variants": cfg["prompt_variants"],
        "tool_mapping": {kind: {"name": tool, "argument_key": field} for kind, (tool, field) in TOOL_BY_KIND.items()},
    }
    try:
        source_cfg = json.loads(base.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        if source_cfg.get("version") != "0.0.32":
            raise ValueError("unexpected source-loader configuration")
        with tempfile.TemporaryDirectory(prefix="ember-v039-") as td:
            model, tokenizer, source, _splits, source_ref = base.load_inputs(source_cfg, Path(td), torch)
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != cfg["source_version"]:
                raise ValueError("v0.0.39 must measure the pinned v0.0.31 step-479 baseline")

            reference_rows = []
            for case in control.reference_cases(cfg):
                generation = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"]))
                score = control.score_reference(case, generation["completion"])
                reference_rows.append({**case, **generation, "score": score})
                print(json.dumps({"event": "reference_case", "id": case["id"], "json_valid": score["envelope_json_valid"], "tool_name_correct": score["tool_name_correct"]}), flush=True)
            reference_gate = control.reference_summary(reference_rows, cfg)

            matrix_rows = []
            for case in build_calibration_cases(cfg):
                generation = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"]))
                score = control.score_case(case, generation["completion"])
                matrix_rows.append({**case, **generation, "score": score})
                print(json.dumps({"event": "calibration_case", "id": case["id"], "kind": case["kind"], "variant": case["variant"], "json_valid": score["envelope_json_valid"], "tool_name_correct": score["tool_name_correct"]}), flush=True)
            matrix_summary, selected_variants = summarize_matrix(matrix_rows, cfg)
            print(json.dumps({"event": "matrix_selected", "selected_variants": selected_variants}), flush=True)

            rows = []
            for case in build_final_cases(cfg, selected_variants):
                generation = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"]))
                score = control.score_case(case, generation["completion"])
                rows.append({**case, **generation, "score": score})
                print(json.dumps({"event": "heldout_case", "id": case["id"], "kind": case["kind"], "variant": case["variant"], "json_valid": score["envelope_json_valid"], "tool_name_correct": score["tool_name_correct"], "slot_exact": score["slot_exact"]}), flush=True)

            baseline_gate = control.baseline_summary(rows, cfg)
            targeted = cohort(rows, CALIBRATION_KINDS)
            perfect = cohort(rows, PERFECT_KINDS)
            kind_metrics = per_kind(rows)
            passed = reference_gate["passed"] and baseline_gate["passed"]
            report.update(
                source={
                    "repo_id": source_ref["repo_id"], "checkpoint_path": source_ref["checkpoint_path"],
                    "revision": source_ref["revision"], "checkpoint_sha256": source_ref["checkpoint_sha256"],
                    "step": source["step"], "version": source["train_config"]["version"],
                },
                reference_cases=reference_rows,
                reference_gate=reference_gate,
                calibration_cases=matrix_rows,
                calibration_summary=matrix_summary,
                selected_variants=selected_variants,
                cases=rows,
                baseline_gate=baseline_gate,
                targeted_kind_summary=targeted,
                perfect_kind_summary=perfect,
                per_kind=kind_metrics,
                status="PASS" if passed else "FAIL",
                meaning="Synthetic targeted prompt calibration followed by frozen 90-case CPU baseline; no training is authorized.",
            )
    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        if "reference_gate" in report and "baseline_gate" in report:
            (output / "summary.md").write_text(summary_markdown(report), encoding="utf-8")

    print(json.dumps({"event": "complete", "status": report["status"], "reference": report["reference_gate"]["metrics"], "baseline": report["baseline_gate"]["metrics"], "selected_variants": report["selected_variants"], "per_kind": report["per_kind"]}), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
