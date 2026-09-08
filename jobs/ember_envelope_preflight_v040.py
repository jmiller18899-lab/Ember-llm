"""CPU-only regression-safe prompt calibration for Ember v0.0.40.

Each weak web-search value kind carries forward its best previously measured
prompt as a mandatory baseline candidate. Two challengers are evaluated only on
synthetic values disjoint from the 90-case held-out battery. A challenger may
replace the baseline only if it beats it by the configured margin. The frozen
selection is then measured once on the same 90-case battery.

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

from jobs import ember_envelope_preflight_v039 as prior

base = prior.base
copy_data = prior.copy_data
control = prior.control
schema_prompt = prior.schema_prompt
DEFAULT_CONFIG = ROOT / "config/ember_envelope_preflight_v0.0.40.json"
TOOL_BY_KIND = dict(prior.TOOL_BY_KIND)
CALIBRATION_KINDS = {"short_code", "long_code", "url", "path", "mixed"}
PERFECT_KINDS = set(copy_data.KINDS) - CALIBRATION_KINDS
BASELINE_BY_KIND = {
    "short_code": "current_exact",
    "long_code": "tool_named",
    "url": "current_exact",
    "path": "prior_natural",
    "mixed": "current_exact",
}


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if path.resolve() != DEFAULT_CONFIG.resolve() or cfg.get("version") != "0.0.40":
        raise ValueError("unsupported v0.0.40 calibration configuration")
    if any(cfg.get(key) is not False for key in (
        "training_authorized", "gpu_training_authorized", "production_authorized"
    )):
        raise ValueError("v0.0.40 is CPU preflight-only; training and production authorization must be false")
    if int(cfg.get("expected_cases", 0)) != 90 or int(cfg.get("expected_reference_cases", 0)) != 4:
        raise ValueError("v0.0.40 requires 90 held-out cases and four reference cases")
    if set(cfg.get("calibration_kinds", [])) != CALIBRATION_KINDS:
        raise ValueError("v0.0.40 calibration kinds changed")
    if int(cfg.get("calibration_values_per_kind", 0)) != 8:
        raise ValueError("v0.0.40 requires eight synthetic values per weak kind")
    if int(cfg.get("replacement_margin", 0)) < 1:
        raise ValueError("v0.0.40 replacement margin must be positive")
    if set(cfg.get("challenger_templates", {})) != {"direct_query", "current_quoted"}:
        raise ValueError("v0.0.40 challenger set changed")
    for template in cfg["challenger_templates"].values():
        if "{value}" not in template:
            raise ValueError("every challenger must contain {value}")
    if set(cfg.get("historical_floor", {})) != CALIBRATION_KINDS:
        raise ValueError("historical floor must cover every weak kind")
    expected_floor = {"short_code": 8, "long_code": 7, "url": 7, "path": 9, "mixed": 9}
    if cfg["historical_floor"] != expected_floor:
        raise ValueError("historical floor changed from measured v0.0.38/v0.0.39 evidence")
    for group in ("reference_gate", "baseline_gate"):
        for key in ("minimum_envelope_json_valid_rate", "minimum_envelope_tool_name_rate"):
            threshold = float(cfg.get(group, {}).get(key, -1))
            if not 0.0 < threshold <= 1.0:
                raise ValueError(f"invalid {group} threshold: {key}")
    return cfg


def baseline_user(kind: str, value: str) -> str:
    if kind == "path":
        return schema_prompt._user_request("web_search", value)
    if BASELINE_BY_KIND[kind] == "tool_named":
        return f'Use web_search to find current information about this exact identifier: "{value}".'
    label = "URL" if kind == "url" else "identifier"
    return f'Search the web for this exact {label}: "{value}".'


def calibration_values(cfg: dict) -> dict[str, list[str]]:
    count = int(cfg["calibration_values_per_kind"])
    used = set(copy_data.HELD_OUT_VALUES)
    out: dict[str, list[str]] = {}
    for kind in sorted(CALIBRATION_KINDS):
        values: list[str] = []
        i = 0
        while len(values) < count:
            variant = i % copy_data.VARIANTS[kind]
            seed = copy_data._digest("v040-calibration", kind, variant, i)
            value = copy_data._render(kind, variant, seed)
            i += 1
            if value in used:
                continue
            used.add(value)
            values.append(value)
        out[kind] = values
    return out


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
            raise ValueError(f"weak kind unexpectedly changed tool mapping: {kind}")
        for value_index, value in enumerate(values):
            candidates = {"baseline": baseline_user(kind, value)}
            candidates.update({name: template.format(value=value) for name, template in cfg["challenger_templates"].items()})
            for candidate_name, user in candidates.items():
                cases.append({
                    "id": f"cal_{kind}_{value_index:02d}_{candidate_name}",
                    "kind": kind,
                    "target": value,
                    "candidate": candidate_name,
                    "expected_tool": tool,
                    "argument_key": field,
                    "prompt": _prompt(kind, value, user),
                })
    expected = len(CALIBRATION_KINDS) * int(cfg["calibration_values_per_kind"]) * 3
    if len(cases) != expected:
        raise ValueError(f"expected {expected} calibration cases, got {len(cases)}")
    return cases


def summarize_matrix(rows: list[dict], cfg: dict) -> tuple[dict, dict[str, str]]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["kind"]][row["candidate"]].append(row)
    selected = {}
    summary = {}
    margin = int(cfg["replacement_margin"])
    expected = int(cfg["calibration_values_per_kind"])
    candidate_order = ["direct_query", "current_quoted"]
    for kind in sorted(CALIBRATION_KINDS):
        metrics = {}
        for name in ["baseline", *candidate_order]:
            items = grouped[kind][name]
            if len(items) != expected:
                raise ValueError(f"incomplete matrix for {kind}/{name}")
            valid = sum(bool(x["score"]["envelope_json_valid"]) for x in items)
            tool = sum(bool(x["score"]["tool_name_correct"]) for x in items)
            metrics[name] = {"cases": len(items), "envelope_json_valid": valid, "tool_name_correct": tool}
        baseline = metrics["baseline"]
        winner = "baseline"
        best = max(candidate_order, key=lambda name: (metrics[name]["tool_name_correct"], metrics[name]["envelope_json_valid"], -candidate_order.index(name)))
        if (
            metrics[best]["tool_name_correct"] >= baseline["tool_name_correct"] + margin
            and metrics[best]["envelope_json_valid"] >= baseline["envelope_json_valid"] + margin
        ):
            winner = best
        selected[kind] = winner
        summary[kind] = {
            "baseline_prompt": BASELINE_BY_KIND[kind],
            "candidates": metrics,
            "selected": winner,
            "replacement_margin": margin,
        }
    return summary, selected


def final_user(kind: str, value: str, selected: dict[str, str], cfg: dict) -> tuple[str, str]:
    if kind not in CALIBRATION_KINDS:
        return "v037_schema_natural", schema_prompt._user_request(TOOL_BY_KIND[kind][0], value)
    choice = selected[kind]
    if choice == "baseline":
        return f"baseline:{BASELINE_BY_KIND[kind]}", baseline_user(kind, value)
    return choice, cfg["challenger_templates"][choice].format(value=value)


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
        variant, user = final_user(kind, value, selected, cfg)
        cases.append({
            "id": f"safe_target_{kind}_{index:02d}",
            "kind": kind,
            "target": value,
            "variant": variant,
            "expected_tool": tool,
            "argument_key": field,
            "prompt": _prompt(kind, value, user),
        })
    if set(counts) != set(copy_data.KINDS) or any(counts[kind] != 10 for kind in copy_data.KINDS):
        raise ValueError("held-out battery must remain balanced at ten cases per kind")
    return cases


def cohort(rows: list[dict], kinds: set[str]) -> dict:
    items = [row for row in rows if row["kind"] in kinds]
    total = len(items)
    valid = sum(bool(row["score"]["envelope_json_valid"]) for row in items)
    tool = sum(bool(row["score"]["tool_name_correct"]) for row in items)
    slot = sum(bool(row["score"]["slot_exact"]) for row in items)
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


def regression_gate(per_kind_metrics: dict, cfg: dict) -> dict:
    checks = {}
    for kind, floor in cfg["historical_floor"].items():
        actual = int(per_kind_metrics[kind]["envelope_json_valid"])
        checks[kind] = {"floor": int(floor), "actual": actual, "passed": actual >= int(floor)}
    return {"passed": all(x["passed"] for x in checks.values()), "checks": checks}


def summary_markdown(report: dict) -> str:
    ref = report["reference_gate"]["metrics"]
    metrics = report["baseline_gate"]["metrics"]
    lines = [
        f"# Ember v0.0.40 regression-safe envelope calibration: {report['status']}", "",
        "CPU calibration and held-out baseline only. No optimizer, training, GPU submission, promotion, deployment, or production integration occurred.", "",
        "| Measurement | Result |", "| --- | ---: |",
        f"| Exact v0.0.8 reference JSON | {ref['envelope_json_valid']}/{ref['cases']} ({ref['envelope_json_valid_rate']:.1%}) |",
        f"| Exact v0.0.8 reference tool name | {ref['envelope_tool_name_correct']}/{ref['cases']} ({ref['envelope_tool_name_rate']:.1%}) |",
        f"| 90-case JSON envelope | {metrics['envelope_json_valid']}/{metrics['cases']} ({metrics['envelope_json_valid_rate']:.1%}) |",
        f"| 90-case correct tool | {metrics['envelope_tool_name_correct']}/{metrics['cases']} ({metrics['envelope_tool_name_rate']:.1%}) |",
        f"| 90-case slot exact | {metrics['slot_exact']}/{metrics['slot_evaluable']} ({metrics['slot_exact_rate']:.1%}) |" if metrics["slot_exact_rate"] is not None else "| 90-case slot exact | undefined |",
        f"| historical no-regression gate | {'PASS' if report['regression_gate']['passed'] else 'FAIL'} |",
        "", "Frozen selection by weak kind:", "",
    ]
    for kind in sorted(report["selected_candidates"]):
        lines.append(f"- `{kind}`: `{report['selected_candidates'][kind]}`")
    lines += ["", f"Reference gate: {'PASS' if report['reference_gate']['passed'] else 'FAIL'}. Global 90-case gate: {'PASS' if report['baseline_gate']['passed'] else 'FAIL'}. No-regression gate: {'PASS' if report['regression_gate']['passed'] else 'FAIL'}." ]
    if report["status"] == "PASS":
        lines += ["", "The regression-safe calibration clears the envelope baseline without falling below any historical per-kind floor. This still authorizes no training; placement learning must be a separate phase."]
    else:
        lines += ["", "At least one required CPU gate failed. Stop here; do not start placement learning or spend GPU from this result."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("v040-results"))
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
        "version": "0.0.40",
        "phase": cfg["phase"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ERROR",
        "code_commit": os.environ.get("GITHUB_SHA"),
        "training_authorized": False,
        "gpu_training_authorized": False,
        "production_authorized": False,
        "historical_floor": cfg["historical_floor"],
        "baseline_by_kind": BASELINE_BY_KIND,
        "challenger_templates": cfg["challenger_templates"],
    }
    try:
        source_cfg = json.loads(base.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        if source_cfg.get("version") != "0.0.32":
            raise ValueError("unexpected source-loader configuration")
        with tempfile.TemporaryDirectory(prefix="ember-v040-") as td:
            model, tokenizer, source, _splits, source_ref = base.load_inputs(source_cfg, Path(td), torch)
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != cfg["source_version"]:
                raise ValueError("v0.0.40 must measure the pinned v0.0.31 step-479 baseline")

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
                print(json.dumps({"event": "calibration_case", "id": case["id"], "kind": case["kind"], "candidate": case["candidate"], "json_valid": score["envelope_json_valid"], "tool_name_correct": score["tool_name_correct"]}), flush=True)
            matrix_summary, selected_candidates = summarize_matrix(matrix_rows, cfg)
            print(json.dumps({"event": "matrix_selected", "selected_candidates": selected_candidates}), flush=True)

            rows = []
            for case in build_final_cases(cfg, selected_candidates):
                generation = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"]))
                score = control.score_case(case, generation["completion"])
                rows.append({**case, **generation, "score": score})
                print(json.dumps({"event": "heldout_case", "id": case["id"], "kind": case["kind"], "variant": case["variant"], "json_valid": score["envelope_json_valid"], "tool_name_correct": score["tool_name_correct"], "slot_exact": score["slot_exact"]}), flush=True)

            baseline_gate = control.baseline_summary(rows, cfg)
            kind_metrics = per_kind(rows)
            regressions = regression_gate(kind_metrics, cfg)
            passed = reference_gate["passed"] and baseline_gate["passed"] and regressions["passed"]
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
                selected_candidates=selected_candidates,
                cases=rows,
                baseline_gate=baseline_gate,
                per_kind=kind_metrics,
                regression_gate=regressions,
                status="PASS" if passed else "FAIL",
                meaning="Regression-safe CPU prompt calibration only; no training is authorized.",
            )
    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        if "reference_gate" in report and "baseline_gate" in report and "regression_gate" in report:
            (output / "summary.md").write_text(summary_markdown(report), encoding="utf-8")

    print(json.dumps({"event": "complete", "status": report["status"], "reference": report["reference_gate"]["metrics"], "baseline": report["baseline_gate"]["metrics"], "selected_candidates": report["selected_candidates"], "per_kind": report["per_kind"], "regression_gate": report["regression_gate"]}), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
