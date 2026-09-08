"""CPU-only structural-subtype envelope calibration for Ember v0.0.41.

This experiment starts from the v0.0.40 regression-safe prompt stack. Only
structural subtypes that contained held-out envelope failures in v0.0.40 are
eligible for prompt replacement. Every subtype carries the v0.0.40 prompt as a
mandatory baseline candidate, challengers are selected on synthetic disjoint
values only, and both kind-level and subtype-level no-regression gates protect
the held-out floor.

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
DEFAULT_CONFIG = ROOT / "config/ember_envelope_preflight_v0.0.41.json"
TOOL_BY_KIND = dict(prior.TOOL_BY_KIND)
WEAK_KINDS = set(prior.CALIBRATION_KINDS)
PERFECT_KINDS = set(copy_data.KINDS) - WEAK_KINDS

SUBTYPE_VARIANT = {
    "short_code/len4": ("short_code", 0),
    "short_code/len5": ("short_code", 1),
    "long_code/4x4": ("long_code", 0),
    "long_code/3x5": ("long_code", 1),
    "url/one_mixed": ("url", 1),
    "url/two_segment": ("url", 0),
    "path/plain_leaf": ("path", 1),
    "mixed/upper": ("mixed", 2),
}
TARGET_SUBTYPES = set(SUBTYPE_VARIANT)

EXPECTED_KIND_FLOOR = {
    "short_code": 8,
    "long_code": 7,
    "url": 7,
    "path": 9,
    "mixed": 9,
}
EXPECTED_SUBTYPE_FLOOR = {
    "short_code/len4": 5,
    "short_code/len5": 3,
    "long_code/4x4": 5,
    "long_code/3x5": 2,
    "url/one_mixed": 2,
    "url/two_segment": 2,
    "path/plain_leaf": 3,
    "mixed/upper": 2,
}


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if path.resolve() != DEFAULT_CONFIG.resolve() or cfg.get("version") != "0.0.41":
        raise ValueError("unsupported v0.0.41 structural calibration configuration")
    if any(cfg.get(key) is not False for key in (
        "training_authorized", "gpu_training_authorized", "production_authorized"
    )):
        raise ValueError("v0.0.41 is CPU preflight-only; training and production authorization must be false")
    if int(cfg.get("expected_cases", 0)) != 90 or int(cfg.get("expected_reference_cases", 0)) != 4:
        raise ValueError("v0.0.41 requires 90 held-out cases and four reference cases")
    if int(cfg.get("calibration_values_per_subtype", 0)) != 8:
        raise ValueError("v0.0.41 requires eight synthetic values per residual subtype")
    if int(cfg.get("replacement_margin", 0)) < 1:
        raise ValueError("v0.0.41 replacement margin must be positive")
    if set(cfg.get("calibration_subtypes", [])) != TARGET_SUBTYPES:
        raise ValueError("v0.0.41 calibration subtypes changed")
    if set(cfg.get("challenger_templates", {})) != {"literal_query", "json_exact", "quoted_text"}:
        raise ValueError("v0.0.41 challenger set changed")
    for template in cfg["challenger_templates"].values():
        if "{value}" not in template:
            raise ValueError("every challenger template must contain {value}")
    if cfg.get("historical_floor") != EXPECTED_KIND_FLOOR:
        raise ValueError("v0.0.41 kind floors changed from v0.0.40 evidence")
    if cfg.get("historical_subtype_floor") != EXPECTED_SUBTYPE_FLOOR:
        raise ValueError("v0.0.41 subtype floors changed from v0.0.40 evidence")
    for group in ("reference_gate", "baseline_gate"):
        for key in ("minimum_envelope_json_valid_rate", "minimum_envelope_tool_name_rate"):
            threshold = float(cfg.get(group, {}).get(key, -1))
            if not 0.0 < threshold <= 1.0:
                raise ValueError(f"invalid {group} threshold: {key}")
    return cfg


def subtype_for(kind: str, value: str) -> str:
    if kind == "short_code":
        name = f"len{len(value)}"
    elif kind == "long_code":
        left, right = value.split("-", 1)
        name = f"{len(left)}x{len(right)}"
    elif kind == "url":
        parts = [part for part in urlsplit(value).path.split("/") if part]
        if len(parts) == 2:
            name = "two_segment"
        elif len(parts) == 1:
            name = "one_mixed" if parts[0] and parts[0][0].islower() else "one_upper"
        else:
            raise ValueError(f"unsupported URL structure: {value}")
    elif kind == "path":
        leaf = Path(value).name
        stem = leaf[:-5] if leaf.endswith(".json") else leaf
        if re.fullmatch(r"result-[A-Za-z0-9]+", stem):
            name = "result_code"
        elif re.fullmatch(r"[A-Za-z]+-\d{2}", stem):
            name = "leaf_numeric"
        else:
            name = "plain_leaf"
    elif kind == "mixed":
        body = value.split("_", 1)[1].rsplit("-", 1)[0]
        letters = [ch for ch in body if ch.isalpha()]
        if letters and all(ch.islower() for ch in letters):
            name = "lower"
        elif letters and all(ch.isupper() for ch in letters):
            name = "upper"
        else:
            name = "mixedcase"
    else:
        return f"{kind}/default"
    return f"{kind}/{name}"


def safe_user(kind: str, value: str) -> tuple[str, str]:
    """Return the strongest v0.0.40 held-out prompt for this kind."""
    if kind in PERFECT_KINDS:
        return "v037_schema_natural", schema_prompt._user_request(TOOL_BY_KIND[kind][0], value)
    if kind == "url":
        return "v040:current_quoted", f'Use web_search to find current information about "{value}".'
    return f"v040:{prior.BASELINE_BY_KIND[kind]}", prior.baseline_user(kind, value)


def _prompt(kind: str, user_request: str) -> str:
    tool, field = TOOL_BY_KIND[kind]
    return (
        f"<|system|>\n{schema_prompt.schema_system(tool, field)}\n"
        f"<|user|>\n{user_request}\n"
        "<|assistant|>\n"
    )


def calibration_values(cfg: dict) -> dict[str, list[str]]:
    count = int(cfg["calibration_values_per_subtype"])
    used = set(copy_data.HELD_OUT_VALUES)
    out: dict[str, list[str]] = {}
    for subtype in sorted(TARGET_SUBTYPES):
        kind, variant = SUBTYPE_VARIANT[subtype]
        values: list[str] = []
        i = 0
        while len(values) < count:
            seed = copy_data._digest("v041-calibration", subtype, i)
            value = copy_data._render(kind, variant, seed)
            i += 1
            if value in used:
                continue
            if subtype_for(kind, value) != subtype:
                continue
            used.add(value)
            values.append(value)
        out[subtype] = values
    return out


def build_calibration_cases(cfg: dict) -> list[dict]:
    cases = []
    for subtype, values in calibration_values(cfg).items():
        kind, _variant = SUBTYPE_VARIANT[subtype]
        tool, field = TOOL_BY_KIND[kind]
        if (tool, field) != ("web_search", "query"):
            raise ValueError(f"targeted subtype unexpectedly changed tool mapping: {subtype}")
        for value_index, value in enumerate(values):
            _baseline_id, baseline_request = safe_user(kind, value)
            candidates = {"baseline": baseline_request}
            candidates.update({
                name: template.format(value=value)
                for name, template in cfg["challenger_templates"].items()
            })
            for candidate_name, user_request in candidates.items():
                cases.append({
                    "id": f"cal_{subtype.replace('/', '_')}_{value_index:02d}_{candidate_name}",
                    "kind": kind,
                    "subtype": subtype,
                    "target": value,
                    "candidate": candidate_name,
                    "expected_tool": tool,
                    "argument_key": field,
                    "prompt": _prompt(kind, user_request),
                })
    expected = len(TARGET_SUBTYPES) * int(cfg["calibration_values_per_subtype"]) * 4
    if len(cases) != expected:
        raise ValueError(f"expected {expected} calibration cases, got {len(cases)}")
    return cases


def summarize_matrix(rows: list[dict], cfg: dict) -> tuple[dict, dict[str, str]]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["subtype"]][row["candidate"]].append(row)

    expected = int(cfg["calibration_values_per_subtype"])
    margin = int(cfg["replacement_margin"])
    challenger_order = ["literal_query", "json_exact", "quoted_text"]
    summary = {}
    selected = {}
    for subtype in sorted(TARGET_SUBTYPES):
        metrics = {}
        for name in ["baseline", *challenger_order]:
            items = grouped[subtype][name]
            if len(items) != expected:
                raise ValueError(f"incomplete calibration matrix for {subtype}/{name}")
            valid = sum(bool(item["score"]["envelope_json_valid"]) for item in items)
            tool = sum(bool(item["score"]["tool_name_correct"]) for item in items)
            metrics[name] = {
                "cases": len(items),
                "envelope_json_valid": valid,
                "tool_name_correct": tool,
            }
        baseline = metrics["baseline"]
        best = max(
            challenger_order,
            key=lambda name: (
                metrics[name]["tool_name_correct"],
                metrics[name]["envelope_json_valid"],
                -challenger_order.index(name),
            ),
        )
        winner = "baseline"
        if (
            metrics[best]["tool_name_correct"] >= baseline["tool_name_correct"] + margin
            and metrics[best]["envelope_json_valid"] >= baseline["envelope_json_valid"] + margin
        ):
            winner = best
        selected[subtype] = winner
        summary[subtype] = {
            "candidates": metrics,
            "selected": winner,
            "replacement_margin": margin,
        }
    return summary, selected


def final_user(kind: str, value: str, selected: dict[str, str], cfg: dict) -> tuple[str, str]:
    subtype = subtype_for(kind, value)
    baseline_id, baseline_request = safe_user(kind, value)
    if subtype not in TARGET_SUBTYPES:
        return baseline_id, baseline_request
    choice = selected[subtype]
    if choice == "baseline":
        return f"{baseline_id}|{subtype}", baseline_request
    return f"{choice}|{subtype}", cfg["challenger_templates"][choice].format(value=value)

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
        subtype = subtype_for(kind, value)
        variant, user_request = final_user(kind, value, selected, cfg)
        cases.append({
            "id": f"subtype_target_{kind}_{index:02d}",
            "kind": kind,
            "subtype": subtype,
            "target": value,
            "variant": variant,
            "expected_tool": tool,
            "argument_key": field,
            "prompt": _prompt(kind, user_request),
        })
    if set(counts) != set(copy_data.KINDS) or any(counts[kind] != 10 for kind in copy_data.KINDS):
        raise ValueError("held-out battery must remain balanced at ten cases per kind")
    return cases


def cohort(rows: list[dict]) -> dict:
    total = len(rows)
    valid = sum(bool(row["score"]["envelope_json_valid"]) for row in rows)
    tool = sum(bool(row["score"]["tool_name_correct"]) for row in rows)
    slot = sum(bool(row["score"]["slot_exact"]) for row in rows)
    return {
        "cases": total,
        "envelope_json_valid": valid,
        "envelope_json_valid_rate": valid / total if total else None,
        "tool_name_correct": tool,
        "tool_name_rate": tool / total if total else None,
        "slot_exact": slot,
        "slot_exact_rate_on_evaluable": slot / tool if tool else None,
    }


def per_subtype(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["subtype"]].append(row)
    return {key: cohort(items) for key, items in sorted(grouped.items())}


def subtype_regression_gate(subtype_metrics: dict, cfg: dict) -> dict:
    checks = {}
    for subtype, floor in cfg["historical_subtype_floor"].items():
        actual = int(subtype_metrics[subtype]["envelope_json_valid"])
        checks[subtype] = {
            "floor": int(floor),
            "actual": actual,
            "passed": actual >= int(floor),
        }
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks}


def summary_markdown(report: dict) -> str:
    ref = report["reference_gate"]["metrics"]
    metrics = report["baseline_gate"]["metrics"]
    lines = [
        f"# Ember v0.0.41 structural-subtype envelope calibration: {report['status']}", "",
        "CPU calibration and held-out baseline only. No optimizer, training, GPU submission, promotion, deployment, or production integration occurred.", "",
        "| Measurement | Result |", "| --- | ---: |",
        f"| Exact v0.0.8 reference JSON | {ref['envelope_json_valid']}/{ref['cases']} ({ref['envelope_json_valid_rate']:.1%}) |",
        f"| Exact v0.0.8 reference tool name | {ref['envelope_tool_name_correct']}/{ref['cases']} ({ref['envelope_tool_name_rate']:.1%}) |",
        f"| 90-case JSON envelope | {metrics['envelope_json_valid']}/{metrics['cases']} ({metrics['envelope_json_valid_rate']:.1%}) |",
        f"| 90-case correct tool | {metrics['envelope_tool_name_correct']}/{metrics['cases']} ({metrics['envelope_tool_name_rate']:.1%}) |",
        f"| 90-case slot exact | {metrics['slot_exact']}/{metrics['slot_evaluable']} ({metrics['slot_exact_rate']:.1%}) |" if metrics["slot_exact_rate"] is not None else "| 90-case slot exact | undefined |",
        f"| kind no-regression gate | {'PASS' if report['kind_regression_gate']['passed'] else 'FAIL'} |",
        f"| subtype no-regression gate | {'PASS' if report['subtype_regression_gate']['passed'] else 'FAIL'} |",
        "", "Frozen selection by residual subtype:", "",
    ]
    for subtype in sorted(report["selected_candidates"]):
        lines.append(f"- `{subtype}`: `{report['selected_candidates'][subtype]}`")
    lines += [
        "",
        f"Reference gate: {'PASS' if report['reference_gate']['passed'] else 'FAIL'}. "
        f"Global 90-case gate: {'PASS' if report['baseline_gate']['passed'] else 'FAIL'}. "
        f"Kind no-regression gate: {'PASS' if report['kind_regression_gate']['passed'] else 'FAIL'}. "
        f"Subtype no-regression gate: {'PASS' if report['subtype_regression_gate']['passed'] else 'FAIL'}.",
    ]
    if report["status"] == "PASS":
        lines += ["", "The structural-subtype calibration clears the 95% envelope baseline without regressing any protected kind or residual subtype. This still authorizes no training; placement learning must be a separate phase."]
    else:
        lines += ["", "At least one required CPU gate failed. Stop here; do not start placement learning or spend GPU from this result."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("v041-results"))
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
        "version": "0.0.41",
        "phase": cfg["phase"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ERROR",
        "code_commit": os.environ.get("GITHUB_SHA"),
        "training_authorized": False,
        "gpu_training_authorized": False,
        "production_authorized": False,
        "historical_floor": cfg["historical_floor"],
        "historical_subtype_floor": cfg["historical_subtype_floor"],
        "calibration_subtypes": sorted(TARGET_SUBTYPES),
        "challenger_templates": cfg["challenger_templates"],
    }
    try:
        source_cfg = json.loads(base.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        if source_cfg.get("version") != "0.0.32":
            raise ValueError("unexpected source-loader configuration")
        with tempfile.TemporaryDirectory(prefix="ember-v041-") as td:
            model, tokenizer, source, _splits, source_ref = base.load_inputs(source_cfg, Path(td), torch)
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != cfg["source_version"]:
                raise ValueError("v0.0.41 must measure the pinned v0.0.31 step-479 baseline")

            reference_rows = []
            for case in control.reference_cases(cfg):
                generation = base.semantic_gate.generate_completion(
                    model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"])
                )
                score = control.score_reference(case, generation["completion"])
                reference_rows.append({**case, **generation, "score": score})
                print(json.dumps({
                    "event": "reference_case",
                    "id": case["id"],
                    "json_valid": score["envelope_json_valid"],
                    "tool_name_correct": score["tool_name_correct"],
                }), flush=True)
            reference_gate = control.reference_summary(reference_rows, cfg)

            calibration_rows = []
            for case in build_calibration_cases(cfg):
                generation = base.semantic_gate.generate_completion(
                    model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"])
                )
                score = control.score_case(case, generation["completion"])
                calibration_rows.append({**case, **generation, "score": score})
                print(json.dumps({
                    "event": "calibration_case",
                    "id": case["id"],
                    "subtype": case["subtype"],
                    "candidate": case["candidate"],
                    "json_valid": score["envelope_json_valid"],
                    "tool_name_correct": score["tool_name_correct"],
                }), flush=True)
            calibration_summary, selected_candidates = summarize_matrix(calibration_rows, cfg)
            print(json.dumps({
                "event": "matrix_selected",
                "selected_candidates": selected_candidates,
            }), flush=True)

            rows = []
            for case in build_final_cases(cfg, selected_candidates):
                generation = base.semantic_gate.generate_completion(
                    model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"])
                )
                score = control.score_case(case, generation["completion"])
                rows.append({**case, **generation, "score": score})
                print(json.dumps({
                    "event": "heldout_case",
                    "id": case["id"],
                    "kind": case["kind"],
                    "subtype": case["subtype"],
                    "variant": case["variant"],
                    "json_valid": score["envelope_json_valid"],
                    "tool_name_correct": score["tool_name_correct"],
                    "slot_exact": score["slot_exact"],
                }), flush=True)

            baseline_gate = control.baseline_summary(rows, cfg)
            kind_metrics = prior.per_kind(rows)
            subtype_metrics = per_subtype(rows)
            kind_gate = prior.regression_gate(kind_metrics, cfg)
            subtype_gate = subtype_regression_gate(subtype_metrics, cfg)
            passed = (
                reference_gate["passed"]
                and baseline_gate["passed"]
                and kind_gate["passed"]
                and subtype_gate["passed"]
            )
            report.update(
                source={
                    "repo_id": source_ref["repo_id"],
                    "checkpoint_path": source_ref["checkpoint_path"],
                    "revision": source_ref["revision"],
                    "checkpoint_sha256": source_ref["checkpoint_sha256"],
                    "step": source["step"],
                    "version": source["train_config"]["version"],
                },
                reference_cases=reference_rows,
                reference_gate=reference_gate,
                calibration_cases=calibration_rows,
                calibration_summary=calibration_summary,
                selected_candidates=selected_candidates,
                cases=rows,
                baseline_gate=baseline_gate,
                per_kind=kind_metrics,
                per_subtype=subtype_metrics,
                kind_regression_gate=kind_gate,
                subtype_regression_gate=subtype_gate,
                status="PASS" if passed else "FAIL",
                meaning="Structural-subtype envelope calibration only; no training is authorized.",
            )
    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        (output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if "reference_gate" in report and "baseline_gate" in report:
            (output / "summary.md").write_text(summary_markdown(report), encoding="utf-8")

    print(json.dumps({
        "event": "complete",
        "status": report["status"],
        "reference": report["reference_gate"]["metrics"],
        "baseline": report["baseline_gate"]["metrics"],
        "selected_candidates": report["selected_candidates"],
        "kind_regression_gate": report["kind_regression_gate"],
        "subtype_regression_gate": report["subtype_regression_gate"],
    }), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())