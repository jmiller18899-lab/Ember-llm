"""CPU-only three-fold residual envelope calibration for Ember v0.0.43.

Starts from the strongest non-regressing v0.0.42 prompt stack (84/90), freezes
URL and mixed gains, and calibrates only the five structural subtypes that still
contain held-out envelope failures. A challenger must be non-regressing on all
three synthetic folds, win on at least two folds, and gain at least two cases
combined before it may replace the frozen safe prompt.

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

from jobs import ember_envelope_preflight_v042 as prior

base = prior.base
copy_data = prior.copy_data
control = prior.control
schema_prompt = prior.schema_prompt
TOOL_BY_KIND = dict(prior.TOOL_BY_KIND)
DEFAULT_CONFIG = ROOT / "config/ember_envelope_preflight_v0.0.43.json"
V042_CFG = json.loads(prior.DEFAULT_CONFIG.read_text(encoding="utf-8"))

TARGET_SUBTYPES = {
    "short_code/len4",
    "short_code/len5",
    "long_code/4x4",
    "long_code/3x5",
    "path/plain_leaf",
}
SUBTYPE_VARIANT = {key: prior.SUBTYPE_VARIANT[key] for key in TARGET_SUBTYPES}
LABEL_BY_KIND = {
    "short_code": "code",
    "long_code": "code",
    "path": "file path",
}
EXPECTED_KIND_FLOOR = {
    "short_code": 8,
    "long_code": 7,
    "url": 10,
    "path": 9,
    "mixed": 10,
}
EXPECTED_SUBTYPE_FLOOR = {
    "short_code/len4": 5,
    "short_code/len5": 3,
    "long_code/4x4": 5,
    "long_code/3x5": 2,
    "url/one_mixed": 4,
    "url/two_segment": 3,
    "path/plain_leaf": 3,
    "mixed/upper": 3,
}


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if path.resolve() != DEFAULT_CONFIG.resolve() or cfg.get("version") != "0.0.43":
        raise ValueError("unsupported v0.0.43 configuration")
    if any(cfg.get(k) is not False for k in (
        "training_authorized", "gpu_training_authorized", "production_authorized"
    )):
        raise ValueError("v0.0.43 is CPU preflight-only")
    if int(cfg.get("expected_cases", 0)) != 90 or int(cfg.get("expected_reference_cases", 0)) != 4:
        raise ValueError("v0.0.43 requires 90 held-out cases and four references")
    if cfg.get("calibration_folds") != ["select", "confirm", "stress"]:
        raise ValueError("v0.0.43 requires select, confirm, and stress folds")
    if set(cfg.get("calibration_subtypes", [])) != TARGET_SUBTYPES:
        raise ValueError("v0.0.43 target subtypes changed")
    if int(cfg.get("calibration_values_per_fold", 0)) != 6:
        raise ValueError("v0.0.43 requires six synthetic values per fold")
    if set(cfg.get("challenger_templates", {})) != {"lookup_exact", "search_literal", "web_lookup"}:
        raise ValueError("v0.0.43 challenger set changed")
    if int(cfg.get("minimum_winning_folds", 0)) < 2:
        raise ValueError("v0.0.43 requires at least two winning folds")
    if int(cfg.get("minimum_combined_gain", 0)) < 2:
        raise ValueError("v0.0.43 requires at least two combined gains")
    if cfg.get("historical_floor") != EXPECTED_KIND_FLOOR:
        raise ValueError("v0.0.43 kind floors changed from v0.0.42 evidence")
    if cfg.get("historical_subtype_floor") != EXPECTED_SUBTYPE_FLOOR:
        raise ValueError("v0.0.43 subtype floors changed from v0.0.42 evidence")
    for template in cfg["challenger_templates"].values():
        if "{value}" not in template or "{label}" not in template:
            raise ValueError("every challenger must contain {label} and {value}")
    for group in ("reference_gate", "baseline_gate"):
        for key in ("minimum_envelope_json_valid_rate", "minimum_envelope_tool_name_rate"):
            threshold = float(cfg.get(group, {}).get(key, -1))
            if not 0.0 < threshold <= 1.0:
                raise ValueError(f"invalid {group} threshold: {key}")
    return cfg


def subtype_for(kind: str, value: str) -> str:
    return prior.subtype_for(kind, value)


def frozen_safe_user(kind: str, value: str) -> tuple[str, str]:
    """Assemble the strongest non-regressing v0.0.42 held-out prompt stack."""
    subtype = subtype_for(kind, value)
    if subtype == "url/two_segment":
        template = V042_CFG["challenger_templates"]["tool_query"]
        return "v042:tool_query|url/two_segment", template.format(label="URL", value=value)
    return prior.assembled_safe_user(kind, value, V042_CFG)


def _prompt(kind: str, user_request: str) -> str:
    tool, field = TOOL_BY_KIND[kind]
    return (
        f"<|system|>\n{schema_prompt.schema_system(tool, field)}\n"
        f"<|user|>\n{user_request}\n"
        "<|assistant|>\n"
    )


def calibration_values(cfg: dict) -> dict[str, dict[str, list[str]]]:
    count = int(cfg["calibration_values_per_fold"])
    used = set(copy_data.HELD_OUT_VALUES)
    out: dict[str, dict[str, list[str]]] = {}
    for subtype in sorted(TARGET_SUBTYPES):
        kind, variant = SUBTYPE_VARIANT[subtype]
        out[subtype] = {}
        for fold in cfg["calibration_folds"]:
            values: list[str] = []
            i = 0
            while len(values) < count:
                seed = copy_data._digest("v043-calibration", fold, subtype, i)
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
    values_by_subtype = calibration_values(cfg)
    for subtype, folds in values_by_subtype.items():
        kind, _variant = SUBTYPE_VARIANT[subtype]
        tool, field = TOOL_BY_KIND[kind]
        if (tool, field) != ("web_search", "query"):
            raise ValueError(f"target subtype changed tool mapping: {subtype}")
        for fold, values in folds.items():
            for value_index, value in enumerate(values):
                _baseline_id, baseline_request = frozen_safe_user(kind, value)
                candidates = {"baseline": baseline_request}
                candidates.update({
                    name: _challenger_user(kind, value, template)
                    for name, template in cfg["challenger_templates"].items()
                })
                for candidate, user_request in candidates.items():
                    cases.append({
                        "id": f"cal_{subtype.replace('/', '_')}_{fold}_{value_index:02d}_{candidate}",
                        "kind": kind,
                        "subtype": subtype,
                        "fold": fold,
                        "target": value,
                        "candidate": candidate,
                        "expected_tool": tool,
                        "argument_key": field,
                        "prompt": _prompt(kind, user_request),
                    })
    expected = len(TARGET_SUBTYPES) * len(cfg["calibration_folds"]) * int(cfg["calibration_values_per_fold"]) * 4
    if len(cases) != expected:
        raise ValueError(f"expected {expected} calibration cases, got {len(cases)}")
    return cases


def summarize_matrix(rows: list[dict], cfg: dict) -> tuple[dict, dict[str, str]]:
    grouped: dict[str, dict[str, dict[str, list[dict]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        grouped[row["subtype"]][row["fold"]][row["candidate"]].append(row)

    candidate_order = ["lookup_exact", "search_literal", "web_lookup"]
    expected = int(cfg["calibration_values_per_fold"])
    minimum_winning_folds = int(cfg["minimum_winning_folds"])
    minimum_combined_gain = int(cfg["minimum_combined_gain"])
    summary = {}
    selected = {}

    for subtype in sorted(TARGET_SUBTYPES):
        metrics = {}
        for fold in cfg["calibration_folds"]:
            metrics[fold] = {}
            for name in ["baseline", *candidate_order]:
                items = grouped[subtype][fold][name]
                if len(items) != expected:
                    raise ValueError(f"incomplete matrix for {subtype}/{fold}/{name}")
                valid = sum(bool(x["score"]["envelope_json_valid"]) for x in items)
                tool = sum(bool(x["score"]["tool_name_correct"]) for x in items)
                metrics[fold][name] = {
                    "cases": len(items),
                    "envelope_json_valid": valid,
                    "tool_name_correct": tool,
                }

        eligible = []
        for name in candidate_order:
            zero_loss = True
            winning_folds = 0
            base_valid_total = base_tool_total = cand_valid_total = cand_tool_total = 0
            for fold in cfg["calibration_folds"]:
                baseline = metrics[fold]["baseline"]
                challenger = metrics[fold][name]
                base_valid_total += baseline["envelope_json_valid"]
                base_tool_total += baseline["tool_name_correct"]
                cand_valid_total += challenger["envelope_json_valid"]
                cand_tool_total += challenger["tool_name_correct"]
                if (
                    challenger["envelope_json_valid"] < baseline["envelope_json_valid"]
                    or challenger["tool_name_correct"] < baseline["tool_name_correct"]
                ):
                    zero_loss = False
                if (
                    challenger["envelope_json_valid"] > baseline["envelope_json_valid"]
                    and challenger["tool_name_correct"] > baseline["tool_name_correct"]
                ):
                    winning_folds += 1
            valid_gain = cand_valid_total - base_valid_total
            tool_gain = cand_tool_total - base_tool_total
            if (
                zero_loss
                and winning_folds >= minimum_winning_folds
                and valid_gain >= minimum_combined_gain
                and tool_gain >= minimum_combined_gain
            ):
                eligible.append((tool_gain, valid_gain, winning_folds, -candidate_order.index(name), name))

        winner = max(eligible)[4] if eligible else "baseline"
        selected[subtype] = winner
        summary[subtype] = {
            "folds": metrics,
            "selected": winner,
            "minimum_winning_folds": minimum_winning_folds,
            "minimum_combined_gain": minimum_combined_gain,
        }
    return summary, selected


def final_user(kind: str, value: str, selected: dict[str, str], cfg: dict) -> tuple[str, str]:
    subtype = subtype_for(kind, value)
    baseline_id, baseline_request = frozen_safe_user(kind, value)
    if subtype not in TARGET_SUBTYPES or selected[subtype] == "baseline":
        return f"{baseline_id}|{subtype}", baseline_request
    choice = selected[subtype]
    request = _challenger_user(kind, value, cfg["challenger_templates"][choice])
    return f"{choice}|{subtype}", request


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
            "id": f"threefold_target_{kind}_{index:02d}",
            "kind": kind,
            "subtype": subtype,
            "target": value,
            "variant": variant,
            "expected_tool": tool,
            "argument_key": field,
            "prompt": _prompt(kind, user_request),
        })
    if set(counts) != set(copy_data.KINDS) or any(counts[k] != 10 for k in copy_data.KINDS):
        raise ValueError("held-out battery must remain balanced")
    return cases


def summary_markdown(report: dict) -> str:
    ref = report["reference_gate"]["metrics"]
    m = report["baseline_gate"]["metrics"]
    lines = [
        f"# Ember v0.0.43 three-fold residual calibration: {report['status']}",
        "",
        "CPU-only. No optimizer, training, GPU submission, promotion, deployment, or integration occurred.",
        "",
        "| Measurement | Result |",
        "| --- | ---: |",
        f"| Exact v0.0.8 reference JSON | {ref['envelope_json_valid']}/{ref['cases']} ({ref['envelope_json_valid_rate']:.1%}) |",
        f"| Exact v0.0.8 reference tool | {ref['envelope_tool_name_correct']}/{ref['cases']} ({ref['envelope_tool_name_rate']:.1%}) |",
        f"| 90-case JSON envelope | {m['envelope_json_valid']}/{m['cases']} ({m['envelope_json_valid_rate']:.1%}) |",
        f"| 90-case correct tool | {m['envelope_tool_name_correct']}/{m['cases']} ({m['envelope_tool_name_rate']:.1%}) |",
        f"| kind no-regression gate | {'PASS' if report['kind_regression_gate']['passed'] else 'FAIL'} |",
        f"| subtype no-regression gate | {'PASS' if report['subtype_regression_gate']['passed'] else 'FAIL'} |",
        "",
        "Frozen selection:",
        "",
    ]
    lines += [f"- `{k}`: `{v}`" for k, v in sorted(report["selected_candidates"].items())]
    lines += [
        "",
        "A PASS validates only the envelope baseline for a separate placement-learning phase; it does not authorize training automatically.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("v043-results"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not os.environ.get("HF_TOKEN", "").strip():
        raise ValueError("HF_TOKEN is required")

    import torch

    torch.set_num_threads(2)
    torch.manual_seed(int(cfg["seed"]))
    torch.use_deterministic_algorithms(True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "version": "0.0.43",
        "phase": cfg["phase"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ERROR",
        "code_commit": os.environ.get("GITHUB_SHA"),
        "training_authorized": False,
        "gpu_training_authorized": False,
        "production_authorized": False,
        "historical_floor": cfg["historical_floor"],
        "historical_subtype_floor": cfg["historical_subtype_floor"],
    }

    try:
        source_cfg = json.loads(base.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="ember-v043-") as td:
            model, tokenizer, source, _splits, source_ref = base.load_inputs(source_cfg, Path(td), torch)
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != cfg["source_version"]:
                raise ValueError("wrong source checkpoint")

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
                    "subtype": case["subtype"],
                    "fold": case["fold"],
                    "candidate": case["candidate"],
                    "json_valid": score["envelope_json_valid"],
                    "tool_name_correct": score["tool_name_correct"],
                }), flush=True)

            calibration_summary, selected = summarize_matrix(calibration_rows, cfg)
            print(json.dumps({"event": "matrix_selected", "selected_candidates": selected}), flush=True)

            rows = []
            for case in build_final_cases(cfg, selected):
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
            subtype_metrics = prior.per_subtype(rows)
            kind_gate = prior.regression_gate(kind_metrics, cfg["historical_floor"])
            subtype_gate = prior.regression_gate(subtype_metrics, cfg["historical_subtype_floor"])
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
                selected_candidates=selected,
                cases=rows,
                baseline_gate=baseline_gate,
                per_kind=kind_metrics,
                per_subtype=subtype_metrics,
                kind_regression_gate=kind_gate,
                subtype_regression_gate=subtype_gate,
                status="PASS" if passed else "FAIL",
                meaning="Envelope calibration only; no training is authorized by this runner.",
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
        "baseline": report["baseline_gate"]["metrics"],
        "selected_candidates": report["selected_candidates"],
        "kind_regression_gate": report["kind_regression_gate"],
        "subtype_regression_gate": report["subtype_regression_gate"],
    }), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
