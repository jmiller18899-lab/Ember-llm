"""CPU-only code-shaped envelope calibration for Ember v0.0.38.

This measures the pinned v0.0.31 step-479 checkpoint only. Relative to v0.0.37,
only short_code and long_code user wording changes. No optimizer, model write,
GPU submission, promotion, or production integration is possible here.
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

from jobs import ember_envelope_preflight_v037 as prior

base = prior.base
copy_data = prior.copy_data
DEFAULT_CONFIG = ROOT / "config/ember_envelope_preflight_v0.0.38.json"
TOOL_BY_KIND = dict(prior.TOOL_BY_KIND)
CALIBRATION_KINDS = {"short_code", "long_code"}


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if path.resolve() != DEFAULT_CONFIG.resolve() or cfg.get("version") != "0.0.38":
        raise ValueError("unsupported v0.0.38 envelope calibration configuration")
    if any(cfg.get(key) is not False for key in (
        "training_authorized", "gpu_training_authorized", "production_authorized"
    )):
        raise ValueError("v0.0.38 is CPU preflight-only; training and production authorization must be false")
    if int(cfg.get("expected_cases", 0)) != 90 or int(cfg.get("expected_reference_cases", 0)) != 4:
        raise ValueError("v0.0.38 requires 90 held-out cases and four reference cases")
    if set(cfg.get("calibration_kinds", [])) != CALIBRATION_KINDS:
        raise ValueError("v0.0.38 calibration kinds changed")
    if cfg.get("code_user_template") != 'Search the web for this exact identifier: "{value}".':
        raise ValueError("v0.0.38 code prompt template changed")
    for group in ("reference_gate", "baseline_gate"):
        for key in ("minimum_envelope_json_valid_rate", "minimum_envelope_tool_name_rate"):
            threshold = float(cfg.get(group, {}).get(key, -1))
            if not 0.0 < threshold <= 1.0:
                raise ValueError(f"invalid {group} threshold: {key}")
    return cfg


def user_request(kind: str, tool: str, target: str, cfg: dict) -> str:
    if kind in CALIBRATION_KINDS:
        return cfg["code_user_template"].format(value=target)
    return prior._user_request(tool, target)


def build_cases(cfg: dict) -> list[dict]:
    diagnostics = list(copy_data.DIAGNOSTICS)
    if len(diagnostics) != 90:
        raise ValueError(f"expected 90 held-out values, got {len(diagnostics)}")
    counts = defaultdict(int)
    cases = []
    for kind, value, _corrupt in diagnostics:
        index = counts[kind]
        counts[kind] += 1
        tool, field = TOOL_BY_KIND[kind]
        prompt = (
            f"<|system|>\n{prior.schema_system(tool, field)}\n"
            f"<|user|>\n{user_request(kind, tool, value, cfg)}\n"
            "<|assistant|>\n"
        )
        cases.append({
            "id": f"codecal_target_{kind}_{index:02d}",
            "kind": kind,
            "target": value,
            "expected_tool": tool,
            "argument_key": field,
            "prompt": prompt,
        })
    if set(counts) != set(copy_data.KINDS) or any(counts[kind] != 10 for kind in copy_data.KINDS):
        raise ValueError("held-out battery must remain balanced at ten cases per kind")
    return cases


def summary_markdown(report: dict) -> str:
    ref = report["reference_gate"]["metrics"]
    metrics = report["baseline_gate"]["metrics"]
    code = report["code_kind_summary"]
    other = report["other_kind_summary"]
    slot_text = (
        f"{metrics['slot_exact']}/{metrics['slot_evaluable']} ({metrics['slot_exact_rate']:.1%})"
        if metrics["slot_exact_rate"] is not None else "undefined"
    )
    lines = [
        f"# Ember v0.0.38 code-shaped envelope calibration: {report['status']}", "",
        "CPU baseline only. No optimizer, training, GPU submission, promotion, or production integration occurred.", "",
        "| Measurement | Result |", "| --- | ---: |",
        f"| Exact v0.0.8 reference JSON | {ref['envelope_json_valid']}/{ref['cases']} ({ref['envelope_json_valid_rate']:.1%}) |",
        f"| Exact v0.0.8 reference tool name | {ref['envelope_tool_name_correct']}/{ref['cases']} ({ref['envelope_tool_name_rate']:.1%}) |",
        f"| 90-case JSON envelope | {metrics['envelope_json_valid']}/{metrics['cases']} ({metrics['envelope_json_valid_rate']:.1%}) |",
        f"| 90-case correct tool | {metrics['envelope_tool_name_correct']}/{metrics['cases']} ({metrics['envelope_tool_name_rate']:.1%}) |",
        f"| 90-case slot exact | {slot_text} |",
        f"| short+long code JSON envelope | {code['envelope_json_valid']}/{code['cases']} ({code['envelope_json_valid_rate']:.1%}) |",
        f"| unchanged 70-case JSON envelope | {other['envelope_json_valid']}/{other['cases']} ({other['envelope_json_valid_rate']:.1%}) |",
        "",
        f"Reference gate: {'PASS' if report['reference_gate']['passed'] else 'FAIL'}. 90-case baseline gate: {'PASS' if report['baseline_gate']['passed'] else 'FAIL'}.",
    ]
    if report["status"] == "PASS":
        lines += ["", "The code-specific framing clears the 95% envelope baseline. This report still authorizes no training; a separate placement-learning phase must be designed before any optimizer or GPU use."]
    else:
        lines += ["", "The envelope baseline remains below 95%. Stop here and do not start a learning phase or spend GPU from this result."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("v038-results"))
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
        "version": "0.0.38",
        "phase": cfg["phase"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ERROR",
        "code_commit": os.environ.get("GITHUB_SHA"),
        "training_authorized": False,
        "gpu_training_authorized": False,
        "production_authorized": False,
        "prompt_reference": cfg["prompt_reference"],
        "calibration_kinds": sorted(CALIBRATION_KINDS),
        "code_user_template": cfg["code_user_template"],
        "tool_mapping": {kind: {"name": tool, "argument_key": field} for kind, (tool, field) in TOOL_BY_KIND.items()},
    }
    try:
        source_cfg = json.loads(base.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        if source_cfg.get("version") != "0.0.32":
            raise ValueError("unexpected source-loader configuration")
        with tempfile.TemporaryDirectory(prefix="ember-v038-") as td:
            model, tokenizer, source, _splits, source_ref = base.load_inputs(source_cfg, Path(td), torch)
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != cfg["source_version"]:
                raise ValueError("v0.0.38 must measure the pinned v0.0.31 step-479 baseline")

            reference_rows = []
            for case in prior.prior.reference_cases(cfg):
                generation = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"]))
                score = prior.prior.score_reference(case, generation["completion"])
                reference_rows.append({**case, **generation, "score": score})
                print(json.dumps({"event": "reference_case", "id": case["id"], "json_valid": score["envelope_json_valid"], "tool_name_correct": score["tool_name_correct"]}), flush=True)
            reference_gate = prior.prior.reference_summary(reference_rows, cfg)

            rows = []
            for case in build_cases(cfg):
                generation = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"]))
                score = prior.prior.score_case(case, generation["completion"])
                rows.append({**case, **generation, "score": score})
                print(json.dumps({"event": "case", "id": case["id"], "kind": case["kind"], "json_valid": score["envelope_json_valid"], "tool_name_correct": score["tool_name_correct"], "slot_exact": score["slot_exact"]}), flush=True)

            baseline_gate = prior.prior.baseline_summary(rows, cfg)
            code_summary = prior.cohort_summary(rows, CALIBRATION_KINDS)
            other_summary = prior.cohort_summary(rows, set(copy_data.KINDS) - CALIBRATION_KINDS)
            passed = reference_gate["passed"] and baseline_gate["passed"]
            report.update(
                source={
                    "repo_id": source_ref["repo_id"], "checkpoint_path": source_ref["checkpoint_path"],
                    "revision": source_ref["revision"], "checkpoint_sha256": source_ref["checkpoint_sha256"],
                    "step": source["step"], "version": source["train_config"]["version"],
                },
                reference_cases=reference_rows,
                reference_gate=reference_gate,
                cases=rows,
                baseline_gate=baseline_gate,
                code_kind_summary=code_summary,
                other_kind_summary=other_summary,
                status="PASS" if passed else "FAIL",
                meaning="Code-shaped user-framing calibration only; no training is authorized.",
            )
    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        if "reference_gate" in report and "baseline_gate" in report:
            (output / "summary.md").write_text(summary_markdown(report), encoding="utf-8")

    print(json.dumps({"event": "complete", "status": report["status"], "reference": report["reference_gate"]["metrics"], "baseline": report["baseline_gate"]["metrics"], "code_kinds": report["code_kind_summary"], "other_kinds": report["other_kind_summary"]}), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
