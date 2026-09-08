"""CPU-only schema-bearing envelope calibration for Ember v0.0.37.

The pinned v0.0.31 step-479 checkpoint is measured only. The four exact v0.0.8
reference prompts remain a control. The 90-case probe changes one variable from
v0.0.36: the system line uses the schema-bearing agent wording from the v0.0.32
semantic curriculum. No optimizer, model write, GPU submission, or promotion is
possible in this runner.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import ember_envelope_preflight_v036 as prior

base = prior.base
copy_data = prior.copy_data
DEFAULT_CONFIG = ROOT / "config/ember_envelope_preflight_v0.0.37.json"
TOOL_BY_KIND = dict(prior.TOOL_BY_KIND)


def schema_system(tool: str, field: str) -> str:
    return f"You are Ember. Use {tool} with JSON arguments. Its only required argument is {field}, a string."


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if path.resolve() != DEFAULT_CONFIG.resolve() or cfg.get("version") != "0.0.37":
        raise ValueError("unsupported v0.0.37 envelope preflight configuration")
    if any(cfg.get(key) is not False for key in (
        "training_authorized", "gpu_training_authorized", "production_authorized"
    )):
        raise ValueError("v0.0.37 is preflight-only; training and production authorization must be false")
    if int(cfg.get("expected_cases", 0)) != 90 or int(cfg.get("expected_reference_cases", 0)) != 4:
        raise ValueError("v0.0.37 requires 90 held-out cases and four reference cases")
    if set(cfg.get("weak_kinds", [])) != {"short_code", "long_code", "url", "path"}:
        raise ValueError("weak-kind calibration cohort changed")
    for group in ("reference_gate", "baseline_gate"):
        for key in ("minimum_envelope_json_valid_rate", "minimum_envelope_tool_name_rate"):
            threshold = float(cfg.get(group, {}).get(key, -1))
            if not 0.0 < threshold <= 1.0:
                raise ValueError(f"invalid {group} threshold: {key}")
    return cfg


def _user_request(tool: str, target: str) -> str:
    if tool == "weather":
        return f"What is the weather in {target} right now?"
    if tool == "calculator":
        return f"Calculate {target}."
    if tool == "web_search":
        return f"Find current information about {target}."
    raise ValueError(tool)


def build_cases() -> list[dict]:
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
            f"<|system|>\n{schema_system(tool, field)}\n"
            f"<|user|>\n{_user_request(tool, value)}\n"
            "<|assistant|>\n"
        )
        cases.append({
            "id": f"schema_target_{kind}_{index:02d}",
            "kind": kind,
            "target": value,
            "expected_tool": tool,
            "argument_key": field,
            "prompt": prompt,
        })
    if set(counts) != set(copy_data.KINDS) or any(counts[kind] != 10 for kind in copy_data.KINDS):
        raise ValueError("held-out battery must remain balanced at ten cases per kind")
    return cases


def cohort_summary(rows: list[dict], kinds: set[str]) -> dict:
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


def summary_markdown(report: dict) -> str:
    ref = report["reference_gate"]["metrics"]
    base_metrics = report["baseline_gate"]["metrics"]
    weak = report["weak_kind_summary"]
    lines = [
        f"# Ember v0.0.37 schema-bearing envelope preflight: {report['status']}", "",
        "CPU baseline only. No optimizer, training, GPU submission, promotion, or production integration occurred.", "",
        "| Measurement | Result |", "| --- | ---: |",
        f"| Exact v0.0.8 reference JSON | {ref['envelope_json_valid']}/{ref['cases']} ({ref['envelope_json_valid_rate']:.1%}) |",
        f"| Exact v0.0.8 reference tool name | {ref['envelope_tool_name_correct']}/{ref['cases']} ({ref['envelope_tool_name_rate']:.1%}) |",
        f"| 90-case JSON envelope | {base_metrics['envelope_json_valid']}/{base_metrics['cases']} ({base_metrics['envelope_json_valid_rate']:.1%}) |",
        f"| 90-case correct tool | {base_metrics['envelope_tool_name_correct']}/{base_metrics['cases']} ({base_metrics['envelope_tool_name_rate']:.1%}) |",
        f"| 90-case slot exact | {base_metrics['slot_exact']}/{base_metrics['slot_evaluable']} ({base_metrics['slot_exact_rate']:.1%}) |" if base_metrics["slot_exact_rate"] is not None else "| 90-case slot exact | undefined |",
        f"| Weak kinds JSON envelope | {weak['envelope_json_valid']}/{weak['cases']} ({weak['envelope_json_valid_rate']:.1%}) |",
        "",
        f"Reference gate: {'PASS' if report['reference_gate']['passed'] else 'FAIL'}. 90-case baseline gate: {'PASS' if report['baseline_gate']['passed'] else 'FAIL'}.",
    ]
    if report["status"] == "PASS":
        lines += ["", "The schema-bearing prompt clears the envelope baseline. slot_exact is now interpretable, but this preflight still authorizes no training."]
    else:
        lines += ["", "The envelope baseline remains below threshold. Stop here; do not spend GPU or start a learning phase from this report."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("v037-results"))
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
        "version": "0.0.37",
        "phase": cfg["phase"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ERROR",
        "code_commit": os.environ.get("GITHUB_SHA"),
        "training_authorized": False,
        "gpu_training_authorized": False,
        "production_authorized": False,
        "prompt_reference": cfg["prompt_reference"],
        "weak_kinds": cfg["weak_kinds"],
        "tool_mapping": {kind: {"name": tool, "argument_key": field} for kind, (tool, field) in TOOL_BY_KIND.items()},
    }
    try:
        source_cfg = json.loads(base.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        if source_cfg.get("version") != "0.0.32":
            raise ValueError("unexpected source-loader configuration")
        with tempfile.TemporaryDirectory(prefix="ember-v037-") as td:
            model, tokenizer, source, _splits, source_ref = base.load_inputs(source_cfg, Path(td), torch)
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != cfg["source_version"]:
                raise ValueError("v0.0.37 must measure the pinned v0.0.31 step-479 baseline")

            reference_rows = []
            for case in prior.reference_cases(cfg):
                generation = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"]))
                score = prior.score_reference(case, generation["completion"])
                reference_rows.append({**case, **generation, "score": score})
                print(json.dumps({"event": "reference_case", "id": case["id"], "json_valid": score["envelope_json_valid"], "tool_name_correct": score["tool_name_correct"]}), flush=True)
            reference_gate = prior.reference_summary(reference_rows, cfg)

            rows = []
            for case in build_cases():
                generation = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"]))
                score = prior.score_case(case, generation["completion"])
                rows.append({**case, **generation, "score": score})
                print(json.dumps({"event": "case", "id": case["id"], "kind": case["kind"], "json_valid": score["envelope_json_valid"], "tool_name_correct": score["tool_name_correct"], "slot_exact": score["slot_exact"]}), flush=True)
            baseline_gate = prior.baseline_summary(rows, cfg)
            weak = cohort_summary(rows, set(cfg["weak_kinds"]))
            strong = cohort_summary(rows, set(copy_data.KINDS) - set(cfg["weak_kinds"]))
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
                weak_kind_summary=weak,
                other_kind_summary=strong,
                status="PASS" if passed else "FAIL",
                meaning="Schema-bearing agent-envelope calibration only; no training is authorized.",
            )
    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        if "reference_gate" in report and "baseline_gate" in report:
            (output / "summary.md").write_text(summary_markdown(report), encoding="utf-8")

    print(json.dumps({"event": "complete", "status": report["status"], "reference": report["reference_gate"]["metrics"], "baseline": report["baseline_gate"]["metrics"], "weak_kinds": report["weak_kind_summary"]}), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
