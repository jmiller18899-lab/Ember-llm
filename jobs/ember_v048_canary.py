"""Run the bounded Ember v0.0.48 two-objective CPU canary."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import ember_v048_data as data
from jobs import ember_v048_objectives as objectives
from jobs import ember_v048_regression as regression
from jobs import ember_v048_gate as gate_logic

base = data.base
DEFAULT_CONFIG = data.DEFAULT_CONFIG


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def state_digest(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def summary(report: dict) -> str:
    be, ae = report["before"]["entry_dev"], report["after"]["entry_dev"]
    bp, ap = report["before"]["placement_dev"], report["after"]["placement_dev"]
    fam = report["after"]["familiar_90"]
    return "\n".join([
        f"# Ember v0.0.48 two-objective CPU canary: {report['status']}", "",
        "CPU optimizer canary only. No GPU, promotion, deployment, or production integration occurred.", "",
        f"- Entry dev top-1: {be['top1']}/{be['cases']} -> {ae['top1']}/{ae['cases']}",
        f"- Entry dev mean loss: {be['mean_loss']:.4f} -> {ae['mean_loss']:.4f}",
        f"- Placement dev exact: {bp['exact_top1']}/{bp['cases']} -> {ap['exact_top1']}/{ap['cases']}",
        f"- Placement token top-1: {bp['token_top1_rate']:.1%} -> {ap['token_top1_rate']:.1%}",
        f"- Placement dev mean loss: {bp['mean_loss']:.4f} -> {ap['mean_loss']:.4f}",
        f"- Familiar 90 envelope/tool: {fam['envelope_json_valid']}/90, {fam['correct_tool']}/90",
        f"- Reference controls: {report['after']['reference']['passed_cases']}/4",
        f"- Existing copy guard: {'PASS' if report['copy_guard']['passed'] else 'FAIL'}",
        f"- Canary gate: {'PASS' if report['decision']['passed'] else 'FAIL'}", "",
        "A PASS is learning evidence only; it is not a fresh promotion evaluation or GPU authorization.", "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("v048-results"))
    args = parser.parse_args()
    cfg = data.load_config(args.config)
    if not os.environ.get("HF_TOKEN", "").strip():
        raise ValueError("HF_TOKEN is required for the pinned private checkpoint")

    import torch
    torch.set_num_threads(2)
    torch.manual_seed(int(cfg["seed"]))
    torch.use_deterministic_algorithms(True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "version": "0.0.48",
        "phase": cfg["phase"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ERROR",
        "code_commit": os.environ.get("GITHUB_SHA"),
        "cpu_learning_authorized": True,
        "gpu_training_authorized": False,
        "production_authorized": False,
        "promotion_authorized": False,
    }
    try:
        source_cfg = json.loads(base.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        values = data.synthetic_values(cfg)
        with tempfile.TemporaryDirectory(prefix="ember-v048-") as td:
            model, tokenizer, source, _splits, source_ref = base.load_inputs(source_cfg, Path(td), torch)
            model.to("cpu").eval()
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != cfg["source_version"]:
                raise ValueError("v0.0.48 source is not v0.0.31 step 479")

            template, template_report = data.discover_template(model, tokenizer, torch, cfg, values["template"])
            entry_train = data.build_cases({k: values["train"][k] for k in sorted(data.ENTRY_SUBTYPES)}, "entry_train")
            place_train = data.build_cases({k: values["train"][k] for k in sorted(data.PLACEMENT_SUBTYPES)}, "placement_train")
            entry_dev = data.build_cases({k: values["development"][k] for k in sorted(data.ENTRY_SUBTYPES)}, "entry_dev")
            place_dev = data.build_cases({k: values["development"][k] for k in sorted(data.PLACEMENT_SUBTYPES)}, "placement_dev")

            before_entry = objectives.entry_probe(model, tokenizer, torch, entry_dev)
            before_place = objectives.placement_probe(model, tokenizer, torch, place_dev, template)
            before_copy = base.copy_diagnostic(model, tokenizer, torch)
            before_familiar = regression.familiar_90(model, tokenizer, torch, cfg)
            before_reference = regression.references(model, tokenizer, torch, cfg)
            if before_familiar["envelope_json_valid"] != 84 or before_familiar["correct_tool"] != 84:
                raise ValueError("source no longer reproduces the familiar 84/90 baseline")
            if not before_reference["passed"]:
                raise ValueError("source no longer reproduces 4/4 reference controls")

            source_hash = state_digest(model)
            history = objectives.run_canary(model, tokenizer, torch, cfg, entry_train, place_train, template)
            candidate_hash = state_digest(model)
            changed = candidate_hash != source_hash

            after_entry = objectives.entry_probe(model, tokenizer, torch, entry_dev)
            after_place = objectives.placement_probe(model, tokenizer, torch, place_dev, template)
            after_copy = base.copy_diagnostic(model, tokenizer, torch)
            copy_guard = base.copy_protection(before_copy, after_copy)
            after_reference = regression.references(model, tokenizer, torch, cfg)
            after_familiar = regression.familiar_90(model, tokenizer, torch, cfg)
            familiar_guard = regression.floor_checks(after_familiar, cfg)
            decision = gate_logic.decide(
                before_entry, after_entry, before_place, after_place,
                after_reference, familiar_guard, copy_guard, changed, cfg,
            )

            report.update(
                status="PASS" if decision["passed"] else "FAIL",
                source={
                    "repo_id": source_ref["repo_id"],
                    "checkpoint_path": source_ref["checkpoint_path"],
                    "revision": source_ref["revision"],
                    "checkpoint_sha256": source_ref["checkpoint_sha256"],
                    "step": source["step"],
                    "version": source["train_config"]["version"],
                    "state_sha256": source_hash,
                },
                template=template_report,
                synthetic_counts={p: {s: len(v) for s, v in groups.items()} for p, groups in values.items()},
                train_rows={"entry": len(entry_train), "placement": len(place_train)},
                development_rows={"entry": len(entry_dev), "placement": len(place_dev)},
                before={
                    "entry_dev": before_entry,
                    "placement_dev": before_place,
                    "familiar_90": before_familiar,
                    "reference": before_reference,
                },
                history=history,
                model_state_changed=changed,
                candidate_state_sha256=candidate_hash,
                after={
                    "entry_dev": after_entry,
                    "placement_dev": after_place,
                    "familiar_90": {**after_familiar, "regression_gate": familiar_guard},
                    "reference": after_reference,
                },
                baseline_copy=before_copy,
                candidate_copy=after_copy,
                copy_guard=copy_guard,
                decision=decision,
                meaning="Bounded CPU learning canary; no fresh promotion or GPU authorization.",
            )
    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        write_json(output / "report.json", report)
        if report.get("status") in {"PASS", "FAIL"}:
            (output / "summary.md").write_text(summary(report), encoding="utf-8")

    print(json.dumps({
        "event": "complete",
        "status": report["status"],
        "entry_gain": report["decision"]["metrics"]["entry_top1_gain"],
        "placement_exact_gain": report["decision"]["metrics"]["placement_exact_gain"],
        "familiar_correct_tool": report["after"]["familiar_90"]["correct_tool"],
    }), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
