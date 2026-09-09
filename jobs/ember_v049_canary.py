"""Run the bounded Ember v0.0.49 protected-learning CPU canary.

v0.0.48 learned entry (19/24 -> 24/24) and placement (1/24 -> 9/24) and lost the
familiar battery (84/90 -> 33/90 correct tool), a reference control, and copy
protection. This canary restarts from the untouched v0.0.31 step-479 source and
asks whether the same two gains survive a much smaller, replay-protected update.

No GPU, promotion, deployment, or production integration is possible here. The
familiar 90 cases and the four historical controls are evaluation-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import ember_v048_canary as v048
from jobs import ember_v048_regression as regression
from jobs import ember_v049_data as data
from jobs import ember_v049_gate as gate_logic
from jobs import ember_v049_objectives as objectives

base = data.base
DEFAULT_CONFIG = data.DEFAULT_CONFIG
write_json = v048.write_json
state_digest = v048.state_digest


def summary(report: dict) -> str:
    be, ae = report["before"]["entry_dev"], report["after"]["entry_dev"]
    bp, ap = report["before"]["placement_dev"], report["after"]["placement_dev"]
    fam_before = report["before"]["familiar_90"]
    fam = report["after"]["familiar_90"]
    drift = report["drift"]
    lines = [
        f"# Ember v0.0.49 protected-learning CPU canary: {report['status']}", "",
        "CPU optimizer canary only. No GPU, promotion, deployment, or production integration occurred.",
        "The familiar 90 cases and four historical controls are evaluation-only.", "",
        "## Did the two objectives still move?", "",
        f"- Entry dev top-1: {be['top1']}/{be['cases']} -> {ae['top1']}/{ae['cases']}",
        f"- Entry dev mean loss: {be['mean_loss']:.4f} -> {ae['mean_loss']:.4f}",
        f"- Placement dev exact: {bp['exact_top1']}/{bp['cases']} -> {ap['exact_top1']}/{ap['cases']}",
        f"- Placement token top-1: {bp['token_top1_rate']:.1%} -> {ap['token_top1_rate']:.1%}",
        f"- Placement dev mean loss: {bp['mean_loss']:.4f} -> {ap['mean_loss']:.4f}", "",
        "## Did protection hold?", "",
        f"- Familiar 90 canonical JSON: {fam_before['envelope_json_valid']}/90 -> {fam['envelope_json_valid']}/90",
        f"- Familiar 90 correct tool: {fam_before['correct_tool']}/90 -> {fam['correct_tool']}/90",
        f"- Reference controls: {report['before']['reference']['passed_cases']}/4 -> {report['after']['reference']['passed_cases']}/4",
        f"- Existing copy guard: {'PASS' if report['copy_guard']['passed'] else 'FAIL'}",
        f"- Max relative parameter drift: {drift['max_relative_drift']:.3e} (limit {drift['limit']:.3e})",
        f"- Trust region respected: {'yes' if drift['within_trust_region'] else 'NO'}",
        (f"- Trust region engaged on {drift['steps_clipped']} steps "
         f"(drift reached {drift['headroom']:.0%} of the cap)"
         if drift.get("engaged")
         else f"- Trust region never engaged: drift reached only {drift['headroom']:.0%} of the cap, "
              "so the learning rate and replay term provided the protection"),
        f"- Replay corpus: {report['replay']['envelope']['captured']} envelope rows over "
        f"{report['replay']['envelope']['kinds_covered']} kinds, "
        f"{report['replay']['copy']['captured']} copy rows over "
        f"{report['replay']['copy']['kinds_covered']} kinds", "",
        "## Interference trajectory", "",
        "| Step | Synthetic envelope probe | Max drift |",
        "| --- | ---: | ---: |",
    ]
    for point in report["interference"]["points"]:
        drift_value = point.get("max_relative_drift")
        lines.append(
            f"| {point['step']} | {point['interference_correct']}/{report['interference']['cases']} "
            f"| {'' if drift_value is None else f'{drift_value:.3e}'} |"
        )
    lines += [
        "",
        f"- Canary gate: {'PASS' if report['decision']['passed'] else 'FAIL'}",
        "",
        "A PASS is learning evidence only. It is not a fresh promotion evaluation and it does not authorize GPU training.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("v049-results"))
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
        "version": "0.0.49",
        "phase": cfg["phase"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ERROR",
        "code_commit": os.environ.get("GITHUB_SHA"),
        "cpu_learning_authorized": True,
        "gpu_training_authorized": False,
        "production_authorized": False,
        "promotion_authorized": False,
        "predecessor": {
            "version": "0.0.48",
            "outcome": "FAIL by catastrophic interference",
            "learning_rate": 0.0000012,
            "familiar_correct_tool_after": 33,
        },
    }
    try:
        source_cfg = json.loads(base.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        used = data.historical_used_values()
        values = data.synthetic_values(cfg)
        for groups in values.values():
            for items in groups.values():
                used.update(items)
        replay_plan = data.replay_values(cfg, used)
        interference_plan = data.interference_values(cfg, used)

        with tempfile.TemporaryDirectory(prefix="ember-v049-") as td:
            model, tokenizer, source, _splits, source_ref = base.load_inputs(source_cfg, Path(td), torch)
            model.to("cpu").eval()
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != cfg["source_version"]:
                raise ValueError("v0.0.49 source is not the untouched v0.0.31 step 479")

            template, template_report = data.discover_template(model, tokenizer, torch, cfg, values["template"])
            entry_train = data.build_cases({k: values["train"][k] for k in sorted(data.ENTRY_SUBTYPES)}, "entry_train")
            place_train = data.build_cases({k: values["train"][k] for k in sorted(data.PLACEMENT_SUBTYPES)}, "placement_train")
            entry_dev = data.build_cases({k: values["development"][k] for k in sorted(data.ENTRY_SUBTYPES)}, "entry_dev")
            place_dev = data.build_cases({k: values["development"][k] for k in sorted(data.PLACEMENT_SUBTYPES)}, "placement_dev")

            envelope_rows, envelope_summary = data.capture_envelope_replay(
                model, tokenizer, torch, cfg, replay_plan["envelope"])
            copy_rows, copy_summary = data.capture_copy_replay(
                model, tokenizer, torch, cfg, replay_plan["copy"])
            report["replay"] = {"envelope": envelope_summary, "copy": copy_summary}
            print(json.dumps({"event": "replay_captured",
                              "envelope": envelope_summary, "copy": copy_summary}), flush=True)
            data.enforce_replay_floor(cfg, envelope_summary, copy_summary)
            replay_rows = envelope_rows + copy_rows

            interference_cases = [
                data.envelope_prompt_for_kind(kind, value)
                for kind, items in sorted(interference_plan.items()) for value in items
            ]

            before_entry = objectives.entry_probe(model, tokenizer, torch, entry_dev)
            before_place = objectives.placement_probe(model, tokenizer, torch, place_dev, template)
            before_interference = objectives.interference_probe(model, tokenizer, torch, cfg, interference_cases)
            before_copy = base.copy_diagnostic(model, tokenizer, torch)
            before_familiar = regression.familiar_90(model, tokenizer, torch, cfg)
            before_reference = regression.references(model, tokenizer, torch, cfg)
            if before_familiar["envelope_json_valid"] != 84 or before_familiar["correct_tool"] != 84:
                raise ValueError("source no longer reproduces the familiar 84/90 baseline")
            if not before_reference["passed"]:
                raise ValueError("source no longer reproduces 4/4 reference controls")

            source_hash = state_digest(model)

            def probe(current_model, _step):
                return {
                    "interference": objectives.interference_probe(
                        current_model, tokenizer, torch, cfg, interference_cases),
                    "entry": objectives.entry_probe(current_model, tokenizer, torch, entry_dev),
                    "placement": objectives.placement_probe(
                        current_model, tokenizer, torch, place_dev, template),
                }

            run = objectives.run_canary(
                model, tokenizer, torch, cfg, entry_train, place_train,
                replay_rows, template, probe=probe,
            )
            candidate_hash = state_digest(model)
            changed = candidate_hash != source_hash

            after_entry = objectives.entry_probe(model, tokenizer, torch, entry_dev)
            after_place = objectives.placement_probe(model, tokenizer, torch, place_dev, template)
            after_interference = objectives.interference_probe(model, tokenizer, torch, cfg, interference_cases)
            after_copy = base.copy_diagnostic(model, tokenizer, torch)
            copy_guard = base.copy_protection(before_copy, after_copy)
            after_reference = regression.references(model, tokenizer, torch, cfg)
            after_familiar = regression.familiar_90(model, tokenizer, torch, cfg)
            familiar_guard = regression.floor_checks(after_familiar, cfg)
            decision = gate_logic.decide(
                before_entry, after_entry, before_place, after_place,
                after_reference, familiar_guard, copy_guard, changed,
                run["final_drift"], cfg,
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
                replay_counts={p: {k: len(v) for k, v in groups.items()} for p, groups in replay_plan.items()},
                train_rows=run["objective_rows"],
                development_rows={"entry": len(entry_dev), "placement": len(place_dev)},
                replay_build=run["replay_build"],
                before={
                    "entry_dev": before_entry,
                    "placement_dev": before_place,
                    "interference": before_interference,
                    "familiar_90": before_familiar,
                    "reference": before_reference,
                },
                history=run["history"],
                trajectory=run["trajectory"],
                drift=run["final_drift"],
                model_state_changed=changed,
                candidate_state_sha256=candidate_hash,
                after={
                    "entry_dev": after_entry,
                    "placement_dev": after_place,
                    "interference": after_interference,
                    "familiar_90": {**after_familiar, "regression_gate": familiar_guard},
                    "reference": after_reference,
                },
                interference=gate_logic.interference_summary(
                    before_interference, after_interference, run["trajectory"]),
                baseline_copy=before_copy,
                candidate_copy=after_copy,
                copy_guard=copy_guard,
                decision=decision,
                meaning="Bounded protected CPU learning canary; no fresh promotion or GPU authorization.",
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
        "max_relative_drift": report["drift"]["max_relative_drift"],
    }), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
