"""Run Ember v0.0.49 protected-replay CPU canary from untouched step 479."""
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

from jobs import ember_v049_replay as replay
from jobs import ember_v048_data as v048d
from jobs import ember_v048_objectives as objectives
from jobs import ember_v048_regression as regression
from jobs import ember_v048_gate as gate_logic

base = replay.base
DEFAULT_CONFIG = replay.DEFAULT_CONFIG


def state_digest(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def summary(report: dict) -> str:
    be, ae = report["before"]["entry_dev"], report["after"]["entry_dev"]
    bp, ap = report["before"]["placement_dev"], report["after"]["placement_dev"]
    bt, at = report["before"]["tool_replay"], report["after"]["tool_replay"]
    bc, ac = report["before"]["copy_replay"], report["after"]["copy_replay"]
    fam = report["after"]["familiar_90"]
    return "\n".join([
        f"# Ember v0.0.49 protected replay CPU canary: {report['status']}", "",
        "Restarted from untouched v0.0.31 step 479. CPU only; no GPU, promotion, deployment, or production integration.", "",
        f"- Entry dev top-1: {be['top1']}/{be['cases']} -> {ae['top1']}/{ae['cases']}",
        f"- Placement dev exact: {bp['exact_top1']}/{bp['cases']} -> {ap['exact_top1']}/{ap['cases']}",
        f"- Placement token top-1: {bp['token_top1_rate']:.1%} -> {ap['token_top1_rate']:.1%}",
        f"- Tool replay token retention: {bt['token_top1_rate']:.1%} -> {at['token_top1_rate']:.1%}",
        f"- Copy replay token retention: {bc['token_top1_rate']:.1%} -> {ac['token_top1_rate']:.1%}",
        f"- Familiar 90 envelope/tool: {fam['envelope_json_valid']}/90, {fam['correct_tool']}/90",
        f"- Reference controls: {report['after']['reference']['passed_cases']}/4",
        f"- Existing copy guard: {'PASS' if report['copy_guard']['passed'] else 'FAIL'}",
        f"- Protected canary gate: {'PASS' if report['decision']['passed'] else 'FAIL'}", "",
        "A PASS is CPU preservation/learning evidence only; it does not authorize GPU training or promotion.", "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("v049-results"))
    args = parser.parse_args()
    cfg = replay.load_config(args.config)
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
    }
    try:
        source_cfg = json.loads(base.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        values = replay.target_values(cfg)
        with tempfile.TemporaryDirectory(prefix="ember-v049-") as td:
            model, tokenizer, source, _splits, source_ref = base.load_inputs(source_cfg, Path(td), torch)
            model.to("cpu").eval()
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != cfg["source_version"]:
                raise ValueError("v0.0.49 source is not untouched v0.0.31 step 479")

            template, template_report = v048d.discover_template(model, tokenizer, torch, cfg, values["template"])
            entry_train = v048d.build_cases({k: values["train"][k] for k in sorted(replay.ENTRY_SUBTYPES)}, "v049_entry_train")
            place_train = v048d.build_cases({k: values["train"][k] for k in sorted(replay.PLACEMENT_SUBTYPES)}, "v049_place_train")
            entry_dev = v048d.build_cases({k: values["development"][k] for k in sorted(replay.ENTRY_SUBTYPES)}, "v049_entry_dev")
            place_dev = v048d.build_cases({k: values["development"][k] for k in sorted(replay.PLACEMENT_SUBTYPES)}, "v049_place_dev")

            tool_rows, tool_replay_report = replay.prepare_tool_replay(model, tokenizer, torch, cfg)
            copy_rows, copy_replay_report = replay.prepare_copy_replay(model, tokenizer, torch, cfg)

            before_entry = objectives.entry_probe(model, tokenizer, torch, entry_dev)
            before_place = objectives.placement_probe(model, tokenizer, torch, place_dev, template)
            before_tool_replay = replay.replay_probe(model, tokenizer, torch, tool_rows)
            before_copy_replay = replay.replay_probe(model, tokenizer, torch, copy_rows)
            before_copy = base.copy_diagnostic(model, tokenizer, torch)
            before_familiar = regression.familiar_90(model, tokenizer, torch, cfg)
            before_reference = regression.references(model, tokenizer, torch, cfg)
            if before_familiar["envelope_json_valid"] != 84 or before_familiar["correct_tool"] != 84:
                raise ValueError("source no longer reproduces familiar 84/90")
            if not before_reference["passed"]:
                raise ValueError("source no longer reproduces 4/4 references")
            if before_tool_replay["token_top1_rate"] < 0.999 or before_copy_replay["token_top1_rate"] < 0.999:
                raise ValueError("source-generated replay rows do not reproduce their own greedy tokens")

            source_hash = state_digest(model)
            history = replay.run_protected_canary(
                model, tokenizer, torch, cfg, entry_train, place_train, template, tool_rows, copy_rows
            )
            candidate_hash = state_digest(model)
            changed = candidate_hash != source_hash

            after_entry = objectives.entry_probe(model, tokenizer, torch, entry_dev)
            after_place = objectives.placement_probe(model, tokenizer, torch, place_dev, template)
            after_tool_replay = replay.replay_probe(model, tokenizer, torch, tool_rows)
            after_copy_replay = replay.replay_probe(model, tokenizer, torch, copy_rows)
            after_copy = base.copy_diagnostic(model, tokenizer, torch)
            copy_guard = base.copy_protection(before_copy, after_copy)
            after_reference = regression.references(model, tokenizer, torch, cfg)
            after_familiar = regression.familiar_90(model, tokenizer, torch, cfg)
            familiar_guard = regression.floor_checks(after_familiar, cfg)
            base_decision = gate_logic.decide(
                before_entry, after_entry, before_place, after_place,
                after_reference, familiar_guard, copy_guard, changed, cfg,
            )
            replay_checks = {
                "tool_replay": after_tool_replay["token_top1_rate"] >= float(cfg["gate"]["minimum_tool_replay_token_top1"]),
                "copy_replay": after_copy_replay["token_top1_rate"] >= float(cfg["gate"]["minimum_copy_replay_token_top1"]),
            }
            decision = {
                **base_decision,
                "checks": {**base_decision["checks"], **replay_checks},
            }
            decision["passed"] = all(decision["checks"].values())

            report.update(
                status="PASS" if decision["passed"] else "FAIL",
                source={
                    "repo_id": source_ref["repo_id"], "checkpoint_path": source_ref["checkpoint_path"],
                    "revision": source_ref["revision"], "checkpoint_sha256": source_ref["checkpoint_sha256"],
                    "step": source["step"], "version": source["train_config"]["version"], "state_sha256": source_hash,
                },
                template=template_report,
                replay_sources={"tool": tool_replay_report, "copy": copy_replay_report},
                train_rows={"entry": len(entry_train), "placement": len(place_train), "tool_replay": len(tool_rows), "copy_replay": len(copy_rows)},
                development_rows={"entry": len(entry_dev), "placement": len(place_dev)},
                before={
                    "entry_dev": before_entry, "placement_dev": before_place,
                    "tool_replay": before_tool_replay, "copy_replay": before_copy_replay,
                    "familiar_90": before_familiar, "reference": before_reference,
                },
                history=history,
                model_state_changed=changed,
                candidate_state_sha256=candidate_hash,
                after={
                    "entry_dev": after_entry, "placement_dev": after_place,
                    "tool_replay": after_tool_replay, "copy_replay": after_copy_replay,
                    "familiar_90": {**after_familiar, "regression_gate": familiar_guard},
                    "reference": after_reference,
                },
                baseline_copy=before_copy,
                candidate_copy=after_copy,
                copy_guard=copy_guard,
                decision=decision,
                meaning="Protected CPU replay canary only; no fresh promotion or GPU authorization.",
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
        "event": "complete", "status": report["status"],
        "entry_gain": report["decision"]["metrics"]["entry_top1_gain"],
        "placement_exact_gain": report["decision"]["metrics"]["placement_exact_gain"],
        "familiar_correct_tool": report["after"]["familiar_90"]["correct_tool"],
        "tool_replay_top1": report["after"]["tool_replay"]["token_top1_rate"],
        "copy_replay_top1": report["after"]["copy_replay"]["token_top1_rate"],
    }), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
