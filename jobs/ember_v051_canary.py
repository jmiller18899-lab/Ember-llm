"""Run Ember v0.0.51 placement-focused frozen-teacher KL CPU canary."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import ember_v051_data as data
from jobs import ember_v050_distill as distill
from jobs import ember_v048_objectives as objectives
from jobs import ember_v048_regression as regression

base = data.base
DEFAULT_CONFIG = data.DEFAULT_CONFIG


def state_digest(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def final_decision(selection: dict, reference: dict, familiar_guard: dict, copy_guard: dict, changed: bool) -> dict:
    checks = {
        "synthetic_selection": bool(selection.get("passed")),
        "reference": bool(reference["passed"]),
        "familiar": bool(familiar_guard["passed"]),
        "copy": bool(copy_guard["passed"]),
        "state_changed": bool(changed),
    }
    return {"passed": all(checks.values()), "checks": checks}


def summary(report: dict) -> str:
    be, ae = report["before"]["entry_dev"], report["after"]["entry_dev"]
    bp, ap = report["before"]["placement_dev"], report["after"]["placement_dev"]
    td, cd = report["after"]["tool_distill"], report["after"]["copy_distill"]
    fam = report["after"]["familiar_90"]
    return "\n".join([
        f"# Ember v0.0.51 placement-focused teacher-KL CPU canary: {report['status']}", "",
        "Restarted from untouched v0.0.31 step 479. Candidate step selected only from fresh synthetic development/distillation evidence.", "",
        f"- Selected optimizer step: {report['selected_step']}",
        f"- Entry dev top-1: {be['top1']}/{be['cases']} -> {ae['top1']}/{ae['cases']}",
        f"- Placement dev exact: {bp['exact_top1']}/{bp['cases']} -> {ap['exact_top1']}/{ap['cases']}",
        f"- Placement token top-1: {bp['token_top1_rate']:.1%} -> {ap['token_top1_rate']:.1%}",
        f"- Tool teacher KL / token retention: {td['teacher_kl']:.6f} / {td['token_top1_rate']:.1%}",
        f"- Copy teacher KL / token retention: {cd['teacher_kl']:.6f} / {cd['token_top1_rate']:.1%}",
        f"- Familiar 90 envelope/tool: {fam['envelope_json_valid']}/90, {fam['correct_tool']}/90",
        f"- Reference controls: {report['after']['reference']['passed_cases']}/4",
        f"- Existing copy guard: {'PASS' if report['copy_guard']['passed'] else 'FAIL'}",
        f"- Final canary gate: {'PASS' if report['decision']['passed'] else 'FAIL'}", "",
        "No GPU training, promotion, deployment, or production integration is authorized by this canary.", "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("v051-results"))
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
        "version": "0.0.51",
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
        values = data.target_values(cfg)
        with tempfile.TemporaryDirectory(prefix="ember-v051-") as td:
            student, tokenizer, source, _splits, source_ref = base.load_inputs(source_cfg, Path(td), torch)
            student.to("cpu").eval()
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != cfg["source_version"]:
                raise ValueError("v0.0.51 source is not untouched v0.0.31 step 479")
            teacher = copy.deepcopy(student).to("cpu").eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)

            template, template_report = data.v048d.discover_template(student, tokenizer, torch, cfg, values["template"])
            entry_train = data.v048d.build_cases({k: values["train"][k] for k in sorted(data.ENTRY_SUBTYPES)}, "v051_entry_train")
            place_train = data.v048d.build_cases({k: values["train"][k] for k in sorted(data.PLACEMENT_SUBTYPES)}, "v051_place_train")
            entry_dev = data.v048d.build_cases({k: values["development"][k] for k in sorted(data.ENTRY_SUBTYPES)}, "v051_entry_dev")
            place_dev = data.v048d.build_cases({k: values["development"][k] for k in sorted(data.PLACEMENT_SUBTYPES)}, "v051_place_dev")
            tool_rows, tool_report = data.prepare_tool_rows(teacher, tokenizer, torch, cfg)
            copy_rows, copy_report = data.prepare_copy_rows(teacher, tokenizer, torch, cfg)

            before_entry = objectives.entry_probe(student, tokenizer, torch, entry_dev)
            before_place = objectives.placement_probe(student, tokenizer, torch, place_dev, template)
            before_tool = distill.distill_probe(student, teacher, tokenizer, torch, tool_rows, int(cfg["tool_distill_batch_size"]), float(cfg["distill_temperature"]))
            before_copy_distill = distill.distill_probe(student, teacher, tokenizer, torch, copy_rows, int(cfg["copy_distill_batch_size"]), float(cfg["distill_temperature"]))
            before_copy = base.copy_diagnostic(student, tokenizer, torch)
            before_familiar = regression.familiar_90(student, tokenizer, torch, cfg)
            before_reference = regression.references(student, tokenizer, torch, cfg)
            if before_familiar["envelope_json_valid"] != 84 or before_familiar["correct_tool"] != 84:
                raise ValueError("source no longer reproduces familiar 84/90")
            if not before_reference["passed"]:
                raise ValueError("source no longer reproduces 4/4 references")
            if before_tool["teacher_kl"] > 1e-7 or before_copy_distill["teacher_kl"] > 1e-7:
                raise ValueError("frozen teacher does not initially match student")

            tool_id, _eos = objectives.token_contract(tokenizer)
            entry_examples = [objectives.supervised_example(tokenizer, case, "entry", template, tool_id) for case in entry_train]
            place_examples = [objectives.supervised_example(tokenizer, case, "placement", template, tool_id) for case in place_train]
            optimizer = torch.optim.AdamW(student.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
            rng = random.Random(int(cfg["seed"]))
            source_hash = state_digest(student)
            history = []
            selected_step = None
            selection = {"passed": False, "checks": {}, "metrics": {}}
            latest_entry, latest_place = before_entry, before_place
            latest_tool, latest_copy_distill = before_tool, before_copy_distill
            student.train()
            for step in range(1, int(cfg["max_optimizer_steps"]) + 1):
                optimizer.zero_grad(set_to_none=True)
                sample = lambda n, size: [rng.randrange(n) for _ in range(size)]
                entry_loss = objectives.batch_loss(student, torch, entry_examples, sample(len(entry_examples), int(cfg["entry_batch_size"])))
                place_loss = objectives.batch_loss(student, torch, place_examples, sample(len(place_examples), int(cfg["placement_batch_size"])))
                tool_kl = distill.batch_teacher_kl(student, teacher, tokenizer, torch, tool_rows, sample(len(tool_rows), int(cfg["tool_distill_batch_size"])), float(cfg["distill_temperature"]))
                copy_kl = distill.batch_teacher_kl(student, teacher, tokenizer, torch, copy_rows, sample(len(copy_rows), int(cfg["copy_distill_batch_size"])), float(cfg["distill_temperature"]))
                total = (
                    float(cfg["entry_loss_weight"]) * entry_loss
                    + float(cfg["placement_loss_weight"]) * place_loss
                    + float(cfg["tool_kl_loss_weight"]) * tool_kl
                    + float(cfg["copy_kl_loss_weight"]) * copy_kl
                )
                if not bool(torch.isfinite(total).item()):
                    raise ValueError("non-finite v0.0.51 canary loss")
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(student.parameters(), float(cfg["gradient_clip"]), error_if_nonfinite=True)
                optimizer.step()
                if step == 1 or step % int(cfg["checkpoint_interval"]) == 0:
                    event = {
                        "step": step,
                        "entry_loss": float(entry_loss.detach()),
                        "placement_loss": float(place_loss.detach()),
                        "tool_kl_loss": float(tool_kl.detach()),
                        "copy_kl_loss": float(copy_kl.detach()),
                        "combined_loss": float(total.detach()),
                        "gradient_norm": float(norm),
                    }
                    history.append(event)
                    print(json.dumps({"event": "placement_teacher_kl_step", **event}), flush=True)
                if step % int(cfg["checkpoint_interval"]) == 0:
                    student.eval()
                    latest_entry = objectives.entry_probe(student, tokenizer, torch, entry_dev)
                    latest_place = objectives.placement_probe(student, tokenizer, torch, place_dev, template)
                    latest_tool = distill.distill_probe(student, teacher, tokenizer, torch, tool_rows, int(cfg["tool_distill_batch_size"]), float(cfg["distill_temperature"]))
                    latest_copy_distill = distill.distill_probe(student, teacher, tokenizer, torch, copy_rows, int(cfg["copy_distill_batch_size"]), float(cfg["distill_temperature"]))
                    selection = distill.selection_checks(before_entry, latest_entry, before_place, latest_place, latest_tool, latest_copy_distill, cfg)
                    print(json.dumps({"event": "selection_checkpoint", "step": step, **selection}), flush=True)
                    if selection["passed"]:
                        selected_step = step
                        break
                    student.train()
            del optimizer
            student.eval()
            if selected_step is None:
                selected_step = int(cfg["max_optimizer_steps"])

            candidate_hash = state_digest(student)
            changed = candidate_hash != source_hash
            after_entry, after_place = latest_entry, latest_place
            after_tool, after_copy_distill = latest_tool, latest_copy_distill
            after_copy = base.copy_diagnostic(student, tokenizer, torch)
            copy_guard = base.copy_protection(before_copy, after_copy)
            after_reference = regression.references(student, tokenizer, torch, cfg)
            after_familiar = regression.familiar_90(student, tokenizer, torch, cfg)
            floor_cfg = {**cfg, "gate": cfg["final_gate"]}
            familiar_guard = regression.floor_checks(after_familiar, floor_cfg)
            decision = final_decision(selection, after_reference, familiar_guard, copy_guard, changed)

            report.update(
                status="PASS" if decision["passed"] else "FAIL",
                source={
                    "repo_id": source_ref["repo_id"], "checkpoint_path": source_ref["checkpoint_path"],
                    "revision": source_ref["revision"], "checkpoint_sha256": source_ref["checkpoint_sha256"],
                    "step": source["step"], "version": source["train_config"]["version"], "state_sha256": source_hash,
                },
                template=template_report,
                distill_sources={"tool": tool_report, "copy": copy_report},
                train_rows={"entry": len(entry_train), "placement": len(place_train), "tool_distill": len(tool_rows), "copy_distill": len(copy_rows)},
                development_rows={"entry": len(entry_dev), "placement": len(place_dev)},
                before={"entry_dev": before_entry, "placement_dev": before_place, "tool_distill": before_tool, "copy_distill": before_copy_distill, "familiar_90": before_familiar, "reference": before_reference},
                history=history,
                selected_step=selected_step,
                selection=selection,
                model_state_changed=changed,
                candidate_state_sha256=candidate_hash,
                after={
                    "entry_dev": after_entry, "placement_dev": after_place,
                    "tool_distill": after_tool, "copy_distill": after_copy_distill,
                    "familiar_90": {**after_familiar, "regression_gate": familiar_guard},
                    "reference": after_reference,
                },
                baseline_copy=before_copy,
                candidate_copy=after_copy,
                copy_guard=copy_guard,
                decision=decision,
                meaning="Placement-focused frozen-teacher CPU canary only; no GPU or promotion authorization.",
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
        "event": "complete", "status": report["status"], "selected_step": report["selected_step"],
        "entry_gain": report["selection"]["metrics"].get("entry_top1_gain"),
        "placement_exact_gain": report["selection"]["metrics"].get("placement_exact_gain"),
        "placement_token_gain": report["selection"]["metrics"].get("placement_token_top1_gain"),
        "familiar_correct_tool": report["after"]["familiar_90"]["correct_tool"],
        "tool_teacher_kl": report["after"]["tool_distill"]["teacher_kl"],
        "copy_teacher_kl": report["after"]["copy_distill"]["teacher_kl"],
    }), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
