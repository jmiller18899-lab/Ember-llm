"""Bounded alternating copy/repair trust-region diagnostic for Ember.

Uses the same eight baseline-failing placement cases and pinned v0.0.31 step-479
source. Each rung resets to pristine. A cycle makes exactly one copy-placement
update, then (only if needed) up to four structure-only repair updates. A cycle
is accepted only when free-running JSON/tool behavior returns to the source
floor (>=7/8) and placement loss does not regress; otherwise model AND optimizer
states roll back to the pre-cycle snapshot. No checkpoint is saved or promoted.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import time

from jobs import ember_rung0_trajectory as trace
from jobs import ember_tiny_overfit as tiny
from jobs import ember_tiny_lr_ladder as ladder
from jobs import ember_tiny_protected_overfit as protected


diag = trace.diag
data = trace.data
objectives = trace.objectives
base = trace.base

COPY_LRS = (4e-7, 8e-7, 1.2e-6)
REPAIR_LR = 1.6e-6
MAX_CYCLES = 16
MAX_REPAIR_STEPS = 4
ACCEPT_EXACT = 6
ACCEPT_STRUCTURE = 7
WALL_SECONDS = 1500


def placement_only_probe(model, tokenizer, torch, selected, template):
    with trace.observation(model, torch):
        return objectives.placement_probe(model, tokenizer, torch, selected, template)


def structure_only_probe(model, tokenizer, torch, cfg, selected):
    return tiny.structure_probe(model, tokenizer, torch, cfg, selected)


def structure_pass(structure):
    return (
        int(structure["envelope_json_valid"]) >= ACCEPT_STRUCTURE
        and int(structure["tool_name_correct"]) >= ACCEPT_STRUCTURE
    )


def optimizer_update(model, optimizer, torch, examples, indices, cfg):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = objectives.batch_loss(model, torch, examples, indices)
    if not bool(torch.isfinite(loss)):
        raise ValueError("non-finite alternating loss")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(cfg["gradient_clip"]), error_if_nonfinite=True
    )
    before = diag.flat_parameters(model, torch)
    optimizer.step()
    model.zero_grad(set_to_none=True)
    model.eval()
    update_l2 = float((diag.flat_parameters(model, torch) - before).norm())
    if not math.isfinite(update_l2) or update_l2 <= 0:
        raise ValueError("finite nonzero update required")
    return {
        "loss": float(loss.detach()),
        "gradient_norm": float(grad_norm),
        "update_l2": update_l2,
    }


def probe_record(cycle, place, structure, baseline_wrong, phase, repair_step=0):
    wrong = tiny.wrong_token_stats(place, baseline_wrong)
    rec = {
        "cycle": int(cycle),
        "phase": phase,
        "repair_step": int(repair_step),
        "exact_top1": int(place["exact_top1"]),
        "token_top1": int(place["token_top1"]),
        "tokens": int(place["tokens"]),
        "mean_loss": float(place["mean_loss"]),
        "wrong_top1": int(wrong["top1"]),
        "wrong_tokens": int(wrong["tokens"]),
        "wrong_mean_probability": float(wrong["mean_target_probability"]),
        "json_valid": int(structure["envelope_json_valid"]),
        "tool_correct": int(structure["tool_name_correct"]),
        "slot_exact_generated": int(structure["slot_exact"]),
    }
    rec["copy_pass"] = rec["exact_top1"] >= ACCEPT_EXACT
    rec["structure_pass"] = structure_pass(structure)
    rec["simultaneous_pass"] = rec["copy_pass"] and rec["structure_pass"]
    return rec


def run_rung(student, tokenizer, torch, cfg, template, selected, baseline_wrong,
             placement_examples, structural_examples, pristine, source_hash, copy_lr):
    student.load_state_dict(pristine)
    student.eval()
    if trace.state_digest(student) != source_hash:
        raise ValueError("rung reset failed")

    copy_opt = torch.optim.AdamW(student.parameters(), lr=float(copy_lr), weight_decay=float(cfg["weight_decay"]))
    repair_opt = torch.optim.AdamW(student.parameters(), lr=REPAIR_LR, weight_decay=float(cfg["weight_decay"]))
    pidx = list(range(len(placement_examples)))
    sidx = list(range(len(structural_examples)))

    place = placement_only_probe(student, tokenizer, torch, selected, template)
    structure = structure_only_probe(student, tokenizer, torch, cfg, selected)
    if not structure_pass(structure):
        raise ValueError("source structure floor no longer reproduced")
    rung = {
        "copy_lr": float(copy_lr),
        "repair_lr": REPAIR_LR,
        "cycles_attempted": 0,
        "cycles_accepted": 0,
        "cycles_rejected": 0,
        "accepted": False,
        "events": [probe_record(0, place, structure, baseline_wrong, "baseline")],
    }
    print(json.dumps({"event":"trust_rung_start", "copy_lr":copy_lr, **rung["events"][-1]}), flush=True)

    for cycle in range(1, MAX_CYCLES + 1):
        rung["cycles_attempted"] = cycle
        pre_model = copy.deepcopy(student.state_dict())
        pre_copy_opt = copy.deepcopy(copy_opt.state_dict())
        pre_repair_opt = copy.deepcopy(repair_opt.state_dict())
        pre_place = placement_only_probe(student, tokenizer, torch, selected, template)

        copy_update = optimizer_update(student, copy_opt, torch, placement_examples, pidx, cfg)
        after_copy_place = placement_only_probe(student, tokenizer, torch, selected, template)
        after_copy_structure = structure_only_probe(student, tokenizer, torch, cfg, selected)
        rec = probe_record(cycle, after_copy_place, after_copy_structure, baseline_wrong, "after_copy")
        rec["copy_update"] = copy_update
        rung["events"].append(rec)
        print(json.dumps({"event":"trust_probe", "copy_lr":copy_lr, **rec}), flush=True)

        repairs = []
        structure = after_copy_structure
        place = after_copy_place
        if not structure_pass(structure):
            for repair_step in range(1, MAX_REPAIR_STEPS + 1):
                repair_update = optimizer_update(student, repair_opt, torch, structural_examples, sidx, cfg)
                place = placement_only_probe(student, tokenizer, torch, selected, template)
                structure = structure_only_probe(student, tokenizer, torch, cfg, selected)
                rrec = probe_record(cycle, place, structure, baseline_wrong, "after_repair", repair_step)
                rrec["repair_update"] = repair_update
                repairs.append(rrec)
                rung["events"].append(rrec)
                print(json.dumps({"event":"trust_probe", "copy_lr":copy_lr, **rrec}), flush=True)
                if structure_pass(structure):
                    break

        placement_nonregression = float(place["mean_loss"]) <= float(pre_place["mean_loss"]) + 1e-7
        cycle_accept = structure_pass(structure) and placement_nonregression
        if not cycle_accept:
            student.load_state_dict(pre_model)
            copy_opt.load_state_dict(pre_copy_opt)
            repair_opt.load_state_dict(pre_repair_opt)
            student.eval()
            rung["cycles_rejected"] += 1
            rollback_place = placement_only_probe(student, tokenizer, torch, selected, template)
            rollback_structure = structure_only_probe(student, tokenizer, torch, cfg, selected)
            rb = probe_record(cycle, rollback_place, rollback_structure, baseline_wrong, "rollback")
            rb["reason"] = {
                "structure_recovered": structure_pass(structure),
                "placement_nonregression": placement_nonregression,
                "repair_steps_used": len(repairs),
            }
            rung["events"].append(rb)
            print(json.dumps({"event":"trust_cycle_rejected", "copy_lr":copy_lr, **rb}), flush=True)
            # If one copy step cannot be repaired inside the trust region, this rung cannot advance safely.
            break

        rung["cycles_accepted"] += 1
        accepted_rec = probe_record(cycle, place, structure, baseline_wrong, "accepted_pair", len(repairs))
        accepted_rec["repair_steps_used"] = len(repairs)
        rung["events"].append(accepted_rec)
        print(json.dumps({"event":"trust_cycle_accepted", "copy_lr":copy_lr, **accepted_rec}), flush=True)
        if accepted_rec["simultaneous_pass"]:
            rung["accepted"] = True
            rung["accepted_cycle"] = cycle
            break

    final_place = placement_only_probe(student, tokenizer, torch, selected, template)
    final_structure = structure_only_probe(student, tokenizer, torch, cfg, selected)
    rung["final_selected"] = final_place
    rung["final_structure"] = final_structure
    rung["final_simultaneous_pass"] = (
        int(final_place["exact_top1"]) >= ACCEPT_EXACT and structure_pass(final_structure)
    )
    rung["accepted"] = bool(rung["accepted"] or rung["final_simultaneous_pass"])
    print(json.dumps({
        "event":"trust_rung_complete", "copy_lr":copy_lr,
        "cycles_attempted":rung["cycles_attempted"], "cycles_accepted":rung["cycles_accepted"],
        "cycles_rejected":rung["cycles_rejected"], "accepted":rung["accepted"],
        "final_exact":final_place["exact_top1"], "final_tokens":final_place["token_top1"],
        "json_valid":final_structure["envelope_json_valid"], "tool_correct":final_structure["tool_name_correct"],
    }), flush=True)
    del copy_opt, repair_opt
    return rung


def summary_markdown(report):
    lines = [
        "# Ember alternating repair / trust-region diagnostic", "",
        f"Execution: {report['status']}",
        "Source: pinned v0.0.31 step 479",
        f"Copy LR rungs: {', '.join(f'{x:.1e}' for x in COPY_LRS)}",
        f"Structure repair LR: {REPAIR_LR:.1e}",
        f"Gate: >= {ACCEPT_EXACT}/8 exact copies AND >= {ACCEPT_STRUCTURE}/8 valid/correct-tool envelopes.",
        "Each accepted cycle = one copy update plus 0-4 structure-only repairs. Failed cycles roll back model and both optimizer states.",
        "", "| Copy LR | Accepted cycles | Rejected cycles | Exact copies | Copy tokens | JSON valid | Tool correct | Pass |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rung in report.get("rungs", []):
        p, s = rung["final_selected"], rung["final_structure"]
        lines.append(
            f"| {rung['copy_lr']:.1e} | {rung['cycles_accepted']} | {rung['cycles_rejected']} | "
            f"{p['exact_top1']}/{p['cases']} | {p['token_top1']}/{p['tokens']} | "
            f"{s['envelope_json_valid']}/{s['cases']} | {s['tool_name_correct']}/{s['cases']} | {rung['accepted']} |"
        )
    lines += ["",
        f"Operating point found: {report.get('operating_point_found', False)}",
        f"Selected copy LR: {report.get('selected_copy_lr')}",
        f"Selected cycle: {report.get('selected_cycle')}",
        f"Interpretation: {report.get('interpretation', 'not available')}", "",
        "No checkpoint was saved, exported, promoted, or integrated. The source checkpoint is unchanged.",
    ]
    if report.get("error"):
        lines.append(f"Error: {report['error']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("alternating-trust-results"))
    args = parser.parse_args()
    cfg = trace.load_config()
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(int(cfg["seed"]))
    torch.use_deterministic_algorithms(True)
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("alternating trust-region wall bound")))
    signal.alarm(WALL_SECONDS)

    report = {
        "schema_version":1, "diagnostic":"ember-alternating-trust-region-v1", "status":"ERROR",
        "created_at":datetime.now(timezone.utc).isoformat(), "code_commit":os.environ.get("GITHUB_SHA"),
        "copy_lrs":list(COPY_LRS), "repair_lr":REPAIR_LR, "max_cycles":MAX_CYCLES,
        "max_repair_steps":MAX_REPAIR_STEPS, "accept_exact":ACCEPT_EXACT, "accept_structure":ACCEPT_STRUCTURE,
        "cpu_learning_authorized":True, "gpu_training_authorized":False,
        "promotion_authorized":False, "production_authorized":False, "checkpoint_export_authorized":False,
        "rungs":[],
    }
    student = pristine = teacher = None
    try:
        with tempfile.TemporaryDirectory(prefix="ember-alternating-trust-") as td:
            student, tokenizer, source, _splits, source_ref = base.load_inputs(
                json.loads(base.DEFAULT_CONFIG.read_text()), Path(td), torch
            )
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != "0.0.31":
                raise ValueError("not pinned v0.0.31 step-479 source")
            if float(student.cfg.dropout) != 0:
                raise ValueError("zero dropout required")
            student.to("cpu").eval(); pristine = copy.deepcopy(student.state_dict())
            source_hash = trace.state_digest(student)
            teacher = copy.deepcopy(student).eval()
            for p in teacher.parameters(): p.requires_grad_(False)
            report.update(source=source_ref, source_state_sha256=source_hash)
            del source, _splits

            values = data.target_values(cfg)
            template, template_report = data.v048d.discover_template(student, tokenizer, torch, cfg, values["template"])
            development = data.v048d.build_cases(
                {k: values["development"][k] for k in sorted(data.PLACEMENT_SUBTYPES)}, "v051_place_dev"
            )
            full_baseline, chosen_rows, selected, selected_baseline, baseline_wrong = ladder.choose_cases(
                student, tokenizer, torch, cfg, template, development
            )
            baseline_structure = tiny.structure_probe(student, tokenizer, torch, cfg, selected)
            if not structure_pass(baseline_structure):
                raise ValueError("selected source no longer has 7/8 valid structure")
            report["template_report"] = template_report
            report["full_baseline"] = full_baseline
            report["selected_case_ids"] = [c["id"] for c in selected]
            report["selection"] = chosen_rows
            report["selected_baseline"] = selected_baseline
            report["baseline_structure"] = baseline_structure

            tool_id, _ = objectives.token_contract(tokenizer)
            placement_examples = [objectives.supervised_example(tokenizer, c, "placement", template, tool_id) for c in selected]
            structural_examples, structural_records = protected.prepare_structural_replay(
                teacher, tokenizer, torch, cfg, selected
            )
            report["structural_replay"] = structural_records
            report["structural_replay_examples"] = len(structural_examples)

            for copy_lr in COPY_LRS:
                rung = run_rung(student, tokenizer, torch, cfg, template, selected, baseline_wrong,
                                placement_examples, structural_examples, pristine, source_hash, copy_lr)
                report["rungs"].append(rung)
                if rung["accepted"]:
                    report["operating_point_found"] = True
                    report["selected_copy_lr"] = float(copy_lr)
                    report["selected_cycle"] = int(rung.get("accepted_cycle", rung["cycles_accepted"]))
                    report["interpretation"] = (
                        "PASS: alternating copy and targeted structure repair found a behavior-constrained path that learns copy placement while retaining the source envelope floor. "
                        "This is a diagnostic operating point and still requires disjoint/generalization validation before any training recipe or promotion."
                    )
                    break
            else:
                report["operating_point_found"] = False
                report["selected_copy_lr"] = None
                report["selected_cycle"] = None
                report["interpretation"] = (
                    "FAIL: none of the bounded copy LRs could accumulate enough copy learning while every accepted cycle stayed inside the 7/8 free-running structure trust region. "
                    "Next inspect which first-generation envelope token moves after the copy update and protect that local decision directly rather than increasing global replay weight."
                )
            report["source_checkpoint_mutated"] = False
            report["checkpoint_saved"] = False
            report["status"] = "COMPLETE"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"event":"alternating_trust_error", "error":report["error"]}), flush=True)
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - started
        diag.write_json(out / "report.json", report)
        (out / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
        if student is not None and pristine is not None:
            student.load_state_dict(pristine)
        print(json.dumps({
            "event":"alternating_trust_complete", "status":report["status"],
            "operating_point_found":report.get("operating_point_found", False),
            "selected_copy_lr":report.get("selected_copy_lr"), "selected_cycle":report.get("selected_cycle"),
            "rungs_run":len(report.get("rungs", [])), "elapsed_seconds":report["elapsed_seconds"],
        }), flush=True)
    if report["status"] != "COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
