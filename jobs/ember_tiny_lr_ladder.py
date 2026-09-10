"""Bounded CPU learning-rate ladder for Ember copy-placement memorization.

Runs the same eight baseline-failing v0.0.51 development cases from the pinned
v0.0.31 step-479 checkpoint. Each learning-rate rung resets to the pristine
source and a fresh optimizer, uses 100% placement CE, and saves no checkpoint.
The ladder stops at the first rung reaching >=6/8 exact copies.
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


diag = trace.diag
data = trace.data
objectives = trace.objectives
base = trace.base

LRS = (4e-7, 1.6e-6, 6.4e-6)
MAX_STEPS = 24
SELECTED_CASES = 8
ACCEPT_EXACT = 6
PROBE_EVERY = 2
WALL_SECONDS = 1200


def choose_cases(model, tokenizer, torch, cfg, template, development):
    with trace.observation(model, torch):
        full_baseline = objectives.placement_probe(model, tokenizer, torch, development, template)
    if (full_baseline["exact_top1"], full_baseline["token_top1"], full_baseline["tokens"]) != (1, 88, 170):
        raise ValueError("source failed published placement baseline reproduction")
    failing = [row for row in full_baseline["rows"] if not row["exact_top1"]]
    failing.sort(key=lambda r: (
        r["token_top1"] / r["target_token_count"],
        -float(r["mean_loss"]),
        r["id"],
    ))
    chosen_rows = failing[:SELECTED_CASES]
    lookup = {case["id"]: case for case in development}
    selected = [lookup[row["id"]] for row in chosen_rows]
    if len(selected) != SELECTED_CASES:
        raise ValueError("could not select eight failing cases")
    with trace.observation(model, torch):
        selected_baseline = objectives.placement_probe(model, tokenizer, torch, selected, template)
    if selected_baseline["exact_top1"] != 0:
        raise ValueError("selected memorization set is no longer all-failing")
    baseline_wrong = [
        (row["id"], i)
        for row in selected_baseline["rows"]
        for i, token in enumerate(row["tokens"])
        if not token["top1"]
    ]
    return full_baseline, chosen_rows, selected, selected_baseline, baseline_wrong


def probe_record(step, probe, baseline_wrong):
    wrong = tiny.wrong_token_stats(probe, baseline_wrong)
    return {
        "step": step,
        "exact_top1": int(probe["exact_top1"]),
        "cases": int(probe["cases"]),
        "token_top1": int(probe["token_top1"]),
        "tokens": int(probe["tokens"]),
        "token_top1_rate": float(probe["token_top1_rate"]),
        "mean_loss": float(probe["mean_loss"]),
        "baseline_wrong_top1": int(wrong["top1"]),
        "baseline_wrong_tokens": int(wrong["tokens"]),
        "baseline_wrong_mean_rank": float(wrong["mean_rank"]),
        "baseline_wrong_mean_target_probability": float(wrong["mean_target_probability"]),
        "baseline_wrong_mean_target_nll": float(wrong["mean_target_nll"]),
    }


def run_rung(student, tokenizer, torch, cfg, template, selected, baseline_wrong, pristine, source_hash, lr):
    student.load_state_dict(pristine)
    student.eval()
    if trace.state_digest(student) != source_hash:
        raise ValueError("rung reset failed pristine source digest")

    tool_id, _ = objectives.token_contract(tokenizer)
    examples = [objectives.supervised_example(tokenizer, c, "placement", template, tool_id) for c in selected]
    indices = list(range(len(examples)))
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=float(lr), weight_decay=float(cfg["weight_decay"])
    )
    if optimizer.state:
        raise ValueError("rung optimizer must start empty")

    with trace.observation(student, torch):
        baseline = objectives.placement_probe(student, tokenizer, torch, selected, template)
    rung = {
        "learning_rate": float(lr),
        "max_steps": MAX_STEPS,
        "steps_executed": 0,
        "accepted": False,
        "probes": [probe_record(0, baseline, baseline_wrong)],
        "updates": [],
    }
    print(json.dumps({"event": "lr_rung_start", "learning_rate": lr,
                      "exact_top1": baseline["exact_top1"], "token_top1": baseline["token_top1"],
                      "tokens": baseline["tokens"]}), flush=True)

    for step in range(1, MAX_STEPS + 1):
        student.train()
        optimizer.zero_grad(set_to_none=True)
        loss = objectives.batch_loss(student, torch, examples, indices)
        if not bool(torch.isfinite(loss)):
            raise ValueError("non-finite placement loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            student.parameters(), float(cfg["gradient_clip"]), error_if_nonfinite=True
        )
        before_flat = diag.flat_parameters(student, torch)
        optimizer.step()
        student.zero_grad(set_to_none=True)
        student.eval()
        update_l2 = float((diag.flat_parameters(student, torch) - before_flat).norm())
        if not math.isfinite(update_l2) or update_l2 <= 0:
            raise ValueError("finite nonzero update required")
        rung["updates"].append({
            "step": step,
            "placement_loss": float(loss.detach()),
            "gradient_norm": float(grad_norm),
            "update_l2": update_l2,
        })
        rung["steps_executed"] = step

        if step % PROBE_EVERY == 0 or step == MAX_STEPS:
            with trace.observation(student, torch):
                probe = objectives.placement_probe(student, tokenizer, torch, selected, template)
            rec = probe_record(step, probe, baseline_wrong)
            rung["probes"].append(rec)
            print(json.dumps({"event": "lr_rung_probe", "learning_rate": lr, **rec}), flush=True)
            if rec["exact_top1"] >= ACCEPT_EXACT:
                rung["accepted"] = True
                rung["accepted_step"] = step
                break

    with trace.observation(student, torch):
        final_selected = objectives.placement_probe(student, tokenizer, torch, selected, template)
    rung["final_selected"] = final_selected
    rung["final_structure"] = tiny.structure_probe(student, tokenizer, torch, cfg, selected)
    rung["accepted"] = int(final_selected["exact_top1"]) >= ACCEPT_EXACT
    if rung["accepted"] and "accepted_step" not in rung:
        rung["accepted_step"] = rung["steps_executed"]
    print(json.dumps({
        "event": "lr_rung_complete",
        "learning_rate": lr,
        "steps": rung["steps_executed"],
        "accepted": rung["accepted"],
        "final_exact": final_selected["exact_top1"],
        "final_token_top1": final_selected["token_top1"],
        "final_tokens": final_selected["tokens"],
        "json_valid": rung["final_structure"]["envelope_json_valid"],
        "tool_correct": rung["final_structure"]["tool_name_correct"],
    }), flush=True)
    del optimizer
    return rung


def summary_markdown(report):
    lines = [
        "# Ember tiny learning-rate overfit ladder",
        "",
        f"Execution: {report['status']}",
        "Source: pinned v0.0.31 step 479",
        f"Selected failing cases: {len(report.get('selected_case_ids', []))}",
        f"Acceptance: >= {ACCEPT_EXACT}/{SELECTED_CASES} exact copies",
        "Objective: 100% placement CE; each rung resets to pristine source with a fresh optimizer.",
        "",
        "| LR | Steps | Exact copies | Copy tokens | Wrong tokens fixed | JSON valid | Tool correct | Accepted |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rung in report.get("rungs", []):
        f = rung["final_selected"]
        p = rung["probes"][-1]
        s = rung["final_structure"]
        lines.append(
            f"| {rung['learning_rate']:.1e} | {rung['steps_executed']} | {f['exact_top1']}/{f['cases']} | "
            f"{f['token_top1']}/{f['tokens']} | {p['baseline_wrong_top1']}/{p['baseline_wrong_tokens']} | "
            f"{s['envelope_json_valid']}/{s['cases']} | {s['tool_name_correct']}/{s['cases']} | {rung['accepted']} |"
        )
    lines += [
        "",
        f"Operating memorization point found: {report.get('operating_point_found', False)}",
        f"Selected learning rate: {report.get('selected_learning_rate')}",
        f"Interpretation: {report.get('interpretation', 'not available')}",
        "",
        "No checkpoint was saved, exported, promoted, or integrated. The source checkpoint is unchanged.",
    ]
    if report.get("error"):
        lines.append(f"Error: {report['error']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("tiny-lr-ladder-results"))
    args = parser.parse_args()

    cfg = trace.load_config()
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(int(cfg["seed"]))
    torch.use_deterministic_algorithms(True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def timeout(_signum, _frame):
        raise TimeoutError("tiny LR ladder exceeded fixed CPU wall bound")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(WALL_SECONDS)

    report = {
        "schema_version": 1,
        "diagnostic": "ember-tiny-lr-overfit-ladder-v1",
        "status": "ERROR",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": os.environ.get("GITHUB_SHA"),
        "learning_rates": list(LRS),
        "max_steps_per_rung": MAX_STEPS,
        "accept_exact": ACCEPT_EXACT,
        "selected_cases_requested": SELECTED_CASES,
        "objective": {"placement_ce": 1.0, "tool_kl": 0.0, "copy_kl": 0.0, "entry": 0.0, "closing": 0.0},
        "cpu_learning_authorized": True,
        "gpu_training_authorized": False,
        "promotion_authorized": False,
        "production_authorized": False,
        "checkpoint_export_authorized": False,
        "rungs": [],
    }

    student = pristine = None
    try:
        with tempfile.TemporaryDirectory(prefix="ember-tiny-lr-ladder-") as td:
            student, tokenizer, source, _splits, source_ref = base.load_inputs(
                json.loads(base.DEFAULT_CONFIG.read_text()), Path(td), torch
            )
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != "0.0.31":
                raise ValueError("not the pinned v0.0.31 step-479 source")
            if float(student.cfg.dropout) != 0:
                raise ValueError("zero dropout required")
            student.to("cpu").eval()
            pristine = copy.deepcopy(student.state_dict())
            source_hash = trace.state_digest(student)
            report.update(source=source_ref, source_state_sha256=source_hash)
            del source, _splits

            values = data.target_values(cfg)
            template, template_report = data.v048d.discover_template(
                student, tokenizer, torch, cfg, values["template"]
            )
            development = data.v048d.build_cases(
                {k: values["development"][k] for k in sorted(data.PLACEMENT_SUBTYPES)},
                "v051_place_dev",
            )
            full_baseline, chosen_rows, selected, selected_baseline, baseline_wrong = choose_cases(
                student, tokenizer, torch, cfg, template, development
            )
            report["template_report"] = template_report
            report["full_baseline"] = full_baseline
            report["selected_case_ids"] = [c["id"] for c in selected]
            report["selection"] = chosen_rows
            report["selected_baseline"] = selected_baseline
            report["baseline_wrong_tokens"] = len(baseline_wrong)
            report["baseline_structure"] = tiny.structure_probe(student, tokenizer, torch, cfg, selected)

            for lr in LRS:
                rung = run_rung(
                    student, tokenizer, torch, cfg, template, selected, baseline_wrong,
                    pristine, source_hash, lr
                )
                report["rungs"].append(rung)
                if rung["accepted"]:
                    report["operating_point_found"] = True
                    report["selected_learning_rate"] = float(lr)
                    report["interpretation"] = (
                        "PASS: Ember can memorize the copy-placement task from the pristine source at this bounded learning rate. "
                        "The copy representation and supervision are therefore learnable; the next problem is preserving envelope/tool behavior while reintroducing mixed-objective protection."
                    )
                    break
            else:
                report["operating_point_found"] = False
                report["selected_learning_rate"] = None
                report["interpretation"] = (
                    "FAIL: none of the bounded learning-rate rungs reached 6/8 exact copies. "
                    "The next diagnostic should inspect supervision alignment/loss masking or model capacity rather than raise mixed-training dose."
                )

            report["source_checkpoint_mutated"] = False
            report["checkpoint_saved"] = False
            report["status"] = "COMPLETE"

    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"event": "tiny_lr_ladder_error", "error": report["error"]}), flush=True)
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - started
        diag.write_json(output / "report.json", report)
        (output / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
        if student is not None and pristine is not None:
            student.load_state_dict(pristine)
        print(json.dumps({
            "event": "tiny_lr_ladder_complete",
            "status": report["status"],
            "operating_point_found": report.get("operating_point_found", False),
            "selected_learning_rate": report.get("selected_learning_rate"),
            "rungs_run": len(report.get("rungs", [])),
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)

    if report["status"] != "COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
