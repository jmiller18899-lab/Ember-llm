"""Bounded CPU memorization diagnostic for Ember placement copying.

Intentionally trains on eight baseline-failing development cases to answer one
question: can the pinned v0.0.31 step-479 model memorize the copy placement
when preservation objectives are removed? This is diagnostic leakage by design.
No checkpoint is exported, promoted, or integrated.
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

ROOT = Path(__file__).resolve().parents[1]

from jobs import ember_rung0_trajectory as trace

diag = trace.diag
data = trace.data
objectives = trace.objectives
regression = trace.regression
base = trace.base

STEPS = 24
SELECTED_CASES = 8
PROBE_STEPS = {0, 4, 8, 12, 16, 20, 24}
ACCEPT_EXACT = 6  # 75% of the deliberately tiny memorization set.
WALL_SECONDS = 1200


def structure_probe(model, tokenizer, torch, cfg, cases):
    rows = []
    with trace.observation(model, torch):
        for case in cases:
            gen = base.semantic_gate.generate_completion(
                model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"])
            )
            score = regression.control.score_case(case, gen["completion"])
            rows.append({
                "id": case["id"],
                "envelope_json_valid": bool(score["envelope_json_valid"]),
                "tool_name_correct": bool(score["tool_name_correct"]),
                "slot_exact": bool(score["slot_exact"]),
                "completion": gen["completion"],
            })
    return {
        "cases": len(rows),
        "envelope_json_valid": sum(r["envelope_json_valid"] for r in rows),
        "tool_name_correct": sum(r["tool_name_correct"] for r in rows),
        "slot_exact": sum(r["slot_exact"] for r in rows),
        "rows": rows,
    }


def wrong_token_stats(probe, baseline_wrong):
    by_id = {row["id"]: row for row in probe["rows"]}
    observations = []
    for case_id, token_index in baseline_wrong:
        token = by_id[case_id]["tokens"][token_index]
        loss = float(token["loss"])
        observations.append({
            "id": case_id,
            "token_index": token_index,
            "top1": bool(token["top1"]),
            "rank": int(token["rank"]),
            "target_probability": math.exp(-loss),
            "target_nll": loss,
        })
    return {
        "tokens": len(observations),
        "top1": sum(o["top1"] for o in observations),
        "mean_rank": sum(o["rank"] for o in observations) / len(observations),
        "mean_target_probability": sum(o["target_probability"] for o in observations) / len(observations),
        "mean_target_nll": sum(o["target_nll"] for o in observations) / len(observations),
        "rows": observations,
    }


def compact_probe(step, probe, baseline_wrong):
    wrong = wrong_token_stats(probe, baseline_wrong)
    event = {
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
    print(json.dumps({"event": "tiny_overfit_probe", **event}), flush=True)
    return {**event, "placement": probe, "baseline_wrong": wrong}


def summary_markdown(report):
    lines = [
        "# Ember tiny overfit diagnostic",
        "",
        f"Execution: {report['status']}",
        f"Source: pinned v0.0.31 step 479",
        f"Selected failing cases: {len(report.get('selected_case_ids', []))}",
        f"Optimizer steps: {report.get('optimizer_steps_executed', 0)}/{STEPS}",
        f"Learning rate: {report.get('learning_rate')}",
        "Objective: 100% placement CE; no tool KL, copy KL, entry loss, or closing loss.",
        "",
        "| Step | Exact copies | Copy tokens | Baseline-wrong tokens fixed | Mean target probability on baseline-wrong tokens |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for p in report.get("probes", []):
        lines.append(
            f"| {p['step']} | {p['exact_top1']}/{p['cases']} | {p['token_top1']}/{p['tokens']} | "
            f"{p['baseline_wrong_top1']}/{p['baseline_wrong_tokens']} | {p['baseline_wrong_mean_target_probability']:.6f} |"
        )
    if report.get("baseline_structure"):
        b, f = report["baseline_structure"], report.get("final_structure", {})
        lines += [
            "",
            f"Generated structure: baseline JSON {b['envelope_json_valid']}/{b['cases']}, tool {b['tool_name_correct']}/{b['cases']}; "
            f"final JSON {f.get('envelope_json_valid')}/{f.get('cases')}, tool {f.get('tool_name_correct')}/{f.get('cases')}.",
        ]
    lines += [
        "",
        f"Acceptance (>= {ACCEPT_EXACT}/{SELECTED_CASES} exact on the memorization set): {report.get('accepted', False)}",
        f"Interpretation: {report.get('interpretation', 'not available')}",
        "",
        "No checkpoint was saved, exported, promoted, or integrated. The source checkpoint is unchanged.",
    ]
    if report.get("error"):
        lines.append(f"Error: {report['error']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("tiny-overfit-results"))
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
        raise TimeoutError("tiny overfit exceeded its fixed CPU wall bound")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(WALL_SECONDS)

    report = {
        "schema_version": 1,
        "diagnostic": "ember-tiny-overfit-v1",
        "status": "ERROR",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": os.environ.get("GITHUB_SHA"),
        "optimizer_steps_executed": 0,
        "steps_requested": STEPS,
        "selected_cases_requested": SELECTED_CASES,
        "accept_exact": ACCEPT_EXACT,
        "learning_rate": float(cfg["control_learning_rate"]),
        "gradient_clip": float(cfg["gradient_clip"]),
        "weight_decay": float(cfg["weight_decay"]),
        "objective": {"placement_ce": 1.0, "tool_kl": 0.0, "copy_kl": 0.0, "entry": 0.0, "closing": 0.0},
        "cpu_learning_authorized": True,
        "gpu_training_authorized": False,
        "promotion_authorized": False,
        "production_authorized": False,
        "checkpoint_export_authorized": False,
        "probes": [],
        "updates": [],
    }

    student = pristine = None
    try:
        with tempfile.TemporaryDirectory(prefix="ember-tiny-overfit-") as td:
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
            make = lambda phase, tag: data.v048d.build_cases(
                {k: values[phase][k] for k in sorted(data.PLACEMENT_SUBTYPES)}, tag
            )
            development = make("development", "v051_place_dev")

            with trace.observation(student, torch):
                full_baseline = objectives.placement_probe(student, tokenizer, torch, development, template)
            if (full_baseline["exact_top1"], full_baseline["token_top1"], full_baseline["tokens"]) != (1, 88, 170):
                raise ValueError("source failed published placement baseline reproduction")

            failing = [row for row in full_baseline["rows"] if not row["exact_top1"]]
            failing.sort(key=lambda r: (
                r["token_top1"] / r["target_token_count"],
                -float(r["mean_loss"]),
                r["id"],
            ))
            chosen_rows = failing[:SELECTED_CASES]
            selected_ids = [row["id"] for row in chosen_rows]
            selected_lookup = {case["id"]: case for case in development}
            selected = [selected_lookup[i] for i in selected_ids]
            if len(selected) != SELECTED_CASES:
                raise ValueError("could not select eight failing cases")

            with trace.observation(student, torch):
                selected_baseline = objectives.placement_probe(student, tokenizer, torch, selected, template)
            baseline_wrong = [
                (row["id"], i)
                for row in selected_baseline["rows"]
                for i, token in enumerate(row["tokens"])
                if not token["top1"]
            ]
            if not baseline_wrong:
                raise ValueError("selected set unexpectedly has no incorrect tokens")

            report["template_report"] = template_report
            report["selected_case_ids"] = selected_ids
            report["selection"] = chosen_rows
            report["full_baseline"] = full_baseline
            report["baseline_wrong_tokens"] = len(baseline_wrong)
            report["probes"].append(compact_probe(0, selected_baseline, baseline_wrong))
            report["baseline_structure"] = structure_probe(student, tokenizer, torch, cfg, selected)

            tool_id, _ = objectives.token_contract(tokenizer)
            examples = [
                objectives.supervised_example(tokenizer, case, "placement", template, tool_id)
                for case in selected
            ]
            indices = list(range(len(examples)))
            optimizer = torch.optim.AdamW(
                student.parameters(),
                lr=float(cfg["control_learning_rate"]),
                weight_decay=float(cfg["weight_decay"]),
            )
            if optimizer.state or trace.state_digest(student) != source_hash:
                raise ValueError("optimizer did not start from pristine source")

            for step in range(1, STEPS + 1):
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
                report["updates"].append({
                    "step": step,
                    "placement_loss": float(loss.detach()),
                    "gradient_norm": float(grad_norm),
                    "update_l2": update_l2,
                })
                report["optimizer_steps_executed"] = step

                if step in PROBE_STEPS:
                    with trace.observation(student, torch):
                        probe = objectives.placement_probe(student, tokenizer, torch, selected, template)
                    report["probes"].append(compact_probe(step, probe, baseline_wrong))

            del optimizer
            with trace.observation(student, torch):
                final_selected = objectives.placement_probe(student, tokenizer, torch, selected, template)
                final_full = objectives.placement_probe(student, tokenizer, torch, development, template)
            report["final_selected"] = final_selected
            report["final_full_placement"] = final_full
            report["final_structure"] = structure_probe(student, tokenizer, torch, cfg, selected)

            accepted = int(final_selected["exact_top1"]) >= ACCEPT_EXACT
            report["accepted"] = accepted
            if accepted:
                report["interpretation"] = (
                    "PASS: Ember can memorize the copy placements once preservation competition and placement down-weighting are removed; "
                    "the next experiment should rebalance the mixed objective rather than rewrite token alignment."
                )
            else:
                report["interpretation"] = (
                    "FAIL: removing preservation competition was insufficient at the original control LR; do not increase the mixed-objective weight yet. "
                    "Next isolate supervision alignment/loss masking, or run a separately bounded learning-rate memorization check before changing production training."
                )
            report["source_checkpoint_mutated"] = False
            report["checkpoint_saved"] = False
            report["status"] = "COMPLETE"

    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"event": "tiny_overfit_error", "error": report["error"]}), flush=True)
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - started
        diag.write_json(output / "report.json", report)
        (output / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
        if student is not None and pristine is not None:
            student.load_state_dict(pristine)
        print(json.dumps({
            "event": "tiny_overfit_complete",
            "status": report["status"],
            "accepted": report.get("accepted", False),
            "steps": report.get("optimizer_steps_executed", 0),
            "final_exact": report.get("final_selected", {}).get("exact_top1"),
            "final_token_top1": report.get("final_selected", {}).get("token_top1"),
            "final_tokens": report.get("final_selected", {}).get("tokens"),
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)

    if report["status"] != "COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
