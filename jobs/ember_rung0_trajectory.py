"""Observe the published rung-0 trajectory without altering its optimization.

One 40-step CPU run; all familiar cases stay outside the optimization batches.
No checkpoint export, promotion, integration, or further run is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import ember_first_update_diagnostic as diag
from jobs import ember_v050_data as data
from jobs import ember_v050_distill as distill
from jobs import ember_v048_objectives as objectives
from jobs import ember_v048_regression as regression
from jobs.ember_v051_canary import state_digest

base = data.base
CONFIG = ROOT / "config/ember_placement_ladder_v0.0.51.json"
CONFIG_SHA256 = "17374128222b2e00024c8522a7097ec37fdc0697ccbcc7f624907af8e70e0d62"
LADDER_COMMIT = "ea3fe6a9f1ccaac59a91e5c6b3e8ebee70700e1c"
WALL_TIME_LIMIT = 1800
MAXIMUM_WALL_TIME_LIMIT = 3300
DEFAULT_FULL_EVAL_STEPS = (1, 40)


def load_config():
    raw = CONFIG.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("published ladder configuration changed")
    return json.loads(raw)


def full_eval_steps(requested, cfg):
    """Steps whose state gets the full preservation battery.

    The first and last steps are always measured: step 1 is the baseline the
    later states are compared against, and the last step carries the published
    endpoint reproduction. Anything else is opt-in, because each battery costs
    wall time and the trace is bounded.
    """
    steps = {1, cfg["steps_per_rung"]}
    for part in str(requested).split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise ValueError(f"full-evaluation steps must be positive integers, got {part!r}")
        step = int(part)
        if not 1 <= step <= cfg["steps_per_rung"]:
            raise ValueError(f"full-evaluation step {step} is outside 1..{cfg['steps_per_rung']}")
        steps.add(step)
    return steps


def batch_schedule(cfg, nplace, ntool, ncopy):
    """The original run_rung RNG sequence, fixed before any evaluation."""
    rng = random.Random(cfg["seed"])
    return [{name: [rng.randrange(n) for _ in range(cfg[size])]
             for name, n, size in (("placement", nplace, "placement_batch_size"),
                                   ("tool", ntool, "tool_distill_batch_size"),
                                   ("copy", ncopy, "copy_distill_batch_size"))}
            for _ in range(cfg["steps_per_rung"])]


def optimizer_step(student, teacher, tokenizer, torch, cfg, examples, tools, copies, batch, optimizer):
    student.train()
    optimizer.zero_grad(set_to_none=True)
    place = objectives.batch_loss(student, torch, examples, batch["placement"])
    tool = distill.batch_teacher_kl(student, teacher, tokenizer, torch, tools, batch["tool"], cfg["distill_temperature"])
    copied = distill.batch_teacher_kl(student, teacher, tokenizer, torch, copies, batch["copy"], cfg["distill_temperature"])
    total = cfg["placement_loss_weight"] * place + cfg["tool_kl_loss_weight"] * tool + cfg["copy_kl_loss_weight"] * copied
    if not bool(torch.isfinite(total)):
        raise ValueError("non-finite rung-0 loss")
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
    optimizer.step()
    student.zero_grad(set_to_none=True)
    student.eval()
    return {"placement_loss": float(place.detach()), "tool_kl_loss": float(tool.detach()),
            "copy_kl_loss": float(copied.detach()), "combined_loss": float(total.detach()),
            "gradient_norm": float(norm)}


@contextmanager
def observation(model, torch):
    """Probes must not modify parameters/buffers or perturb training RNG."""
    before = state_digest(model)
    rng = torch.get_rng_state().clone()
    py_rng = random.getstate()
    mode = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(mode)
        torch.set_rng_state(rng)
        random.setstate(py_rng)
        if state_digest(model) != before:
            raise ValueError("evaluation modified the model state")


def short_codes(model, tokenizer, torch, cfg):
    rows = []
    for case in regression.v045.frozen_v044_cases():
        if case["kind"] != "short_code":
            continue
        gen = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], cfg["generation_budget"])
        score = regression.control.score_case(case, gen["completion"])
        rows.append({"id": case["id"], "kind": case["kind"], "subtype": case["subtype"],
                     "prompt": case["prompt"], "score": score, **gen})
    return {"rows": rows, "correct_tool": sum(r["score"]["envelope_json_valid"] and r["score"]["tool_name_correct"] for r in rows)}


def placement_learning(before, after, cfg):
    if before["tokens"] != after["tokens"] or before["cases"] != after["cases"]:
        raise ValueError("placement denominators changed")
    checks = {"exact_gain": after["exact_top1"] - before["exact_top1"] >= cfg["operating_point"]["minimum_placement_exact_gain"],
              "token_gain": after["token_top1_rate"] - before["token_top1_rate"] >= cfg["operating_point"]["minimum_placement_token_top1_gain"]}
    return {"passed": all(checks.values()), "checks": checks}


def full_evaluation(model, teacher, tokenizer, torch, cfg, tools, copies, before, placement):
    familiar = diag.familiar(model, tokenizer, torch, cfg)
    reference = regression.references(model, tokenizer, torch, cfg)
    copied = base.copy_diagnostic(model, tokenizer, torch)
    tool_kl = distill.distill_probe(model, teacher, tokenizer, torch, tools,
                                   cfg["tool_distill_batch_size"], cfg["distill_temperature"])
    copy_kl = distill.distill_probe(model, teacher, tokenizer, torch, copies,
                                   cfg["copy_distill_batch_size"], cfg["distill_temperature"])
    result = {"familiar": familiar, "reference": reference, "copy": copied,
              "tool_distill": tool_kl, "copy_distill": copy_kl, "placement": placement}
    if before is None:
        return result
    retention = diag.retained_case_ids(before["familiar"], familiar)
    floors = regression.floor_checks(familiar, {**cfg, "gate": cfg["operating_point"]})
    copy_guard = base.copy_protection(before["copy"], copied)
    learning = placement_learning(before["placement"], placement, cfg)
    checks = {"placement_learning": learning["passed"], "familiar_floors": floors["passed"],
              "all_source_cases_retained": retention["passed"], "copy": copy_guard["passed"],
              "reference": reference["passed"],
              "tool_kl_budget": tool_kl["teacher_kl"] <= cfg["operating_point"]["maximum_tool_teacher_kl"],
              "copy_kl_budget": copy_kl["teacher_kl"] <= cfg["operating_point"]["maximum_copy_teacher_kl"]}
    result.update(retention=retention, familiar_guard=floors, copy_guard=copy_guard,
                  learning_gate=learning, checks=checks, operating_point_found=all(checks.values()),
                  familiar_divergences=diag.divergence_rows(teacher, model, tokenizer, torch,
                      before["familiar"], familiar, retention["lost_ids"]))
    return result


def timing_summary(steps, full):
    first_loss = next((r["step"] for r in steps if not r["short_code_retention"]["passed"]), None)
    first_learning = next((r["step"] for r in steps if r["learning"]["passed"]), None)
    overlap = [r["step"] for r in steps if r["learning"]["passed"] and r["short_code_retention"]["passed"]]
    verified = [int(step) for step, r in full.items() if r["operating_point_found"]]
    unverified = [step for step in overlap if str(step) not in full]
    return {"first_short_code_loss_step": first_loss, "first_placement_learning_step": first_learning,
            "learning_and_short_code_retention_steps": overlap, "fully_passing_steps": verified,
            "overlap_steps_without_full_evaluation": unverified,
            "operating_point_found": bool(verified), "all_potential_windows_evaluated": not unverified}


def summary_markdown(report):
    timing = report.get("timing", {})
    lines = ["# Ember rung-0 CPU trajectory", "", f"Execution: {report['status']}",
             f"Optimizer steps: {report['optimizer_steps_executed']}/40", "",
             "Same source, original ladder data, LR 1e-7 and original 0.20/0.55/0.25 objective weights.",
             "Every-step short-code and placement observations; full guards at steps "
             f"{', '.join(str(s) for s in report.get('full_evaluation_steps', []))} and the first learning-gate pass.", "",
             f"First short-code loss: {timing.get('first_short_code_loss_step')}",
             f"First placement-learning pass: {timing.get('first_placement_learning_step')}",
             f"Fully passing steps: {timing.get('fully_passing_steps', [])}", "",
             "| Step | Short-code cases retained | Placement exact | Placement tokens | Learning gate |",
             "| ---: | ---: | ---: | ---: | --- |"]
    for row in report["steps"]:
        p = row["placement"]
        lines.append(f"| {row['step']} | {row['short_code_retention']['retained']}/8 | {p['exact_top1']}/24 | {p['token_top1']}/{p['tokens']} | {row['learning']['passed']} |")
    lines += ["", "Full evaluation states:"]
    for step, result in report["full_evaluations"].items():
        failed = [k for k, ok in result["checks"].items() if not ok]
        lines.append(f"- Step {step}: familiar {result['familiar']['correct_tool']}/90; copy guard {result['copy_guard']['passed']}; failed checks {failed}.")
    lines += ["", f"Endpoint reproduction: {report.get('endpoint_reproduction')}",
              f"Potential windows without full evaluation: {timing.get('overlap_steps_without_full_evaluation', [])}",
              "No model weights were exported, promoted or integrated. Full reports retain generated responses and first-divergence evidence."]
    if "error" in report:
        lines.append(f"Error: {report['error']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("rung0-trajectory-results"))
    parser.add_argument("--full-eval-steps", default=",".join(str(s) for s in DEFAULT_FULL_EVAL_STEPS),
                        help="comma-separated steps to run the full preservation battery on; "
                             "the first and last steps are always included")
    parser.add_argument("--wall-time-limit", type=int, default=WALL_TIME_LIMIT,
                        help="CPU bound in seconds; raise it when extra batteries are requested")
    args = parser.parse_args()
    cfg = load_config()
    eval_steps = full_eval_steps(args.full_eval_steps, cfg)
    if not 600 <= args.wall_time_limit <= MAXIMUM_WALL_TIME_LIMIT:
        raise ValueError(f"wall-time limit must be 600..{MAXIMUM_WALL_TIME_LIMIT} seconds")
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(cfg["seed"])
    torch.use_deterministic_algorithms(True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def timeout(_signum, _frame):
        raise TimeoutError(f"rung-0 trace exceeded its fixed {args.wall_time_limit}-second CPU bound")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.wall_time_limit)
    report = {"schema_version": 1, "diagnostic": "ember-rung0-trajectory-v1", "status": "ERROR",
              "created_at": datetime.now(timezone.utc).isoformat(), "code_commit": os.environ.get("GITHUB_SHA"),
              "replicated_ladder_commit": LADDER_COMMIT, "config": cfg, "config_sha256": CONFIG_SHA256,
              "optimizer_steps_executed": 0, "steps": [], "full_evaluations": {},
              "full_evaluation_steps": sorted(eval_steps), "wall_time_limit_seconds": args.wall_time_limit,
              "gpu_training_authorized": False, "promotion_authorized": False, "production_authorized": False}
    student = teacher = pristine = None
    try:
        with tempfile.TemporaryDirectory(prefix="ember-rung0-trace-") as td:
            student, tokenizer, source, _splits, source_ref = base.load_inputs(json.loads(base.DEFAULT_CONFIG.read_text()), Path(td), torch)
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != "0.0.31":
                raise ValueError("not the pinned v0.0.31 step-479 source")
            if float(student.cfg.dropout) != 0:
                raise ValueError("zero dropout required to reproduce rung 0 with interleaved evaluation")
            student.to("cpu").eval()
            pristine = copy.deepcopy(student.state_dict())
            source_hash = state_digest(student)
            teacher = copy.deepcopy(student).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            report.update(source=source_ref, source_state_sha256=source_hash)
            del source, _splits
            diag.event("preparation", stage="original_ladder_data")
            values = data.target_values(cfg)
            template, template_report = data.v048d.discover_template(student, tokenizer, torch, cfg, values["template"])
            make = lambda phase, tag: data.v048d.build_cases({k: values[phase][k] for k in sorted(data.PLACEMENT_SUBTYPES)}, tag)
            train = make("train", "v051_place_train")
            development = make("development", "v051_place_dev")
            tools, tool_report = data.prepare_tool_rows(teacher, tokenizer, torch, cfg)
            diag.event("preparation", stage="tool_rows", kept=tool_report)
            copies, copy_report = data.prepare_copy_rows(teacher, tokenizer, torch, cfg)
            token_id, _ = objectives.token_contract(tokenizer)
            examples = [objectives.supervised_example(tokenizer, c, "placement", template, token_id) for c in train]
            schedule = batch_schedule(cfg, len(examples), len(tools), len(copies))
            prepared = {"values": values, "template": template, "template_report": template_report,
                        "train": train, "development": development, "tools": tools, "copies": copies,
                        "tool_report": tool_report, "copy_report": copy_report, "schedule": schedule}
            diag.write_json(output / "data.json", prepared)
            report["data_sha256"] = hashlib.sha256((output / "data.json").read_bytes()).hexdigest()
            report["distillation_coverage"] = {"tool": tool_report, "copy": copy_report,
                "tool_by_subtype": dict(Counter(data.v044.subtype_for(r["kind"], r["target"]) for r in tools))}
            with observation(student, torch):
                placement = objectives.placement_probe(student, tokenizer, torch, development, template)
                before = full_evaluation(student, teacher, tokenizer, torch, cfg, tools, copies, None, placement)
            if (before["familiar"]["correct_tool"], before["familiar"]["envelope_json_valid"]) != (84, 84) or not before["reference"]["passed"]:
                raise ValueError("source failed familiar/reference reproduction")
            if (placement["exact_top1"], placement["token_top1"], placement["tokens"]) != (1, 88, 170):
                raise ValueError("source failed exact published ladder placement reproduction")
            if max(before[k]["teacher_kl"] for k in ("tool_distill", "copy_distill")) > 1e-7:
                raise ValueError("source differs from the frozen teacher")
            report["before"] = before
            before_short = {"rows": [r for r in before["familiar"]["rows"] if r["kind"] == "short_code"]}
            optimizer = torch.optim.AdamW(student.parameters(), lr=cfg["control_learning_rate"], weight_decay=cfg["weight_decay"])
            if optimizer.state or state_digest(student) != source_hash:
                raise ValueError("optimization did not start from the pristine source and empty state")
            diag.event("baseline", familiar=84, placement_exact=1, placement_tokens=88, total_tokens=170)
            first_learning = None
            for step, batch in enumerate(schedule, 1):
                losses = optimizer_step(student, teacher, tokenizer, torch, cfg, examples, tools, copies, batch, optimizer)
                report["optimizer_steps_executed"] = step
                if step == 1 and abs(losses["placement_loss"] - 1.8641787767410278) > 1e-5:
                    raise ValueError("first training batch failed original-rung reproduction")
                with observation(student, torch):
                    place = objectives.placement_probe(student, tokenizer, torch, development, template)
                    short = short_codes(student, tokenizer, torch, cfg)
                    retained = diag.retained_case_ids(before_short, short)
                    learning = placement_learning(placement, place, cfg)
                    row = {"step": step, "losses": losses, "placement": place, "short_codes": short,
                           "short_code_retention": retained, "learning": learning,
                           "short_code_divergences": diag.divergence_rows(teacher, student, tokenizer, torch,
                               before_short, short, retained["lost_ids"])}
                    report["steps"].append(row)
                    first_pass = learning["passed"] and first_learning is None
                    if first_pass:
                        first_learning = step
                    diag.event("trace_step", step=step, retained=retained["retained"], lost_ids=retained["lost_ids"],
                               placement_exact=place["exact_top1"], placement_tokens=place["token_top1"], learning=learning["passed"])
                    if step in eval_steps or first_pass:
                        diag.event("full_evaluation_start", step=step)
                        result = full_evaluation(student, teacher, tokenizer, torch, cfg, tools, copies, before, place)
                        report["full_evaluations"][str(step)] = result
                        diag.event("full_evaluation_complete", step=step, familiar=result["familiar"]["correct_tool"],
                                   retained=result["retention"]["retained"], copy_guard=result["copy_guard"]["passed"], checks=result["checks"])
                diag.write_json(output / "report.json", report)
            del optimizer
            final = report["full_evaluations"]["40"]
            report["endpoint_reproduction"] = {
                "placement_exact": final["placement"]["exact_top1"] == 2,
                "placement_tokens": final["placement"]["token_top1"] == 95,
                "familiar_json": final["familiar"]["envelope_json_valid"] == 82,
                "familiar_tool": final["familiar"]["correct_tool"] == 82,
                "tool_kl": abs(final["tool_distill"]["teacher_kl"] - 0.0007525200498468406) <= 1e-5,
                "copy_kl": abs(final["copy_distill"]["teacher_kl"] - 0.00025023536363733) <= 1e-5,
                "copy_failure": [k for k, ok in final["copy_guard"]["checks"].items() if not ok] == ["expanded_cases_retained"]}
            report["timing"] = timing_summary(report["steps"], report["full_evaluations"])
            report["replication_passed"] = all(report["endpoint_reproduction"].values())
            report["status"] = "COMPLETE" if report["replication_passed"] else "REPLICATION_MISMATCH"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        signal.alarm(0)
        if student is not None and pristine is not None:
            student.load_state_dict(pristine)
            report["pristine_restore_verified"] = state_digest(student) == report["source_state_sha256"]
            report["teacher_unchanged"] = teacher is not None and state_digest(teacher) == report["source_state_sha256"]
            if not report["pristine_restore_verified"] or not report["teacher_unchanged"]:
                report["status"] = "ERROR"
                report["integrity_error"] = "source/teacher final state mismatch"
        report["elapsed_seconds"] = time.monotonic() - started
        diag.write_json(output / "report.json", report)
        (output / "summary.md").write_text(summary_markdown(report))
    diag.event("complete", status=report["status"], timing=report.get("timing"),
               endpoint_reproduction=report.get("endpoint_reproduction"), elapsed_seconds=report["elapsed_seconds"])
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
