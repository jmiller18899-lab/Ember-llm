"""Targeted protected overfit diagnostic for Ember copy placement.

Uses the same eight baseline-failing v0.0.51 development cases and pinned
v0.0.31 step-479 source as the tiny LR ladder. Each rung resets to the pristine
source. Copy placement is trained at LR 6.4e-6 while a masked structural replay
loss protects teacher-generated tool-envelope tokens but never supervises the
query value itself. No checkpoint is saved, promoted, exported, or integrated.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import tempfile
import time

from jobs import ember_rung0_trajectory as trace
from jobs import ember_tiny_overfit as tiny
from jobs import ember_tiny_lr_ladder as ladder


diag = trace.diag
data = trace.data
objectives = trace.objectives
regression = trace.regression
base = trace.base

LR = 6.4e-6
STRUCTURE_WEIGHTS = (0.15, 0.30, 0.45, 0.60)
MAX_STEPS = 24
PROBE_EVERY = 2
ACCEPT_EXACT = 6
ACCEPT_STRUCTURE = 7
WALL_SECONDS = 1500


def _value_span(text: str) -> tuple[int, int]:
    m = re.search(r'"query"\s*:\s*"', text)
    if m is None:
        raise ValueError("no query-value start")
    start = m.end()
    i = start
    while i < len(text):
        ch = text[i]
        if ch == '"':
            return start, i
        if ch == '\\':
            i += 2
        else:
            i += 1
    raise ValueError("no query-value end")


def structural_replay_example(tokenizer, case: dict, generated_ids: list[int], completion: str):
    """Teacher-force structural response tokens while masking the copied value.

    We only keep examples whose source generation is valid JSON and correct-tool.
    Tokens touching the query-value character span are masked, so this protection
    objective cannot reinforce the source model's wrong argument value.
    """
    start_char, end_char = _value_span(completion)
    char_spans = []
    prev = 0
    for i in range(1, len(generated_ids) + 1):
        decoded = tokenizer.decode(generated_ids[:i])
        if not completion.startswith(decoded):
            raise ValueError("generated ids do not round-trip")
        char_spans.append((prev, len(decoded)))
        prev = len(decoded)

    prompt_ids = tokenizer.encode(case["prompt"])
    x = list(prompt_ids) + list(generated_ids[:-1])
    y = [-100] * len(x)
    response_start = len(prompt_ids) - 1
    supervised = 0
    for token_index, token_id in enumerate(generated_ids):
        c0, c1 = char_spans[token_index]
        touches_value = not (c1 <= start_char or c0 >= end_char)
        pos = response_start + token_index
        if pos >= len(y):
            break
        if not touches_value:
            y[pos] = int(token_id)
            supervised += 1
    if supervised < 4:
        raise ValueError("too few structural replay tokens")
    return (x, y), supervised


def prepare_structural_replay(teacher, tokenizer, torch, cfg, selected):
    examples = []
    records = []
    with trace.observation(teacher, torch):
        for case in selected:
            gen = base.semantic_gate.generate_completion(
                teacher, tokenizer, torch, case["prompt"], int(cfg["generation_budget"])
            )
            score = regression.control.score_case(case, gen["completion"])
            if not (score["envelope_json_valid"] and score["tool_name_correct"]):
                records.append({"id": case["id"], "accepted": False, "reason": "source_structure_invalid"})
                continue
            try:
                example, supervised = structural_replay_example(
                    tokenizer, case, [int(v) for v in gen["generated_ids"]], gen["completion"]
                )
            except ValueError as exc:
                records.append({"id": case["id"], "accepted": False, "reason": str(exc)})
                continue
            examples.append(example)
            records.append({"id": case["id"], "accepted": True, "supervised_tokens": supervised})
    if len(examples) < ACCEPT_STRUCTURE:
        raise ValueError(f"need at least {ACCEPT_STRUCTURE} valid structural replay examples, got {len(examples)}")
    return examples, records


def simultaneous_probe(student, tokenizer, torch, cfg, selected, template, baseline_wrong, step):
    with trace.observation(student, torch):
        place = objectives.placement_probe(student, tokenizer, torch, selected, template)
    structure = tiny.structure_probe(student, tokenizer, torch, cfg, selected)
    wrong = tiny.wrong_token_stats(place, baseline_wrong)
    rec = {
        "step": step,
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
    rec["structure_pass"] = rec["json_valid"] >= ACCEPT_STRUCTURE and rec["tool_correct"] >= ACCEPT_STRUCTURE
    rec["simultaneous_pass"] = rec["copy_pass"] and rec["structure_pass"]
    return rec, place, structure


def run_rung(student, teacher, tokenizer, torch, cfg, template, selected, baseline_wrong,
             placement_examples, structural_examples, pristine, source_hash, structure_weight):
    student.load_state_dict(pristine)
    student.eval()
    if trace.state_digest(student) != source_hash:
        raise ValueError("rung reset failed")
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=float(cfg["weight_decay"]))
    pidx = list(range(len(placement_examples)))
    sidx = list(range(len(structural_examples)))
    rung = {
        "learning_rate": LR,
        "structure_weight": float(structure_weight),
        "placement_weight": float(1.0 - structure_weight),
        "steps_executed": 0,
        "accepted": False,
        "probes": [],
        "updates": [],
    }
    rec, _, _ = simultaneous_probe(student, tokenizer, torch, cfg, selected, template, baseline_wrong, 0)
    rung["probes"].append(rec)
    print(json.dumps({"event":"protected_rung_start", "structure_weight":structure_weight, **rec}), flush=True)

    for step in range(1, MAX_STEPS + 1):
        student.train()
        optimizer.zero_grad(set_to_none=True)
        place_loss = objectives.batch_loss(student, torch, placement_examples, pidx)
        structure_loss = objectives.batch_loss(student, torch, structural_examples, sidx)
        total = (1.0 - structure_weight) * place_loss + structure_weight * structure_loss
        if not bool(torch.isfinite(total)):
            raise ValueError("non-finite protected loss")
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            student.parameters(), float(cfg["gradient_clip"]), error_if_nonfinite=True
        )
        before = diag.flat_parameters(student, torch)
        optimizer.step()
        student.zero_grad(set_to_none=True)
        student.eval()
        update_l2 = float((diag.flat_parameters(student, torch) - before).norm())
        if not math.isfinite(update_l2) or update_l2 <= 0:
            raise ValueError("finite nonzero update required")
        rung["steps_executed"] = step
        rung["updates"].append({
            "step": step,
            "placement_loss": float(place_loss.detach()),
            "structure_loss": float(structure_loss.detach()),
            "combined_loss": float(total.detach()),
            "gradient_norm": float(grad_norm),
            "update_l2": update_l2,
        })
        if step % PROBE_EVERY == 0 or step == MAX_STEPS:
            rec, place, structure = simultaneous_probe(
                student, tokenizer, torch, cfg, selected, template, baseline_wrong, step
            )
            rung["probes"].append(rec)
            print(json.dumps({"event":"protected_rung_probe", "structure_weight":structure_weight, **rec}), flush=True)
            if rec["simultaneous_pass"]:
                rung["accepted"] = True
                rung["accepted_step"] = step
                rung["final_selected"] = place
                rung["final_structure"] = structure
                break
    if not rung["accepted"]:
        rec, place, structure = simultaneous_probe(
            student, tokenizer, torch, cfg, selected, template, baseline_wrong, rung["steps_executed"]
        )
        rung["final_selected"] = place
        rung["final_structure"] = structure
    print(json.dumps({
        "event":"protected_rung_complete",
        "structure_weight":structure_weight,
        "steps":rung["steps_executed"],
        "accepted":rung["accepted"],
        "final_exact":rung["final_selected"]["exact_top1"],
        "final_tokens":rung["final_selected"]["token_top1"],
        "json_valid":rung["final_structure"]["envelope_json_valid"],
        "tool_correct":rung["final_structure"]["tool_name_correct"],
    }), flush=True)
    del optimizer
    return rung


def summary_markdown(report):
    lines = [
        "# Ember tiny protected overfit diagnostic", "",
        f"Execution: {report['status']}",
        "Source: pinned v0.0.31 step 479",
        f"Learning rate: {LR:.1e}",
        f"Target gate: >= {ACCEPT_EXACT}/8 exact copies AND >= {ACCEPT_STRUCTURE}/8 valid/correct-tool envelopes.",
        "Structural replay masks every token touching the query value; it protects only envelope syntax/tool structure.",
        "",
        "| Structure weight | Placement weight | Steps | Exact copies | Copy tokens | JSON valid | Tool correct | Pass |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rung in report.get("rungs", []):
        p, s = rung["final_selected"], rung["final_structure"]
        lines.append(
            f"| {rung['structure_weight']:.2f} | {rung['placement_weight']:.2f} | {rung['steps_executed']} | "
            f"{p['exact_top1']}/{p['cases']} | {p['token_top1']}/{p['tokens']} | "
            f"{s['envelope_json_valid']}/{s['cases']} | {s['tool_name_correct']}/{s['cases']} | {rung['accepted']} |"
        )
    lines += ["",
        f"Simultaneous operating point found: {report.get('operating_point_found', False)}",
        f"Selected structure weight: {report.get('selected_structure_weight')}",
        f"Selected step: {report.get('selected_step')}",
        f"Interpretation: {report.get('interpretation', 'not available')}", "",
        "No checkpoint was saved, exported, promoted, or integrated. The source checkpoint is unchanged.",
    ]
    if report.get("error"):
        lines.append(f"Error: {report['error']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("tiny-protected-results"))
    args = parser.parse_args()
    cfg = trace.load_config()
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(int(cfg["seed"]))
    torch.use_deterministic_algorithms(True)
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("protected overfit wall bound")))
    signal.alarm(WALL_SECONDS)
    report = {
        "schema_version":1, "diagnostic":"ember-tiny-protected-overfit-v1", "status":"ERROR",
        "created_at":datetime.now(timezone.utc).isoformat(), "code_commit":os.environ.get("GITHUB_SHA"),
        "learning_rate":LR, "structure_weights":list(STRUCTURE_WEIGHTS), "max_steps_per_rung":MAX_STEPS,
        "accept_exact":ACCEPT_EXACT, "accept_structure":ACCEPT_STRUCTURE,
        "cpu_learning_authorized":True, "gpu_training_authorized":False,
        "promotion_authorized":False, "production_authorized":False, "checkpoint_export_authorized":False,
        "rungs":[],
    }
    student = teacher = pristine = None
    try:
        with tempfile.TemporaryDirectory(prefix="ember-tiny-protected-") as td:
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
            report["template_report"] = template_report
            report["selected_case_ids"] = [c["id"] for c in selected]
            report["selection"] = chosen_rows
            report["selected_baseline"] = selected_baseline
            report["baseline_structure"] = tiny.structure_probe(student, tokenizer, torch, cfg, selected)

            tool_id, _ = objectives.token_contract(tokenizer)
            placement_examples = [objectives.supervised_example(tokenizer, c, "placement", template, tool_id) for c in selected]
            structural_examples, structural_records = prepare_structural_replay(teacher, tokenizer, torch, cfg, selected)
            report["structural_replay"] = structural_records
            report["structural_replay_examples"] = len(structural_examples)

            for sw in STRUCTURE_WEIGHTS:
                rung = run_rung(student, teacher, tokenizer, torch, cfg, template, selected, baseline_wrong,
                                placement_examples, structural_examples, pristine, source_hash, sw)
                report["rungs"].append(rung)
                if rung["accepted"]:
                    report["operating_point_found"] = True
                    report["selected_structure_weight"] = float(sw)
                    report["selected_step"] = int(rung["accepted_step"])
                    report["interpretation"] = (
                        "PASS: targeted non-value structural replay preserved at least 7/8 tool envelopes while the copy objective reached at least 6/8 exact. "
                        "This establishes a bounded simultaneous operating point for further validation, not a production recipe."
                    )
                    break
            else:
                report["operating_point_found"] = False
                report["selected_structure_weight"] = None
                report["selected_step"] = None
                report["interpretation"] = (
                    "FAIL: none of the bounded structural-replay weights simultaneously preserved >=7/8 envelopes and reached >=6/8 exact copies. "
                    "The next diagnostic should alter protection geometry or learning-rate/step coupling rather than change copy supervision."
                )
            report["status"] = "COMPLETE"; report["checkpoint_saved"] = False; report["source_checkpoint_mutated"] = False
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"event":"protected_overfit_error", "error":report["error"]}), flush=True)
    finally:
        signal.alarm(0); report["elapsed_seconds"] = time.monotonic() - started
        diag.write_json(out / "report.json", report)
        (out / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
        if student is not None and pristine is not None: student.load_state_dict(pristine)
        print(json.dumps({"event":"protected_overfit_complete", "status":report["status"],
                          "operating_point_found":report.get("operating_point_found",False),
                          "selected_structure_weight":report.get("selected_structure_weight"),
                          "selected_step":report.get("selected_step"),
                          "elapsed_seconds":report["elapsed_seconds"]}), flush=True)
    if report["status"] != "COMPLETE": raise SystemExit(2)


if __name__ == "__main__":
    main()
