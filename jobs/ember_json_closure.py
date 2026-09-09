"""Bounded CPU test of fresh JSON-closing margin projection on Ember rung 0."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import ember_rung0_trajectory as trace

diag, data, objectives, regression, base = trace.diag, trace.data, trace.objectives, trace.regression, trace.base
PROBE_STEPS = {1, 12, 13, 23, 40}
MAX_ATTEMPTS = 48
WALL_SECONDS = 1800
MAX_RESIDUAL = .001
PRIOR_VALUES = ROOT / "config/ember_json_closure_prior_values.json"
PROJECTION_MODE = "projection"
NORM_MATCHED_MODE = "norm-matched"


def closure_position(tokenizer, ids, field="query"):
    """Locate the first closing-quote token without constraining value tokens.

    Reject a token that straddles the final value characters and closing quote.
    Escaped quotes inside the value are parsed with the JSON decoder.
    """
    text = tokenizer.decode(ids)
    match = re.search(r'"' + re.escape(field) + r'"\s*:\s*', text)
    if match is None:
        raise ValueError("argument field not found")
    value, consumed = json.JSONDecoder().raw_decode(text[match.end():])
    if not isinstance(value, str):
        raise ValueError("argument must be a JSON string")
    closing = match.end() + consumed - 1
    if text[closing] != '"':
        raise ValueError("closing quote not found")
    for i in range(len(ids)):
        prefix = tokenizer.decode(ids[:i])
        extended = tokenizer.decode(ids[:i+1])
        if len(prefix) <= closing < len(extended):
            if prefix != text[:closing]:
                raise ValueError("closing token also contains value characters or unstable decoding")
            if not extended.startswith(prefix + '"'):
                raise ValueError("closing token does not start with the closing quote")
            return i
    raise ValueError("closing token could not be aligned")


def fresh_value(phase, variant, index, used):
    for nonce in range(10000):
        digest = data.copy_data._digest("json-closure-20260909-v1", phase, variant, index, nonce)
        value = data.copy_data._render("short_code", variant, digest)
        if value not in used:
            used.add(value)
            return value
    raise ValueError("fresh target namespace exhausted")


def prepare_rows(teacher, tokenizer, torch):
    used = diag.historical_values()
    used.update(json.loads(PRIOR_VALUES.read_text())["excluded_values"])
    excluded = set(used)
    cohorts, attempts = {}, []
    for phase, quotas in (("anchors", (6, 2)), ("holdout", (4, 2))):
        rows = []
        for variant, quota in enumerate(quotas):
            kept = 0
            for i in range(MAX_ATTEMPTS):
                target = fresh_value(phase, variant, i, used)
                case = {"id": f"json_{phase}_{variant}_{i:02d}", **data.tool_prompt("short_code", target)}
                gen = base.semantic_gate.generate_completion(teacher, tokenizer, torch, case["prompt"], 96)
                score = regression.control.score_case(case, gen["completion"])
                accepted = bool(score["envelope_json_valid"] and score["tool_name_correct"] and gen["generated_ids"])
                reason, position = None, None
                if accepted:
                    try:
                        position = closure_position(tokenizer, gen["generated_ids"])
                    except ValueError as exc:
                        accepted, reason = False, str(exc)
                attempts.append({"phase": phase, "variant": variant, "target": target, "accepted": accepted, "rejection": reason})
                if accepted:
                    rows.append({**case, "variant": variant, "source_ids": gen["generated_ids"],
                                 "source_completion": gen["completion"], "source_score": score,
                                 "closing_position": position})
                    kept += 1
                if kept == quota:
                    break
            diag.event("closure_quota", phase=phase, variant=variant, kept=kept, required=quota)
            if kept != quota:
                raise ValueError(f"fresh structurally correct closure quota failed: {phase}/{variant}")
        cohorts[phase] = rows
    targets = [r["target"] for rows in cohorts.values() for r in rows]
    if len(targets) != len(set(targets)) or set(targets) & excluded:
        raise ValueError("fresh closure cohorts overlap each other or previous values")
    return cohorts, attempts, len(excluded)


def build_basis(student, tokenizer, torch, rows):
    basis, anchors = [], []
    for row in rows:
        position = row["closing_position"]
        target_id = row["source_ids"][position]
        prefix = tokenizer.encode(row["prompt"]) + row["source_ids"][:position]
        logits, _ = student(torch.tensor([prefix], dtype=torch.long))
        v = logits[0, -1]
        if int(v.detach().argmax()) != target_id:
            raise ValueError("source closing token is not reproduced at its own prefix")
        others = v.detach().clone()
        others[target_id] = -torch.inf
        alternative = int(others.argmax())
        margin = v[target_id] - v[alternative]
        if float(margin.detach()) <= 0:
            raise ValueError("strictly positive source closing margin required")
        grad = diag.gradient_vector(margin, student, torch)
        independent = diag.add_basis(grad, basis, torch)
        anchors.append({"id": row["id"], "target": row["target"], "prefix_ids": prefix,
                        "response_position": position, "source_token_id": target_id,
                        "source_token": tokenizer.decode([target_id]), "alternative_token_id": alternative,
                        "alternative_token": tokenizer.decode([alternative]), "source_margin": float(margin.detach()),
                        "independent_direction": independent})
        diag.event("closure_direction", id=row["id"], rank=len(basis), margin=float(margin.detach()))
    if len(basis) != len(rows):
        raise ValueError("the eight prescribed closure directions were not independent")
    return basis, anchors


def apply_projected_proposal(student, before, raw, basis, torch):
    projected = diag.project_update(raw, basis, torch)
    raw_norm = float(raw.norm())
    projected_norm = float(projected.norm())
    if not all(math.isfinite(n) for n in (raw_norm, projected_norm)) or raw_norm == 0:
        raise ValueError("finite nonzero optimizer proposal required")
    mathematical = sum(float(torch.dot(q, projected))**2 for q in basis)**.5 / max(projected_norm, 1e-30)
    if projected_norm > raw_norm * (1+1e-5) or mathematical > 1e-5:
        raise ValueError("mathematical projection failed")
    diag.assign_vector(student, before + projected, torch)
    actual = diag.flat_parameters(student, torch) - before
    norm = float(actual.norm())
    residual = sum(float(torch.dot(q, actual))**2 for q in basis)**.5 / max(norm, 1e-30)
    if not all(math.isfinite(n) for n in (norm, residual)) or norm == 0 or residual > MAX_RESIDUAL:
        raise ValueError(f"actual float32 projection failed: norm={norm}, residual={residual}")
    return {"raw_l2": raw_norm, "projected_l2": projected_norm, "actual_l2": norm,
            "retained_norm_fraction": projected_norm/raw_norm,
            "actual_residual_fraction": residual, "mathematical_residual_fraction": mathematical,
            "direction_cosine": float(torch.dot(actual, raw))/max(norm*raw_norm, 1e-30)}


def apply_norm_matched_proposal(student, before, raw, basis, torch):
    """The control arm: the projection's magnitude schedule, its direction removal dropped.

    The scale is the fraction the projection would have retained on this same
    proposal, so the two arms shrink each step identically and differ only in
    whether the closure directions are removed. It is recomputed here rather than
    replayed from the projection run, whose per-step fractions were not published;
    the realized fractions are reported so the two magnitude profiles can be
    compared.
    """
    projected = diag.project_update(raw, basis, torch)
    raw_norm = float(raw.norm())
    projected_norm = float(projected.norm())
    del projected
    if not all(math.isfinite(n) for n in (raw_norm, projected_norm)) or raw_norm == 0:
        raise ValueError("finite nonzero optimizer proposal required")
    if projected_norm > raw_norm * (1+1e-5):
        raise ValueError("projected norm exceeded the proposal it came from")
    scale = projected_norm/raw_norm
    diag.assign_vector(student, before + raw*scale, torch)
    actual = diag.flat_parameters(student, torch) - before
    norm = float(actual.norm())
    if not math.isfinite(norm) or norm == 0:
        raise ValueError(f"norm-matched update failed: norm={norm}")
    cosine = float(torch.dot(actual, raw))/max(norm*raw_norm, 1e-30)
    residual = sum(float(torch.dot(q, actual))**2 for q in basis)**.5 / max(norm, 1e-30)
    if abs(norm - projected_norm) > max(1e-5*raw_norm, 1e-12) or cosine < 1-1e-5:
        raise ValueError(f"norm-matched update missed its target: norm={norm}, cosine={cosine}")
    return {"raw_l2": raw_norm, "projected_l2": projected_norm, "actual_l2": norm,
            "retained_norm_fraction": scale, "realized_norm_fraction": norm/raw_norm,
            "actual_residual_fraction": residual, "mathematical_residual_fraction": None,
            "direction_cosine": cosine}


def fresh_probe(student, tokenizer, torch, rows):
    output = []
    for row in rows:
        gen = base.semantic_gate.generate_completion(student, tokenizer, torch, row["prompt"], 96)
        score = regression.control.score_case(row, gen["completion"])
        output.append({"id": row["id"], "target": row["target"], "score": score, **gen})
    passed = sum(r["score"]["envelope_json_valid"] and r["score"]["tool_name_correct"] for r in output)
    return {"rows": output, "cases": len(output), "structurally_correct": passed,
            "all_retained": passed == len(output), "exact_arguments": sum(r["score"]["slot_exact"] for r in output)}


def final_checks(full, fresh, anchor_margins, updates, mode=PROJECTION_MODE):
    checks = dict(full["checks"])
    checks.update(fresh_anchors_retained=fresh["anchors"]["all_retained"],
                  fresh_holdout_retained=fresh["holdout"]["all_retained"],
                  protected_closing_tokens_still_top1=all(r["source_token_still_top1"] for r in anchor_margins))
    stats = [r["projection"] for r in updates]
    if mode == PROJECTION_MODE:
        checks["all_40_nonzero_projected_updates"] = len(updates)==40 and all(
            r["actual_l2"]>0 and r["actual_residual_fraction"]<=MAX_RESIDUAL for r in stats)
    else:
        checks["all_40_nonzero_norm_matched_updates"] = len(updates)==40 and all(
            r["actual_l2"]>0 and r["direction_cosine"]>=1-1e-5
            and abs(r["realized_norm_fraction"]-r["retained_norm_fraction"])<=1e-5 for r in stats)
    return checks


def summary(report):
    lines = [f"# Ember JSON-closing CPU result ({report.get('mode', PROJECTION_MODE)} arm)", "", f"Execution: {report['status']}",
             f"Steps: {len(report['updates'])}/40", f"Endpoint passed: {report.get('endpoint_passed', False)}", "",
             "| Observed step | Short-code cases retained | Exact placement | Placement tokens | Learning gate |",
             "| ---: | ---: | ---: | ---: | --- |"]
    for r in report["probes"]:
        lines.append(f"| {r['step']} | {r['retention']['retained']}/8 | {r['placement']['exact_top1']}/24 | {r['placement']['token_top1']}/170 | {r['learning']['passed']} |")
    if "checks" in report:
        lines += ["", "Failed endpoint checks: " + ", ".join(k for k,v in report["checks"].items() if not v)]
    lines += ["", "Only steps 1, 12, 13, 23 and 40 were probed; full familiar/copy/reference/KL guards were run on the source and step 40.",
              "Fixed source-derived closing-token directions from eight fresh examples; six other fresh examples never supply gradients."
              if report.get("mode", PROJECTION_MODE) == PROJECTION_MODE else
              "Control arm: the same per-step shrinkage as the projection, with its direction removal dropped.",
              "This is an endpoint experiment, not an exhaustive timing search or promotion evaluation. No checkpoint was exported or integrated."]
    if "error" in report:
        lines.append(f"Error: {report['error']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("json-closure-results"))
    parser.add_argument("--mode", choices=(PROJECTION_MODE, NORM_MATCHED_MODE), default=PROJECTION_MODE,
                        help="projection removes the closure directions; norm-matched keeps them and "
                             "applies only the same per-step shrinkage, isolating direction from magnitude")
    args = parser.parse_args()
    cfg = trace.load_config()
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(cfg["seed"])
    torch.use_deterministic_algorithms(True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def timeout(_signum, _frame):
        raise TimeoutError("JSON closure experiment exceeded its 30-minute CPU bound")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(WALL_SECONDS)
    report = {"status": "ERROR", "experiment": "ember-json-closure-v1", "mode": args.mode, "created_at": datetime.now(timezone.utc).isoformat(),
              "code_commit": os.environ.get("GITHUB_SHA"), "original_recipe": cfg, "updates": [], "probes": [],
              "gpu_training_authorized": False, "promotion_authorized": False, "production_authorized": False,
              "comparison_run_id": 34369464055, "probe_steps": sorted(PROBE_STEPS)}
    student = teacher = pristine = None
    try:
        with tempfile.TemporaryDirectory(prefix="ember-json-closure-") as td:
            student, tokenizer, source, splits, source_ref = base.load_inputs(json.loads(base.DEFAULT_CONFIG.read_text()), Path(td), torch)
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != "0.0.31" or float(student.cfg.dropout) != 0:
                raise ValueError("pinned v0.0.31 step 479 with zero dropout required")
            student.to("cpu").eval()
            report["source"], report["source_state_sha256"] = source_ref, trace.state_digest(student)
            pristine = copy.deepcopy(student.state_dict())
            teacher = copy.deepcopy(student).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            del source, splits
            cohorts, attempts, excluded_count = prepare_rows(teacher, tokenizer, torch)
            basis, anchors = build_basis(student, tokenizer, torch, cohorts["anchors"])
            report["projection"] = {"rank": len(basis), "anchors": anchors, "fixed_at_source": True,
                                    "mode": args.mode, "directions_removed": args.mode == PROJECTION_MODE}
            values = data.target_values(cfg)
            template, template_report = data.v048d.discover_template(student, tokenizer, torch, cfg, values["template"])
            make = lambda phase, tag: data.v048d.build_cases({k:values[phase][k] for k in sorted(data.PLACEMENT_SUBTYPES)}, tag)
            train, dev = make("train", "v051_place_train"), make("development", "v051_place_dev")
            tools, tool_report = data.prepare_tool_rows(teacher, tokenizer, torch, cfg)
            copies, copy_report = data.prepare_copy_rows(teacher, tokenizer, torch, cfg)
            token_id, _ = objectives.token_contract(tokenizer)
            examples = [objectives.supervised_example(tokenizer,c,"placement",template,token_id) for c in train]
            schedule = trace.batch_schedule(cfg,len(examples),len(tools),len(copies))
            diag.write_json(output/"data.json", {"cohorts":cohorts,"attempts":attempts,"excluded_count":excluded_count,
                "values":values,"template":template,"template_report":template_report,"tools":tools,"copies":copies,
                "tool_report":tool_report,"copy_report":copy_report,"schedule":schedule,"anchors":anchors})
            report["data_sha256"] = hashlib.sha256((output/"data.json").read_bytes()).hexdigest()
            if trace.state_digest(student) != report["source_state_sha256"]:
                raise ValueError("preparation changed the pristine model")
            with trace.observation(student,torch):
                place = objectives.placement_probe(student,tokenizer,torch,dev,template)
                before = trace.full_evaluation(student,teacher,tokenizer,torch,cfg,tools,copies,None,place)
            if (before["familiar"]["correct_tool"],before["familiar"]["envelope_json_valid"],place["exact_top1"],place["token_top1"],place["tokens"]) != (84,84,1,88,170) or not before["reference"]["passed"]:
                raise ValueError("original baseline reproduction failed")
            report["before"] = before
            before_short = {"rows":[r for r in before["familiar"]["rows"] if r["kind"]=="short_code"]}
            optimizer = torch.optim.AdamW(student.parameters(),lr=cfg["control_learning_rate"],weight_decay=cfg["weight_decay"])
            if optimizer.state:
                raise ValueError("optimizer must start empty")
            diag.event("baseline", familiar=84, placement_exact=1, placement_tokens=88, closure_directions=len(basis))
            for step,batch in enumerate(schedule,1):
                theta = diag.flat_parameters(student,torch)
                losses = trace.optimizer_step(student,teacher,tokenizer,torch,cfg,examples,tools,copies,batch,optimizer)
                raw = diag.flat_parameters(student,torch)-theta
                apply = apply_projected_proposal if args.mode == PROJECTION_MODE else apply_norm_matched_proposal
                stats = apply(student,theta,raw,basis,torch)
                del theta,raw
                report["updates"].append({"step":step,"losses":losses,"projection":stats})
                if step==1 and abs(losses["placement_loss"]-1.8641787767410278)>1e-5:
                    raise ValueError("first original training batch failed reproduction")
                if step in PROBE_STEPS:
                    with trace.observation(student,torch):
                        placement = objectives.placement_probe(student,tokenizer,torch,dev,template)
                        short = trace.short_codes(student,tokenizer,torch,cfg)
                        retained = diag.retained_case_ids(before_short,short)
                        margins = diag.measure_anchors(student,torch,anchors)
                        fresh = {phase:fresh_probe(student,tokenizer,torch,rows) for phase,rows in cohorts.items()}
                        row = {"step":step,"placement":placement,"short_codes":short,"retention":retained,
                               "learning":trace.placement_learning(place,placement,cfg),"anchor_margins":margins,"fresh":fresh,
                               "divergences":diag.divergence_rows(teacher,student,tokenizer,torch,before_short,short,retained["lost_ids"])}
                        report["probes"].append(row)
                        diag.event("closure_probe",step=step,retained=retained["retained"],lost=retained["lost_ids"],
                                   placement_exact=placement["exact_top1"],placement_tokens=placement["token_top1"],
                                   fresh_anchors=fresh["anchors"]["structurally_correct"],fresh_holdout=fresh["holdout"]["structurally_correct"])
                diag.write_json(output/"report.json",report)
            del optimizer
            with trace.observation(student,torch):
                final = trace.full_evaluation(student,teacher,tokenizer,torch,cfg,tools,copies,before,report["probes"][-1]["placement"])
            report["final"] = final
            report["checks"] = final_checks(final,fresh,margins,report["updates"],args.mode)
            report["endpoint_passed"] = all(report["checks"].values())
            report["status"] = "COMPLETE"
    except Exception as exc:
        report["error"] = {"type":type(exc).__name__,"message":str(exc)}
        raise
    finally:
        signal.alarm(0)
        if student is not None and pristine is not None:
            student.load_state_dict(pristine)
            report["pristine_restore_verified"] = trace.state_digest(student)==report["source_state_sha256"]
            report["teacher_unchanged"] = teacher is not None and trace.state_digest(teacher)==report["source_state_sha256"]
            if not report["pristine_restore_verified"] or not report["teacher_unchanged"]:
                report["status"] = "ERROR"
                report["endpoint_passed"] = False
        report["elapsed_seconds"] = time.monotonic()-started
        diag.write_json(output/"report.json",report)
        (output/"summary.md").write_text(summary(report))
    diag.event("complete",status=report["status"],endpoint_passed=report.get("endpoint_passed"),checks=report.get("checks"),elapsed_seconds=report["elapsed_seconds"])
    return 0 if report["status"]=="COMPLETE" else 1


if __name__=="__main__":
    raise SystemExit(main())
