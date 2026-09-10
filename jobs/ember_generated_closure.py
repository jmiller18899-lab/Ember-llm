"""Bounded CPU test of closing punctuation after candidate-generated argument text."""
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
from jobs import ember_json_closure as closure

diag, data, objectives, regression, base = trace.diag, trace.data, trace.objectives, trace.regression, trace.base
PROBE_STEPS = {1, 12, 13, 23, 40}
WALL_SECONDS = 1800


CLOSURE_WEIGHT = .20
REFRESH_STEPS = set(range(1, 41, 5))
PRIOR_VALUES = ROOT / "config/ember_generated_closure_prior_values.json"


def fresh_value(phase, variant, index, used):
    for nonce in range(10000):
        digest = data.copy_data._digest("generated-closure-20260909-v1", phase, variant, index, nonce)
        value = data.copy_data._render("short_code", variant, digest)
        if value not in used:
            used.add(value)
            return value
    raise ValueError("fresh target namespace exhausted")


def prepare_rows(teacher, tokenizer, torch):
    # Reuse source structural quotas, but a new namespace and complete exclusions.
    old_value, old_prior = closure.fresh_value, closure.PRIOR_VALUES
    try:
        closure.fresh_value, closure.PRIOR_VALUES = fresh_value, PRIOR_VALUES
        return closure.prepare_rows(teacher, tokenizer, torch)
    finally:
        closure.fresh_value, closure.PRIOR_VALUES = old_value, old_prior


def generated_value_end(text):
    """Accept only the expected tool envelope and a complete escaped string body.

    An unescaped quote or newline/EOS marks the value's end. Never guess a
    boundary for budget-truncated output, incomplete escapes or bad scaffolding.
    """
    match = re.match(r'^\s*<\|tool\|>\s*\{"arguments":\{"query":"', text)
    if match is None:
        raise ValueError("unexpected argument envelope")
    start, i = match.end(), match.end()
    while i < len(text):
        ch = text[i]
        if ch == '"' or ch in '\r\n' or text.startswith('<|endoftext|>', i):
            json.loads('"' + text[start:i] + '"')
            return i, ('closed' if ch == '"' else 'unclosed')
        if ch == '\\':
            i += 2
        else:
            i += 1
    raise ValueError("no observed value termination")


def punctuation_ids(tokenizer, row):
    ids = row['source_ids']
    pos = row['closing_position']
    tail = tokenizer.decode(ids)[len(tokenizer.decode(ids[:pos])):]
    chosen = []
    for end in range(pos + 1, len(ids) + 1):
        text = tokenizer.decode(ids[:end])[len(tokenizer.decode(ids[:pos])):]
        if not text or any(ch not in '\"}, \t\r\n' for ch in text):
            break
        chosen = ids[pos:end]
    rendered = tokenizer.decode(ids[:pos] + chosen)[len(tokenizer.decode(ids[:pos])):]
    if not rendered.startswith('\"}') or not chosen or not tail.startswith(rendered):
        raise ValueError("no pure closing-punctuation token sequence")
    return chosen


def structural_example(tokenizer, row, generated_ids):
    text = tokenizer.decode(generated_ids)
    boundary, kind = generated_value_end(text)
    positions = [i for i in range(1, len(generated_ids)+1)
                 if tokenizer.decode(generated_ids[:i]) == text[:boundary]]
    if not positions:
        raise ValueError("value/ending boundary straddles a token")
    prefix = generated_ids[:positions[0]]
    suffix = punctuation_ids(tokenizer, row)
    # Check the source punctuation decodes identically in this new context.
    rendered = tokenizer.decode(prefix + suffix)[len(text[:boundary]):]
    source_rendered = tokenizer.decode(row['source_ids'][:row['closing_position']] + suffix)[len(tokenizer.decode(row['source_ids'][:row['closing_position']])):]
    if not tokenizer.decode(prefix + suffix).startswith(text[:boundary]) or rendered != source_rendered:
        raise ValueError("closing punctuation is context-dependent")
    context = tokenizer.encode(row['prompt']) + prefix
    x = context + suffix[:-1]
    y = [-100] * len(x)
    y[len(context)-1:] = suffix
    return (x, y), {'id':row['id'], 'target':row['target'], 'generated_ids':generated_ids,
                    'completion':text, 'value_end':boundary, 'termination':kind,
                    'prefix_ids':prefix, 'suffix_ids':suffix, 'suffix_text':rendered,
                    'value_changed_from_source':prefix != row['source_ids'][:row['closing_position']],
                    'supervised_tokens':len(suffix)}


def refresh_examples(student, tokenizer, torch, rows, step):
    examples, records, rejected = [], [], []
    with trace.observation(student, torch):
        for row in rows:
            gen = base.semantic_gate.generate_completion(student, tokenizer, torch, row['prompt'], 96)
            try:
                example, record = structural_example(tokenizer, row, gen['generated_ids'])
            except ValueError as exc:
                rejected.append({'id':row['id'], 'reason':str(exc), **gen})
                continue
            if len(example[0]) > student.cfg.block_size:
                raise ValueError("structural example exceeds context limit")
            examples.append(example)
            records.append(record)
    if not examples:
        raise ValueError("no usable candidate-generated closing examples")
    audit = {'step':step, 'rows':records, 'rejected':rejected,
             'unclosed':sum(r['termination']=='unclosed' for r in records),
             'changed':sum(r['value_changed_from_source'] for r in records)}
    diag.event('prefix_refresh', step=step, accepted=len(records), rejected=len(rejected),
               unclosed=audit['unclosed'], changed=audit['changed'])
    return examples, audit


def optimizer_step(student, teacher, tokenizer, torch, cfg, examples, tools, copies, batch, optimizer, structural, closing_weight=CLOSURE_WEIGHT):
    student.train()
    optimizer.zero_grad(set_to_none=True)
    place = objectives.batch_loss(student, torch, examples, batch['placement'])
    tool = trace.distill.batch_teacher_kl(student, teacher, tokenizer, torch, tools, batch['tool'], cfg['distill_temperature'])
    copied = trace.distill.batch_teacher_kl(student, teacher, tokenizer, torch, copies, batch['copy'], cfg['distill_temperature'])
    closing = objectives.batch_loss(student, torch, structural, list(range(len(structural))))
    total = cfg['placement_loss_weight']*place + cfg['tool_kl_loss_weight']*tool + cfg['copy_kl_loss_weight']*copied + closing_weight*closing
    if not bool(torch.isfinite(total)):
        raise ValueError('non-finite generated-prefix objective')
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    student.zero_grad(set_to_none=True)
    student.eval()
    return {'placement_loss':float(place.detach()), 'tool_kl_loss':float(tool.detach()),
            'copy_kl_loss':float(copied.detach()), 'closing_loss':float(closing.detach()),
            'combined_loss':float(total.detach()), 'gradient_norm':float(norm)}


def prepare_new_holdout(teacher, tokenizer, torch, previous_attempts):
    used = diag.historical_values() | set(json.loads(PRIOR_VALUES.read_text())["excluded_values"])
    used.update(r["target"] for r in previous_attempts)
    rows, attempts = [], []
    for variant, quota in enumerate((4, 2)):
        kept = 0
        for i in range(48):
            for nonce in range(10000):
                digest = data.copy_data._digest("closure005-fresh-holdout-20260910", variant, i, nonce)
                target = data.copy_data._render("short_code", variant, digest)
                if target not in used:
                    used.add(target)
                    break
            else:
                raise ValueError("new holdout namespace exhausted")
            case = {"id": f"closure005_holdout_{variant}_{i:02d}", **data.tool_prompt("short_code", target)}
            gen = base.semantic_gate.generate_completion(teacher, tokenizer, torch, case["prompt"], 96)
            score = regression.control.score_case(case, gen["completion"])
            accepted = bool(score["envelope_json_valid"] and score["tool_name_correct"])
            attempts.append({"phase":"fresh_holdout", "variant":variant, "target":target, "accepted":accepted})
            if accepted:
                rows.append({**case, "source_score":score, "source_completion":gen["completion"], "source_ids":gen["generated_ids"]})
                kept += 1
            if kept == quota:
                break
        if kept != quota:
            raise ValueError("new source-valid holdout quota failed")
    return rows, attempts


def fresh_probe(student, tokenizer, torch, rows):
    output = []
    for row in rows:
        gen = base.semantic_gate.generate_completion(student, tokenizer, torch, row["prompt"], 96)
        score = regression.control.score_case(row, gen["completion"])
        output.append({"id": row["id"], "target": row["target"], "score": score, **gen})
    passed = sum(r["score"]["envelope_json_valid"] and r["score"]["tool_name_correct"] for r in output)
    return {"rows": output, "cases": len(output), "structurally_correct": passed,
            "all_retained": passed == len(output), "exact_arguments": sum(r["score"]["slot_exact"] for r in output)}


def final_checks(full, fresh, updates):
    checks = dict(full["checks"])
    checks.update(fresh_training_structure_retained=fresh["anchors"]["all_retained"],
                  fresh_holdout_retained=fresh["holdout"]["all_retained"],
                  all_40_nonzero_updates=len(updates)==40 and all(math.isfinite(r["actual_l2"]) and r["actual_l2"]>0 for r in updates))
    if "fresh_holdout" in fresh:
        checks["new_fresh_holdout_retained"] = fresh["fresh_holdout"]["all_retained"]
    return checks


def summary(report):
    lines = ["# Ember generated-prefix JSON closure CPU result", "", f"Execution: {report['status']}",
             f"Steps: {len(report['updates'])}/40", f"Punctuation weight: {report['closing_weight']}", f"Endpoint passed: {report.get('endpoint_passed', False)}", "",
             "| Observed step | Short-code cases retained | Exact placement | Placement tokens | Learning gate |",
             "| ---: | ---: | ---: | ---: | --- |"]
    for r in report["probes"]:
        lines.append(f"| {r['step']} | {r['retention']['retained']}/8 | {r['placement']['exact_top1']}/24 | {r['placement']['token_top1']}/170 | {r['learning']['passed']} |")
    if "checks" in report:
        lines += ["", "Failed endpoint checks: " + ", ".join(k for k,v in report["checks"].items() if not v)]
    lines += ["", "Only steps 1, 12, 13, 23 and 40 were probed; full familiar/copy/reference/KL guards were run on the source and step 40.",
              "Eight fresh training prompts regenerated every five updates; only pure closing punctuation receives structural labels. Six separate fresh holdout prompts never supply gradients.",
              "This is an endpoint experiment, not an exhaustive timing search or promotion evaluation. No checkpoint was exported or integrated."]
    if "error" in report:
        lines.append(f"Error: {report['error']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("generated-closure-results"))
    parser.add_argument("--closing-weight", type=float, choices=(.05, .20), default=.20)
    parser.add_argument("--new-holdout", action="store_true")
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
    report = {"status": "ERROR", "experiment": "ember-generated-closure-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "code_commit": os.environ.get("GITHUB_SHA"), "original_recipe": cfg, "updates": [], "probes": [],
              "gpu_training_authorized": False, "promotion_authorized": False, "production_authorized": False,
              "comparison_run_ids": [34369464055,34387531160], "probe_steps": sorted(PROBE_STEPS),
              "closing_weight": args.closing_weight, "new_holdout": args.new_holdout, "refresh_steps": sorted(REFRESH_STEPS), "refreshes": []}
    student = teacher = pristine = None
    try:
        with tempfile.TemporaryDirectory(prefix="ember-generated-closure-") as td:
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
            if args.new_holdout:
                cohorts["fresh_holdout"], new_attempts = prepare_new_holdout(teacher, tokenizer, torch, attempts)
                attempts.extend(new_attempts)
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
                "tool_report":tool_report,"copy_report":copy_report,"schedule":schedule})
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
            diag.event("baseline", familiar=84, placement_exact=1, placement_tokens=88, closing_weight=args.closing_weight)
            for step,batch in enumerate(schedule,1):
                if step in REFRESH_STEPS:
                    structural, audit = refresh_examples(student, tokenizer, torch, cohorts["anchors"], step)
                    report["refreshes"].append(audit)
                theta = diag.flat_parameters(student,torch)
                losses = optimizer_step(student,teacher,tokenizer,torch,cfg,examples,tools,copies,batch,optimizer,structural,closing_weight=args.closing_weight)
                actual_norm = float((diag.flat_parameters(student,torch)-theta).norm())
                if not math.isfinite(actual_norm) or actual_norm <= 0:
                    raise ValueError("finite nonzero update required")
                del theta
                report["updates"].append({"step":step,"losses":losses,"actual_l2":actual_norm})
                if step==1 and abs(losses["placement_loss"]-1.8641787767410278)>1e-5:
                    raise ValueError("first original training batch failed reproduction")
                if step in PROBE_STEPS:
                    with trace.observation(student,torch):
                        placement = objectives.placement_probe(student,tokenizer,torch,dev,template)
                        short = trace.short_codes(student,tokenizer,torch,cfg)
                        retained = diag.retained_case_ids(before_short,short)
                        fresh = {phase:fresh_probe(student,tokenizer,torch,rows) for phase,rows in cohorts.items()}
                        row = {"step":step,"placement":placement,"short_codes":short,"retention":retained,
                               "learning":trace.placement_learning(place,placement,cfg),"fresh":fresh,
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
            report["checks"] = final_checks(final,fresh,report["updates"])
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
