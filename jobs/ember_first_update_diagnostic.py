"""One CPU Adam proposal, evaluated with and without a synthetic margin projection.

This is a diagnostic, not a training launcher or candidate promotion workflow.
The frozen familiar/copy/reference batteries never supply optimization gradients.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import copy
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import ember_v051_data as data
from jobs import ember_v050_distill as distill
from jobs import ember_v048_objectives as objectives
from jobs import ember_v048_regression as regression
from jobs.ember_v051_canary import state_digest

base = data.base
CONFIG = ROOT / "config/ember_first_update_diagnostic.json"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def event(name, **fields):
    print(json.dumps({"event": name, **fields}, allow_nan=False), flush=True)


def flat_parameters(model, torch):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def assign_vector(model, vector, torch):
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(vector[offset:offset + p.numel()].view_as(p))
            offset += p.numel()
    if offset != vector.numel():
        raise ValueError("parameter-vector shape mismatch")


def gradient_vector(loss, model, torch):
    parameters = list(model.parameters())
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for p, g in zip(parameters, gradients)])


def add_basis(vector, basis, torch):
    """Two-pass modified Gram-Schmidt; reject numerically dependent directions."""
    initial_norm = float(torch.linalg.vector_norm(vector))
    if initial_norm <= 1e-12:
        return False
    vector = vector / initial_norm
    for _ in range(2):
        for q in basis:
            vector.add_(q, alpha=-float(torch.dot(q, vector)))
    norm = float(torch.linalg.vector_norm(vector))
    if norm <= 1e-6:
        return False
    basis.append(vector / norm)
    return True


def project_update(delta, basis, torch):
    projected = delta.clone()
    for _ in range(2):
        for q in basis:
            projected.add_(q, alpha=-float(torch.dot(q, projected)))
    return projected


def retained_case_ids(before, after):
    old = {r["id"] for r in before["rows"]
           if r["score"]["envelope_json_valid"] and r["score"]["tool_name_correct"]}
    new = {r["id"] for r in after["rows"]
           if r["score"]["envelope_json_valid"] and r["score"]["tool_name_correct"]}
    return {"passed": old <= new, "source_passed": len(old), "retained": len(old & new),
            "lost_ids": sorted(old - new), "gained_ids": sorted(new - old)}


def historical_values():
    used = data.historical_used_values()
    cfg = data.load_config()
    used.update(v for phase in data.target_values(cfg).values() for group in phase.values() for v in group)
    for family in ("tool", "copy"):
        used.update(r["target"] for r in data.distill_value_rows(cfg, family))
    return used


def fresh_value(kind, variant, tag, index, used):
    for nonce in range(10000):
        digest = data.copy_data._digest("first-update-20260909-v1", tag, kind, variant, index, nonce)
        value = data.copy_data._render(kind, variant, digest)
        if value not in used:
            used.add(value)
            return value
    raise ValueError(f"fresh value space exhausted for {kind}/{variant}")


def new_target_values(used):
    out = {}
    for phase, count in (("template", 4), ("train", 6), ("development", 6)):
        out[phase] = {}
        for subtype in sorted(data.TARGET_SUBTYPES):
            kind, variant = data.v048d.SUBTYPE_VARIANT[subtype]
            group = [fresh_value(kind, variant, phase, i, used) for i in range(count)]
            if any(data.v044.subtype_for(kind, v) != subtype for v in group):
                raise ValueError("fresh target subtype changed")
            out[phase][subtype] = group
    return out


def preservation_rows(model, tokenizer, torch, cfg, used):
    tools, copies, attempts = [], [], []
    quotas = [("short_code", 0, 8), ("short_code", 1, 2)]
    quotas.extend((kind, 0, 2) for kind in data.copy_data.KINDS if kind != "short_code")
    for kind, variant, quota in quotas:
        kept = 0
        for i in range(cfg["max_attempts_per_quota"]):
            value = fresh_value(kind, variant, "tool-preservation", i, used)
            case = {"id": f"fresh_tool_{kind}_{variant}_{i:02d}", **data.tool_prompt(kind, value)}
            gen = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], 96)
            score = data.v048d.control.score_case(case, gen["completion"])
            accepted = bool(score["envelope_json_valid"] and score["tool_name_correct"] and gen["generated_ids"])
            attempts.append({"family": "tool", "kind": kind, "variant": variant, "target": value, "accepted": accepted})
            if accepted:
                tools.append({**case, "variant": variant, "source_ids": gen["generated_ids"],
                              "source_completion": gen["completion"], "source_score": score})
                kept += 1
            if kept == quota:
                break
        event("preservation_quota", kind=kind, variant=variant, kept=kept, required=quota)
        if kept != quota:
            raise ValueError(f"teacher-correct envelope/tool quota unavailable: {kind}/{variant}: {kept}/{quota}")
    for kind in data.copy_data.KINDS:
        for i in range(cfg["max_attempts_per_quota"]):
            value = fresh_value(kind, 0, "copy-preservation", i, used)
            prompt = data.copy_data.prompt(value, data.copy_data._corrupt(value, 3), data.copy_data._corrupt(value, 11))
            gen = base.semantic_gate.generate_completion(model, tokenizer, torch, prompt, 96)
            accepted = bool(gen["generated_ids"] and not gen["completion"].lstrip().startswith(data.v048d.v045.TOOL))
            attempts.append({"family": "copy", "kind": kind, "variant": 0, "target": value, "accepted": accepted})
            if accepted:
                copies.append({"id": f"fresh_copy_{kind}_{i:02d}", "kind": kind, "target": value,
                               "prompt": prompt, "source_ids": gen["generated_ids"], "source_completion": gen["completion"]})
                break
        else:
            raise ValueError(f"no direct-response preservation row for {kind}")
    return tools, copies, attempts


def margin_basis(model, tokenizer, torch, rows):
    basis, anchors = [], []
    for row in rows:
        prompt = tokenizer.encode(row["prompt"])
        generated = row["source_ids"]
        x = torch.tensor([prompt + generated[:-1]], dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(x)
            response = logits[0, len(prompt) - 1:]
            targets = torch.tensor(generated, dtype=torch.long)
            others = response.clone()
            others[torch.arange(len(targets)), targets] = -torch.inf
            best_other, runner = others.max(dim=-1)
            margins = response[torch.arange(len(targets)), targets] - best_other
            position = int(margins.argmin())
            if float(margins[position]) < -1e-5:
                raise ValueError("source teacher-forced winners do not reproduce generated tokens")
            alternative = int(runner[position])
        prefix = prompt + generated[:position]
        logits, _ = model(torch.tensor([prefix], dtype=torch.long))
        margin = logits[0, -1, generated[position]] - logits[0, -1, alternative]
        grad = gradient_vector(margin, model, torch)
        norm = float(torch.linalg.vector_norm(grad))
        included = add_basis(grad, basis, torch)
        anchors.append({"id": row["id"], "target": row["target"], "response_position": position,
                        "source_token_id": generated[position], "alternative_token_id": alternative,
                        "source_margin": float(margin.detach()), "gradient_norm": norm,
                        "independent_direction": included, "prefix_ids": prefix})
        event("margin_direction", id=row["id"], rank=len(basis), source_margin=float(margin.detach()))
    if not basis:
        raise ValueError("no nonzero synthetic preservation directions")
    return basis, anchors


def measure_anchors(model, torch, anchors):
    rows = []
    with torch.no_grad():
        for a in anchors:
            logits, _ = model(torch.tensor([a["prefix_ids"]], dtype=torch.long))
            v = logits[0, -1]
            margin = float(v[a["source_token_id"]] - v[a["alternative_token_id"]])
            rows.append({"id": a["id"], "margin": margin, "margin_change": margin-a["source_margin"],
                         "source_token_still_top1": int(v.argmax()) == a["source_token_id"]})
    return rows


def familiar(model, tokenizer, torch, cfg):
    rows, by_kind, by_subtype = [], Counter(), Counter()
    for case in regression.v045.frozen_v044_cases():
        gen = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], cfg["generation_budget"])
        score = regression.control.score_case(case, gen["completion"])
        rows.append({"id": case["id"], "kind": case["kind"], "subtype": case["subtype"],
                     "prompt": case["prompt"], "score": score, **gen})
        if score["envelope_json_valid"] and score["tool_name_correct"]:
            by_kind[case["kind"]] += 1
            by_subtype[case["subtype"]] += 1
    return {"cases": len(rows), "envelope_json_valid": sum(r["score"]["envelope_json_valid"] for r in rows),
            "correct_tool": sum(r["score"]["tool_name_correct"] for r in rows),
            "by_kind": dict(by_kind), "by_subtype": dict(by_subtype), "rows": rows}


def evaluate(model, teacher, tokenizer, torch, cfg, entry, placement, template, tools, copies, label):
    model.eval()
    result = {}
    result["entry"] = objectives.entry_probe(model, tokenizer, torch, entry)
    result["placement"] = objectives.placement_probe(model, tokenizer, torch, placement, template)
    event("evaluate", arm=label, stage="familiar_90")
    result["familiar"] = familiar(model, tokenizer, torch, cfg)
    result["reference"] = regression.references(model, tokenizer, torch, cfg)
    event("evaluate", arm=label, stage="historical_copy")
    result["copy"] = base.copy_diagnostic(model, tokenizer, torch)
    result["tool_distill"] = distill.distill_probe(model, teacher, tokenizer, torch, tools)
    result["copy_distill"] = distill.distill_probe(model, teacher, tokenizer, torch, copies)
    event("evaluation_complete", arm=label, familiar=result["familiar"]["correct_tool"],
          placement_exact=result["placement"]["exact_top1"], placement_tokens=result["placement"]["token_top1"])
    return result


def divergence_rows(source_model, model, tokenizer, torch, before, after, lost_ids):
    previous = {r["id"]: r for r in before["rows"]}
    out = []
    for row in after["rows"]:
        old = previous[row["id"]]
        a, b = old["generated_ids"], row["generated_ids"]
        position = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
        if position is None:
            if len(a) != len(b):
                out.append({"id": row["id"], "length_changed": True, "lost_case": row["id"] in lost_ids})
            continue
        details = {"id": row["id"], "first_changed_position": position, "source_token_id": a[position],
                   "candidate_token_id": b[position], "source_token": tokenizer.decode([a[position]]),
                   "candidate_token": tokenizer.decode([b[position]]), "lost_case": row["id"] in lost_ids}
        if details["lost_case"]:
            prefix = tokenizer.encode(row["prompt"]) + a[:position]
            with torch.no_grad():
                for name, m in (("source", source_model), ("candidate", model)):
                    logits, _ = m(torch.tensor([prefix], dtype=torch.long))
                    details[name + "_margin_at_divergence"] = float(logits[0,-1,a[position]] - logits[0,-1,b[position]])
        out.append(details)
    return out


def update_statistics(model, theta, actual, raw, basis, torch):
    norm = float(torch.linalg.vector_norm(actual))
    raw_norm = float(torch.linalg.vector_norm(raw))
    qdots = [float(torch.dot(q, actual)) for q in basis]
    per_tensor, offset = [], 0
    for name, p in model.named_parameters():
        v = actual[offset:offset+p.numel()]
        per_tensor.append({"name": name, "elements": p.numel(), "changed_elements": int((v != 0).sum()),
                           "delta_l2": float(torch.linalg.vector_norm(v)), "delta_max_abs": float(v.abs().max())})
        offset += p.numel()
    return {"delta_l2": norm, "source_relative_l2": norm/float(torch.linalg.vector_norm(theta)),
            "cosine_to_raw_proposal": float(torch.dot(actual, raw))/(norm*raw_norm) if norm*raw_norm else 0.0,
            "basis_dot_actual_update": qdots,
            "basis_residual_fraction": sum(x*x for x in qdots)**0.5/norm if norm else 0.0,
            "per_tensor": per_tensor}


def timeout_handler(_signum, _frame):
    raise TimeoutError("bounded first-update diagnostic reached its wall-time limit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("first-update-results"))
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text())
    if cfg["optimizer_steps"] != 1 or cfg["arms"] != ["unrestricted", "projected"]:
        raise ValueError("diagnostic must remain one proposal and two fixed arms")
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(cfg["seed"])
    torch.use_deterministic_algorithms(True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(cfg["wall_time_limit_seconds"])
    report = {"schema_version": 1, "diagnostic": "ember-first-update-v1", "status": "ERROR",
              "created_at": datetime.now(timezone.utc).isoformat(), "code_commit": os.environ.get("GITHUB_SHA"),
              "config": cfg, "optimizer_steps_executed": 0, "arms": {},
              "gpu_training_authorized": False, "promotion_authorized": False}
    try:
        old_cfg = data.load_config()
        objective_cfg = {**old_cfg, "learning_rate": cfg["learning_rate"]}
        report["objective_config"] = objective_cfg
        with tempfile.TemporaryDirectory(prefix="ember-first-update-") as td:
            student, tokenizer, source, _splits, source_ref = base.load_inputs(json.loads(base.DEFAULT_CONFIG.read_text()), Path(td), torch)
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != "0.0.31":
                raise ValueError("not the pristine v0.0.31 step-479 source")
            report["source"] = source_ref
            report["source_model_config"] = source["model_config"]
            if float(student.cfg.dropout) != 0:
                raise ValueError("training/evaluation dropout parity must hold for this diagnostic")
            del source, _splits
            student.to("cpu").eval()
            teacher = copy.deepcopy(student).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            theta = flat_parameters(student, torch)
            source_hash = state_digest(student)
            report["source_state_sha256"] = source_hash
            report["unique_parameters"] = theta.numel()
            used = historical_values()
            protected = set(used)
            values = new_target_values(used)
            template, template_report = data.v048d.discover_template(student, tokenizer, torch, old_cfg, values["template"])
            tools, copies, attempts = preservation_rows(student, tokenizer, torch, cfg, used)
            fresh = used - protected
            if any(r["target"] in protected for r in tools + copies):
                raise ValueError("preservation target overlaps historical values")
            report["data"] = {"target_values": values, "template": template_report,
                              "tool_preservation": tools, "copy_preservation": copies,
                              "attempts": attempts, "fresh_value_count": len(fresh),
                              "historical_value_count": len(protected), "historical_overlap": 0,
                              "unpublished_ladder_values_available": False,
                              "teacher_correct_definition": "Tool rows preserve valid canonical JSON and correct tool name; exact argument copying is reported separately. Copy rows retain the published direct-response filter."}
            write_json(output / "data.json", report["data"])
            def cases(phase, subtypes, name):
                return data.v048d.build_cases({k: values[phase][k] for k in sorted(subtypes)}, name)
            entry_train = cases("train", data.ENTRY_SUBTYPES, "entry_train")
            place_train = cases("train", data.PLACEMENT_SUBTYPES, "place_train")
            entry_dev = cases("development", data.ENTRY_SUBTYPES, "entry_dev")
            place_dev = cases("development", data.PLACEMENT_SUBTYPES, "place_dev")
            short_rows = [r for r in tools if r["kind"] == "short_code" and r["variant"] == 0]
            if len(short_rows) != 8 or any(len(r["target"]) != 4 for r in short_rows):
                raise ValueError("eight fresh four-character anchors required")
            basis, anchors = margin_basis(student, tokenizer, torch, short_rows)
            report["projection"] = {"rank": len(basis), "anchors": anchors,
                                    "definition": "Euclidean projection of the actual AdamW proposal onto the orthogonal complement of eight fresh short-code source-margin gradients."}
            tool_id, _ = objectives.token_contract(tokenizer)
            entry_examples = [objectives.supervised_example(tokenizer, c, "entry", template, tool_id) for c in entry_train]
            place_examples = [objectives.supervised_example(tokenizer, c, "placement", template, tool_id) for c in place_train]
            entry_idx = [0, 6, 12, 18]
            place_idx = [0, 1, 6, 7, 12, 13, 18, 19]
            tool_idx = [tools.index(r) for r in short_rows[:4]] + [i for i,r in enumerate(tools) if r["kind"] != "short_code"][::2][:4]
            copy_idx = list(range(8))
            report["batches"] = {"entry_ids": [entry_train[i]["id"] for i in entry_idx],
                                 "placement_ids": [place_train[i]["id"] for i in place_idx],
                                 "tool_ids": [tools[i]["id"] for i in tool_idx], "copy_ids": [copies[i]["id"] for i in copy_idx]}
            total_grad = torch.zeros_like(theta)
            preservation_grad = torch.zeros_like(theta)
            components = {}
            component_fns = [
                ("entry", old_cfg["entry_loss_weight"], lambda: objectives.batch_loss(student, torch, entry_examples, entry_idx)),
                ("placement", old_cfg["placement_loss_weight"], lambda: objectives.batch_loss(student, torch, place_examples, place_idx)),
                ("tool_kl", old_cfg["tool_kl_loss_weight"], lambda: distill.batch_teacher_kl(student, teacher, tokenizer, torch, tools, tool_idx, 1.0)),
                ("copy_kl", old_cfg["copy_kl_loss_weight"], lambda: distill.batch_teacher_kl(student, teacher, tokenizer, torch, copies, copy_idx, 1.0)),
            ]
            for name, weight, fn in component_fns:
                loss = fn()
                grad = gradient_vector(weight * loss, student, torch)
                components[name] = {"loss": float(loss.detach()), "weight": weight,
                                    "weighted_gradient_l2": float(torch.linalg.vector_norm(grad))}
                total_grad.add_(grad)
                if name.endswith("kl"):
                    preservation_grad.add_(grad)
                del grad, loss
            report["step0_gradients"] = {"components": components,
                                         "preservation_sum_l2": float(torch.linalg.vector_norm(preservation_grad)),
                                         "combined_l2": float(torch.linalg.vector_norm(total_grad))}
            del preservation_grad
            optimizer = torch.optim.AdamW(student.parameters(), lr=cfg["learning_rate"], weight_decay=old_cfg["weight_decay"])
            if optimizer.state or state_digest(student) != source_hash:
                raise ValueError("first proposal must start from pristine model and empty optimizer")
            report["optimizer_initial_state_entries"] = len(optimizer.state)
            offset = 0
            for p in student.parameters():
                p.grad = total_grad[offset:offset+p.numel()].reshape_as(p).clone()
                offset += p.numel()
            torch.nn.utils.clip_grad_norm_(student.parameters(), old_cfg["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            report["optimizer_steps_executed"] += 1
            raw = flat_parameters(student, torch) - theta
            del optimizer, total_grad
            student.zero_grad(set_to_none=True)
            projected = project_update(raw, basis, torch)
            raw_norm = float(torch.linalg.vector_norm(raw))
            projected_norm = float(torch.linalg.vector_norm(projected))
            residual = max(abs(float(torch.dot(q, projected))) for q in basis) / max(projected_norm, 1e-30)
            if projected_norm > raw_norm * (1 + 1e-5) or residual > 1e-5:
                raise ValueError("projection failed its non-expansion/orthogonality checks")
            report["projection"].update(raw_proposal_l2=raw_norm, projected_proposal_l2=projected_norm,
                                        mathematical_projection_residual=residual,
                                        removed_l2=float(torch.linalg.vector_norm(raw-projected)))
            event("proposal_fixed", raw_l2=raw_norm, projected_l2=projected_norm, rank=len(basis))
            assign_vector(student, theta, torch)
            if state_digest(student) != source_hash:
                raise ValueError("pristine restore failed before evaluation")
            # All proposal choices and projection directions are fixed before any frozen-battery evaluation.
            before = evaluate(student, teacher, tokenizer, torch, old_cfg, entry_dev, place_dev, template, tools, copies, "source")
            report["before"] = before
            if before["familiar"]["envelope_json_valid"] != 84 or before["familiar"]["correct_tool"] != 84 or not before["reference"]["passed"]:
                raise ValueError("source failed the 84/90 and 4/4 reproduction gates")
            if before["tool_distill"]["teacher_kl"] > 1e-7 or before["copy_distill"]["teacher_kl"] > 1e-7:
                raise ValueError("source and frozen teacher do not reproduce initial KL equality")
            write_json(output / "report.json", report)
            for name, delta in (("unrestricted", raw), ("projected", projected)):
                assign_vector(student, theta, torch)
                restored_hash = state_digest(student)
                if restored_hash != source_hash:
                    raise ValueError("arm pristine restore failed")
                assign_vector(student, theta + delta, torch)
                actual = flat_parameters(student, torch) - theta
                stats = update_statistics(student, theta, actual, raw, basis, torch)
                after = evaluate(student, teacher, tokenizer, torch, old_cfg, entry_dev, place_dev, template, tools, copies, name)
                retention = retained_case_ids(before["familiar"], after["familiar"])
                floors = regression.floor_checks(after["familiar"], {**old_cfg, "gate": old_cfg["final_gate"]})
                copy_guard = base.copy_protection(before["copy"], after["copy"])
                learning = distill.selection_checks(before["entry"], after["entry"], before["placement"], after["placement"], after["tool_distill"], after["copy_distill"], old_cfg)
                checks = {"all_source_familiar_cases_retained": retention["passed"], "familiar_floors": floors["passed"],
                          "reference": after["reference"]["passed"], "copy": copy_guard["passed"],
                          "existing_synthetic_learning_gate": learning["passed"], "nonzero_update": bool((actual != 0).any())}
                if name == "projected":
                    checks["actual_projection_residual"] = stats["basis_residual_fraction"] <= cfg["maximum_actual_projection_residual_fraction"]
                report["arms"][name] = {"restored_source_sha256": restored_hash, "candidate_sha256": state_digest(student),
                                       "actual_update": stats, "after": after, "retention": retention,
                                       "familiar_guard": floors, "copy_guard": copy_guard, "learning_gate": learning,
                                       "anchors": measure_anchors(student, torch, anchors),
                                       "familiar_divergences": divergence_rows(teacher, student, tokenizer, torch, before["familiar"], after["familiar"], retention["lost_ids"]),
                                       "checks": checks, "operating_point_found": all(checks.values())}
                write_json(output / "report.json", report)
                event("arm_complete", arm=name, retained=retention["retained"], lost=retention["lost_ids"], checks=checks)
                del actual
            assign_vector(student, theta, torch)
            report["final_pristine_restore_verified"] = state_digest(student) == source_hash
            report["teacher_unchanged"] = state_digest(teacher) == source_hash
            if not report["final_pristine_restore_verified"] or not report["teacher_unchanged"]:
                raise ValueError("final source/teacher integrity check failed")
            report["status"] = "COMPLETE"
            report["operating_point_found"] = any(a["operating_point_found"] for a in report["arms"].values())
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - start
        write_json(output / "report.json", report)
        lines = ["# Ember bounded first-update CPU diagnostic", "", f"Execution: {report['status']}",
                 f"Optimizer proposals executed: {report['optimizer_steps_executed']}", "",
                 "| Arm | Familiar JSON | Correct tool | Source cases retained | Placement exact | Placement token top-1 | Copy guard | Operating point |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
        if "before" in report:
            b = report["before"]
            lines.append(f"| Source | {b['familiar']['envelope_json_valid']}/90 | {b['familiar']['correct_tool']}/90 | 84/84 | {b['placement']['exact_top1']}/{b['placement']['cases']} | {b['placement']['token_top1_rate']:.4%} | baseline | — |")
        for name,a in report["arms"].items():
            b = a["after"]
            lines.append(f"| {name} | {b['familiar']['envelope_json_valid']}/90 | {b['familiar']['correct_tool']}/90 | {a['retention']['retained']}/84 | {b['placement']['exact_top1']}/{b['placement']['cases']} | {b['placement']['token_top1_rate']:.4%} | {'PASS' if a['copy_guard']['passed'] else 'FAIL'} | {a['operating_point_found']} |")
        lines += ["", "Exactly one AdamW proposal; the two arms differ only by projection after the optimizer. Frozen evaluation cases never supply gradients or tune the projection.",
                  "The committed v0.0.51 objective weights are used (0.05 entry, 0.20 placement, 0.50 tool KL, 0.25 copy KL), with learning rate 1e-7 and entirely fresh values relative to available published corpora.",
                  "The later rung-0 ladder code/data were unavailable. This is a separate diagnostic, not an exact reconstruction of that ladder. No checkpoint is saved, promoted, or deployed."]
        if "error" in report:
            lines += ["", f"Error: {report['error']}"]
        (output / "summary.md").write_text("\n".join(lines) + "\n")
    event("complete", status=report["status"], operating_point_found=report["operating_point_found"], elapsed_seconds=report["elapsed_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
