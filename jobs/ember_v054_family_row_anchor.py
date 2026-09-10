"""Four-row tool-family calibration diagnostic for saved Ember v0.0.53.

The model is frozen except vocabulary-facing weight rows for the four first
family-distinguishing tokens used by weather, calculator, web_search, get_time.
If the LM head is tied to the token embedding this is literally four rows of the
tied matrix; if the architecture exposes separate vocabulary-facing matrices,
only those same four rows are trainable in each.

Training uses the 16 development tool requests and a balanced 4-way CE at the
first family token, conditioned on correct canonical arguments. The unchanged
8 tool held-out cases are validation only. Copy is probed every step and a rung
is rolled back immediately below 22/35. Full generation/structure gates run only
on promising teacher-forced family points.

CPU-only diagnostic. No checkpoint save, export, promotion, or integration.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

from huggingface_hub import HfApi
import torch
import torch.nn.functional as F

from jobs import ember_hf_eval as ev
from jobs import ember_v053_heldout_routing as held
from jobs import ember_v053_first_token_anchor as base
from jobs import ember_v054_routing_repair as repair
from jobs import ember_v054_tool_family_diagnostic as family
from jobs import ember_v054_family_token_localize as localize
from jobs import ember_alternating_trust_region as trust

OUT = Path("v054-family-row-anchor")
LRS = (1e-4, 3e-4, 1e-3, 3e-3)
MAX_STEPS = 24
COPY_FLOOR = 22
STRUCTURE_FLOOR = 7
TEACHER_HELDOUT_TRIGGER = 6
GRAD_CLIP = 1.0


def vocabulary_row_parameters(model):
    vocab = int(model.cfg.vocab_size)
    rows = [(name, p) for name, p in model.named_parameters() if p.ndim == 2 and int(p.shape[0]) == vocab]
    if not rows:
        raise RuntimeError("no vocabulary-facing row parameter found")
    return rows


def freeze_to_vocab_rows(model):
    row_params = vocabulary_row_parameters(model)
    ids = {id(p) for _, p in row_params}
    for p in model.parameters():
        p.requires_grad = id(p) in ids
    return row_params


def family_context(tokenizer, case, shared_prefix):
    args_json = json.dumps(case["arguments"], separators=(",", ":"))
    prefix = '<|tool|>\n{"arguments":' + args_json + ',"name":'
    return [int(x) for x in tokenizer.encode(case["prompt"] + prefix)] + shared_prefix


def family_token_ids(tokenizer, contract):
    _ids, shared, next_ids = localize.family_token_contract(tokenizer, contract.get("shared_prefix_id"))
    return shared, next_ids


def family_probe(model, tokenizer, cases, shared, next_ids):
    rows = []
    correct = 0
    with torch.inference_mode():
        for c in cases:
            context = family_context(tokenizer, c, shared)
            logits, _ = model(torch.tensor([context], dtype=torch.long), None)
            next_logits = logits[0, -1]
            ranked = sorted(family.FAMILIES, key=lambda name: float(next_logits[next_ids[name]]), reverse=True)
            expected = c["expected_tool"]
            winner = ranked[0]
            correct += int(winner == expected)
            strongest_wrong = max(float(next_logits[next_ids[name]]) for name in family.FAMILIES if name != expected)
            expected_logit = float(next_logits[next_ids[expected]])
            rows.append({
                "id": c["id"], "expected": expected, "winner": winner,
                "expected_rank": ranked.index(expected) + 1,
                "expected_margin": expected_logit - strongest_wrong,
                "candidate_logits": {name: float(next_logits[next_ids[name]]) for name in family.FAMILIES},
            })
    return {"correct": correct, "total": len(cases), "rows": rows}


def update_once(model, tokenizer, optimizer, train_cases, shared, next_ids, row_params):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    family_to_index = {name: i for i, name in enumerate(family.FAMILIES)}
    losses = []
    margins = []
    for c in train_cases:
        context = family_context(tokenizer, c, shared)
        logits, _ = model(torch.tensor([context], dtype=torch.long), None)
        next_logits = logits[0, -1]
        candidate = torch.stack([next_logits[next_ids[name]] for name in family.FAMILIES])
        target_index = family_to_index[c["expected_tool"]]
        loss = F.cross_entropy(candidate.unsqueeze(0), torch.tensor([target_index], dtype=torch.long))
        (loss / len(train_cases)).backward()
        losses.append(float(loss.detach()))
        target_logit = float(candidate[target_index].detach())
        wrong = max(float(candidate[i].detach()) for i in range(len(family.FAMILIES)) if i != target_index)
        margins.append(target_logit - wrong)

    selected_rows = set(int(x) for x in next_ids.values())
    pre_mask_sq = 0.0
    post_mask_sq = 0.0
    for _, p in row_params:
        if p.grad is None:
            continue
        pre_mask_sq += float((p.grad.detach().float() ** 2).sum())
        mask = torch.zeros_like(p.grad)
        for rid in selected_rows:
            mask[rid].copy_(p.grad[rid])
        p.grad.copy_(mask)
        post_mask_sq += float((p.grad.detach().float() ** 2).sum())
    grad_norm = torch.nn.utils.clip_grad_norm_([p for _, p in row_params], GRAD_CLIP, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    return {
        "ce_mean": sum(losses) / len(losses),
        "mean_target_margin": sum(margins) / len(margins),
        "pre_mask_grad_norm": pre_mask_sq ** 0.5,
        "selected_row_grad_norm": post_mask_sq ** 0.5,
        "clipped_grad_norm": float(grad_norm),
    }


def generation_gates(model, tokenizer, heldout_cases, fixed_cases, fixed_generation, cfg, selected, template):
    held_eval = held.evaluate(model, tokenizer, "family-row-candidate")
    fixed_route = base.routing_probe(model, tokenizer, fixed_cases)
    fixed_gen = base.generation_probe(model, tokenizer, fixed_cases, fixed_generation)
    structure = base.structure_probe(model, tokenizer, cfg, selected)
    placement = base.placement_probe(model, tokenizer, selected, template)
    hm = held_eval["metrics"]
    gates = {
        "heldout_tool_entry": hm["tool_first_token_pass"] == hm["tool_total"],
        "heldout_tool_family": hm["tool_name_correct"] >= 6,
        "heldout_direct_nonregression": hm["direct_first_token_pass"] >= 3 and hm["direct_generation_pass"] >= 3,
        "fixed_route": fixed_route["direct_argmax_tool_count"] == 0 and fixed_route["tool_argmax_tool_count"] == 4,
        "fixed_generation": fixed_gen["direct_pass"] == 4 and fixed_gen["tool_pass"] == 4,
        "copy_preserved": int(placement["token_top1"]) >= COPY_FLOOR,
        "structure_preserved": int(structure["envelope_json_valid"]) >= STRUCTURE_FLOOR and int(structure["tool_name_correct"]) >= STRUCTURE_FLOOR,
    }
    return {
        "heldout": held_eval,
        "fixed_route": fixed_route,
        "fixed_generation": fixed_gen,
        "structure": structure,
        "placement": placement,
        "gates": gates,
        "accepted": all(gates.values()),
    }


def summary_markdown(report):
    lines = [
        "# Ember v0.0.54 four-row tool-family calibration", "",
        f"Status: {report['status']}",
        "Source: exact saved v0.0.53 step-9 candidate.",
        f"Vocabulary-facing parameters: {', '.join(report.get('vocab_row_parameters', []))}",
        f"Only four family-distinguishing rows are allowed to change: {report.get('family_token_ids', {})}",
        f"LR ladder: {', '.join(f'{x:.1e}' for x in LRS)}; max {MAX_STEPS} steps/rung.",
        "", "| LR | Step | Dev family | Held-out family | Copy | Candidate full gate |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rung in report.get("rungs", []):
        for p in rung.get("steps", []):
            lines.append(
                f"| {rung['lr']:.1e} | {p['step']} | {p['development_family']['correct']}/16 | "
                f"{p['heldout_family']['correct']}/8 | {p['copy_tokens']}/35 | {p.get('candidate_accepted', '—')} |"
            )
    lines += ["", f"Operating point found: {report.get('operating_point_found', False)}",
              f"Selected LR: {report.get('selected_lr')}", f"Selected step: {report.get('selected_step')}",
              "", "No checkpoint was saved, exported, promoted, or integrated."]
    return "\n".join(lines) + "\n"


def main():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    torch.set_num_threads(2)
    torch.manual_seed(20260910)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="ember-v054-family-row-") as td:
        work = Path(td)
        archive = ev.download_verified(ev.PACKAGE_URL, work / "ember.zip", ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        import sys
        sys.path.insert(0, str(work / "src" / "ember"))

        api = HfApi(token=token)
        owner = api.whoami()["name"]
        repo = f"{owner}/{held.MODEL_NAME}"
        model, tokenizer, checkpoint = held.load_full(repo, work, token)
        model.eval()
        pristine = copy.deepcopy(model.state_dict())
        pristine_hash = trust.trace.state_digest(model)

        contract, _, _ = repair.routing_contract(tokenizer)
        shared, next_ids = family_token_ids(tokenizer, contract)
        row_params = freeze_to_vocab_rows(model)
        row_param_names = [name for name, _ in row_params]
        selected_rows = sorted(set(next_ids.values()))
        if len(selected_rows) != 4:
            raise RuntimeError(f"expected exactly four distinct family rows, got {selected_rows}")

        development, heldout_tools = family.make_cases()
        cfg, template, template_report, selected, selected_ids, *_ = base.build_placement_fixture(work)
        fixed_spec = json.loads(Path("config/ember_v0.0.8_eval.json").read_text())
        fixed_cases = [c for c in fixed_spec["cases"] if c["kind"] in {"direct_response", "tool_call"}]
        fixed_generation = fixed_spec["generation"]

        baseline_dev = family_probe(model, tokenizer, development, shared, next_ids)
        baseline_held = family_probe(model, tokenizer, heldout_tools, shared, next_ids)
        baseline_place = base.placement_probe(model, tokenizer, selected, template)
        if int(baseline_place["token_top1"]) != 22:
            raise RuntimeError(f"v0.0.53 copy baseline drifted: {baseline_place['token_top1']}/35")

        report = {
            "schema_version": 1,
            "diagnostic": "ember-v054-four-row-family-calibration-v1",
            "status": "RUNNING",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_repo": repo,
            "checkpoint": held.BEST_PATH,
            "checkpoint_sha256": held.BEST_SHA256,
            "source_state_sha256": pristine_hash,
            "vocab_row_parameters": row_param_names,
            "family_token_ids": next_ids,
            "family_token_text": {name: tokenizer.decode([tid]) for name, tid in next_ids.items()},
            "baseline": {"development_family": baseline_dev, "heldout_family": baseline_held, "placement": baseline_place},
            "rungs": [],
            "operating_point_found": False,
            "selected_lr": None,
            "selected_step": None,
            "cpu_only": True,
            "checkpoint_save_authorized": False,
            "promotion_authorized": False,
        }

        found = False
        for lr in LRS:
            model.load_state_dict(pristine)
            model.eval()
            row_params = freeze_to_vocab_rows(model)
            if trust.trace.state_digest(model) != pristine_hash:
                raise RuntimeError("rung reset failed")
            optimizer = torch.optim.AdamW([p for _, p in row_params], lr=float(lr), weight_decay=0.0)
            rung = {"lr": float(lr), "steps": [], "rolled_back": False}
            for step in range(1, MAX_STEPS + 1):
                update = update_once(model, tokenizer, optimizer, development, shared, next_ids, row_params)
                dev_probe = family_probe(model, tokenizer, development, shared, next_ids)
                held_probe = family_probe(model, tokenizer, heldout_tools, shared, next_ids)
                place = base.placement_probe(model, tokenizer, selected, template)
                copy_tokens = int(place["token_top1"])
                record = {
                    "step": step, "update": update,
                    "development_family": dev_probe,
                    "heldout_family": held_probe,
                    "copy_tokens": copy_tokens,
                }
                print(json.dumps({
                    "event": "family_row_probe", "lr": lr, "step": step,
                    "development_family": dev_probe["correct"],
                    "heldout_family": held_probe["correct"],
                    "copy_tokens": copy_tokens,
                    "update": update,
                }), flush=True)

                if copy_tokens < COPY_FLOOR:
                    record["rollback_reason"] = "copy"
                    rung["steps"].append(record)
                    rung["rolled_back"] = True
                    model.load_state_dict(pristine)
                    print(json.dumps({"event":"family_row_rollback","lr":lr,"step":step,"reason":"copy"}), flush=True)
                    break

                if held_probe["correct"] >= TEACHER_HELDOUT_TRIGGER:
                    candidate = generation_gates(model, tokenizer, heldout_tools, fixed_cases, fixed_generation, cfg, selected, template)
                    record["candidate"] = candidate
                    record["candidate_accepted"] = bool(candidate["accepted"] and held_probe["correct"] == 8 and dev_probe["correct"] == 16)
                    print(json.dumps({
                        "event":"family_row_full_gate", "lr":lr, "step":step,
                        "dev_family":dev_probe["correct"], "heldout_teacher_family":held_probe["correct"],
                        "heldout_free_tool_family":candidate["heldout"]["metrics"]["tool_name_correct"],
                        "heldout_direct":candidate["heldout"]["metrics"]["direct_generation_pass"],
                        "copy_tokens":candidate["placement"]["token_top1"],
                        "json":candidate["structure"]["envelope_json_valid"],
                        "tool_structure":candidate["structure"]["tool_name_correct"],
                        "gates":candidate["gates"], "accepted":record["candidate_accepted"],
                    }), flush=True)
                    if record["candidate_accepted"]:
                        report["operating_point_found"] = True
                        report["selected_lr"] = float(lr)
                        report["selected_step"] = step
                        report["selected_probe"] = record
                        found = True
                rung["steps"].append(record)
                if found:
                    break
            report["rungs"].append(rung)
            del optimizer
            if found:
                break

        report["status"] = "PASS"
        report["elapsed_seconds"] = time.monotonic() - started
        if not found:
            # Preserve the best teacher-forced point for diagnosis, not as a checkpoint.
            all_steps = [(r["lr"], s) for r in report["rungs"] for s in r["steps"] if s.get("copy_tokens", 0) >= COPY_FLOOR]
            if all_steps:
                lr, best = max(all_steps, key=lambda item: (item[1]["heldout_family"]["correct"], item[1]["development_family"]["correct"], item[1]["step"]))
                report["best_diagnostic_lr"] = lr
                report["best_diagnostic_step"] = best["step"]
                report["best_diagnostic_heldout_family"] = best["heldout_family"]["correct"]
                report["best_diagnostic_development_family"] = best["development_family"]["correct"]
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (OUT / "summary.md").write_text(summary_markdown(report))
        print(json.dumps({
            "event":"family_row_complete",
            "operating_point_found":report["operating_point_found"],
            "selected_lr":report["selected_lr"], "selected_step":report["selected_step"],
            "best_diagnostic_heldout_family":report.get("best_diagnostic_heldout_family"),
            "best_diagnostic_development_family":report.get("best_diagnostic_development_family"),
            "elapsed_seconds":report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
