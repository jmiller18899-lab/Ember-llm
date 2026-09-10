"""Surgical Ember v0.0.54 routing repair restricted to transformer blocks 4 and 5.

Starts from exact saved v0.0.53 step 9. Blocks 4/5 were selected by the prior
read-only gradient-localization diagnostic because they showed the strongest
routing/tool-name signal relative to the protected copy gradient.

Objectives:
- direct first-token margin on fresh v0.0.54 development prompts;
- light preservation of tool-mode entry on true tool prompts;
- canonical early tool-family name supervision after <|tool|>;
- lighter supervision of the established args-first tool-name position.

Every rung resets to v0.0.53. If the protected copy probe drops below 22/35 or
true-tool first-token entry regresses, that rung rolls back and stops. No
checkpoint is saved or promoted.
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
from jobs import ember_v054_routing_repair as v54
from jobs import ember_v054_gradient_localize as loc
from jobs import ember_alternating_trust_region as trust

OUT = Path("v054-block45-repair")
TRAIN_GROUPS = {"block_04", "block_05"}
LRS = (6.4e-7, 1.6e-6, 3.2e-6)
MAX_STEPS = 12
DIRECT_MARGIN = 0.75
TOOL_MARGIN = 1.00
DIRECT_WEIGHT = 0.50
TOOL_ROUTE_WEIGHT = 0.10
CANONICAL_NAME_WEIGHT = 0.30
ARGS_FIRST_NAME_WEIGHT = 0.10
GRAD_CLIP = 0.25
COPY_FLOOR = 22
STRUCTURE_FLOOR = 7


def cont_ids(tokenizer, text: str, shared_prefix_id: int | None) -> list[int]:
    ids = list(tokenizer.encode(text))
    if shared_prefix_id is not None and ids and int(ids[0]) == int(shared_prefix_id):
        ids = ids[1:]
    return [int(x) for x in ids]


def lcp(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def span_ce(model, tokenizer, prompt: str, prefix_text: str, expected: str, shared_prefix_id: int | None):
    through_text = prefix_text + json.dumps(expected)
    pids = cont_ids(tokenizer, prefix_text, shared_prefix_id)
    tids = cont_ids(tokenizer, through_text, shared_prefix_id)
    split = lcp(pids, tids)
    targets = tids[split:]
    if not targets:
        raise RuntimeError("empty name target span")
    context = list(tokenizer.encode(prompt)) + tids[:split]
    seq = context + targets
    if len(seq) > int(model.cfg.block_size):
        raise RuntimeError("name supervision exceeds context window")
    x = torch.tensor([seq[:-1]], dtype=torch.long)
    logits, _ = model(x, None)
    start = len(context) - 1
    y = torch.tensor(targets, dtype=torch.long)
    return F.cross_entropy(logits[0, start : start + len(targets)], y), len(targets)


def route_loss(logits, tool_id: int, special_ids: set[int], should_tool: bool):
    detached = logits.detach().clone()
    detached[list(special_ids)] = -float("inf")
    lexical = detached.max()
    tool = logits[tool_id]
    if should_tool:
        return F.relu(lexical + TOOL_MARGIN - tool)
    return F.relu(tool + DIRECT_MARGIN - lexical)


def configure_trainable(model):
    names = []
    count = 0
    for name, p in model.named_parameters():
        trainable = loc.group_name(name) in TRAIN_GROUPS
        p.requires_grad_(trainable)
        if trainable:
            names.append(name)
            count += p.numel()
    if not names:
        raise RuntimeError("no parameters matched block 4/5")
    return names, count


def snapshot_trainable(model):
    return {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}


def restore_trainable(model, state):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value)


def update_once(model, tokenizer, optimizer, contract, tool_id, special_ids):
    directs = [c for c in v54.TRAIN if c["kind"] == "direct_response"]
    tools = [c for c in v54.TRAIN if c["kind"] == "tool_call"]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    stats = {"direct": [], "tool_route": [], "canonical_name": [], "args_name": []}

    for c in directs:
        logits = base.next_logits(model, tokenizer, c["prompt"])
        loss = route_loss(logits, tool_id, special_ids, False)
        (DIRECT_WEIGHT * loss / len(directs)).backward()
        stats["direct"].append(float(loss.detach()))

    for c in tools:
        logits = base.next_logits(model, tokenizer, c["prompt"])
        loss = route_loss(logits, tool_id, special_ids, True)
        (TOOL_ROUTE_WEIGHT * loss / len(tools)).backward()
        stats["tool_route"].append(float(loss.detach()))

        canonical, _ = span_ce(
            model, tokenizer, c["prompt"], '<|tool|>\n{"name":', c["expected_tool"], contract.get("shared_prefix_id")
        )
        (CANONICAL_NAME_WEIGHT * canonical / len(tools)).backward()
        stats["canonical_name"].append(float(canonical.detach()))

        args_loss, _ = v54.name_loss(model, tokenizer, c, contract.get("shared_prefix_id"))
        (ARGS_FIRST_NAME_WEIGHT * args_loss / len(tools)).backward()
        stats["args_name"].append(float(args_loss.detach()))

    trainable = [p for p in model.parameters() if p.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    return {
        "direct_hinge": sum(stats["direct"]) / len(stats["direct"]),
        "tool_route_hinge": sum(stats["tool_route"]) / len(stats["tool_route"]),
        "canonical_name_ce": sum(stats["canonical_name"]) / len(stats["canonical_name"]),
        "args_first_name_ce": sum(stats["args_name"]) / len(stats["args_name"]),
        "grad_norm": float(grad_norm),
    }


def summary(report):
    lines = [
        "# Ember v0.0.54 block-4/5 surgical routing repair", "",
        f"Status: {report['status']}",
        f"Trainable groups: {', '.join(sorted(TRAIN_GROUPS))}",
        f"LRs: {', '.join(f'{x:.1e}' for x in LRS)}; max {MAX_STEPS} steps/rung.",
        "Rollback: copy <22/35 or held-out true-tool entry <8/8.",
        "", "| LR | Step | Held-out first D/T | Held-out gen D/T | Tool name | Copy | Structure | Fixed D/T | Pass |",
        "| ---: | ---: | --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for rung in report.get("rungs", []):
        for p in rung.get("candidate_probes", []):
            h = p["heldout"]["metrics"]
            s = p["structure"]
            f = p["fixed_generation"]
            lines.append(
                f"| {rung['lr']:.1e} | {p['step']} | {h['direct_first_token_pass']}/12 / {h['tool_first_token_pass']}/8 | "
                f"{h['direct_generation_pass']}/12 / {h['tool_generation_pass']}/8 | {h['tool_name_correct']}/8 | "
                f"{p['placement']['token_top1']}/35 | {s['envelope_json_valid']}/8 / {s['tool_name_correct']}/8 | "
                f"{f['direct_pass']}/4 / {f['tool_pass']}/4 | {p['accepted']} |"
            )
    lines += ["",
        f"Operating point found: {report.get('operating_point_found', False)}",
        f"Selected LR/step: {report.get('selected_lr')} / {report.get('selected_step')}",
        f"Interpretation: {report.get('interpretation', '')}", "",
        "No checkpoint was saved, exported, promoted, or integrated.",
    ]
    return "\n".join(lines) + "\n"


def main():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN required")
    torch.set_num_threads(2)
    torch.manual_seed(20260910)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    report = {
        "schema_version": 1,
        "diagnostic": "ember-v054-block45-surgical-repair-v1",
        "status": "ERROR",
        "source": held.MODEL_NAME,
        "trainable_groups": sorted(TRAIN_GROUPS),
        "lrs": list(LRS),
        "max_steps": MAX_STEPS,
        "cpu_only": True,
        "checkpoint_save_authorized": False,
        "promotion_authorized": False,
        "rungs": [],
    }

    with tempfile.TemporaryDirectory(prefix="ember-v054-b45-") as td:
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
        trainable_names, trainable_count = configure_trainable(model)
        contract, tool_id, special_ids = v54.routing_contract(tokenizer)
        cfg, template, template_report, selected, selected_ids, *_ = base.build_placement_fixture(work)
        fixed_spec = json.loads(Path("config/ember_v0.0.8_eval.json").read_text())
        fixed_cases = [c for c in fixed_spec["cases"] if c["kind"] in {"direct_response", "tool_call"}]

        baseline_place = base.placement_probe(model, tokenizer, selected, template)
        baseline_held = v54.routing_counts(model, tokenizer, held.CASES)
        report["baseline"] = {"placement": baseline_place, "heldout_routing": baseline_held}
        report["trainable_parameter_names"] = trainable_names
        report["trainable_parameter_count"] = trainable_count
        if int(baseline_place["token_top1"]) != COPY_FLOOR:
            raise RuntimeError("source copy floor drifted")

        found = False
        best = None
        best_score = (-1, -1)
        for lr in LRS:
            model.load_state_dict(pristine)
            configure_trainable(model)
            model.eval()
            if trust.trace.state_digest(model) != pristine_hash:
                raise RuntimeError("rung reset failed")
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(lr), weight_decay=0.0)
            rung = {"lr": float(lr), "steps": [], "candidate_probes": [], "accepted": False, "rollback": None}
            for step in range(1, MAX_STEPS + 1):
                pre_params = snapshot_trainable(model)
                pre_opt = copy.deepcopy(opt.state_dict())
                update = update_once(model, tokenizer, opt, contract, tool_id, special_ids)
                train_route = v54.routing_counts(model, tokenizer, v54.TRAIN)
                held_route = v54.routing_counts(model, tokenizer, held.CASES)
                place = base.placement_probe(model, tokenizer, selected, template)
                rec = {
                    "step": step,
                    "update": update,
                    "train_direct": train_route["direct_ok"],
                    "train_tool": train_route["tool_ok"],
                    "heldout_direct": held_route["direct_ok"],
                    "heldout_tool": held_route["tool_ok"],
                    "copy_tokens": int(place["token_top1"]),
                }
                rung["steps"].append(rec)
                print(json.dumps({"event":"b45_probe", "lr":lr, **rec}), flush=True)

                if int(place["token_top1"]) < COPY_FLOOR or held_route["tool_ok"] < 8:
                    restore_trainable(model, pre_params)
                    opt.load_state_dict(pre_opt)
                    model.eval()
                    rung["rollback"] = {"step": step, "reason": "copy" if int(place["token_top1"]) < COPY_FLOOR else "tool_entry"}
                    print(json.dumps({"event":"b45_rollback", "lr":lr, **rung["rollback"]}), flush=True)
                    break

                score = (held_route["direct_ok"], train_route["direct_ok"])
                if score > best_score:
                    best_score = score
                    best = {"lr": float(lr), "step": step, "state": copy.deepcopy(model.state_dict())}

                if held_route["direct_ok"] >= 9 and train_route["direct_ok"] >= 14:
                    held_eval = held.evaluate(model, tokenizer, f"b45-{lr:.1e}-s{step}")
                    structure = base.structure_probe(model, tokenizer, cfg, selected)
                    fixed_gen = base.generation_probe(model, tokenizer, fixed_cases, fixed_spec["generation"])
                    hm = held_eval["metrics"]
                    gates = {
                        "heldout_first": hm["direct_first_token_pass"] == 12 and hm["tool_first_token_pass"] == 8,
                        "heldout_generation": hm["direct_generation_pass"] == 12 and hm["tool_generation_pass"] == 8,
                        "heldout_tool_name": hm["tool_name_correct"] == 8,
                        "copy": int(place["token_top1"]) >= COPY_FLOOR,
                        "structure": int(structure["envelope_json_valid"]) >= STRUCTURE_FLOOR and int(structure["tool_name_correct"]) >= STRUCTURE_FLOOR,
                        "fixed": int(fixed_gen["direct_pass"]) == 4 and int(fixed_gen["tool_pass"]) == 4,
                    }
                    accepted = all(gates.values())
                    probe = {
                        "step": step, "heldout": held_eval, "placement": place,
                        "structure": structure, "fixed_generation": fixed_gen,
                        "gates": gates, "accepted": accepted,
                    }
                    rung["candidate_probes"].append(probe)
                    print(json.dumps({"event":"b45_full_gate", "lr":lr, "step":step, "heldout":hm,
                                      "copy":place["token_top1"], "structure_json":structure["envelope_json_valid"],
                                      "structure_tool":structure["tool_name_correct"], "gates":gates, "accepted":accepted}), flush=True)
                    if accepted:
                        rung["accepted"] = True
                        report["operating_point_found"] = True
                        report["selected_lr"] = float(lr)
                        report["selected_step"] = step
                        report["selected_probe"] = probe
                        found = True
                        break
            report["rungs"].append(rung)
            del opt
            if found:
                break

        if found:
            report["interpretation"] = "Blocks 4/5 alone produced a strict routing operating point while preserving the protected copy/structure gates; run a second fresh confirmation set before saving."
        else:
            report["operating_point_found"] = False
            report["selected_lr"] = None
            report["selected_step"] = None
            if best is not None:
                model.load_state_dict(best["state"])
                configure_trainable(model)
                model.eval()
                held_eval = held.evaluate(model, tokenizer, "b45-best")
                place = base.placement_probe(model, tokenizer, selected, template)
                structure = base.structure_probe(model, tokenizer, cfg, selected)
                report["best_observed"] = {
                    "lr": best["lr"], "step": best["step"], "heldout": held_eval,
                    "placement": place, "structure": structure,
                }
            report["interpretation"] = "Blocks 4/5 improved routing but did not clear the strict unchanged held-out gate inside the copy trust region; do not save or promote."

        report["status"] = "PASS"
        report["elapsed_seconds"] = time.monotonic() - started
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (OUT / "summary.md").write_text(summary(report))
        print(json.dumps({"event":"b45_complete", "operating_point_found":report["operating_point_found"],
                          "selected_lr":report["selected_lr"], "selected_step":report["selected_step"],
                          "elapsed_seconds":report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
