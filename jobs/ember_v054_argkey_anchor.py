"""Bounded 4-way argument-key margin anchor for saved Ember v0.0.53 step 9.

The anchor acts only after the canonical tool prefix:
    <|tool|>\n{"arguments":{
and teaches the correct argument-key family:
    weather -> "location":
    calculator -> "expression":
    web_search -> "query":
    get_time -> "timezone":

Training prompts are distinct from the unchanged held-out and fixed eval sets.
Only transformer blocks 4 and 5 are trainable. Every rung resets to the exact
saved v0.0.53 checkpoint. Copy <22/35 or true-tool entry regression causes an
immediate rung rollback. CPU-only; no checkpoint save, export, promotion, or
production integration.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
import time
import zipfile

from huggingface_hub import HfApi
import torch
import torch.nn.functional as F

from jobs import ember_hf_eval as ev
from jobs import ember_v052_first_token_logits as ft
from jobs import ember_v053_heldout_routing as held
from jobs import ember_v053_first_token_anchor as base
from jobs import ember_v054_tool_family_prefix as fam
from jobs import ember_alternating_trust_region as trust

OUT = Path("v054-argkey-anchor")
TRAIN_GROUPS = {"block_04", "block_05"}
LRS = (8e-7, 1.6e-6, 3.2e-6)
MAX_STEPS = 10
FAMILY_MARGIN = 1.0
GRAD_CLIP = 0.25
COPY_FLOOR = 22
STRUCTURE_FLOOR = 7

SYSTEM_1 = (
    "You are Ember. Available tools are weather, calculator, web_search, and get_time. "
    "Use the single tool that actually matches the request. Tool calls use JSON arguments."
)
SYSTEM_2 = (
    "You are Ember. Choose among weather, calculator, web_search, and get_time when live data, "
    "arithmetic, web lookup, or current local time is required. Return a JSON tool call."
)


def prompt(system: str, user: str) -> str:
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"


def tool(case_id: str, user: str, expected_tool: str, system: str = SYSTEM_1) -> dict:
    return {"id": case_id, "kind": "tool_call", "user": user, "expected_tool": expected_tool, "prompt": prompt(system, user)}


# Deliberately distinct from held.CASES, config/ember_v0.0.8_eval.json, and prior v0.0.54 development prompts.
TRAIN = [
    tool("ak_weather_seattle", "Is it raining in Seattle right now?", "weather"),
    tool("ak_weather_tokyo", "What is the live temperature in Tokyo?", "weather", SYSTEM_2),
    tool("ak_weather_buffalo", "Is it snowing in Buffalo at the moment?", "weather"),
    tool("ak_weather_sf", "What are the current wind conditions in San Francisco?", "weather", SYSTEM_2),
    tool("ak_weather_austin", "What is the weather in Austin right now?", "weather"),
    tool("ak_calc_mul", "Calculate 73 multiplied by 48.", "calculator", SYSTEM_2),
    tool("ak_calc_div", "What is 9876 divided by 24?", "calculator"),
    tool("ak_calc_pct", "Calculate 12.5 percent of 640.", "calculator", SYSTEM_2),
    tool("ak_calc_pow", "Compute 3 to the power of 7.", "calculator"),
    tool("ak_calc_sub", "What is 1250 minus 487?", "calculator", SYSTEM_2),
    tool("ak_search_ruby", "Find the newest stable Ruby release.", "web_search"),
    tool("ak_search_ubuntu", "What is the latest Ubuntu LTS release?", "web_search", SYSTEM_2),
    tool("ak_search_fda", "Find a recent official FDA announcement.", "web_search"),
    tool("ak_search_docker", "What is the current stable Docker Engine release?", "web_search", SYSTEM_2),
    tool("ak_search_ts", "Find the latest stable TypeScript release.", "web_search"),
    tool("ak_time_tokyo", "What time is it in Tokyo right now?", "get_time", SYSTEM_2),
    tool("ak_time_cairo", "Tell me the current local time in Cairo.", "get_time"),
    tool("ak_time_auckland", "What is the time in Auckland at this moment?", "get_time", SYSTEM_2),
    tool("ak_time_mexico", "Give me the current time in Mexico City.", "get_time"),
    tool("ak_time_reykjavik", "What time is it in Reykjavik right now?", "get_time", SYSTEM_2),
]


def group_name(name: str) -> str:
    m = re.search(r"(?:^|\.)(?:h|blocks|layers)\.(\d+)(?:\.|$)", name)
    if m:
        return f"block_{int(m.group(1)):02d}"
    return "other"


def configure_trainable(model):
    names = []
    count = 0
    for name, p in model.named_parameters():
        yes = group_name(name) in TRAIN_GROUPS
        p.requires_grad_(yes)
        if yes:
            names.append(name)
            count += p.numel()
    if not names:
        raise RuntimeError("no block 4/5 parameters found")
    return names, count


def sequence_logprob(model, tokenizer, prompt_text: str, continuation: str):
    prefix_ids = list(tokenizer.encode(prompt_text + fam.PREFIX))
    full_ids = list(tokenizer.encode(prompt_text + fam.PREFIX + continuation))
    split = 0
    for a, b in zip(prefix_ids, full_ids):
        if a != b:
            break
        split += 1
    targets = [int(x) for x in full_ids[split:]]
    context = [int(x) for x in full_ids[:split]]
    if not targets or not context:
        raise RuntimeError("empty argument-key context/target")
    seq = context + targets
    if len(seq) > int(model.cfg.block_size):
        raise RuntimeError("argument-key sequence exceeds context window")
    x = torch.tensor([seq[:-1]], dtype=torch.long)
    logits, _ = model(x, None)
    logp = F.log_softmax(logits[0], dim=-1)
    start = len(context) - 1
    vals = [logp[start + i, tid] for i, tid in enumerate(targets)]
    return torch.stack(vals).sum(), len(targets)


def family_loss(model, tokenizer, case: dict):
    scores = {}
    token_counts = {}
    for family, key in fam.KEYS.items():
        s, n = sequence_logprob(model, tokenizer, case["prompt"], key)
        scores[family] = s
        token_counts[family] = n
    expected = scores[case["expected_tool"]]
    wrong = torch.stack([score for family, score in scores.items() if family != case["expected_tool"]]).max()
    return F.relu(wrong + FAMILY_MARGIN - expected), {
        "expected": float(expected.detach()),
        "best_wrong": float(wrong.detach()),
        "token_counts": token_counts,
    }


def update_once(model, tokenizer, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for c in TRAIN:
        loss, _ = family_loss(model, tokenizer, c)
        (loss / len(TRAIN)).backward()
        losses.append(float(loss.detach()))
    params = [p for p in model.parameters() if p.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    return {"family_hinge_mean": sum(losses) / len(losses), "grad_norm": float(grad_norm)}


def family_probe(model, tokenizer, cases):
    rows = [fam.inspect_case(model, tokenizer, c) for c in cases]
    return {"rows": rows, "aggregate": fam.aggregate(rows)}


def tool_entry_probe(model, tokenizer, cases):
    rows = []
    good = 0
    for c in cases:
        r = ft.inspect_case(model, tokenizer, c)
        ok = bool(r["argmax_is_tool"])
        good += int(ok)
        rows.append({"id": c["id"], "ok": ok, "tool_probability": float(r["tool_probability"]), "tool_rank": int(r["tool_rank"])})
    return {"good": good, "total": len(cases), "rows": rows}


def tool_generation_probe(model, tokenizer, cases, generation):
    rows = []
    correct = 0
    torch.manual_seed(int(generation["seed"]))
    for c in cases:
        completion = ev.generate_completion(model, tokenizer, torch, c["prompt"], generation)
        score = ev.score_case(c, completion)
        ok = bool(score.get("tool_name_matches")) and bool(score.get("json_valid"))
        correct += int(ok)
        rows.append({"id": c["id"], "completion": completion, "score": score, "correct": ok})
    return {"correct": correct, "total": len(cases), "rows": rows}


def summary(report):
    lines = [
        "# Ember v0.0.54 4-way argument-key margin anchor", "",
        f"Status: {report['status']}",
        "Source: exact saved v0.0.53 step-9 checkpoint.",
        f"Trainable groups: {', '.join(sorted(TRAIN_GROUPS))}",
        f"Family margin: {FAMILY_MARGIN}; LRs: {', '.join(f'{x:.1e}' for x in LRS)}; max {MAX_STEPS} steps/rung.",
        "", "| LR | Step | Train key | Held-out key | Fixed key | Held-out tool entry | Copy | Structure | Held-out tool gen | Fixed D/T gen | Pass |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for rung in report.get("rungs", []):
        for p in rung.get("candidate_probes", []):
            s = p["structure"]
            fg = p["fixed_generation"]
            lines.append(
                f"| {rung['lr']:.1e} | {p['step']} | {p['train_family']['aggregate']['correct']}/20 | "
                f"{p['heldout_family']['aggregate']['correct']}/8 | {p['fixed_family']['aggregate']['correct']}/4 | "
                f"{p['heldout_entry']['good']}/8 | {p['placement']['token_top1']}/35 | "
                f"{s['envelope_json_valid']}/8 / {s['tool_name_correct']}/8 | {p['heldout_tool_generation']['correct']}/8 | "
                f"{fg['direct_pass']}/4 / {fg['tool_pass']}/4 | {p['accepted']} |"
            )
    lines += ["", f"Operating point found: {report.get('operating_point_found', False)}", f"Selected LR/step: {report.get('selected_lr')} / {report.get('selected_step')}", f"Interpretation: {report.get('interpretation', '')}", "", "No checkpoint was saved, exported, promoted, or integrated."]
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
    report = {"schema_version": 1, "diagnostic": "ember-v054-argkey-anchor-v1", "status": "ERROR", "lrs": list(LRS), "max_steps": MAX_STEPS, "family_margin": FAMILY_MARGIN, "trainable_groups": sorted(TRAIN_GROUPS), "cpu_only": True, "checkpoint_save_authorized": False, "promotion_authorized": False, "rungs": []}

    with tempfile.TemporaryDirectory(prefix="ember-v054-argkey-") as td:
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
        names, count = configure_trainable(model)
        report["trainable_parameter_names"] = names
        report["trainable_parameter_count"] = count

        cfg, template, template_report, selected, selected_ids, *_ = base.build_placement_fixture(work)
        fixed_spec = json.loads(Path("config/ember_v0.0.8_eval.json").read_text())
        fixed_all = [c for c in fixed_spec["cases"] if c["kind"] in {"direct_response", "tool_call"}]
        fixed_tools = [c for c in fixed_all if c["kind"] == "tool_call"]
        held_tools = [c for c in held.CASES if c["kind"] == "tool_call"]

        baseline = {
            "placement": base.placement_probe(model, tokenizer, selected, template),
            "heldout_family": family_probe(model, tokenizer, held_tools),
            "fixed_family": family_probe(model, tokenizer, fixed_tools),
            "train_family": family_probe(model, tokenizer, TRAIN),
            "heldout_entry": tool_entry_probe(model, tokenizer, held_tools),
        }
        report["baseline"] = baseline
        report["template_report"] = template_report
        report["selected_case_ids"] = selected_ids
        if int(baseline["placement"]["token_top1"]) != COPY_FLOOR:
            raise RuntimeError("v0.0.53 copy baseline drifted")
        if baseline["heldout_entry"]["good"] != len(held_tools):
            raise RuntimeError("v0.0.53 held-out tool-entry baseline drifted")

        best = None
        best_score = (-1, -1, -1)
        found = False
        for lr in LRS:
            model.load_state_dict(pristine)
            configure_trainable(model)
            model.eval()
            if trust.trace.state_digest(model) != pristine_hash:
                raise RuntimeError("rung reset failed")
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(lr), weight_decay=0.0)
            rung = {"lr": float(lr), "steps": [], "candidate_probes": [], "rollback": None, "accepted": False}

            for step in range(1, MAX_STEPS + 1):
                before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
                update = update_once(model, tokenizer, opt)
                place = base.placement_probe(model, tokenizer, selected, template)
                entry = tool_entry_probe(model, tokenizer, held_tools)
                hfamily = family_probe(model, tokenizer, held_tools)
                tfamily = family_probe(model, tokenizer, TRAIN)
                compact = {"step": step, "update": update, "train_family_correct": tfamily["aggregate"]["correct"], "heldout_family_correct": hfamily["aggregate"]["correct"], "heldout_entry": entry["good"], "copy_tokens": int(place["token_top1"])}
                rung["steps"].append(compact)
                print(json.dumps({"event": "argkey_probe", "lr": lr, **compact}), flush=True)

                if int(place["token_top1"]) < COPY_FLOOR or entry["good"] < len(held_tools):
                    params = dict(model.named_parameters())
                    with torch.no_grad():
                        for n, v in before.items():
                            params[n].copy_(v)
                    rung["rollback"] = {"step": step, "reason": "copy" if int(place["token_top1"]) < COPY_FLOOR else "tool_entry"}
                    print(json.dumps({"event": "argkey_rollback", "lr": lr, **rung["rollback"]}), flush=True)
                    break

                score = (int(hfamily["aggregate"]["correct"]), int(tfamily["aggregate"]["correct"]), int(place["token_top1"]))
                if score > best_score:
                    best_score = score
                    best = {"lr": float(lr), "step": step, "state": copy.deepcopy(model.state_dict())}

                # Expensive gates only once held-out family routing becomes promising.
                if int(hfamily["aggregate"]["correct"]) >= 6:
                    ffamily = family_probe(model, tokenizer, fixed_tools)
                    structure = base.structure_probe(model, tokenizer, cfg, selected)
                    fixed_generation = base.generation_probe(model, tokenizer, fixed_all, fixed_spec["generation"])
                    held_generation = tool_generation_probe(model, tokenizer, held_tools, held.GENERATION)
                    accepted = (
                        int(hfamily["aggregate"]["correct"]) == 8
                        and int(tfamily["aggregate"]["correct"]) >= 18
                        and int(ffamily["aggregate"]["correct"]) == 4
                        and entry["good"] == 8
                        and int(place["token_top1"]) >= COPY_FLOOR
                        and int(structure["envelope_json_valid"]) >= STRUCTURE_FLOOR
                        and int(structure["tool_name_correct"]) >= STRUCTURE_FLOOR
                        and fixed_generation["direct_pass"] == 4
                        and fixed_generation["tool_pass"] == 4
                        and held_generation["correct"] >= 6
                    )
                    probe = {"step": step, "train_family": tfamily, "heldout_family": hfamily, "fixed_family": ffamily, "heldout_entry": entry, "placement": place, "structure": structure, "fixed_generation": fixed_generation, "heldout_tool_generation": held_generation, "accepted": accepted}
                    rung["candidate_probes"].append(probe)
                    print(json.dumps({"event": "argkey_full_gate", "lr": lr, "step": step, "heldout_family": hfamily["aggregate"]["correct"], "train_family": tfamily["aggregate"]["correct"], "fixed_family": ffamily["aggregate"]["correct"], "copy": place["token_top1"], "structure": [structure["envelope_json_valid"], structure["tool_name_correct"]], "heldout_tool_gen": held_generation["correct"], "fixed_gen": [fixed_generation["direct_pass"], fixed_generation["tool_pass"]], "accepted": accepted}), flush=True)
                    if accepted:
                        rung["accepted"] = True
                        report["selected_lr"] = float(lr)
                        report["selected_step"] = step
                        report["selected_probe"] = probe
                        found = True
                        break
            report["rungs"].append(rung)
            if found:
                break

        report["operating_point_found"] = found
        if found:
            report["status"] = "OPERATING_POINT_FOUND"
            report["interpretation"] = "A block-4/5-only 4-way argument-key margin anchor achieved the family gate while preserving routing, copy, structure, and fixed behavior."
        else:
            report["status"] = "NO_OPERATING_POINT"
            report["best_score"] = list(best_score)
            if best is not None:
                report["best_lr"] = best["lr"]
                report["best_step"] = best["step"]
                model.load_state_dict(best["state"])
                configure_trainable(model)
                report["best_diagnostic"] = {
                    "train_family": family_probe(model, tokenizer, TRAIN),
                    "heldout_family": family_probe(model, tokenizer, held_tools),
                    "fixed_family": family_probe(model, tokenizer, fixed_tools),
                    "heldout_entry": tool_entry_probe(model, tokenizer, held_tools),
                    "placement": base.placement_probe(model, tokenizer, selected, template),
                    "structure": base.structure_probe(model, tokenizer, cfg, selected),
                    "heldout_tool_generation": tool_generation_probe(model, tokenizer, held_tools, held.GENERATION),
                    "fixed_generation": base.generation_probe(model, tokenizer, fixed_all, fixed_spec["generation"]),
                }
            report["interpretation"] = "The exact argument-key decision is learnable only if held-out family accuracy rises without violating the protected copy/tool-entry floors; no strict point met every gate in this ladder."

    report["elapsed_seconds"] = time.monotonic() - started
    (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (OUT / "summary.md").write_text(summary(report))
    print(json.dumps({"event": "argkey_complete", "status": report["status"], "operating_point_found": report["operating_point_found"], "selected_lr": report.get("selected_lr"), "selected_step": report.get("selected_step"), "best_score": report.get("best_score"), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
