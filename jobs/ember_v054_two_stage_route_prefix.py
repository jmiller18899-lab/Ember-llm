"""Two-stage bounded routing repair for Ember v0.0.54.

Stage A repairs direct-vs-tool entry using only transformer blocks 0-1, where the
localization diagnostic found most direct-routing gradient. Its update gradient
is projected orthogonal to the protected copy gradient and is rolled back if the
22/35 copy floor or 8/8 true-tool entry is lost.

Stage B starts from the best safe Stage-A state and uses only blocks 3-5. It
teaches the *actual generated prefix* after <|tool|> to declare tool name and
argument schema before the value, e.g. {"name":"weather","arguments":{"location":
The evaluator is JSON-key-order agnostic, so this is contract-compatible. Stage
B is also copy-gradient projected and rollback guarded.

The existing v0.0.53 20-case challenge is now a development set. No checkpoint
is saved. Any passing point must still pass a fresh confirmation set later.
CPU-only; no promotion or production integration.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import math
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

OUT = Path("v054-two-stage-route-prefix")
A_GROUPS = {"block_00", "block_01"}
B_GROUPS = {"block_03", "block_04", "block_05"}
A_LRS = (8e-7, 1.6e-6, 3.2e-6)
B_LRS = (8e-7, 1.6e-6, 3.2e-6)
A_STEPS = 16
B_STEPS = 10
DIRECT_MARGIN = 1.0
TOOL_MARGIN = 1.0
GRAD_CLIP = 0.25
COPY_FLOOR = 22
STRUCTURE_FLOOR = 7
ARG_KEY = {
    "weather": "location",
    "calculator": "expression",
    "web_search": "query",
    "get_time": "timezone",
}


def configure_groups(model, groups: set[str]):
    names, count = [], 0
    for name, p in model.named_parameters():
        active = loc.group_name(name) in groups
        p.requires_grad_(active)
        if active:
            names.append(name)
            count += p.numel()
    if not names:
        raise RuntimeError(f"no parameters matched {sorted(groups)}")
    return names, count


def route_margin(logits, tool_id: int, special_ids: set[int], should_tool: bool):
    detached = logits.detach().clone()
    detached[list(special_ids)] = -float("inf")
    lexical = detached.max()
    tool = logits[tool_id]
    if should_tool:
        return F.relu(lexical + TOOL_MARGIN - tool)
    return F.relu(tool + DIRECT_MARGIN - lexical)


def clone_grads(params):
    return [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params]


def dot_grads(a, b):
    return sum(float((x * y).sum().item()) for x, y in zip(a, b))


def norm_grads(a):
    return math.sqrt(max(0.0, sum(float((x * x).sum().item()) for x in a)))


def projected_step(model, optimizer, repair_backward, copy_examples):
    """Take a repair step with its trainable gradient orthogonal to copy gradient."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    repair_stats = repair_backward()
    repair = clone_grads(params)

    optimizer.zero_grad(set_to_none=True)
    copy_loss = trust.objectives.batch_loss(model, torch, copy_examples, list(range(len(copy_examples))))
    if not bool(torch.isfinite(copy_loss).item()):
        raise RuntimeError("non-finite protected copy loss")
    copy_loss.backward()
    copy_grad = clone_grads(params)

    dot = dot_grads(repair, copy_grad)
    copy_sq = sum(float((g * g).sum().item()) for g in copy_grad)
    alpha = dot / copy_sq if copy_sq > 1e-20 else 0.0
    # Always remove the copy-gradient component. Discrete copy top-1 is guarded too.
    projected = [r - alpha * c for r, c in zip(repair, copy_grad)]
    optimizer.zero_grad(set_to_none=True)
    for p, g in zip(params, projected):
        p.grad = g
    projected_norm = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()

    rn, cn = norm_grads(repair), norm_grads(copy_grad)
    return {
        **repair_stats,
        "copy_loss": float(copy_loss.detach()),
        "repair_grad_norm": rn,
        "copy_grad_norm": cn,
        "repair_copy_cosine": dot / (rn * cn) if rn > 0 and cn > 0 else 0.0,
        "projection_alpha": alpha,
        "projected_grad_norm_preclip": float(projected_norm),
    }


def stage_a_backward(model, tokenizer, tool_id, special_ids):
    directs = [c for c in v54.TRAIN if c["kind"] == "direct_response"]
    tools = [c for c in v54.TRAIN if c["kind"] == "tool_call"]
    model.train()
    vals_d, vals_t = [], []
    for c in directs:
        loss = route_margin(base.next_logits(model, tokenizer, c["prompt"]), tool_id, special_ids, False)
        (0.80 * loss / len(directs)).backward()
        vals_d.append(float(loss.detach()))
    for c in tools:
        loss = route_margin(base.next_logits(model, tokenizer, c["prompt"]), tool_id, special_ids, True)
        (0.20 * loss / len(tools)).backward()
        vals_t.append(float(loss.detach()))
    return {
        "direct_hinge": sum(vals_d) / len(vals_d),
        "tool_hinge": sum(vals_t) / len(vals_t),
    }


def lcp(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if int(x) != int(y):
            break
        n += 1
    return n


def prefix_ce(model, tokenizer, case: dict):
    tool = case["expected_tool"]
    key = ARG_KEY[tool]
    forced = case["prompt"] + "<|tool|>\n"
    target = '{"name":' + json.dumps(tool) + ',"arguments":{' + json.dumps(key) + ':'
    a = list(tokenizer.encode(forced))
    b = list(tokenizer.encode(forced + target))
    split = lcp(a, b)
    targets = [int(x) for x in b[split:]]
    if not targets:
        raise RuntimeError(f"empty name-first prefix targets for {case['id']}")
    context = [int(x) for x in b[:split]]
    seq = context + targets
    if len(seq) > int(model.cfg.block_size):
        raise RuntimeError(f"prefix sequence too long for {case['id']}")
    x = torch.tensor([seq[:-1]], dtype=torch.long)
    logits, _ = model(x, None)
    start = len(context) - 1
    y = torch.tensor(targets, dtype=torch.long)
    selected = logits[0, start:start + len(targets)]
    return F.cross_entropy(selected, y), len(targets)


def stage_b_backward(model, tokenizer, tool_id, special_ids):
    directs = [c for c in v54.TRAIN if c["kind"] == "direct_response"]
    tools = [c for c in v54.TRAIN if c["kind"] == "tool_call"]
    model.train()
    prefix_vals, prefix_tokens, direct_vals, tool_vals = [], 0, [], []
    for c in tools:
        loss, n = prefix_ce(model, tokenizer, c)
        (0.75 * loss / len(tools)).backward()
        prefix_vals.append(float(loss.detach()))
        prefix_tokens += n
        r = route_margin(base.next_logits(model, tokenizer, c["prompt"]), tool_id, special_ids, True)
        (0.05 * r / len(tools)).backward()
        tool_vals.append(float(r.detach()))
    for c in directs:
        r = route_margin(base.next_logits(model, tokenizer, c["prompt"]), tool_id, special_ids, False)
        (0.20 * r / len(directs)).backward()
        direct_vals.append(float(r.detach()))
    return {
        "namefirst_prefix_ce": sum(prefix_vals) / len(prefix_vals),
        "namefirst_prefix_tokens": prefix_tokens,
        "direct_preserve_hinge": sum(direct_vals) / len(direct_vals),
        "tool_preserve_hinge": sum(tool_vals) / len(tool_vals),
    }


def snapshot(model):
    return copy.deepcopy(model.state_dict())


def restore(model, state):
    model.load_state_dict(state)
    model.eval()


def route_score(model, tokenizer):
    train = v54.routing_counts(model, tokenizer, v54.TRAIN)
    dev = v54.routing_counts(model, tokenizer, held.CASES)
    return train, dev


def full_gate(model, tokenizer, cfg, selected, template, fixed_cases, fixed_generation, label):
    dev_eval = held.evaluate(model, tokenizer, label)
    place = base.placement_probe(model, tokenizer, selected, template)
    structure = base.structure_probe(model, tokenizer, cfg, selected)
    fixed = base.generation_probe(model, tokenizer, fixed_cases, fixed_generation)
    m = dev_eval["metrics"]
    gates = {
        "dev_first": m["direct_first_token_pass"] == 12 and m["tool_first_token_pass"] == 8,
        "dev_generation": m["direct_generation_pass"] == 12 and m["tool_generation_pass"] == 8,
        "dev_tool_name": m["tool_name_correct"] == 8,
        "copy": int(place["token_top1"]) >= COPY_FLOOR,
        "structure": int(structure["envelope_json_valid"]) >= STRUCTURE_FLOOR and int(structure["tool_name_correct"]) >= STRUCTURE_FLOOR,
        "fixed": int(fixed["direct_pass"]) == 4 and int(fixed["tool_pass"]) == 4,
    }
    return {
        "dev": dev_eval,
        "placement": place,
        "structure": structure,
        "fixed": fixed,
        "gates": gates,
        "accepted": all(gates.values()),
    }


def summary_md(report):
    a = report.get("stage_a_best", {})
    b = report.get("stage_b_best", {})
    lines = [
        "# Ember v0.0.54 two-stage route/prefix diagnostic", "",
        f"Status: {report['status']}",
        "Source: exact saved v0.0.53 step-9 checkpoint.",
        "Stage A: blocks 0-1, direct-vs-tool routing, copy-gradient projection.",
        "Stage B: blocks 3-5, generated name-first tool prefix/schema, copy-gradient projection.",
        "The prior 20-case challenge is development only; a fresh confirmation is required before save.", "",
        f"Stage-A best: LR={a.get('lr')} step={a.get('step')} dev direct={a.get('dev_direct')}/12 tool={a.get('dev_tool')}/8 copy={a.get('copy_tokens')}/35.",
        f"Stage-B best: LR={b.get('lr')} step={b.get('step')} dev direct={b.get('dev_direct')}/12 tool={b.get('dev_tool')}/8 copy={b.get('copy_tokens')}/35.",
        f"Operating point found: {report.get('operating_point_found', False)}", "",
    ]
    gate = report.get("final_gate")
    if gate:
        m = gate["dev"]["metrics"]
        s = gate["structure"]
        lines += [
            f"Final dev first-token direct/tool: {m['direct_first_token_pass']}/12 / {m['tool_first_token_pass']}/8",
            f"Final dev generation direct/tool: {m['direct_generation_pass']}/12 / {m['tool_generation_pass']}/8",
            f"Final tool-name correct: {m['tool_name_correct']}/8",
            f"Copy: {gate['placement']['token_top1']}/35",
            f"Structure JSON/tool: {s['envelope_json_valid']}/8 / {s['tool_name_correct']}/8",
            f"Fixed routing generation D/T: {gate['fixed']['direct_pass']}/4 / {gate['fixed']['tool_pass']}/4",
            f"Accepted: {gate['accepted']}", "",
        ]
    lines += ["No checkpoint was saved, exported, promoted, or integrated.\n"]
    return "\n".join(lines)


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
        "diagnostic": "ember-v054-two-stage-route-prefix-v1",
        "status": "ERROR",
        "source": held.MODEL_NAME,
        "stage_a_groups": sorted(A_GROUPS),
        "stage_b_groups": sorted(B_GROUPS),
        "stage_a_lrs": list(A_LRS),
        "stage_b_lrs": list(B_LRS),
        "cpu_only": True,
        "checkpoint_save_authorized": False,
        "promotion_authorized": False,
        "prior_challenge_is_development": True,
        "fresh_confirmation_required": True,
        "stage_a": [],
        "stage_b": [],
    }

    with tempfile.TemporaryDirectory(prefix="ember-v054-two-stage-") as td:
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
        pristine = snapshot(model)
        pristine_hash = trust.trace.state_digest(model)
        if checkpoint.get("train_config", {}).get("version") != "0.0.53":
            raise RuntimeError("source is not saved v0.0.53")

        contract, tool_id, special_ids = v54.routing_contract(tokenizer)
        cfg, template, template_report, selected, selected_ids, *_ = base.build_placement_fixture(work)
        copy_tool_id, _ = trust.objectives.token_contract(tokenizer)
        copy_examples = [
            trust.objectives.supervised_example(tokenizer, c, "placement", template, copy_tool_id)
            for c in selected
        ]
        fixed_spec = json.loads(Path("config/ember_v0.0.8_eval.json").read_text())
        fixed_cases = [c for c in fixed_spec["cases"] if c["kind"] in {"direct_response", "tool_call"}]
        base_place = base.placement_probe(model, tokenizer, selected, template)
        if int(base_place["token_top1"]) != COPY_FLOOR:
            raise RuntimeError("v0.0.53 copy baseline drifted")
        base_train, base_dev = route_score(model, tokenizer)
        report["baseline"] = {
            "copy": base_place,
            "train_routing": base_train,
            "dev_routing": base_dev,
            "state_sha256": pristine_hash,
            "selected_case_ids": selected_ids,
            "template": template_report,
        }

        # Stage A: find best safe direct-routing state.
        best_a = None
        best_a_score = (-1, -1, float("-inf"))
        for lr in A_LRS:
            restore(model, pristine)
            configure_groups(model, A_GROUPS)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(lr), weight_decay=0.0)
            rung = {"lr": float(lr), "steps": [], "rollback": None}
            for step in range(1, A_STEPS + 1):
                pre = snapshot(model)
                preopt = copy.deepcopy(opt.state_dict())
                stats = projected_step(
                    model, opt,
                    lambda: stage_a_backward(model, tokenizer, tool_id, special_ids),
                    copy_examples,
                )
                train, dev = route_score(model, tokenizer)
                place = base.placement_probe(model, tokenizer, selected, template)
                rec = {
                    "step": step, "stats": stats,
                    "train_direct": train["direct_ok"], "train_tool": train["tool_ok"],
                    "dev_direct": dev["direct_ok"], "dev_tool": dev["tool_ok"],
                    "copy_tokens": int(place["token_top1"]), "copy_mean_loss": float(place["mean_loss"]),
                }
                rung["steps"].append(rec)
                print(json.dumps({"event":"two_stage_a_probe", "lr":lr, **rec}), flush=True)
                if int(place["token_top1"]) < COPY_FLOOR or dev["tool_ok"] < 8:
                    restore(model, pre); opt.load_state_dict(preopt)
                    rung["rollback"] = {"step": step, "reason": "copy" if int(place["token_top1"]) < COPY_FLOOR else "tool_entry"}
                    print(json.dumps({"event":"two_stage_a_rollback", "lr":lr, **rung["rollback"]}), flush=True)
                    break
                score = (int(dev["direct_ok"]), int(train["direct_ok"]), -float(place["mean_loss"]))
                if score > best_a_score:
                    best_a_score = score
                    best_a = {"lr": float(lr), "step": step, "state": snapshot(model), "dev_direct":dev["direct_ok"], "dev_tool":dev["tool_ok"], "train_direct":train["direct_ok"], "copy_tokens":int(place["token_top1"])}
                if dev["direct_ok"] == 12 and train["direct_ok"] >= 14:
                    break
            report["stage_a"].append(rung)
            del opt

        if best_a is None:
            raise RuntimeError("no safe Stage-A state")
        report["stage_a_best"] = {k:v for k,v in best_a.items() if k != "state"}
        restore(model, best_a["state"])
        stage_a_hash = trust.trace.state_digest(model)
        report["stage_a_state_sha256"] = stage_a_hash

        # Stage B: name-first prefix/schema repair from best Stage-A state.
        best_b = None
        best_b_score = (-1, -1, -1, float("-inf"))
        for lr in B_LRS:
            restore(model, best_a["state"])
            configure_groups(model, B_GROUPS)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(lr), weight_decay=0.0)
            rung = {"lr": float(lr), "steps": [], "rollback": None, "end_gate": None}
            safe_state = snapshot(model)
            for step in range(1, B_STEPS + 1):
                pre = snapshot(model)
                preopt = copy.deepcopy(opt.state_dict())
                stats = projected_step(
                    model, opt,
                    lambda: stage_b_backward(model, tokenizer, tool_id, special_ids),
                    copy_examples,
                )
                train, dev = route_score(model, tokenizer)
                place = base.placement_probe(model, tokenizer, selected, template)
                rec = {
                    "step": step, "stats": stats,
                    "train_direct": train["direct_ok"], "train_tool": train["tool_ok"],
                    "dev_direct": dev["direct_ok"], "dev_tool": dev["tool_ok"],
                    "copy_tokens": int(place["token_top1"]), "copy_mean_loss": float(place["mean_loss"]),
                }
                rung["steps"].append(rec)
                print(json.dumps({"event":"two_stage_b_probe", "lr":lr, **rec}), flush=True)
                # Preserve Stage-A direct gain within one case while learning prefix.
                min_direct = max(3, int(best_a["dev_direct"]) - 1)
                if int(place["token_top1"]) < COPY_FLOOR or dev["tool_ok"] < 8 or dev["direct_ok"] < min_direct:
                    restore(model, pre); opt.load_state_dict(preopt)
                    rung["rollback"] = {"step": step, "reason": "copy" if int(place["token_top1"]) < COPY_FLOOR else ("tool_entry" if dev["tool_ok"] < 8 else "direct_regression")}
                    print(json.dumps({"event":"two_stage_b_rollback", "lr":lr, **rung["rollback"]}), flush=True)
                    break
                safe_state = snapshot(model)

            restore(model, safe_state)
            end_gate = full_gate(model, tokenizer, cfg, selected, template, fixed_cases, fixed_spec["generation"], f"two-stage-b-{lr:.1e}")
            rung["end_gate"] = end_gate
            m = end_gate["dev"]["metrics"]
            score = (int(m["tool_name_correct"]), int(m["direct_generation_pass"]), int(m["tool_generation_pass"]), -float(end_gate["placement"]["mean_loss"]))
            print(json.dumps({
                "event":"two_stage_b_full_gate", "lr":lr,
                "dev_first_direct":m["direct_first_token_pass"], "dev_first_tool":m["tool_first_token_pass"],
                "dev_gen_direct":m["direct_generation_pass"], "dev_gen_tool":m["tool_generation_pass"],
                "tool_name":m["tool_name_correct"], "copy":end_gate["placement"]["token_top1"],
                "structure_json":end_gate["structure"]["envelope_json_valid"], "structure_tool":end_gate["structure"]["tool_name_correct"],
                "accepted":end_gate["accepted"], "gates":end_gate["gates"],
            }), flush=True)
            if score > best_b_score:
                best_b_score = score
                best_b = {"lr":float(lr), "step":len(rung["steps"]), "state":snapshot(model), "gate":end_gate,
                          "dev_direct":m["direct_first_token_pass"], "dev_tool":m["tool_first_token_pass"], "copy_tokens":int(end_gate["placement"]["token_top1"])}
            report["stage_b"].append(rung)
            if end_gate["accepted"]:
                break
            del opt

        if best_b is None:
            raise RuntimeError("no safe Stage-B state")
        report["stage_b_best"] = {k:v for k,v in best_b.items() if k not in {"state","gate"}}
        report["final_gate"] = best_b["gate"]
        report["operating_point_found"] = bool(best_b["gate"]["accepted"])
        report["best_state_sha256"] = trust.trace.state_digest(model)
        report["interpretation"] = (
            "A simultaneous route/tool-family/copy operating point was found; run a fresh confirmation set before save."
            if report["operating_point_found"] else
            "No simultaneous operating point yet; use the split-stage evidence to choose the next bounded repair."
        )
        report["elapsed_seconds"] = time.monotonic() - started
        report["status"] = "PASS"
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (OUT / "summary.md").write_text(summary_md(report))
        print(json.dumps({"event":"two_stage_complete", "operating_point_found":report["operating_point_found"],
                          "stage_a_best":report["stage_a_best"], "stage_b_best":report["stage_b_best"],
                          "elapsed_seconds":report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
