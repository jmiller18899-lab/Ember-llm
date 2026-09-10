"""Bounded v0.0.54 args-first schema-key anchor.

Starts from a deterministic reproduction of the best safe v0.0.54 two-stage
state (10/12 development direct routing, 8/8 tool entry, 22/35 protected copy).
The new objective acts at the proven tool-family fork under:
    <|tool|>\n{"arguments":{
It classifies the next schema key among location/expression/query/timezone and
lightly teaches the complete key span. It never trains argument values.

Only transformer blocks 3-5 are trainable. Every repair gradient is projected
against the protected copy gradient; every step is rolled back if copy drops
below 22/35, true-tool entry drops below 8/8, or development direct routing
falls below 9/12. The prior 20-case challenge is development only; any passing
point still requires a brand-new confirmation set before save.

CPU-only. No checkpoint save/export/promotion/integration.
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
from jobs import ember_v054_two_stage_route_prefix as ts
from jobs import ember_v054_schema_key_logits as sk
from jobs import ember_alternating_trust_region as trust

OUT = Path("v054-schema-key-anchor")
TRAIN_GROUPS = {"block_03", "block_04", "block_05"}
LRS = (8e-7, 1.6e-6, 3.2e-6)
MAX_STEPS = 12
SCHEMA_MARGIN = 0.50
FIRST_CLASS_WEIGHT = 0.60
KEY_SPAN_WEIGHT = 0.20
DIRECT_PRESERVE_WEIGHT = 0.15
TOOL_ENTRY_WEIGHT = 0.05
COPY_FLOOR = 22
STRUCTURE_FLOOR = 7
MIN_DEV_DIRECT = 9
GRAD_CLIP = 0.25
KEY_ORDER = ("location", "expression", "query", "timezone")


def configure_trainable(model):
    names, count = [], 0
    for name, p in model.named_parameters():
        active = ts.loc.group_name(name) in TRAIN_GROUPS
        p.requires_grad_(active)
        if active:
            names.append(name)
            count += p.numel()
    if not names:
        raise RuntimeError("no block 3-5 parameters found")
    return names, count


def key_context(tokenizer, prompt: str):
    base_text = prompt + '<|tool|>\n{"arguments":{'
    enc = {
        key: [int(x) for x in tokenizer.encode(base_text + sk.KEY_SUFFIX[key])]
        for key in KEY_ORDER
    }
    common = sk.lcp_many(list(enc.values()))
    if common < 1:
        raise RuntimeError("schema candidates have no common token prefix")
    common_ids = list(next(iter(enc.values())))[:common]
    first_ids = {}
    tails = {}
    for key in KEY_ORDER:
        tail = enc[key][common:]
        if not tail:
            raise RuntimeError(f"empty candidate tail for {key}")
        first_ids[key] = int(tail[0])
        tails[key] = tail
    if len(set(first_ids.values())) != len(KEY_ORDER):
        raise RuntimeError(f"schema first tokens are not unique: {first_ids}")
    return common_ids, first_ids, tails


def schema_losses(model, tokenizer, case: dict):
    expected = sk.EXPECTED_KEY[case["expected_tool"]]
    common, first_ids, tails = key_context(tokenizer, case["prompt"])
    x = torch.tensor([common], dtype=torch.long)
    logits, _ = model(x, None)
    next_logits = logits[0, -1]
    candidate_logits = torch.stack([next_logits[first_ids[k]] for k in KEY_ORDER])
    target_index = KEY_ORDER.index(expected)
    class_ce = F.cross_entropy(candidate_logits[None, :], torch.tensor([target_index], dtype=torch.long))
    correct = candidate_logits[target_index]
    wrong = torch.cat((candidate_logits[:target_index], candidate_logits[target_index + 1:])).max()
    margin = F.relu(wrong + SCHEMA_MARGIN - correct)

    # Light span CE for the complete correct key, still stopping before its value.
    tail = tails[expected]
    seq = common + tail
    inp = torch.tensor([seq[:-1]], dtype=torch.long)
    all_logits, _ = model(inp, None)
    start = len(common) - 1
    selected = all_logits[0, start:start + len(tail)]
    y = torch.tensor(tail, dtype=torch.long)
    span_ce = F.cross_entropy(selected, y)
    return class_ce + margin, span_ce, {
        "expected_key": expected,
        "candidate_logits": {k: float(candidate_logits[i].detach()) for i, k in enumerate(KEY_ORDER)},
        "class_ce": float(class_ce.detach()),
        "margin_loss": float(margin.detach()),
        "span_ce": float(span_ce.detach()),
        "key_tokens": len(tail),
    }


def repair_backward(model, tokenizer, tool_id, special_ids):
    directs = [c for c in v54.TRAIN if c["kind"] == "direct_response"]
    tools = [c for c in v54.TRAIN if c["kind"] == "tool_call"]
    model.train()
    class_vals, span_vals, direct_vals, tool_vals = [], [], [], []
    margin_vals = []
    for c in tools:
        class_loss, span_loss, detail = schema_losses(model, tokenizer, c)
        (FIRST_CLASS_WEIGHT * class_loss / len(tools)).backward()
        (KEY_SPAN_WEIGHT * span_loss / len(tools)).backward()
        class_vals.append(detail["class_ce"])
        margin_vals.append(detail["margin_loss"])
        span_vals.append(detail["span_ce"])

        route = ts.route_margin(base.next_logits(model, tokenizer, c["prompt"]), tool_id, special_ids, True)
        (TOOL_ENTRY_WEIGHT * route / len(tools)).backward()
        tool_vals.append(float(route.detach()))

    for c in directs:
        route = ts.route_margin(base.next_logits(model, tokenizer, c["prompt"]), tool_id, special_ids, False)
        (DIRECT_PRESERVE_WEIGHT * route / len(directs)).backward()
        direct_vals.append(float(route.detach()))

    return {
        "schema_class_ce": sum(class_vals) / len(class_vals),
        "schema_margin": sum(margin_vals) / len(margin_vals),
        "schema_span_ce": sum(span_vals) / len(span_vals),
        "direct_hinge": sum(direct_vals) / len(direct_vals),
        "tool_hinge": sum(tool_vals) / len(tool_vals),
    }


def schema_probe(model, tokenizer, cases):
    rows = []
    correct = 0
    query_wins = 0
    for c in [r for r in cases if r["kind"] == "tool_call"]:
        expected = sk.EXPECTED_KEY[c["expected_tool"]]
        base_text = c["prompt"] + '<|tool|>\n{"arguments":{'
        scores = sk.continuation_scores(model, tokenizer, base_text, sk.KEY_SUFFIX)
        choice = max(scores["candidates"], key=lambda k: scores["candidates"][k]["first_probability"])
        correct += int(choice == expected)
        query_wins += int(choice == "query")
        rows.append({
            "id": c["id"],
            "expected_tool": c["expected_tool"],
            "expected_key": expected,
            "choice": choice,
            "probabilities": {k: v["first_probability"] for k, v in scores["candidates"].items()},
            "ranks": {k: v["first_rank"] for k, v in scores["candidates"].items()},
        })
    return {"correct": correct, "total": len(rows), "query_wins": query_wins, "rows": rows}


def reproduce_base(model, tokenizer, selected, template):
    # Exact best-safe state from prior two-stage run.
    contract, tool_id, special_ids = v54.routing_contract(tokenizer)
    copy_tool_id, _ = trust.objectives.token_contract(tokenizer)
    copy_examples = [
        trust.objectives.supervised_example(tokenizer, c, "placement", template, copy_tool_id)
        for c in selected
    ]
    ts.configure_groups(model, ts.A_GROUPS)
    opt_a = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=8e-7, weight_decay=0.0)
    ts.projected_step(model, opt_a, lambda: ts.stage_a_backward(model, tokenizer, tool_id, special_ids), copy_examples)
    ts.configure_groups(model, ts.B_GROUPS)
    opt_b = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3.2e-6, weight_decay=0.0)
    for _ in range(10):
        ts.projected_step(model, opt_b, lambda: ts.stage_b_backward(model, tokenizer, tool_id, special_ids), copy_examples)
    route = v54.routing_counts(model, tokenizer, held.CASES)
    place = base.placement_probe(model, tokenizer, selected, template)
    schema = schema_probe(model, tokenizer, held.CASES)
    if int(route["direct_ok"]) != 10 or int(route["tool_ok"]) != 8 or int(place["token_top1"]) != 22 or int(schema["correct"]) != 3:
        raise RuntimeError(
            f"v0.0.54-best reproduction drifted: direct={route['direct_ok']} tool={route['tool_ok']} "
            f"copy={place['token_top1']} schema={schema['correct']}"
        )
    return copy.deepcopy(model.state_dict()), {
        "routing": route, "placement": place, "schema": schema,
    }, copy_examples, tool_id, special_ids


def full_gate(model, tokenizer, cfg, selected, template, fixed_cases, fixed_generation, label):
    dev_eval = held.evaluate(model, tokenizer, label)
    place = base.placement_probe(model, tokenizer, selected, template)
    structure = base.structure_probe(model, tokenizer, cfg, selected)
    fixed = base.generation_probe(model, tokenizer, fixed_cases, fixed_generation)
    schema = schema_probe(model, tokenizer, held.CASES)
    m = dev_eval["metrics"]
    gates = {
        "dev_first": m["direct_first_token_pass"] == 12 and m["tool_first_token_pass"] == 8,
        "dev_generation": m["direct_generation_pass"] == 12 and m["tool_generation_pass"] == 8,
        "dev_tool_name": m["tool_name_correct"] == 8,
        "dev_schema": schema["correct"] == 8,
        "copy": int(place["token_top1"]) >= COPY_FLOOR,
        "structure": int(structure["envelope_json_valid"]) >= STRUCTURE_FLOOR and int(structure["tool_name_correct"]) >= STRUCTURE_FLOOR,
        "fixed": int(fixed["direct_pass"]) == 4 and int(fixed["tool_pass"]) == 4,
    }
    return {
        "dev": dev_eval, "schema": schema, "placement": place,
        "structure": structure, "fixed": fixed, "gates": gates,
        "accepted": all(gates.values()),
    }


def summary_md(report):
    lines = [
        "# Ember v0.0.54 args-first schema-key anchor", "",
        f"Status: {report['status']}",
        "Source: reproduced best-safe v0.0.54 two-stage state.",
        "Target: first schema-key decision under `<|tool|>\\n{\"arguments\":{`.",
        f"Trainable: {', '.join(sorted(TRAIN_GROUPS))}; copy floor {COPY_FLOOR}/35.",
        "Prior 20-case challenge is development only; fresh confirmation required before save.", "",
        "| LR | Best step | Dev direct | Dev tool entry | Dev schema key | Copy | Full tool-name | Accepted |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rung in report.get("rungs", []):
        b = rung.get("best")
        if not b:
            lines.append(f"| {rung['lr']:.1e} | — | — | — | — | — | — | False |")
            continue
        fg = b.get("full_gate")
        tool_name = fg["dev"]["metrics"]["tool_name_correct"] if fg else "—"
        accepted = fg["accepted"] if fg else False
        lines.append(
            f"| {rung['lr']:.1e} | {b['step']} | {b['dev_direct']}/12 | {b['dev_tool']}/8 | "
            f"{b['dev_schema']}/8 | {b['copy_tokens']}/35 | {tool_name}/8 | {accepted} |"
        )
    lines += ["",
        f"Operating point found: {report.get('operating_point_found', False)}",
        f"Selected LR/step: {report.get('selected_lr')} / {report.get('selected_step')}",
        f"Interpretation: {report.get('interpretation', '')}", "",
        "No checkpoint was saved, exported, promoted, or integrated.\n",
    ]
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
        "diagnostic": "ember-v054-args-first-schema-key-anchor-v1",
        "status": "ERROR",
        "lrs": list(LRS),
        "max_steps": MAX_STEPS,
        "trainable_groups": sorted(TRAIN_GROUPS),
        "cpu_only": True,
        "checkpoint_save_authorized": False,
        "promotion_authorized": False,
        "fresh_confirmation_required": True,
        "rungs": [],
    }

    with tempfile.TemporaryDirectory(prefix="ember-v054-schema-anchor-") as td:
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
        if checkpoint.get("train_config", {}).get("version") != "0.0.53":
            raise RuntimeError("source is not exact saved v0.0.53")
        cfg, template, template_report, selected, selected_ids, *_ = base.build_placement_fixture(work)
        best_base, base_metrics, copy_examples, tool_id, special_ids = reproduce_base(model, tokenizer, selected, template)
        report["base"] = base_metrics
        report["base_state_sha256"] = trust.trace.state_digest(model)
        report["selected_case_ids"] = selected_ids
        report["template"] = template_report

        fixed_spec = json.loads(Path("config/ember_v0.0.8_eval.json").read_text())
        fixed_cases = [c for c in fixed_spec["cases"] if c["kind"] in {"direct_response", "tool_call"}]

        global_best = None
        global_score = (-1, -1, -1, float("-inf"))
        found = False
        for lr in LRS:
            model.load_state_dict(best_base); model.eval()
            configure_trainable(model)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(lr), weight_decay=0.0)
            rung = {"lr": float(lr), "steps": [], "rollback": None, "best": None}
            rung_best = None
            rung_score = (-1, -1, -1, float("-inf"))

            for step in range(1, MAX_STEPS + 1):
                pre = copy.deepcopy(model.state_dict())
                preopt = copy.deepcopy(opt.state_dict())
                stats = ts.projected_step(
                    model, opt,
                    lambda: repair_backward(model, tokenizer, tool_id, special_ids),
                    copy_examples,
                )
                train_route = v54.routing_counts(model, tokenizer, v54.TRAIN)
                dev_route = v54.routing_counts(model, tokenizer, held.CASES)
                dev_schema = schema_probe(model, tokenizer, held.CASES)
                train_schema = schema_probe(model, tokenizer, v54.TRAIN)
                place = base.placement_probe(model, tokenizer, selected, template)
                rec = {
                    "step": step, "stats": stats,
                    "train_direct": train_route["direct_ok"], "train_tool": train_route["tool_ok"],
                    "train_schema": train_schema["correct"],
                    "dev_direct": dev_route["direct_ok"], "dev_tool": dev_route["tool_ok"],
                    "dev_schema": dev_schema["correct"], "dev_query_wins": dev_schema["query_wins"],
                    "copy_tokens": int(place["token_top1"]), "copy_mean_loss": float(place["mean_loss"]),
                }
                rung["steps"].append(rec)
                print(json.dumps({"event":"schema_anchor_probe", "lr":lr, **rec}), flush=True)

                if int(place["token_top1"]) < COPY_FLOOR or int(dev_route["tool_ok"]) < 8 or int(dev_route["direct_ok"]) < MIN_DEV_DIRECT:
                    model.load_state_dict(pre); model.eval(); configure_trainable(model); opt.load_state_dict(preopt)
                    reason = "copy" if int(place["token_top1"]) < COPY_FLOOR else ("tool_entry" if int(dev_route["tool_ok"]) < 8 else "direct_regression")
                    rung["rollback"] = {"step": step, "reason": reason}
                    print(json.dumps({"event":"schema_anchor_rollback", "lr":lr, **rung["rollback"]}), flush=True)
                    break

                score = (int(dev_schema["correct"]), int(dev_route["direct_ok"]), int(train_schema["correct"]), -float(place["mean_loss"]))
                if score > rung_score:
                    rung_score = score
                    rung_best = {
                        "step": step, "state": copy.deepcopy(model.state_dict()),
                        "dev_direct": int(dev_route["direct_ok"]), "dev_tool": int(dev_route["tool_ok"]),
                        "dev_schema": int(dev_schema["correct"]), "train_schema": int(train_schema["correct"]),
                        "copy_tokens": int(place["token_top1"]), "copy_mean_loss": float(place["mean_loss"]),
                    }

                if int(dev_schema["correct"]) >= 6 and int(dev_route["direct_ok"]) >= 10:
                    gate = full_gate(model, tokenizer, cfg, selected, template, fixed_cases, fixed_spec["generation"], f"schema-anchor-{lr:.1e}-s{step}")
                    print(json.dumps({
                        "event":"schema_anchor_full_gate", "lr":lr, "step":step,
                        "dev_first_direct":gate["dev"]["metrics"]["direct_first_token_pass"],
                        "dev_gen_direct":gate["dev"]["metrics"]["direct_generation_pass"],
                        "dev_gen_tool":gate["dev"]["metrics"]["tool_generation_pass"],
                        "tool_name":gate["dev"]["metrics"]["tool_name_correct"],
                        "schema":gate["schema"]["correct"], "copy":gate["placement"]["token_top1"],
                        "structure_json":gate["structure"]["envelope_json_valid"],
                        "structure_tool":gate["structure"]["tool_name_correct"],
                        "accepted":gate["accepted"], "gates":gate["gates"],
                    }), flush=True)
                    if rung_best and rung_best["step"] == step:
                        rung_best["full_gate"] = gate
                    if gate["accepted"]:
                        rung_best = rung_best or {"step":step, "state":copy.deepcopy(model.state_dict())}
                        rung_best["full_gate"] = gate
                        rung["best"] = {k:v for k,v in rung_best.items() if k != "state"}
                        report["operating_point_found"] = True
                        report["selected_lr"] = float(lr)
                        report["selected_step"] = step
                        report["selected_gate"] = gate
                        found = True
                        break

            if rung_best:
                model.load_state_dict(rung_best["state"]); model.eval()
                if "full_gate" not in rung_best:
                    rung_best["full_gate"] = full_gate(model, tokenizer, cfg, selected, template, fixed_cases, fixed_spec["generation"], f"schema-anchor-best-{lr:.1e}")
                rung["best"] = {k:v for k,v in rung_best.items() if k != "state"}
                if rung_score > global_score:
                    global_score = rung_score
                    global_best = {"lr":float(lr), **rung_best}
            report["rungs"].append(rung)
            del opt
            if found:
                break

        if not found:
            report["operating_point_found"] = False
            report["selected_lr"] = None
            report["selected_step"] = None
            if global_best:
                report["best_safe"] = {k:v for k,v in global_best.items() if k != "state"}
            report["interpretation"] = (
                "The schema-key anchor improved the proven family fork but did not clear every development/copy/structure gate. "
                "Use the best-safe failure pattern to choose the next bounded repair."
            )
        else:
            report["interpretation"] = (
                "The args-first schema-key anchor cleared the development gates. Run a brand-new confirmation set before saving v0.0.54."
            )

        report["elapsed_seconds"] = time.monotonic() - started
        report["status"] = "PASS"
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (OUT / "summary.md").write_text(summary_md(report))
        print(json.dumps({
            "event":"schema_anchor_complete",
            "operating_point_found":report["operating_point_found"],
            "selected_lr":report["selected_lr"], "selected_step":report["selected_step"],
            "best_safe":report.get("best_safe"), "elapsed_seconds":report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
