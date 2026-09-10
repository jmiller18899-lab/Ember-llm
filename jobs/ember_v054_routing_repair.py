"""Bounded v0.0.54 routing repair diagnostic on saved Ember v0.0.53 step 9.

Training prompts are deliberately distinct from the v0.0.53 held-out challenge.
The repair has two narrow objectives only:
  1) first-token direct-vs-tool margin across fresh generic all-tools prompts;
  2) tool-name token supervision on true tool cases.

It does not train full direct answers or full JSON envelopes. Candidate points are
validated against the unchanged v0.0.53 held-out challenge, the original fixed
routing cases, the 22/35 copy-token floor, and the structural floor.

CPU-only. No checkpoint save, export, promotion, or production integration.
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
from jobs import ember_v052_first_token_logits as ft
from jobs import ember_v053_first_token_anchor as base
from jobs import ember_v053_first_token_margin_anchor as margin
from jobs import ember_v053_heldout_routing as held
from jobs import ember_alternating_trust_region as trust

OUT = Path("v054-routing-repair")
MODEL_NAME = "ember-v0.0.53-t4"
ANCHOR_LRS = (8e-8, 2.4e-7, 6.4e-7)
MAX_STEPS = 12
DIRECT_MARGIN = 0.75
TOOL_MARGIN = 1.50
NAME_WEIGHT = 0.25
GRAD_CLIP = 0.25
COPY_TOKEN_FLOOR = 22
STRUCTURE_FLOOR = 7

SYSTEM_1 = (
    "You are Ember. Available tools: weather, calculator, web_search, get_time. "
    "Use a tool only for live weather, arithmetic, current web facts, or current local time. "
    "Otherwise answer directly. Tool calls use JSON arguments."
)
SYSTEM_2 = (
    "You are Ember. You can answer normally or call weather, calculator, web_search, or get_time. "
    "Do not call a tool merely because its name or words like current/latest appear in the request. "
    "Call one only when the task truly requires that capability."
)
SYSTEM_3 = (
    "You are Ember. Choose between a direct answer and these tools: weather, calculator, web_search, get_time. "
    "Writing, rewriting, explaining, summarizing, and planning are direct tasks unless live data or arithmetic is requested."
)


def make_prompt(system: str, user: str) -> str:
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"


def direct(case_id: str, user: str, system: str = SYSTEM_1) -> dict:
    return {"id": case_id, "kind": "direct_response", "user": user, "prompt": make_prompt(system, user)}


def tool(case_id: str, user: str, expected_tool: str, arguments: dict, system: str = SYSTEM_1) -> dict:
    completion = "<|tool|>\n" + json.dumps(
        {"arguments": arguments, "name": expected_tool}, separators=(",", ":")
    ) + "\n<|endoftext|>"
    return {
        "id": case_id,
        "kind": "tool_call",
        "user": user,
        "prompt": make_prompt(system, user),
        "expected_tool": expected_tool,
        "arguments": arguments,
        "target_completion": completion,
    }


# These are training/development prompts, not any prompt from held.CASES.
TRAIN = [
    direct("d_welcome_alt", "Write one warm sentence welcoming someone to a project.", SYSTEM_3),
    direct("d_bug_rephrase", "Turn this into a clear title: settings page acts weird after save.", SYSTEM_2),
    direct("d_weather_definition", "Explain what a weather forecast is without checking today's forecast.", SYSTEM_2),
    direct("d_weather_rewrite", "Rewrite this heading: weather screen is broken again.", SYSTEM_3),
    direct("d_calculator_definition", "In one sentence, explain what a calculator does.", SYSTEM_2),
    direct("d_calculator_reason", "Give one reason calculators are useful for students.", SYSTEM_3),
    direct("d_timezone_definition", "Explain the idea of a time zone without telling me the current time.", SYSTEM_2),
    direct("d_time_phrase", "Rewrite the phrase current time display as a short feature title.", SYSTEM_3),
    direct("d_search_definition", "Define web search in plain English without performing a search.", SYSTEM_2),
    direct("d_latest_rephrase", "Rewrite latest release info into a professional section heading.", SYSTEM_3),
    direct("d_current_rephrase", "Make this friendlier: current status unavailable, try later.", SYSTEM_2),
    direct("d_summary_alt", "Summarize this: tests passed and the build is ready for review.", SYSTEM_3),
    direct("d_plan_alt", "Give two short steps for testing a password reset form.", SYSTEM_1),
    direct("d_compare_alt", "Compare a folder and a file in one sentence.", SYSTEM_3),
    direct("d_sentiment_alt", "Is the sentence 'The fix worked perfectly' positive, negative, or neutral?", SYSTEM_1),
    direct("d_search_title", "Shorten this title without searching: find latest information page.", SYSTEM_2),

    tool("t_weather_boston", "What is the weather in Boston right now?", "weather", {"location": "Boston"}, SYSTEM_1),
    tool("t_weather_denver", "Do I need a jacket for the weather in Denver at the moment?", "weather", {"location": "Denver"}, SYSTEM_2),
    tool("t_weather_phoenix", "What is the live temperature in Phoenix?", "weather", {"location": "Phoenix"}, SYSTEM_3),
    tool("t_weather_dublin", "Is it raining in Dublin right now?", "weather", {"location": "Dublin"}, SYSTEM_1),
    tool("t_calc_multiply", "Calculate 812 multiplied by 37.", "calculator", {"expression": "812*37"}, SYSTEM_2),
    tool("t_calc_divide", "What is 1440 divided by 12?", "calculator", {"expression": "1440/12"}, SYSTEM_3),
    tool("t_calc_percent", "Calculate 18 percent of 250.", "calculator", {"expression": "0.18*250"}, SYSTEM_1),
    tool("t_calc_power", "Compute 2 to the power of 12.", "calculator", {"expression": "2**12"}, SYSTEM_2),
    tool("t_search_go", "Find the newest stable Go release.", "web_search", {"query": "newest stable Go release"}, SYSTEM_1),
    tool("t_search_kernel", "What is the current stable Linux kernel release?", "web_search", {"query": "current stable Linux kernel release"}, SYSTEM_2),
    tool("t_search_postgres", "Find the latest PostgreSQL major release.", "web_search", {"query": "latest PostgreSQL major release"}, SYSTEM_3),
    tool("t_search_nasa", "Find a recent official NASA announcement.", "web_search", {"query": "recent official NASA announcement"}, SYSTEM_1),
    tool("t_time_honolulu", "What time is it in Honolulu right now?", "get_time", {"timezone": "Honolulu"}, SYSTEM_2),
    tool("t_time_berlin", "Tell me the current local time in Berlin.", "get_time", {"timezone": "Berlin"}, SYSTEM_3),
    tool("t_time_singapore", "What is the time in Singapore at this moment?", "get_time", {"timezone": "Singapore"}, SYSTEM_1),
    tool("t_time_buenos", "Give me the current time in Buenos Aires.", "get_time", {"timezone": "Buenos Aires"}, SYSTEM_2),
]


def continuation_ids(tokenizer, text: str, shared_prefix_id: int | None) -> list[int]:
    ids = list(tokenizer.encode(text))
    if shared_prefix_id is not None and ids and int(ids[0]) == int(shared_prefix_id):
        ids = ids[1:]
    return [int(x) for x in ids]


def lcp_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def name_loss(model, tokenizer, case: dict, shared_prefix_id: int | None):
    target = case["target_completion"]
    quoted = json.dumps(case["expected_tool"])
    pos = target.find(quoted)
    if pos < 0:
        raise RuntimeError(f"tool name missing from target for {case['id']}")
    prefix_text = target[:pos]
    through_name_text = target[: pos + len(quoted)]
    prefix_ids = continuation_ids(tokenizer, prefix_text, shared_prefix_id)
    through_ids = continuation_ids(tokenizer, through_name_text, shared_prefix_id)
    split = lcp_len(prefix_ids, through_ids)
    targets = through_ids[split:]
    if not targets:
        raise RuntimeError(f"empty tool-name target span for {case['id']}")
    context = list(tokenizer.encode(case["prompt"])) + through_ids[:split]
    sequence = context + targets
    if len(sequence) > int(model.cfg.block_size):
        raise RuntimeError(f"tool-name training sequence too long for {case['id']}")
    x = torch.tensor([sequence[:-1]], dtype=torch.long)
    logits, _ = model(x, None)
    start = len(context) - 1
    selected = logits[0, start : start + len(targets)]
    y = torch.tensor(targets, dtype=torch.long)
    return F.cross_entropy(selected, y), len(targets)


def routing_contract(tokenizer):
    contract = ev.special_token_contract(tokenizer)
    if not contract["atomic"] or not contract["unique"]:
        raise RuntimeError("special-token contract must remain atomic and unique")
    tool_id = ft.marker_id(contract, "<|tool|>")
    special_ids = {ft.marker_id(contract, marker) for marker in ev.SPECIAL_TOKENS}
    return contract, tool_id, special_ids


def route_margin_loss(logits, tool_id: int, special_ids: set[int], should_tool: bool):
    lexical = torch.stack([logits[i] for i in range(int(logits.shape[0])) if i not in special_ids]).max().detach()
    tool_logit = logits[tool_id]
    if should_tool:
        violation = lexical + TOOL_MARGIN - tool_logit
    else:
        violation = tool_logit + DIRECT_MARGIN - lexical
    return F.relu(violation)


def update_once(model, tokenizer, optimizer, contract, tool_id, special_ids):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    route_losses = []
    tool_cases = [c for c in TRAIN if c["kind"] == "tool_call"]
    for c in TRAIN:
        logits = base.next_logits(model, tokenizer, c["prompt"])
        loss = route_margin_loss(logits, tool_id, special_ids, c["kind"] == "tool_call")
        (loss / len(TRAIN)).backward()
        route_losses.append(float(loss.detach()))

    name_losses = []
    name_tokens = 0
    for c in tool_cases:
        loss, count = name_loss(model, tokenizer, c, contract.get("shared_prefix_id"))
        (NAME_WEIGHT * loss / len(tool_cases)).backward()
        name_losses.append(float(loss.detach()))
        name_tokens += int(count)

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    return {
        "route_hinge_mean": sum(route_losses) / len(route_losses),
        "tool_name_ce_mean": sum(name_losses) / len(name_losses),
        "tool_name_target_tokens": name_tokens,
        "gradient_norm": float(grad_norm),
    }


def routing_counts(model, tokenizer, cases):
    direct_total = tool_total = direct_ok = tool_ok = 0
    rows = []
    for c in cases:
        first = ft.inspect_case(model, tokenizer, c)
        if c["kind"] == "direct_response":
            direct_total += 1
            ok = not bool(first["argmax_is_tool"])
            direct_ok += int(ok)
        else:
            tool_total += 1
            ok = bool(first["argmax_is_tool"])
            tool_ok += int(ok)
        rows.append({
            "id": c["id"],
            "kind": c["kind"],
            "expected_tool": c.get("expected_tool"),
            "ok": ok,
            "argmax": first["argmax_token"],
            "tool_probability": float(first["tool_probability"]),
            "tool_rank": int(first["tool_rank"]),
            "tool_margin": float(first["tool_minus_best_non_special_logit"]),
        })
    return {
        "direct_ok": direct_ok,
        "direct_total": direct_total,
        "tool_ok": tool_ok,
        "tool_total": tool_total,
        "rows": rows,
    }


def summary_markdown(report):
    lines = [
        "# Ember v0.0.54 bounded routing repair", "",
        f"Status: {report['status']}",
        "Source: exact saved v0.0.53 step-9 candidate.",
        "Training prompts are distinct from the unchanged v0.0.53 held-out challenge.",
        f"LR ladder: {', '.join(f'{x:.1e}' for x in ANCHOR_LRS)}; max {MAX_STEPS} steps/rung.",
        f"Margins: direct={DIRECT_MARGIN:.2f}, tool={TOOL_MARGIN:.2f}; tool-name weight={NAME_WEIGHT:.2f}.",
        "",
        "| LR | Step | Train D/T | Held-out D/T first-token | Held-out D/T gen | Tool name | Copy | Structure | Fixed D/T | Pass |",
        "| ---: | ---: | --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for rung in report.get("rungs", []):
        for p in rung.get("candidate_probes", []):
            h = p["heldout"]["metrics"]
            f = p["fixed_generation"]
            s = p["structure"]
            lines.append(
                f"| {rung['lr']:.1e} | {p['step']} | "
                f"{p['train_routing']['direct_ok']}/{p['train_routing']['direct_total']} / {p['train_routing']['tool_ok']}/{p['train_routing']['tool_total']} | "
                f"{h['direct_first_token_pass']}/{h['direct_total']} / {h['tool_first_token_pass']}/{h['tool_total']} | "
                f"{h['direct_generation_pass']}/{h['direct_total']} / {h['tool_generation_pass']}/{h['tool_total']} | "
                f"{h['tool_name_correct']}/{h['tool_total']} | {p['placement']['token_top1']}/35 | "
                f"{s['envelope_json_valid']}/8 / {s['tool_name_correct']}/8 | "
                f"{f['direct_pass']}/4 / {f['tool_pass']}/4 | {p['accepted']} |"
            )
    lines += ["",
        f"Operating point found: {report.get('operating_point_found', False)}",
        f"Selected LR: {report.get('selected_lr')}",
        f"Selected step: {report.get('selected_step')}",
        f"Interpretation: {report.get('interpretation', '')}", "",
        "No checkpoint was saved, exported, promoted, or integrated.",
    ]
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

    report = {
        "schema_version": 1,
        "diagnostic": "ember-v054-bounded-routing-repair-v1",
        "status": "ERROR",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model": MODEL_NAME,
        "anchor_lrs": list(ANCHOR_LRS),
        "max_steps": MAX_STEPS,
        "direct_margin": DIRECT_MARGIN,
        "tool_margin": TOOL_MARGIN,
        "tool_name_weight": NAME_WEIGHT,
        "training_case_count": len(TRAIN),
        "training_direct": sum(c["kind"] == "direct_response" for c in TRAIN),
        "training_tools": sum(c["kind"] == "tool_call" for c in TRAIN),
        "heldout_case_count": len(held.CASES),
        "cpu_only": True,
        "checkpoint_save_authorized": False,
        "promotion_authorized": False,
        "rungs": [],
    }

    with tempfile.TemporaryDirectory(prefix="ember-v054-routing-") as td:
        work = Path(td)
        archive = ev.download_verified(ev.PACKAGE_URL, work / "ember.zip", ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        import sys
        sys.path.insert(0, str(work / "src" / "ember"))

        api = HfApi(token=token)
        owner = api.whoami()["name"]
        repo = f"{owner}/{MODEL_NAME}"
        model, tokenizer, checkpoint = held.load_full(repo, work, token)
        model.eval()
        pristine = copy.deepcopy(model.state_dict())
        pristine_hash = trust.trace.state_digest(model)
        contract, tool_id, special_ids = routing_contract(tokenizer)

        # Preserve the placement/structure benchmark from the v0.0.53 line.
        cfg, template, template_report, selected, selected_ids, source_ref, *_ = base.build_placement_fixture(work)
        fixed_spec = json.loads(Path("config/ember_v0.0.8_eval.json").read_text())
        fixed_cases = [c for c in fixed_spec["cases"] if c["kind"] in {"direct_response", "tool_call"}]

        baseline_train = routing_counts(model, tokenizer, TRAIN)
        baseline_heldout = routing_counts(model, tokenizer, held.CASES)
        baseline_place = base.placement_probe(model, tokenizer, selected, template)
        baseline_structure = base.structure_probe(model, tokenizer, cfg, selected)
        report["baseline"] = {
            "train_routing": baseline_train,
            "heldout_routing": baseline_heldout,
            "placement": baseline_place,
            "structure": baseline_structure,
            "state_sha256": pristine_hash,
        }
        if int(baseline_place["token_top1"]) != COPY_TOKEN_FLOOR or int(baseline_place["tokens"]) != 35:
            raise RuntimeError("v0.0.53 copy floor drifted")
        if int(baseline_structure["envelope_json_valid"]) < STRUCTURE_FLOOR or int(baseline_structure["tool_name_correct"]) < STRUCTURE_FLOOR:
            raise RuntimeError("v0.0.53 structure floor drifted")

        found = False
        best_snapshot = None
        best_score = (-1, -1, -1)
        for lr in ANCHOR_LRS:
            model.load_state_dict(pristine)
            model.eval()
            if trust.trace.state_digest(model) != pristine_hash:
                raise RuntimeError("rung reset failed")
            opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.0)
            rung = {"lr": float(lr), "steps": [], "candidate_probes": [], "accepted": False}
            for step in range(1, MAX_STEPS + 1):
                update = update_once(model, tokenizer, opt, contract, tool_id, special_ids)
                train_route = routing_counts(model, tokenizer, TRAIN)
                held_route = routing_counts(model, tokenizer, held.CASES)
                simple = {
                    "step": step,
                    "update": update,
                    "train_direct": train_route["direct_ok"],
                    "train_tool": train_route["tool_ok"],
                    "heldout_direct": held_route["direct_ok"],
                    "heldout_tool": held_route["tool_ok"],
                }
                rung["steps"].append(simple)
                print(json.dumps({"event": "v054_route_probe", "lr": lr, **simple}), flush=True)

                score = (held_route["direct_ok"], held_route["tool_ok"], train_route["direct_ok"])
                if score > best_score:
                    best_score = score
                    best_snapshot = {
                        "lr": float(lr),
                        "step": step,
                        "state": copy.deepcopy(model.state_dict()),
                        "routing": {"train": train_route, "heldout": held_route},
                    }

                # Expensive gates only when first-token behavior is already substantially repaired.
                if held_route["direct_ok"] >= 10 and held_route["tool_ok"] == 8 and train_route["direct_ok"] >= 15 and train_route["tool_ok"] == 16:
                    held_eval = held.evaluate(model, tokenizer, f"v054-{lr:.1e}-s{step}")
                    place = base.placement_probe(model, tokenizer, selected, template)
                    structure = base.structure_probe(model, tokenizer, cfg, selected)
                    fixed_generation = base.generation_probe(model, tokenizer, fixed_cases, fixed_spec["generation"])
                    hm = held_eval["metrics"]
                    gates = {
                        "heldout_first_token": hm["direct_first_token_pass"] == hm["direct_total"] and hm["tool_first_token_pass"] == hm["tool_total"],
                        "heldout_generation": hm["direct_generation_pass"] == hm["direct_total"] and hm["tool_generation_pass"] == hm["tool_total"],
                        "heldout_tool_names": hm["tool_name_correct"] == hm["tool_total"],
                        "copy_preserved": int(place["token_top1"]) >= COPY_TOKEN_FLOOR,
                        "structure_preserved": int(structure["envelope_json_valid"]) >= STRUCTURE_FLOOR and int(structure["tool_name_correct"]) >= STRUCTURE_FLOOR,
                        "fixed_generation_preserved": int(fixed_generation["direct_pass"]) == 4 and int(fixed_generation["tool_pass"]) == 4,
                    }
                    accepted = all(gates.values())
                    probe = {
                        "step": step,
                        "train_routing": train_route,
                        "heldout": held_eval,
                        "placement": place,
                        "structure": structure,
                        "fixed_generation": fixed_generation,
                        "gates": gates,
                        "accepted": accepted,
                    }
                    rung["candidate_probes"].append(probe)
                    print(json.dumps({
                        "event": "v054_full_gate", "lr": lr, "step": step,
                        "heldout": hm,
                        "copy_tokens": place["token_top1"],
                        "json_valid": structure["envelope_json_valid"],
                        "tool_correct": structure["tool_name_correct"],
                        "fixed_direct": fixed_generation["direct_pass"],
                        "fixed_tool": fixed_generation["tool_pass"],
                        "gates": gates,
                        "accepted": accepted,
                    }), flush=True)
                    if accepted:
                        rung["accepted"] = True
                        rung["accepted_step"] = step
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

        if not found:
            report["operating_point_found"] = False
            report["selected_lr"] = None
            report["selected_step"] = None
            if best_snapshot is not None:
                model.load_state_dict(best_snapshot["state"])
                model.eval()
                best_held = held.evaluate(model, tokenizer, "v054-best-diagnostic")
                best_place = base.placement_probe(model, tokenizer, selected, template)
                best_structure = base.structure_probe(model, tokenizer, cfg, selected)
                report["best_observed"] = {
                    "lr": best_snapshot["lr"],
                    "step": best_snapshot["step"],
                    "routing": best_snapshot["routing"],
                    "heldout": best_held,
                    "placement": best_place,
                    "structure": best_structure,
                }
            report["interpretation"] = (
                "No strict operating point cleared the unchanged 20-case routing challenge while preserving copy/structure. "
                "Use best_observed to localize the remaining routing/tool-name errors; do not save or promote."
            )
        else:
            report["interpretation"] = (
                "A bounded multi-prompt routing + tool-name repair cleared the unchanged v0.0.53 held-out challenge and preserved prior gates. "
                "It should still face a second fresh confirmation set before checkpoint save/promotion."
            )

        report["elapsed_seconds"] = time.monotonic() - started
        report["status"] = "PASS"
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (OUT / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
        print(json.dumps({
            "event": "v054_complete",
            "operating_point_found": report["operating_point_found"],
            "selected_lr": report["selected_lr"],
            "selected_step": report["selected_step"],
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
