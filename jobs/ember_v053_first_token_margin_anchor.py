"""Rank-margin refinement of the Ember v0.0.53 first-token routing anchor.

Starts from the exact saved v0.0.52 candidate. The loss acts on the <|tool|>
first-token logit only: direct cases push it below the strongest lexical token
by a small margin; explicit tool cases preserve it above the strongest lexical
token by a margin. The lexical comparison is detached, so the anchor does not
train a replacement prose token.

CPU-only, bounded, no checkpoint save/export/promotion/integration.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import gc
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
from jobs import ember_v052_first_token_logits as ft
from jobs import ember_v053_first_token_anchor as base
from jobs import ember_alternating_trust_region as trust

OUT = Path("v053-first-token-margin-anchor")
CANDIDATE_NAME = "ember-v0.0.52-t4"
ANCHOR_LRS = (4e-8, 1.6e-7, 6.4e-7)
MAX_STEPS = 12
DIRECT_MARGIN = 0.25
TOOL_MARGIN = 1.0
COPY_TOKEN_FLOOR = 22
STRUCTURE_FLOOR = 7
GRAD_CLIP = 0.25


def routing_contract(tokenizer):
    contract = ev.special_token_contract(tokenizer)
    if not contract["atomic"] or not contract["unique"]:
        raise RuntimeError("special-token contract must be atomic and unique")
    special_ids = {ft.marker_id(contract, marker) for marker in ev.SPECIAL_TOKENS}
    tool_id = ft.marker_id(contract, "<|tool|>")
    return tool_id, special_ids


def margin_loss(logits, tool_id: int, special_ids: set[int], should_tool: bool):
    lexical_ids = [i for i in range(int(logits.shape[0])) if i not in special_ids]
    lexical = logits[lexical_ids].max().detach()
    tool = logits[tool_id]
    if should_tool:
        violation = lexical + TOOL_MARGIN - tool
    else:
        violation = tool + DIRECT_MARGIN - lexical
    return F.relu(violation), float(tool.detach() - lexical)


def anchor_update(model, tokenizer, optimizer, cases, tool_id, special_ids):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    direct_losses, tool_losses, pre_margins = [], [], {}
    for case in cases:
        logits = base.next_logits(model, tokenizer, case["prompt"])
        should_tool = case["kind"] == "tool_call"
        loss, raw_margin = margin_loss(logits, tool_id, special_ids, should_tool)
        scale = 0.5 / (4 if should_tool else 4)
        (loss * scale).backward()
        (tool_losses if should_tool else direct_losses).append(float(loss.detach()))
        pre_margins[case["id"]] = raw_margin
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    return {
        "direct_hinge": sum(direct_losses) / len(direct_losses),
        "tool_hinge": sum(tool_losses) / len(tool_losses),
        "gradient_norm": float(grad_norm),
        "pre_update_tool_minus_lexical": pre_margins,
    }


def compact_route(route):
    direct = {r["id"]: {
        "argmax": r["argmax_token"],
        "tool_probability": r["tool_probability"],
        "tool_rank": r["tool_rank"],
        "tool_margin": r["tool_minus_best_non_special_logit"],
        "best_lexical": r["best_non_special_token"],
    } for r in route["rows"] if r["kind"] == "direct_response"}
    tools = {r["id"]: {
        "argmax": r["argmax_token"],
        "tool_probability": r["tool_probability"],
        "tool_rank": r["tool_rank"],
        "tool_margin": r["tool_minus_best_non_special_logit"],
    } for r in route["rows"] if r["kind"] == "tool_call"}
    return {"direct": direct, "tools": tools}


def summary_markdown(report):
    lines = [
        "# Ember v0.0.53 first-token rank-margin anchor", "",
        f"Status: {report['status']}",
        "Starts from saved v0.0.52 candidate.",
        f"Direct margin: tool logit must be >= {DIRECT_MARGIN:.2f} below best lexical token; tool cases preserve >= {TOOL_MARGIN:.2f} positive margin.",
        f"LR ladder: {', '.join(f'{x:.1e}' for x in ANCHOR_LRS)}; max {MAX_STEPS} steps/rung.",
        "", "| LR | Step | Direct tool argmax | Tool tool-argmax | Copy | JSON | Tool structure | Generation D/T | Pass |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for rung in report.get("rungs", []):
        for p in rung.get("candidate_probes", []):
            lines.append(
                f"| {rung['lr']:.1e} | {p['step']} | {p['routing']['direct_argmax_tool_count']}/4 | "
                f"{p['routing']['tool_argmax_tool_count']}/4 | {p['placement']['token_top1']}/35 | "
                f"{p['structure']['envelope_json_valid']}/8 | {p['structure']['tool_name_correct']}/8 | "
                f"{p['generation']['direct_pass']}/4 / {p['generation']['tool_pass']}/4 | {p['accepted']} |"
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
    torch.manual_seed(1337)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    report = {
        "schema_version": 1,
        "diagnostic": "ember-v053-first-token-rank-margin-anchor-v1",
        "status": "ERROR",
        "anchor_lrs": list(ANCHOR_LRS),
        "max_steps": MAX_STEPS,
        "direct_margin": DIRECT_MARGIN,
        "tool_margin": TOOL_MARGIN,
        "copy_floor": COPY_TOKEN_FLOOR,
        "structure_floor": STRUCTURE_FLOOR,
        "cpu_only": True,
        "checkpoint_save_authorized": False,
        "promotion_authorized": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rungs": [],
    }

    with tempfile.TemporaryDirectory(prefix="ember-v053-margin-") as td:
        work = Path(td)
        archive = ev.download_verified(ev.PACKAGE_URL, work / "ember.zip", ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        import sys
        sys.path.insert(0, str(work / "src" / "ember"))

        spec = json.loads(Path("config/ember_v0.0.8_eval.json").read_text())
        cases = [c for c in spec["cases"] if c["kind"] in {"direct_response", "tool_call"}]
        api = HfApi(token=token)
        owner = api.whoami()["name"]
        repo = f"{owner}/{CANDIDATE_NAME}"
        checkpoint_path = ft.candidate_checkpoint_path(api, repo)
        model, tokenizer, checkpoint = ft.load_model(repo, checkpoint_path, work, work / "src" / "ember")
        if checkpoint.get("train_config", {}).get("version") != "0.0.52":
            raise RuntimeError("must start from saved v0.0.52 candidate")
        model.eval()
        pristine = copy.deepcopy(model.state_dict())
        pristine_hash = trust.trace.state_digest(model)
        tool_id, special_ids = routing_contract(tokenizer)

        cfg, template, template_report, selected, selected_ids, source_ref, *_ = base.build_placement_fixture(work)
        baseline_route = base.routing_probe(model, tokenizer, cases)
        baseline_place = base.placement_probe(model, tokenizer, selected, template)
        baseline_structure = base.structure_probe(model, tokenizer, cfg, selected)
        if baseline_route["direct_argmax_tool_count"] != 2 or baseline_route["tool_argmax_tool_count"] != 4:
            raise RuntimeError("v52 routing baseline drifted")
        if int(baseline_place["token_top1"]) != 22 or int(baseline_place["tokens"]) != 35:
            raise RuntimeError("v52 copy baseline drifted")
        if int(baseline_structure["envelope_json_valid"]) < 7 or int(baseline_structure["tool_name_correct"]) < 7:
            raise RuntimeError("v52 structural floor drifted")
        report["baseline"] = {
            "routing": baseline_route,
            "routing_compact": compact_route(baseline_route),
            "placement": baseline_place,
            "structure": baseline_structure,
        }
        report["candidate_repo"] = repo
        report["candidate_checkpoint"] = checkpoint_path
        report["candidate_state_sha256"] = pristine_hash
        report["selected_case_ids"] = selected_ids

        found = False
        for lr in ANCHOR_LRS:
            model.load_state_dict(pristine)
            model.eval()
            if trust.trace.state_digest(model) != pristine_hash:
                raise RuntimeError("rung reset failed")
            opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.0)
            rung = {"lr": float(lr), "steps": [], "candidate_probes": [], "accepted": False}
            for step in range(1, MAX_STEPS + 1):
                update = anchor_update(model, tokenizer, opt, cases, tool_id, special_ids)
                route = base.routing_probe(model, tokenizer, cases)
                record = {
                    "step": step,
                    "update": update,
                    "direct_argmax_tool_count": route["direct_argmax_tool_count"],
                    "tool_argmax_tool_count": route["tool_argmax_tool_count"],
                    "routing": compact_route(route),
                }
                rung["steps"].append(record)
                print(json.dumps({"event":"margin_anchor_probe", "lr":lr, **record}), flush=True)

                if route["direct_argmax_tool_count"] == 0 and route["tool_argmax_tool_count"] == 4:
                    place = base.placement_probe(model, tokenizer, selected, template)
                    structure = base.structure_probe(model, tokenizer, cfg, selected)
                    generation = base.generation_probe(model, tokenizer, cases, spec["generation"])
                    gates = base.gate_record(route, generation, place, structure)
                    accepted = all(gates.values())
                    candidate_probe = {
                        "step": step,
                        "routing": route,
                        "routing_compact": compact_route(route),
                        "placement": place,
                        "structure": structure,
                        "generation": generation,
                        "gates": gates,
                        "accepted": accepted,
                    }
                    rung["candidate_probes"].append(candidate_probe)
                    print(json.dumps({
                        "event":"margin_anchor_gate", "lr":lr, "step":step,
                        "copy_tokens":place["token_top1"],
                        "json_valid":structure["envelope_json_valid"],
                        "tool_correct":structure["tool_name_correct"],
                        "direct_generation_pass":generation["direct_pass"],
                        "tool_generation_pass":generation["tool_pass"],
                        "gates":gates, "accepted":accepted,
                    }), flush=True)
                    if accepted:
                        rung["accepted"] = True
                        rung["accepted_step"] = step
                        report["operating_point_found"] = True
                        report["selected_lr"] = float(lr)
                        report["selected_step"] = step
                        report["selected_probe"] = candidate_probe
                        found = True
                        break
            report["rungs"].append(rung)
            del opt
            if found:
                break

        if found:
            report["interpretation"] = (
                "The rank-margin first-token anchor found a simultaneous routing/copy/structure operating point on top of v0.0.52."
            )
        else:
            report["operating_point_found"] = False
            report["selected_lr"] = None
            report["selected_step"] = None
            report["interpretation"] = (
                "No tested rank-margin anchor point simultaneously cleared first-token routing, full generation, copy preservation, and structure gates."
            )
        report["elapsed_seconds"] = time.monotonic() - started
        report["status"] = "PASS"
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (OUT / "summary.md").write_text(summary_markdown(report))
        print(json.dumps({"event":"margin_anchor_complete", "operating_point_found":report["operating_point_found"], "selected_lr":report["selected_lr"], "selected_step":report["selected_step"], "elapsed_seconds":report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
