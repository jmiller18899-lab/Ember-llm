"""Bounded first-token routing-anchor diagnostic on saved Ember v0.0.52.

Starts from the saved v0.0.52 trust-region candidate. Direct-response prompts
penalize only the <|tool|> continuation at token 1; explicit tool prompts
reinforce only the <|tool|> continuation at token 1. No replacement prose is
taught. Each LR rung resets to the exact saved v0.0.52 weights.

An operating point requires:
- 0/4 direct prompts with <|tool|> as first-token argmax,
- 4/4 explicit tool prompts with <|tool|> as first-token argmax,
- all four direct generations remain tool-free and all four tool generations
  still satisfy the existing evaluator,
- placement token top-1 remains >=22/35,
- selected free-running envelope structure remains >=7/8 JSON-valid and
  correct-tool.

CPU-only diagnostic. No checkpoint save, export, promotion, or integration.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import gc
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
from jobs import ember_alternating_trust_region as trust

OUT = Path("v053-first-token-anchor")
CANDIDATE_NAME = "ember-v0.0.52-t4"
ANCHOR_LRS = (4e-8, 1.6e-7, 6.4e-7)
MAX_STEPS = 8
DIRECT_CASES = 4
TOOL_CASES = 4
COPY_TOKEN_FLOOR = 22
STRUCTURE_FLOOR = 7
GRAD_CLIP = 0.25
WALL_SECONDS = 1200


def tool_id_for(tokenizer) -> int:
    contract = ev.special_token_contract(tokenizer)
    if not contract["atomic"] or not contract["unique"]:
        raise RuntimeError("special-token contract must be atomic and unique")
    return ft.marker_id(contract, "<|tool|>")


def next_logits(model, tokenizer, prompt: str):
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    logits, _ = model(x, None)
    return logits[0, -1]


def binary_tool_loss(logits, tool_id: int, should_tool: bool):
    tool = logits[tool_id]
    other = torch.logsumexp(torch.cat((logits[:tool_id], logits[tool_id + 1 :])), dim=0)
    # Stable -log p(tool) for tool cases or -log(1-p(tool)) for direct cases.
    return F.softplus(other - tool) if should_tool else F.softplus(tool - other)


def anchor_update(model, tokenizer, optimizer, cases: list[dict], tool_id: int):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    direct_losses = []
    tool_losses = []
    for case in cases:
        logits = next_logits(model, tokenizer, case["prompt"])
        should_tool = case["kind"] == "tool_call"
        loss = binary_tool_loss(logits, tool_id, should_tool)
        # Equal total weight for direct and tool groups, independent of per-case count.
        scale = 0.5 / (TOOL_CASES if should_tool else DIRECT_CASES)
        (loss * scale).backward()
        (tool_losses if should_tool else direct_losses).append(float(loss.detach()))
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    return {
        "direct_loss": sum(direct_losses) / len(direct_losses),
        "tool_loss": sum(tool_losses) / len(tool_losses),
        "gradient_norm": float(grad_norm),
    }


def routing_probe(model, tokenizer, cases: list[dict]) -> dict:
    rows = []
    direct_bad = 0
    tool_good = 0
    with torch.inference_mode():
        for case in cases:
            row = ft.inspect_case(model, tokenizer, case)
            rows.append({"id": case["id"], "kind": case["kind"], **row})
            if case["kind"] == "direct_response":
                direct_bad += int(row["argmax_is_tool"])
            else:
                tool_good += int(row["argmax_is_tool"])
    return {
        "direct_argmax_tool_count": direct_bad,
        "tool_argmax_tool_count": tool_good,
        "rows": rows,
    }


def generation_probe(model, tokenizer, cases: list[dict], generation: dict) -> dict:
    rows = []
    direct_pass = 0
    tool_pass = 0
    torch.manual_seed(int(generation["seed"]))
    for case in cases:
        completion = ev.generate_completion(model, tokenizer, torch, case["prompt"], generation)
        score = ev.score_case(case, completion)
        rows.append({"id": case["id"], "kind": case["kind"], "completion": completion, "score": score})
        if case["kind"] == "direct_response":
            direct_pass += int(score["passed"])
        else:
            tool_pass += int(score["passed"])
    return {"direct_pass": direct_pass, "tool_pass": tool_pass, "rows": rows}


def build_placement_fixture(work: Path):
    """Reconstruct the exact eight v0.0.52 placement cases from pinned v31."""
    cfg = trust.trace.load_config()
    source_model, source_tok, source, _splits, source_ref = trust.base.load_inputs(
        json.loads(trust.base.DEFAULT_CONFIG.read_text()), work / "placement-source", torch
    )
    if source.get("step") != 479 or source.get("train_config", {}).get("version") != "0.0.31":
        raise RuntimeError("placement fixture source is not pinned v0.0.31 step 479")
    source_model.to("cpu").eval()
    values = trust.data.target_values(cfg)
    template, template_report = trust.data.v048d.discover_template(
        source_model, source_tok, torch, cfg, values["template"]
    )
    development = trust.data.v048d.build_cases(
        {k: values["development"][k] for k in sorted(trust.data.PLACEMENT_SUBTYPES)},
        "v051_place_dev",
    )
    full_baseline, chosen_rows, selected, selected_baseline, baseline_wrong = trust.ladder.choose_cases(
        source_model, source_tok, torch, cfg, template, development
    )
    selected_ids = [c["id"] for c in selected]
    expected = [
        "v051_place_dev_short_code_len5_03",
        "v051_place_dev_short_code_len5_00",
        "v051_place_dev_short_code_len5_02",
        "v051_place_dev_short_code_len4_03",
        "v051_place_dev_short_code_len5_01",
        "v051_place_dev_long_code_3x5_05",
        "v051_place_dev_short_code_len4_01",
        "v051_place_dev_long_code_3x5_02",
    ]
    if selected_ids != expected:
        raise RuntimeError(f"placement selection drifted: {selected_ids}")
    if ev.special_token_contract(source_tok)["signatures"] is None:
        raise RuntimeError("source tokenizer contract unavailable")
    del source_model, source, _splits
    gc.collect()
    return cfg, template, template_report, selected, selected_ids, source_ref, full_baseline, chosen_rows, selected_baseline, baseline_wrong


def placement_probe(model, tokenizer, selected, template):
    with trust.trace.observation(model, torch):
        return trust.objectives.placement_probe(model, tokenizer, torch, selected, template)


def structure_probe(model, tokenizer, cfg, selected):
    return trust.tiny.structure_probe(model, tokenizer, torch, cfg, selected)


def gate_record(route, generation, place, structure) -> dict:
    return {
        "routing_first_token_pass": route["direct_argmax_tool_count"] == 0 and route["tool_argmax_tool_count"] == 4,
        "generation_routing_pass": generation["direct_pass"] == 4 and generation["tool_pass"] == 4,
        "copy_preserved": int(place["token_top1"]) >= COPY_TOKEN_FLOOR,
        "structure_preserved": int(structure["envelope_json_valid"]) >= STRUCTURE_FLOOR
        and int(structure["tool_name_correct"]) >= STRUCTURE_FLOOR,
    }


def summary_markdown(report: dict) -> str:
    lines = [
        "# Ember v0.0.53 first-token routing-anchor diagnostic",
        "",
        f"Status: {report['status']}",
        "Starting checkpoint: saved Ember v0.0.52 trust-region candidate",
        "Anchor: direct prompts penalize only <|tool|> at token 1; tool prompts reinforce only <|tool|> at token 1.",
        f"LR ladder: {', '.join(f'{lr:.1e}' for lr in ANCHOR_LRS)}; max {MAX_STEPS} steps per rung.",
        f"Gates: 0/4 direct first-token tool entries, 4/4 tool first-token entries, 4/4 direct + 4/4 tool generation routing, >= {COPY_TOKEN_FLOOR}/35 copy tokens, >= {STRUCTURE_FLOOR}/8 structure.",
        "",
        "| LR | Step | Direct tool argmax | Tool tool-argmax | Copy tokens | JSON | Tool structure | Full generation D/T | Pass |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for rung in report.get("rungs", []):
        if not rung.get("candidate_probes"):
            lines.append(f"| {rung['lr']:.1e} | — | — | — | — | — | — | — | False |")
            continue
        for p in rung["candidate_probes"]:
            lines.append(
                f"| {rung['lr']:.1e} | {p['step']} | {p['routing']['direct_argmax_tool_count']}/4 | "
                f"{p['routing']['tool_argmax_tool_count']}/4 | {p['placement']['token_top1']}/35 | "
                f"{p['structure']['envelope_json_valid']}/8 | {p['structure']['tool_name_correct']}/8 | "
                f"{p['generation']['direct_pass']}/4 / {p['generation']['tool_pass']}/4 | {p['accepted']} |"
            )
    lines += [
        "",
        f"Operating point found: {report.get('operating_point_found', False)}",
        f"Selected LR: {report.get('selected_lr')}",
        f"Selected step: {report.get('selected_step')}",
        f"Interpretation: {report.get('interpretation', '')}",
        "",
        "No checkpoint was saved, exported, promoted, or integrated.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
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
        "diagnostic": "ember-v053-first-token-routing-anchor-v1",
        "status": "ERROR",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "anchor_lrs": list(ANCHOR_LRS),
        "max_steps": MAX_STEPS,
        "copy_token_floor": COPY_TOKEN_FLOOR,
        "structure_floor": STRUCTURE_FLOOR,
        "cpu_only": True,
        "checkpoint_save_authorized": False,
        "promotion_authorized": False,
        "rungs": [],
    }

    with tempfile.TemporaryDirectory(prefix="ember-v053-anchor-") as td:
        work = Path(td)
        # Install the verified Ember source package used by the existing evaluator.
        archive = ev.download_verified(ev.PACKAGE_URL, work / "ember.zip", ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        import sys
        sys.path.insert(0, str(work / "src" / "ember"))

        spec = json.loads(Path("config/ember_v0.0.8_eval.json").read_text())
        routing_cases = [c for c in spec["cases"] if c["kind"] in {"direct_response", "tool_call"}]
        if len(routing_cases) != 8:
            raise RuntimeError(f"expected 8 routing cases, got {len(routing_cases)}")

        api = HfApi(token=token)
        owner = api.whoami()["name"]
        candidate_repo = f"{owner}/{CANDIDATE_NAME}"
        candidate_path = ft.candidate_checkpoint_path(api, candidate_repo)
        model, tokenizer, checkpoint = ft.load_model(candidate_repo, candidate_path, work, work / "src" / "ember")
        if checkpoint.get("train_config", {}).get("version") != "0.0.52":
            raise RuntimeError("anchor must start from saved v0.0.52 candidate")
        if float(model.cfg.dropout) != 0:
            raise RuntimeError("zero dropout required")
        model.eval()
        pristine = copy.deepcopy(model.state_dict())
        pristine_hash = trust.trace.state_digest(model)
        tool_id = tool_id_for(tokenizer)

        cfg, template, template_report, selected, selected_ids, source_ref, full_baseline, chosen_rows, selected_baseline, baseline_wrong = build_placement_fixture(work)
        baseline_place = placement_probe(model, tokenizer, selected, template)
        baseline_structure = structure_probe(model, tokenizer, cfg, selected)
        baseline_route = routing_probe(model, tokenizer, routing_cases)
        if int(baseline_place["token_top1"]) != 22 or int(baseline_place["tokens"]) != 35:
            raise RuntimeError(f"saved v52 copy baseline drifted: {baseline_place['token_top1']}/{baseline_place['tokens']}")
        if int(baseline_structure["envelope_json_valid"]) < 7 or int(baseline_structure["tool_name_correct"]) < 7:
            raise RuntimeError("saved v52 structure baseline drifted below 7/8")
        if baseline_route["direct_argmax_tool_count"] != 2 or baseline_route["tool_argmax_tool_count"] != 4:
            raise RuntimeError(f"saved v52 routing baseline drifted: {baseline_route}")

        report.update({
            "candidate_repo": candidate_repo,
            "candidate_checkpoint": candidate_path,
            "candidate_state_sha256": pristine_hash,
            "placement_source": source_ref,
            "selected_case_ids": selected_ids,
            "template_report": template_report,
            "baseline": {
                "routing": baseline_route,
                "placement": baseline_place,
                "structure": baseline_structure,
            },
        })

        found = False
        for lr in ANCHOR_LRS:
            model.load_state_dict(pristine)
            model.eval()
            if trust.trace.state_digest(model) != pristine_hash:
                raise RuntimeError("v52 rung reset failed")
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.0)
            rung = {"lr": float(lr), "steps": [], "candidate_probes": [], "accepted": False}

            for step in range(1, MAX_STEPS + 1):
                update = anchor_update(model, tokenizer, optimizer, routing_cases, tool_id)
                route = routing_probe(model, tokenizer, routing_cases)
                compact = {
                    "step": step,
                    "update": update,
                    "direct_argmax_tool_count": route["direct_argmax_tool_count"],
                    "tool_argmax_tool_count": route["tool_argmax_tool_count"],
                    "direct_tool_probabilities": {
                        r["id"]: r["tool_probability"] for r in route["rows"] if r["kind"] == "direct_response"
                    },
                    "tool_tool_probabilities": {
                        r["id"]: r["tool_probability"] for r in route["rows"] if r["kind"] == "tool_call"
                    },
                }
                rung["steps"].append(compact)
                print(json.dumps({"event": "anchor_probe", "lr": lr, **compact}), flush=True)

                # Only pay for copy/structure/free-generation probes when first-token routing is clean.
                if route["direct_argmax_tool_count"] == 0 and route["tool_argmax_tool_count"] == 4:
                    place = placement_probe(model, tokenizer, selected, template)
                    structure = structure_probe(model, tokenizer, cfg, selected)
                    generation = generation_probe(model, tokenizer, routing_cases, spec["generation"])
                    gates = gate_record(route, generation, place, structure)
                    accepted = all(gates.values())
                    candidate_probe = {
                        "step": step,
                        "routing": route,
                        "placement": place,
                        "structure": structure,
                        "generation": generation,
                        "gates": gates,
                        "accepted": accepted,
                    }
                    rung["candidate_probes"].append(candidate_probe)
                    print(json.dumps({
                        "event": "anchor_candidate_gate",
                        "lr": lr,
                        "step": step,
                        "direct_first_token_tool": route["direct_argmax_tool_count"],
                        "tool_first_token_tool": route["tool_argmax_tool_count"],
                        "copy_tokens": place["token_top1"],
                        "json_valid": structure["envelope_json_valid"],
                        "tool_correct": structure["tool_name_correct"],
                        "direct_generation_pass": generation["direct_pass"],
                        "tool_generation_pass": generation["tool_pass"],
                        "gates": gates,
                        "accepted": accepted,
                    }), flush=True)
                    if accepted:
                        rung["accepted"] = True
                        rung["accepted_step"] = step
                        report["operating_point_found"] = True
                        report["selected_lr"] = float(lr)
                        report["selected_step"] = int(step)
                        report["selected_probe"] = candidate_probe
                        found = True
                        break
            report["rungs"].append(rung)
            del optimizer
            if found:
                break

        if found:
            p = report["selected_probe"]
            report["interpretation"] = (
                "A bounded first-token routing anchor found a simultaneous point that removes accidental direct-mode tool entry, "
                "keeps explicit tool routing intact, and preserves the saved v0.0.52 copy/structure gains."
            )
        else:
            report["operating_point_found"] = False
            report["selected_lr"] = None
            report["selected_step"] = None
            report["interpretation"] = (
                "The tested first-token anchor ladder did not satisfy routing, full-generation, copy-preservation, and structure gates simultaneously."
            )

        report["elapsed_seconds"] = time.monotonic() - started
        report["status"] = "PASS"
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (OUT / "summary.md").write_text(summary_markdown(report))
        print(json.dumps({
            "event": "anchor_complete",
            "operating_point_found": report["operating_point_found"],
            "selected_lr": report["selected_lr"],
            "selected_step": report["selected_step"],
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
