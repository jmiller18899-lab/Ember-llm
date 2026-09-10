"""Trace the first free-running tool-envelope divergence for Ember v0.0.53/v0.0.54.

Compares two model states on the unchanged eight held-out tool cases:
1. the exact saved v0.0.53 step-9 checkpoint;
2. an in-memory deterministic reproduction of the strongest safe blocks-3..5
   repair point (LR 3.2e-6, step 9).

For each case, greedily follows the model from the real prompt and compares the
actual continuation against the shortest canonical family prefix:

    <|tool|>\n{"arguments":{"location":
    <|tool|>\n{"arguments":{"expression":
    <|tool|>\n{"arguments":{"query":
    <|tool|>\n{"arguments":{"timezone":

The trace stops at the FIRST mismatching token and records target/selected rank,
probability, logits, and gap under the actual free-running context. This avoids
teacher-forcing past an earlier mistake.

Read-only with respect to saved checkpoints. CPU-only. No checkpoint save,
export, promotion, or integration.
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

from jobs import ember_hf_eval as ev
from jobs import ember_v052_first_token_logits as ft
from jobs import ember_v053_heldout_routing as held
from jobs import ember_v053_first_token_anchor as anchor
from jobs import ember_v054_routing_repair as v54
from jobs import ember_v054_tool_family_prefix as family
from jobs import ember_v054_block345_family as b345
from jobs import ember_alternating_trust_region as trust

OUT = Path("v054-free-run-divergence")
REPAIR_LR = 3.2e-6
REPAIR_STEPS = 9
CANONICAL = {
    "weather": '<|tool|>\n{"arguments":{"location":',
    "calculator": '<|tool|>\n{"arguments":{"expression":',
    "web_search": '<|tool|>\n{"arguments":{"query":',
    "get_time": '<|tool|>\n{"arguments":{"timezone":',
}


def display(tokenizer, token_id: int, shared_prefix_id: int | None) -> str:
    return ft.display_token(tokenizer, int(token_id), shared_prefix_id)


def continuation_ids(tokenizer, text: str, shared_prefix_id: int | None) -> list[int]:
    ids = list(tokenizer.encode(text))
    if shared_prefix_id is not None and ids and int(ids[0]) == int(shared_prefix_id):
        ids = ids[1:]
    return [int(x) for x in ids]


def token_stats(logits, token_id: int) -> dict:
    logits = logits.float().cpu()
    probs = torch.softmax(logits, dim=-1)
    value = float(logits[int(token_id)])
    rank = 1 + int((logits > logits[int(token_id)]).sum().item())
    return {
        "id": int(token_id),
        "logit": value,
        "probability": float(probs[int(token_id)]),
        "rank": rank,
    }


def trace_case(model, tokenizer, case: dict, shared_prefix_id: int | None) -> dict:
    expected_tool = case["expected_tool"]
    desired_text = CANONICAL[expected_tool]
    desired_ids = continuation_ids(tokenizer, desired_text, shared_prefix_id)
    context = [int(x) for x in tokenizer.encode(case["prompt"])]
    matched = []

    for position, desired_id in enumerate(desired_ids, start=1):
        x = torch.tensor([context], dtype=torch.long)
        with torch.inference_mode():
            logits, _ = model(x, None)
        next_logits = logits[0, -1].float().cpu()
        selected_id = int(torch.argmax(next_logits).item())
        desired = token_stats(next_logits, desired_id)
        selected = token_stats(next_logits, selected_id)
        if selected_id != desired_id:
            return {
                "id": case["id"],
                "expected_tool": expected_tool,
                "desired_prefix": desired_text,
                "matched_tokens": len(matched),
                "first_divergence_position": position,
                "matched_text": tokenizer.decode(matched) if matched else "",
                "desired_token": {
                    **desired,
                    "text": display(tokenizer, desired_id, shared_prefix_id),
                },
                "selected_token": {
                    **selected,
                    "text": display(tokenizer, selected_id, shared_prefix_id),
                },
                "selected_minus_desired_logit": selected["logit"] - desired["logit"],
                "desired_minus_selected_logit": desired["logit"] - selected["logit"],
                "canonical_prefix_fully_matched": False,
            }
        matched.append(selected_id)
        context.append(selected_id)

    return {
        "id": case["id"],
        "expected_tool": expected_tool,
        "desired_prefix": desired_text,
        "matched_tokens": len(matched),
        "first_divergence_position": None,
        "matched_text": tokenizer.decode(matched),
        "desired_token": None,
        "selected_token": None,
        "selected_minus_desired_logit": None,
        "desired_minus_selected_logit": None,
        "canonical_prefix_fully_matched": True,
    }


def first_token_direct_probe(model, tokenizer, cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        first = ft.inspect_case(model, tokenizer, case)
        rows.append({
            "id": case["id"],
            "argmax_token": first["argmax_token"],
            "argmax_is_tool": bool(first["argmax_is_tool"]),
            "tool_probability": float(first["tool_probability"]),
            "tool_rank": int(first["tool_rank"]),
            "tool_margin": float(first["tool_minus_best_non_special_logit"]),
            "best_lexical": first["best_non_special_token"],
        })
    return rows


def summarize_state(rows: list[dict]) -> dict:
    by_tool = {}
    positions = []
    for row in rows:
        bucket = by_tool.setdefault(row["expected_tool"], {
            "cases": 0,
            "full_prefix_matches": 0,
            "divergence_positions": [],
        })
        bucket["cases"] += 1
        bucket["full_prefix_matches"] += int(row["canonical_prefix_fully_matched"])
        if row["first_divergence_position"] is not None:
            bucket["divergence_positions"].append(int(row["first_divergence_position"]))
            positions.append(int(row["first_divergence_position"]))
    for bucket in by_tool.values():
        vals = bucket["divergence_positions"]
        bucket["mean_divergence_position"] = sum(vals) / len(vals) if vals else None
    return {
        "cases": len(rows),
        "full_prefix_matches": sum(int(r["canonical_prefix_fully_matched"]) for r in rows),
        "mean_divergence_position": sum(positions) / len(positions) if positions else None,
        "by_tool": by_tool,
    }


def render_summary(report: dict) -> str:
    lines = [
        "# Ember free-running tool-envelope first-divergence trace",
        "",
        "Source: exact saved v0.0.53 step-9 checkpoint.",
        f"Repair comparison: blocks 3-5, LR {REPAIR_LR:.1e}, step {REPAIR_STEPS}, reproduced in memory only.",
        "The trace uses the model's actual greedy context and stops at the first mismatch; it does not teacher-force past errors.",
        "",
        "| Case | Tool | v53 first mismatch | v53 desired→selected | v54-s9 first mismatch | v54-s9 desired→selected |",
        "| --- | --- | ---: | --- | ---: | --- |",
    ]
    source = {r["id"]: r for r in report["states"]["v053_source"]["traces"]}
    repair = {r["id"]: r for r in report["states"]["block345_step9"]["traces"]}
    for case_id in source:
        a = source[case_id]
        b = repair[case_id]
        def cell(r):
            if r["canonical_prefix_fully_matched"]:
                return "FULL"
            return f"`{r['desired_token']['text']}` → `{r['selected_token']['text']}` (rank {r['desired_token']['rank']}, gap {r['selected_minus_desired_logit']:+.3f})"
        lines.append(
            f"| {case_id} | {a['expected_tool']} | "
            f"{a['first_divergence_position'] if a['first_divergence_position'] is not None else 'FULL'} | {cell(a)} | "
            f"{b['first_divergence_position'] if b['first_divergence_position'] is not None else 'FULL'} | {cell(b)} |"
        )
    lines += [
        "",
        "## Aggregate",
        "",
        f"- v0.0.53 canonical family prefixes fully matched: {report['states']['v053_source']['aggregate']['full_prefix_matches']}/8.",
        f"- blocks-3..5 step-9 prefixes fully matched: {report['states']['block345_step9']['aggregate']['full_prefix_matches']}/8.",
        "",
        "No saved model state was changed. No checkpoint was saved, exported, promoted, or integrated.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN required")
    torch.set_num_threads(2)
    torch.manual_seed(20260910)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="ember-free-divergence-") as td:
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
            raise RuntimeError("expected saved v0.0.53 checkpoint")
        model.eval()
        source_hash = trust.trace.state_digest(model)
        pristine = copy.deepcopy(model.state_dict())
        contract, tool_id, special_ids = v54.routing_contract(tokenizer)
        shared_prefix = contract.get("shared_prefix_id")

        tool_cases = [c for c in held.CASES if c["kind"] == "tool_call"]
        if len(tool_cases) != 8:
            raise RuntimeError(f"expected 8 held-out tool cases, got {len(tool_cases)}")
        direct_cases = [c for c in held.CASES if c["id"] in {"direct_latest_rewrite", "direct_summary"}]

        source_traces = [trace_case(model, tokenizer, c, shared_prefix) for c in tool_cases]
        source_direct = first_token_direct_probe(model, tokenizer, direct_cases)

        # Deterministically reproduce the strongest safe blocks-3..5 point.
        model.load_state_dict(pristine)
        b345.configure_trainable(model)
        model.eval()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=REPAIR_LR,
            weight_decay=0.0,
        )
        updates = []
        for step in range(1, REPAIR_STEPS + 1):
            update = b345.update_once(model, tokenizer, optimizer, contract, tool_id, special_ids)
            updates.append({"step": step, **update})
        del optimizer
        model.eval()

        cfg, template, template_report, selected, selected_ids, *_ = anchor.build_placement_fixture(work)
        place = anchor.placement_probe(model, tokenizer, selected, template)
        structure = anchor.structure_probe(model, tokenizer, cfg, selected)
        held_route = v54.routing_counts(model, tokenizer, held.CASES)
        if int(place["token_top1"]) != 22:
            raise RuntimeError(f"step-9 reproduction copy mismatch: {place['token_top1']}/35")
        if int(held_route["direct_ok"]) != 10 or int(held_route["tool_ok"]) != 8:
            raise RuntimeError(
                f"step-9 reproduction routing mismatch: direct={held_route['direct_ok']}/12 tool={held_route['tool_ok']}/8"
            )
        repaired_hash = trust.trace.state_digest(model)
        if repaired_hash == source_hash:
            raise RuntimeError("repair reproduction did not change model state")

        repaired_traces = [trace_case(model, tokenizer, c, shared_prefix) for c in tool_cases]
        repaired_direct = first_token_direct_probe(model, tokenizer, direct_cases)

        report = {
            "schema_version": 1,
            "diagnostic": "ember-v054-free-running-first-divergence-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_repo": repo,
            "source_checkpoint": held.BEST_PATH,
            "source_checkpoint_sha256": held.BEST_SHA256,
            "source_state_sha256": source_hash,
            "repair": {
                "trainable_groups": sorted(b345.TRAIN_GROUPS),
                "learning_rate": REPAIR_LR,
                "steps": REPAIR_STEPS,
                "updates": updates,
                "state_sha256": repaired_hash,
                "verification": {
                    "heldout_direct_first_token": int(held_route["direct_ok"]),
                    "heldout_tool_first_token": int(held_route["tool_ok"]),
                    "copy_tokens": int(place["token_top1"]),
                    "copy_tokens_total": int(place["tokens"]),
                    "structure_json": int(structure["envelope_json_valid"]),
                    "structure_tool": int(structure["tool_name_correct"]),
                },
            },
            "canonical_prefixes": CANONICAL,
            "states": {
                "v053_source": {
                    "traces": source_traces,
                    "aggregate": summarize_state(source_traces),
                    "remaining_direct_failures": source_direct,
                },
                "block345_step9": {
                    "traces": repaired_traces,
                    "aggregate": summarize_state(repaired_traces),
                    "remaining_direct_failures": repaired_direct,
                },
            },
            "read_only_saved_state": True,
            "checkpoint_save_authorized": False,
            "promotion_authorized": False,
            "elapsed_seconds": time.monotonic() - started,
        }
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (OUT / "summary.md").write_text(render_summary(report), encoding="utf-8")

        print(json.dumps({
            "event": "free_divergence_complete",
            "source_full_prefix_matches": report["states"]["v053_source"]["aggregate"]["full_prefix_matches"],
            "repair_full_prefix_matches": report["states"]["block345_step9"]["aggregate"]["full_prefix_matches"],
            "repair_verification": report["repair"]["verification"],
            "source": [
                {
                    "id": r["id"],
                    "tool": r["expected_tool"],
                    "position": r["first_divergence_position"],
                    "desired": None if r["desired_token"] is None else r["desired_token"]["text"],
                    "selected": None if r["selected_token"] is None else r["selected_token"]["text"],
                    "desired_rank": None if r["desired_token"] is None else r["desired_token"]["rank"],
                    "gap": r["selected_minus_desired_logit"],
                }
                for r in source_traces
            ],
            "repair": [
                {
                    "id": r["id"],
                    "tool": r["expected_tool"],
                    "position": r["first_divergence_position"],
                    "desired": None if r["desired_token"] is None else r["desired_token"]["text"],
                    "selected": None if r["selected_token"] is None else r["selected_token"]["text"],
                    "desired_rank": None if r["desired_token"] is None else r["desired_token"]["rank"],
                    "gap": r["selected_minus_desired_logit"],
                }
                for r in repaired_traces
            ],
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
