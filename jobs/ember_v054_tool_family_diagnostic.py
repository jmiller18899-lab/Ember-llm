"""Tool-family classifier diagnostic for saved Ember v0.0.53 step 9.

Read-only, CPU-only diagnostic. It isolates the choice among weather, calculator,
web_search, and get_time after tool mode has already been selected.

For each true tool request it scores all four tool-name strings under three
contexts:
  1) name_first: counterfactual JSON where the name is chosen immediately after
     <|tool|>, before arguments;
  2) args_first_correct: the existing arguments-first order, conditioned on the
     correct canonical arguments;
  3) args_first_generated: the existing order, conditioned on the model's own
     greedy prefix up to the tool-name value when such a prefix is available.

It also records logits at the first token position where the four tool names
become distinguishable. Development tool prompts are distinct from the eight
held-out challenge prompts; both full and INT4 checkpoints are evaluated.

No weights are changed, saved, promoted, or integrated.
"""
from __future__ import annotations

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
from jobs import ember_v054_routing_repair as repair

OUT = Path("v054-tool-family-diagnostic")
FAMILIES = ("weather", "calculator", "web_search", "get_time")

HELDOUT_ARGS = {
    "tool_weather_umbrella": {"location": "Portland"},
    "tool_weather_temperature": {"location": "Miami"},
    "tool_calculator_divide": {"expression": "9187/43"},
    "tool_calculator_percent": {"expression": "0.175*864"},
    "tool_search_node": {"query": "newest stable Node.js release"},
    "tool_search_python": {"query": "most recent official Python security release"},
    "tool_time_lisbon": {"timezone": "Lisbon"},
    "tool_time_seoul": {"timezone": "Seoul"},
}

GENERATION = {
    "max_new_tokens": 48,
    "temperature": 1.0,
    "top_k": 1,
    "seed": 20260910,
}


def continuation_ids(tokenizer, text: str, shared_prefix_id: int | None) -> list[int]:
    ids = [int(x) for x in tokenizer.encode(text)]
    if shared_prefix_id is not None and ids and ids[0] == int(shared_prefix_id):
        ids = ids[1:]
    return ids


def common_prefix(seqs: list[list[int]]) -> list[int]:
    if not seqs:
        return []
    out = []
    for items in zip(*seqs):
        if len(set(items)) != 1:
            break
        out.append(int(items[0]))
    return out


def score_continuation(model, tokenizer, prompt: str, prefix: str, continuation: str, shared_prefix_id: int | None) -> dict:
    context = [int(x) for x in tokenizer.encode(prompt + prefix)]
    targets = continuation_ids(tokenizer, continuation, shared_prefix_id)
    if not targets:
        raise RuntimeError("empty candidate continuation")
    sequence = context + targets
    if len(sequence) > int(model.cfg.block_size):
        raise RuntimeError("tool-family scoring sequence exceeds block size")
    x = torch.tensor([sequence[:-1]], dtype=torch.long)
    with torch.inference_mode():
        logits, _ = model(x, None)
        start = len(context) - 1
        selected = logits[0, start : start + len(targets)]
        logp = F.log_softmax(selected, dim=-1)
        y = torch.tensor(targets, dtype=torch.long)
        token_logps = logp[torch.arange(len(targets)), y]
    return {
        "token_ids": targets,
        "token_count": len(targets),
        "total_logprob": float(token_logps.sum().item()),
        "mean_logprob": float(token_logps.mean().item()),
        "token_logprobs": [float(x) for x in token_logps.tolist()],
    }


def family_scores(model, tokenizer, prompt: str, prefix: str, shared_prefix_id: int | None) -> dict:
    rows = {}
    raw = []
    for family in FAMILIES:
        continuation = json.dumps(family) + "}"
        row = score_continuation(model, tokenizer, prompt, prefix, continuation, shared_prefix_id)
        rows[family] = row
        raw.append(row["total_logprob"])
    normalizer = torch.logsumexp(torch.tensor(raw, dtype=torch.float64), dim=0).item()
    for family in FAMILIES:
        rows[family]["normalized_sequence_probability"] = math.exp(rows[family]["total_logprob"] - normalizer)
    ranked = sorted(FAMILIES, key=lambda f: rows[f]["total_logprob"], reverse=True)
    return {"ranked": ranked, "scores": rows}


def divergence_probe(model, tokenizer, prompt: str, prefix: str, shared_prefix_id: int | None) -> dict:
    name_ids = {family: continuation_ids(tokenizer, json.dumps(family), shared_prefix_id) for family in FAMILIES}
    shared = common_prefix(list(name_ids.values()))
    if any(len(ids) <= len(shared) for ids in name_ids.values()):
        return {"available": False, "reason": "candidate ends inside common prefix", "name_token_ids": name_ids}
    context = [int(x) for x in tokenizer.encode(prompt + prefix)] + shared
    x = torch.tensor([context], dtype=torch.long)
    with torch.inference_mode():
        logits, _ = model(x, None)
        next_logits = logits[0, -1]
        probs = torch.softmax(next_logits, dim=-1)
    candidate_next = {family: int(ids[len(shared)]) for family, ids in name_ids.items()}
    rows = {}
    for family, tid in candidate_next.items():
        logit = float(next_logits[tid].item())
        rank = int((next_logits > next_logits[tid]).sum().item()) + 1
        rows[family] = {
            "token_id": tid,
            "token_text": tokenizer.decode([tid]),
            "logit": logit,
            "probability": float(probs[tid].item()),
            "vocab_rank": rank,
        }
    ranked = sorted(FAMILIES, key=lambda f: rows[f]["logit"], reverse=True)
    return {
        "available": True,
        "shared_name_prefix_ids": shared,
        "shared_name_prefix_text": tokenizer.decode(shared) if shared else "",
        "candidate_next": rows,
        "ranked": ranked,
    }


def generated_name_prefix(model, tokenizer, prompt: str) -> dict:
    completion = ev.generate_completion(model, tokenizer, torch, prompt, GENERATION)
    marker = '"name":'
    pos = completion.find(marker)
    if pos < 0:
        return {"available": False, "completion": completion, "reason": "name field not found"}
    prefix = completion[: pos + len(marker)]
    if not prefix.startswith("<|tool|>"):
        return {"available": False, "completion": completion, "reason": "completion did not begin in tool mode"}
    return {"available": True, "completion": completion, "prefix": prefix}


def make_cases() -> tuple[list[dict], list[dict]]:
    development = []
    for c in repair.TRAIN:
        if c["kind"] != "tool_call":
            continue
        development.append({
            "id": c["id"],
            "split": "development",
            "prompt": c["prompt"],
            "expected_tool": c["expected_tool"],
            "arguments": c["arguments"],
            "user": c["user"],
        })
    challenge = []
    for c in held.CASES:
        if c["kind"] != "tool_call":
            continue
        if c["id"] not in HELDOUT_ARGS:
            raise RuntimeError(f"missing held-out canonical arguments for {c['id']}")
        challenge.append({
            "id": c["id"],
            "split": "heldout",
            "prompt": c["prompt"],
            "expected_tool": c["expected_tool"],
            "arguments": HELDOUT_ARGS[c["id"]],
            "user": c["user"],
        })
    if len(development) != 16 or len(challenge) != 8:
        raise RuntimeError(f"unexpected case counts: development={len(development)}, heldout={len(challenge)}")
    return development, challenge


def evaluate_case(model, tokenizer, case: dict, shared_prefix_id: int | None) -> dict:
    expected = case["expected_tool"]
    name_first_prefix = '<|tool|>\n{"name":'
    args_json = json.dumps(case["arguments"], separators=(",", ":"))
    args_first_prefix = '<|tool|>\n{"arguments":' + args_json + ',"name":'

    modes = {}
    for mode, prefix in (("name_first", name_first_prefix), ("args_first_correct", args_first_prefix)):
        scores = family_scores(model, tokenizer, case["prompt"], prefix, shared_prefix_id)
        divergence = divergence_probe(model, tokenizer, case["prompt"], prefix, shared_prefix_id)
        modes[mode] = {
            "prefix": prefix,
            "winner": scores["ranked"][0],
            "expected_rank": scores["ranked"].index(expected) + 1,
            "expected_probability": scores["scores"][expected]["normalized_sequence_probability"],
            "scores": scores,
            "divergence": divergence,
        }

    generated = generated_name_prefix(model, tokenizer, case["prompt"])
    if generated["available"]:
        scores = family_scores(model, tokenizer, case["prompt"], generated["prefix"], shared_prefix_id)
        divergence = divergence_probe(model, tokenizer, case["prompt"], generated["prefix"], shared_prefix_id)
        generated.update({
            "winner": scores["ranked"][0],
            "expected_rank": scores["ranked"].index(expected) + 1,
            "expected_probability": scores["scores"][expected]["normalized_sequence_probability"],
            "scores": scores,
            "divergence": divergence,
        })
    modes["args_first_generated"] = generated

    return {
        **case,
        "modes": modes,
    }


def summarize(rows: list[dict]) -> dict:
    metrics = {}
    for split in ("development", "heldout", "all"):
        subset = rows if split == "all" else [r for r in rows if r["split"] == split]
        split_metrics = {}
        for mode in ("name_first", "args_first_correct", "args_first_generated"):
            available = [r for r in subset if r["modes"][mode].get("available", True)]
            correct = sum(r["modes"][mode].get("winner") == r["expected_tool"] for r in available)
            avg_expected_p = (
                sum(float(r["modes"][mode].get("expected_probability", 0.0)) for r in available) / len(available)
                if available else 0.0
            )
            split_metrics[mode] = {
                "correct": int(correct),
                "available": len(available),
                "total": len(subset),
                "accuracy": float(correct / len(available)) if available else 0.0,
                "mean_expected_sequence_probability": avg_expected_p,
            }
        metrics[split] = split_metrics
    return metrics


def confusion(rows: list[dict], mode: str, split: str) -> dict:
    out = {expected: {pred: 0 for pred in FAMILIES} for expected in FAMILIES}
    for r in rows:
        if split != "all" and r["split"] != split:
            continue
        m = r["modes"][mode]
        if not m.get("available", True) or "winner" not in m:
            continue
        out[r["expected_tool"]][m["winner"]] += 1
    return out


def evaluate_model(model, tokenizer, label: str, cases: list[dict]) -> dict:
    contract = ev.special_token_contract(tokenizer)
    if not contract["atomic"] or not contract["unique"]:
        raise RuntimeError("special-token contract must remain atomic and unique")
    shared_prefix_id = contract.get("shared_prefix_id")
    torch.manual_seed(int(GENERATION["seed"]))
    rows = []
    for case in cases:
        row = evaluate_case(model, tokenizer, case, shared_prefix_id)
        rows.append(row)
        compact = {
            "event": "tool_family_case",
            "model": label,
            "split": case["split"],
            "id": case["id"],
            "expected": case["expected_tool"],
            "name_first": row["modes"]["name_first"]["winner"],
            "args_first_correct": row["modes"]["args_first_correct"]["winner"],
            "generated_available": row["modes"]["args_first_generated"]["available"],
            "args_first_generated": row["modes"]["args_first_generated"].get("winner"),
        }
        print(json.dumps(compact), flush=True)
    return {
        "label": label,
        "metrics": summarize(rows),
        "confusion": {
            split: {
                mode: confusion(rows, mode, split)
                for mode in ("name_first", "args_first_correct", "args_first_generated")
            }
            for split in ("development", "heldout", "all")
        },
        "cases": rows,
    }


def summary_markdown(report: dict) -> str:
    lines = [
        "# Ember v0.0.54 tool-family classifier diagnostic",
        "",
        "Saved v0.0.53 step-9 candidate; read-only CPU diagnostic.",
        "24 true tool requests: 16 development + 8 unchanged held-out challenge cases.",
        "",
        "| Model | Split | Name-first | Args-first correct | Args-first generated |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for model_key in ("full", "int4"):
        for split in ("development", "heldout", "all"):
            m = report[model_key]["metrics"][split]
            cells = []
            for mode in ("name_first", "args_first_correct", "args_first_generated"):
                x = m[mode]
                cells.append(f"{x['correct']}/{x['available']}")
            lines.append(f"| {model_key.upper()} | {split} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines += ["", "Interpretation hints:",
              "- Strong name-first but weak args-first implies generation order/path dependence is the main problem.",
              "- Weak name-first and args-first implies tool-family representation itself is weak.",
              "- Strong correct-args but weak generated-prefix implies wrong argument schema/content commits the model to the wrong family.",
              "", "No weights were changed, saved, promoted, or integrated."]
    return "\n".join(lines) + "\n"


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    api = HfApi(token=token)
    owner = api.whoami()["name"]
    repo = f"{owner}/{held.MODEL_NAME}"
    development, challenge = make_cases()
    cases = development + challenge

    with tempfile.TemporaryDirectory(prefix="ember-v054-family-") as td:
        work = Path(td)
        archive = ev.download_verified(ev.PACKAGE_URL, work / "ember.zip", ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        import sys
        sys.path.insert(0, str(work / "src" / "ember"))

        full_model, full_tok, full_checkpoint = held.load_full(repo, work, token)
        full = evaluate_model(full_model, full_tok, "full", cases)
        del full_model, full_checkpoint

        int4_model, int4_tok = held.load_int4(repo, work, token)
        int4 = evaluate_model(int4_model, int4_tok, "int4", cases)
        del int4_model

    report = {
        "schema_version": 1,
        "diagnostic": "ember-v054-tool-family-classifier-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_repo": repo,
        "run_id": held.RUN_ID,
        "checkpoint_sha256": held.BEST_SHA256,
        "int4_sha256": held.INT4_SHA256,
        "families": list(FAMILIES),
        "development_cases": len(development),
        "heldout_cases": len(challenge),
        "generation": GENERATION,
        "full": full,
        "int4": int4,
        "read_only": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
    print(json.dumps({
        "event": "tool_family_complete",
        "full": full["metrics"],
        "int4": int4["metrics"],
        "elapsed_seconds": report["elapsed_seconds"],
    }), flush=True)


if __name__ == "__main__":
    main()
