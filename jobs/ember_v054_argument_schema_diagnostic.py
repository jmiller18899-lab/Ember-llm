"""Read-only argument-schema diagnostic for saved Ember v0.0.53 step 9.

The tool-family calibration experiments showed that the correct family becomes
much more recoverable when canonical arguments are supplied, while free-running
family selection remains unchanged. This diagnostic moves one decision earlier.

For each true tool request, after the fixed prefix:
    <|tool|>\n{"arguments":{
it scores the four canonical argument-key schemas:
    weather     -> "location":
    calculator  -> "expression":
    web_search  -> "query":
    get_time    -> "timezone":

It records both whole-key sequence probability and the first token where those
four schema keys diverge. It evaluates 16 development + 8 unchanged held-out
requests on the exact saved full and INT4 v0.0.53 checkpoints. On held-out cases
it also generates once and extracts the actual first argument key, so the
teacher-forced classifier can be compared with the free-running trajectory.

CPU-only, read-only. No optimizer step, checkpoint write, promotion, or
production integration.
"""
from __future__ import annotations

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

from jobs import ember_hf_eval as ev
from jobs import ember_v053_heldout_routing as held
from jobs import ember_v054_routing_repair as repair
from jobs import ember_v054_tool_family_diagnostic as family

OUT = Path("v054-argument-schema-diagnostic")
SCHEMAS = {
    "weather": "location",
    "calculator": "expression",
    "web_search": "query",
    "get_time": "timezone",
}
FAMILIES = tuple(SCHEMAS)
PREFIX = '<|tool|>\n{"arguments":{'
GENERATION = {"max_new_tokens": 48, "temperature": 1.0, "top_k": 1, "seed": 20260910}


def make_cases():
    dev, challenge = family.make_cases()
    return dev, challenge


def candidate_text(tool_family: str) -> str:
    return json.dumps(SCHEMAS[tool_family]) + ":"


def whole_schema_scores(model, tokenizer, prompt: str, shared_prefix_id: int | None) -> dict:
    rows = {}
    raw = []
    for tool_family in FAMILIES:
        row = family.score_continuation(
            model, tokenizer, prompt, PREFIX, candidate_text(tool_family), shared_prefix_id
        )
        rows[tool_family] = row
        raw.append(row["total_logprob"])
    z = torch.logsumexp(torch.tensor(raw, dtype=torch.float64), dim=0).item()
    for tool_family in FAMILIES:
        rows[tool_family]["normalized_sequence_probability"] = float(
            torch.exp(torch.tensor(rows[tool_family]["total_logprob"] - z, dtype=torch.float64)).item()
        )
    ranked = sorted(FAMILIES, key=lambda f: rows[f]["total_logprob"], reverse=True)
    return {"ranked": ranked, "scores": rows}


def divergence_probe(model, tokenizer, prompt: str, shared_prefix_id: int | None) -> dict:
    key_ids = {
        tool_family: family.continuation_ids(tokenizer, json.dumps(key), shared_prefix_id)
        for tool_family, key in SCHEMAS.items()
    }
    shared = family.common_prefix(list(key_ids.values()))
    if any(len(ids) <= len(shared) for ids in key_ids.values()):
        return {"available": False, "reason": "schema key ended in shared prefix", "key_ids": key_ids}
    context = [int(x) for x in tokenizer.encode(prompt + PREFIX)] + shared
    with torch.inference_mode():
        logits, _ = model(torch.tensor([context], dtype=torch.long), None)
        next_logits = logits[0, -1]
        probs = torch.softmax(next_logits, dim=-1)
    next_ids = {tool_family: int(ids[len(shared)]) for tool_family, ids in key_ids.items()}
    rows = {}
    for tool_family, tid in next_ids.items():
        rows[tool_family] = {
            "token_id": tid,
            "token_text": tokenizer.decode([tid]),
            "logit": float(next_logits[tid]),
            "probability": float(probs[tid]),
            "vocab_rank": int((next_logits > next_logits[tid]).sum().item()) + 1,
        }
    ranked = sorted(FAMILIES, key=lambda f: rows[f]["logit"], reverse=True)
    return {
        "available": True,
        "shared_schema_prefix_ids": shared,
        "shared_schema_prefix_text": tokenizer.decode(shared) if shared else "",
        "candidate_next": rows,
        "ranked": ranked,
        "first_distinguishing_token_ids": next_ids,
    }


def generated_argument_key(model, tokenizer, prompt: str) -> dict:
    completion = ev.generate_completion(model, tokenizer, torch, prompt, GENERATION)
    # Be tolerant of whitespace while requiring the first key inside arguments.
    match = re.search(r'"arguments"\s*:\s*\{\s*"([^"]+)"\s*:', completion)
    key = match.group(1) if match else None
    family_name = next((f for f, schema in SCHEMAS.items() if schema == key), None)
    return {"completion": completion, "argument_key": key, "schema_family": family_name}


def evaluate_case(model, tokenizer, case: dict, shared_prefix_id: int | None, generate: bool) -> dict:
    expected = case["expected_tool"]
    whole = whole_schema_scores(model, tokenizer, case["prompt"], shared_prefix_id)
    first = divergence_probe(model, tokenizer, case["prompt"], shared_prefix_id)
    row = {
        "id": case["id"],
        "split": case["split"],
        "user": case["user"],
        "expected_tool": expected,
        "expected_schema": SCHEMAS[expected],
        "whole_key": {
            "winner": whole["ranked"][0],
            "expected_rank": whole["ranked"].index(expected) + 1,
            "expected_probability": whole["scores"][expected]["normalized_sequence_probability"],
            "scores": whole,
        },
        "first_distinguishing_token": {
            **first,
            "winner": first["ranked"][0] if first.get("available") else None,
            "expected_rank": first["ranked"].index(expected) + 1 if first.get("available") else None,
        },
    }
    if generate:
        row["free_generation"] = generated_argument_key(model, tokenizer, case["prompt"])
    return row


def metrics(rows: list[dict], split: str) -> dict:
    subset = rows if split == "all" else [r for r in rows if r["split"] == split]
    whole_correct = sum(r["whole_key"]["winner"] == r["expected_tool"] for r in subset)
    first_correct = sum(r["first_distinguishing_token"]["winner"] == r["expected_tool"] for r in subset)
    generated = [r for r in subset if "free_generation" in r]
    generated_correct = sum(r["free_generation"].get("schema_family") == r["expected_tool"] for r in generated)
    return {
        "total": len(subset),
        "whole_key_correct": whole_correct,
        "whole_key_accuracy": whole_correct / len(subset) if subset else 0.0,
        "first_token_correct": first_correct,
        "first_token_accuracy": first_correct / len(subset) if subset else 0.0,
        "generated_available": len(generated),
        "generated_schema_correct": generated_correct,
        "generated_schema_accuracy": generated_correct / len(generated) if generated else None,
    }


def confusion(rows: list[dict], field: str, split: str) -> dict:
    out = {expected: {pred: 0 for pred in FAMILIES} for expected in FAMILIES}
    for r in rows:
        if split != "all" and r["split"] != split:
            continue
        if field == "whole":
            pred = r["whole_key"]["winner"]
        elif field == "first":
            pred = r["first_distinguishing_token"]["winner"]
        else:
            pred = r.get("free_generation", {}).get("schema_family")
        if pred in FAMILIES:
            out[r["expected_tool"]][pred] += 1
    return out


def evaluate_model(model, tokenizer, label: str, development: list[dict], challenge: list[dict]) -> dict:
    contract = ev.special_token_contract(tokenizer)
    if not contract["atomic"] or not contract["unique"]:
        raise RuntimeError("special-token contract drifted")
    shared_prefix_id = contract.get("shared_prefix_id")
    rows = []
    for case in development:
        row = evaluate_case(model, tokenizer, case, shared_prefix_id, generate=False)
        rows.append(row)
        print(json.dumps({
            "event": "schema_case", "model": label, "split": "development", "id": case["id"],
            "expected": case["expected_tool"], "whole": row["whole_key"]["winner"],
            "first": row["first_distinguishing_token"]["winner"],
        }), flush=True)
    torch.manual_seed(int(GENERATION["seed"]))
    for case in challenge:
        row = evaluate_case(model, tokenizer, case, shared_prefix_id, generate=True)
        rows.append(row)
        print(json.dumps({
            "event": "schema_case", "model": label, "split": "heldout", "id": case["id"],
            "expected": case["expected_tool"], "whole": row["whole_key"]["winner"],
            "first": row["first_distinguishing_token"]["winner"],
            "generated_key": row["free_generation"]["argument_key"],
            "generated_family": row["free_generation"]["schema_family"],
        }), flush=True)
    return {
        "label": label,
        "metrics": {split: metrics(rows, split) for split in ("development", "heldout", "all")},
        "confusion": {
            split: {
                "whole_key": confusion(rows, "whole", split),
                "first_token": confusion(rows, "first", split),
                "free_generation": confusion(rows, "generated", split),
            }
            for split in ("development", "heldout", "all")
        },
        "cases": rows,
    }


def summary_markdown(report: dict) -> str:
    lines = [
        "# Ember v0.0.54 argument-schema diagnostic", "",
        "Exact saved v0.0.53 step-9 candidate; read-only CPU evaluation.",
        "Schema decision tested immediately after `<|tool|>\\n{\"arguments\":{`.", "",
        "| Model | Split | Whole schema key | First distinguishing token | Free generated schema |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for model_key in ("full", "int4"):
        for split in ("development", "heldout", "all"):
            m = report[model_key]["metrics"][split]
            generated = "—" if not m["generated_available"] else f"{m['generated_schema_correct']}/{m['generated_available']}"
            lines.append(
                f"| {model_key.upper()} | {split} | {m['whole_key_correct']}/{m['total']} | "
                f"{m['first_token_correct']}/{m['total']} | {generated} |"
            )
    lines += ["", "Interpretation:",
              "- Weak schema classification here means the wrong family trajectory begins before argument values and before the tool-name field.",
              "- Strong teacher-forced schema but weak generated schema would instead point to an even earlier structural/prefix divergence.",
              "", "No training, checkpoint save, export, promotion, or integration occurred."]
    return "\n".join(lines) + "\n"


def main():
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

    with tempfile.TemporaryDirectory(prefix="ember-v054-schema-") as td:
        work = Path(td)
        archive = ev.download_verified(ev.PACKAGE_URL, work / "ember.zip", ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        import sys
        sys.path.insert(0, str(work / "src" / "ember"))

        full_model, full_tok, full_checkpoint = held.load_full(repo, work, token)
        full = evaluate_model(full_model, full_tok, "full", development, challenge)
        del full_model, full_checkpoint

        int4_model, int4_tok = held.load_int4(repo, work, token)
        int4 = evaluate_model(int4_model, int4_tok, "int4", development, challenge)
        del int4_model

    report = {
        "schema_version": 1,
        "diagnostic": "ember-v054-argument-schema-classifier-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_repo": repo,
        "run_id": held.RUN_ID,
        "checkpoint_sha256": held.BEST_SHA256,
        "int4_sha256": held.INT4_SHA256,
        "schemas": SCHEMAS,
        "development_cases": len(development),
        "heldout_cases": len(challenge),
        "full": full,
        "int4": int4,
        "read_only": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (OUT / "summary.md").write_text(summary_markdown(report))
    print(json.dumps({
        "event": "argument_schema_complete",
        "full": full["metrics"], "int4": int4["metrics"],
        "elapsed_seconds": report["elapsed_seconds"],
    }), flush=True)


if __name__ == "__main__":
    main()
