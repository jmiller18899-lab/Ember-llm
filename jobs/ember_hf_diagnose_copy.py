# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "huggingface-hub>=1.4",
#   "sentencepiece>=0.2",
#   "torch>=2.4",
# ]
# ///
"""Diagnose Ember exact-copy, prompt conditioning, and tokenizer behavior on CPU.

This is a read-only diagnostic. It does not train, mutate, upload, or promote a
checkpoint. It compares the authoritative v0.0.7 baseline with the accepted
experimental v0.0.9 checkpoint on deterministic unseen values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile

PACKAGE_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/main/ember-v0.0.7-hf-ready.zip"
PACKAGE_SHA256 = "27e8f7c80317652a22b3d58a0bd474724491a685dfe9e20c0b997b7c5907a289"
DEFAULT_MODELS = ("ember-v0.0.7-t4", "ember-v0.0.9-t4")
EOT = "<|endoftext|>"

CASES = (
    {"id": "short_code", "value": "Q7M4", "alternate": "R8N5"},
    {"id": "long_code", "value": "V9K2-4R7P", "alternate": "W8L3-5S6Q"},
    {"id": "digits", "value": "58310429", "alternate": "69421530"},
    {"id": "model_id", "value": "openai/gpt-6-astra", "alternate": "openai/gpt-5.6-sol"},
    {"id": "url", "value": "https://example.test/a7Q9", "alternate": "https://example.test/b8R2"},
    {"id": "path", "value": "/tmp/ember/Q7M4/result.json", "alternate": "/tmp/ember/R8N5/output.json"},
    {"id": "entity", "value": "Northfield Zephyr", "alternate": "Westhaven Orion"},
    {"id": "expression", "value": "53*19+7", "alternate": "61*17+9"},
    {"id": "mixed", "value": "acct_Q7m4-5831", "alternate": "acct_R8n5-6942"},
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, destination: Path, expected_sha256: str) -> Path:
    urllib.request.urlretrieve(url, destination)
    actual = sha256_file(destination)
    if actual != expected_sha256:
        raise RuntimeError(f"download checksum mismatch for {url}: {actual}")
    return destination


def prompt_for(value: str) -> str:
    return (
        "<|system|>\nYou are Ember. Copy the target from the current user message exactly. "
        "Do not substitute, normalize, calculate, explain, or call a tool. Stop at endoftext.\n"
        f"<|user|>\nIgnore distractors old=K2P8 and fallback=77291. TARGET={value}. "
        "Reply with TARGET exactly once.\n<|assistant|>\n"
    )


def expected_completion(value: str) -> str:
    return f"{value}\n{EOT}\n"


def checkpoint_path(files: list[str]) -> str:
    preferred = sorted(path for path in files if path.startswith("checkpoints/") and path.endswith("/best.pt"))
    if preferred:
        return preferred[-1]
    fallback = sorted(path for path in files if path.endswith("best.pt"))
    if fallback:
        return fallback[-1]
    raise RuntimeError("model repository has no best.pt checkpoint")


def eos_greedy(model, tokenizer, torch, prompt: str, max_new: int = 64) -> tuple[str, bool]:
    prompt_ids = list(tokenizer.encode(prompt))
    eot_ids = list(tokenizer.encode(EOT))
    if not eot_ids:
        raise RuntimeError("tokenizer cannot encode endoftext")
    eot_id = int(eot_ids[-1])
    seq = torch.tensor([prompt_ids], dtype=torch.long)
    generated: list[int] = []
    stopped = False
    with torch.inference_mode():
        for _ in range(max_new):
            x = seq[:, -int(model.cfg.block_size):]
            logits, _ = model(x, None)
            nxt = int(torch.argmax(logits[0, -1]).item())
            generated.append(nxt)
            seq = torch.cat([seq, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
            if nxt == eot_id:
                stopped = True
                break
    return tokenizer.decode(generated), stopped


def completion_nll(model, tokenizer, torch, prompt: str, completion: str) -> dict:
    prompt_ids = list(tokenizer.encode(prompt))
    full_ids = list(tokenizer.encode(prompt + completion))
    if full_ids[: len(prompt_ids)] != prompt_ids:
        return {"valid": False, "reason": "tokenization_changed_at_completion_boundary"}
    if len(full_ids) > int(model.cfg.block_size) + 1:
        return {"valid": False, "reason": "sequence_exceeds_context"}
    x = torch.tensor([full_ids[:-1]], dtype=torch.long)
    y = torch.tensor([full_ids[1:]], dtype=torch.long)
    first_target = max(0, len(prompt_ids) - 1)
    with torch.inference_mode():
        logits, _ = model(x, None)
        log_probs = torch.log_softmax(logits[0], dim=-1)
        positions = list(range(first_target, y.size(1)))
        if not positions:
            return {"valid": False, "reason": "no_completion_tokens"}
        token_nll = [-float(log_probs[pos, int(y[0, pos])].item()) for pos in positions]
    return {
        "valid": True,
        "mean_nll": sum(token_nll) / len(token_nll),
        "max_nll": max(token_nll),
        "tokens": len(token_nll),
    }


def tokenizer_metrics(tokenizer, value: str) -> dict:
    ids = list(tokenizer.encode(value))
    decoded = tokenizer.decode(ids)
    embedded = prompt_for(value)
    full_ids = list(tokenizer.encode(embedded))
    prefix = prompt_for("")
    prefix_ids = list(tokenizer.encode(prefix))
    return {
        "chars": len(value),
        "standalone_tokens": len(ids),
        "tokens_per_char": len(ids) / max(1, len(value)),
        "roundtrip_exact": decoded.strip() == value.strip(),
        "prompt_tokens": len(full_ids),
        "empty_target_prompt_tokens": len(prefix_ids),
        "prompt_token_delta": len(full_ids) - len(prefix_ids),
    }


def normalize_generated(text: str) -> str:
    visible = text.split(EOT, 1)[0]
    return visible.strip()


def diagnose_case(model, tokenizer, torch, case: dict) -> dict:
    value = str(case["value"])
    alternate = str(case["alternate"])
    prompt = prompt_for(value)
    generated, stopped = eos_greedy(model, tokenizer, torch, prompt)
    expected = completion_nll(model, tokenizer, torch, prompt, expected_completion(value))
    corrupted = completion_nll(model, tokenizer, torch, prompt, expected_completion(alternate))
    margin = None
    expected_wins = False
    if expected.get("valid") and corrupted.get("valid"):
        margin = float(corrupted["mean_nll"]) - float(expected["mean_nll"])
        expected_wins = margin > 0.0
    normalized = normalize_generated(generated)
    return {
        "id": case["id"],
        "value": value,
        "alternate": alternate,
        "tokenizer": tokenizer_metrics(tokenizer, value),
        "generation": {
            "text": generated,
            "normalized": normalized,
            "exact": normalized == value,
            "contains_target": value in normalized,
            "contains_alternate": alternate in normalized,
            "clean_stop": stopped,
        },
        "teacher_forcing": {
            "expected": expected,
            "corrupted": corrupted,
            "nll_margin_corrupt_minus_expected": margin,
            "expected_wins": expected_wins,
        },
    }


def summarize(cases: list[dict]) -> dict:
    count = len(cases)
    margins = [row["teacher_forcing"]["nll_margin_corrupt_minus_expected"] for row in cases]
    margins = [float(value) for value in margins if value is not None and math.isfinite(float(value))]
    return {
        "cases": count,
        "tokenizer_roundtrip_rate": sum(row["tokenizer"]["roundtrip_exact"] for row in cases) / count,
        "mean_tokens_per_char": sum(row["tokenizer"]["tokens_per_char"] for row in cases) / count,
        "exact_copy_rate": sum(row["generation"]["exact"] for row in cases) / count,
        "target_containment_rate": sum(row["generation"]["contains_target"] for row in cases) / count,
        "clean_stop_rate": sum(row["generation"]["clean_stop"] for row in cases) / count,
        "teacher_forced_expected_win_rate": sum(row["teacher_forcing"]["expected_wins"] for row in cases) / count,
        "mean_nll_margin_corrupt_minus_expected": (sum(margins) / len(margins)) if margins else None,
    }


def classify(baseline: dict, candidate: dict) -> dict:
    b = baseline["summary"]
    c = candidate["summary"]
    if c["tokenizer_roundtrip_rate"] < 1.0:
        verdict = "tokenizer_roundtrip_failure"
        next_step = "Repair tokenizer encode/decode fidelity before more SFT."
    elif c["teacher_forced_expected_win_rate"] < 0.75:
        verdict = "conditioning_or_foundation_bottleneck"
        next_step = "Stop copy-canary SFT iteration; strengthen general pretraining and token-level copy curriculum."
    elif c["exact_copy_rate"] < 0.75 and c["teacher_forced_expected_win_rate"] >= 0.75:
        verdict = "decoding_or_sequence_control_bottleneck"
        next_step = "Keep the checkpoint; inspect EOS/greedy decoding and output formatting before more training."
    elif c["exact_copy_rate"] >= 0.75:
        verdict = "copy_mechanism_healthy"
        next_step = "Move to the stronger semantic tool-argument/result gate before additional training."
    else:
        verdict = "mixed"
        next_step = "Inspect per-case NLL margins and token fragmentation before deciding on the next phase."
    return {
        "verdict": verdict,
        "next_step": next_step,
        "delta_exact_copy_rate_v009_minus_v007": c["exact_copy_rate"] - b["exact_copy_rate"],
        "delta_teacher_forced_win_rate_v009_minus_v007": c["teacher_forced_expected_win_rate"] - b["teacher_forced_expected_win_rate"],
        "delta_mean_nll_margin_v009_minus_v007": (
            None if b["mean_nll_margin_corrupt_minus_expected"] is None or c["mean_nll_margin_corrupt_minus_expected"] is None
            else c["mean_nll_margin_corrupt_minus_expected"] - b["mean_nll_margin_corrupt_minus_expected"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--output", default="ember-diagnostics.json")
    args = parser.parse_args()
    if tuple(args.models) != DEFAULT_MODELS:
        raise RuntimeError("diagnostic currently requires baseline v0.0.7 and accepted v0.0.9 in that order")
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required to read private Ember checkpoints")

    import torch
    from huggingface_hub import HfApi, hf_hub_download

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    api = HfApi(token=token)
    report = {
        "schema_version": 1,
        "diagnostic": "ember-copy-conditioning-v1",
        "read_only": True,
        "models": [],
    }
    with tempfile.TemporaryDirectory(prefix="ember-diagnose-") as temporary:
        work = Path(temporary)
        archive = download_verified(PACKAGE_URL, work / "ember.zip", PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        root = work / "src" / "ember"
        import sys
        sys.path.insert(0, str(root))
        from src.checkpoint import load_checkpoint
        from src.model import EmberGPT, ModelConfig
        from src.tokenizer import tokenizer_from_state_dict

        for model_name in args.models:
            repo_id = f"Jmiller18899/{model_name}"
            files = list(api.list_repo_files(repo_id, repo_type="model"))
            remote_checkpoint = checkpoint_path(files)
            local_checkpoint = Path(hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                filename=remote_checkpoint,
                token=token,
                local_dir=work / model_name,
            ))
            checkpoint = load_checkpoint(local_checkpoint, device="cpu")
            tokenizer = tokenizer_from_state_dict(checkpoint["tokenizer"])
            model = EmberGPT(ModelConfig(**checkpoint["model_config"]))
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            cases = [diagnose_case(model, tokenizer, torch, case) for case in CASES]
            report["models"].append({
                "model": model_name,
                "repo": repo_id,
                "checkpoint": remote_checkpoint,
                "checkpoint_sha256": sha256_file(local_checkpoint),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "summary": summarize(cases),
                "cases": cases,
            })
            del model, checkpoint

    report["comparison"] = classify(report["models"][0], report["models"][1])
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "diagnostic": report["diagnostic"],
        "baseline": report["models"][0]["summary"],
        "candidate": report["models"][1]["summary"],
        "comparison": report["comparison"],
        "output": str(output),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
