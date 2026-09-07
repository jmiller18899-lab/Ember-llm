# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["huggingface-hub>=1.4", "sentencepiece>=0.2", "torch>=2.4"]
# ///
"""Ember v0.0.24: CPU-only token-support and capacity diagnostic.

This phase does not train. It loads the strongest v0.0.20 best checkpoint and
inspects the held-out literal-copy decisions token by token. For each diagnostic
continuation position it records the expected token rank, correct-vs-best-wrong
logit gap, tokenizer piece, and support in the original leakage-safe synthetic
training curriculum globally, within the same format, and at the same relative
position. The goal is to distinguish sparse token/data support from a near-miss
objective/calibration problem or a deeper contextual/model-capacity miss.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

CONFIG_COMMIT = "b108908e7a4f3f8cccf8df15e031d0e3d2d32d36"
CONFIG_URL = (
    "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/"
    f"{CONFIG_COMMIT}/config/ember_token_diagnostic_v0.0.24.json"
)
BASE_PIN = "67a94d20530901a2524da8010408d5f2c728a165"
BASE_URL = (
    "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/"
    f"{BASE_PIN}/jobs/ember_hf_sft_v015.py"
)
DATA_PIN = "14995a94a4d1594463e266c4c5fed0ecec329da9"
DATA_URL = (
    "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/"
    f"{DATA_PIN}/jobs/ember_sft_data_v015.py"
)
SOURCE_REPO = "Jmiller18899/ember-v0.0.20-t4"
EXPECTED_OWNER = "Jmiller18899"


def fetch(url: str, path: Path) -> Path:
    urllib.request.urlretrieve(url, path)
    return path


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"unable to load module {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_piece(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return f"<decode-error:{int(token_id)}>"


def resolve_source(api: HfApi, token: str, work: Path, base):
    state_path = Path(hf_hub_download(
        repo_id=SOURCE_REPO,
        repo_type="model",
        filename="run-state.json",
        token=token,
        local_dir=work / "source-state",
    ))
    state = json.loads(state_path.read_text())
    if state.get("status") != "evaluation_complete":
        raise RuntimeError(f"v0.0.20 source state is not evaluation_complete: {state}")
    run_id = str(state.get("run_id", "")).strip()
    if not run_id:
        raise RuntimeError("v0.0.20 run-state.json has no run_id")
    remote = f"checkpoints/{run_id}/best.pt"
    checkpoint = Path(hf_hub_download(
        repo_id=SOURCE_REPO,
        repo_type="model",
        filename=remote,
        token=token,
        local_dir=work / "source-model",
    ))
    return checkpoint, {
        "repo": SOURCE_REPO,
        "checkpoint": remote,
        "sha256": base.sha(checkpoint),
        "run_id": run_id,
        "promotion": state.get("promotion"),
        "best_step": state.get("best_step"),
    }


def token_layout(tokenizer, prompt: str, value: str, base):
    prompt_ids = list(tokenizer.encode(prompt))
    full = list(tokenizer.encode(prompt + base.completion_for(value)))
    if full[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError("v0.0.24 prompt/completion boundary changed")
    eot_id = int(tokenizer.encode(base.EOT)[-1])
    eot_positions = [i for i in range(len(prompt_ids), len(full)) if int(full[i]) == eot_id]
    if not eot_positions:
        raise RuntimeError("v0.0.24 token layout has no EOS")
    y = full[1:]
    first_target = len(prompt_ids) - 1
    eos_target = eot_positions[0] - 1
    copy_positions = list(range(first_target, eos_target))
    continuation_positions = list(range(first_target + 1, eos_target))
    return {
        "prompt_ids": prompt_ids,
        "full": full,
        "y": y,
        "copy_positions": copy_positions,
        "continuation_positions": continuation_positions,
    }


def build_support(tokenizer, rows, base):
    global_tokens = Counter()
    kind_tokens = defaultdict(Counter)
    kind_offset_tokens = defaultdict(Counter)
    kind_prev_pairs = defaultdict(Counter)
    kind_lengths = defaultdict(list)

    for row in rows:
        kind = str(row["kind"])
        layout = token_layout(tokenizer, row["prompt"], str(row["value"]), base)
        y = layout["y"]
        copy_positions = layout["copy_positions"]
        continuation_positions = layout["continuation_positions"]
        kind_lengths[kind].append(len(copy_positions))
        for offset, pos in enumerate(continuation_positions):
            token_id = int(y[pos])
            prev_id = int(y[pos - 1])
            global_tokens[token_id] += 1
            kind_tokens[kind][token_id] += 1
            kind_offset_tokens[(kind, offset)][token_id] += 1
            kind_prev_pairs[kind][(prev_id, token_id)] += 1

    return {
        "global_tokens": global_tokens,
        "kind_tokens": kind_tokens,
        "kind_offset_tokens": kind_offset_tokens,
        "kind_prev_pairs": kind_prev_pairs,
        "kind_lengths": kind_lengths,
    }


def inspect_case(model, tokenizer, value: str, kind: str, device, torch, base, support, cfg):
    prompt = base.prompt_for(value)
    layout = token_layout(tokenizer, prompt, value, base)
    full = layout["full"]
    y = layout["y"]
    copy_positions = layout["copy_positions"]
    continuation_positions = layout["continuation_positions"]
    x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits, _ = model(x, None)
        logits = logits[0].float()

    positions = []
    first_error_offset = None
    for offset, pos in enumerate(continuation_positions):
        row = logits[pos]
        expected_id = int(y[pos])
        predicted_id = int(torch.argmax(row).item())
        expected_logit = float(row[expected_id].item())
        rank = 1 + int((row > row[expected_id]).sum().item())
        top2_values, top2_indices = torch.topk(row, k=2)
        if int(top2_indices[0].item()) == expected_id:
            best_wrong_id = int(top2_indices[1].item())
            best_wrong_logit = float(top2_values[1].item())
        else:
            best_wrong_id = int(top2_indices[0].item())
            best_wrong_logit = float(top2_values[0].item())
        gap = expected_logit - best_wrong_logit
        prev_id = int(y[pos - 1])
        correct = predicted_id == expected_id
        if not correct and first_error_offset is None:
            first_error_offset = offset
        positions.append({
            "offset": offset,
            "correct": correct,
            "expected_id": expected_id,
            "expected_piece": safe_piece(tokenizer, expected_id),
            "predicted_id": predicted_id,
            "predicted_piece": safe_piece(tokenizer, predicted_id),
            "expected_rank": rank,
            "expected_logit": round(expected_logit, 6),
            "best_wrong_id": best_wrong_id,
            "best_wrong_piece": safe_piece(tokenizer, best_wrong_id),
            "best_wrong_logit": round(best_wrong_logit, 6),
            "gap": round(gap, 6),
            "support_global": int(support["global_tokens"][expected_id]),
            "support_same_kind": int(support["kind_tokens"][kind][expected_id]),
            "support_same_kind_offset": int(support["kind_offset_tokens"][(kind, offset)][expected_id]),
            "support_same_kind_prev_pair": int(support["kind_prev_pairs"][kind][(prev_id, expected_id)]),
        })

    first_error = None
    evidence_label = "clean"
    if first_error_offset is not None:
        first_error = positions[first_error_offset]
        sparse = (
            first_error["support_same_kind"] < int(cfg["strong_same_kind_support"])
            or first_error["support_same_kind_offset"] < int(cfg["strong_same_offset_support"])
        )
        near = (
            first_error["expected_rank"] <= int(cfg["near_miss_max_rank"])
            and first_error["gap"] >= float(cfg["near_miss_min_gap"])
        )
        deep = (
            first_error["expected_rank"] >= int(cfg["deep_miss_min_rank"])
            or first_error["gap"] <= float(cfg["deep_miss_max_gap"])
        )
        if sparse:
            evidence_label = "support_sparse"
        elif near:
            evidence_label = "near_miss_with_support"
        elif deep:
            evidence_label = "deep_context_miss_with_support"
        else:
            evidence_label = "moderate_context_miss_with_support"

    copy_token_ids = [int(y[pos]) for pos in copy_positions]
    return {
        "kind": kind,
        "target_chars": len(value),
        "copy_tokens": len(copy_token_ids),
        "chars_per_copy_token": round(len(value) / max(1, len(copy_token_ids)), 4),
        "copy_token_pieces": [safe_piece(tokenizer, tid) for tid in copy_token_ids],
        "continuation_tokens": len(continuation_positions),
        "first_error_offset": first_error_offset,
        "evidence_label": evidence_label,
        "first_error": first_error,
        "positions": positions,
    }


def summarize_hypothesis(cases):
    failed = [c for c in cases if c["first_error_offset"] is not None]
    labels = Counter(c["evidence_label"] for c in failed)
    if not failed:
        primary = "no_teacher_forced_continuation_failures"
    elif labels["support_sparse"] >= 3:
        primary = "tokenizer_or_training_support_bottleneck"
    elif labels["near_miss_with_support"] >= 3:
        primary = "decision_margin_or_objective_bottleneck"
    elif labels["deep_context_miss_with_support"] >= 3:
        primary = "context_representation_or_model_capacity_bottleneck"
    else:
        primary = "mixed_contextual_bottleneck"
    return {
        "failed_cases": len(failed),
        "label_counts": dict(labels),
        "primary_evidence_hypothesis": primary,
        "note": "Heuristic evidence classification; use raw rank/gap/support fields for the training decision.",
    }


def main():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required")

    import torch

    api = HfApi(token=token)
    identity = api.whoami()
    owner = str(identity.get("name", "")).strip()
    if owner.casefold() != EXPECTED_OWNER.casefold():
        raise RuntimeError(f"HF_TOKEN belongs to {owner!r}; expected {EXPECTED_OWNER!r}")

    with tempfile.TemporaryDirectory(prefix="ember-v024-diag-") as td:
        work = Path(td)
        cfg = json.loads(fetch(CONFIG_URL, work / "config.json").read_text())
        if cfg.get("version") != "0.0.24" or cfg.get("phase") != "token-support-capacity-diagnostic":
            raise RuntimeError("unexpected v0.0.24 diagnostic config")

        base = load_module(fetch(BASE_URL, work / "base.py"), "ember_v015_base")
        data = load_module(fetch(DATA_URL, work / "data.py"), "ember_v015_data")
        train_rows = data.build_examples("train", int(cfg["train_examples"]))
        val_rows = data.build_examples("validation", 450)
        data.assert_clean(train_rows, val_rows)

        package = base.fetch(base.PACKAGE_URL, work / "ember.zip", base.PACKAGE_SHA256)
        with zipfile.ZipFile(package) as z:
            z.extractall(work / "src")
        sys.path.insert(0, str(work / "src" / "ember"))
        from src.checkpoint import load_checkpoint
        from src.model import EmberGPT, ModelConfig
        from src.tokenizer import tokenizer_from_state_dict

        source_path, source_info = resolve_source(api, token, work, base)
        source = load_checkpoint(source_path, device="cpu")
        source_cfg = source.get("train_config") or {}
        if str(source_cfg.get("version")) != "0.0.20":
            raise RuntimeError(f"expected v0.0.20 checkpoint, got {source_cfg}")
        if str(source.get("run_id", "")) != source_info["run_id"]:
            raise RuntimeError("v0.0.20 checkpoint run_id does not match run-state.json")

        tokenizer = tokenizer_from_state_dict(source["tokenizer"])
        model = EmberGPT(ModelConfig(**source["model_config"]))
        model.load_state_dict(source["model_state"])
        model.eval()

        support = build_support(tokenizer, train_rows, base)
        cases = []
        for idx, pair in enumerate(base.DIAGNOSTICS):
            value = str(pair[0])
            kind = str(data.KINDS[idx])
            cases.append(inspect_case(model, tokenizer, value, kind, "cpu", torch, base, support, cfg))

        hypothesis = summarize_hypothesis(cases)
        compact_failures = []
        for case in cases:
            if case["first_error"] is None:
                continue
            compact_failures.append({
                "kind": case["kind"],
                "first_error_offset": case["first_error_offset"],
                "chars_per_copy_token": case["chars_per_copy_token"],
                "evidence_label": case["evidence_label"],
                "first_error": case["first_error"],
            })

        report = {
            "status": "PASS",
            "version": "0.0.24",
            "phase": cfg["phase"],
            "source": source_info,
            "training_support_rows": len(train_rows),
            "held_out_leakage": False,
            "failed_case_summary": compact_failures,
            "hypothesis": hypothesis,
            "cases": cases,
        }
        print(json.dumps(report, indent=2), flush=True)
        print(f"EMBER_V024_SOURCE_CHECKPOINT={source_info['checkpoint']}", flush=True)
        print(f"EMBER_V024_SOURCE_SHA256={source_info['sha256']}", flush=True)
        print("EMBER_V024_FAILED_CASE_SUMMARY=" + json.dumps(compact_failures, separators=(",", ":")), flush=True)
        print("EMBER_V024_HYPOTHESIS=" + json.dumps(hypothesis, separators=(",", ":")), flush=True)
        print("EMBER_HF_V024_DIAGNOSTIC=PASS", flush=True)


if __name__ == "__main__":
    main()
