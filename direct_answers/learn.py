"""Bounded native-PyTorch CPU canary with the routing layers frozen."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from tool_assistant.runtime import hidden, load_model, sha256
from . import quality

MARKERS = ("<|system|>", "<|user|>", "<|assistant|>", "<|tool|>", "<|tool_result|>", "<|endoftext|>")


def configure_training(model):
    model.requires_grad_(False)
    for module in (model.blocks[5], model.ln_f):
        module.requires_grad_(True)
    selected = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not selected or not all(name.startswith(("blocks.5.", "ln_f.")) for name in selected):
        raise RuntimeError("Unexpected trainable parameter scope")
    return selected


def frozen_hash(model):
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        if not p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def encode_row(tokenizer, row, block_size):
    rendered = quality.prompt(row["user"])
    prefix = tokenizer.encode(rendered)
    ids = tokenizer.encode(rendered + row["answer"] + "<|endoftext|>")
    if ids[:len(prefix)] != prefix:
        raise RuntimeError("Tokenizer changed the prompt/answer boundary")
    if len(ids) <= len(prefix) or len(ids) - 1 > block_size:
        raise RuntimeError("Invalid or oversized answer example")
    x = torch.tensor(ids[:-1], dtype=torch.long)
    y = torch.tensor(ids[1:], dtype=torch.long)
    y[:len(prefix) - 1] = -100
    return x, y


def batch(rows):
    longest = max(len(x) for x, _ in rows)
    x = torch.zeros((len(rows), longest), dtype=torch.long)
    y = torch.full((len(rows), longest), -100, dtype=torch.long)
    for i, (a, b) in enumerate(rows):
        x[i, :len(a)] = a
        y[i, :len(b)] = b
    return x, y


def mean_loss(model, rows):
    total = count = 0
    with torch.inference_mode():
        for start in range(0, len(rows), 2):
            x, y = batch(rows[start:start + 2])
            logits, _ = model(x)
            total += float(F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum", ignore_index=-100))
            count += int((y != -100).sum())
    return total / count


def generate(model, tokenizer, user):
    marker_ids = {m: tokenizer.encode(m)[-1] for m in MARKERS}
    if len(set(marker_ids.values())) != len(MARKERS):
        raise RuntimeError("Conversation markers are not distinct")
    ids = tokenizer.encode(quality.prompt(user))
    generated = []
    stopped = False
    with torch.inference_mode():
        for _ in range(min(64, model.cfg.block_size - len(ids))):
            logits, _ = model(torch.tensor([ids + generated], dtype=torch.long))
            scores = logits[0, -1].clone()
            for marker in MARKERS[:-1]:
                scores[marker_ids[marker]] = -torch.inf
            token = int(scores.argmax())
            if token == marker_ids["<|endoftext|>"]:
                stopped = True
                break
            generated.append(token)
    return {"text": tokenizer.decode(generated).strip(), "stopped_at_eot": stopped, "generated_tokens": len(generated)}


def evaluate(model, tokenizer, rows):
    output = []
    for row in rows:
        generation = generate(model, tokenizer, row["user"])
        semantic = quality.semantic_check(generation["text"], row["check"])
        generic = quality.generic_quality(generation["text"])
        passed = generation["stopped_at_eot"] and semantic["passed"] and generic["passed"]
        output.append({"id": row["id"], "family": row["family"], "user": row["user"], **generation,
                       "semantic": semantic, "quality": generic, "passed": passed})
    return {"passed": sum(r["passed"] for r in output), "total": len(output), "rows": output}


def validate_data(data, tool_training):
    groups = (data["train"], data["development"], data["confirmation"])
    if tuple(map(len, groups)) != (72, 12, 12):
        raise ValueError("Canary split sizes changed")
    normalized = lambda s: " ".join(s.casefold().split())
    seen = set()
    for group in groups:
        prompts = {normalized(r["user"]) for r in group}
        if len(prompts) != len(group) or prompts & seen:
            raise ValueError("Direct-answer data leakage")
        seen |= prompts
    forbidden = {normalized(c["user"]) for c in quality.CASES + tool_training}
    if seen & forbidden:
        raise ValueError("New direct data overlaps old quality cases or router data")
    return True


def state_of(selected):
    return {name: p.detach().clone() for name, p in selected.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("Refusing to overwrite a previous learning trial")
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(20260911)
    torch.use_deterministic_algorithms(True)
    data_path = Path(__file__).parent / "canary-data.json"
    data = json.loads(data_path.read_text())
    cfg = data["training_config"]
    tool_data = json.loads((Path(__file__).parent.parent / "tool_assistant/data/router-training.json").read_text())["cases"]
    validate_data(data, tool_data)
    model, tokenizer, manifest = load_model(args.bundle, "full")
    selected = configure_training(model)
    model.eval()  # Deterministic dropout-free learning still computes gradients.
    frozen_before = frozen_hash(model)
    routing_probes = [c["prompt"] for c in tool_data[:12]]
    features_before = torch.stack([hidden(model, tokenizer, p) for p in routing_probes])
    train = [encode_row(tokenizer, r, model.cfg.block_size) for r in data["train"]]
    dev = [encode_row(tokenizer, r, model.cfg.block_size) for r in data["development"]]
    baseline_train = mean_loss(model, train)
    baseline_dev = mean_loss(model, dev)
    baseline_quality = evaluate(model, tokenizer, quality.CASES)
    optimizer = torch.optim.AdamW(list(selected.values()), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best_dev, best_step, best_state = baseline_dev, 0, state_of(selected)
    records, used_ids = [], []
    rng = torch.Generator().manual_seed(20260911)
    order = []
    started = time.monotonic()
    for step in range(1, cfg["steps"] + 1):
        if time.monotonic() - started > cfg["maximum_training_seconds"]:
            raise RuntimeError("CPU training time budget exceeded")
        if len(order) < cfg["batch_size"]:
            order = torch.randperm(len(train), generator=rng).tolist()
        indices, order = order[:cfg["batch_size"]], order[cfg["batch_size"]:]
        used_ids.extend(data["train"][i]["id"] for i in indices)
        x, y = batch([train[i] for i in indices])
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(selected.values()), cfg["clip_norm"], error_if_nonfinite=True)
        optimizer.step()
        if step in cfg["checkpoints"]:
            dev_loss = mean_loss(model, dev)
            train_loss = mean_loss(model, train)
            row = {"step": step, "train_loss": train_loss, "development_loss": dev_loss}
            records.append(row)
            torch.save({"step": step, "trainable_state": state_of(selected), "optimizer": optimizer.state_dict(),
                        "rng_state": torch.get_rng_state(), "shuffle_rng_state": rng.get_state(), "remaining_order": order,
                        "source_manifest_sha256": sha256(args.bundle / "manifest.json"), "data_sha256": sha256(data_path), "config": cfg},
                       args.out / f"checkpoint-{step}.pt")
            if dev_loss < best_dev:
                best_dev, best_step, best_state = dev_loss, step, state_of(selected)
            print(json.dumps({"event": "direct_learning_checkpoint", **row}), flush=True)
    with torch.no_grad():
        for name, p in selected.items():
            p.copy_(best_state[name])
    frozen_after = frozen_hash(model)
    if frozen_before != frozen_after:
        raise RuntimeError("Protected model parameters changed")
    features_after = torch.stack([hidden(model, tokenizer, p) for p in routing_probes])
    torch.testing.assert_close(features_before, features_after, rtol=0, atol=0)
    selected_path = args.out / "selected-adapter.pt"
    torch.save({"trainable_state": best_state, "selected_step": best_step, "source_manifest_sha256": sha256(args.bundle / "manifest.json")}, selected_path)
    # Reload the saved artifact before measuring its generated answers.
    saved = torch.load(selected_path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["trainable_state"], strict=False)
    candidate_quality = evaluate(model, tokenizer, quality.CASES)
    progress = best_step > 0 and best_dev <= baseline_dev * 0.9 and candidate_quality["passed"] >= baseline_quality["passed"] + 3
    confirmation = None
    if progress:
        # This file is never used for gradients or checkpoint selection.
        confirmation = evaluate(model, tokenizer, data["confirmation"])
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "precision": "full",
        "config": cfg, "data_sha256": sha256(data_path), "source_manifest_sha256": sha256(args.bundle / "manifest.json"),
        "trainable_names": list(selected), "trainable_parameters": sum(p.numel() for p in selected.values()),
        "frozen_parameter_sha256_before": frozen_before, "frozen_parameter_sha256_after": frozen_after,
        "block04_features_identical": True, "router_heads_changed": False,
        "baseline_train_loss": baseline_train, "baseline_development_loss": baseline_dev,
        "selected_step": best_step, "selected_development_loss": best_dev, "checkpoints": records,
        "training_examples_seen": len(set(used_ids)), "training_example_order": used_ids,
        "baseline_quality": baseline_quality, "candidate_quality": candidate_quality,
        "development_progress_gate": progress, "confirmation_consumed": progress, "confirmation": confirmation,
        "selected_adapter_sha256": sha256(selected_path), "production_ready": False,
        "int4_candidate_tested": False, "elapsed_training_and_final_eval_seconds": time.monotonic() - started,
        "interpretation": "A small last-block learning test. Task checks are deterministic lexical checks; inspect the raw answers. This does not prove broad conversational quality or authorize promotion.",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    summary = ["# Independent Ember direct-answer canary", "",
               f"Old 24 development task checks: {baseline_quality['passed']}/24 → {candidate_quality['passed']}/24.",
               f"Development answer-token loss: {baseline_dev:.4f} → {best_dev:.4f}; selected step {best_step}.",
               f"Development progress gate: **{progress}**. Fresh direct confirmation consumed: **{progress}**.",
               "Routing-layer parameters and observed block_04 features remained byte-for-byte unchanged.",
               "", report["interpretation"]]
    if confirmation:
        summary.append(f"Fresh direct task checks: {confirmation['passed']}/{confirmation['total']}.")
    (args.out / "summary.md").write_text("\n\n".join(summary) + "\n")
    print(json.dumps({"event": "direct_learning_complete", "selected_step": best_step,
                      "baseline_quality": baseline_quality["passed"], "candidate_quality": candidate_quality["passed"],
                      "baseline_development_loss": baseline_dev, "selected_development_loss": best_dev,
                      "progress": progress, "fresh_confirmation": None if confirmation is None else f"{confirmation['passed']}/12"}), flush=True)


if __name__ == "__main__":
    main()
