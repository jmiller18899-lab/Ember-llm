"""Bounded grounded-answer learning; archived tool parameters remain frozen."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from tool_assistant.runtime import hidden, load_model, sha256
from tool_assistant.evidence_v5 import emit_json
from . import quality
from .learn import configure_training, frozen_hash, encode_row, batch, mean_loss, state_of
from .learn import evaluate as old_evaluate
from .grounded_v2 import evaluate, validate_data


LOCK_FILES = (
    'direct_answers/learn.py', 'direct_answers/quality.py', 'direct_answers/canary-data.json',
    'direct_answers/learn_v2.py', 'direct_answers/grounded_v2.py', 'direct_answers/grounded-v2-data.json',
    'tool_assistant/runtime.py', 'tool_assistant/evidence_v5.py', 'tool_assistant/build.py',
    'tests/test_direct_grounded_v2.py', 'tests/test_direct_answer_learning.py',
    '.github/workflows/ember-direct-grounded-v2.yml',
)


def verify_source():
    root=Path(__file__).resolve().parents[1]
    lock=json.loads((root/'config/ember_direct_v2_source_lock.json').read_text())
    if set(lock.get('files',{})) != set(LOCK_FILES):raise ValueError('Incomplete source freeze')
    for name,digest in lock['files'].items():
        if sha256(root/name)!=digest:raise ValueError('Frozen experiment source changed: '+name)
    return lock


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
    source_freeze = verify_source()
    data_path = Path(__file__).parent / "grounded-v2-data.json"
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
    baseline_quality = evaluate(model, tokenizer, data["development"])
    historical_baseline = old_evaluate(model, tokenizer, quality.CASES)
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
    candidate_quality = evaluate(model, tokenizer, data["development"])
    historical_candidate = old_evaluate(model, tokenizer, quality.CASES)
    progress = best_step > 0 and best_dev <= baseline_dev * 0.9 and candidate_quality["passed"] >= baseline_quality["passed"] + 3 and historical_candidate["passed"] >= historical_baseline["passed"]
    confirmation = None
    if progress:
        # This file is never used for gradients or checkpoint selection.
        confirmation = evaluate(model, tokenizer, data["confirmation"])
    report = {
        "schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(), "precision": "full",
        "config": cfg, "data_sha256": sha256(data_path), "source_manifest_sha256": sha256(args.bundle / "manifest.json"),
        "trainable_names": list(selected), "trainable_parameters": sum(p.numel() for p in selected.values()),
        "frozen_parameter_sha256_before": frozen_before, "frozen_parameter_sha256_after": frozen_after,
        "block04_features_identical": True, "router_heads_changed": False,
        "baseline_train_loss": baseline_train, "baseline_development_loss": baseline_dev,
        "selected_step": best_step, "selected_development_loss": best_dev, "checkpoints": records,
        "training_examples_seen": len(set(used_ids)), "training_example_order": used_ids,
        "baseline_quality": baseline_quality, "candidate_quality": candidate_quality,
        "historical_baseline_quality": historical_baseline, "historical_candidate_quality": historical_candidate,
        "confirmation_gate_passed": confirmation is not None and confirmation["passed"] == confirmation["total"],
        "source_freeze": source_freeze,
        "quality_rule": "Conservative normalized reference match on 48 development tasks; historical24 lexical scores retained separately",
        "development_progress_gate": progress, "confirmation_consumed": progress, "confirmation": confirmation,
        "selected_adapter_sha256": sha256(selected_path), "production_ready": False,
        "int4_candidate_tested": False, "elapsed_training_and_final_eval_seconds": time.monotonic() - started,
        "interpretation": "A bounded last-block trial on source-grounded reference tasks. Reference matching can reject valid paraphrases and is not broad answer quality. Historical24 lexical scores are separate. No production promotion.",
    }
    if source_freeze != verify_source():
        raise RuntimeError("Experiment source changed during measurement")
    (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    emit_json(args.out / "report.json", "direct-grounded-v2-report.json")
    summary = ["# Ember grounded direct-answer v2", "",
               f"Grounded development reference checks: {baseline_quality['passed']}/48 → {candidate_quality['passed']}/48.",
               f"Lexical rule on the same new development cases: {baseline_quality['legacy_passed']}/48 → {candidate_quality['legacy_passed']}/48.",
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
                      "progress": progress, "fresh_confirmation": None if confirmation is None else f"{confirmation['passed']}/24"}), flush=True)


if __name__ == "__main__":
    main()
