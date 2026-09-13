"""Bounded CPU UFC comparison: pre-2022 train, 2022-23 validation, 2024+ test.

The three fixed seeds are ensembled; test results never select checkpoints,
seeds, hyperparameters, or the preferred model family.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from jobs.ember_bsq import BSQForecaster
from jobs.ember_bsq_data import chronological_split, load_examples, timestamp
from jobs.ember_bsq_train import ROOT, MODES, load_config, metrics, run, tensors

BOUNDARIES = (timestamp("2022-01-01T00:00:00Z"), timestamp("2024-01-01T00:00:00Z"))
SEEDS = (17, 29, 43)


def checkpoint_probabilities(path, rows):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = BSQForecaster(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    raw, _ = tensors(rows)
    norm = checkpoint["normalization"]
    x = (raw - norm["feature_mean"]) / norm["feature_scale"]
    with torch.no_grad():
        logits = model(x)[0]
        if checkpoint.get("symmetric_binary", False):
            swapped = -x - 2 * norm["feature_mean"] / norm["feature_scale"]
            logits = (logits - model(swapped)[0]) / 2
        return logits.sigmoid()


def logistic_baseline(parts, output):
    train_x, train_y = tensors(parts[0])
    # Zero-centered difference features and a zero intercept enforce A/B symmetry.
    train_x = train_x[:, 0]
    scale = train_x.square().mean(0).sqrt().clamp_min(1e-6)
    weights = torch.zeros(train_x.shape[1], requires_grad=True)
    optimizer = torch.optim.Adam([weights], lr=.02)
    val_x, val_y = tensors(parts[1])
    best, selected_step, best_weights = float("inf"), 0, None
    for step in range(1, 301):
        logits = (train_x / scale) @ weights
        loss = F.binary_cross_entropy_with_logits(logits, train_y) + .01 * weights.square().sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 25 == 0:
            with torch.no_grad():
                validation = float(F.binary_cross_entropy_with_logits((val_x[:, 0] / scale) @ weights, val_y))
            if validation < best:
                best, selected_step, best_weights = validation, step, weights.detach().clone()
    checkpoint = {"weights": best_weights, "scale": scale, "selected_step": selected_step}
    torch.save(checkpoint, output / "logistic.pt")
    return {name: ((tensors(part)[0][:, 0] / scale) @ best_weights).sigmoid()
            for name, part in zip(("train", "validation", "test"), parts)}


def score(probabilities, rows):
    target = tensors(rows)[1]
    result = metrics(torch.logit(probabilities.clamp(1e-6, 1 - 1e-6)), target, "binary")
    result["fights"] = len(rows)
    result["correct"] = int(((probabilities >= .5) == target.bool()).sum())
    return result


def event_bootstrap_difference(left, right, rows, repeats=1000):
    """Paired log-loss interval, resampling cards rather than independent fights."""
    target = np.array([r.target for r in rows])
    def loss(p):
        p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
        return -(target * np.log(p) + (1 - target) * np.log1p(-p))
    differences = loss(left) - loss(right)
    events = sorted({r.event_id for r in rows})
    indices = {event: i for i, event in enumerate(events)}
    codes = np.array([indices[r.event_id] for r in rows])
    sums = np.bincount(codes, weights=differences)
    counts = np.bincount(codes)
    rng = np.random.default_rng(20260913)
    samples = []
    for _ in range(repeats):
        selected = rng.integers(0, len(events), len(events))
        samples.append(sums[selected].sum() / counts[selected].sum())
    return {"mean_log_loss_difference": float(differences.mean()),
            "event_bootstrap_95_percent_interval": [float(x) for x in np.quantile(samples, [.025, .975])],
            "events": len(events), "resamples": repeats, "negative_favors": "bsq"}


def benchmark(data, cfg, output):
    torch.set_num_threads(1)
    rows = load_examples(data, cfg)
    parts, split = chronological_split(rows, BOUNDARIES)
    raw = {r["sample_id"]: r for r in map(json.loads, data.read_text().splitlines())}
    output.mkdir(parents=True, exist_ok=False)
    neural = {mode: {part: [] for part in ("validation", "test")} for mode in MODES}
    seed_results = {}
    for seed in SEEDS:
        local = copy.deepcopy(cfg)
        local["seed"] = seed
        folder = output / f"seed-{seed}"
        report = run(local, data, folder, "historical_ufc_retrospective", BOUNDARIES, symmetric_binary=True)
        seed_results[str(seed)] = report["models"]
        for mode in MODES:
            for name, part in zip(("validation", "test"), parts[1:]):
                neural[mode][name].append(checkpoint_probabilities(folder / f"{mode}.pt", part))
        print(json.dumps({"completed_seed": seed, "test_not_used_for_selection": True}), flush=True)
    probabilities = {mode: {name: torch.stack(values).mean(0) for name, values in sections.items()}
                     for mode, sections in neural.items()}
    probabilities["logistic"] = logistic_baseline(parts, output)
    probabilities["elo"] = {name: torch.tensor([raw[r.sample_id]["elo_probability"] for r in part])
                             for name, part in zip(("validation", "test"), parts[1:])}
    probabilities["constant"] = {name: torch.full((len(part),), sum(r.target for r in parts[0]) / len(parts[0]))
                                  for name, part in zip(("validation", "test"), parts[1:])}
    scores = {model: {name: score(p[name], part) for name, part in zip(("validation", "test"), parts[1:])}
              for model, p in probabilities.items()}
    selected = min(scores, key=lambda model: scores[model]["validation"]["log_loss"])
    result = {"status": "completed", "task": cfg["target_name"], "data_kind": "historical_ufc_retrospective",
              "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(), "split": split,
              "seeds": SEEDS, "max_steps_per_model": cfg["max_steps"], "symmetric_probabilities": True,
              "selected_by_validation_log_loss": selected, "scores": scores, "per_seed": seed_results,
              "bsq_minus_continuous": event_bootstrap_difference(probabilities["bsq"]["test"], probabilities["continuous"]["test"], parts[2]),
              "production_integrated": False, "ember_gpt_weights_updated": False,
              "interpretation": "retrospective held-out evidence; not a live betting or promotion gate"}
    (output / "benchmark.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (output / "test_predictions.jsonl").open("w") as stream:
        for i, row in enumerate(parts[2]):
            source = raw[row.sample_id]
            record = {"fight_id": row.sample_id, "event_id": row.event_id,
                      "fighter_a": source["fighter_a"], "fighter_b": source["fighter_b"],
                      "event_date": source["event_date"], "target": row.target,
                      "probabilities": {m: float(p["test"][i]) for m, p in probabilities.items()}}
            stream.write(json.dumps(record) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/ember_ufc_forecast.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(args.data, load_config(args.config), args.output_dir)
    print(json.dumps({"status": result["status"], "selected_by_validation": result["selected_by_validation_log_loss"],
                      "test_scores": {k: {m: v["test"][m] for m in ("accuracy", "log_loss", "brier", "fights")}
                                      for k, v in result["scores"].items()}}, indent=2))


if __name__ == "__main__":
    main()
