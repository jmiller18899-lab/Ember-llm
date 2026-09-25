"""Run an offline CPU comparison of continuous, BSQ, and hybrid forecasts.

python -m jobs.ember_bsq_train --smoke --output-dir /tmp/ember-bsq-smoke
No EmberGPT checkpoint is loaded or updated by this experiment.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from jobs.ember_bsq import BSQForecaster
from jobs.ember_bsq_data import chronological_split, load_examples, write_smoke_data

ROOT = Path(__file__).resolve().parents[1]
MODES = ("continuous", "bsq", "hybrid")


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "target_name", "task", "horizon_seconds", "feature_names",
                "history_length", "bits", "hidden_size", "seed", "max_steps", "batch_size",
                "learning_rate", "eval_every", "reconstruction_weight", "commitment_weight", "entropy_weight"}
    if set(cfg) != required:
        raise ValueError(f"config keys differ: {sorted(set(cfg) ^ required)}")
    if cfg["schema_version"] != "ember-bsq-forecast-v1" or cfg["task"] not in {"binary", "regression"}:
        raise ValueError("unsupported schema_version or task")
    if not isinstance(cfg["target_name"], str) or not cfg["target_name"].strip():
        raise ValueError("target_name must describe the prediction target")
    names = cfg["feature_names"]
    if not isinstance(names, list) or not 1 <= len(names) <= 64:
        raise ValueError("feature_names must contain 1 to 64 names")
    if any(not isinstance(n, str) or not n.strip() for n in names) or len(set(names)) != len(names):
        raise ValueError("feature_names must be unique nonempty strings")
    for name, low, high in (("history_length", 1, 64), ("bits", 2, 32), ("hidden_size", 4, 128),
                            ("max_steps", 1, 1000), ("batch_size", 1, 256), ("eval_every", 1, 1000),
                            ("seed", 0, 2**31 - 1)):
        if type(cfg[name]) is not int or not low <= cfg[name] <= high:
            raise ValueError(f"{name} must be an integer between {low} and {high}")
    for name in ("horizon_seconds", "learning_rate", "reconstruction_weight", "commitment_weight", "entropy_weight"):
        value = cfg[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if cfg["horizon_seconds"] <= 0 or not 0 < cfg["learning_rate"] <= .1:
        raise ValueError("horizon must be positive and learning_rate must be in (0, 0.1]")
    return cfg


def tensors(rows):
    x = torch.tensor([r.features for r in rows], dtype=torch.float32)
    y = torch.tensor([r.target for r in rows], dtype=torch.float32)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("numeric values overflow float32")
    return x, y


def metrics(prediction: torch.Tensor, target: torch.Tensor, task: str) -> dict:
    if not torch.isfinite(prediction).all():
        raise ValueError("non-finite model predictions")
    if task == "regression":
        return {"mse": float(F.mse_loss(prediction, target)),
                "mae": float(F.l1_loss(prediction, target))}
    probability = prediction.sigmoid()
    bins = []
    for i in range(5):
        mask = (probability >= i / 5) & ((probability < (i + 1) / 5) if i < 4 else (probability <= 1))
        if mask.any():
            bins.append({"lower": i / 5, "upper": (i + 1) / 5, "count": int(mask.sum()),
                         "mean_probability": float(probability[mask].mean()),
                         "observed_frequency": float(target[mask].mean())})
    return {"log_loss": float(F.binary_cross_entropy_with_logits(prediction, target)),
            "brier": float(((probability - target) ** 2).mean()),
            "accuracy": float(((probability >= .5) == target.bool()).float().mean()),
            "reliability_bins": bins}


def fit_one(cfg, mode, partitions, normalization, output_dir, symmetric_binary=False):
    torch.manual_seed(cfg["seed"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    model_args = {"feature_count": len(cfg["feature_names"]), "bits": cfg["bits"],
                  "hidden_size": cfg["hidden_size"], "mode": mode}
    model = BSQForecaster(**model_args).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=.01)
    train_x, train_y = partitions[0]
    val_x, val_y = partitions[1]
    test_x, test_y = partitions[2]
    y_mean, y_scale = normalization["target_mean"], normalization["target_scale"]

    def forward(x):
        prediction, auxiliary = model(x)
        if symmetric_binary:
            swapped = -x - 2 * normalization["feature_mean"] / normalization["feature_scale"]
            other_prediction, other_auxiliary = model(swapped)
            prediction = (prediction - other_prediction) / 2
            auxiliary = {k: (v + other_auxiliary[k]) / 2 for k, v in auxiliary.items()}
        return prediction, auxiliary

    def forecast_loss(pred, target):
        if cfg["task"] == "binary":
            return F.binary_cross_entropy_with_logits(pred, target)
        return F.mse_loss(pred, (target - y_mean) / y_scale)

    best_loss, best_step, best_state = math.inf, 0, None
    for step in range(1, cfg["max_steps"] + 1):
        model.train()
        index = torch.randint(len(train_x), (min(cfg["batch_size"], len(train_x)),), generator=generator)
        prediction, auxiliary = forward(train_x[index])
        loss = forecast_loss(prediction, train_y[index])
        for name in ("reconstruction", "commitment", "entropy"):
            loss = loss + cfg[name + "_weight"] * auxiliary[name]
        if not torch.isfinite(loss):
            raise ValueError(f"{mode}: non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
            model.eval()
            with torch.no_grad():
                validation_loss = float(forecast_loss(forward(val_x)[0], val_y))
            if math.isfinite(validation_loss) and validation_loss < best_loss:
                best_loss, best_step = validation_loss, step
                best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise ValueError(f"{mode}: no finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_prediction = forward(val_x)[0] * y_scale + y_mean
        test_prediction = forward(test_x)[0] * y_scale + y_mean
    checkpoint = {"schema_version": cfg["schema_version"], "model_args": model_args,
                  "model_state_dict": best_state, "normalization": normalization,
                  "config": cfg, "selected_step": best_step, "symmetric_binary": symmetric_binary}
    torch.save(checkpoint, output_dir / f"{mode}.pt")
    return {"parameters": sum(p.numel() for p in model.parameters()), "selected_step": best_step,
            "validation": metrics(validation_prediction, val_y, cfg["task"]),
            "test": metrics(test_prediction, test_y, cfg["task"])}


def predict_checkpoint(path: Path, features, feature_names: list[str]) -> dict:
    """Local numeric inference; probabilities are model estimates, not calibrated guarantees."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["schema_version"] != "ember-bsq-forecast-v1":
        raise ValueError("unsupported checkpoint schema")
    cfg, norm = checkpoint["config"], checkpoint["normalization"]
    if feature_names != cfg["feature_names"]:
        raise ValueError("feature_names and their order must match the trained model")
    x = torch.tensor(features, dtype=torch.float32)
    if list(x.shape) != [cfg["history_length"], len(feature_names)] or not torch.isfinite(x).all():
        raise ValueError("provide a finite history matrix with the trained dimensions")
    model = BSQForecaster(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        normalized = ((x - norm["feature_mean"]) / norm["feature_scale"]).unsqueeze(0)
        raw = model(normalized)[0][0]
        if checkpoint.get("symmetric_binary", False):
            swapped = -normalized - 2 * norm["feature_mean"] / norm["feature_scale"]
            raw = (raw - model(swapped)[0][0]) / 2
        if not torch.isfinite(raw):
            raise ValueError("non-finite prediction")
        value = float(raw.sigmoid()) if cfg["task"] == "binary" else float(raw * norm["target_scale"] + norm["target_mean"])
    return {"target_name": cfg["target_name"], "horizon_seconds": cfg["horizon_seconds"],
            "probability" if cfg["task"] == "binary" else "prediction": value,
            "status": "experimental_uncalibrated"}


def run(cfg: dict, data: Path, output_dir: Path, data_kind: str = "user_supplied_unverified",
        boundaries: tuple[float, float] | None = None, symmetric_binary=False) -> dict:
    torch.set_num_threads(1)
    if symmetric_binary and cfg["task"] != "binary":
        raise ValueError("sign-swap symmetry requires a binary difference-feature task")
    rows = load_examples(data, cfg)
    parts, split = chronological_split(rows, boundaries)
    if cfg["task"] == "binary" and {r.target for r in parts[0]} != {0., 1.}:
        raise ValueError("binary training examples must contain both classes")
    raw_parts = [tensors(part) for part in parts]
    train_x, train_y = raw_parts[0]
    normalization = {"feature_mean": train_x.mean((0, 1)),
                     "feature_scale": train_x.std((0, 1), unbiased=False).clamp_min(1e-6),
                     "target_mean": float(train_y.mean()) if cfg["task"] == "regression" else 0.,
                     "target_scale": float(train_y.std(unbiased=False).clamp_min(1e-6)) if cfg["task"] == "regression" else 1.}
    if not all(torch.isfinite(v).all() if isinstance(v, torch.Tensor) else math.isfinite(v)
               for v in normalization.values()):
        raise ValueError("non-finite training normalization")
    prepared = [((x - normalization["feature_mean"]) / normalization["feature_scale"], y)
                for x, y in raw_parts]
    if any(not torch.isfinite(x).all() for x, _ in prepared):
        raise ValueError("non-finite standardized features")
    # Refuse to overwrite prior evidence or any existing checkpoint directory.
    output_dir.mkdir(parents=True, exist_ok=False)
    models = {mode: fit_one(cfg, mode, prepared, normalization, output_dir, symmetric_binary) for mode in MODES}
    constant = train_y.mean()
    if cfg["task"] == "binary":
        constant = torch.logit(constant.clamp(1e-6, 1 - 1e-6))
    baseline = {name: metrics(torch.full_like(y, constant), y, cfg["task"])
                for name, (_, y) in zip(("train", "validation", "test"), raw_parts)}
    report = {"status": "completed", "data_kind": data_kind, "schema_version": cfg["schema_version"],
              "real_world_quality": "not_established", "ember_gpt_weights_updated": False,
              "production_integrated": False, "device": "cpu", "config": cfg,
              "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
              "split": split, "symmetric_binary": symmetric_binary, "constant_baseline": baseline, "models": models,
              "selection_rule": "checkpoint selected by validation forecast loss; test reported only afterward",
              "limitations": ["one time split and one seed; repeat on fresh periods before adoption",
                              "probabilities are uncalibrated; reliability bins describe this test only",
                              "availability declarations cannot detect hidden leakage inside source features"]}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/ember_bsq_forecast.json")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path)
    source.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="ember-bsq-") as temporary:
            data = Path(temporary) / "synthetic.jsonl"
            write_smoke_data(data, cfg)
            report = run(cfg, data, args.output_dir, "synthetic_smoke")
    else:
        report = run(cfg, args.data, args.output_dir)
    print(json.dumps({"status": report["status"], "data_kind": report["data_kind"],
                      "real_world_quality": report["real_world_quality"],
                      "report": str(args.output_dir / "report.json")}))


if __name__ == "__main__":
    main()
