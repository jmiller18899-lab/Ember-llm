from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import pytest
import torch

from jobs.ember_bsq import BinarySphericalQuantizer
from jobs.ember_bsq_data import chronological_split, load_examples, write_smoke_data
from jobs.ember_bsq_train import load_config, predict_checkpoint, run

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dataset(tmp_path):
    cfg = load_config(ROOT / "config/ember_bsq_forecast.json")
    path = tmp_path / "data.jsonl"
    write_smoke_data(path, cfg)
    return cfg, path


def test_bsq_unit_sphere_and_learning_gradient():
    latent = torch.tensor([[.2, -.9, .3, -.1], [.8, .2, -.5, -.7]], requires_grad=True)
    code, commitment, entropy = BinarySphericalQuantizer(4)(latent)
    assert torch.allclose(code.norm(dim=-1), torch.ones(2))
    assert torch.allclose(code.abs(), torch.full_like(code, .5))
    loss = (code * torch.arange(4)).sum() + commitment + .01 * entropy
    loss.backward()
    assert torch.isfinite(latent.grad).all()
    assert latent.grad.abs().sum() > 0


def test_zero_latent_is_a_valid_finite_code():
    latent = torch.zeros(2, 8, requires_grad=True)
    code, commitment, entropy = BinarySphericalQuantizer(8)(latent)
    assert torch.allclose(code, torch.full_like(code, 1 / math.sqrt(8)))
    (code.sum() + commitment + entropy).backward()
    assert torch.isfinite(latent.grad).all()


@pytest.mark.parametrize("mutation, message", [
    ({"label_available_at": "2020-01-01T00:00:00Z"}, "forecast horizon"),
    ({"observed_at": ["2030-01-01T00:00:00Z", "2030-01-01T00:01:00Z",
                       "2030-01-01T00:02:00Z", "2030-01-01T00:03:00Z"]}, "future observations"),
    ({"as_of": "2025-01-01T00:00:00"}, "timezone"),
    ({"target": .5}, "binary targets"),
    ({"features": [[float("nan"), 0]] * 4}, "finite numbers"),
    ({"feature_names": ["context", "signal"]}, "feature_names"),
])
def test_reject_invalid_or_leaky_rows(dataset, mutation, message):
    cfg, path = dataset
    lines = path.read_text().splitlines()
    row = json.loads(lines[0])
    row.update(mutation)
    lines[0] = json.dumps(row)
    path.write_text("\n".join(lines))
    with pytest.raises(ValueError, match=message):
        load_examples(path, cfg)


def test_split_purges_unavailable_labels_and_keeps_equal_times_together(dataset):
    cfg, path = dataset
    rows = load_examples(path, cfg)
    rows[0] = replace(rows[0], label_available_at=rows[-1].label_available_at)
    # A tied prediction timestamp may never land on both sides of a boundary.
    rows.append(replace(rows[144], sample_id="tie", event_id="tie"))
    parts, counts = chronological_split(list(reversed(rows)))
    assert counts["purged"] >= 1
    assert all(r.sample_id != rows[0].sample_id for r in parts[0])
    for earlier, later in zip(parts, parts[1:]):
        assert max(r.label_available_at for r in earlier) < min(r.as_of for r in later)
        assert not ({r.as_of for r in earlier} & {r.as_of for r in later})


def test_related_events_cannot_cross_splits(dataset):
    cfg, path = dataset
    rows = load_examples(path, cfg)
    rows[-1] = replace(rows[-1], event_id=rows[0].event_id)
    with pytest.raises(ValueError, match="same event crosses"):
        chronological_split(rows)


@pytest.mark.parametrize("task", ["binary", "regression"])
def test_training_learns_synthetic_signal_and_checkpoint_roundtrip(tmp_path, task):
    cfg = load_config(ROOT / "config/ember_bsq_forecast.json")
    cfg["task"] = task
    path = tmp_path / "examples.jsonl"
    write_smoke_data(path, cfg)
    output = tmp_path / "results"
    report = run(cfg, path, output, "synthetic_smoke")
    score = "log_loss" if task == "binary" else "mse"
    assert report["models"]["bsq"]["test"][score] < report["constant_baseline"]["test"][score]
    assert report["real_world_quality"] == "not_established"
    assert report["ember_gpt_weights_updated"] is False
    row = json.loads(path.read_text().splitlines()[-1])
    for mode in ("continuous", "bsq", "hybrid"):
        a = predict_checkpoint(output / f"{mode}.pt", row["features"], cfg["feature_names"])
        b = predict_checkpoint(output / f"{mode}.pt", row["features"], cfg["feature_names"])
        assert a == b
        assert math.isfinite(a["probability" if task == "binary" else "prediction"])
        if task == "binary":
            assert 0 <= a["probability"] <= 1
    with pytest.raises(FileExistsError):
        run(cfg, path, output)


def test_normalization_uses_only_training_rows(dataset, tmp_path):
    cfg, path = dataset
    rows = load_examples(path, cfg)
    parts, _ = chronological_split(rows)
    expected = torch.tensor([r.features for r in parts[0]]).mean((0, 1))
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    later_ids = {r.sample_id for r in parts[1] + parts[2]}
    for row in lines:
        if row["sample_id"] in later_ids:
            row["features"] = [[v + 100 for v in observation] for observation in row["features"]]
    path.write_text("\n".join(json.dumps(row) for row in lines))
    cfg["max_steps"] = 1
    output = tmp_path / "normalized"
    run(cfg, path, output)
    checkpoint = torch.load(output / "bsq.pt", weights_only=True)
    assert torch.equal(checkpoint["normalization"]["feature_mean"], expected)
