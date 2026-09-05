from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "jobs" / "ember_sft_data_v015.py"
TRAINER = ROOT / "jobs" / "ember_hf_sft_v015.py"
CONFIG = ROOT / "config" / "ember_copy_warmup_v0.0.15.json"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_is_narrow_first_token_phase_from_v009():
    cfg = json.loads(CONFIG.read_text())
    assert cfg["version"] == "0.0.15"
    assert cfg["phase"] == "first-token-literal-copy-warmup"
    assert cfg["source_model_name"] == "ember-v0.0.9-t4"
    assert cfg["first_token_weight"] > cfg["copy_token_weight"] > cfg["eos_token_weight"]
    assert cfg["minimum_first_token_top20_rate"] >= 7 / 9
    assert cfg["minimum_first_token_top5_rate"] >= 5 / 9
    assert cfg["minimum_exact_copy_rate"] >= 5 / 9


def test_dataset_is_diverse_direct_copy_only_and_held_out_clean():
    data = load(DATA, "ember_sft_data_v015")
    train = data.build_examples("train", 900)
    val = data.build_examples("validation", 180)
    data.assert_clean(train, val)
    assert {row["kind"] for row in train} == set(data.KINDS)
    for row in train + val:
        assert row["completion"].startswith(row["value"])
        assert row["completion"].endswith("<|endoftext|>\n")
        assert row["prompt"].count(row["value"]) == 1
        assert "<|tool|>" not in row["completion"]
        assert row["value"] not in data.HELD_OUT_VALUES


def test_train_validation_values_do_not_overlap():
    data = load(DATA, "ember_sft_data_v015_overlap")
    train = data.build_examples("train", 3600)
    val = data.build_examples("validation", 450)
    assert not ({r["value"] for r in train} & {r["value"] for r in val})


def test_diagnostic_cases_match_cpu_suite_shapes():
    trainer = load(TRAINER, "ember_hf_sft_v015")
    values = [value for value, _ in trainer.DIAGNOSTICS]
    assert len(values) == 9
    assert "Q7M4" in values
    assert "openai/gpt-6-astra" in values
    assert "https://example.test/a7Q9" in values
    assert "/tmp/ember/Q7M4/result.json" in values
    assert "53*19+7" in values


def test_checkpoint_score_prioritizes_behavior_over_loss():
    trainer = load(TRAINER, "ember_hf_sft_v015_score")
    weak = {"metrics": {"first_token_top20_rate": 0.2, "first_token_top5_rate": 0.1, "exact_copy_rate": 0.0,
                         "teacher_forced_expected_win_rate": 1.0, "mean_first_token_rank": 50.0}}
    strong = {"metrics": {"first_token_top20_rate": 0.8, "first_token_top5_rate": 0.6, "exact_copy_rate": 0.6,
                           "teacher_forced_expected_win_rate": 0.9, "mean_first_token_rank": 4.0}}
    assert trainer.score(strong, 2.0) > trainer.score(weak, 0.01)


def test_asset_pin_is_immutable_commit_and_not_branch_name():
    trainer = load(TRAINER, "ember_hf_sft_v015_pin")
    assert len(trainer.ASSET_PIN) == 40
    int(trainer.ASSET_PIN, 16)
    assert trainer.SOURCE_SHA256 == "8299d52e8a852b9bd3e8403e086b48fd42d2babfd51c74eb94af29bd87ef2d13"
