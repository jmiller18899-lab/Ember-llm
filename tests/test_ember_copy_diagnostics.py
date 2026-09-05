from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "jobs" / "ember_hf_diagnose_copy.py"


def load_module():
    spec = importlib.util.spec_from_file_location("ember_hf_diagnose_copy", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cases_cover_multiple_copy_shapes_and_are_unique():
    module = load_module()
    ids = [row["id"] for row in module.CASES]
    values = [row["value"] for row in module.CASES]
    alternates = [row["alternate"] for row in module.CASES]
    assert len(module.CASES) >= 8
    assert len(ids) == len(set(ids))
    assert len(values) == len(set(values))
    assert not (set(values) & set(alternates))
    assert {"short_code", "digits", "model_id", "url", "path", "expression"}.issubset(ids)


def test_prompt_contains_target_and_distractors_without_leaking_alternate():
    module = load_module()
    for row in module.CASES:
        prompt = module.prompt_for(row["value"])
        assert row["value"] in prompt
        assert row["alternate"] not in prompt
        assert "old=K2P8" in prompt
        assert "fallback=77291" in prompt
        assert "TARGET=" in prompt


def test_normalize_generated_stops_at_endoftext():
    module = load_module()
    assert module.normalize_generated("Q7M4\n<|endoftext|>\nnoise") == "Q7M4"
    assert module.normalize_generated("  Q7M4  ") == "Q7M4"


def fake_model(summary):
    return {"summary": summary}


def base_summary(**overrides):
    payload = {
        "tokenizer_roundtrip_rate": 1.0,
        "mean_tokens_per_char": 0.5,
        "exact_copy_rate": 0.1,
        "target_containment_rate": 0.1,
        "clean_stop_rate": 1.0,
        "first_token_top1_rate": 0.0,
        "first_token_top5_rate": 0.2,
        "first_token_top20_rate": 0.4,
        "mean_first_token_rank": 40.0,
        "mean_top1_logprob_gap_over_expected": 2.0,
        "teacher_forced_expected_win_rate": 0.4,
        "mean_nll_margin_corrupt_minus_expected": -0.2,
    }
    payload.update(overrides)
    return payload


def test_classifier_flags_foundation_when_correct_completion_is_not_preferred():
    module = load_module()
    baseline = fake_model(base_summary())
    candidate = fake_model(base_summary(teacher_forced_expected_win_rate=0.55, mean_nll_margin_corrupt_minus_expected=-0.05))
    result = module.classify(baseline, candidate)
    assert result["verdict"] == "conditioning_or_foundation_bottleneck"
    assert "pretraining" in result["next_step"]


def test_classifier_flags_dominant_response_prior_when_copy_signal_exists_but_start_is_not_competitive():
    module = load_module()
    baseline = fake_model(base_summary())
    candidate = fake_model(base_summary(
        teacher_forced_expected_win_rate=0.9,
        mean_nll_margin_corrupt_minus_expected=0.8,
        first_token_top5_rate=0.3,
        exact_copy_rate=0.0,
    ))
    result = module.classify(baseline, candidate)
    assert result["verdict"] == "copy_signal_overwhelmed_by_response_prior"
    assert "literal-copy warmup" in result["next_step"]


def test_classifier_separates_continuation_control_when_copy_start_is_competitive():
    module = load_module()
    baseline = fake_model(base_summary())
    candidate = fake_model(base_summary(
        teacher_forced_expected_win_rate=0.9,
        mean_nll_margin_corrupt_minus_expected=0.8,
        first_token_top5_rate=0.9,
        exact_copy_rate=0.3,
    ))
    result = module.classify(baseline, candidate)
    assert result["verdict"] == "sequence_decoding_after_copy_start_bottleneck"


def test_classifier_accepts_healthy_copy_mechanism():
    module = load_module()
    baseline = fake_model(base_summary())
    candidate = fake_model(base_summary(
        teacher_forced_expected_win_rate=1.0,
        mean_nll_margin_corrupt_minus_expected=1.2,
        first_token_top5_rate=1.0,
        exact_copy_rate=0.9,
        target_containment_rate=0.9,
    ))
    result = module.classify(baseline, candidate)
    assert result["verdict"] == "copy_mechanism_healthy"


def test_checkpoint_prefers_nested_best_checkpoint():
    module = load_module()
    files = ["best.pt", "checkpoints/run-a/best.pt", "checkpoints/run-b/best.pt", "latest.pt"]
    assert module.checkpoint_path(files) == "checkpoints/run-b/best.pt"
