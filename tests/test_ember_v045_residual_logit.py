from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from jobs import ember_residual_logit_v045 as v045


ROOT = Path(__file__).resolve().parents[1]


def test_v045_is_diagnostic_only_and_pins_the_six_failures():
    cfg = v045.load_config(v045.DEFAULT_CONFIG)
    assert cfg["version"] == "0.0.45"
    assert cfg["training_authorized"] is False
    assert cfg["gpu_training_authorized"] is False
    assert cfg["production_authorized"] is False
    assert cfg["placement_learning_authorized"] is False
    assert cfg["expected_target_cases"] == 24
    assert cfg["expected_failures"] == 6
    assert set(cfg["target_subtypes"]) == v045.TARGET_SUBTYPES
    assert set(cfg["expected_failure_ids"]) == v045.EXPECTED_FAILURE_IDS


def test_v045_reuses_the_v044_prompt_stack_without_new_prompt_templates():
    frozen = v045.frozen_v044_cases()
    direct = v045.prior.build_final_cases(v045.V044_CONFIG, v045.V044_SYSTEM_SELECTION)
    assert len(frozen) == len(direct) == 90
    assert [(c["id"], c["prompt"]) for c in frozen] == [(c["id"], c["prompt"]) for c in direct]
    assert v045.prompt_digest(frozen) == v045.prompt_digest(direct)


def test_v045_target_cohort_is_all_cases_in_the_failure_bearing_subtypes():
    cases = v045.target_cases()
    assert len(cases) == 24
    assert len({c["id"] for c in cases}) == 24
    counts = {}
    for case in cases:
        counts[case["subtype"]] = counts.get(case["subtype"], 0) + 1
        assert case["subtype"] in v045.TARGET_SUBTYPES
        assert case["system_variant"] == "v037_schema_system"
    assert counts == {
        "short_code/len4": 5,
        "short_code/len5": 5,
        "long_code/4x4": 5,
        "long_code/3x5": 5,
        "path/plain_leaf": 4,
    }


def test_forced_tool_intervention_forces_only_one_marker_token():
    source = inspect.getsource(v045.forced_tool_completion)
    assert "generated = [tool_id]" in source
    assert "prompt_ids + [tool_id]" in source
    assert "case[\"target\"]" not in source
    assert "argument_key" not in source
    assert "optimizer" not in source
    assert ".backward(" not in source


def test_runner_contains_no_training_or_gpu_execution_path():
    source = (ROOT / "jobs/ember_residual_logit_v045.py").read_text(encoding="utf-8")
    forbidden = ["torch.optim", ".backward(", "optimizer.step", ".cuda(", "device=\"cuda\"", "to(\"cuda\")"]
    assert not any(token in source for token in forbidden)


def _row(case_id: str, valid: bool, rank: int, deficit: float, forced: bool):
    margin = -deficit if deficit else 0.5
    return {
        "id": case_id,
        "subtype": "short_code/len4",
        "target_token_count": 3,
        "baseline_score": {"envelope_json_valid": valid},
        "first_token": {
            "tool_rank": rank,
            "tool_margin_vs_best_other": margin,
            "tool_logit_deficit": deficit,
            "tool_probability": 0.2,
        },
        "forced_score": {"envelope_json_valid": forced, "tool_name_correct": forced},
    }


def test_interpretation_marks_near_boundary_only_when_rescue_and_logits_support_it():
    cfg = v045.load_config(v045.DEFAULT_CONFIG)
    failures = [
        _row(case_id, False, 2, 0.4, True)
        for case_id in sorted(v045.EXPECTED_FAILURE_IDS)
    ]
    passes = [_row(f"pass_{i}", True, 1, 0.0, True) for i in range(18)]
    result = v045.interpret(failures + passes, cfg)
    assert result["forced_rescue_count"] == 6
    assert result["near_boundary_count"] == 6
    assert result["regime"] == "downstream_intact_near_envelope_entry_boundary"


def test_interpretation_does_not_call_a_deep_rank_near_boundary():
    cfg = v045.load_config(v045.DEFAULT_CONFIG)
    failures = [
        _row(case_id, False, 20, 5.0, True)
        for case_id in sorted(v045.EXPECTED_FAILURE_IDS)
    ]
    passes = [_row(f"pass_{i}", True, 1, 0.0, True) for i in range(18)]
    result = v045.interpret(failures + passes, cfg)
    assert result["forced_rescue_count"] == 6
    assert result["near_boundary_count"] == 0
    assert result["competitive_count"] == 0
    assert result["regime"] == "downstream_intact_different_first_token_regime"


def test_config_rejects_any_attempt_to_authorize_learning(tmp_path):
    cfg = json.loads(v045.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    cfg["placement_learning_authorized"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError):
        v045.load_config(path)
