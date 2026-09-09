from __future__ import annotations

from collections import Counter
import copy
import json

import pytest

from jobs import ember_envelope_preflight_v035 as preflight

CFG = json.loads(preflight.DEFAULT_CONFIG.read_text())


def test_config_remains_cpu_preflight_only():
    cfg = preflight.load_config(preflight.DEFAULT_CONFIG)
    assert cfg["version"] == "0.0.35"
    assert cfg["expected_cases"] == 90
    assert cfg["training_authorized"] is False
    assert cfg["gpu_training_authorized"] is False
    assert cfg["production_authorized"] is False
    assert cfg["baseline_gate"] == {
        "minimum_envelope_json_valid_rate": 0.95,
        "minimum_envelope_tool_name_rate": 0.95,
    }


def test_only_v008_demonstrated_tool_vocabulary_is_used():
    assert {tool for tool, _ in preflight.TOOL_BY_KIND.values()} == {
        "weather", "calculator", "web_search"
    }
    assert preflight.TOOL_BY_KIND["entity"] == ("weather", "location")
    assert preflight.TOOL_BY_KIND["digits"] == ("calculator", "expression")
    assert preflight.TOOL_BY_KIND["expression"] == ("calculator", "expression")
    for kind in ("short_code", "long_code", "model_id", "url", "path", "mixed"):
        assert preflight.TOOL_BY_KIND[kind] == ("web_search", "query")


def test_system_lines_are_exact_v008_agent_lines():
    assert preflight.SYSTEM_BY_TOOL == {
        "weather": "You are Ember. When current weather is requested, call the weather tool with JSON arguments.",
        "calculator": "You are Ember. Use the calculator tool for arithmetic and provide JSON arguments.",
        "web_search": "You are Ember. Use web_search when the user requests current information.",
    }


def test_90_cases_are_balanced_natural_and_keep_two_distractors():
    cases = preflight.build_cases()
    assert len(cases) == 90
    assert Counter(c["kind"] for c in cases) == dict.fromkeys(preflight.copy_data.KINDS, 10)
    for case in cases:
        assert case["prompt"].startswith("<|system|>\nYou are Ember.")
        assert case["prompt"].endswith("\n<|assistant|>\n")
        assert "TARGET=" not in case["prompt"]
        assert "Reply with TARGET" not in case["prompt"]
        assert case["target"] in case["prompt"]
        assert case["old"] in case["prompt"]
        assert case["fallback"] in case["prompt"]
        assert len({case["target"], case["old"], case["fallback"]}) == 3


def test_scoring_preserves_right_envelope_wrong_value_measurement():
    case = next(c for c in preflight.build_cases() if c["kind"] == "entity")
    completion = '<|tool|>\n{"name":"weather","arguments":{"location":"Detroit"}}\n<|endoftext|>\n'
    score = preflight.score_case(case, completion)
    assert score["envelope_json_valid"] is True
    assert score["tool_name_correct"] is True
    if case["target"] == "Detroit":
        assert score["slot_exact"] is True
    else:
        assert score["slot_exact"] is False
        assert score["right_envelope_tool_wrong_value"] is True


def _rows(json_valid: int, tool_correct: int):
    cases = preflight.build_cases()
    out = []
    for i, case in enumerate(cases):
        valid = i < json_valid
        correct = valid and i < tool_correct
        out.append({
            "id": case["id"],
            "kind": case["kind"],
            "score": {
                "envelope_json_valid": valid,
                "tool_name_correct": correct,
                "slot_exact": False,
                "right_envelope_tool_wrong_value": correct,
                "clean_stop": True,
            },
        })
    return out


def test_baseline_gate_still_requires_86_of_90_for_both_metrics():
    assert preflight.summarize(_rows(86, 86), CFG)["passed"] is True
    assert preflight.summarize(_rows(85, 85), CFG)["passed"] is False
    assert preflight.summarize(_rows(90, 85), CFG)["passed"] is False


def test_training_cannot_be_enabled_in_this_phase(tmp_path, monkeypatch):
    cfg = copy.deepcopy(CFG)
    cfg["gpu_training_authorized"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(preflight, "DEFAULT_CONFIG", path)
    with pytest.raises(ValueError, match="preflight-only"):
        preflight.load_config(path)
