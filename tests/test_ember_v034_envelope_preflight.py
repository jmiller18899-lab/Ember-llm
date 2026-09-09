from __future__ import annotations

from collections import Counter
import copy
import json

import pytest

from jobs import ember_envelope_preflight_v034 as preflight

CFG = json.loads(preflight.DEFAULT_CONFIG.read_text())


def completion(name: str, key: str, value: str) -> str:
    return f'<|tool|>\n{{"name":"{name}","arguments":{{"{key}":"{value}"}}}}\n<|endoftext|>\n'


def test_config_is_preflight_only_and_gates_the_baseline_envelope():
    cfg = preflight.load_config(preflight.DEFAULT_CONFIG)
    assert cfg["version"] == "0.0.34"
    assert cfg["expected_cases"] == 90
    assert cfg["training_authorized"] is False
    assert cfg["gpu_training_authorized"] is False
    assert cfg["production_authorized"] is False
    assert cfg["baseline_gate"] == {
        "minimum_envelope_json_valid_rate": 0.95,
        "minimum_envelope_tool_name_rate": 0.95,
    }


def test_held_out_battery_is_90_balanced_agent_style_cases():
    cases = preflight.build_cases()
    assert len(cases) == 90
    assert Counter(case["kind"] for case in cases) == dict.fromkeys(preflight.copy_data.KINDS, 10)
    assert set(preflight.TOOL_BY_KIND) == set(preflight.copy_data.KINDS)
    for case in cases:
        assert case["prompt"].startswith("<|system|>\nYou are Ember.")
        assert "\n<|user|>\n" in case["prompt"]
        assert case["prompt"].endswith("\n<|assistant|>\n")
        assert "TARGET=" not in case["prompt"]
        assert "Reply with TARGET" not in case["prompt"]
        assert case["target"] in case["prompt"]
        assert case["old"] in case["prompt"]
        assert case["fallback"] in case["prompt"]


def test_tool_mapping_keeps_value_kinds_semantically_paired():
    assert preflight.TOOL_BY_KIND["entity"] == ("weather", "location")
    assert preflight.TOOL_BY_KIND["expression"] == ("calculator", "expression")
    assert preflight.TOOL_BY_KIND["digits"] == ("calculator", "expression")
    assert preflight.TOOL_BY_KIND["model_id"] == ("web_search", "query")
    assert preflight.TOOL_BY_KIND["short_code"] == ("web_search", "query")
    assert preflight.TOOL_BY_KIND["url"] == ("fetch_url", "url")
    assert preflight.TOOL_BY_KIND["path"] == ("read_file", "path")
    assert preflight.TOOL_BY_KIND["long_code"] == ("lookup", "key")
    assert preflight.TOOL_BY_KIND["mixed"] == ("lookup", "key")


def test_right_envelope_and_tool_with_wrong_value_is_not_slot_exact():
    case = next(c for c in preflight.build_cases() if c["kind"] == "entity")
    score = preflight.score_case(case, completion(case["expected_tool"], case["argument_key"], "Wrong Place"))
    assert score["envelope_json_valid"] is True
    assert score["tool_name_correct"] is True
    assert score["slot_exact"] is False
    assert score["right_envelope_tool_wrong_value"] is True


def test_exact_value_in_expected_slot_is_slot_exact():
    case = next(c for c in preflight.build_cases() if c["kind"] == "path")
    score = preflight.score_case(case, completion(case["expected_tool"], case["argument_key"], case["target"]))
    assert score["envelope_json_valid"] is True
    assert score["tool_name_correct"] is True
    assert score["slot_exact"] is True
    assert score["right_envelope_tool_wrong_value"] is False


def test_wrong_tool_and_invalid_json_are_not_slot_evidence():
    case = next(c for c in preflight.build_cases() if c["kind"] == "expression")
    wrong_tool = preflight.score_case(case, completion("weather", case["argument_key"], case["target"]))
    assert wrong_tool["envelope_json_valid"] is True
    assert wrong_tool["tool_name_correct"] is False
    assert wrong_tool["slot_exact"] is False
    broken = preflight.score_case(case, "<|tool|>\n{not json}\n<|endoftext|>\n")
    assert broken["envelope_json_valid"] is False
    assert broken["tool_name_correct"] is False
    assert broken["slot_exact"] is False


def scored_rows(json_valid=90, tool_correct=90, slot_exact=0):
    cases = preflight.build_cases()
    rows = []
    for i, case in enumerate(cases):
        valid = i < json_valid
        correct = valid and i < tool_correct
        exact = correct and i < slot_exact
        rows.append({
            "id": case["id"],
            "kind": case["kind"],
            "score": {
                "envelope_json_valid": valid,
                "tool_name_correct": correct,
                "slot_exact": exact,
                "right_envelope_tool_wrong_value": correct and not exact,
                "clean_stop": True,
            },
        })
    return rows


def test_baseline_gate_requires_high_json_and_tool_name_rates_before_slot_exact():
    passing = preflight.summarize(scored_rows(json_valid=86, tool_correct=86, slot_exact=4), CFG)
    assert passing["passed"] is True
    assert passing["metrics"]["slot_evaluable"] == 86
    assert passing["metrics"]["slot_exact"] == 4
    assert passing["metrics"]["slot_exact_rate"] == pytest.approx(4 / 86)
    json_fail = preflight.summarize(scored_rows(json_valid=85, tool_correct=85), CFG)
    assert json_fail["passed"] is False
    assert json_fail["checks"]["baseline_envelope_json_valid"] is False
    tool_fail = preflight.summarize(scored_rows(json_valid=90, tool_correct=85), CFG)
    assert tool_fail["passed"] is False
    assert tool_fail["checks"]["baseline_envelope_tool_name"] is False


def test_future_phase_cannot_enable_training_in_this_config(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    cfg = copy.deepcopy(CFG)
    cfg["training_authorized"] = True
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(preflight, "DEFAULT_CONFIG", path)
    with pytest.raises(ValueError, match="preflight-only"):
        preflight.load_config(path)


def test_summary_explicitly_says_no_training_or_gpu():
    gate = preflight.summarize(scored_rows(), CFG)
    report = {"status": "PASS", "baseline_gate": gate}
    text = preflight.summary_markdown(report)
    assert "No optimizer was created" in text
    assert "no training, GPU submission, promotion, or production integration" in text
    assert "slot_exact" in text
