from __future__ import annotations

from collections import Counter
import copy
import json

import pytest

from jobs import ember_envelope_preflight_v036 as preflight

CFG = json.loads(preflight.DEFAULT_CONFIG.read_text())


def test_config_is_preflight_only_with_two_envelope_gates():
    cfg = preflight.load_config(preflight.DEFAULT_CONFIG)
    assert cfg["version"] == "0.0.36"
    assert cfg["expected_reference_cases"] == 4
    assert cfg["expected_cases"] == 90
    assert cfg["reference_gate"] == {
        "minimum_envelope_json_valid_rate": 1.0,
        "minimum_envelope_tool_name_rate": 1.0,
    }
    assert cfg["baseline_gate"] == {
        "minimum_envelope_json_valid_rate": 0.95,
        "minimum_envelope_tool_name_rate": 0.95,
    }
    assert cfg["training_authorized"] is False
    assert cfg["gpu_training_authorized"] is False
    assert cfg["production_authorized"] is False


def test_reference_probe_is_exact_four_v008_tool_prompts():
    cases = preflight.reference_cases(CFG)
    assert [c["id"] for c in cases] == [
        "tool_weather", "tool_calculator", "tool_web_search", "tool_get_time"
    ]
    assert [c["expected_tool"] for c in cases] == [
        "weather", "calculator", "web_search", "get_time"
    ]
    assert "What is the weather in Detroit right now?" in cases[0]["prompt"]
    assert "Calculate 347 multiplied by 28." in cases[1]["prompt"]
    assert "Find the latest published release of Python." in cases[2]["prompt"]
    assert "What time is it in Tokyo?" in cases[3]["prompt"]


def test_90_case_probe_has_one_requested_value_and_no_added_distractors():
    cases = preflight.build_cases()
    assert len(cases) == 90
    assert Counter(c["kind"] for c in cases) == dict.fromkeys(preflight.copy_data.KINDS, 10)
    diagnostic = {(kind, value): corrupt for kind, value, corrupt in preflight.copy_data.DIAGNOSTICS}
    all_values = [value for _kind, value, _corrupt in preflight.copy_data.DIAGNOSTICS]
    for case in cases:
        assert case["target"] in case["prompt"]
        assert diagnostic[(case["kind"], case["target"])] not in case["prompt"]
        assert "TARGET=" not in case["prompt"]
        assert "fallback" not in case["prompt"].lower()
        assert "Ignore " not in case["prompt"]
        assert case["prompt"].endswith("\n<|assistant|>\n")


def test_tool_mapping_stays_inside_demonstrated_v008_vocabulary_for_90_cases():
    assert {tool for tool, _ in preflight.TOOL_BY_KIND.values()} == {
        "weather", "calculator", "web_search"
    }


def test_reference_scoring_separates_json_and_tool_name():
    case = {"expected_tool": "weather"}
    exact = '<|tool|>\n{"name":"weather","arguments":{"location":"Detroit"}}\n<|endoftext|>\n'
    wrong = '<|tool|>\n{"name":"calculator","arguments":{"expression":"1+1"}}\n<|endoftext|>\n'
    broken = 'Detroit\n<|endoftext|>\n'
    assert preflight.score_reference(case, exact)["tool_name_correct"] is True
    wrong_score = preflight.score_reference(case, wrong)
    assert wrong_score["envelope_json_valid"] is True
    assert wrong_score["tool_name_correct"] is False
    assert preflight.score_reference(case, broken)["envelope_json_valid"] is False


def test_reference_gate_requires_all_four_cases():
    rows = []
    for i in range(4):
        rows.append({"score": {"envelope_json_valid": True, "tool_name_correct": i < 4, "clean_stop": True}})
    assert preflight.reference_summary(rows, CFG)["passed"] is True
    rows[-1]["score"]["tool_name_correct"] = False
    assert preflight.reference_summary(rows, CFG)["passed"] is False


def test_90_case_baseline_gate_still_requires_86_of_90():
    cases = preflight.build_cases()
    def rows(valid, correct):
        out = []
        for i, case in enumerate(cases):
            is_valid = i < valid
            is_correct = is_valid and i < correct
            out.append({
                "id": case["id"], "kind": case["kind"],
                "score": {
                    "envelope_json_valid": is_valid,
                    "tool_name_correct": is_correct,
                    "slot_exact": False,
                    "right_envelope_tool_wrong_value": is_correct,
                    "clean_stop": True,
                },
            })
        return out
    assert preflight.baseline_summary(rows(86, 86), CFG)["passed"] is True
    assert preflight.baseline_summary(rows(85, 85), CFG)["passed"] is False
    assert preflight.baseline_summary(rows(90, 85), CFG)["passed"] is False


def test_training_authorization_cannot_be_enabled(tmp_path, monkeypatch):
    cfg = copy.deepcopy(CFG)
    cfg["training_authorized"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(preflight, "DEFAULT_CONFIG", path)
    with pytest.raises(ValueError, match="preflight-only"):
        preflight.load_config(path)
