from __future__ import annotations

from collections import Counter
import copy
import json

import pytest

from jobs import ember_envelope_preflight_v037 as preflight
from jobs import ember_sft_data_semantic_v1 as semantic_data

CFG = json.loads(preflight.DEFAULT_CONFIG.read_text())


def test_config_is_cpu_preflight_only_with_95_percent_baseline_gate():
    cfg = preflight.load_config(preflight.DEFAULT_CONFIG)
    assert cfg["version"] == "0.0.37"
    assert cfg["expected_cases"] == 90 and cfg["expected_reference_cases"] == 4
    assert set(cfg["weak_kinds"]) == {"short_code", "long_code", "url", "path"}
    assert cfg["baseline_gate"] == {
        "minimum_envelope_json_valid_rate": 0.95,
        "minimum_envelope_tool_name_rate": 0.95,
    }
    assert cfg["training_authorized"] is False
    assert cfg["gpu_training_authorized"] is False
    assert cfg["production_authorized"] is False


def test_schema_system_line_is_exact_v032_semantic_curriculum_wording():
    for family in (
        "tool_call/weather_literal", "tool_call/calculator_literal", "tool_call/search_literal"
    ):
        tool, field = semantic_data.tool_spec(family.split("/", 1)[1])
        assert preflight.schema_system(tool, field) == semantic_data.system_for(family)


def test_90_cases_keep_single_natural_target_and_schema_line():
    cases = preflight.build_cases()
    assert len(cases) == 90
    assert Counter(c["kind"] for c in cases) == dict.fromkeys(preflight.copy_data.KINDS, 10)
    diagnostic = {(kind, value): corrupt for kind, value, corrupt in preflight.copy_data.DIAGNOSTICS}
    for case in cases:
        assert case["target"] in case["prompt"]
        assert diagnostic[(case["kind"], case["target"])] not in case["prompt"]
        assert "TARGET=" not in case["prompt"]
        assert "fallback" not in case["prompt"].lower()
        assert "Its only required argument is" in case["prompt"]
        assert case["prompt"].endswith("\n<|assistant|>\n")


def test_weak_cohort_is_exactly_40_cases():
    cases = preflight.build_cases()
    weak = {"short_code", "long_code", "url", "path"}
    rows = []
    for case in cases:
        rows.append({
            "kind": case["kind"],
            "score": {
                "envelope_json_valid": True,
                "tool_name_correct": True,
                "slot_exact": False,
            },
        })
    summary = preflight.cohort_summary(rows, weak)
    assert summary["cases"] == 40
    assert summary["envelope_json_valid"] == 40
    assert summary["tool_name_correct"] == 40
    assert summary["slot_exact"] == 0


def test_reference_control_remains_exact_v008_four_cases():
    cases = preflight.prior.reference_cases(CFG)
    assert len(cases) == 4
    assert [case["expected_tool"] for case in cases] == [
        "weather", "calculator", "web_search", "get_time"
    ]


def test_training_authorization_cannot_be_enabled(tmp_path, monkeypatch):
    cfg = copy.deepcopy(CFG)
    cfg["training_authorized"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(preflight, "DEFAULT_CONFIG", path)
    with pytest.raises(ValueError, match="preflight-only"):
        preflight.load_config(path)
