from __future__ import annotations

from collections import Counter
import copy
import json

import pytest

from jobs import ember_envelope_preflight_v037 as v037
from jobs import ember_envelope_preflight_v038 as preflight

CFG = json.loads(preflight.DEFAULT_CONFIG.read_text())


def test_config_is_cpu_only_and_keeps_95_percent_global_gate():
    cfg = preflight.load_config(preflight.DEFAULT_CONFIG)
    assert cfg["version"] == "0.0.38"
    assert cfg["expected_cases"] == 90 and cfg["expected_reference_cases"] == 4
    assert set(cfg["calibration_kinds"]) == {"short_code", "long_code"}
    assert cfg["baseline_gate"] == {
        "minimum_envelope_json_valid_rate": 0.95,
        "minimum_envelope_tool_name_rate": 0.95,
    }
    assert cfg["training_authorized"] is False
    assert cfg["gpu_training_authorized"] is False
    assert cfg["production_authorized"] is False


def test_only_short_and_long_code_user_wording_changes_from_v037():
    new_cases = preflight.build_cases(CFG)
    old_cases = v037.build_cases()
    assert len(new_cases) == len(old_cases) == 90
    old = {(case["kind"], case["target"]): case for case in old_cases}
    changed = 0
    for case in new_cases:
        previous = old[(case["kind"], case["target"])]
        if case["kind"] in preflight.CALIBRATION_KINDS:
            changed += 1
            assert case["prompt"] != previous["prompt"]
            assert f'Search the web for this exact identifier: "{case["target"]}".' in case["prompt"]
            assert case["expected_tool"] == "web_search"
            assert case["argument_key"] == "query"
        else:
            assert case["prompt"] == previous["prompt"]
    assert changed == 20


def test_battery_remains_balanced_and_has_single_target_only():
    cases = preflight.build_cases(CFG)
    assert Counter(c["kind"] for c in cases) == dict.fromkeys(preflight.copy_data.KINDS, 10)
    corrupt = {(kind, value): bad for kind, value, bad in preflight.copy_data.DIAGNOSTICS}
    for case in cases:
        assert case["target"] in case["prompt"]
        assert corrupt[(case["kind"], case["target"])] not in case["prompt"]
        assert "TARGET=" not in case["prompt"]
        assert "fallback" not in case["prompt"].lower()
        assert "Its only required argument is" in case["prompt"]
        assert case["prompt"].endswith("\n<|assistant|>\n")


def test_exact_v008_reference_control_remains_four_cases():
    cases = v037.prior.reference_cases(CFG)
    assert len(cases) == 4
    assert [case["expected_tool"] for case in cases] == [
        "weather", "calculator", "web_search", "get_time"
    ]


def test_training_authorization_cannot_be_enabled(tmp_path, monkeypatch):
    cfg = copy.deepcopy(CFG)
    cfg["gpu_training_authorized"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(preflight, "DEFAULT_CONFIG", path)
    with pytest.raises(ValueError, match="CPU preflight-only"):
        preflight.load_config(path)
