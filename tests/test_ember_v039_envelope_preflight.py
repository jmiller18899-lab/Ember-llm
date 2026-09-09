from __future__ import annotations

from collections import Counter
import copy
import json

import pytest

from jobs import ember_envelope_preflight_v039 as preflight

CFG = json.loads(preflight.DEFAULT_CONFIG.read_text())


def test_config_is_cpu_only_and_keeps_95_percent_global_gate():
    cfg = preflight.load_config(preflight.DEFAULT_CONFIG)
    assert cfg["version"] == "0.0.39"
    assert cfg["expected_cases"] == 90 and cfg["expected_reference_cases"] == 4
    assert set(cfg["calibration_kinds"]) == preflight.CALIBRATION_KINDS
    assert cfg["calibration_values_per_kind"] == 4
    assert cfg["baseline_gate"] == {
        "minimum_envelope_json_valid_rate": 0.95,
        "minimum_envelope_tool_name_rate": 0.95,
    }
    assert cfg["training_authorized"] is False
    assert cfg["gpu_training_authorized"] is False
    assert cfg["production_authorized"] is False


def test_synthetic_calibration_values_are_disjoint_from_heldout_battery():
    values = preflight.calibration_values(CFG)
    assert set(values) == preflight.CALIBRATION_KINDS
    flat = []
    for kind, items in values.items():
        assert len(items) == 4
        assert len(set(items)) == 4
        flat.extend(items)
        assert all(item not in preflight.copy_data.HELD_OUT_VALUES for item in items), kind
    assert len(flat) == len(set(flat)) == 20


def test_prompt_matrix_has_60_cases_and_preserves_schema_tool_shape():
    cases = preflight.build_calibration_cases(CFG)
    assert len(cases) == 60
    counts = Counter((case["kind"], case["variant"]) for case in cases)
    for kind in preflight.CALIBRATION_KINDS:
        for variant in CFG["variant_priority"]:
            assert counts[(kind, variant)] == 4
    for case in cases:
        assert case["target"] in case["prompt"]
        assert case["expected_tool"] == "web_search"
        assert case["argument_key"] == "query"
        assert "Use web_search with JSON arguments" in case["prompt"]
        assert case["prompt"].endswith("\n<|assistant|>\n")


def test_current_exact_keeps_v038_code_wording_available():
    template = CFG["prompt_variants"]["current_exact"]
    for kind in ("short_code", "long_code"):
        text = preflight._user_from_template(kind, "Q7M4", template)
        assert text == 'Search the web for this exact identifier: "Q7M4".'


def test_matrix_selection_is_deterministic_and_uses_priority_for_ties():
    rows = []
    for kind in sorted(preflight.CALIBRATION_KINDS):
        for name in CFG["variant_priority"]:
            for i in range(4):
                success = name in {"current_exact", "tool_named"}
                rows.append({
                    "kind": kind,
                    "variant": name,
                    "score": {
                        "envelope_json_valid": success,
                        "tool_name_correct": success,
                    },
                })
    summary, selected = preflight.summarize_matrix(rows, CFG)
    assert set(summary) == preflight.CALIBRATION_KINDS
    assert all(name == "current_exact" for name in selected.values())


def test_final_90_cases_change_only_targeted_kinds():
    selected = {kind: "tool_named" for kind in preflight.CALIBRATION_KINDS}
    cases = preflight.build_final_cases(CFG, selected)
    assert len(cases) == 90
    assert Counter(case["kind"] for case in cases) == dict.fromkeys(preflight.copy_data.KINDS, 10)
    for case in cases:
        if case["kind"] in preflight.CALIBRATION_KINDS:
            assert case["variant"] == "tool_named"
            assert "Use web_search to find current information" in case["prompt"]
        else:
            assert case["variant"] == "v037_schema_natural"
            tool, _field = preflight.TOOL_BY_KIND[case["kind"]]
            expected_user = preflight.schema_prompt._user_request(tool, case["target"])
            assert f"<|user|>\n{expected_user}\n" in case["prompt"]


def test_exact_v008_reference_control_remains_four_cases():
    cases = preflight.control.reference_cases(CFG)
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
    with pytest.raises(ValueError, match="CPU preflight-only"):
        preflight.load_config(path)
