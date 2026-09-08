from __future__ import annotations

import copy
import json

import pytest

from jobs import ember_envelope_preflight_v040 as preflight

CFG = json.loads(preflight.DEFAULT_CONFIG.read_text())


def test_config_is_cpu_only_and_keeps_95_percent_gate():
    cfg = preflight.load_config(preflight.DEFAULT_CONFIG)
    assert cfg["version"] == "0.0.40"
    assert cfg["expected_cases"] == 90
    assert cfg["expected_reference_cases"] == 4
    assert cfg["calibration_values_per_kind"] == 8
    assert cfg["replacement_margin"] == 2
    assert cfg["baseline_gate"] == {
        "minimum_envelope_json_valid_rate": 0.95,
        "minimum_envelope_tool_name_rate": 0.95,
    }
    assert cfg["training_authorized"] is False
    assert cfg["gpu_training_authorized"] is False
    assert cfg["production_authorized"] is False


def test_historical_floors_match_measured_best_prior_prompts():
    assert CFG["historical_floor"] == {
        "short_code": 8,
        "long_code": 7,
        "url": 7,
        "path": 9,
        "mixed": 9,
    }
    assert preflight.BASELINE_BY_KIND == {
        "short_code": "current_exact",
        "long_code": "tool_named",
        "url": "current_exact",
        "path": "prior_natural",
        "mixed": "current_exact",
    }


def test_path_baseline_restores_v038_natural_prompt():
    value = "/tmp/ember/Q7M4/result.json"
    assert preflight.baseline_user("path", value) == f"Find current information about {value}."


def test_calibration_matrix_is_disjoint_balanced_and_includes_baseline():
    cases = preflight.build_calibration_cases(CFG)
    assert len(cases) == 120
    heldout = set(preflight.copy_data.HELD_OUT_VALUES)
    by_kind = {}
    for case in cases:
        assert case["target"] not in heldout
        by_kind.setdefault(case["kind"], []).append(case)
    assert set(by_kind) == preflight.CALIBRATION_KINDS
    for kind, items in by_kind.items():
        assert len(items) == 24
        assert {item["candidate"] for item in items} == {"baseline", "direct_query", "current_quoted"}
        assert sum(item["candidate"] == "baseline" for item in items) == 8


def _fake_rows(kind: str, baseline: int, direct: int, quoted: int):
    rows = []
    for name, count in (("baseline", baseline), ("direct_query", direct), ("current_quoted", quoted)):
        for i in range(8):
            ok = i < count
            rows.append({
                "kind": kind,
                "candidate": name,
                "score": {"envelope_json_valid": ok, "tool_name_correct": ok},
            })
    return rows


def test_challenger_must_beat_baseline_by_full_margin():
    rows = []
    for kind in sorted(preflight.CALIBRATION_KINDS):
        if kind == "short_code":
            rows.extend(_fake_rows(kind, baseline=5, direct=6, quoted=7))
        else:
            rows.extend(_fake_rows(kind, baseline=6, direct=7, quoted=6))
    summary, selected = preflight.summarize_matrix(rows, CFG)
    assert selected["short_code"] == "current_quoted"
    for kind in preflight.CALIBRATION_KINDS - {"short_code"}:
        assert selected[kind] == "baseline"
    assert summary["short_code"]["replacement_margin"] == 2


def test_final_battery_is_90_and_perfect_kinds_stay_schema_natural():
    selected = {kind: "baseline" for kind in preflight.CALIBRATION_KINDS}
    cases = preflight.build_final_cases(CFG, selected)
    assert len(cases) == 90
    assert all(case["target"] in case["prompt"] for case in cases)
    for case in cases:
        if case["kind"] in preflight.PERFECT_KINDS:
            assert case["variant"] == "v037_schema_natural"


def test_regression_gate_blocks_any_kind_below_historical_floor():
    metrics = {}
    for kind in preflight.copy_data.KINDS:
        floor = CFG["historical_floor"].get(kind, 10)
        metrics[kind] = {"envelope_json_valid": floor}
    assert preflight.regression_gate(metrics, CFG)["passed"] is True
    metrics["path"]["envelope_json_valid"] = 8
    gate = preflight.regression_gate(metrics, CFG)
    assert gate["passed"] is False
    assert gate["checks"]["path"] == {"floor": 9, "actual": 8, "passed": False}


def test_reference_control_remains_exact_four_historical_cases():
    cases = preflight.control.reference_cases(CFG)
    assert len(cases) == 4
    assert [case["expected_tool"] for case in cases] == ["weather", "calculator", "web_search", "get_time"]


def test_training_authorization_cannot_be_enabled(tmp_path, monkeypatch):
    cfg = copy.deepcopy(CFG)
    cfg["gpu_training_authorized"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(preflight, "DEFAULT_CONFIG", path)
    with pytest.raises(ValueError, match="CPU preflight-only"):
        preflight.load_config(path)
