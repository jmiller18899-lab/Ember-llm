from __future__ import annotations

from collections import Counter
import copy
import json

import pytest

from jobs import ember_envelope_preflight_v041 as preflight

CFG = json.loads(preflight.DEFAULT_CONFIG.read_text())


def test_config_is_cpu_only_and_keeps_95_percent_gate():
    cfg = preflight.load_config(preflight.DEFAULT_CONFIG)
    assert cfg["version"] == "0.0.41"
    assert cfg["expected_cases"] == 90
    assert cfg["expected_reference_cases"] == 4
    assert cfg["baseline_gate"] == {
        "minimum_envelope_json_valid_rate": 0.95,
        "minimum_envelope_tool_name_rate": 0.95,
    }
    assert cfg["historical_floor"] == preflight.EXPECTED_KIND_FLOOR
    assert cfg["historical_subtype_floor"] == preflight.EXPECTED_SUBTYPE_FLOOR
    assert cfg["training_authorized"] is False
    assert cfg["gpu_training_authorized"] is False
    assert cfg["production_authorized"] is False


def test_classifier_covers_residual_structures_from_v040():
    examples = {
        ("short_code", "Q7M4"): "short_code/len4",
        ("short_code", "R6GBH"): "short_code/len5",
        ("long_code", "V9K2-4R7P"): "long_code/4x4",
        ("long_code", "6AR-AWF98"): "long_code/3x5",
        ("url", "https://example.test/a7Q9"): "url/one_mixed",
        ("url", "https://example.test/XBL6/v5k7"): "url/two_segment",
        ("url", "https://example.test/A3KX"): "url/one_upper",
        ("path", "/tmp/ember/KQ6B/result.json"): "path/plain_leaf",
        ("path", "/tmp/ember/QMB9/result-tvks.json"): "path/result_code",
        ("path", "/tmp/ember/SE64/payload-96.json"): "path/leaf_numeric",
        ("mixed", "acct_GZ2V-4554"): "mixed/upper",
        ("mixed", "acct_lvsy-8951"): "mixed/lower",
        ("mixed", "acct_Q7m4-5831"): "mixed/mixedcase",
    }
    for (kind, value), expected in examples.items():
        assert preflight.subtype_for(kind, value) == expected


def test_heldout_structural_counts_match_v040_evidence():
    counts = Counter(
        preflight.subtype_for(kind, value)
        for kind, value, _corrupt in preflight.copy_data.DIAGNOSTICS
        if kind in preflight.WEAK_KINDS
    )
    assert {key: counts[key] for key in preflight.TARGET_SUBTYPES} == {
        "short_code/len4": 6,
        "short_code/len5": 4,
        "long_code/4x4": 6,
        "long_code/3x5": 4,
        "url/one_mixed": 4,
        "url/two_segment": 3,
        "path/plain_leaf": 4,
        "mixed/upper": 3,
    }


def test_calibration_values_are_disjoint_and_subtype_exact():
    values = preflight.calibration_values(CFG)
    assert set(values) == preflight.TARGET_SUBTYPES
    flat = []
    for subtype, items in values.items():
        assert len(items) == 8
        kind, _variant = preflight.SUBTYPE_VARIANT[subtype]
        for value in items:
            assert value not in preflight.copy_data.HELD_OUT_VALUES
            assert preflight.subtype_for(kind, value) == subtype
        flat.extend(items)
    assert len(flat) == len(set(flat))


def test_calibration_matrix_has_baseline_and_three_challengers_per_value():
    cases = preflight.build_calibration_cases(CFG)
    assert len(cases) == 8 * 8 * 4
    grouped = Counter((case["subtype"], case["target"]) for case in cases)
    assert set(grouped.values()) == {4}
    candidates = Counter(case["candidate"] for case in cases)
    assert candidates == {
        "baseline": 64,
        "literal_query": 64,
        "json_exact": 64,
        "quoted_text": 64,
    }


def test_selector_requires_two_case_margin_over_baseline():
    rows = []
    for subtype in sorted(preflight.TARGET_SUBTYPES):
        for candidate, good in (
            ("baseline", 5),
            ("literal_query", 6),
            ("json_exact", 7),
            ("quoted_text", 4),
        ):
            for i in range(8):
                ok = i < good
                rows.append({
                    "subtype": subtype,
                    "candidate": candidate,
                    "score": {"envelope_json_valid": ok, "tool_name_correct": ok},
                })
    _summary, selected = preflight.summarize_matrix(rows, CFG)
    assert set(selected.values()) == {"json_exact"}


def test_final_battery_is_90_and_only_target_subtypes_are_replaceable():
    selected = {subtype: "json_exact" for subtype in preflight.TARGET_SUBTYPES}
    cases = preflight.build_final_cases(CFG, selected)
    assert len(cases) == 90
    assert Counter(case["kind"] for case in cases) == dict.fromkeys(preflight.copy_data.KINDS, 10)
    for case in cases:
        if case["subtype"] in preflight.TARGET_SUBTYPES:
            assert case["variant"].startswith("json_exact|")
        else:
            assert not case["variant"].startswith("json_exact|")


def test_subtype_regression_gate_protects_each_v040_floor():
    metrics = {
        key: {"envelope_json_valid": floor}
        for key, floor in preflight.EXPECTED_SUBTYPE_FLOOR.items()
    }
    assert preflight.subtype_regression_gate(metrics, CFG)["passed"] is True
    metrics["url/one_mixed"]["envelope_json_valid"] = 1
    gate = preflight.subtype_regression_gate(metrics, CFG)
    assert gate["passed"] is False
    assert gate["checks"]["url/one_mixed"]["passed"] is False


def test_training_authorization_cannot_be_enabled(tmp_path, monkeypatch):
    cfg = copy.deepcopy(CFG)
    cfg["training_authorized"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(preflight, "DEFAULT_CONFIG", path)
    with pytest.raises(ValueError, match="CPU preflight-only"):
        preflight.load_config(path)
