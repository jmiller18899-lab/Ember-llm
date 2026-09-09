from __future__ import annotations

from collections import Counter
import json

from jobs import ember_envelope_preflight_v042 as preflight

CFG = json.loads(preflight.DEFAULT_CONFIG.read_text())


def test_config_is_cpu_only_and_keeps_95_percent_gate():
    cfg = preflight.load_config(preflight.DEFAULT_CONFIG)
    assert cfg["version"] == "0.0.42"
    assert cfg["expected_cases"] == 90
    assert cfg["baseline_gate"] == {
        "minimum_envelope_json_valid_rate": 0.95,
        "minimum_envelope_tool_name_rate": 0.95,
    }
    assert cfg["historical_floor"] == preflight.EXPECTED_KIND_FLOOR
    assert cfg["historical_subtype_floor"] == preflight.EXPECTED_SUBTYPE_FLOOR
    assert cfg["training_authorized"] is False
    assert cfg["gpu_training_authorized"] is False
    assert cfg["production_authorized"] is False


def test_assembled_stack_preserves_v041_gains_and_restores_len5():
    _, url = preflight.assembled_safe_user("url", "https://example.test/a7Q9", CFG)
    assert "exact quoted text" in url
    _, mixed = preflight.assembled_safe_user("mixed", "acct_GZ2V-4554", CFG)
    assert "literal query string" in mixed
    _, short5 = preflight.assembled_safe_user("short_code", "R6GBH", CFG)
    assert short5 == 'Search the web for this exact identifier: "R6GBH".'


def test_calibration_values_are_two_fold_disjoint_and_exact_subtype():
    values = preflight.calibration_values(CFG)
    assert set(values) == preflight.TARGET_SUBTYPES
    flat = []
    for subtype, folds in values.items():
        assert set(folds) == {"select", "confirm"}
        kind, _variant = preflight.SUBTYPE_VARIANT[subtype]
        for fold, items in folds.items():
            assert len(items) == 6
            for value in items:
                assert value not in preflight.copy_data.HELD_OUT_VALUES
                assert preflight.subtype_for(kind, value) == subtype
            flat.extend(items)
    assert len(flat) == len(set(flat))


def test_calibration_matrix_size_and_candidate_balance():
    cases = preflight.build_calibration_cases(CFG)
    assert len(cases) == 6 * 2 * 6 * 4
    assert Counter(case["candidate"] for case in cases) == {
        "baseline": 72,
        "typed_exact": 72,
        "tool_query": 72,
        "query_exact": 72,
    }
    assert Counter(case["fold"] for case in cases) == {"select": 144, "confirm": 144}


def test_selector_requires_improvement_on_both_folds():
    rows = []
    for subtype in sorted(preflight.TARGET_SUBTYPES):
        for fold in ("select", "confirm"):
            for candidate, good in (
                ("baseline", 3),
                ("typed_exact", 4),
                ("tool_query", 5 if fold == "select" else 3),
                ("query_exact", 2),
            ):
                for i in range(6):
                    ok = i < good
                    rows.append({
                        "subtype": subtype,
                        "fold": fold,
                        "candidate": candidate,
                        "score": {"envelope_json_valid": ok, "tool_name_correct": ok},
                    })
    _summary, selected = preflight.summarize_matrix(rows, CFG)
    assert set(selected.values()) == {"typed_exact"}


def test_final_battery_is_90_and_only_target_subtypes_replace():
    selected = {subtype: "tool_query" for subtype in preflight.TARGET_SUBTYPES}
    cases = preflight.build_final_cases(CFG, selected)
    assert len(cases) == 90
    assert Counter(case["kind"] for case in cases) == dict.fromkeys(preflight.copy_data.KINDS, 10)
    for case in cases:
        if case["subtype"] in preflight.TARGET_SUBTYPES:
            assert case["variant"].startswith("tool_query|")
        elif case["subtype"] == "url/one_mixed":
            assert case["variant"].startswith("v041:quoted_text")
        elif case["subtype"] == "mixed/upper":
            assert case["variant"].startswith("v041:literal_query")


def test_regression_gate_rejects_any_floor_loss():
    metrics = {key: {"envelope_json_valid": floor} for key, floor in CFG["historical_floor"].items()}
    assert preflight.regression_gate(metrics, CFG["historical_floor"])["passed"] is True
    metrics["short_code"]["envelope_json_valid"] = 7
    assert preflight.regression_gate(metrics, CFG["historical_floor"])["passed"] is False
