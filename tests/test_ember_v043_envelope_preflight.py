from __future__ import annotations

from collections import Counter
import json

from jobs import ember_envelope_preflight_v043 as preflight

CFG = json.loads(preflight.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_config_is_cpu_only_and_keeps_95_percent_gate():
    cfg = preflight.load_config(preflight.DEFAULT_CONFIG)
    assert cfg["version"] == "0.0.43"
    assert cfg["calibration_folds"] == ["select", "confirm", "stress"]
    assert cfg["baseline_gate"] == {
        "minimum_envelope_json_valid_rate": 0.95,
        "minimum_envelope_tool_name_rate": 0.95,
    }
    assert cfg["historical_floor"] == preflight.EXPECTED_KIND_FLOOR
    assert cfg["historical_subtype_floor"] == preflight.EXPECTED_SUBTYPE_FLOOR
    assert cfg["training_authorized"] is False
    assert cfg["gpu_training_authorized"] is False
    assert cfg["production_authorized"] is False


def test_only_five_residual_subtypes_are_calibrated():
    assert preflight.TARGET_SUBTYPES == {
        "short_code/len4",
        "short_code/len5",
        "long_code/4x4",
        "long_code/3x5",
        "path/plain_leaf",
    }
    assert "url/two_segment" not in preflight.TARGET_SUBTYPES
    assert "mixed/upper" not in preflight.TARGET_SUBTYPES


def test_v042_url_and_mixed_gains_are_frozen():
    _id, user = preflight.frozen_safe_user("url", "https://example.test/ABCD/efgh")
    assert 'Use web_search with query "https://example.test/ABCD/efgh".' == user
    _id, user = preflight.frozen_safe_user("url", "https://example.test/a7Q9")
    assert "exact quoted text" in user
    _id, user = preflight.frozen_safe_user("mixed", "acct_ABCD-5831")
    assert "literal query string" in user


def test_calibration_values_are_threefold_disjoint_and_exact_subtype():
    values = preflight.calibration_values(CFG)
    assert set(values) == preflight.TARGET_SUBTYPES
    flat = []
    for subtype, folds in values.items():
        assert set(folds) == {"select", "confirm", "stress"}
        kind, _variant = preflight.SUBTYPE_VARIANT[subtype]
        for items in folds.values():
            assert len(items) == 6
            for value in items:
                assert value not in preflight.copy_data.HELD_OUT_VALUES
                assert preflight.subtype_for(kind, value) == subtype
            flat.extend(items)
    assert len(flat) == len(set(flat))


def test_calibration_matrix_has_baseline_and_three_challengers():
    cases = preflight.build_calibration_cases(CFG)
    assert len(cases) == 5 * 3 * 6 * 4
    assert Counter(case["candidate"] for case in cases) == {
        "baseline": 90,
        "lookup_exact": 90,
        "search_literal": 90,
        "web_lookup": 90,
    }


def _rows_for_selector():
    rows = []
    folds = ["select", "confirm", "stress"]
    for subtype in sorted(preflight.TARGET_SUBTYPES):
        # baseline = 3/6 on every fold
        scores = {
            "baseline": [3, 3, 3],
            # valid: wins two folds, ties one, combined +2, never loses
            "lookup_exact": [4, 4, 3],
            # invalid: strong overall but loses one fold
            "search_literal": [5, 2, 5],
            # invalid: only one winning fold
            "web_lookup": [4, 3, 3],
        }
        for candidate, fold_scores in scores.items():
            for fold, good in zip(folds, fold_scores):
                for i in range(6):
                    ok = i < good
                    rows.append({
                        "subtype": subtype,
                        "fold": fold,
                        "candidate": candidate,
                        "score": {
                            "envelope_json_valid": ok,
                            "tool_name_correct": ok,
                        },
                    })
    return rows


def test_selector_requires_zero_loss_and_two_of_three_wins():
    _summary, selected = preflight.summarize_matrix(_rows_for_selector(), CFG)
    assert set(selected.values()) == {"lookup_exact"}


def test_final_battery_stays_90_and_url_mixed_are_not_replaceable():
    selected = {subtype: "lookup_exact" for subtype in preflight.TARGET_SUBTYPES}
    cases = preflight.build_final_cases(CFG, selected)
    assert len(cases) == 90
    assert Counter(case["kind"] for case in cases) == dict.fromkeys(preflight.copy_data.KINDS, 10)
    for case in cases:
        if case["subtype"] in preflight.TARGET_SUBTYPES:
            assert case["variant"].startswith("lookup_exact|")
        if case["kind"] in {"url", "mixed"}:
            assert not case["variant"].startswith("lookup_exact|")


def test_runner_contains_no_optimizer_or_training_path():
    text = (preflight.ROOT / "jobs/ember_envelope_preflight_v043.py").read_text(encoding="utf-8")
    assert "optimizer.step(" not in text
    assert "loss.backward(" not in text
    assert "torch.optim" not in text
