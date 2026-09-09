from __future__ import annotations

from collections import Counter
import copy
import json

import pytest

from jobs import ember_envelope_preflight_v043 as prior
from jobs import ember_envelope_preflight_v044 as preflight

CFG = json.loads(preflight.DEFAULT_CONFIG.read_text(encoding="utf-8"))
ALL_BASELINE = {subtype: "baseline" for subtype in preflight.TARGET_SUBTYPES}


def test_config_is_cpu_only_and_keeps_every_v043_gate():
    cfg = preflight.load_config(preflight.DEFAULT_CONFIG)
    assert cfg["version"] == "0.0.44"
    assert cfg["calibration_folds"] == ["select", "confirm", "stress"]
    assert cfg["baseline_gate"] == {
        "minimum_envelope_json_valid_rate": 0.95,
        "minimum_envelope_tool_name_rate": 0.95,
    }
    assert cfg["reference_gate"] == {
        "minimum_envelope_json_valid_rate": 1.0,
        "minimum_envelope_tool_name_rate": 1.0,
    }
    assert cfg["historical_floor"] == prior.EXPECTED_KIND_FLOOR
    assert cfg["historical_subtype_floor"] == prior.EXPECTED_SUBTYPE_FLOOR
    assert cfg["training_authorized"] is False
    assert cfg["gpu_training_authorized"] is False
    assert cfg["production_authorized"] is False


def test_only_code_shaped_residual_subtypes_are_calibrated():
    assert preflight.TARGET_SUBTYPES == {
        "short_code/len4",
        "short_code/len5",
        "long_code/4x4",
        "long_code/3x5",
    }
    # path/plain_leaf already consumed its user-prompt change in v0.0.43 and is
    # not a code lookup, so it must stay frozen rather than move again.
    assert "path/plain_leaf" not in preflight.TARGET_SUBTYPES


def test_frozen_user_stack_reproduces_the_measured_v043_prompts_exactly():
    measured = prior.build_final_cases(preflight.V043_CFG, preflight.V043_SELECTION)
    mine = preflight.build_final_cases(CFG, ALL_BASELINE)
    assert len(measured) == len(mine) == 90
    for before, after in zip(measured, mine):
        assert before["target"] == after["target"]
        assert before["prompt"] == after["prompt"]


def test_v043_path_selection_is_carried_forward_not_reverted():
    """v0.0.43 froze web_lookup for path/plain_leaf. v0.0.44 inherits it, not the
    pre-v0.0.43 phrasing, because that stack is what measured 84/90."""
    assert preflight.V043_SELECTION["path/plain_leaf"] == "web_lookup"
    variant, user = preflight.frozen_user("path", "/tmp/ember/Q7M4/result.json")
    assert variant == "web_lookup|path/plain_leaf"
    assert user == 'Look up the exact file path "/tmp/ember/Q7M4/result.json" with web_search.'


def test_only_the_system_line_varies_within_a_calibration_case():
    cases = preflight.build_calibration_cases(CFG)
    by_value = {}
    for case in cases:
        by_value.setdefault((case["subtype"], case["fold"], case["target"]), {})[
            case["candidate"]
        ] = case["prompt"]
    assert by_value
    for prompts in by_value.values():
        users = {prompt.split("<|user|>\n", 1)[1] for prompt in prompts.values()}
        assert len(users) == 1, "the user request must be identical across candidates"
        systems = {prompt.split("<|user|>", 1)[0] for prompt in prompts.values()}
        assert len(systems) == len(prompts), "each candidate must supply a distinct system line"


def test_challenger_systems_never_instruct_value_placement():
    for name, template in CFG["challenger_systems"].items():
        rendered = template.format(tool="web_search", field="query").lower()
        for phrase in CFG["forbidden_system_phrases"]:
            assert phrase.lower() not in rendered, f"{name} leaks a placement instruction"


def test_a_placement_instruction_is_rejected_rather_than_silently_accepted():
    bad = copy.deepcopy(CFG)
    bad["challenger_systems"]["code_activation"] = (
        "You are Ember. Copy the code into query. Its only required argument is {field}, a string."
    )
    with pytest.raises(ValueError, match="instructs value placement"):
        preflight.assert_no_placement_instruction(bad)


def test_challengers_change_activation_and_keep_the_schema_sentence():
    baseline = preflight.baseline_system("short_code")
    schema = CFG["schema_sentence"].format(field="query")
    assert baseline.endswith(schema)
    for name, template in CFG["challenger_systems"].items():
        rendered = preflight.challenger_system("short_code", template)
        assert rendered.endswith(schema), f"{name} altered the schema half of the system line"
        assert rendered != baseline, f"{name} is not a change at all"


def test_calibration_values_are_disjoint_from_the_battery_and_from_v043():
    v043_values = set()
    for folds in prior.calibration_values(preflight.V043_CFG).values():
        for values in folds.values():
            v043_values.update(values)
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
                assert value not in v043_values
                assert preflight.subtype_for(kind, value) == subtype
            flat.extend(items)
    assert len(flat) == len(set(flat))


def test_calibration_matrix_has_baseline_and_three_system_challengers():
    cases = preflight.build_calibration_cases(CFG)
    assert len(cases) == 4 * 3 * 6 * 4
    assert Counter(case["candidate"] for case in cases) == {
        "baseline": 72,
        "code_activation": 72,
        "identifier_capability": 72,
        "reference_lookup": 72,
    }


def _rows_for_selector():
    rows = []
    folds = ["select", "confirm", "stress"]
    for subtype in sorted(preflight.TARGET_SUBTYPES):
        scores = {
            "baseline": [3, 3, 3],
            # wins two folds, ties one, combined +2, never loses
            "code_activation": [4, 4, 3],
            # strong overall but loses a fold
            "identifier_capability": [5, 2, 5],
            # only one winning fold
            "reference_lookup": [4, 3, 3],
        }
        for candidate, fold_scores in scores.items():
            for fold, good in zip(folds, fold_scores):
                for i in range(6):
                    ok = i < good
                    rows.append({
                        "subtype": subtype,
                        "fold": fold,
                        "candidate": candidate,
                        "score": {"envelope_json_valid": ok, "tool_name_correct": ok},
                    })
    return rows


def test_selector_requires_zero_loss_and_two_of_three_wins():
    summary, selected = preflight.summarize_matrix(_rows_for_selector(), CFG)
    assert set(selected.values()) == {"code_activation"}
    for entry in summary.values():
        assert entry["synthetic_gain"]["envelope_json_valid"] == 2


def test_selector_keeps_the_baseline_when_nothing_clears_the_rule():
    rows = [
        {**row, "score": {"envelope_json_valid": False, "tool_name_correct": False}}
        for row in _rows_for_selector()
    ]
    _summary, selected = preflight.summarize_matrix(rows, CFG)
    assert set(selected.values()) == {"baseline"}


def test_selected_system_reaches_only_code_cases_and_battery_stays_90():
    selected = {subtype: "code_activation" for subtype in preflight.TARGET_SUBTYPES}
    frozen = preflight.build_final_cases(CFG, ALL_BASELINE)
    cases = preflight.build_final_cases(CFG, selected)
    assert len(cases) == 90
    assert Counter(case["kind"] for case in cases) == dict.fromkeys(preflight.copy_data.KINDS, 10)
    changed = [
        after for before, after in zip(frozen, cases) if before["prompt"] != after["prompt"]
    ]
    assert {case["kind"] for case in changed} == {"short_code", "long_code"}
    assert {case["subtype"] for case in changed} == preflight.TARGET_SUBTYPES
    for case in cases:
        if case["subtype"] in preflight.TARGET_SUBTYPES:
            assert case["system_variant"].startswith("code_activation|")
        else:
            assert case["system_variant"] == "v037_schema_system"


def test_transfer_report_pairs_synthetic_gain_with_heldout_result():
    summary, _selected = preflight.summarize_matrix(_rows_for_selector(), CFG)
    subtype_metrics = {
        subtype: {"envelope_json_valid": 6} for subtype in preflight.TARGET_SUBTYPES
    }
    transfer = preflight.transfer_report(summary, subtype_metrics)
    assert set(transfer) == preflight.TARGET_SUBTYPES
    entry = transfer["short_code/len4"]
    assert entry["synthetic_gain_envelope_json_valid"] == 2
    assert entry["heldout_envelope_json_valid"] == 6
    assert entry["heldout_floor"] == prior.EXPECTED_SUBTYPE_FLOOR["short_code/len4"]
    assert entry["heldout_gain_over_floor"] == 6 - entry["heldout_floor"]


def test_runner_contains_no_optimizer_or_training_path():
    text = (preflight.ROOT / "jobs/ember_envelope_preflight_v044.py").read_text(encoding="utf-8")
    assert "optimizer.step(" not in text
    assert "loss.backward(" not in text
    assert "torch.optim" not in text
