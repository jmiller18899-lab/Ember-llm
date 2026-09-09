from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from jobs import ember_structural_prefix_v046 as v046


ROOT = Path(__file__).resolve().parents[1]


def test_v046_is_cpu_diagnostic_only_and_pins_both_failure_cohorts():
    cfg = v046.load_config(v046.DEFAULT_CONFIG)
    assert cfg["version"] == "0.0.46"
    assert cfg["expected_residual_cases"] == 24
    assert cfg["expected_structural_cases"] == 17
    assert cfg["expected_nonrescued_failures"] == 4
    assert cfg["expected_structural_passes"] == 13
    assert cfg["expected_entry_suppression_cases"] == 2
    assert set(cfg["nonrescued_failure_ids"]) == v046.NONRESCUED_FAILURE_IDS
    assert set(cfg["entry_suppression_ids"]) == v046.ENTRY_SUPPRESSION_IDS
    for key in (
        "training_authorized",
        "gpu_training_authorized",
        "production_authorized",
        "placement_learning_authorized",
    ):
        assert cfg[key] is False


def test_v046_reuses_the_exact_v045_residual_prompts():
    current = v046.residual_cases()
    prior = v046.prior.target_cases()
    assert len(current) == len(prior) == 24
    assert [(c["id"], c["prompt"]) for c in current] == [(c["id"], c["prompt"]) for c in prior]


def test_structural_cohort_is_four_nonrescued_failures_plus_all_matched_passes():
    cases = v046.structural_cases()
    assert len(cases) == 17
    counts = {}
    for case in cases:
        counts[case["subtype"]] = counts.get(case["subtype"], 0) + 1
        assert case["id"] not in v046.ENTRY_SUPPRESSION_IDS
    assert counts == v046.EXPECTED_STRUCTURAL_COUNTS
    ids = {case["id"] for case in cases}
    assert v046.NONRESCUED_FAILURE_IDS <= ids
    assert len(ids - v046.NONRESCUED_FAILURE_IDS) == 13


def test_entry_suppression_cohort_is_separate_and_exact():
    cases = v046.entry_suppression_cases()
    assert {case["id"] for case in cases} == v046.ENTRY_SUPPRESSION_IDS
    assert not ({case["id"] for case in cases} & v046.NONRESCUED_FAILURE_IDS)


def test_canonical_prefix_stops_before_any_argument_value_token():
    cfg = v046.load_config(v046.DEFAULT_CONFIG)
    prefix = v046.canonical_prefix(cfg)
    assert prefix == '<|tool|>{"name":"web_search","arguments":{"query":"'
    assert prefix.endswith('"')
    for case in v046.structural_cases():
        assert case["target"] not in prefix


def test_structural_probe_cannot_receive_or_reference_the_target_value():
    signature = inspect.signature(v046.structural_prefix_probe)
    assert "target" not in signature.parameters
    source = inspect.getsource(v046.structural_prefix_probe)
    assert "target" not in source
    assert "argument_key" not in source
    assert "slot_exact" not in source


def test_runner_contains_no_training_or_gpu_path():
    source = (ROOT / "jobs/ember_structural_prefix_v046.py").read_text(encoding="utf-8")
    forbidden = [
        "torch.optim",
        ".backward(",
        "optimizer.step",
        ".cuda(",
        'device="cuda"',
        'to("cuda")',
        "save_checkpoint(",
    ]
    assert not any(token in source for token in forbidden)


def _probe(complete: bool, first_stage: str | None = None):
    steps = [
        {
            "stage": "tool_marker",
            "greedy_match": True,
            "expected_rank": 1,
            "expected_margin_vs_best_other": 2.0,
        },
        {
            "stage": "json_open",
            "greedy_match": complete,
            "expected_rank": 1 if complete else 3,
            "expected_margin_vs_best_other": 1.0 if complete else -1.0,
        },
    ]
    mismatch = None
    if not complete:
        mismatch = {
            "index": 1,
            "stage": first_stage or "json_open",
            "expected_rank": 3,
            "expected_margin_vs_best_other": -1.0,
            "greedy_token_text": "x",
        }
        steps[1]["stage"] = mismatch["stage"]
    return {
        "all_structural_tokens_greedy": complete,
        "first_non_greedy": mismatch,
        "steps": steps,
    }


def test_interpretation_localizes_internal_structural_divergence_when_controls_are_stable():
    cfg = v046.load_config(v046.DEFAULT_CONFIG)
    rows = [
        {"id": case_id, "prefix_probe": _probe(False, "arguments_key")}
        for case_id in sorted(v046.NONRESCUED_FAILURE_IDS)
    ]
    rows += [
        {"id": f"pass_{i}", "prefix_probe": _probe(True)}
        for i in range(13)
    ]
    result = v046.interpret(rows, cfg)
    assert result["matched_prefix_stable"] is True
    assert result["regime"] == "nonrescued_failures_diverge_within_structural_prefix"


def test_interpretation_calls_out_later_failure_when_all_four_prefixes_are_top1():
    cfg = v046.load_config(v046.DEFAULT_CONFIG)
    rows = [
        {"id": case_id, "prefix_probe": _probe(True)}
        for case_id in sorted(v046.NONRESCUED_FAILURE_IDS)
    ]
    rows += [
        {"id": f"pass_{i}", "prefix_probe": _probe(True)}
        for i in range(13)
    ]
    result = v046.interpret(rows, cfg)
    assert result["regime"] == "nonrescued_failures_diverge_after_structural_prefix"


def test_config_rejects_learning_authorization(tmp_path):
    cfg = json.loads(v046.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    cfg["placement_learning_authorized"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError):
        v046.load_config(path)
