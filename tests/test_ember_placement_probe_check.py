from __future__ import annotations

import json

import pytest

from jobs import ember_placement_probe_check as check


def _probe(rows):
    """Shape a probe result the way objectives.placement_probe returns one."""
    tokens = sum(len(r) for r in rows)
    return {
        "cases": len(rows),
        "exact_top1": sum(1 for r in rows if all(t["top1"] for t in r)),
        "exact_top1_rate": sum(1 for r in rows if all(t["top1"] for t in r)) / len(rows),
        "token_top1": sum(1 for r in rows for t in r if t["top1"]),
        "tokens": tokens,
        "token_top1_rate": sum(1 for r in rows for t in r if t["top1"]) / tokens,
        "mean_loss": 1.0,
        "rows": [
            {
                "id": f"c{i}", "subtype": "short_code/len4",
                "target_token_count": len(r),
                "exact_top1": all(t["top1"] for t in r),
                "token_top1": sum(1 for t in r if t["top1"]),
                "mean_loss": 1.0,
                "tokens": r,
            }
            for i, r in enumerate(rows)
        ],
    }


def _tok(top1: bool, rank: int = 1):
    return {"expected_id": 7, "top1": top1, "loss": 0.1, "rank": rank}


CFG = check.load_config()


def test_diagnostic_reuses_the_v050_config_verbatim():
    assert CFG["version"] == "0.0.50"
    assert CFG["selection_gate"]["minimum_placement_token_top1_gain"] == 0.03
    assert CFG["gpu_training_authorized"] is False


def test_a_config_that_authorizes_anything_is_refused(tmp_path):
    bad = dict(CFG)
    bad["gpu_training_authorized"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="authorizes anything"):
        check.load_config(path)


def test_breakdown_separates_the_boundary_token_from_continuation():
    # Every case: first token wrong at rank 40, three continuation tokens right.
    rows = [[_tok(False, 40), _tok(True), _tok(True), _tok(True)] for _ in range(6)]
    breakdown = check.positional_breakdown(_probe(rows))
    assert breakdown["by_index"][0] == {"index": 0, "tokens": 6, "top1": 0, "rate": 0.0}
    assert all(entry["rate"] == 1.0 for entry in breakdown["by_index"][1:])
    assert breakdown["first_token"]["top1_rate"] == 0.0
    assert breakdown["first_token"]["median_rank"] == 40
    assert breakdown["later_tokens"]["top1_rate"] == 1.0


def test_verdict_flags_a_metric_dominated_by_already_correct_positions():
    rows = [[_tok(False, 40), _tok(True), _tok(True), _tok(True)] for _ in range(6)]
    probe = _probe(rows)
    breakdown = check.positional_breakdown(probe)
    head = check.headroom(probe, breakdown, CFG)
    result = check.verdict(head, breakdown)
    assert result["metric_dominated_by_easy_positions"] is True
    assert result["probe_is_a_usable_placement_signal"] is False
    assert any("boundary token" in finding for finding in result["findings"])


def test_verdict_flags_a_saturated_metric():
    rows = [[_tok(True), _tok(True), _tok(True), _tok(True)] for _ in range(6)]
    probe = _probe(rows)
    breakdown = check.positional_breakdown(probe)
    head = check.headroom(probe, breakdown, CFG)
    result = check.verdict(head, breakdown)
    assert result["metric_saturated"] is True
    assert result["probe_is_a_usable_placement_signal"] is False


def test_a_healthy_probe_is_reported_as_usable():
    """Errors spread across positions, so no single position carries the metric."""
    rows = [[_tok(False, 5), _tok(False, 3), _tok(True), _tok(True)] for _ in range(3)]
    rows += [[_tok(True), _tok(True), _tok(False, 4), _tok(True)] for _ in range(3)]
    probe = _probe(rows)
    breakdown = check.positional_breakdown(probe)
    head = check.headroom(probe, breakdown, CFG)
    result = check.verdict(head, breakdown)
    assert breakdown["first_token"]["top1_rate"] == 0.5
    assert result["metric_saturated"] is False
    assert result["metric_dominated_by_easy_positions"] is False
    assert result["token_gate_reachable"] is True
    assert result["probe_is_a_usable_placement_signal"] is True


def test_headroom_counts_cases_one_token_from_exact():
    rows = [
        [_tok(False, 9), _tok(True), _tok(True)],   # one flip from exact
        [_tok(False, 9), _tok(False, 4), _tok(True)],  # two wrong
        [_tok(True), _tok(True), _tok(True)],       # already exact
    ]
    probe = _probe(rows)
    head = check.headroom(probe, check.positional_breakdown(probe), CFG)
    assert head["cases_one_flip_from_exact"] == 1
    assert head["cases_already_exact"] == 1
    assert head["wrong_first_tokens"] == 2
    assert head["wrong_position_counts"] == [0, 1, 2]


def test_headroom_reports_when_the_token_gate_cannot_be_reached():
    # Only one token is wrong out of 40, so fixing every first token gains 2.5%,
    # under the configured 3% requirement.
    rows = [[_tok(False, 2), _tok(True), _tok(True), _tok(True)]] + [
        [_tok(True), _tok(True), _tok(True), _tok(True)] for _ in range(9)
    ]
    probe = _probe(rows)
    head = check.headroom(probe, check.positional_breakdown(probe), CFG)
    assert head["available_token_gain"] == pytest.approx(0.025)
    assert head["required_token_gain"] == 0.03
    assert head["token_gate_reachable"] is False


def test_degenerate_single_token_targets_are_counted():
    rows = [[_tok(True)], [_tok(True), _tok(True)]]
    lengths = check.target_lengths(_probe(rows))
    assert lengths["minimum"] == 1
    assert lengths["degenerate_single_token_cases"] == 1


def test_runner_reads_and_never_writes_a_model():
    text = (check.ROOT / "jobs/ember_placement_probe_check.py").read_text(encoding="utf-8")
    for forbidden in ("optimizer.step(", "loss.backward(", "torch.optim",
                      "upload_file", "create_repo", "cuda"):
        assert forbidden not in text


def test_diagnostic_can_rebuild_the_v050_values_without_another_test_importing_the_shim():
    """The v0.0.50 value bookkeeping only works once the v0.0.49 replay compat
    shim is imported. Its own test suite gets that by side effect from another
    test file; this diagnostic must not depend on collection order."""
    values = check.v050d.target_values(CFG)
    assert set(values) == {"template", "train", "development"}
    for groups in values.values():
        assert groups
        for items in groups.values():
            assert items
