"""Guards for the Ember v0.0.28 boundary-focus objective.

v0.0.27 moved sequence_margin_health by +0.2025 and exact copy by nothing. These
tests pin the three things that follow: mining must prefer sequences that can
still flip over sequences that cannot, the objective must span every decision an
exact copy tests, and a large continuous gain must be reported as an advance.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAINER_V016 = ROOT / "jobs" / "ember_hf_sft_v016.py"
TRAINER_V015 = ROOT / "jobs" / "ember_hf_sft_v015.py"
TRAINER_V028 = ROOT / "jobs" / "ember_hf_sft_v028.py"
DATA_V026 = ROOT / "jobs" / "ember_sft_data_v026.py"
CONFIG = ROOT / "config" / "ember_boundary_focus_v0.0.28.json"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helpers():
    trainer = load(TRAINER_V028, "ember_hf_sft_v028_helpers")
    namespace: dict = {}
    exec(trainer.HELPERS, namespace)
    return namespace


def cfg():
    return json.loads(CONFIG.read_text())


class StubModel:
    def __init__(self, logits):
        self._logits = logits

    def __call__(self, x, _):
        return self._logits, None


def encoded_rows(torch, rows, c, vocab=8):
    """Build (logits, y, w) laid out exactly as base.encode_row does.

    Position 0 carries first_token_weight, the middle carries copy_token_weight,
    the last supervised position carries eos_token_weight, and a trailing
    position carries the plain 1.0 weight that encode_row leaves behind.
    """
    width = max(len(r) for r in rows) + 1
    logits = torch.zeros((len(rows), width, vocab))
    y = torch.full((len(rows), width), -100, dtype=torch.long)
    w = torch.zeros((len(rows), width))
    for i, row in enumerate(rows):
        for t, gap in enumerate(row):
            token = 1 + (t % (vocab - 2))
            logits[i, t, token] = gap
            y[i, t] = token
        y[i, len(row)] = 1
        w[i, : len(row)] = float(c["copy_token_weight"])
        w[i, 0] = float(c["first_token_weight"])
        w[i, len(row) - 1] = float(c["eos_token_weight"])
        w[i, len(row)] = 1.0          # the trailing token encode_row leaves at 1.0
    return logits, y, w


def test_v028_transform_targets_exist_and_runtime_compiles():
    trainer = load(TRAINER_V028, "ember_hf_sft_v028_targets")
    assert trainer.unmatched_transform_targets(TRAINER_V016.read_text()) == []
    runtime = trainer.apply_transforms(TRAINER_V016.read_text())
    compile(runtime, "ember_hf_sft_v028_runtime", "exec")

    assert "--preflight-only" in runtime
    assert "v0.0.28 training requires explicitly approved T4 GPU" in runtime
    assert runtime.index('print("EMBER_HF_V028_PREFLIGHT=PASS"') < runtime.index(
        "torch.cuda.is_available()"
    )
    assert runtime.index("assert_format_parity(data, train_rows)") < runtime.index(
        "source_path, source_info = resolve_v027_source"
    )
    assert "best_score = v028_score(baseline, -1e9)" in runtime
    assert "step=-1" in runtime
    assert "loss = boundary_focus_loss(model, x, y, w, cfg, torch) / accum" in runtime
    assert 'sample_boundary_focus_indices(train, int(cfg["batch_size"])' in runtime
    assert 'SOURCE_REPO = "Jmiller18899/ember-v0.0.27-t4"' in runtime
    assert "ember_boundary_focus_v0.0.28.json" in runtime
    assert "ember_sft_data_v026.py" in runtime


def test_v028_preflight_reports_whether_a_paid_run_is_worth_launching():
    trainer = load(TRAINER_V028, "ember_hf_sft_v028_preflight")
    runtime = trainer.apply_transforms(TRAINER_V016.read_text())
    assert "EMBER_V028_SEQUENCES_WITHIN_REACH=" in runtime
    assert "NO_REACHABLE_SEQUENCES_DO_NOT_LAUNCH_T4" in runtime
    assert 'baseline["distribution"]' in runtime
    # The advice has to print before the GPU guard, or it is not free.
    assert runtime.index("NO_REACHABLE_SEQUENCES_DO_NOT_LAUNCH_T4") < runtime.index(
        "torch.cuda.is_available()"
    )


def test_v028_config_keeps_v026_training_knobs_and_narrows_the_margin():
    c = cfg()
    assert c["version"] == "0.0.28"
    assert c["phase"] == "boundary-focus-repair"
    assert c["source_model_name"] == "ember-v0.0.27-t4"
    # Not adding steps and not raising the learning rate is the point.
    assert c["learning_rate"] == 1.2e-6
    assert c["max_steps"] == 600
    assert c["copy_token_weight"] == 8.0
    # Pressure moves toward the boundary from both sides.
    assert c["sequence_margin"] < 0.61, "v0.0.27 spent effort widening already-safe margins"
    assert c["boundary_band_low"] < 0.0
    assert 0.0 < c["out_of_band_weight"] < 1.0
    # The health statistic keeps its v0.0.27 scale so the series stays readable.
    assert c["margin_health_ceiling"] == 1.2 and c["margin_health_floor"] == -2.0
    assert c["protected_expanded_sequence_margin_health"] <= -0.6846


def test_v028_supervised_span_covers_the_whole_exact_copy_conjunction():
    """The gap that made the first token and EOS invisible to v0.0.27."""
    torch = pytest.importorskip("torch")
    h = helpers()
    c = cfg()
    logits, y, w = encoded_rows(torch, [[1.0, 1.0, 1.0, 1.0, 1.0]], c)
    copy_mask, span_mask = h["v028_supervised_span"](y, w, c, torch)

    # encode_row's copy weight lands only on the interior.
    assert copy_mask[0].tolist() == [False, True, True, True, False, False]
    # The span adds the first token and the EOS decision, and stops there: the
    # trailing 1.0-weight position is not part of the exact-copy event.
    assert span_mask[0].tolist() == [True, True, True, True, True, False]


def test_v028_boundary_mining_prefers_a_flippable_row_over_a_hopeless_one():
    """The v0.0.27 misallocation, stated as a test.

    relu(margin - gap) is monotone in how badly a row fails, so v0.0.27 ranked a
    row at -3.0 above a row at -0.05 for every mined slot. Those are the rows
    least likely to become exact copies.
    """
    torch = pytest.importorskip("torch")
    h = helpers()
    c = cfg()

    hopeless = [[-3.0, 5.0, 5.0, 5.0]]
    flippable = [[-0.05, 5.0, 5.0, 5.0]]
    safe = [[5.0, 5.0, 5.0, 5.0]]

    def priority(rows):
        logits, y, w = encoded_rows(torch, rows, c)
        _, span = h["v028_supervised_span"](y, w, c, torch)
        gap, _ = h["v028_gaps"](logits, y, span, torch)
        worst_hinge, worst_gap, _ = h["v028_row_worst"](gap, span, c, torch)
        far = (float(c["sequence_margin"]) - float(c["boundary_band_low"])) * float(
            c["out_of_band_weight"]
        )
        in_band = worst_gap.ge(float(c["boundary_band_low"]))
        return float(
            torch.where(in_band, worst_hinge, torch.full_like(worst_hinge, far))[0]
        )

    # v0.0.27's ranking would be hopeless > flippable > safe. v0.0.28 inverts the
    # first pair while still leaving the hopeless row above the already-safe one.
    assert priority(flippable) > priority(hopeless) > priority(safe)


def test_v028_loss_down_weights_hopeless_rows_without_abandoning_them():
    torch = pytest.importorskip("torch")
    h = helpers()
    c = cfg()

    def sequence_term(rows):
        logits, y, w = encoded_rows(torch, rows, c)
        _, span = h["v028_supervised_span"](y, w, c, torch)
        gap, _ = h["v028_gaps"](logits, y, span, torch)
        worst_hinge, worst_gap, has_targets = h["v028_row_worst"](gap, span, c, torch)
        focus, in_band = h["v028_boundary_weight"](worst_gap, c, torch)
        rows_f = has_targets.to(worst_hinge.dtype)
        return float((worst_hinge * focus * rows_f).sum() / rows_f.sum().clamp_min(1.0)), bool(
            in_band[0]
        )

    flippable_term, flippable_in_band = sequence_term([[-0.05, 5.0, 5.0, 5.0]])
    hopeless_term, hopeless_in_band = sequence_term([[-3.0, 5.0, 5.0, 5.0]])

    assert flippable_in_band and not hopeless_in_band
    # Still learning, just not owning the batch.
    assert hopeless_term > 0.0
    assert hopeless_term == pytest.approx(
        (c["sequence_margin"] + 3.0) * c["out_of_band_weight"], abs=1e-5
    )
    assert flippable_term == pytest.approx(c["sequence_margin"] + 0.05, abs=1e-5)


def test_v028_loss_reduces_to_cross_entropy_when_every_decision_is_safe():
    torch = pytest.importorskip("torch")
    h = helpers()
    c = cfg()
    logits, y, w = encoded_rows(torch, [[c["sequence_margin"] + 2.0] * 4], c)
    total = float(h["boundary_focus_loss"](StubModel(logits), None, y, w, c, torch))
    raw = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction="none"
    ).reshape_as(y)
    active = y.ne(-100)
    expected = float((raw * w * active).sum() / (w * active).sum())
    assert total == pytest.approx(expected, abs=1e-4)


def test_v028_loss_sees_a_weak_first_token_that_v027_could_not():
    """A row whose only weak decision is its first token.

    v0.0.27's sequence term selected on the copy weight, which encode_row
    overwrites at that position, so this row contributed nothing.
    """
    torch = pytest.importorskip("torch")
    h = helpers()
    c = cfg()
    weak_first = [[-1.0] + [c["sequence_margin"] + 2.0] * 3]
    logits, y, w = encoded_rows(torch, weak_first, c)

    copy_mask, span_mask = h["v028_supervised_span"](y, w, c, torch)
    copy_gap, _ = h["v028_gaps"](logits, y, copy_mask, torch)
    span_gap, _ = h["v028_gaps"](logits, y, span_mask, torch)
    # Invisible under the copy mask, visible under the span mask.
    assert float(copy_gap.min()) > c["sequence_margin"]
    assert float(span_gap.min()) == pytest.approx(-1.0, abs=1e-5)

    _, worst_gap, _ = h["v028_row_worst"](span_gap, span_mask, c, torch)
    assert float(worst_gap[0]) == pytest.approx(-1.0, abs=1e-5)


def test_v028_progress_calls_the_v027_result_an_advance():
    """v0.0.27 gained 0.2025 of margin health and was reported FLAT."""
    h = helpers()
    c = cfg()

    def diag(exact, continuation, health):
        return {
            "metrics": {
                "expanded_exact_copy_rate": exact,
                "expanded_continuation_top1_rate": continuation,
                "expanded_sequence_margin_health": health,
                "expanded_full_sequence_margin_health": health,
                "exact_copy_rate": 4 / 9,
                "continuation_top1_rate": 0.8696,
            }
        }

    replay = h["v028_progress"](diag(0.2222, 0.8151, -0.8871), diag(0.2222, 0.8205, -0.6846), c)
    assert replay["verdict"] == "ADVANCED"
    assert replay["deltas"]["expanded_sequence_margin_health"]["gain"] == pytest.approx(
        0.2025, abs=1e-4
    )

    stalled = h["v028_progress"](diag(0.2222, 0.8205, -0.6846), diag(0.2222, 0.8210, -0.6800), c)
    assert stalled["verdict"] == "FLAT"


def test_v028_selection_uses_the_full_span_statistic():
    h = helpers()
    score = h["v028_score"]

    def checkpoint(full_health, exact):
        return {
            "gates": {"not_regressed": True},
            "metrics": {
                "expanded_full_sequence_margin_health": full_health,
                "expanded_exact_copy_rate": exact,
                "expanded_continuation_top1_rate": 0.8205,
                "exact_copy_rate": 4 / 9,
                "continuation_top1_rate": 0.8696,
            },
        }

    assert score(checkpoint(-0.50, 19 / 90), 0.9) > score(checkpoint(-0.68, 20 / 90), 0.9)
    regressed = checkpoint(0.9, 30 / 90)
    regressed["gates"]["not_regressed"] = False
    assert score(checkpoint(-0.68, 20 / 90), 0.9) > score(regressed, 0.1)
    # The seeded baseline cannot be displaced by a tie.
    assert score(checkpoint(-0.68, 20 / 90), -1e9) > score(checkpoint(-0.68, 20 / 90), 0.9)


def test_v028_distribution_helpers_expose_the_reachable_band():
    h = helpers()
    c = cfg()
    gaps = [-2.4, -1.6, -1.1, -0.6, -0.3, -0.05, 0.2, 0.9, 1.5]
    percentiles = h["_v028_percentiles"](gaps, c["margin_report_percentiles"])
    assert percentiles["p50"] == -0.3
    histogram = h["_v028_histogram"](gaps, c["margin_histogram_edges"])
    assert histogram["<-2.00"] == 1
    assert histogram[">=1.20"] == 1
    assert sum(histogram.values()) == len(gaps)
    within = [g for g in gaps if c["boundary_band_low"] <= g < 0.0]
    assert within == [-0.6, -0.3, -0.05]


def test_v028_format_parity_still_holds_for_the_carried_curriculum():
    h = helpers()
    data = load(DATA_V026, "ember_sft_data_v026_for_v028")
    parity = h["assert_format_parity"](data, data.build_examples("train", 3600))
    assert parity["templates_supported"] == len(data.DIAGNOSTICS)
    assert parity["min_train_rows_for_a_held_out_template"] > 0
