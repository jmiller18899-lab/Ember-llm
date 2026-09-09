"""Guards for the Ember v0.0.27 sequence-completion objective.

v0.0.26 showed that copy errors are now independent across positions, so exact
copy is limited by each sequence's single worst token rather than by its average
token. These tests pin the two properties that follow: the loss must respond to
the worst position rather than the mean, and checkpoint selection must not be
decided by a one-case fluctuation in a 90-case count.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAINER_V016 = ROOT / "jobs" / "ember_hf_sft_v016.py"
TRAINER_V027 = ROOT / "jobs" / "ember_hf_sft_v027.py"
DATA_V026 = ROOT / "jobs" / "ember_sft_data_v026.py"
CONFIG = ROOT / "config" / "ember_sequence_completion_v0.0.27.json"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helpers():
    """The helper block the transform injects, evaluated on its own."""
    trainer = load(TRAINER_V027, "ember_hf_sft_v027_helpers")
    namespace: dict = {}
    exec(trainer.HELPERS, namespace)
    return namespace


def cfg():
    return json.loads(CONFIG.read_text())


class StubModel:
    """Stands in for EmberGPT: the losses only need (logits, _) back."""

    def __init__(self, logits):
        self._logits = logits

    def __call__(self, x, _):
        return self._logits, None


def build_batch(torch, rows, vocab=6, copy_weight=8.0):
    """rows: list of lists of (correct_token, gap). Builds logits/y/w directly."""
    width = max(len(r) for r in rows)
    logits = torch.zeros((len(rows), width, vocab))
    y = torch.full((len(rows), width), -100, dtype=torch.long)
    w = torch.zeros((len(rows), width))
    for i, row in enumerate(rows):
        for t, (token, gap) in enumerate(row):
            logits[i, t, :] = 0.0
            competitor = (token + 1) % vocab
            logits[i, t, competitor] = 0.0
            logits[i, t, token] = gap
            y[i, t] = token
            w[i, t] = copy_weight
    return logits, y, w


def test_v027_transform_targets_exist_in_the_pinned_scaffold():
    trainer = load(TRAINER_V027, "ember_hf_sft_v027_targets")
    missing = trainer.unmatched_transform_targets(TRAINER_V016.read_text())
    assert missing == [], f"transform targets missing from the scaffold: {missing}"


def test_v027_transformed_trainer_keeps_its_protections():
    trainer = load(TRAINER_V027, "ember_hf_sft_v027_runtime")
    runtime = trainer.apply_transforms(TRAINER_V016.read_text())
    compile(runtime, "ember_hf_sft_v027_runtime", "exec")

    assert "--preflight-only" in runtime
    assert "v0.0.27 training requires explicitly approved T4 GPU" in runtime
    assert runtime.index('print("EMBER_HF_V027_PREFLIGHT=PASS"') < runtime.index(
        "torch.cuda.is_available()"
    )
    assert runtime.index("assert_format_parity(data, train_rows)") < runtime.index(
        "source_path, source_info = resolve_v026_source"
    )
    assert "best_score = v027_score(baseline, -1e9)" in runtime
    assert "step=-1" in runtime
    assert "loss = sequence_completion_loss(model, x, y, w, cfg, torch) / accum" in runtime
    assert "sample_sequence_hard_indices(train, int(cfg[\"batch_size\"])" in runtime
    assert "v027_progress(baseline, final, cfg)" in runtime


def test_v027_continues_from_v026_without_changing_the_curriculum():
    trainer = load(TRAINER_V027, "ember_hf_sft_v027_source")
    runtime = trainer.apply_transforms(TRAINER_V016.read_text())
    assert 'SOURCE_REPO = "Jmiller18899/ember-v0.0.26-t4"' in runtime
    assert "ember-v0.0.15-t4" not in runtime
    assert "ember_sequence_completion_v0.0.27.json" in runtime
    # The curriculum is deliberately unchanged, so the data pin still names the
    # commit that introduced it.
    assert "ember_sft_data_v026.py" in runtime
    assert len(trainer.DATA_COMMIT) == 40 and len(trainer.CONFIG_COMMIT) == 40


def test_v027_config_changes_the_objective_and_nothing_else():
    c = cfg()
    assert c["version"] == "0.0.27"
    assert c["phase"] == "sequence-completion-repair"
    assert c["source_model_name"] == "ember-v0.0.26-t4"
    # Everything v0.0.26 established stays put: this run tests the objective.
    assert c["learning_rate"] == 1.2e-6
    assert c["copy_token_weight"] == 8.0
    assert c["first_token_weight"] == 0.4
    assert c["eos_token_weight"] == 2.5
    assert c["train_examples"] == 3600
    # The new part: the worst position needs a wider cushion than the average one.
    assert c["sequence_margin"] > c["margin"]
    assert c["sequence_margin_loss_weight"] >= c["margin_loss_weight"]
    assert 0.0 < c["hard_batch_fraction"] < 1.0
    # Protection sits at the v0.0.26 result, with slack only where the metric is
    # coarse. Expanded exact copy has a standard deviation near four cases.
    assert c["protected_legacy_exact_copy_rate"] == 0.4444444444
    assert c["protected_expanded_exact_copy_rate"] < 0.2222
    assert c["protected_expanded_continuation_top1_rate"] <= 0.8151


def test_v027_selection_prefers_margin_health_over_a_one_case_fluctuation():
    """The v0.0.26 selection scenario, replayed against the new score.

    Step 219 measured 20/90 exact copy at 0.8151 continuation; a later
    checkpoint measured 19/90 at 0.8300. Independence predicts 18.8 and 21.6
    cases respectively, so the later checkpoint is the better model and the old
    selector took the worse one.
    """
    h = helpers()
    score = h["v027_score"]

    def checkpoint(margin_health, expanded_exact, expanded_continuation):
        return {
            "gates": {"not_regressed": True},
            "metrics": {
                "expanded_sequence_margin_health": margin_health,
                "expanded_exact_copy_rate": expanded_exact,
                "expanded_continuation_top1_rate": expanded_continuation,
                "exact_copy_rate": 4 / 9,
                "continuation_top1_rate": 0.8696,
            },
        }

    step_219 = checkpoint(0.42, 20 / 90, 0.8151)
    later = checkpoint(0.55, 19 / 90, 0.8300)
    assert score(later, 0.9) > score(step_219, 0.9)

    # A regressed checkpoint can never win, however good its margins look.
    regressed = checkpoint(0.99, 30 / 90, 0.95)
    regressed["gates"]["not_regressed"] = False
    assert score(step_219, 0.9) > score(regressed, 0.1)

    # The seeded baseline carries a +1e9 tiebreaker, so a tie cannot displace it.
    baseline = checkpoint(0.42, 20 / 90, 0.8151)
    assert score(baseline, -1e9) > score(step_219, 0.9)


def test_v027_progress_separates_a_real_advance_from_promotion():
    h = helpers()
    c = cfg()

    def diag(exact, continuation, health):
        return {
            "metrics": {
                "expanded_exact_copy_rate": exact,
                "expanded_continuation_top1_rate": continuation,
                "expanded_sequence_margin_health": health,
                "exact_copy_rate": 4 / 9,
                "continuation_top1_rate": 0.8696,
            }
        }

    # v0.0.26's own numbers: promotion failed, but the run advanced.
    v026 = h["v027_progress"](diag(0.1889, 0.7534, 0.1), diag(0.2222, 0.8151, 0.4), c)
    assert v026["verdict"] == "ADVANCED"
    assert v026["deltas"]["expanded_continuation_top1_rate"]["gain"] == pytest.approx(0.0617, abs=1e-3)

    flat = h["v027_progress"](diag(0.2222, 0.8151, 0.4), diag(0.2222, 0.8160, 0.4), c)
    assert flat["verdict"] == "FLAT"

    worse = h["v027_progress"](diag(0.2222, 0.8151, 0.4), diag(0.2000, 0.8000, 0.3), c)
    assert worse["verdict"] == "REGRESSED"


def test_v027_length_buckets_cover_every_sequence_length():
    h = helpers()
    edges = cfg()["length_buckets"]
    assert [h["_v027_bucket"](n, edges) for n in (1, 4, 5, 8, 9, 12, 13, 40)] == [
        "<=4", "<=4", "<=8", "<=8", "<=12", "<=12", ">12", ">12",
    ]


def test_v027_format_parity_still_holds_for_the_carried_curriculum():
    h = helpers()
    data = load(DATA_V026, "ember_sft_data_v026_for_v027")
    train = data.build_examples("train", 3600)
    parity = h["assert_format_parity"](data, train)
    assert parity["held_out_cases"] == len(data.DIAGNOSTICS)
    assert parity["min_train_rows_for_a_held_out_template"] > 0


# --- numerical guards; these need torch, which CI installs before running tests ---


def test_v027_copy_gaps_measure_the_distance_to_the_best_competitor():
    torch = pytest.importorskip("torch")
    h = helpers()
    logits, y, w = build_batch(torch, [[(1, 0.7), (2, 2.5)]])
    gap, best_is_correct, target_mask, _ = h["_v027_copy_gaps"](logits, y, w, cfg(), torch)
    assert torch.allclose(gap, torch.tensor([0.7, 2.5]), atol=1e-5)
    assert bool(best_is_correct.all())
    assert int(target_mask.sum()) == 2


def test_v027_sequence_term_tracks_the_worst_position_not_the_mean():
    """The property the whole phase rests on.

    Two rows with the same *mean* hinge must not score the same: the row whose
    failure is concentrated in one position is the one that cannot produce an
    exact copy, and it must carry more loss.
    """
    torch = pytest.importorskip("torch")
    h = helpers()
    c = cfg()
    margin = c["sequence_margin"]

    concentrated = [[(1, -1.0), (2, margin + 2.0), (3, margin + 2.0), (4, margin + 2.0)]]
    spread = [[(1, margin - 0.55), (2, margin - 0.55), (3, margin - 0.55), (4, margin - 0.55)]]

    def sequence_term(rows):
        logits, y, w = build_batch(torch, rows)
        gap, _, target_mask, _ = h["_v027_copy_gaps"](logits, y, w, c, torch)
        hinge = torch.zeros_like(y, dtype=gap.dtype)
        hinge[target_mask] = torch.nn.functional.relu(margin - gap)
        return float(hinge.max(dim=1).values.mean())

    # Equal mean hinge by construction: 2.2/4 = 0.55 each.
    concentrated_mean = (margin + 1.0) / 4
    assert concentrated_mean == pytest.approx(0.55, abs=1e-6)
    assert sequence_term(concentrated) > sequence_term(spread)
    assert sequence_term(concentrated) == pytest.approx(margin + 1.0, abs=1e-5)
    assert sequence_term(spread) == pytest.approx(0.55, abs=1e-5)


def test_v027_loss_is_zero_extra_when_every_sequence_is_comfortably_correct():
    torch = pytest.importorskip("torch")
    h = helpers()
    c = cfg()
    safe = [[(1, c["sequence_margin"] + 1.0), (2, c["sequence_margin"] + 1.0)]]
    logits, y, w = build_batch(torch, safe)
    gap, _, target_mask, _ = h["_v027_copy_gaps"](logits, y, w, c, torch)
    hinge = torch.zeros_like(y, dtype=gap.dtype)
    hinge[target_mask] = torch.nn.functional.relu(c["sequence_margin"] - gap)
    assert float(hinge.max()) == 0.0

    total = h["sequence_completion_loss"](StubModel(logits), None, y, w, c, torch)
    cross_entropy_only = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction="none"
    ).reshape_as(y)
    active = y.ne(-100)
    expected = float((cross_entropy_only * w * active).sum() / (w * active).sum())
    assert float(total) == pytest.approx(expected, abs=1e-4)


def test_v027_loss_penalises_a_single_bad_position_in_a_good_sequence():
    torch = pytest.importorskip("torch")
    h = helpers()
    c = cfg()
    good = [[(1, c["sequence_margin"] + 1.0)] * 4]
    one_bad = [[(1, c["sequence_margin"] + 1.0)] * 3 + [(2, -1.0)]]
    a = float(h["sequence_completion_loss"](StubModel(build_batch(torch, good)[0]),
                                            None, *build_batch(torch, good)[1:], c, torch))
    logits_b, y_b, w_b = build_batch(torch, one_bad)
    b = float(h["sequence_completion_loss"](StubModel(logits_b), None, y_b, w_b, c, torch))
    assert b > a


def test_v027_hard_mining_selects_the_rows_with_the_worst_single_position():
    torch = pytest.importorskip("torch")
    h = helpers()
    c = dict(cfg())
    c["hard_batch_fraction"] = 0.5
    # 2 mined slots x 8 = 16 candidates drawn from 2 rows, so the failing row is
    # in the pool for any seed worth testing.
    c["hard_candidate_multiplier"] = 8

    # Row 0 fails badly at one position; row 1 is comfortable everywhere.
    rows = [
        [(1, -3.0), (2, 5.0)],
        [(1, 5.0), (2, 5.0)],
    ]
    logits, y, w = build_batch(torch, rows)
    # Each row's x encodes its own index, so the stub can return that row's
    # logits and the test measures selection rather than shape.
    train = [
        {"x": torch.full((2,), i, dtype=torch.long), "y": y[i], "w": w[i]}
        for i in range(len(rows))
    ]

    class IndexedStub:
        def __call__(self, x, _):
            return logits[x[:, 0]], None

    for seed in range(5):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        picked = h["sample_sequence_hard_indices"](
            train, 4, c, generator, IndexedStub(), "cpu", torch
        )
        assert len(picked) == 4
        assert all(0 <= i < len(rows) for i in picked)
        # Row 0 is the only one with a failing position, so both mined slots
        # take it; the remaining two slots stay uniform.
        assert set(picked[:2]) == {0}, f"seed {seed} did not mine the failing row: {picked}"
