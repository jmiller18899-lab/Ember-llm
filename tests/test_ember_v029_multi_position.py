"""Guards for the Ember v0.0.29 multi-position objective.

v0.0.28 flipped eight strings by pushing each row's single worst decision. The
v0.0.28 numbers say the average failing row carries about 1.66 wrong decisions,
so one pass can retire at most one of them. These tests pin the top-k statistic
that follows, the gate-comparison fix that v0.0.28's 5/9 exposed, and the band
accounting that says whether the boundary pipeline is still flowing.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAINER_V016 = ROOT / "jobs" / "ember_hf_sft_v016.py"
TRAINER_V028 = ROOT / "jobs" / "ember_hf_sft_v028.py"
TRAINER_V029 = ROOT / "jobs" / "ember_hf_sft_v029.py"
DATA_V026 = ROOT / "jobs" / "ember_sft_data_v026.py"
CONFIG = ROOT / "config" / "ember_multi_position_v0.0.29.json"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helpers(path: Path = TRAINER_V029, name: str = "ember_hf_sft_v029_helpers"):
    namespace: dict = {}
    exec(load(path, name).HELPERS, namespace)
    return namespace


def cfg():
    return json.loads(CONFIG.read_text())


def encoded_rows(torch, rows, c, vocab=8):
    """(logits, y, w) laid out exactly as base.encode_row does."""
    width = max(len(r) for r in rows) + 1
    logits = torch.zeros((len(rows), width, vocab))
    y = torch.full((len(rows), width), -100, dtype=torch.long)
    w = torch.zeros((len(rows), width))
    for i, row in enumerate(rows):
        for t, gap in enumerate(row):
            logits[i, t, 1 + (t % (vocab - 2))] = gap
            y[i, t] = 1 + (t % (vocab - 2))
        y[i, len(row)] = 1
        w[i, : len(row)] = float(c["copy_token_weight"])
        w[i, 0] = float(c["first_token_weight"])
        w[i, len(row) - 1] = float(c["eos_token_weight"])
        w[i, len(row)] = 1.0
    return logits, y, w


def worst_k(h, torch, rows, c):
    logits, y, w = encoded_rows(torch, rows, c)
    _, span = h["v029_supervised_span"](y, w, c, torch)
    gap, _ = h["v029_gaps"](logits, y, span, torch)
    return h["v029_row_worst"](gap, span, c, torch)


def test_v029_transform_targets_exist_and_runtime_compiles():
    trainer = load(TRAINER_V029, "ember_hf_sft_v029_targets")
    assert trainer.unmatched_transform_targets(TRAINER_V016.read_text()) == []
    runtime = trainer.apply_transforms(TRAINER_V016.read_text())
    compile(runtime, "ember_hf_sft_v029_runtime", "exec")

    assert runtime.index('print("EMBER_HF_V029_PREFLIGHT=PASS"') < runtime.index(
        "torch.cuda.is_available()"
    )
    assert runtime.index("assert_format_parity(data, train_rows)") < runtime.index(
        "source_path, source_info = resolve_v028_source"
    )
    assert "best_score = v029_score(baseline, -1e9)" in runtime
    assert "loss = multi_position_loss(model, x, y, w, cfg, torch) / accum" in runtime
    assert 'sample_multi_position_indices(train, int(cfg["batch_size"])' in runtime
    assert "EMBER_V029_GATE_DISTANCE" in runtime
    assert "EMBER_V029_BAND_FLOW" in runtime


def test_v029_continues_from_v028_the_strongest_checkpoint():
    trainer = load(TRAINER_V029, "ember_hf_sft_v029_source")
    runtime = trainer.apply_transforms(TRAINER_V016.read_text())
    assert 'SOURCE_REPO = "Jmiller18899/ember-v0.0.28-t4"' in runtime
    assert 'if str(source_cfg.get("version")) != "0.0.28":' in runtime
    assert "ember-v0.0.15-t4" not in runtime and "ember-v0.0.27-t4" not in runtime
    assert "ember_multi_position_v0.0.29.json" in runtime
    assert "ember_sft_data_v026.py" in runtime          # curriculum still unchanged


def test_v029_config_changes_only_the_worst_k_statistic():
    c = cfg()
    previous = json.loads((ROOT / "config" / "ember_boundary_focus_v0.0.28.json").read_text())
    assert c["version"] == "0.0.29" and c["phase"] == "multi-position-repair"
    assert c["source_model_name"] == "ember-v0.0.28-t4"
    for key in (
        "learning_rate", "max_steps", "batch_size", "gradient_accumulation_steps",
        "copy_token_weight", "first_token_weight", "eos_token_weight", "margin",
        "margin_loss_weight", "sequence_margin", "sequence_margin_loss_weight",
        "boundary_band_low", "out_of_band_weight", "hard_batch_fraction",
        "margin_health_floor", "margin_health_ceiling", "train_examples",
    ):
        assert c[key] == previous[key], f"{key} drifted from v0.0.28"
    # 730 x (1 - 0.8658) = 98 wrong continuation decisions over 62 failing rows,
    # plus 5 first-token failures, is about 1.66 wrong decisions per failing row.
    assert c["sequence_worst_k"] == 2
    assert c["gate_tolerance"] > 0


def test_v029_gate_comparison_accepts_a_rate_that_equals_its_bar():
    """The bug v0.0.28 exposed by landing on exactly 5/9."""
    h = helpers()
    c = cfg()
    assert 5 / 9 < c["minimum_exact_copy_rate"], "the stored decimal is above 5/9, as expected"
    assert h["v029_meets"](5 / 9, c["minimum_exact_copy_rate"], c)
    assert h["v029_meets"](7 / 9, c["minimum_first_token_top20_rate"], c)
    assert h["v029_meets"](8 / 9, c["minimum_first_token_top5_rate"], c)
    # It still refuses a rate that is genuinely short.
    assert not h["v029_meets"](4 / 9, c["minimum_exact_copy_rate"], c)
    assert not h["v029_meets"](0.8986, c["minimum_continuation_top1_rate"], c)


def test_v029_worst_k_reduces_to_v028_at_k_equals_one():
    torch = pytest.importorskip("torch")
    h29 = helpers()
    h28 = helpers(TRAINER_V028, "ember_hf_sft_v028_helpers_for_v029")
    c = dict(cfg())
    c["sequence_worst_k"] = 1
    rows = [[-0.4, 0.1, -0.9, 2.0]]

    logits, y, w = encoded_rows(torch, rows, c)
    _, span = h29["v029_supervised_span"](y, w, c, torch)
    gap, _ = h29["v029_gaps"](logits, y, span, torch)
    new_sum, new_gap, _ = h29["v029_row_worst"](gap, span, c, torch)
    old_max, old_gap, _ = h28["v028_row_worst"](gap, span, c, torch)
    assert float(new_sum[0]) == pytest.approx(float(old_max[0]), abs=1e-6)
    assert float(new_gap[0]) == pytest.approx(float(old_gap[0]), abs=1e-6)


def test_v029_worst_k_pushes_a_second_bad_decision_that_v028_ignored():
    """The whole point: the average failing row has more than one wrong decision."""
    torch = pytest.importorskip("torch")
    h = helpers()
    c = cfg()
    margin = c["sequence_margin"]
    safe = margin + 2.0

    one_bad, _, _ = worst_k(h, torch, [[-0.4, safe, safe, safe]], c)
    two_bad, _, _ = worst_k(h, torch, [[-0.4, -0.3, safe, safe]], c)

    # k = 2 sums the two worst hinges, so the second failure is now carried.
    assert float(one_bad[0]) == pytest.approx(margin + 0.4, abs=1e-5)
    assert float(two_bad[0]) == pytest.approx((margin + 0.4) + (margin + 0.3), abs=1e-5)
    assert float(two_bad[0]) > float(one_bad[0])


def test_v029_worst_k_handles_the_shortest_row_encode_row_can_produce():
    """Three decisions: first token, one copy position, EOS.

    encode_row always leaves at least one copy-weighted position, because the
    copy block spans the value's tokens plus the trailing newline, so the span
    mask is never empty for a real row.
    """
    torch = pytest.importorskip("torch")
    h = helpers()
    c = cfg()
    safe = c["sequence_margin"] + 2.0
    logits, y, w = encoded_rows(torch, [[-0.2, safe, -0.1]], c)
    copy_mask, span = h["v029_supervised_span"](y, w, c, torch)
    assert int(copy_mask.sum()) == 1 and int(span.sum()) == 3

    total, gap, has_targets = worst_k(h, torch, [[-0.2, safe, -0.1]], c)
    assert bool(has_targets[0])
    assert float(gap[0]) == pytest.approx(-0.2, abs=1e-5)
    assert float(total[0]) == pytest.approx(
        (c["sequence_margin"] + 0.2) + (c["sequence_margin"] + 0.1), abs=1e-5
    )


def test_v029_mining_still_prefers_flippable_rows_over_hopeless_ones():
    torch = pytest.importorskip("torch")
    h = helpers()
    c = cfg()

    def priority(rows):
        total, gap, _ = worst_k(h, torch, rows, c)
        far = (float(c["sequence_margin"]) - float(c["boundary_band_low"])) * float(
            c["out_of_band_weight"]
        )
        in_band = gap.ge(float(c["boundary_band_low"]))
        return float(torch.where(in_band, total, torch.full_like(total, far))[0])

    safe = c["sequence_margin"] + 2.0
    assert priority([[-0.05, safe, safe, safe]]) > priority([[-3.0, safe, safe, safe]])
    assert priority([[-3.0, safe, safe, safe]]) > priority([[safe, safe, safe, safe]])


def test_v029_loss_reduces_to_cross_entropy_when_every_decision_is_safe():
    torch = pytest.importorskip("torch")
    h = helpers()
    c = cfg()
    logits, y, w = encoded_rows(torch, [[c["sequence_margin"] + 2.0] * 4], c)

    class StubModel:
        def __call__(self, x, _):
            return logits, None

    total = float(h["multi_position_loss"](StubModel(), None, y, w, c, torch))
    raw = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction="none"
    ).reshape_as(y)
    active = y.ne(-100)
    assert total == pytest.approx(float((raw * w * active).sum() / (w * active).sum()), abs=1e-4)


def test_v029_band_flow_reconstructs_the_v028_transitions():
    """v0.0.28 went 16 -> 21 in the band while 8 crossed, so 13 entered from below."""
    h = helpers()
    c = cfg()

    def side(gaps):
        return {"expanded_cases": [{"value": f"case-{i}", "span_min_gap": g}
                                   for i, g in enumerate(gaps)]}

    # 8 in-band cases cross; 13 below-band cases rise into the band.
    before = [-0.3] * 16 + [-1.4] * 54 + [0.5] * 20
    after = [0.2] * 8 + [-0.3] * 8 + [-0.3] * 13 + [-1.4] * 41 + [0.5] * 20
    flow = h["v029_band_flow"](side(before), side(after), c)
    assert flow["entered_band_from_below"] == 13
    assert len(flow["crossed_into_correct"]) == 8
    assert flow["fell_out_of_correct"] == 0
    assert flow["transitions"]["within_reach"]["correct"] == 8


def test_v029_gate_distance_shows_that_a_0_90_gate_on_69_tokens_is_really_63_69():
    h = helpers()
    c = cfg()
    final = {
        "expanded_cases": [{"value": f"c{i}"} for i in range(90)],
        "metrics": {
            "expanded_exact_copy_rate": 28 / 90,
            "expanded_continuation_top1_rate": 632 / 730,
            "expanded_continuation_tokens": 730,
            "exact_copy_rate": 5 / 9,
            "continuation_top1_rate": 62 / 69,
            "legacy_continuation_tokens": 69,
        },
    }
    d = h["v029_gate_distance"](final, c)
    assert d["expanded_exact_copy_rate"]["needs"] == "36/90"
    assert d["expanded_exact_copy_rate"]["short_by"] == 8
    assert d["expanded_continuation_top1_rate"]["needs"] == "657/730"
    assert d["expanded_continuation_top1_rate"]["short_by"] == 25
    # The gate reads 0.90 but on 69 tokens it can only be met at 63/69.
    assert d["continuation_top1_rate"]["needs"] == "63/69"
    assert d["continuation_top1_rate"]["short_by"] == 1
    assert d["continuation_top1_rate"]["effective_gate"] == pytest.approx(0.9130, abs=1e-4)
    # Legacy exact copy at exactly 5/9 is already met, thanks to the tolerance.
    assert d["exact_copy_rate"]["short_by"] == 0


def test_v029_protection_holds_the_v028_result():
    c = cfg()
    assert c["protected_legacy_exact_copy_rate"] == c["minimum_exact_copy_rate"]
    assert c["protected_expanded_exact_copy_rate"] <= 28 / 90
    assert c["protected_expanded_continuation_top1_rate"] <= 0.8658
    assert c["protected_expanded_first_token_top1_rate"] <= 0.9444
    assert c["protected_expanded_full_sequence_margin_health"] <= -0.5865


def test_v029_format_parity_still_holds_for_the_carried_curriculum():
    h = helpers()
    data = load(DATA_V026, "ember_sft_data_v026_for_v029")
    parity = h["assert_format_parity"](data, data.build_examples("train", 3600))
    assert parity["templates_supported"] == len(data.DIAGNOSTICS)
