"""Guards for Ember v0.0.33: the copy target moves inside a tool-call envelope.

The v0.0.32 gate found the promoted checkpoint producing a correct envelope, a
correct tool name and a correct argument key while failing exactly one check --
arguments_grounded -- on all four tool calls, on a model that reproduces 43 of
90 held-out literal strings exactly. These tests pin the one change that
follows: the copy weight lands on the value inside the envelope, and nothing
else about the phase moves.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAINER_V016 = ROOT / "jobs" / "ember_hf_sft_v016.py"
TRAINER_V033 = ROOT / "jobs" / "ember_hf_sft_v033.py"
DATA_V026 = ROOT / "jobs" / "ember_sft_data_v026.py"
CONFIG_V031 = ROOT / "config" / "ember_multi_position_v0.0.31.json"
CONFIG_V033 = ROOT / "config" / "ember_envelope_copy_v0.0.33.json"

EOT = "<|endoftext|>"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def helpers():
    namespace: dict = {"json": json}
    exec(load(TRAINER_V033, "ember_hf_sft_v033_helpers").HELPERS, namespace)
    return namespace


@pytest.fixture(scope="module")
def cfg():
    """The real config, with room for the character-level test tokenizer.

    CharTokenizer emits one id per character, so these prompts need far more
    than the 256-token block the real BPE uses. Everything else is untouched.
    """
    loaded = json.loads(CONFIG_V033.read_text())
    loaded["block_size"] = 4096
    return loaded


class CharTokenizer:
    """Prefix-exact tokenizer: one id per character, EOT as a single id.

    Real BPE can merge across a boundary, which is why the encoder checks and
    the preflight has a floor. Here the mapping is exact, so the weight layout
    and the index arithmetic can be asserted precisely.
    """

    def encode(self, text):
        ids = []
        i = 0
        while i < len(text):
            if text.startswith(EOT, i):
                ids.append(1)
                i += len(EOT)
            else:
                ids.append(ord(text[i]) % 4000 + 10)
                i += 1
        return ids


class MergingTokenizer(CharTokenizer):
    """Rewrites the boundary token, as a BPE merge across it would.

    Dropping a token would not be caught: a truncated prefix is still a prefix.
    Changing the last token's identity is what a real merge does.
    """

    def encode(self, text):
        ids = super().encode(text)
        if text.endswith('"'):
            ids[-1] = 7777
        return ids


def bare_row(value="Q7M4X", kind="entity"):
    return {
        "id": "row-1", "kind": kind, "value": value,
        "prompt": (
            "<|system|>\nYou are Ember.\n<|user|>\nIgnore old=AAAA and fallback=BBBB. "
            f"TARGET={value}. Reply with TARGET exactly once.\n<|assistant|>\n"
        ),
    }


# --- shape -----------------------------------------------------------------

def test_the_completion_is_the_envelope_the_checkpoint_already_emits(helpers):
    row = helpers["to_envelope_row"](bare_row())
    assert row["completion"] == (
        '<|tool|>\n{"arguments":{"location":"Q7M4X"},"name":"weather"}\n<|endoftext|>\n'
    )
    assert row["completion_head"] == '<|tool|>\n{"arguments":{"location":"'
    # Key order matches the v0.0.31 generations, so the envelope being taught is
    # the one the model already produces.
    assert row["completion"].index('"arguments"') < row["completion"].index('"name"')


def test_the_target_appears_exactly_once_on_each_side(helpers):
    row = helpers["to_envelope_row"](bare_row())
    assert row["prompt"].count("Q7M4X") == 1
    assert row["completion"].count("Q7M4X") == 1


def test_the_tool_mapping_covers_the_tools_the_eval_battery_uses(helpers):
    pairs = set(helpers["ENVELOPE_TOOLS"].values())
    for tool, key in (("weather", "location"), ("calculator", "expression"),
                      ("web_search", "query")):
        assert (tool, key) in pairs, f"{tool}/{key} is in the v0.0.8 battery"
    # Every kind in the carried curriculum has a mapping.
    data = load(DATA_V026, "ember_sft_data_v026_for_v033")
    for kind in data.KINDS:
        assert kind in helpers["ENVELOPE_TOOLS"], kind


def test_distractors_are_carried_over_from_the_bare_value_curriculum(helpers):
    old, fallback = helpers["envelope_distractors"](bare_row()["prompt"])
    assert (old, fallback) == ("AAAA", "BBBB")
    row = helpers["to_envelope_row"](bare_row())
    assert "old=AAAA" in row["prompt"] and "fallback=BBBB" in row["prompt"]


def test_every_carried_curriculum_row_converts(helpers):
    data = load(DATA_V026, "ember_sft_data_v026_convert")
    for row in data.build_examples("train", 300):
        envelope = helpers["to_envelope_row"](row)
        assert envelope["value"] == row["value"]
        assert envelope["completion"].startswith(envelope["completion_head"])
        assert envelope["completion"].endswith(EOT + "\n")


# --- the weight layout, which is the whole phase ---------------------------

def test_copy_weight_lands_on_the_value_and_not_the_punctuation(helpers, cfg):
    torch = pytest.importorskip("torch")
    row = helpers["to_envelope_row"](bare_row())
    item, reason = helpers["encode_envelope_row"](CharTokenizer(), row, cfg, torch)
    assert reason is None and item is not None

    w, y = item["w"], item["y"]
    start, end = item["value_span"]
    # The value is five characters, so five predicting positions.
    assert end - start == len("Q7M4X")
    # First value token carries first_token_weight, the rest carry copy weight.
    assert float(w[start]) == pytest.approx(cfg["first_token_weight"], abs=1e-6)
    for pos in range(start + 1, end):
        assert float(w[pos]) == pytest.approx(cfg["copy_token_weight"], abs=1e-6), pos
    # Everything else that is supervised is scaffolding or EOS.
    assert float(w[item["eos_target"]]) == pytest.approx(cfg["eos_token_weight"], abs=1e-6)
    supervised = [i for i in range(len(w)) if float(w[i]) > 0]
    scaffolding = [i for i in supervised
                   if not (start <= i < end) and i != item["eos_target"]]
    assert scaffolding, "the envelope should still be supervised, just weakly"
    for pos in scaffolding:
        assert float(w[pos]) == pytest.approx(cfg["envelope_token_weight"], abs=1e-6), pos
    # Nothing outside the completion is supervised.
    assert all(int(y[i]) == -100 for i in range(len(y)) if float(w[i]) == 0)


def test_the_value_span_predicts_the_value_and_nothing_else(helpers, cfg):
    torch = pytest.importorskip("torch")
    tokenizer = CharTokenizer()
    row = helpers["to_envelope_row"](bare_row())
    item, _ = helpers["encode_envelope_row"](tokenizer, row, cfg, torch)
    start, end = item["value_span"]
    predicted = [int(item["y"][pos]) for pos in range(start, end)]
    assert predicted == tokenizer.encode("Q7M4X")


def test_the_supervised_span_reaches_the_value_boundaries_and_the_eos(helpers, cfg):
    """Under the envelope encoder the copy block no longer touches the end, so
    the EOS decision has to be added back explicitly."""
    torch = pytest.importorskip("torch")
    row = helpers["to_envelope_row"](bare_row())
    item, _ = helpers["encode_envelope_row"](CharTokenizer(), row, cfg, torch)
    y = item["y"].unsqueeze(0)
    w = item["w"].unsqueeze(0)
    copy_mask, span = helpers["v033_supervised_span"](y, w, cfg, torch)
    start, end = item["value_span"]

    assert copy_mask[0][start] is not None
    assert not bool(copy_mask[0][start]), "the first value token is not copy-weighted"
    assert all(bool(copy_mask[0][pos]) for pos in range(start + 1, end))
    # The span adds the value's first token, the token after it, and EOS.
    assert bool(span[0][start])
    assert bool(span[0][end])
    assert bool(span[0][item["eos_target"]]), "the envelope must be made to close"


# --- the assumption that cannot be checked without the real tokenizer -------

def test_a_row_whose_value_boundary_merges_is_rejected_not_mis_encoded(helpers, cfg):
    torch = pytest.importorskip("torch")

    item, reason = helpers["encode_envelope_row"](
        MergingTokenizer(), helpers["to_envelope_row"](bare_row()), cfg, torch
    )
    assert item is None
    assert reason in {"head_value_boundary", "value_tail_boundary", "prompt_head_boundary"}


def test_too_many_unencodable_rows_fails_the_preflight_rather_than_the_paid_run(helpers, cfg):
    torch = pytest.importorskip("torch")

    rows = [bare_row(f"CODE{i:03d}") for i in range(20)]
    with pytest.raises(RuntimeError, match="encode cleanly inside the envelope"):
        helpers["encode_envelope_rows"](MergingTokenizer(), rows, cfg, torch, "train")

    # And a clean tokenizer reports a full encodable fraction.
    encoded, summary = helpers["encode_envelope_rows"](CharTokenizer(), rows, cfg, torch, "train")
    assert len(encoded) == 20
    assert summary["encodable_fraction"] == 1.0
    assert summary["rejected"] == {}


def test_a_single_token_value_is_skipped_rather_than_encoded_without_a_copy_span(helpers, cfg):
    """A one-token value leaves no copy block once the first token is
    re-weighted, so the row would train nothing. It is skipped and counted."""
    torch = pytest.importorskip("torch")
    row = helpers["to_envelope_row"](bare_row("Q"))
    item, reason = helpers["encode_envelope_row"](CharTokenizer(), row, cfg, torch)
    assert item is None and reason == "value_too_short"


# --- the phase changes one thing -------------------------------------------

def test_only_the_target_placement_changes(cfg):
    previous = json.loads(CONFIG_V031.read_text())
    for key in (
        "learning_rate", "max_steps", "batch_size", "gradient_accumulation_steps",
        "warmup_steps", "min_lr_ratio", "copy_token_weight", "first_token_weight",
        "eos_token_weight", "margin", "margin_loss_weight", "sequence_margin",
        "sequence_margin_loss_weight", "sequence_worst_k", "boundary_band_low",
        "out_of_band_weight", "hard_batch_fraction", "hard_candidate_multiplier",
        "train_examples", "margin_health_floor", "margin_health_ceiling",
    ):
        assert cfg[key] == previous[key], f"{key} drifted; only placement should move"
    assert cfg["source_model_name"] == "ember-v0.0.31-t4"
    assert cfg["phase"] == "envelope-placed-copy"
    assert cfg["envelope_token_weight"] < cfg["copy_token_weight"]
    assert 0.0 < cfg["minimum_envelope_encodable_fraction"] <= 1.0


def test_the_bare_value_capability_is_protected(cfg):
    # v0.0.31 measured 43/90 expanded exact copy and 662/730 continuation.
    assert cfg["protected_expanded_exact_copy_rate"] <= 43 / 90
    assert cfg["protected_expanded_continuation_top1_rate"] <= 662 / 730
    tolerance = cfg["gate_tolerance"]
    # 6/9 is 0.6666666666666666 against a stored 0.6666666667; the tolerance
    # added in v0.0.29 is what makes that bar reachable.
    assert 6 / 9 >= cfg["protected_legacy_exact_copy_rate"] - tolerance
    assert cfg["protected_legacy_continuation_top1_rate"] <= 65 / 69


def test_the_runtime_trains_on_envelopes_and_continues_from_v031():
    trainer = load(TRAINER_V033, "ember_hf_sft_v033_runtime")
    assert trainer.unmatched_transform_targets(TRAINER_V016.read_text()) == []
    runtime = trainer.apply_transforms(TRAINER_V016.read_text())
    compile(runtime, "ember_hf_sft_v033_runtime", "exec")

    assert 'SOURCE_REPO = "Jmiller18899/ember-v0.0.31-t4"' in runtime
    assert 'if str(source_cfg.get("version")) != "0.0.31":' in runtime
    assert "ember_envelope_copy_v0.0.33.json" in runtime
    assert "ember_sft_data_v026.py" in runtime, "the curriculum values are unchanged"
    # Training data goes through the envelope encoder, not the bare-value one.
    assert "encode_envelope_rows(tokenizer, train_rows" in runtime
    assert "base.encode_row(tokenizer, row, cfg, torch) for row in train_rows" not in runtime
    # Every prior guard survives.
    assert "best_score = v033_score(baseline, -1e9)" in runtime
    assert "assert_format_parity(data, train_rows)" in runtime
    assert runtime.index('print("EMBER_HF_V033_PREFLIGHT=PASS"') < runtime.index(
        "torch.cuda.is_available()"
    )
    assert "loss = multi_position_loss(model, x, y, w, cfg, torch) / accum" in runtime


def test_selection_leads_on_the_slot_but_cannot_trade_away_bare_value_copying(helpers):
    score = helpers["v033_score"]

    def checkpoint(slot_health, slot_exact, expanded_exact, protected=True):
        return {
            "gates": {"not_regressed": protected},
            "metrics": {
                "envelope_slot_margin_health": slot_health,
                "envelope_slot_exact_rate": slot_exact,
                "expanded_full_sequence_margin_health": -0.24,
                "expanded_exact_copy_rate": expanded_exact,
                "expanded_continuation_top1_rate": 662 / 730,
                "exact_copy_rate": 6 / 9,
                "continuation_top1_rate": 65 / 69,
            },
        }

    better_slot = checkpoint(-0.30, 20 / 90, 43 / 90)
    worse_slot = checkpoint(-0.80, 18 / 90, 43 / 90)
    assert score(better_slot, 0.9) > score(worse_slot, 0.9)

    # A checkpoint that wins the slot by losing the bare-value capability is
    # disqualified before the slot metrics are even compared.
    traded = checkpoint(0.9, 60 / 90, 10 / 90, protected=False)
    assert score(worse_slot, 0.9) > score(traded, 0.1)

    # The seeded baseline still cannot be displaced by a tie.
    assert score(better_slot, -1e9) > score(better_slot, 0.9)


def test_the_preflight_reports_the_slot_baseline_before_any_gpu_guard():
    trainer = load(TRAINER_V033, "ember_hf_sft_v033_preflight")
    runtime = trainer.apply_transforms(TRAINER_V016.read_text())
    assert "EMBER_V033_BASELINE_SLOT_EXACT" in runtime
    assert "EMBER_V033_TRAIN_ENCODING" in runtime
    assert runtime.index("EMBER_V033_BASELINE_SLOT_EXACT") < runtime.index(
        "torch.cuda.is_available()"
    )


def test_the_preflight_persists_its_report_rather_than_only_printing_it():
    """A detached job's stdout lives only in its log stream.

    The v0.0.32 evaluator had to learn this after a run produced numbers nobody
    could read back. The preflight returns before every other upload in the
    scaffold, so it needs its own.
    """
    trainer = load(TRAINER_V033, "ember_hf_sft_v033_preflight_report")
    runtime = trainer.apply_transforms(TRAINER_V016.read_text())
    target = "preflight/v0.0.33-preflight-latest.json"
    assert target in runtime
    assert runtime.index(target) < runtime.index("torch.cuda.is_available()")
    # The report carries the four numbers the launch decision needs.
    for field in ("train_encoding", "validation_encoding",
                  "baseline_slot_exact_rate", "baseline_slot_margin_health"):
        assert field in runtime, field


def test_the_report_contract_still_holds():
    """The same producer/consumer check that v0.0.29 failed."""
    import re
    trainer = load(TRAINER_V033, "ember_hf_sft_v033_contract")
    runtime = trainer.apply_transforms(TRAINER_V016.read_text())
    required = {a or b for a, b in
                re.findall(r'report\["progress"\](?:\["(\w+)"\]|\.get\("(\w+)")', runtime)}
    namespace: dict = {"json": json}
    exec(trainer.HELPERS, namespace)
    cfg = json.loads(CONFIG_V033.read_text())

    def diag(slot_exact):
        return {"metrics": {
            "envelope_slot_exact_rate": slot_exact,
            "envelope_slot_margin_health": -0.4,
            "envelope_slot_continuation_top1_rate": 0.8,
            "envelope_json_valid_rate": 1.0,
            "envelope_tool_name_rate": 1.0,
            "expanded_exact_copy_rate": 43 / 90,
            "expanded_continuation_top1_rate": 662 / 730,
            "expanded_sequence_margin_health": -0.2,
            "expanded_full_sequence_margin_health": -0.24,
            "expanded_sequences_within_reach": 0.2,
            "expanded_first_token_top1_rate": 89 / 90,
            "exact_copy_rate": 6 / 9,
            "continuation_top1_rate": 65 / 69,
            "legacy_continuation_tokens": 69,
            "expanded_continuation_tokens": 730,
        }, "expanded_cases": [{"value": f"c{i}", "span_min_gap": -0.5} for i in range(90)]}

    result = namespace["v033_progress"](diag(0.10), diag(0.30), cfg)
    assert required - set(result) == set()
    assert result["verdict"] == "ADVANCED", "a slot gain must register as progress"
