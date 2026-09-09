"""Guards for the Ember v0.0.26 format-parity repair.

The v0.0.16 -> v0.0.25 sequence spent ten runs modifying loss shaping and
curriculum volume while the training curriculum could not emit the structure of
four of the nine held-out diagnostic values. These tests make that class of
mistake fail in CPU CI instead of on a paid T4.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "jobs" / "ember_curriculum_audit_v026.py"
DATA_V015 = ROOT / "jobs" / "ember_sft_data_v015.py"
DATA_V026 = ROOT / "jobs" / "ember_sft_data_v026.py"
TRAINER_V015 = ROOT / "jobs" / "ember_hf_sft_v015.py"
TRAINER_V016 = ROOT / "jobs" / "ember_hf_sft_v016.py"
TRAINER_V026 = ROOT / "jobs" / "ember_hf_sft_v026.py"
CONFIG = ROOT / "config" / "ember_format_parity_v0.0.26.json"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_audit(data: Path, diagnostics: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(AUDIT), "--data", str(data), "--diagnostics", str(diagnostics), *extra],
        capture_output=True,
        text=True,
    )


def test_audit_reproduces_the_v015_format_gap():
    """The audit must still report the gap that stalled v0.0.16 through v0.0.25."""
    result = run_audit(DATA_V015, TRAINER_V015, "--assert-parity")
    assert result.returncode == 1, result.stdout
    assert "FORMAT_GAP" in result.stdout
    # model_id, url, path and mixed are the four exact-copy failures whose
    # template the v0.0.15 curriculum never produces; entity differs in
    # structure too but the model copies it correctly anyway.
    for kind in ("model_id", "url", "path", "mixed"):
        assert f"{kind:12s} NO" in result.stdout, f"{kind} should be reported unsupported"


def test_v026_curriculum_supports_every_held_out_template():
    result = run_audit(DATA_V026, DATA_V026, "--assert-parity")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FORMAT_PARITY" in result.stdout


def test_v026_battery_is_large_enough_to_measure_a_real_change():
    data = load(DATA_V026, "ember_sft_data_v026_battery")
    assert len(data.DIAGNOSTICS) == 90
    # A 9-case battery moves in 11.1-point steps, so a one-case difference is
    # indistinguishable from noise. 90 cases resolve ~1.1 points.
    assert 1.0 / len(data.DIAGNOSTICS) < 0.02
    kinds = {kind for kind, _, _ in data.DIAGNOSTICS}
    assert kinds == set(data.KINDS)
    values = [value for _, value, _ in data.DIAGNOSTICS]
    assert len(set(values)) == len(values)
    for _, value, corrupt in data.DIAGNOSTICS:
        assert corrupt != value
    # Generated corruptions are single-character substitutions, so the
    # teacher-forced comparison isolates the token the model got wrong rather
    # than a length difference. The nine hand-written legacy twins predate that
    # rule and are kept verbatim.
    for _, value, corrupt in data.DIAGNOSTICS[9:]:
        assert len(corrupt) == len(value)


def test_v026_legacy_cases_are_preserved_verbatim():
    """The protected v0.0.20 metric only stays comparable if these do not move."""
    data = load(DATA_V026, "ember_sft_data_v026_legacy")
    # The v0.0.15 trainer imports huggingface_hub at module scope, so its
    # battery is read as text rather than imported.
    source = TRAINER_V015.read_text()
    for kind, value, corrupt in data.LEGACY_DIAGNOSTICS:
        assert f'("{value}", "{corrupt}")' in source, f"legacy case drifted: {value}"
    assert len(data.LEGACY_DIAGNOSTICS) == 9
    assert data.DIAGNOSTICS[:9] == data.LEGACY_DIAGNOSTICS


def test_v026_splits_are_disjoint_unbiased_and_leakage_free():
    data = load(DATA_V026, "ember_sft_data_v026_splits")
    train = data.build_examples("train", 3600)
    validation = data.build_examples("validation", 450)
    data.assert_clean(train, validation)
    assert len(train) == 3600 and len(validation) == 450
    for row in train + validation:
        assert row["completion"].startswith(row["value"])
        assert row["completion"].endswith("<|endoftext|>\n")
        assert row["prompt"].count(row["value"]) == 1
        assert "<|tool|>" not in row["completion"]
    # v0.0.15 reserved "2-9A-H" for train and "J-Z" for validation, so no
    # training target could ever begin with a letter from the second half of the
    # alphabet. v0.0.26 splits on stream position instead.
    leading = {row["value"][0] for row in train}
    assert leading & set("JKLMNPQRSTUVWXYZ"), "train split is still restricted to a sub-alphabet"


def test_v026_curriculum_is_deterministic():
    a = load(DATA_V026, "ember_sft_data_v026_det_a").build_examples("train", 600)
    b = load(DATA_V026, "ember_sft_data_v026_det_b").build_examples("train", 600)
    assert [row["value"] for row in a] == [row["value"] for row in b]


def test_v026_evaluation_prompt_matches_the_training_distribution():
    """v0.0.15 evaluated with distractors its own generator could never produce."""
    data = load(DATA_V026, "ember_sft_data_v026_prompt")
    train = data.build_examples("train", 900)
    reachable = {row["value"] for row in train}
    for _, value, _ in data.DIAGNOSTICS[:9]:
        prompt = data.prompt_for(value)
        assert prompt.count(value) == 1
        old = prompt.split("old=", 1)[1].split(" ", 1)[0]
        fallback = prompt.split("fallback=", 1)[1].split(".", 1)[0].split(" ", 1)[0]
        for slot in (old, fallback):
            assert slot not in data.HELD_OUT_VALUES
        assert "K2P8" not in prompt and "77291" not in prompt
    # The legacy prompt is retained verbatim for the protected comparison.
    assert "old=K2P8 and fallback=77291" in data.legacy_prompt_for("Q7M4")
    assert reachable  # the training curriculum is non-empty


def test_v026_config_restores_a_learning_rate_that_can_move_the_model():
    cfg = json.loads(CONFIG.read_text())
    assert cfg["version"] == "0.0.26"
    assert cfg["phase"] == "format-parity-copy-repair"
    assert cfg["source_model_name"] == "ember-v0.0.20-t4"
    assert cfg["first_token_weight"] < cfg["copy_token_weight"]
    # v0.0.21..v0.0.25 annealed the learning rate to 1.6e-7..2.4e-7, where 420
    # AdamW steps move each weight by at most ~1e-4 and every metric freezes.
    assert cfg["learning_rate"] >= 1e-6, "v0.0.26 must escape the annealing spiral"
    assert cfg["learning_rate"] <= 2.5e-6, "and must not exceed the last learning rate that worked"
    assert cfg["expanded_diagnostic_cases"] == 90
    assert cfg["protected_legacy_exact_copy_rate"] == 0.4444444444
    assert cfg["minimum_expanded_exact_copy_gain"] > 0


def test_v026_trainer_transform_targets_exist_in_the_pinned_scaffold():
    """Every replace_required target must match the pinned v0.0.16 scaffold.

    The v0.0.26 trainer rewrites the v0.0.16 source at run time. A drifted target
    raises inside a paid job; this test catches it in CI instead.
    """
    trainer = load(TRAINER_V026, "ember_hf_sft_v026_dryrun")
    scaffold = TRAINER_V016.read_text()
    missing = trainer.unmatched_transform_targets(scaffold)
    assert missing == [], f"transform targets missing from the scaffold: {missing}"


def test_v026_transformed_trainer_keeps_the_preflight_and_baseline_protections():
    """Check the source the job actually executes, not just the wrapper."""
    trainer = load(TRAINER_V026, "ember_hf_sft_v026_runtime")
    runtime = trainer.apply_transforms(TRAINER_V016.read_text())
    compile(runtime, "ember_hf_sft_v026_runtime", "exec")

    # A preflight must still return before any GPU work.
    assert "--preflight-only" in runtime
    assert "v0.0.26 training requires explicitly approved T4 GPU" in runtime
    preflight_return = runtime.index('print("EMBER_HF_V026_PREFLIGHT=PASS"')
    cuda_guard = runtime.index("torch.cuda.is_available()")
    assert preflight_return < cuda_guard

    # Format parity is asserted before the source checkpoint is even fetched.
    assert runtime.index("assert_format_parity(data, train_rows)") < runtime.index(
        "source_path, source_info = resolve_v020_source"
    )

    # best.pt is seeded with the protected v0.0.20 weights at step -1.
    assert "best_score = v026_score(baseline, -1e9)" in runtime
    assert "step=-1" in runtime

    # Selection and promotion run off the ninety-case battery.
    assert "current = v026_score(diag, val_loss)" in runtime
    assert "v026_expanded_gain(baseline, final, cfg)" in runtime
    # One definition plus three call sites: baseline, per-eval, and final.
    assert runtime.count("v026_diagnostic(model, tokenizer, data, cfg") == 4


def test_v026_trainer_starts_from_the_protected_v020_checkpoint():
    trainer = load(TRAINER_V026, "ember_hf_sft_v026_source")
    runtime = trainer.apply_transforms(TRAINER_V016.read_text())
    assert 'SOURCE_REPO = "Jmiller18899/ember-v0.0.20-t4"' in runtime
    assert "ember-v0.0.15-t4" not in runtime
    assert "ember_format_parity_v0.0.26.json" in runtime
    assert "ember_sft_data_v026.py" in runtime
    # The asset pin must name a commit that carries both v0.0.26 assets.
    assert len(trainer.ASSET_COMMIT) == 40
