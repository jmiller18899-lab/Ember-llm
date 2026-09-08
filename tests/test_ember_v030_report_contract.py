"""Contract guards between the Ember trainers' producers and their reporters.

v0.0.29 crashed with KeyError('gate_distance') after every artifact had already
uploaded. Both helpers existed and both print statements existed; nothing
connected them, because v029_progress() never attached the fields to its return
value. Each half was tested in isolation and the seam between them was not.

These tests derive the required keys from the transform text itself, so the
contract cannot drift silently again. The first test proves they catch the real
defect by asserting that v0.0.29 still fails the same check.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAINER_V016 = ROOT / "jobs" / "ember_hf_sft_v016.py"
TRAINER_V029 = ROOT / "jobs" / "ember_hf_sft_v029.py"
TRAINER_V030 = ROOT / "jobs" / "ember_hf_sft_v030.py"
CONFIG_V029 = ROOT / "config" / "ember_multi_position_v0.0.29.json"
CONFIG_V030 = ROOT / "config" / "ember_multi_position_v0.0.30.json"

PROGRESS_KEY = re.compile(r'report\["progress"\](?:\["(\w+)"\]|\.get\("(\w+)")')
DIAGNOSTIC_KEY = re.compile(r'\b(?:baseline|final)\["(\w+)"\]')


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_source(trainer_path: Path, name: str) -> str:
    return load(trainer_path, name).apply_transforms(TRAINER_V016.read_text())


def helpers(trainer_path: Path, name: str) -> dict:
    namespace: dict = {}
    exec(load(trainer_path, name).HELPERS, namespace)
    return namespace


def required_progress_keys(runtime: str) -> set[str]:
    """Every key the reporter reads out of the progress result."""
    return {a or b for a, b in PROGRESS_KEY.findall(runtime)}


def returned_dict_keys(source: str, function: str) -> set[str]:
    """String keys of the dict a function returns, read statically."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            for inner in ast.walk(node):
                if isinstance(inner, ast.Return) and isinstance(inner.value, ast.Dict):
                    return {
                        k.value for k in inner.value.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)
                    }
    raise AssertionError(f"{function} has no dict return")


def synthetic_diagnostic(version: str, exact: float, continuation: float, health: float) -> dict:
    cases = [
        {"value": f"case-{i}", "span_min_gap": -1.4 + 0.02 * i, "kind": "digits"}
        for i in range(90)
    ]
    return {
        "metrics": {
            "expanded_exact_copy_rate": exact,
            "expanded_continuation_top1_rate": continuation,
            "expanded_continuation_tokens": 730,
            "expanded_sequence_margin_health": health,
            "expanded_full_sequence_margin_health": health,
            "expanded_sequences_within_reach": 0.2,
            "expanded_first_token_top1_rate": 0.9667,
            "exact_copy_rate": 6 / 9,
            "continuation_top1_rate": 65 / 69,
            "legacy_continuation_tokens": 69,
        },
        "expanded_cases": cases,
    }


def test_the_contract_check_catches_the_real_v029_defect():
    """Without this, the tests below prove nothing.

    v0.0.29 shipped with the reporter reading gate_distance and band_flow out of
    a progress result that carried neither.
    """
    runtime = runtime_source(TRAINER_V029, "v029_contract_probe")
    required = required_progress_keys(runtime)
    assert {"gate_distance", "band_flow"} <= required, "the reporter did read both fields"

    produced = returned_dict_keys(load(TRAINER_V029, "v029_helpers_probe").HELPERS, "v029_progress")
    missing = required - produced
    assert missing == {"gate_distance", "band_flow"}, (
        "v0.0.29 is expected to be missing exactly these two; got %r" % missing
    )


def test_v030_progress_supplies_every_field_the_reporter_reads():
    runtime = runtime_source(TRAINER_V030, "v030_contract")
    required = required_progress_keys(runtime)
    assert required, "the reporter should read at least one progress field"

    cfg = json.loads(CONFIG_V030.read_text())
    h = helpers(TRAINER_V030, "v030_contract_helpers")
    result = h["v030_progress"](
        synthetic_diagnostic("baseline", 32 / 90, 642 / 730, -0.4361),
        synthetic_diagnostic("final", 36 / 90, 657 / 730, -0.3000),
        cfg,
    )
    missing = required - set(result)
    assert missing == set(), f"the reporter reads fields progress does not supply: {missing}"
    assert result["verdict"] in {"ADVANCED", "FLAT", "REGRESSED"}
    assert isinstance(result["gate_distance"], dict)
    assert isinstance(result["band_flow"], dict)


def test_v030_diagnostic_supplies_every_field_the_run_reads_from_it():
    runtime = runtime_source(TRAINER_V030, "v030_diag_contract")
    required = DIAGNOSTIC_KEY.findall(runtime)
    produced = returned_dict_keys(
        load(TRAINER_V030, "v030_diag_helpers").HELPERS, "v030_diagnostic"
    )
    missing = {key for key in required if key not in produced}
    assert missing == set(), f"the run reads diagnostic fields that are never produced: {missing}"


def test_v030_trailing_prints_cannot_fail_a_run_whose_artifacts_are_uploaded():
    """Every upload happens before these prints, so they read defensively."""
    runtime = runtime_source(TRAINER_V030, "v030_defensive")
    last_upload = runtime.rindex("base.upload(")
    for field in ("gate_distance", "band_flow", "final_distribution"):
        marker = f'.get("{field}", {{}})'
        assert marker in runtime, f"{field} is not read defensively"
        assert runtime.index(marker) > last_upload, f"{field} is printed before the last upload"


def test_v030_continues_from_the_v029_step_479_checkpoint():
    runtime = runtime_source(TRAINER_V030, "v030_source")
    assert 'SOURCE_REPO = "Jmiller18899/ember-v0.0.29-t4"' in runtime
    assert 'if str(source_cfg.get("version")) != "0.0.29":' in runtime
    assert "ember-v0.0.28-t4" not in runtime
    # The run-state written by v0.0.29 is what resolves the checkpoint; it
    # uploaded before the reporter failed.
    assert 'state.get("status") != "evaluation_complete"' in runtime
    assert "ember_multi_position_v0.0.30.json" in runtime
    assert "ember_sft_data_v026.py" in runtime


def test_v030_changes_nothing_but_the_source_and_the_reporter():
    a = json.loads(CONFIG_V029.read_text())
    b = json.loads(CONFIG_V030.read_text())
    for key in (
        "learning_rate", "max_steps", "batch_size", "gradient_accumulation_steps",
        "warmup_steps", "min_lr_ratio", "copy_token_weight", "first_token_weight",
        "eos_token_weight", "margin", "margin_loss_weight", "sequence_margin",
        "sequence_margin_loss_weight", "sequence_worst_k", "boundary_band_low",
        "out_of_band_weight", "hard_batch_fraction", "train_examples",
        "minimum_expanded_exact_copy_rate", "minimum_expanded_continuation_top1_rate",
        "minimum_exact_copy_rate", "minimum_continuation_top1_rate",
    ):
        assert a[key] == b[key], f"{key} drifted; v0.0.30 is meant to change only the source"
    assert b["source_model_name"] == "ember-v0.0.29-t4"
    # 730 x (1 - 0.8795) = 88 wrong continuation decisions over 58 failing rows,
    # plus 3 first-token failures, is 1.57 per failing row: k = 2 still matches.
    assert b["sequence_worst_k"] == 2


def test_v030_protection_holds_the_v029_result_including_the_2_3_bar():
    cfg = json.loads(CONFIG_V030.read_text())
    h = helpers(TRAINER_V030, "v030_protection")
    # 6/9 is 0.6666666666666666 and the stored bar is 0.6666666667; the raw
    # comparison the scaffold uses would reject a checkpoint that holds it.
    assert not (6 / 9 >= cfg["protected_legacy_exact_copy_rate"])
    assert h["v030_meets"](6 / 9, cfg["protected_legacy_exact_copy_rate"], cfg)
    assert not h["v030_meets"](5 / 9, cfg["protected_legacy_exact_copy_rate"], cfg)
    assert cfg["protected_expanded_exact_copy_rate"] <= 32 / 90
    assert cfg["protected_expanded_continuation_top1_rate"] <= 642 / 730
    assert cfg["protected_expanded_full_sequence_margin_health"] <= -0.4361
