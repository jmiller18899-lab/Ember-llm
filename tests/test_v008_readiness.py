from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG = ROOT / "config" / "ember_agent_t4_long_v0.0.8.json"
EVAL_SPEC = ROOT / "config" / "ember_v0.0.8_eval.json"
TRAIN_JOB = ROOT / "jobs" / "ember_hf_train_v008.py"
EVAL_JOB = ROOT / "jobs" / "ember_hf_eval.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def constant(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"missing constant {name} in {path}")


def load_eval_module():
    spec = importlib.util.spec_from_file_location("ember_hf_eval", EVAL_JOB)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_train_module():
    spec = importlib.util.spec_from_file_location("ember_hf_train_v008", TRAIN_JOB)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_long_run_keeps_validated_architecture_and_uses_fresh_schedule():
    candidate = json.loads(TRAIN_CONFIG.read_text(encoding="utf-8"))
    with zipfile.ZipFile(ROOT / "ember-v0.0.7-hf-ready.zip") as archive:
        baseline = json.loads(archive.read("ember/config/ember_agent_t4_validation.json"))

    architecture_keys = {
        "block_size",
        "n_layer",
        "n_head",
        "n_embd",
        "dropout",
        "position_encoding",
        "norm_type",
        "mlp_type",
    }
    assert {key: candidate[key] for key in architecture_keys} == {
        key: baseline[key] for key in architecture_keys
    }
    assert candidate["tokenizer"] == baseline["tokenizer"]
    assert candidate["version"] == "0.0.8"
    assert candidate["initialization"] == "from_scratch"
    assert candidate["resume_scope"] == "v0.0.8_only"


def test_long_run_budget_and_checkpoint_math_is_explicit():
    config = json.loads(TRAIN_CONFIG.read_text(encoding="utf-8"))
    expected_tokens = (
        config["max_steps"]
        * config["batch_size"]
        * config["block_size"]
        * config["gradient_accumulation_steps"]
    )
    assert config["max_steps"] == 3000
    assert config["warmup_steps"] == 150
    assert config["save_interval"] == 500
    assert config["hub_checkpoint_interval"] == 500
    assert config["max_steps"] % config["hub_checkpoint_interval"] == 0
    assert config["expected_training_tokens"] == expected_tokens == 49_152_000
    assert 3.2 < expected_tokens / config["verified_corpus_tokens"] < 3.4


def test_remote_downloads_are_content_pinned():
    assert constant(TRAIN_JOB, "TRAIN_CONFIG_SHA256") == sha256(TRAIN_CONFIG)
    assert constant(EVAL_JOB, "EVAL_SPEC_SHA256") == sha256(EVAL_SPEC)
    expected_package = "27e8f7c80317652a22b3d58a0bd474724491a685dfe9e20c0b997b7c5907a289"
    assert constant(TRAIN_JOB, "PACKAGE_SHA256") == expected_package
    assert constant(EVAL_JOB, "PACKAGE_SHA256") == expected_package


def test_eval_spec_has_balanced_held_out_contract_groups():
    spec = json.loads(EVAL_SPEC.read_text(encoding="utf-8"))
    cases = spec["cases"]
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids)) == 12
    counts = {
        kind: sum(case["kind"] == kind for case in cases)
        for kind in {"tool_call", "direct_response", "tool_result_response"}
    }
    assert counts == {"tool_call": 4, "direct_response": 4, "tool_result_response": 4}
    assert all(case["prompt"].endswith("<|assistant|>\n") for case in cases)
    assert all("<|endoftext|>" not in case["prompt"] for case in cases)
    assert spec["validation_loss"] == {"batch_size": 8, "batches": 16}


def test_tool_contract_scoring_requires_marker_json_name_and_arguments():
    module = load_eval_module()
    case = {"kind": "tool_call", "expected_tool": "weather"}
    valid = '<|tool|>\n{"name":"weather","arguments":{"location":"Detroit"}}'
    invalid_name = '<|tool|>\n{"name":"search","arguments":{"q":"Detroit"}}'
    missing_marker = '{"name":"weather","arguments":{"location":"Detroit"}}'
    assert module.score_case(case, valid)["passed"] is True
    assert module.score_case(case, invalid_name)["passed"] is False
    assert module.score_case(case, missing_marker)["passed"] is False


def test_json_extractor_handles_braces_inside_strings():
    module = load_eval_module()
    payload = module.extract_json_object('prefix {"arguments":{"query":"a { brace }"}} suffix')
    assert payload == {"arguments": {"query": "a { brace }"}}


def test_direct_and_tool_result_scores_reject_recursive_tool_calls():
    module = load_eval_module()
    for kind in ("direct_response", "tool_result_response"):
        assert module.score_case({"kind": kind}, "A clear answer.")["passed"] is True
        assert module.score_case({"kind": kind}, "<|tool|> {} ")["passed"] is False


def test_promotion_compares_the_same_fixed_validation_measurement():
    module = load_eval_module()
    baseline = {
        "checkpoint": {"best_validation_loss": 1.0},
        "metrics": {
            "fixed_validation_loss": 4.0,
            "valid_tool_call_rate": 0.25,
            "direct_response_rate": 0.75,
            "tool_result_response_rate": 0.75,
        },
    }
    candidate = {
        "checkpoint": {"best_validation_loss": 9.0},
        "metrics": {
            "fixed_validation_loss": 3.9,
            "valid_tool_call_rate": 0.5,
            "direct_response_rate": 1.0,
            "tool_result_response_rate": 1.0,
        },
        "technical_pass": True,
    }
    thresholds = json.loads(EVAL_SPEC.read_text(encoding="utf-8"))["promotion"]
    promotion = module.compare_for_promotion(candidate, baseline, thresholds)
    assert promotion["relative_validation_loss_improvement"] == pytest.approx(0.025)
    assert promotion["promotion_eligible"] is True


def test_paid_long_run_is_manual_only_and_marker_was_not_changed():
    workflow = (ROOT / ".github" / "workflows" / "ember-hf.yml").read_text(encoding="utf-8")
    marker = (ROOT / ".github" / "ember-hf.trigger").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "steps.request.outputs.mode == 'train-v008'" in workflow
    assert "--flavor t4-small" in workflow
    assert "--timeout 3h" in workflow
    assert marker.startswith("mode: train\n")
    assert "train-v008" not in marker


def test_paid_run_state_lock_rejects_live_and_completed_runs():
    module = load_train_module()
    with pytest.raises(RuntimeError, match="appears active"):
        module.assert_no_live_duplicate({
            "status": "running",
            "updated_at": module.utc_now().isoformat(),
        })
    with pytest.raises(RuntimeError, match="already complete"):
        module.assert_no_live_duplicate({"status": "training_complete_pending_eval"})
    module.assert_no_live_duplicate({
        "status": "running",
        "updated_at": "2000-01-01T00:00:00+00:00",
    })
    module.assert_no_live_duplicate({"status": "error"})


def test_no_hugging_face_token_is_committed():
    token_pattern = re.compile(r"hf_[A-Za-z0-9]{20,}")
    checked = [TRAIN_JOB, EVAL_JOB, TRAIN_CONFIG, EVAL_SPEC]
    assert not any(token_pattern.search(path.read_text(encoding="utf-8")) for path in checked)
