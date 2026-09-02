from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "ember_agent_tool_sft_v0.0.9.json"
EVAL_SPEC = ROOT / "config" / "ember_v0.0.8_eval.json"
PACKAGE = ROOT / "ember-v0.0.7-hf-ready.zip"
DATA_MODULE = ROOT / "jobs" / "ember_sft_data_v009.py"
SFT_JOB = ROOT / "jobs" / "ember_hf_sft_v009.py"
WORKFLOW = ROOT / ".github" / "workflows" / "ember-hf.yml"
TRIGGER = ROOT / ".github" / "ember-hf.trigger"


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


def load_data_module():
    spec = importlib.util.spec_from_file_location("ember_sft_data_v009", DATA_MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sft_job_module():
    spec = importlib.util.spec_from_file_location("ember_hf_sft_v009", SFT_JOB)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sft_config_is_small_completion_only_and_source_pinned():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["version"] == "0.0.9"
    assert config["phase"] == "tool-routing-sft"
    assert config["source_model_name"] == "ember-v0.0.8-t4"
    assert config["output_model_name"] == "ember-v0.0.9-t4"
    assert config["completion_only_loss"] is True
    assert config["held_out_promotion_cases"] == 12
    assert config["train_examples"] == 1152
    assert config["validation_examples"] == 144
    assert config["block_size"] == 256
    assert config["max_steps"] == 600
    assert config["learning_rate"] <= 0.00005
    assert config["hub_checkpoint_interval"] == 200
    assert config["max_steps"] % config["hub_checkpoint_interval"] == 0


def test_sft_data_is_balanced_deterministic_and_split_clean():
    module = load_data_module()
    train = module.build_sft_examples("train", 1152, 20260828)
    validation = module.build_sft_examples("validation", 144, 20260828)
    assert Counter(row["kind"] for row in train) == {
        "tool_call": 384,
        "direct_response": 384,
        "tool_result_response": 384,
    }
    assert Counter(row["kind"] for row in validation) == {
        "tool_call": 48,
        "direct_response": 48,
        "tool_result_response": 48,
    }
    assert module.dataset_sha256(train) == module.dataset_sha256(
        module.build_sft_examples("train", 1152, 20260828)
    )
    assert {row["id"] for row in train}.isdisjoint(row["id"] for row in validation)
    assert {row["prompt"] for row in train}.isdisjoint(row["prompt"] for row in validation)


def test_tool_examples_use_the_exact_canonical_json_envelope():
    module = load_data_module()
    rows = module.build_sft_examples("validation", 144, 20260828)
    tools_seen = set()
    for row in rows:
        assert row["prompt"].endswith("<|assistant|>\n")
        assert row["completion"].endswith("<|endoftext|>\n")
        if row["kind"] == "tool_call":
            marker, payload_line, *_ = row["completion"].splitlines()
            payload = json.loads(payload_line)
            assert marker == "<|tool|>"
            assert payload == {
                "arguments": payload["arguments"],
                "name": row["expected_tool"],
            }
            assert isinstance(payload["arguments"], dict) and payload["arguments"]
            tools_seen.add(payload["name"])
        else:
            assert "<|tool|>" not in row["completion"]
    assert tools_seen == {"weather", "calculator", "web_search", "get_time"}


def test_official_promotion_prompts_are_held_out():
    module = load_data_module()
    spec = json.loads(EVAL_SPEC.read_text(encoding="utf-8"))
    rows = module.build_sft_examples("train", 1152) + module.build_sft_examples(
        "validation", 144
    )
    module.assert_held_out_clean(rows, [case["prompt"] for case in spec["cases"]])
    combined = "\n".join(row["prompt"] + row["completion"] for row in rows)
    forbidden = (
        "What is the weather in Detroit right now?",
        "Calculate 347 multiplied by 28.",
        "Find the latest published release of Python.",
        "What time is it in Tokyo?",
        "website thing not working",
    )
    assert not any(text in combined for text in forbidden)


def test_remote_training_inputs_are_content_pinned():
    assert constant(SFT_JOB, "CONFIG_SHA256") == sha256(CONFIG)
    assert constant(SFT_JOB, "DATA_SHA256") == sha256(DATA_MODULE)
    assert constant(SFT_JOB, "EVAL_SPEC_SHA256") == sha256(EVAL_SPEC)
    assert constant(SFT_JOB, "PACKAGE_SHA256") == sha256(PACKAGE)


def test_sft_job_masks_prompts_tracks_metrics_and_persists_both_formats():
    source = SFT_JOB.read_text(encoding="utf-8")
    assert "targets = torch.full((block_size,), -100" in source
    assert "first_completion_prediction" in source
    assert "trackio.init(" in source
    assert "trackio.log(" in source
    assert "export_int4_checkpoint(best_path)" in source
    assert "training_complete_pending_eval" in source
    assert "official_cases_used_for_training\": 0" in source
    assert "ember-v0.0.9-t4" in source
    assert "clawagent" not in source.lower()


def test_completion_only_encoding_masks_every_prompt_target():
    import torch

    module = load_sft_job_module()

    class DummyTokenizer:
        def encode(self, text):
            return [1] + [10 + ord(char) for char in text]

    row = {
        "id": "mask-check",
        "prompt": "prompt",
        "completion": "answer",
    }
    inputs, targets = module.encode_examples(DummyTokenizer(), [row], 32, torch)[0]
    prompt_length = len(DummyTokenizer().encode(row["prompt"]))
    assert torch.all(targets[: prompt_length - 1] == -100)
    assert targets[prompt_length - 1].item() == DummyTokenizer().encode(
        row["prompt"] + row["completion"]
    )[prompt_length]
    assert torch.any(targets[prompt_length - 1 :] != -100)
    assert inputs.shape == targets.shape == (32,)


def test_workflow_exposes_manual_sft_and_cpu_eval_without_default_gpu_launch():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    marker = TRIGGER.read_text(encoding="utf-8")
    assert "sft-v009" in workflow
    assert "preflight-v009" in workflow
    assert "eval-v009" in workflow
    assert "steps.request.outputs.mode == 'preflight-v009'" in workflow
    assert "--preflight-only" in workflow
    assert "steps.request.outputs.mode == 'sft-v009'" in workflow
    assert "--name ember-v0-0-9-tool-sft" in workflow
    assert "--flavor t4-small" in workflow
    mode = marker.splitlines()[0].split(":", 1)[1].strip()
    assert mode not in {"train", "train-v008", "sft-v009", "sft-v010"}


def test_no_hugging_face_token_is_committed_for_v009():
    token_pattern = re.compile(r"hf_[A-Za-z0-9]{20,}")
    checked = [CONFIG, DATA_MODULE, SFT_JOB, WORKFLOW]
    assert not any(token_pattern.search(path.read_text(encoding="utf-8")) for path in checked)
