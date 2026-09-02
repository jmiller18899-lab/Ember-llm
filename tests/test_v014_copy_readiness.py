from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "ember_agent_copy_canary_v0.0.14.json"
EVAL_SPEC = ROOT / "config" / "ember_v0.0.8_eval.json"
PACKAGE = ROOT / "ember-v0.0.7-hf-ready.zip"
DATA_MODULE = ROOT / "jobs" / "ember_sft_data_v014.py"
SFT_JOB = ROOT / "jobs" / "ember_hf_sft_v014.py"
WORKFLOW = ROOT / ".github" / "workflows" / "ember-hf.yml"
TRIGGER = ROOT / ".github" / "ember-hf.trigger"
VALIDATE = ROOT / ".github" / "workflows" / "ember-validate.yml"

# Recorded from Jmiller18899/ember-v0.0.13-t4/training-summary.json after the
# 160-step short-copy canary ended in failed_internal_smoke on 2026-09-02.
V013_SMOKE_FAILURES = (
    {
        "id": "validation-tool-00000",
        "kind": "tool_call",
        "focus_terms": ["WX-200239"],
        "expected": '<|tool|>\n{"arguments":{"location":"WX-200239"},"name":"weather"}\n<|endoftext|>\n',
        "got": '<|tool|>\n{"arguments":{"location":"WX-2009"},"name":"weather"}\n<|endoftext|>',
    },
    {
        "id": "validation-tool-00001",
        "kind": "tool_call",
        "focus_terms": ["290*86"],
        "expected": '<|tool|>\n{"arguments":{"expression":"290*86"},"name":"calculator"}\n<|endoftext|>\n',
        "got": '<|tool|>\n{"arguments":{"expression":"330*64"},"name":"calculator"}\n<|endoftext|>',
    },
    {
        "id": "validation-tool-00002",
        "kind": "tool_call",
        "focus_terms": ["REL-200323"],
        "expected": '<|tool|>\n{"arguments":{"query":"REL-200323"},"name":"web_search"}\n<|endoftext|>\n',
        "got": '<|tool|>\n{"arguments":{"query":"REL-200309"},"name":"web_search"}\n<|endoftext|>',
    },
    {
        "id": "validation-direct-00000",
        "kind": "direct_response",
        "focus_terms": ["ECHO-200415"],
        "expected": "ECHO-200415\n<|endoftext|>\n",
        "got": "ECHO-2315\n<|endoftext|>",
    },
    {
        "id": "validation-direct-00003",
        "kind": "direct_response",
        "focus_terms": ["DEMO-200128"],
        "expected": "1. Submit valid and invalid credentials in demo DEMO-200128.\n<|endoftext|>\n",
        "got": "1. Prepare a controlled example and check an API response.\n2. Record the result and verify the expected behavior.\n<|endoftext|>",
    },
    {
        "id": "validation-result-00001",
        "kind": "tool_result_response",
        "focus_terms": ["24940"],
        "expected": "The supplied result is 24940.\n<|endoftext|>\n",
        "got": "The supplied result is 7150.\n<|endoftext|>",
    },
)


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
    spec = importlib.util.spec_from_file_location("ember_sft_data_v014", DATA_MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sft_job_module():
    spec = importlib.util.spec_from_file_location("ember_hf_sft_v014", SFT_JOB)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v013_canary_copied_templates_but_not_exact_ids():
    module = load_data_module()
    for row in V013_SMOKE_FAILURES:
        assert module.focus_ok(row["expected"], row["focus_terms"])
        assert not module.focus_ok(row["got"], row["focus_terms"])
        if row["kind"] == "tool_call":
            assert "<|tool|>" in row["got"]


def test_v014_config_refuses_failed_v013_recipe():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["version"] == "0.0.14"
    assert config["phase"] == "literal-copy-canary"
    assert config["source_model_name"] == "ember-v0.0.9-t4"
    assert config["output_model_name"] == "ember-v0.0.14-t4"
    assert config["source_checkpoint_sha256"] == (
        "8299d52e8a852b9bd3e8403e086b48fd42d2babfd51c74eb94af29bd87ef2d13"
    )
    assert config["max_steps"] == 480
    assert config["max_steps"] >= 400
    assert config["learning_rate"] <= 0.000004
    assert config["semantic_token_weight"] == 5.0
    assert config["semantic_token_weight"] <= 6.0
    assert config["eos_token_weight"] == 2.0
    assert config["train_examples"] == 2400
    assert config["validation_examples"] == 300
    assert config["hub_checkpoint_interval"] == 160
    assert config["max_steps"] % config["hub_checkpoint_interval"] == 0
    seen = config["max_steps"] * config["batch_size"] * config["gradient_accumulation_steps"]
    assert seen >= 3 * config["train_examples"]


def test_v014_data_is_literal_copy_with_repeated_short_codes():
    module = load_data_module()
    train = module.build_examples("train", 2400)
    validation = module.build_examples("validation", 300)
    assert Counter(row["kind"] for row in train) == {
        "tool_call": 800,
        "direct_response": 800,
        "tool_result_response": 800,
    }
    assert Counter(row["kind"] for row in validation) == {
        "tool_call": 100,
        "direct_response": 100,
        "tool_result_response": 100,
    }
    assert module.dataset_sha256(train) == module.dataset_sha256(module.build_examples("train", 2400))
    assert {row["id"] for row in train}.isdisjoint(row["id"] for row in validation)
    assert {row["prompt"] for row in train}.isdisjoint(row["prompt"] for row in validation)
    train_focus = {row["focus_terms"][0] for row in train}
    val_focus = {row["focus_terms"][0] for row in validation}
    assert train_focus.isdisjoint(val_focus)
    for row in train + validation:
        assert len(row["focus_terms"]) == 1
        term = row["focus_terms"][0]
        assert term in row["prompt"]
        assert term in row["completion"]
        assert row["prompt"].count(term) >= 2
        assert " " not in term
        assert 3 <= len(term) <= 8
        assert not re.fullmatch(r"[A-Z]{2,}-\d{6}", term)


def test_v014_tool_examples_use_canonical_json_and_short_ids():
    module = load_data_module()
    rows = module.build_examples("validation", 300)
    tools_seen = set()
    for row in rows:
        assert row["prompt"].endswith("<|assistant|>\n")
        assert row["completion"].endswith("<|endoftext|>\n")
        if row["kind"] == "tool_call":
            marker, payload_line, *_ = row["completion"].splitlines()
            payload = json.loads(payload_line)
            assert marker == "<|tool|>"
            assert payload["name"] == row["expected_tool"]
            assert payload["arguments"]
            tools_seen.add(payload["name"])
            value = next(iter(payload["arguments"].values()))
            assert row["focus_terms"] == [value]
        else:
            assert "<|tool|>" not in row["completion"]
    assert tools_seen == {"weather", "calculator", "web_search", "get_time"}


def test_official_promotion_prompts_are_held_out():
    module = load_data_module()
    spec = json.loads(EVAL_SPEC.read_text(encoding="utf-8"))
    rows = module.build_examples("train", 2400) + module.build_examples("validation", 300)
    module.assert_held_out_clean(rows, [case["prompt"] for case in spec["cases"]])
    combined = "\n".join(row["prompt"] + row["completion"] for row in rows)
    forbidden = (
        "What is the weather in Detroit right now?",
        "Calculate 347 multiplied by 28.",
        "Find the latest published release of Python.",
        "What time is it in Tokyo?",
        "website thing not working",
        "Ann Arbor",
        "Seattle",
        "Rapidton-7669",
        "WX-200239",
        "ECHO-200415",
    )
    assert not any(text in combined for text in forbidden)


def test_remote_training_inputs_are_content_pinned():
    pin = constant(SFT_JOB, "ASSET_PIN")
    assert re.fullmatch(r"[0-9a-f]{40}", pin)
    assert set(pin) != {"0"}
    assert constant(SFT_JOB, "CONFIG_SHA256") == sha256(CONFIG)
    assert constant(SFT_JOB, "DATA_SHA256") == sha256(DATA_MODULE)
    assert constant(SFT_JOB, "PACKAGE_SHA256") == sha256(PACKAGE)
    assert constant(SFT_JOB, "SOURCE_SHA256") == (
        "8299d52e8a852b9bd3e8403e086b48fd42d2babfd51c74eb94af29bd87ef2d13"
    )
    assert constant(SFT_JOB, "SOURCE_REPO") == "Jmiller18899/ember-v0.0.9-t4"
    source = SFT_JOB.read_text(encoding="utf-8")
    assert "ember-v0.0.11-t4" not in source
    assert "ember-v0.0.12-t4" not in source
    assert "ember-v0.0.13-t4" not in source
    assert "refuses_failed_v013_init" in source


def test_v014_job_masks_prompts_stops_on_eos_and_persists_both_formats():
    source = SFT_JOB.read_text(encoding="utf-8")
    assert "y = torch.full((block_size,), -100" in source
    assert "semantic_positions" in source
    assert "def eos_greedy" in source
    assert "export_int4_checkpoint(best_path)" in source
    assert "training_complete_pending_eval" in source
    assert "failed_internal_smoke" in source
    assert "three passes over the training set" in source
    assert "clawagent" not in source.lower()


def test_character_aligned_encoding_masks_prompt_and_upweights_focus():
    import torch

    module = load_sft_job_module()

    class DummyTokenizer:
        def encode(self, text):
            return [1] + [10 + (ord(char) % 80) for char in text]

    row = {
        "id": "mask-check",
        "prompt": "prompt ",
        "completion": "GYHH\n<|endoftext|>\n",
        "focus_terms": ["GYHH"],
    }
    encoded = module.encode_row(DummyTokenizer(), row, 64, 5.0, 2.0, torch)
    prompt_length = len(DummyTokenizer().encode(row["prompt"]))
    assert torch.all(encoded["y"][: prompt_length - 1] == -100)
    assert encoded["semantic_positions"] >= 1
    assert float(encoded["w"].max()) >= 5.0


def test_workflow_exposes_manual_v014_modes_without_default_gpu_launch():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    marker = TRIGGER.read_text(encoding="utf-8")
    validate = VALIDATE.read_text(encoding="utf-8")
    assert "sft-v014" in workflow
    assert "preflight-v014" in workflow
    assert "steps.request.outputs.mode == 'preflight-v014'" in workflow
    assert "steps.request.outputs.mode == 'sft-v014'" in workflow
    assert "--name ember-v0-0-14-literal-copy" in workflow
    assert "--flavor t4-small" in workflow
    assert "ember_agent_copy_canary_v0.0.14.json" in validate
    mode = marker.splitlines()[0].split(":", 1)[1].strip()
    assert mode != "sft-v014"
    assert "ember-hf.trigger" not in "".join(
        line for line in workflow.splitlines() if "sft-v014" in line
    )


def test_no_hugging_face_token_is_committed_for_v014():
    token_pattern = re.compile(r"hf_[A-Za-z0-9]{20,}")
    checked = [CONFIG, DATA_MODULE, SFT_JOB, WORKFLOW]
    assert not any(token_pattern.search(path.read_text(encoding="utf-8")) for path in checked)
