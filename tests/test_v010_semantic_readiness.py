from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "ember_agent_tool_sft_v0.0.10.json"
EVAL_SPEC = ROOT / "config" / "ember_v0.0.10_eval.json"
LEGACY_EVAL_SPEC = ROOT / "config" / "ember_v0.0.8_eval.json"
PACKAGE = ROOT / "ember-v0.0.7-hf-ready.zip"
DATA_MODULE = ROOT / "jobs" / "ember_sft_data_v010.py"
SFT_JOB = ROOT / "jobs" / "ember_hf_sft_v010.py"
EVAL_JOB = ROOT / "jobs" / "ember_hf_eval.py"
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
    spec = importlib.util.spec_from_file_location("ember_sft_data_v010", DATA_MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_eval_module():
    spec = importlib.util.spec_from_file_location("ember_hf_eval", EVAL_JOB)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sft_job_module():
    spec = importlib.util.spec_from_file_location("ember_hf_sft_v010", SFT_JOB)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepared_cases():
    spec = json.loads(EVAL_SPEC.read_text(encoding="utf-8"))
    module = load_eval_module()
    return spec, {
        case["id"]: module.prepare_case(case, spec)
        for case in spec["cases"]
    }


def test_sft_config_is_completion_only_from_accepted_v009():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["version"] == "0.0.10"
    assert config["phase"] == "semantic-fidelity-sft"
    assert config["source_model_name"] == "ember-v0.0.9-t4"
    assert config["source_evaluation_label"] == "candidate-v0.0.9"
    assert config["output_model_name"] == "ember-v0.0.10-t4"
    assert config["completion_only_loss"] is True
    assert config["held_out_promotion_cases"] == 12
    assert config["train_examples"] == 1152
    assert config["validation_examples"] == 144
    assert config["block_size"] == 256
    assert config["max_steps"] == 400
    assert config["learning_rate"] <= 0.00002
    assert config["hub_checkpoint_interval"] == 200
    assert config["max_steps"] % config["hub_checkpoint_interval"] == 0
    assert config["internal_minimum_valid_tool_call_rate"] >= 0.75
    assert config["internal_minimum_clean_stop_rate"] >= 0.75


def test_sft_data_is_balanced_deterministic_and_copy_grounded():
    module = load_data_module()
    train = module.build_sft_examples("train", 1152, 20260902)
    validation = module.build_sft_examples("validation", 144, 20260902)
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
        module.build_sft_examples("train", 1152, 20260902)
    )
    assert {row["id"] for row in train}.isdisjoint(row["id"] for row in validation)
    assert {row["prompt"] for row in train}.isdisjoint(row["prompt"] for row in validation)


def test_tool_examples_copy_prompt_values_and_stop_at_endoftext():
    module = load_data_module()
    rows = module.build_sft_examples("validation", 144, 20260902)
    tools_seen = set()
    for row in rows:
        assert row["completion"].endswith("<|endoftext|>\n")
        assert row["completion"].count("<|endoftext|>") == 1
        if row["kind"] == "tool_call":
            marker, payload_line, *_ = row["completion"].splitlines()
            payload = json.loads(payload_line)
            assert marker == "<|tool|>"
            assert payload["name"] == row["expected_tool"]
            blob = " ".join(str(value) for value in payload["arguments"].values())
            for expected in row["expected_arguments"].values():
                tokens = expected if isinstance(expected, list) else [expected]
                for token in tokens:
                    assert str(token) in blob
                    assert str(token) in row["prompt"]
            tools_seen.add(payload["name"])
        else:
            assert "<|tool|>" not in row["completion"]
            for fact in row["required_facts"]:
                assert str(fact).casefold() in row["completion"].casefold()
    assert tools_seen == {"weather", "calculator", "web_search", "get_time"}


def test_official_promotion_prompts_are_held_out_from_v010():
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
        '"result":9716',
        '"latency_ms":84',
    )
    assert not any(text in combined for text in forbidden)


def test_legacy_structural_spec_stays_frozen():
    assert sha256(LEGACY_EVAL_SPEC) == "e006aa0f7c50797e0466a87fa3f1e35a1f00a63baf1f5113cf6e574844079bd4"
    legacy = json.loads(LEGACY_EVAL_SPEC.read_text(encoding="utf-8"))
    semantic = json.loads(EVAL_SPEC.read_text(encoding="utf-8"))
    assert [case["prompt"] for case in legacy["cases"]] == [case["prompt"] for case in semantic["cases"]]
    assert all("expected_arguments" in case for case in semantic["cases"] if case["kind"] == "tool_call")
    assert all("required_facts" in case for case in semantic["cases"] if case["kind"] != "tool_call")
    assert semantic["scoring"] == {"require_clean_stop": True, "require_endoftext": True}
    promotion = semantic["promotion"]
    assert "minimum_relative_validation_loss_improvement" not in promotion
    assert promotion["maximum_relative_validation_loss_regression"] == 0.05
    assert promotion["minimum_tool_name_rate"] == 0.75
    assert promotion["minimum_tool_argument_rate"] == 0.5
    assert promotion["minimum_valid_tool_call_rate"] == 0.5
    assert promotion["minimum_tool_result_response_rate"] == 0.75
    assert promotion["minimum_clean_stop_rate"] == 0.75
    assert promotion["minimum_direct_response_rate"] == 0.75


def test_wrong_tool_arguments_fail_the_semantic_gate():
    module = load_eval_module()
    spec, cases = prepared_cases()
    wrong = {
        "tool_weather": '<|tool|>\n{"name":"weather","arguments":{"location":"Austin"}}\n<|endoftext|>\n',
        "tool_calculator": '<|tool|>\n{"name":"calculator","arguments":{"expression":"330"}}\n<|endoftext|>\n',
        "tool_web_search": '<|tool|>\n{"name":"web_search","arguments":{"query":"the latest SQLite release"}}\n<|endoftext|>\n',
        "tool_get_time": '<|tool|>\n{"name":"get_time","arguments":{"timezone":"America/Anchorage"}}\n<|endoftext|>\n',
    }
    correct = {
        "tool_weather": '<|tool|>\n{"name":"weather","arguments":{"location":"Detroit"}}\n<|endoftext|>\n',
        "tool_calculator": '<|tool|>\n{"name":"calculator","arguments":{"expression":"347 * 28"}}\n<|endoftext|>\n',
        "tool_web_search": '<|tool|>\n{"name":"web_search","arguments":{"query":"latest published release of Python"}}\n<|endoftext|>\n',
        "tool_get_time": '<|tool|>\n{"name":"get_time","arguments":{"timezone":"Asia/Tokyo"}}\n<|endoftext|>\n',
    }
    for case_id, completion in wrong.items():
        score = module.score_case(cases[case_id], completion)
        assert score["passed"] is False
        assert score["tool_name_matches"] is True
        assert score["arguments_match"] is False
    for case_id, completion in correct.items():
        assert module.score_case(cases[case_id], completion)["passed"] is True
    austin = wrong["tool_weather"]
    assert module.score_case({"kind": "tool_call", "expected_tool": "weather"}, austin)["passed"] is True


def test_tool_result_answers_must_use_supplied_facts():
    module = load_eval_module()
    _, cases = prepared_cases()
    invented = {
        "result_calculator": "57 multiplied by 2 equals 675.\n<|endoftext|>\n",
        "result_service_status": "It is 45°F and cloudy in Detroit.\n<|endoftext|>\n",
        "result_weather": "It is 12°F and snowing in Austin.\n<|endoftext|>\n",
        "result_search": "No updates were found.\n<|endoftext|>\n",
    }
    faithful = {
        "result_calculator": "347 multiplied by 28 equals 9716.\n<|endoftext|>\n",
        "result_service_status": "The gateway service is healthy with 84 ms latency.\n<|endoftext|>\n",
        "result_weather": "It is 72°F and sunny in Detroit.\n<|endoftext|>\n",
        "result_search": "All validation checks passed.\n<|endoftext|>\n",
    }
    for case_id, completion in invented.items():
        score = module.score_case(cases[case_id], completion)
        assert score["passed"] is False
        assert score["facts_present"] is False
    for case_id, completion in faithful.items():
        assert module.score_case(cases[case_id], completion)["passed"] is True


def test_generation_after_endoftext_fails_clean_stop():
    module = load_eval_module()
    _, cases = prepared_cases()
    valid = '<|tool|>\n{"name":"weather","arguments":{"location":"Detroit"}}\n<|endoftext|>\n'
    continued = valid + "<|user|>\nWhat about Austin?\n"
    missing_stop = '<|tool|>\n{"name":"weather","arguments":{"location":"Detroit"}}\n'
    assert module.score_case(cases["tool_weather"], valid)["passed"] is True
    continued_score = module.score_case(cases["tool_weather"], continued)
    assert continued_score["passed"] is False
    assert continued_score["clean_stop"] is False
    missing_score = module.score_case(cases["tool_weather"], missing_stop)
    assert missing_score["passed"] is False
    assert missing_score["endoftext_present"] is False


def test_baseline_rescore_rejects_v009_style_wrong_arguments():
    module = load_eval_module()
    spec, _ = prepared_cases()
    recorded = [
        {
            "id": "tool_weather",
            "kind": "tool_call",
            "completion": '<|tool|>\n{"name":"weather","arguments":{"location":"Austin"}}\n<|endoftext|>\n',
        },
        {
            "id": "tool_calculator",
            "kind": "tool_call",
            "completion": '<|tool|>\n{"name":"calculator","arguments":{"expression":"330"}}\n<|endoftext|>\n',
        },
        {
            "id": "tool_web_search",
            "kind": "tool_call",
            "completion": '<|tool|>\n{"name":"web_search","arguments":{"query":"SQLite"}}\n<|endoftext|>\n',
        },
        {
            "id": "tool_get_time",
            "kind": "tool_call",
            "completion": '<|tool|>\n{"name":"get_time","arguments":{"timezone":"America/Anchorage"}}\n<|endoftext|>\n',
        },
        {
            "id": "direct_greeting",
            "kind": "direct_response",
            "completion": "Hello there, friend.\n<|endoftext|>\n",
        },
        {
            "id": "direct_rewrite",
            "kind": "direct_response",
            "completion": "Website is not working.\n<|endoftext|>\n",
        },
        {
            "id": "direct_explain",
            "kind": "direct_response",
            "completion": "A checkpoint stores model weights.\n<|endoftext|>\n",
        },
        {
            "id": "direct_plan",
            "kind": "direct_response",
            "completion": "1. Open the login form.\n2. Submit a test account.\n<|endoftext|>\n",
        },
        {
            "id": "result_weather",
            "kind": "tool_result_response",
            "completion": "It is 72 and sunny.\n<|endoftext|>\n",
        },
        {
            "id": "result_calculator",
            "kind": "tool_result_response",
            "completion": "57 multiplied by 2 equals 675.\n<|endoftext|>\n",
        },
        {
            "id": "result_search",
            "kind": "tool_result_response",
            "completion": "All validation checks passed.\n<|endoftext|>\n",
        },
        {
            "id": "result_service_status",
            "kind": "tool_result_response",
            "completion": "It is 45°F and cloudy.\n<|endoftext|>\n",
        },
    ]
    metrics = module.rescore_recorded_cases(spec, recorded)
    assert metrics["tool_name_rate"] == 1.0
    assert metrics["tool_argument_rate"] == 0.0
    assert metrics["valid_tool_call_rate"] == 0.0
    assert metrics["tool_result_response_rate"] == 0.5


def test_semantic_promotion_requires_behavior_not_just_loss():
    module = load_eval_module()
    thresholds = json.loads(EVAL_SPEC.read_text(encoding="utf-8"))["promotion"]
    baseline = {
        "metrics": {
            "fixed_validation_loss": 3.4088,
            "valid_tool_call_rate": 0.0,
            "tool_name_rate": 1.0,
            "tool_argument_rate": 0.0,
            "direct_response_rate": 1.0,
            "tool_result_response_rate": 0.5,
            "clean_stop_rate": 0.5,
        }
    }
    loss_only = {
        "technical_pass": True,
        "metrics": {
            "fixed_validation_loss": 2.6,
            "valid_tool_call_rate": 0.0,
            "tool_name_rate": 1.0,
            "tool_argument_rate": 0.0,
            "direct_response_rate": 1.0,
            "tool_result_response_rate": 0.5,
            "clean_stop_rate": 0.5,
        },
    }
    loss_only_result = module.compare_for_promotion(loss_only, baseline, thresholds)
    assert loss_only_result["checks"]["validation_loss_not_regressed"] is True
    assert loss_only_result["checks"]["absolute_tool_argument_gate"] is False
    assert loss_only_result["promotion_eligible"] is False

    collapsed_routing = {
        "technical_pass": True,
        "metrics": {
            "fixed_validation_loss": 3.4,
            "valid_tool_call_rate": 0.5,
            "tool_name_rate": 0.5,
            "tool_argument_rate": 0.5,
            "direct_response_rate": 1.0,
            "tool_result_response_rate": 0.75,
            "clean_stop_rate": 0.75,
        },
    }
    assert module.compare_for_promotion(collapsed_routing, baseline, thresholds)[
        "promotion_eligible"
    ] is False

    direct_regression = {
        "technical_pass": True,
        "metrics": {
            "fixed_validation_loss": 3.4,
            "valid_tool_call_rate": 0.75,
            "tool_name_rate": 1.0,
            "tool_argument_rate": 0.75,
            "direct_response_rate": 0.5,
            "tool_result_response_rate": 0.75,
            "clean_stop_rate": 0.75,
        },
    }
    assert module.compare_for_promotion(direct_regression, baseline, thresholds)[
        "promotion_eligible"
    ] is False

    bad_loss = {
        "technical_pass": True,
        "metrics": {
            "fixed_validation_loss": 3.7,
            "valid_tool_call_rate": 0.75,
            "tool_name_rate": 1.0,
            "tool_argument_rate": 0.75,
            "direct_response_rate": 1.0,
            "tool_result_response_rate": 0.75,
            "clean_stop_rate": 0.75,
        },
    }
    assert module.compare_for_promotion(bad_loss, baseline, thresholds)[
        "promotion_eligible"
    ] is False

    no_int4 = {
        "technical_pass": False,
        "metrics": {
            "fixed_validation_loss": 3.45,
            "valid_tool_call_rate": 0.75,
            "tool_name_rate": 1.0,
            "tool_argument_rate": 0.75,
            "direct_response_rate": 1.0,
            "tool_result_response_rate": 0.75,
            "clean_stop_rate": 0.75,
        },
    }
    assert module.compare_for_promotion(no_int4, baseline, thresholds)["promotion_eligible"] is False

    eligible = {
        "technical_pass": True,
        "metrics": {
            "fixed_validation_loss": 3.45,
            "valid_tool_call_rate": 0.75,
            "tool_name_rate": 1.0,
            "tool_argument_rate": 0.75,
            "direct_response_rate": 1.0,
            "tool_result_response_rate": 0.75,
            "clean_stop_rate": 0.75,
        },
    }
    result = module.compare_for_promotion(eligible, baseline, thresholds)
    assert result["checks"]["validation_loss_not_regressed"] is True
    assert result["checks"]["absolute_tool_name_gate"] is True
    assert result["checks"]["absolute_tool_argument_gate"] is True
    assert result["checks"]["absolute_clean_stop_gate"] is True
    assert result["checks"]["direct_response_non_regression"] is True
    assert result["checks"]["technical_gate"] is True
    assert result["promotion_eligible"] is True


def test_remote_training_inputs_are_content_pinned():
    assert constant(SFT_JOB, "CONFIG_SHA256") == sha256(CONFIG)
    assert constant(SFT_JOB, "DATA_SHA256") == sha256(DATA_MODULE)
    assert constant(SFT_JOB, "EVAL_SPEC_SHA256") == sha256(EVAL_SPEC)
    assert constant(SFT_JOB, "SOURCE_EVAL_SPEC_SHA256") == sha256(LEGACY_EVAL_SPEC)
    assert constant(SFT_JOB, "PACKAGE_SHA256") == sha256(PACKAGE)
    assert constant(EVAL_JOB, "EVAL_SPEC_V010_SHA256") == sha256(EVAL_SPEC)
    assert constant(EVAL_JOB, "EVAL_SPEC_SHA256") == sha256(LEGACY_EVAL_SPEC)


def test_sft_job_requires_accepted_v009_and_masks_prompts():
    source = SFT_JOB.read_text(encoding="utf-8")
    assert "targets = torch.full((block_size,), -100" in source
    assert "first_completion_prediction" in source
    assert "expected_arguments" in source
    assert "required_facts" in source
    assert "clean_stop" in source
    assert "accepted v0.0.9 promotion-eligible checkpoint" in source
    assert "ember-v0.0.9-t4" in source
    assert "ember-v0.0.10-t4" in source
    assert "official_cases_used_for_training\": 0" in source
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


def test_workflow_exposes_manual_semantic_sft_without_default_gpu_launch():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    marker = TRIGGER.read_text(encoding="utf-8")
    assert "sft-v010" in workflow
    assert "preflight-v010" in workflow
    assert "eval-v010" in workflow
    assert "--eval-spec v0.0.10" in workflow
    assert "steps.request.outputs.mode == 'sft-v010'" in workflow
    assert "--name ember-v0-0-10-semantic-sft" in workflow
    assert "--flavor t4-small" in workflow
    mode = marker.splitlines()[0].split(":", 1)[1].strip()
    assert mode not in {"train", "train-v008", "sft-v009", "sft-v010"}


def test_no_hugging_face_token_is_committed_for_v010():
    token_pattern = re.compile(r"hf_[A-Za-z0-9]{20,}")
    checked = [CONFIG, DATA_MODULE, SFT_JOB, EVAL_JOB, EVAL_SPEC, WORKFLOW]
    assert not any(token_pattern.search(path.read_text(encoding="utf-8")) for path in checked)
