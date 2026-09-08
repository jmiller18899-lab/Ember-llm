from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from jobs import ember_sft_data_semantic_v1 as data
from jobs import ember_semantic_data_preflight as preflight

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "config/ember_semantic_v1.json").read_text())
CONFIG = json.loads((ROOT / "config/ember_semantic_data_v1.json").read_text())


def row(family, group=0, side=0):
    return data.make_record(family, group, side, CONFIG["seed"])


def refresh(record):
    record["prompt"] = data.render_prompt(record["messages"])
    return record


def result(record, value):
    record["messages"][-1]["content"] = data.compact(value)
    return refresh(record)


def test_complete_dataset_is_balanced_reproducible_and_excludes_frozen_cases():
    splits = data.build_dataset()
    report = data.validate_dataset(splits)
    assert report["corrupted_targets_rejected"] == 6912
    assert {k: v["rows"] for k, v in report["splits"].items()} == {"train": 2880, "validation": 576}
    assert preflight.check_separation(splits, SPEC)["normalized_unique_inputs"] == 3456
    preflight.verify_frozen(CONFIG)
    for split, rows in splits.items():
        assert data.jsonl_bytes(rows) == data.jsonl_bytes(data.build_dataset()[split])
        assert all(r["completion"].endswith("\n<|endoftext|>") for r in rows)


@pytest.mark.parametrize("payload", [
    {"name": "weather", "arguments": {"location": "Eugene"}},
    {"name": "weather", "arguments": {"city": "Asheville"}},
    {"name": "weather", "arguments": {"location": 12}},
    {"name": "weather", "arguments": {"location": "Asheville", "extra": True}},
    {"name": "get_time", "arguments": {"location": "Asheville"}},
    {"name": "weather", "arguments": '{"location":"Asheville"}'},
])
def test_correct_tool_name_and_nonempty_arguments_are_insufficient(payload):
    record = row("tool_call/weather_literal")
    record["completion"] = data.done(data.TOOL + "\n" + data.compact(payload))
    record["expected_tool"] = payload["name"]  # Matching metadata cannot override the request.
    with pytest.raises(ValueError):
        data.validate_row(record)


def test_oracle_reads_requested_argument_with_quotes_braces_and_distractor():
    record = row("tool_call/search_choice")
    record["messages"][1]["content"] = r'Use web_search with query "patch \"beta\" {preview}", not "old build".'
    refresh(record)
    record["completion"] = data.done('<|tool|>\n' + r'{"arguments":{"query":"patch \"beta\" {preview}"},"name":"web_search"}')
    data.validate_row(record)
    record["messages"][1]["content"] = r'Use web_search with query "old build", not "patch \"beta\" {preview}".'
    refresh(record)
    with pytest.raises(ValueError, match="actual request"):
        data.validate_row(record)


@pytest.mark.parametrize("family,answer", [
    ("tool_result_response/weather_f", "It is 19°F and clear in Asheville."),
    ("tool_result_response/weather_c", "It is -13°F and clear in Asheville."),
    ("tool_result_response/arithmetic", "The result is 155."),
    ("tool_result_response/service", "The worker is unhealthy."),
    ("tool_result_response/error", "It is sunny in Asheville."),
    ("tool_result_response/search", "I found a result."),
    ("tool_result_response/checks", "All 70 checks passed."),
])
def test_wrong_sign_unit_value_status_or_invented_success_is_rejected(family, answer):
    record = row(family)
    record["completion"] = data.done(answer)
    with pytest.raises(ValueError, match="actual request or tool result"):
        data.validate_row(record)


def test_mutating_result_requires_mutating_the_answer_and_validates_arithmetic():
    record = result(row("tool_result_response/weather_f"), {"temperature_f": -8, "condition": "rainy"})
    with pytest.raises(ValueError):
        data.validate_row(record)
    record["completion"] = data.done("It is -8°F and rainy in Asheville.")
    data.validate_row(record)
    record = result(row("tool_result_response/arithmetic"), {"result": 999})
    record["completion"] = data.done("The result is 999.")
    with pytest.raises(ValueError, match="arithmetic result is false"):
        data.validate_row(record)


@pytest.mark.parametrize("family,value", [
    ("tool_result_response/weather_f", {"temperature_f": True, "condition": "clear"}),
    ("tool_result_response/weather_c", {"temperature_c": "-13", "condition": "clear"}),
    ("tool_result_response/service", {"status": "healthy", "latency_ms": True}),
    ("tool_result_response/error", {"error": "timeout", "condition": "sunny"}),
    ("tool_result_response/search", {"results": ""}),
    ("tool_result_response/missing_owner", {"status": "healthy", "owner": None}),
    ("tool_result_response/checks", {"total_checks": 7, "failed_checks": 8}),
])
def test_invalid_result_fields_and_types_fail_before_targets_are_accepted(family, value):
    with pytest.raises(ValueError):
        data.validate_row(result(row(family), value))


@pytest.mark.parametrize("change", ["name", "extra", "marker", "duplicate"])
def test_result_history_is_a_strict_call_to_the_correct_tool(change):
    record = row("tool_result_response/weather_f")
    if change == "name":
        record["messages"][2]["content"] = '<|tool|>\n{"name":"web_search","arguments":{"location":"Asheville"}}'
    elif change == "extra":
        record["messages"][2]["content"] = '<|tool|>\n{"name":"weather","arguments":{"location":"Asheville"},"extra":true}'
    elif change == "duplicate":
        record["messages"][2]["content"] = '<|tool|>\n{"name":"web_search","name":"weather","arguments":{"location":"Asheville"}}'
    else:
        record["messages"][2]["content"] = record["messages"][2]["content"].removeprefix("<|tool|>\n")
    with pytest.raises(ValueError):
        data.validate_row(refresh(record))


def test_missing_information_and_explicit_false_are_preserved():
    missing = row("direct_response/missing_owner", side=1)
    assert missing["completion"] == data.done("Owner unavailable.")
    missing["completion"] = data.done("The owner is Omar.")
    with pytest.raises(ValueError):
        data.validate_row(missing)
    saved = row("direct_response/saved", side=1)
    assert "was not saved" in saved["completion"]
    saved["completion"] = saved["completion"].replace("was not saved", "was saved")
    with pytest.raises(ValueError):
        data.validate_row(saved)


def test_definitions_are_specific_and_do_not_use_the_old_generic_fallback():
    assert data.DEFINITIONS["checkpoint"] == "A checkpoint stores model weights and training state so an interrupted training run can continue."
    assert len(set(data.DEFINITIONS.values())) == 18
    assert not any("a practical mechanism" in answer for answer in data.DEFINITIONS.values())


def test_benchmark_exclusion_ignores_system_and_normalizes_json_and_whitespace():
    record = row("tool_call/weather_literal")
    messages = preflight.parse_prompt(SPEC["cases"][0]["prompt"])
    messages[0]["content"] = "Different system instructions cannot hide a benchmark prompt."
    record["messages"] = messages
    with pytest.raises(ValueError, match="benchmark input"):
        preflight.check_separation({"train": [record], "validation": []}, SPEC)
    record = row("tool_call/weather_literal")
    record["completion"] = data.done('<|tool|>\n{"name":"weather","arguments":{"location":"  DETROIT  "}}')
    with pytest.raises(ValueError, match="benchmark tool arguments"):
        preflight.check_separation({"train": [record], "validation": []}, SPEC)
    first = row("tool_result_response/weather_f")
    second = copy.deepcopy(first)
    second["messages"][0]["content"] = "A different system"
    second["messages"][2]["content"] = '<|tool|> {"name": "weather", "arguments": {"location": "Asheville"}}'
    with pytest.raises(ValueError, match="duplicate normalized input"):
        preflight.check_separation({"train": [first], "validation": [second]}, SPEC)


def test_counterfactual_pairs_cannot_be_split_by_swapping_valid_rows():
    splits = data.build_dataset()
    family = splits["train"][0]["family"]
    i = next(i for i, r in enumerate(splits["validation"]) if r["family"] == family)
    splits["train"][0], splits["validation"][i] = splits["validation"][i], splits["train"][0]
    splits["train"][0]["split"], splits["validation"][i]["split"] = "train", "validation"
    with pytest.raises(ValueError, match="pair"):
        data.validate_dataset(splits)


def test_completion_loss_masks_first_target_eos_and_padding_without_truncation():
    torch = pytest.importorskip("torch")
    class Tokens:
        def encode(self, text):
            return [7, 8] if text == "prompt" else [7, 8, 11, 12, 2]
    record = {"id": "tiny", "prompt": "prompt", "completion": "answer"}
    (x, y), sizes = preflight.encode_row(Tokens(), record, 8, 4, 2, torch)
    assert x.tolist() == [7, 8, 11, 12, 2, 2, 2, 2]
    assert y.tolist() == [-100, 11, 12, 2, -100, -100, -100, -100]
    assert sizes == {"prompt": 2, "completion": 3, "total": 5}
    with pytest.raises(ValueError, match="context overflow"):
        preflight.encode_row(Tokens(), record, 3, 4, 2, torch)
    with pytest.raises(ValueError, match="generation context overflow"):
        preflight.encode_row(Tokens(), record, 5, 4, 2, torch)


@pytest.mark.parametrize("ids", [[7, 9, 11, 2], [7, 8, 2, 11], [7, 8, 2, 2]])
def test_unstable_boundaries_and_missing_or_repeated_final_eos_fail(ids):
    torch = pytest.importorskip("torch")
    class Tokens:
        def encode(self, text):
            return [7, 8] if text == "prompt" else ids
    with pytest.raises(ValueError):
        preflight.encode_row(Tokens(), {"id": "bad", "prompt": "prompt", "completion": "answer"}, 8, 4, 2, torch)
