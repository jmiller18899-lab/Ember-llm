"""Guards for the Ember v0.0.32 semantic quality gate.

v0.0.31 passed the legacy promotion evaluator on every rate and its own record
still says the completions are "noisy after the first scored segment". These
tests pin the difference: for every quality failure the legacy rubric accepts,
the strict rubric must reject, and for a genuinely good completion both must
agree. A gate that cannot fail on a known-bad input is not a gate.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "jobs" / "ember_hf_semantic_eval_v032.py"
SPEC = ROOT / "config" / "ember_semantic_quality_v0.0.32.json"
LEGACY_SPEC = ROOT / "config" / "ember_v0.0.8_eval.json"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return load(JOB, "ember_semantic_eval_v032")


@pytest.fixture(scope="module")
def spec():
    return json.loads(SPEC.read_text())


def case_of(spec, case_id):
    return next(c for c in spec["cases"] if c["id"] == case_id)


def both(gate, spec, case_id, completion):
    case = case_of(spec, case_id)
    strict = gate.score_case(case, completion, spec["quality"])
    return gate.score_legacy(case, completion), strict


def failing(strict):
    return sorted(k for k, v in strict["checks"].items() if not v)


GOOD_WEATHER = '<|tool|>{"name": "weather", "arguments": {"location": "Detroit"}}<|endoftext|>'
GOOD_DIRECT = "Hello there, it is good to meet you today.<|endoftext|>"


def test_the_job_imports_without_torch_or_huggingface_hub(gate):
    """The rubric has to be exercisable offline, or it cannot be tested."""
    assert hasattr(gate, "score_case") and hasattr(gate, "score_legacy")


def test_a_grounded_tool_call_passes_both_rubrics(gate, spec):
    legacy, strict = both(gate, spec, "tool_weather", GOOD_WEATHER)
    assert legacy and strict["passed"], failing(strict)


def test_a_good_direct_response_passes_both_rubrics(gate, spec):
    legacy, strict = both(gate, spec, "direct_greeting", GOOD_DIRECT)
    assert legacy and strict["passed"], failing(strict)


@pytest.mark.parametrize(
    "label,case_id,completion,expected_failures",
    [
        (
            "arguments that name nothing from the prompt",
            "tool_weather",
            '<|tool|>{"name": "weather", "arguments": {"x": 1}}<|endoftext|>',
            {"required_keys_present", "arguments_grounded"},
        ),
        (
            "the wrong city",
            "tool_weather",
            '<|tool|>{"name": "weather", "arguments": {"location": "Boston"}}<|endoftext|>',
            {"arguments_grounded"},
        ),
        (
            "arguments as an opaque non-JSON string",
            "tool_web_search",
            '<|tool|>{"name": "web_search", "arguments": "stuff"}<|endoftext|>',
            {"arguments_are_object", "required_keys_present", "arguments_grounded"},
        ),
        (
            "an invented user turn after the call",
            "tool_weather",
            '<|tool|>{"name": "weather", "arguments": {"location": "Detroit"}}'
            "<|user|> and tomorrow?<|endoftext|>",
            {"no_trailing_noise", "no_extra_turn_markers"},
        ),
        (
            "trailing prose after the call",
            "tool_weather",
            '<|tool|>{"name": "weather", "arguments": {"location": "Detroit"}} '
            "and then I will check the forecast<|endoftext|>",
            {"no_trailing_noise"},
        ),
        (
            "no EOS inside the budget",
            "tool_weather",
            '<|tool|>{"name": "weather", "arguments": {"location": "Detroit"}}',
            {"clean_stop"},
        ),
        (
            "two tool calls",
            "tool_weather",
            '<|tool|>{"name": "weather", "arguments": {"location": "Detroit"}}'
            '<|tool|>{"name": "weather", "arguments": {"location": "Detroit"}}<|endoftext|>',
            {"single_tool_marker", "no_trailing_noise"},
        ),
    ],
)
def test_legacy_accepts_what_the_strict_gate_rejects(
    gate, spec, label, case_id, completion, expected_failures
):
    """Each of these is a real quality failure the legacy rubric scores as a pass."""
    legacy, strict = both(gate, spec, case_id, completion)
    assert legacy, f"{label}: this case is only interesting if legacy accepts it"
    assert not strict["passed"], label
    assert set(failing(strict)) == expected_failures, f"{label}: {failing(strict)}"


@pytest.mark.parametrize(
    "label,completion,expected_failures",
    [
        ("three characters", "ok.<|endoftext|>", {"long_enough", "enough_words"}),
        (
            "a degenerate loop",
            "the weather in the weather in the weather in the weather in "
            "the weather in the weather in<|endoftext|>",
            {"no_degenerate_repetition"},
        ),
        (
            "an invented conversation",
            "Hello there, nice to meet you.<|user|> and you?<|endoftext|>",
            {"no_extra_turn_markers"},
        ),
        (
            "never stopping",
            "Hello there, it is good to meet you today and I will keep going",
            {"clean_stop"},
        ),
    ],
)
def test_direct_responses_that_legacy_waves_through(gate, spec, label, completion, expected_failures):
    legacy, strict = both(gate, spec, "direct_greeting", completion)
    assert legacy, f"{label}: only interesting if legacy accepts it"
    assert not strict["passed"], label
    assert set(failing(strict)) == expected_failures, f"{label}: {failing(strict)}"


def test_a_missing_tool_call_fails_both(gate, spec):
    legacy, strict = both(gate, spec, "tool_weather", "I think it is sunny.<|endoftext|>")
    assert not legacy and not strict["passed"]


def test_calculator_grounding_needs_both_operands(gate, spec):
    _, ok = both(gate, spec, "tool_calculator",
                 '<|tool|>{"name": "calculator", "arguments": {"expression": "347*28"}}<|endoftext|>')
    assert ok["passed"], failing(ok)
    _, half = both(gate, spec, "tool_calculator",
                   '<|tool|>{"name": "calculator", "arguments": {"expression": "347*29"}}<|endoftext|>')
    assert not half["passed"]
    assert half["grounded_missing"] == ["28"]


def test_grounding_case_sensitivity_follows_the_spec(gate, spec):
    # weather grounding is case-insensitive, so a lowercased city still counts.
    _, relaxed = both(gate, spec, "tool_weather",
                      '<|tool|>{"name": "weather", "arguments": {"location": "detroit"}}<|endoftext|>')
    assert relaxed["checks"]["arguments_grounded"]
    assert case_of(spec, "tool_calculator")["grounding_case_sensitive"] is True


def test_the_report_names_every_case_the_two_rubrics_disagree_on(gate, spec):
    rows = []
    for case_id, completion in (
        ("tool_weather", '<|tool|>{"name": "weather", "arguments": {"x": 1}}<|endoftext|>'),
        ("tool_calculator", '<|tool|>{"name": "calculator", "arguments": {"expression": "347*28"}}<|endoftext|>'),
        ("tool_web_search", '<|tool|>{"name": "web_search", "arguments": {"query": "latest Python release"}}<|endoftext|>'),
        ("tool_get_time", '<|tool|>{"name": "get_time", "arguments": {"timezone": "Tokyo"}}<|endoftext|>'),
        ("direct_greeting", GOOD_DIRECT),
        ("direct_rewrite", GOOD_DIRECT),
        ("direct_explain", GOOD_DIRECT),
        ("direct_plan", GOOD_DIRECT),
        ("result_weather", GOOD_DIRECT),
        ("result_calculator", GOOD_DIRECT),
        ("result_search", GOOD_DIRECT),
        ("result_service_status", GOOD_DIRECT),
    ):
        case = case_of(spec, case_id)
        rows.append({
            "id": case_id, "kind": case["kind"], "completion": completion,
            "strict": gate.score_case(case, completion, spec["quality"]),
            "legacy_passed": gate.score_legacy(case, completion),
        })
    summary = gate.aggregate(rows, spec)
    assert summary["metrics"]["legacy_pass_rate"] == 1.0
    assert summary["metrics"]["overall_strict_pass_rate"] == pytest.approx(11 / 12)
    assert summary["legacy_passes_but_strict_fails"] == ["tool_weather"]
    assert summary["passed"] is False


def test_every_spec_case_maps_to_a_legacy_prompt(spec):
    legacy = json.loads(LEGACY_SPEC.read_text())
    prompts = {case["id"] for case in legacy["cases"]}
    assert {case["id"] for case in spec["cases"]} == prompts
    assert spec["candidate_model_name"] == "ember-v0.0.31-t4"
    # Every gate demands perfection; this is a pre-deployment bar, not a trend.
    assert all(value == 1.0 for value in spec["gates"].values())


def test_the_job_never_trains_uploads_or_promotes():
    source = JOB.read_text()
    for forbidden in ("upload_file", "save_checkpoint", "loss.backward", "optimizer", "cuda"):
        assert forbidden not in source, f"a read-only evaluator must not reference {forbidden}"
