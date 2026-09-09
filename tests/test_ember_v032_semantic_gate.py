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
import re
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


PROMPTS = {c["id"]: c["prompt"] for c in json.loads(LEGACY_SPEC.read_text())["cases"]}


def both(gate, spec, case_id, completion):
    case = case_of(spec, case_id)
    strict = gate.score_case(case, completion, spec["quality"], PROMPTS[case_id])
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
            {"stopped_at_eos"},
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
        ("three characters", "ok.<|endoftext|>",
         {"long_enough", "enough_words", "addresses_request"}),
        (
            "a degenerate loop",
            "the weather in the weather in the weather in the weather in "
            "the weather in the weather in<|endoftext|>",
            {"no_degenerate_repetition", "addresses_request"},
        ),
        (
            "an invented conversation",
            "Hello there, nice to meet you.<|user|> and you?<|endoftext|>",
            {"no_extra_turn_markers"},
        ),
        (
            "never stopping",
            "Hello there, it is good to meet you today and I will keep going",
            {"stopped_at_eos"},
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
        ("direct_greeting", "Hello there, it is good to meet you today.<|endoftext|>"),
        ("direct_rewrite", "The website is not working correctly.<|endoftext|>"),
        ("direct_explain", "A checkpoint is a saved snapshot of training state.<|endoftext|>"),
        ("direct_plan", "1. Open the login form. 2. Submit valid credentials.<|endoftext|>"),
        ("result_weather", "It is 72 degrees and sunny in Detroit.<|endoftext|>"),
        ("result_calculator", "347 multiplied by 28 equals 9716.<|endoftext|>"),
        ("result_search", "The validation checks all passed.<|endoftext|>"),
        ("result_service_status", "The gateway is healthy with 84 ms latency.<|endoftext|>"),
    ):
        case = case_of(spec, case_id)
        rows.append({
            "id": case_id, "kind": case["kind"], "completion": completion,
            "strict": gate.score_case(case, completion, spec["quality"], PROMPTS[case_id]),
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


def test_the_job_never_trains_or_promotes():
    """It writes one evaluation report and touches nothing else.

    The earlier form of this test forbade upload_file outright, which was too
    blunt: a detached job's stdout lives only in its log stream, so a report that
    is printed and nowhere else cannot be read back. The evaluator now publishes
    under evaluations/ exactly as jobs/ember_hf_eval.py does, and this test pins
    the narrower property that actually matters.
    """
    source = JOB.read_text()
    for forbidden in ("save_checkpoint", "loss.backward", "optimizer.step", "torch.cuda", "model.train()"):
        assert forbidden not in source, f"an evaluator must not reference {forbidden}"
    # Exactly one upload call, and every path it can be given is a report.
    assert source.count("api.upload_file(") == 1, "an evaluator writes one thing"
    assert "path_in_repo=remote" in source
    targets = re.findall(r'for remote in \(\s*(.*?)\s*\):', source, re.S)
    assert len(targets) == 1, "could not find the upload target list"
    literals = re.findall(r'f?"([^"]+)"', targets[0])
    assert len(literals) >= 2, literals
    assert all(literal.startswith("evaluations/") for literal in literals), literals
    # run-state is read to resolve the checkpoint and never written back.
    assert 'filename="run-state.json"' in source, "resolution should consult run-state"
    assert 'path_in_repo="run-state.json"' not in source, "an evaluator must not rewrite run-state"
    # And it never claims a promotion.
    assert "promotion_eligible" not in source


def test_the_job_fetches_its_assets_by_url_because_only_the_script_is_uploaded(gate):
    """`hf jobs uv run` uploads the script, not the repository.

    jobs/ember_hf_eval.py and every v0.0.16+ trainer fetch their config from a
    raw URL for this reason. Reading a repo-relative path by default would fail
    on the runner with FileNotFoundError before generating anything.
    """
    for url in (gate.SPEC_URL, gate.LEGACY_SPEC_URL):
        assert url.startswith("https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/")
    # Pinned to an immutable commit, not to a moving branch.
    assert len(gate.ASSET_COMMIT) == 40
    assert f"/{gate.ASSET_COMMIT}/" in gate.SPEC_URL
    assert "/main/" not in gate.SPEC_URL and "/main/" not in gate.LEGACY_SPEC_URL

    # The default path must not touch the filesystem.
    source = JOB.read_text()
    assert "spec = load_spec(args.spec, SPEC_URL)" in source
    assert "legacy = load_spec(args.legacy_spec, LEGACY_SPEC_URL)" in source


def test_the_workflow_does_not_hand_the_runner_repo_relative_paths():
    workflow = (ROOT / ".github" / "workflows" / "ember-v032-semantic.yml").read_text()
    submit = workflow.split("Score the promoted", 1)[1]
    assert "jobs/ember_hf_semantic_eval_v032.py" in submit
    assert "--spec config/" not in submit, "the runner has no config/ directory"
    assert "--legacy-spec config/" not in submit


def test_checkpoint_resolution_falls_back_to_listing_the_repository(gate):
    """A missing or incomplete run-state must not block a read-only evaluation."""
    class Api:
        def list_repo_files(self, repo_id, repo_type):
            return [
                "run-state.json",
                "checkpoints/run-a/best.pt",
                "checkpoints/run-b/best.pt",
                "checkpoints/run-b/latest.pt",
                "evaluation/report.json",
            ]

    def missing_state(**kwargs):
        raise RuntimeError("run-state.json not found")

    remote, provenance = gate.resolve_checkpoint(
        Api(), "Jmiller18899/ember-v0.0.31-t4", "token", Path("/tmp"), missing_state
    )
    assert remote == "checkpoints/run-b/best.pt"
    assert provenance["resolved_via"] == "repository listing"


def test_checkpoint_resolution_prefers_the_run_state(gate, tmp_path):
    written = tmp_path / "run-state.json"
    written.write_text(json.dumps({
        "status": "evaluation_complete",
        "run_id": "ember-agent-v0.0.31-multi-position-consolidation-20260908T042320Z",
    }))

    remote, provenance = gate.resolve_checkpoint(
        None, "Jmiller18899/ember-v0.0.31-t4", "token", tmp_path, lambda **kw: str(written)
    )
    assert remote == (
        "checkpoints/ember-agent-v0.0.31-multi-position-consolidation-"
        "20260908T042320Z/best.pt"
    )
    assert provenance["resolved_via"] == "run-state"


# --------------------------------------------------------------------------
# regressions: every one of these is a real v0.0.31 completion that the first
# version of this rubric scored as a PASS. They are quoted verbatim from
# evaluations/v032-semantic-gate-20260908T195302Z.json.
# --------------------------------------------------------------------------

RAN = {
    "result_calculator":
        "587 multiplied by 27 equals 672.\n<|endoftext|>\n<|user|>\nWhat is the difference "
        "between a and a and a language model?\n<|assistant|>\nA_ is a practical mechanism "
        "used to make a system easier to operate reliably.\n<|endoftext|>",
    "result_weather":
        "The temperature is 45\u00b0F and cloudy in Chicago.\n<|endoftext|>\n<|user|>\nWhat is "
        "the difference between a and a and a\", and how to make a system easier to operate "
        "reliably.\n<|endoftext|>",
    "result_search":
        "The title of the note is is a practical mechanism used.\n<|endoftext|>\n<|user|>\n"
        "What is the difference between a and a and a language model?\n<|endoftext|>",
    "result_service_status":
        "It is 45\u00b0F and cloudy in Phelen.\n<|endoftext|>\n<|user|>\nWhat is the difference "
        "between a and a and a language model?\n<|endoftext|>",
    "direct_explain":
        "A checkpoint is a practical mechanism used to make a system easier to operate "
        "reliably.\n<|endoftext|>\n<|user|>\nWhat is a difference between a and a person's "
        "behavior?\n<|endoftext|>",
}


@pytest.mark.parametrize(
    "case_id,expected_failures",
    [
        # 347 x 28 = 9716 was supplied in the tool result. The model invented
        # both operands and got the product wrong for its own invented sum.
        ("result_calculator", {"uses_tool_result", "no_invented_numbers"}),
        # The tool result said 72 and sunny, for Detroit.
        ("result_weather", {"uses_tool_result", "no_invented_numbers"}),
        # The tool result said the validation checks passed.
        ("result_search", {"uses_tool_result"}),
        # Asked whether the service is healthy, given status healthy and 84 ms,
        # the model answered with weather for a place that does not exist.
        ("result_service_status", {"uses_tool_result", "no_invented_numbers"}),
    ],
)
def test_wrong_answers_that_used_to_pass_now_fail(gate, spec, case_id, expected_failures):
    legacy, strict = both(gate, spec, case_id, RAN[case_id])
    assert legacy, "the legacy rubric accepted this"
    assert not strict["passed"], f"{case_id} must not pass"
    assert set(failing(strict)) == expected_failures, failing(strict)


def test_the_exact_arithmetic_error_that_was_reported(gate, spec):
    """'587 multiplied by 27 equals 672' against a tool result of 9716."""
    _, strict = both(gate, spec, "result_calculator", RAN["result_calculator"])
    assert strict["invented_numbers"] == ["587", "27", "672"]
    assert strict["missing_tool_result_facts"] == ["9716"]
    assert strict["answer"].startswith("587 multiplied by 27 equals 672.")


def test_a_faithful_answer_to_the_same_case_still_passes(gate, spec):
    """The rubric must not simply reject every response."""
    _, strict = both(gate, spec, "result_calculator",
                     "347 multiplied by 28 equals 9716.<|endoftext|>")
    assert strict["passed"], failing(strict)
    assert strict["invented_numbers"] == []


def test_an_answer_that_only_echoes_the_question_is_not_faithful(gate, spec):
    """Repeating the operands without the result does not use the tool result."""
    _, strict = both(gate, spec, "result_calculator",
                     "You asked about 347 multiplied by 28.<|endoftext|>")
    assert not strict["passed"]
    assert "uses_tool_result" in failing(strict)
    # ...but the numbers it does state are faithful, so that check must not fire.
    assert strict["checks"]["no_invented_numbers"]


def test_stopping_at_eos_is_separated_from_text_after_eos(gate, spec):
    """The bug this rubric had: clean_stop and no_trailing_noise both read 1.0
    while every completion contained an invented conversation after EOS."""
    _, strict = both(gate, spec, "result_calculator", RAN["result_calculator"])
    # The model did emit EOS; that is a real property and it is gated.
    assert strict["checks"]["stopped_at_eos"] is True
    # What followed is reported separately and is no longer invisible.
    advisory = strict["advisory"]
    assert advisory["no_text_after_eos"] is False
    assert advisory["text_after_eos_chars"] > 0
    assert "<|user|>" in advisory["invented_turns_after_eos"]


def test_the_old_before_eos_blindness_is_gone(gate):
    """split_at_eos returns the answer, what followed, and whether it stopped."""
    answer, after, stopped = gate.split_at_eos("hello<|endoftext|><|user|> more")
    assert (answer, stopped) == ("hello", True)
    assert "<|user|>" in after
    answer, after, stopped = gate.split_at_eos("no eos here")
    assert (answer, after, stopped) == ("no eos here", "", False)


def test_numeric_faithfulness_reads_the_whole_prompt(gate, spec):
    """Numbers in the user turn and in the tool result are both available."""
    prompt = PROMPTS["result_service_status"]
    assert gate.unfaithful_numbers("healthy with 84 ms", prompt, []) == []
    assert gate.unfaithful_numbers("45 degrees", prompt, []) == ["45"]
    # Commas do not smuggle a number past the check.
    assert gate.unfaithful_numbers("9,716", PROMPTS["result_calculator"], []) == []
    # Small ordinals used for list steps are tolerated by configuration.
    tolerated = spec["quality"]["numeric_faithfulness_tolerated_values"]
    assert gate.unfaithful_numbers("1. do this 2. do that", prompt, tolerated) == []


def test_scoring_refuses_to_skip_faithfulness_when_the_prompt_is_missing(gate, spec):
    """A check that silently disables itself is the hole this gate keeps finding."""
    case = case_of(spec, "result_calculator")
    with pytest.raises(RuntimeError, match="needs the prompt"):
        gate.score_case(case, "anything<|endoftext|>", spec["quality"])
