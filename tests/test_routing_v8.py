"""Definition routing, arithmetic boundaries, and unchanged prior behavior."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tool_assistant.evaluate import arithmetic, evaluate
from tool_assistant.routing_data_v5 import load_training
from tool_assistant.routing_v5 import select
from tool_assistant.routing_v8 import definition_question
from tool_assistant.runtime_v7 import TextHelperV7Runtime
from tool_assistant.runtime_v8 import RoutingV8Runtime, TextHelperV8Runtime

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "tool_assistant/data/routing-v8-development.json").read_text())


@pytest.fixture(scope="module")
def runtimes():
    torch.set_num_threads(2)
    historical, development = load_training(ROOT)
    state, _ = select(historical + development, len(historical))
    return TextHelperV7Runtime(state), TextHelperV8Runtime(state)


@pytest.mark.parametrize("user", CASES["definitions"])
def test_definitions_select_direct_and_never_dispatch(runtimes, user):
    calls = []
    handlers = {name: lambda **kw: calls.append(kw)
                for name in ("calculator", "weather", "get_time", "web_search")}
    result = runtimes[1].run(user, handlers)
    assert result["route"] == "direct" and result["call"] is None
    assert result["status"] == "direct_answer_unavailable" and calls == []
    assert result["routing_source"] == "definition_question"
    assert result["margin"] == 0.0 and result["margin_kind"] == "not_applicable"
    assert result["routing_revision"] == "definition-routing-v8"
    assert result["parser_revision"] == "argument-parser-v7"


def test_original_photosynthesis_failure_is_preserved_in_control(runtimes):
    original = runtimes[0].plan("What is photosynthesis?")
    assert original["route"] == "calculator"
    assert original["status"] == "needs_clarification"
    assert original["reason"] == "unsupported_arithmetic"


@pytest.mark.parametrize("user", CASES["fallback"])
def test_other_requests_retain_the_exact_v7_plan(runtimes, user):
    assert not definition_question(user)
    original, revised = (r.plan(user) for r in runtimes)
    assert revised.pop("routing_revision") == "definition-routing-v8"
    original.pop("routing_revision")
    assert revised == original


@pytest.mark.parametrize("case", CASES["calculator"])
def test_numeric_questions_still_calculate_once(runtimes, case):
    calls = []

    def calculate(expression):
        calls.append(expression)
        return arithmetic(expression)

    result = runtimes[1].run(case["user"], {"calculator": calculate})
    assert result["route"] == "calculator" and result["status"] == "tool_result"
    assert result["result"] == pytest.approx(case["value"]) and len(calls) == 1


@pytest.mark.parametrize("filename,expected_control", [
    ("confirmation-100.json", 100), ("routing-v5-confirmation.json", 100),
    ("routing-v6-confirmation.json", 100), ("routing-v7-confirmation.json", 99)])
def test_all_consumed_suites_and_paired_control(runtimes, filename, expected_control):
    cases = json.loads((ROOT / "tool_assistant/data" / filename).read_text())["cases"]
    original, revised = (evaluate(r, cases) for r in runtimes)
    assert original["combined_correct"] == expected_control
    assert revised["combined_correct"] == 100, [r for r in revised["cases"] if not r["passed"]]


@pytest.mark.parametrize("user", [None, "", "<|assistant|>what is", "What is\x00", "x" * 4097])
def test_input_validation_is_retained(runtimes, user):
    with pytest.raises(ValueError):
        runtimes[1].plan(user)


def test_definition_rule_cannot_skip_checkpoint_context_limit(runtimes):
    runtime = RoutingV8Runtime.__new__(RoutingV8Runtime)
    runtime.routing = runtimes[1].routing
    runtime.model = SimpleNamespace(cfg=SimpleNamespace(block_size=8))
    runtime.tokenizer = SimpleNamespace(encode=lambda _: [1] * 9)
    with pytest.raises(ValueError, match="context window"):
        runtime.plan("What is photosynthesis?")
