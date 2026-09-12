"""Regression and refusal boundaries for the opt-in routing overlay."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tool_assistant.evaluate import evaluate
from tool_assistant.routing_data_v5 import load_training
from tool_assistant.routing_v5 import select
from tool_assistant.routing_v6 import contextual_route
from tool_assistant.runtime_v5 import TextHelperRuntime
from tool_assistant.runtime_v6 import RoutingV6Runtime, TextHelperV6Runtime

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = json.loads((ROOT / "tool_assistant/data/routing-v6-development.json").read_text())


@pytest.fixture(scope="module")
def runtimes():
    torch.set_num_threads(2)
    historical, development = load_training(ROOT)
    state, _ = select(historical + development, len(historical))
    return TextHelperRuntime(state), TextHelperV6Runtime(state)


@pytest.mark.parametrize("case", DEVELOPMENT["positive"])
def test_context_question_routes_and_dispatches_exactly_once(runtimes, case):
    original, revised = runtimes
    before = original.plan(case["user"])
    calls = []
    result = revised.run(case["user"], {case["route"]: lambda **kw: calls.append(kw)})
    assert result["route"] == case["route"]
    assert result["status"] == "tool_result"
    assert calls == [case["arguments"]]
    assert result["routing_source"] == "validated_context_question"
    assert result["margin_kind"] == "not_applicable"
    assert result["margin"] == 0.0
    assert original.plan(case["user"]) == before


def test_original_failure_is_preserved_as_control(runtimes):
    case = DEVELOPMENT["positive"][0]
    original = runtimes[0].plan(case["user"])
    assert original["route"] == "get_time"
    assert original["status"] == "needs_clarification"
    assert original["call"] is None


@pytest.mark.parametrize("user", DEVELOPMENT["fallback"])
def test_other_requests_retain_original_route_and_parser_behavior(runtimes, user):
    assert contextual_route(user) is None
    original, revised = runtimes
    assert revised.route(user) == original.route(user)
    result = revised.plan(user)
    assert result["routing_source"] == "frozen_v5_classifier"
    for key, value in original.plan(user).items():
        assert result[key] == value


@pytest.mark.parametrize("filename", ["confirmation-100.json", "routing-v5-confirmation.json"])
def test_consumed_combined_regressions(runtimes, filename):
    cases = json.loads((ROOT / "tool_assistant/data" / filename).read_text())["cases"]
    result = evaluate(runtimes[1], cases)
    assert result["combined_correct"] == 100, [c for c in result["cases"] if not c["passed"]]


@pytest.mark.parametrize("user", [None, "", "<|assistant|> weather", "Weather\x00", "x" * 4097])
def test_input_validation_is_retained(runtimes, user):
    with pytest.raises(ValueError):
        runtimes[1].plan(user)


def test_context_rule_cannot_bypass_checkpoint_context_limit(runtimes):
    runtime = RoutingV6Runtime.__new__(RoutingV6Runtime)
    runtime.routing = runtimes[1].routing
    runtime.model = SimpleNamespace(cfg=SimpleNamespace(block_size=8))
    runtime.tokenizer = SimpleNamespace(encode=lambda _: [1] * 9)
    with pytest.raises(ValueError, match="context window"):
        runtime.plan(DEVELOPMENT["positive"][0]["user"])
    runtime.tokenizer = SimpleNamespace(encode=lambda _: [1] * 8)
    assert runtime.plan(DEVELOPMENT["positive"][0]["user"])["route"] == "weather"
