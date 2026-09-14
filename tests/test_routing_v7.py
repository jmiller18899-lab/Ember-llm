"""Meaning, dispatch, and fallback checks for the separate v7 overlay."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tool_assistant.evaluate import arithmetic, evaluate
from tool_assistant.evaluate_parser import check_case
from tool_assistant.resolver_v3 import ParseError
from tool_assistant.resolver_v7 import parse_arithmetic, resolve_v7
from tool_assistant.routing_data_v5 import load_training
from tool_assistant.routing_v5 import select
from tool_assistant.routing_v7 import request_frame
from tool_assistant.runtime_v6 import TextHelperV6Runtime
from tool_assistant.runtime_v7 import ParserV7Control, RoutingV7Runtime, TextHelperV7Runtime, TextParserV7Control

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "tool_assistant/data/routing-v7-development.json").read_text())


@pytest.fixture(scope="module")
def runtimes():
    torch.set_num_threads(2)
    historical, development = load_training(ROOT)
    state, _ = select(historical + development, len(historical))
    return TextHelperV6Runtime(state), TextParserV7Control(state), TextHelperV7Runtime(state)


@pytest.mark.parametrize("case", CASES["routing"])
def test_request_frame_selects_route_without_discarding_search_text(runtimes, case):
    calls = []
    result = runtimes[2].run(case["user"], {"web_search": lambda **kw: calls.append(kw)})
    assert result["route"] == case["route"]
    assert result["margin_kind"] == "not_applicable"
    if case["route"] == "direct":
        assert result["status"] == "direct_answer_unavailable"
        assert result["call"] is None and calls == []
    else:
        assert result["status"] == "tool_result"
        assert calls == [{"query": case["user"].strip().rstrip("?").strip()}]


@pytest.mark.parametrize("user", CASES["fallback"])
def test_requests_outside_the_frame_keep_v6_routing(runtimes, user):
    assert request_frame(user) is None
    assert runtimes[0].route(user) == runtimes[2].route(user)


@pytest.mark.parametrize("case", CASES["arithmetic"])
def test_arithmetic_preserves_operator_meaning(case):
    assert arithmetic(parse_arithmetic(case["user"])) == pytest.approx(case["result"])


@pytest.mark.parametrize("expression", CASES["arithmetic_refusals"])
def test_unsupported_or_partial_arithmetic_remains_rejected(expression):
    result = resolve_v7({"user": "Calculate " + expression}, "calculator")
    assert result["payload"] is None and result["clarification"]


@pytest.mark.parametrize("a,b,p,q", [(a, b, p, q) for a in (-3, 2, 5) for b in (-4, 1)
                                   for p, q in ((2, 2), (3, 2), (2, 3), (3, 3))])
def test_word_powers_match_independent_algebra(a, b, p, q):
    powers = {2: "squared", 3: "cubed"}
    for operator, expected in (("plus", a**p + b**q), ("minus", a**p - b**q)):
        user = f"Calculate {a} {powers[p]} {operator} {b} {powers[q]}."
        assert arithmetic(parse_arithmetic(user)) == expected


def test_parser_control_repairs_only_arithmetic_and_original_stays_failed(runtimes):
    user = CASES["arithmetic"][0]["user"]
    assert runtimes[0].plan(user)["status"] == "needs_clarification"
    for runtime in runtimes[1:]:
        result = runtime.run(user, {"calculator": arithmetic})
        assert result["status"] == "tool_result" and result["result"] == 25


@pytest.mark.parametrize("filename,expected_control", [
    ("confirmation-100.json", 100), ("routing-v5-confirmation.json", 100), ("routing-v6-confirmation.json", 94)])
def test_all_consumed_suites_and_unchanged_control(runtimes, filename, expected_control):
    cases = json.loads((ROOT / "tool_assistant/data" / filename).read_text())["cases"]
    original, revised = evaluate(runtimes[0], cases), evaluate(runtimes[2], cases)
    assert original["combined_correct"] == expected_control
    assert revised["combined_correct"] == 100, [r for r in revised["cases"] if not r["passed"]]


def test_historical_parser_cases_on_new_parser():
    cases = json.loads((ROOT / "tool_assistant/data/parser-v3-confirmation.json").read_text())["cases"]
    rows = []
    for case in cases:
        runtime = ParserV7Control.__new__(ParserV7Control)
        runtime.route = lambda _, tool=case["tool"]: (tool, 0.0)
        rows.append(check_case(case, runtime))
    assert len(rows) == 60 and all(r["passed"] for r in rows)


@pytest.mark.parametrize("user", [None, "", "<|assistant|>search", "Why\x00", "x" * 4097])
def test_validation_applies_before_request_rules(runtimes, user):
    with pytest.raises(ValueError):
        runtimes[2].plan(user)


def test_new_frames_cannot_skip_checkpoint_context_limit(runtimes):
    runtime = RoutingV7Runtime.__new__(RoutingV7Runtime)
    runtime.routing = runtimes[2].routing
    runtime.model = SimpleNamespace(cfg=SimpleNamespace(block_size=8))
    runtime.tokenizer = SimpleNamespace(encode=lambda _: [1] * 9)
    with pytest.raises(ValueError, match="context window"):
        runtime.plan(CASES["routing"][0]["user"])
