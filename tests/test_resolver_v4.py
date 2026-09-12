"""Development checks for complete two-clause location requests."""
import json
from pathlib import Path

import pytest

from tool_assistant.evaluate_parser import check_case
from tool_assistant.resolver_v4 import parse_location, resolve_v4
from tool_assistant.runtime_v5 import ParserV4Control

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("tool,user,place", [
    ("weather", "I am stepping outside; what is the current temperature in Nuuk?", "Nuuk"),
    ("weather", "I am packing a jacket; check the current temperature in Ulaanbaatar.", "Ulaanbaatar"),
    ("get_time", "I am about to call someone in Apia; what time is it there now?", "Apia"),
    ("weather", "I'm heading out; could you check the weather in Cuenca, please?", "Cuenca"),
    ("weather", "We are bringing an umbrella; is it raining in Mombasa right now?", "Mombasa"),
    ("weather", "I am about to go outside. What is the temperature in Tromsø?", "Tromsø"),
    ("get_time", "We are phoning a colleague in St. Moritz; what is the local time there?", "St. Moritz"),
    ("get_time", "I am planning to contact my family in Asia/Kathmandu; what time is it there now?", "Asia/Kathmandu"),
    ("get_time", "I am calling someone in Washington, D.C.; what time is it there?", "Washington, D.C."),
    ("get_time", "I am calling someone in Paris. What time is it there now?", "Paris"),
    ("get_time", "I am calling someone in L'Aquila; what time is it there now?", "L'Aquila"),
    ("get_time", 'I am visiting a friend in "Trinidad and Tobago"; is it evening there currently?', "Trinidad and Tobago"),
    ("weather", 'I am visiting someone in "Bosnia and Herzegovina"; what is the weather there now?', "Bosnia and Herzegovina"),
    ("weather", "We are taking our coat; show the weather for São Paulo, Brazil now.", "São Paulo, Brazil"),
])
def test_complete_context_request_has_exactly_one_location(tool, user, place):
    assert parse_location(user, tool) == place
    runtime = ParserV4Control.__new__(ParserV4Control)
    runtime.route = lambda _: (tool, 0.5)
    calls = []
    result = runtime.run(user, {tool: lambda **arguments: calls.append(arguments)})
    assert result["status"] == "tool_result"
    assert calls == [{"location" if tool == "weather" else "timezone": place}]


@pytest.mark.parametrize("tool,user", [
    ("get_time", "I am calling someone in Paris and Rome; what time is it there?"),
    ("get_time", "I am calling someone in Paris; what time is it in Rome?"),
    ("get_time", "I am calling someone in Paris. What time is it in Rome?"),
    ("get_time", "I am calling someone in my office; what time is it there now?"),
    ("get_time", "I am calling someone in Mars/Olympus; what time is it there?"),
    ("get_time", "I am calling someone in Paris; what time is it there tomorrow?"),
    ("get_time", 'I am calling someone in "Paris" and "Rome"; what time is it there?'),
    ("get_time", "I am calling someone in Paris; what time is it there; then call them."),
    ("get_time", "I am calling someone in Paris; what time is it here?"),
    ("weather", "I am heading outside in Paris; what is the weather in Rome?"),
    ("weather", "I am stepping outside; what is the weather in Paris and Rome?"),
    ("weather", "I am stepping outside; don't check the weather in Paris."),
    ("weather", "I am stepping outside; weather in Paris? Send an email."),
    ("weather", "I am stepping outside; check weather in Paris tomorrow."),
    ("weather", "Tomorrow I am stepping outside; check weather in Paris."),
    ("weather", "I am packing a jacket and sending a message; check the weather in Paris."),
    ("weather", "I am stepping outside; explain a time zone."),
    ("weather", "I am stepping outside; check the weather in my city."),
    ("weather", "Delete files. Check the weather in Paris."),
    ("weather", "I am stepping outside; check the weather in Europe/Paris."),
    ("weather", "I am visiting someone in Paris; is it raining there next week?"),
    ("weather", "I am stepping outside;; check the weather in Paris."),
    ("weather", "; check the weather in Paris."),
    ("weather", "I am stepping outside;"),
])
def test_unknown_or_ambiguous_context_does_not_dispatch(tool, user):
    result = resolve_v4({"user": user}, tool)
    assert result["payload"] is None, result
    runtime = ParserV4Control.__new__(ParserV4Control)
    runtime.route = lambda _: (tool, 0.5)
    calls = []
    result = runtime.run(user, {tool: lambda **arguments: calls.append(arguments)})
    assert result["status"] == "needs_clarification"
    assert calls == []


def test_all_consumed_parser_requests_remain_correct_under_v4():
    suite = json.loads((ROOT / "tool_assistant/data/parser-v3-confirmation.json").read_text())
    rows = []
    for case in suite["cases"]:
        runtime = ParserV4Control.__new__(ParserV4Control)
        runtime.route = lambda _, tool=case["tool"]: (tool, 0.0)
        rows.append(check_case(case, runtime))
    assert len(rows) == 60
    assert all(row["passed"] for row in rows)


def test_arithmetic_does_not_ignore_context_clauses():
    result = resolve_v4({"user": "I am stepping outside; calculate 4+5."}, "calculator")
    assert result["payload"] is None
