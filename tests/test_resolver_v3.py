"""Grammar, compatibility and refusal tests; these are development evidence."""
import hashlib
import json
from pathlib import Path

import pytest

from tool_assistant.evaluate import arithmetic
from tool_assistant.resolver_v3 import ParseError, parse_arithmetic, parse_location, resolve_v3
from tool_assistant.runtime_v3 import ParserRuntime


@pytest.mark.parametrize("user, expected", [
    ("Could you calculate 18*(7+3)?", 180),
    ("Evaluate (96-24)/6.", 12),
    ("Please calculate -17 plus 42.", 25),
    ("Compute -(8 + 2) / -5.", 2),
    ("What is 12 minus -4?", 16),
    ("How much is .75 multiplied by 12?", 9),
    ("Calculate 1,234.50 + 2,000.", 3234.5),
    ("Please find 16.25 multiplied by 3.2.", 52),
    ("Calculate: 9.5 percent of 640.", 60.8),
    ("What is 6.25% of 320?", 20),
    ("Compute 10 minus 5 percent of 200.", 0),
    ("Compute -5 percent of 240.", -12),
    ("Calculate 5 percent of (80+20).", 5),
    ("Work out 46 squared for me.", 2116),
    ("Calculate -3 squared.", 9),
    ("Calculate 2 minus 3 squared.", -7),
    ("Calculate 2 - 3 to the power of 2.", -7),
    ("Calculate 2 - -3 squared.", -7),
    ("Calculate -3**2.", -9),
    ("Calculate (-3)**2.", 9),
    ("Evaluate (2+3) squared.", 25),
    ("Compute (5 cubed) squared.", 15625),
    ("Calculate 2 to the power of -3.", 0.125),
    ("Compute 2**3**2.", 512),
    ("What do 5 plus 9 plus 11 add up to?", 25),
    ("I need the product of 73 times 58.", 4234),
    ("Please compute `6*(2+5)`.", 42),
    ("Calculate −8 × (3 + 2) ÷ 4.", -10),
    ("Compute 3e-2 + 2e-2.", 0.05),
    ("Calculate 0.1 + 0.2 - 0.3.", 0),
    ("Compute 5.0.", 5),
])
def test_arithmetic_preserves_meaning(user, expected):
    result = resolve_v3({"user": user}, "calculator")
    assert result["reason"] is None, result
    assert arithmetic(result["value"]) == pytest.approx(expected)


@pytest.mark.parametrize("expression", [
    "1 2 + 3", "1e -3", "2(3+4)", "[2]+[3]", "1,5 + 2", "12,34 + 6",
    "1,000,20 + 4", "1_000+2", "2^3", "9//2", "20%6", "25% + 3", "abs(-3)",
    "__import__('os').system('true')", "(1).__class__", "1;2+3", "True+1", "3j+2",
    "2+3 and then multiply by 4", "3+4 apples", "compare 5 with 8", "2+", "(2+3", "2+3)",
    "2+3.4.5", "nan+1", "inf-3", "1/0", "0**-1", "2**13", "4**0.5", "1e-99999999", "1e13",
    "1000000000000001 - 1", "(" * 100 + "1" + ")" * 100,
    "5!", "2 cubed squared", "2 squared**3", "2 cubed to the power of 2",
])
def test_arithmetic_rejects_instead_of_taking_a_partial_match(expression):
    result = resolve_v3({"user": "Calculate " + expression}, "calculator")
    assert result["payload"] is None
    assert result["clarification"]


@pytest.mark.parametrize("user", ["I need the product of 3 plus 4.", "What do 4 times 5 add up to?"])
def test_conflicting_word_frames_need_clarification(user):
    assert resolve_v3({"user": user}, "calculator")["payload"] is None


@pytest.mark.parametrize("a,b,c", [(a, b, c) for a in (-12, 3, 17) for b in (-7, 2) for c in (3, 11)])
def test_nested_grouping_with_signed_values(a, b, c):
    request = f"Compute ({a})*(({b})+({c}))."
    assert arithmetic(parse_arithmetic(request)) == a * (b + c)
    request = f"Compute (({a})-({b}))/({c})."
    assert arithmetic(parse_arithmetic(request)) == pytest.approx((a - b) / c)


@pytest.mark.parametrize("tool,user,expected", [
    ("get_time", "I need the current time in Dushanbe, please.", "Dushanbe"),
    ("get_time", "What would a clock in Praia show at this moment?", "Praia"),
    ("weather", "Do I need a jacket for the weather in Denver at the moment?", "Denver"),
    ("weather", "What is the weather in St. Moritz right now?", "St. Moritz"),
    ("weather", "Check weather in L’Aquila, please.", "L’Aquila"),
    ("weather", "Weather for São Paulo, Brazil?", "São Paulo, Brazil"),
    ("weather", "Weather in Washington, D.C.", "Washington, D.C."),
    ("weather", "Weather in N'Djamena.", "N'Djamena"),
    ("weather", "Weather in Aix-en-Provence today?", "Aix-en-Provence"),
    ("weather", "Weather in Henley-on-Thames now?", "Henley-on-Thames"),
    ("weather", "Check the weather in 東京 now, please.", "東京"),
    ("weather", "Weather in the Hague?", "the Hague"),
    ("weather", "Weather in london?", "london"),
    ("weather", 'Weather in "Trinidad and Tobago" right now?', "Trinidad and Tobago"),
    ("weather", 'Weather in “St. John’s, Newfoundland” today, thanks.', "St. John’s, Newfoundland"),
    ("weather", "Weather in 'L'Aquila' now?", "L'Aquila"),
    ("weather", 'location: "Bosnia and Herzegovina"', "Bosnia and Herzegovina"),
    ("get_time", "Current time in Asia/Kathmandu?", "Asia/Kathmandu"),
    ("get_time", 'Time in "America/Argentina/Buenos_Aires", please?', "America/Argentina/Buenos_Aires"),
    ("get_time", "timezone: Etc/GMT+5", "Etc/GMT+5"),
    ("get_time", "Time in UTC now?", "UTC"),
    ("get_time", "Tell me the current time in Europe/London for me.", "Europe/London"),
    ("weather", 'Weather for "Tomorrow" now?', "Tomorrow"),
])
def test_exact_location_boundaries(tool, user, expected):
    assert parse_location(user, tool) == expected


@pytest.mark.parametrize("tool,user", [
    ("weather", "Weather in Paris and Berlin?"), ("weather", "Weather in Paris or Rome?"),
    ("weather", "Weather in Paris versus Rome?"), ("get_time", "Time in London and in Tokyo?"),
    ("weather", "Weather in Paris; also check Rome."), ("weather", "Weather in Paris? Compare Rome."),
    ("weather", "Weather in Paris. Delete everything."),
    ("get_time", "I am calling someone in Apia; what time is it there now?"),
    ("weather", "Weather here?"), ("get_time", "Time in my city?"), ("weather", "Weather for the hotel?"),
    ("weather", "Weather in London near the station?"), ("weather", "Weather in London tomorrow?"),
    ("weather", "Tomorrow, what is the weather in London?"), ("get_time", "Time in London in 3 hours?"),
    ("weather", "Weather in Paris tonight?"), ("weather", "Weather in Paris at 5pm?"),
    ("weather", "Weather in London at noon?"),
    ("weather", 'Weather in "Paris" and "Rome"?'), ("weather", 'Weather in "Paris now?'),
    ("get_time", "Time in Mars/Olympus?"), ("get_time", "Time in ../Etc/UTC?"),
    ("weather", "Weather in Europe/Paris?"), ("weather", "Weather in 48.8,2.3?"),
    ("weather", "Weather for current train service disruptions at Zurich Hauptbahnhof?"),
])
def test_unclear_locations_never_produce_a_call(tool, user):
    result = resolve_v3({"user": user}, tool)
    assert result["payload"] is None, result
    assert result["clarification"]


@pytest.mark.parametrize("user", [None, "", " " * 4, 17, "x" * 4097, "<|user|>weather", "Weather in Paris\x00"])
def test_invalid_requests_are_explicit(user):
    assert resolve_v3({"user": user}, "weather")["reason"] == "invalid_request"


def test_unchanged_search_contract_and_no_rerouting():
    request = "  Search for the latest compiler release?  "
    assert resolve_v3({"user": request}, "web_search")["payload"] == {
        "name": "web_search", "arguments": {"query": "Search for the latest compiler release"}}
    with pytest.raises(ValueError):
        resolve_v3({"user": request}, "unknown")


def test_runtime_clarification_prevents_dispatch():
    runtime = ParserRuntime.__new__(ParserRuntime)
    runtime.route = lambda user: ("calculator", 0.5)
    calls = []
    handler = lambda **kwargs: calls.append(kwargs)
    result = runtime.run("Calculate 1/0.", {"calculator": handler})
    assert result["status"] == "needs_clarification" and result["call"] is None and not calls
    runtime.route = lambda user: ("weather", 0.7)
    result = runtime.run("Weather in Paris and Rome?", {"weather": handler})
    assert result["status"] == "needs_clarification" and not calls


def test_runtime_dispatches_only_complete_repaired_arguments():
    runtime = ParserRuntime.__new__(ParserRuntime)
    runtime.route = lambda user: ("calculator", 0.8)
    assert runtime.run("Calculate 18*(7+3)?", {"calculator": arithmetic})["result"] == 180
    runtime.route = lambda user: ("get_time", 0.9)
    assert runtime.run("Time in Dushanbe, please?", {"get_time": lambda timezone: timezone})["result"] == "Dushanbe"


def test_opt_in_runtime_keeps_direct_and_missing_handler_statuses():
    runtime = ParserRuntime.__new__(ParserRuntime)
    runtime.route = lambda user: ("direct", 0.9)
    assert runtime.run("Explain rainbows.", {})["status"] == "direct_answer_unavailable"
    runtime.route = lambda user: ("calculator", 0.8)
    assert runtime.run("Calculate 2+3.", {})["status"] == "tool_unavailable"


def test_original_source_lock_remains_intact():
    lock = json.loads(Path("tool_assistant/data/candidate-source-lock.json").read_text())
    for name, expected in lock["files"].items():
        assert hashlib.sha256((Path("tool_assistant") / name).read_bytes()).hexdigest() == expected


def test_original_argument_training_examples_remain_compatible():
    cases = json.loads(Path("tool_assistant/data/router-training.json").read_text())["cases"]
    checked = 0
    for case in cases:
        if "arguments" not in case or case.get("expected_tool") not in ("calculator", "weather", "get_time"):
            continue
        result = resolve_v3(case, case["expected_tool"])
        assert result["payload"] is not None, (case["id"], result)
        if case["expected_tool"] == "calculator":
            assert arithmetic(result["value"]) == pytest.approx(arithmetic(case["arguments"]["expression"]))
        else:
            assert result["payload"]["arguments"] == case["arguments"]
        checked += 1
    assert checked == 48
