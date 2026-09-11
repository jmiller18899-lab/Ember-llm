from types import SimpleNamespace

from tool_assistant.evaluate_parser import check_case, evaluate


def test_evaluation_rejects_arithmetic_with_the_wrong_meaning():
    case = {"id": "oracle", "tool": "calculator", "user": "Compute 4*(2+3).",
            "expected": {"arguments": {"expression": "4*(2+3)"}, "result": 11}}
    # Matching expression text is insufficient when its evaluated value is wrong.
    assert evaluate([case])[0]["passed"] is False


def test_evaluation_rejects_extra_location_text():
    case = {"id": "location", "tool": "weather", "user": "Weather in Quito today?",
            "expected": {"arguments": {"location": "Quito today"}}}
    assert evaluate([case])[0]["passed"] is False


def test_evaluation_checks_that_clarification_really_prevents_dispatch():
    case = {"id": "refusal", "tool": "calculator", "user": "Compute 6!",
            "expected": {"needs_clarification": True}}

    def unsafe_run(user, handlers):
        handlers["calculator"](expression="6")
        return {"status": "needs_clarification", "call": None, "reason": "unclear", "message": "Clarify"}

    assert check_case(case, SimpleNamespace(run=unsafe_run))["passed"] is False


def test_evaluation_counts_both_correct_arguments_and_correct_refusals():
    cases = [
        {"id": "call", "tool": "calculator", "user": "Compute 4*(2+3).",
         "expected": {"arguments": {"expression": "4*(2+3)"}, "result": 20}},
        {"id": "refuse", "tool": "calculator", "user": "Compute 6!",
         "expected": {"needs_clarification": True}},
    ]
    assert all(row["passed"] for row in evaluate(cases))
