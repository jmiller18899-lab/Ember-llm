import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("temporal_scoped", Path("jobs/ember_4b_temporal_scoped_rule.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
consolidation = importlib.util.spec_from_file_location("consolidation1", Path("jobs/ember_4b_consolidation1.py"))
cmod = importlib.util.module_from_spec(consolidation)
consolidation.loader.exec_module(cmod)


def test_gate_fires_on_every_temporal_case():
    # Duration-only and tracking-without-ask-anchor rows are negative controls:
    # the scoped gate must stay silent so exact benchmark prompts stay byte-identical.
    mismatches = [
        c["id"]
        for c in mod.TEMPORAL_CASES
        if bool(mod.temporal_gate(c["prompt"])) != bool(c["trigger"])
    ]
    assert mismatches == []
    triggered = [c for c in mod.TEMPORAL_CASES if c["trigger"]]
    assert triggered and all(mod.temporal_gate(c["prompt"]) for c in triggered)


@pytest.mark.parametrize("lane", ["arithmetic", "grounding", "drafting"])
def test_gate_silent_on_benchmark_style_prompts(lane):
    # Includes the duration-based and bare "when did it arrive" time-reasoning rows.
    assert not [r["prompt"] for r in cmod.rows(lane) if mod.temporal_gate(r["prompt"])]


def test_unfired_prompt_keeps_exact_system_prompt():
    p = "A bus left at 8:15 AM and traveled for 50 minutes. When did it arrive?"
    assert mod.system_for(p) == mod.SYSTEM
    assert mod.system_for(mod.TEMPORAL_CASES[0]["prompt"]).endswith(mod.TEMPORAL_RULE)


@pytest.mark.parametrize("text", [
    "I don't know when it was delivered.",
    "I need the tracking information to tell you when it arrived.",
    "There's not enough information to determine that.",
    "The delivery date isn't stated; do you have tracking?",
])
def test_abstentions_pass_without_literal_not_enough(text):
    assert mod.rubric_pass(mod.TEMPORAL_CASES[0], text)


def test_context_time_leak_fails():
    by = {c["id"]: c for c in mod.TEMPORAL_CASES}
    assert not mod.rubric_pass(by["rule-01"], "It was delivered tonight.")
    assert not mod.rubric_pass(by["rule-03"], "The bus arrived this afternoon.")


def test_explicit_answers():
    by = {c["id"]: c for c in mod.TEMPORAL_CASES}
    assert mod.rubric_pass(by["rule-02"], "It was delivered tonight at 7:26 PM.")
    assert not mod.rubric_pass(by["rule-02"], "It was delivered tonight.")
    assert mod.rubric_pass(by["rule-04"], "1:05 PM")
    assert mod.rubric_pass(by["rule-06"], "3:17 PM")
    assert mod.rubric_pass(by["rule-07"], "2:15 PM")
    assert not mod.rubric_pass(by["rule-07"], "11:30 AM")
    assert mod.rubric_pass(by["rule-08"], "10:18 AM")
