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
    assert all(mod.temporal_gate(c["prompt"]) for c in mod.TEMPORAL_CASES)


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
    assert not mod.rubric_pass(mod.TEMPORAL_CASES[0], "It was delivered tonight.")
    assert not mod.rubric_pass(mod.TEMPORAL_CASES[1], "The bus arrived this afternoon.")


def test_explicit_answers():
    by = {c["id"]: c for c in mod.TEMPORAL_CASES}
    assert mod.rubric_pass(by["temporal-tracking-explicit"], "It was delivered Tuesday at 2:14 PM.")
    assert not mod.rubric_pass(by["temporal-tracking-explicit"], "It was delivered tonight.")
    assert mod.rubric_pass(by["temporal-log-explicit"], "07:19")
    assert mod.rubric_pass(by["temporal-duration-train"], "6:50 PM")
    assert not mod.rubric_pass(by["temporal-duration-ferry"], "11:30 AM")
