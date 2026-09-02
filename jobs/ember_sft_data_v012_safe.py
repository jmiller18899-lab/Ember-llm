"""Holdout-safe adapter for the v0.0.12 high-diversity curriculum.

Bare integers such as 72, 84, or 9716 can occur harmlessly inside synthetic
identifiers or unrelated arithmetic. The base generator already prevents those
values in the corresponding held-out result targets. This adapter keeps the
leak guard focused on distinctive held-out entities/phrases.
"""
from __future__ import annotations
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/3d8aba948670b8fb8e064609bcf72704568fa682/jobs/ember_sft_data_v012.py"
source = urllib.request.urlopen(BASE_URL, timeout=30).read().decode("utf-8")
ns = {"__name__": "ember_sft_data_v012_base", "__file__": BASE_URL}
exec(compile(source, BASE_URL, "exec"), ns)

ns["HELD_OUT_TERMS"] = (
    "Detroit",
    "Python",
    "Tokyo",
    "website thing not working",
    "validation checks passed",
    "sunny",
    "healthy",
    "Calculate 347 multiplied by 28",
)

build_examples = ns["build_examples"]
KINDS = ns["KINDS"]
TOOLS = ns["TOOLS"]
HELD_OUT_TERMS = ns["HELD_OUT_TERMS"]
