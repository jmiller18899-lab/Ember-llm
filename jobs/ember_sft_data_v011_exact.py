"""Exact-value adapter for Ember v0.0.11 curriculum.

Loads the pinned v0.0.11 curriculum and patches direct-title examples so the
supervised target preserves the user's semantic value byte-for-byte instead of
normalizing acronyms such as API -> Api.
"""
from __future__ import annotations

import types
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/12645f8166c7ae583eb6f645f672a4352efb4f5d/jobs/ember_sft_data_v011.py"
source = urllib.request.urlopen(BASE_URL, timeout=30).read().decode("utf-8")
ns: dict = {"__name__": "ember_sft_data_v011_base", "__file__": BASE_URL}
exec(compile(source, BASE_URL, "exec"), ns)

_base_direct = ns["direct_example"]


def direct_example(split: str, i: int) -> dict:
    row = _base_direct(split, i)
    if i % 4 == 1:
        nouns = ns["DIRECT_NOUNS"] if split == "train" else ns["VAL_DIRECT_NOUNS"]
        variant = i // 4
        noun = nouns[variant % len(nouns)]
        row["completion"] = ns["done"](f"{noun} Is Not Working")
        row["focus_terms"] = [noun]
    return row


# build_examples resolves direct_example through its original globals dict, so
# replace that binding before exporting it.
ns["direct_example"] = direct_example
build_examples = ns["build_examples"]
KINDS = ns["KINDS"]
TOOLS = ns["TOOLS"]
HELD_OUT_TERMS = ns["HELD_OUT_TERMS"]
