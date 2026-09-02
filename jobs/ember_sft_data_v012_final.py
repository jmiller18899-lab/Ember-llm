"""Final holdout-safe, target-consistent v0.0.12 curriculum adapter."""
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

_base_direct = ns["direct_example"]


def direct_example(split: str, i: int) -> dict:
    row = _base_direct(split, i)
    if i % 4 == 3:
        variant = i // 4
        value = ns["code"](split, variant, "demo")
        row["completion"] = ns["done"](
            f"1. Submit valid and invalid credentials in the login form for demo {value}.\n"
            "2. Verify the expected success and error states."
        )
        row["focus_terms"] = ["login", value]
    return row


ns["direct_example"] = direct_example
_base_build = ns["build_examples"]


def build_examples(split: str, total: int) -> list[dict]:
    rows = _base_build(split, total)
    for row in rows:
        completion = row["completion"]
        missing = [str(term) for term in row["focus_terms"] if str(term) not in completion]
        if missing:
            raise RuntimeError(f"semantic focus absent from target: {row['id']} {missing}")
    return rows


KINDS = ns["KINDS"]
TOOLS = ns["TOOLS"]
HELD_OUT_TERMS = ns["HELD_OUT_TERMS"]
