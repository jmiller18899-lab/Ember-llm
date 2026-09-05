"""Diverse literal-copy warmup curriculum for Ember v0.0.15.

Each completion begins immediately with the current target and ends at EOS.
The objective is to suppress generic response openings and make prompt-conditioned
copy tokens competitive at generation start before tool JSON is reintroduced.
"""
from __future__ import annotations

import hashlib

EOT = "<|endoftext|>"
ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
TRAIN_PREFIXES = ALPHABET[:16]
VAL_PREFIXES = ALPHABET[16:]
KINDS = ("short_code", "long_code", "digits", "model_id", "url", "path", "entity", "expression", "mixed")
HELD_OUT_VALUES = {
    "Q7M4", "V9K2-4R7P", "58310429", "openai/gpt-6-astra",
    "https://example.test/a7Q9", "/tmp/ember/Q7M4/result.json",
    "Northfield Zephyr", "53*19+7", "acct_Q7m4-5831",
}


def prompt(value: str, distractor_a: str, distractor_b: str) -> str:
    return (
        "<|system|>\nYou are Ember. Copy TARGET from the current user message exactly. "
        "Do not explain, normalize, calculate, or call a tool. Stop at endoftext.\n"
        f"<|user|>\nIgnore old={distractor_a} and fallback={distractor_b}. TARGET={value}. "
        "Reply with TARGET exactly once.\n<|assistant|>\n"
    )


def completion(value: str) -> str:
    return f"{value}\n{EOT}\n"


def _digest(split: str, kind: str, i: int, salt: str = "") -> bytes:
    return hashlib.sha256(f"ember-v015|{split}|{kind}|{i}|{salt}".encode()).digest()


def _code(split: str, kind: str, i: int, length: int = 4) -> str:
    prefixes = TRAIN_PREFIXES if split == "train" else VAL_PREFIXES
    d = _digest(split, kind, i)
    value = prefixes[d[0] % len(prefixes)] + "".join(ALPHABET[b % len(ALPHABET)] for b in d[1:length])
    return value


def _value(split: str, kind: str, i: int) -> str:
    d = _digest(split, kind, i)
    if kind == "short_code":
        return _code(split, kind, i, 4)
    if kind == "long_code":
        return f"{_code(split, kind, i, 4)}-{_code(split, kind, i + 10000, 4)}"
    if kind == "digits":
        base = int.from_bytes(d[:4], "big") % 90000000 + 10000000
        return str(base)
    if kind == "model_id":
        vendor = ("openai", "anthropic", "meta", "mistral")[d[0] % 4]
        family = _code(split, kind, i, 4).lower()
        return f"{vendor}/ember-{family}-{10 + d[1] % 90}b"
    if kind == "url":
        return f"https://example.test/{_code(split, kind, i, 4)}/{_code(split, kind, i + 1, 4).lower()}"
    if kind == "path":
        return f"/tmp/ember/{_code(split, kind, i, 4)}/result-{_code(split, kind, i + 2, 4).lower()}.json"
    if kind == "entity":
        left = ("Northfield", "Westhaven", "Rivergate", "Stonebridge", "Clearwater", "Pinecrest")[d[0] % 6]
        right = ("Zephyr", "Orion", "Harbor", "Summit", "Vector", "Nimbus")[d[1] % 6]
        return f"{left} {right} {_code(split, kind, i, 4)}"
    if kind == "expression":
        a = 12 + d[0] % 80
        b = 2 + d[1] % 40
        c = 1 + d[2] % 20
        op = ("*", "+", "-")[d[3] % 3]
        return f"{a}{op}{b}+{c}"
    if kind == "mixed":
        return f"acct_{_code(split, kind, i, 4).lower()}-{int.from_bytes(d[4:6], 'big') % 9000 + 1000}"
    raise ValueError(kind)


def build_examples(split: str, total: int) -> list[dict]:
    if split not in {"train", "validation"}:
        raise ValueError(split)
    rows = []
    used = set()
    for i in range(total):
        kind = KINDS[i % len(KINDS)]
        value = _value(split, kind, i)
        distractor_a = _value(split, kind, i + total + 17)
        distractor_b = _value(split, KINDS[(i + 3) % len(KINDS)], i + total + 31)
        if value in HELD_OUT_VALUES or value in used:
            value = _value(split, kind, i + 100000)
        used.add(value)
        row = {
            "id": f"{split}-{kind}-{i:05d}",
            "kind": kind,
            "value": value,
            "prompt": prompt(value, distractor_a, distractor_b),
            "completion": completion(value),
            "focus_terms": [value],
        }
        if not row["completion"].startswith(value):
            raise RuntimeError(f"completion does not start with target: {row['id']}")
        if row["prompt"].count(value) != 1:
            raise RuntimeError(f"target must appear exactly once in prompt: {row['id']}")
        rows.append(row)
    return rows


def assert_clean(train: list[dict], validation: list[dict]) -> None:
    train_values = {row["value"] for row in train}
    val_values = {row["value"] for row in validation}
    if train_values & val_values:
        raise RuntimeError("train/validation value overlap")
    leaked = HELD_OUT_VALUES & (train_values | val_values)
    if leaked:
        raise RuntimeError(f"diagnostic held-out leakage: {sorted(leaked)}")
