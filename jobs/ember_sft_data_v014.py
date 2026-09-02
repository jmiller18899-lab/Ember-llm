"""Literal unique-value copy curriculum for Ember v0.0.14.

v0.0.13 finished 160 T4 steps from v0.0.9 and ended failed_internal_smoke.
It learned tool JSON and EOS stopping, then invented nearby identifiers
(WX-200239 -> WX-2009). Three recipe bugs made exact copy fail:

* codes used a shared TAG-NNNNNN pattern whose 6-digit tails share BPE pieces;
* some calculator focus terms were format conversions, not prompt copies;
* 160 steps times batch 8 times accum 2 is about one epoch, so greedy decode
  still sampled the template instead of the current prompt.

This dataset uses one 4-character code (or a short verbatim expression) per
example, repeats that span in the prompt, and keeps completions short.
Official promotion entities stay held out.
"""
from __future__ import annotations

import hashlib
import json

KINDS = ("tool_call", "direct_response", "tool_result_response")
TOOLS = ("weather", "calculator", "web_search", "get_time")
ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
TRAIN_PREFIXES = ALPHABET[:16]
VAL_PREFIXES = ALPHABET[16:]
HELD_OUT_TERMS = (
    "Detroit",
    "Python",
    "Tokyo",
    "website thing not working",
    "validation checks passed",
    "Calculate 347 multiplied by 28",
    "sunny",
    "healthy",
    "WX-200239",
    "Rapidton-7669",
    "Ann Arbor",
    "Seattle",
)
FORBIDDEN_NUMBERS = {"347", "28", "9716", "84", "72"}


def compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def prompt(system: str, user: str, history: str = "") -> str:
    return f"<|system|>\n{system}\n<|user|>\n{user}\n{history}<|assistant|>\n"


def done(text: str) -> str:
    return text.rstrip() + "\n<|endoftext|>\n"


def tool_done(name: str, arguments: dict) -> str:
    return done("<|tool|>\n" + compact({"name": name, "arguments": arguments}))


def _held_out(text: str) -> bool:
    low = text.casefold()
    return any(term.casefold() in low for term in HELD_OUT_TERMS)


def code(split: str, i: int, tag: str, used: set[str] | None = None) -> str:
    if split not in {"train", "validation"}:
        raise ValueError(split)
    prefixes = TRAIN_PREFIXES if split == "train" else VAL_PREFIXES
    digest = hashlib.sha256(f"{split}|{tag}|{i}|ember-v0.0.14-literal-copy".encode("utf-8")).digest()
    for offset in range(256):
        block = hashlib.sha256(digest + offset.to_bytes(2, "big")).digest()
        token = prefixes[block[0] % len(prefixes)] + "".join(
            ALPHABET[byte % len(ALPHABET)] for byte in block[1:4]
        )
        if _held_out(token) or "-" in token:
            continue
        if used is not None and token in used:
            continue
        if used is not None:
            used.add(token)
        return token
    raise RuntimeError(f"unable to allocate a clean identifier for {split} {tag} {i}")


def expression(split: str, i: int, used: set[str] | None = None) -> str:
    digest = hashlib.sha256(f"{split}|expr|{i}|ember-v0.0.14".encode("utf-8")).digest()
    for offset in range(512):
        block = hashlib.sha256(digest + offset.to_bytes(2, "big")).digest()
        left = (12 + (block[0] % 40)) if split == "train" else (52 + (block[0] % 40))
        right = 2 + (block[1] % 28)
        op = ("*", "+", "-")[block[2] % 3]
        if str(left) in FORBIDDEN_NUMBERS or str(right) in FORBIDDEN_NUMBERS:
            continue
        token = f"{left}{op}{right}"
        if _held_out(token):
            continue
        if used is not None and token in used:
            continue
        if used is not None:
            used.add(token)
        return token
    raise RuntimeError(f"unable to allocate a clean expression for {split} {i}")


def mention(value: str) -> str:
    return f"The exact code is {value}."


def tool_example(split: str, i: int, used: set[str]) -> dict:
    tool = TOOLS[i % 4]
    variant = i // 4
    system = (
        "You are Ember. Call the requested tool. Copy the exact code from the "
        "current user message into the JSON arguments and stop at endoftext."
    )
    if tool == "weather":
        value = code(split, variant, "wx", used)
        user = f"{mention(value)} What is the weather in {value} right now?"
        args = {"location": value}
        focus = [value]
    elif tool == "calculator":
        value = expression(split, variant, used)
        user = f"{mention(value)} Use a calculator for {value}."
        args = {"expression": value}
        focus = [value]
    elif tool == "web_search":
        value = code(split, variant, "rel", used)
        user = f"{mention(value)} Search for current information about {value}."
        args = {"query": value}
        focus = [value]
    else:
        value = code(split, variant, "tz", used)
        user = f"{mention(value)} What time is it in {value}?"
        args = {"timezone": value}
        focus = [value]
    return {
        "id": f"{split}-tool-{i:05d}",
        "kind": "tool_call",
        "prompt": prompt(system, user),
        "completion": tool_done(tool, args),
        "expected_tool": tool,
        "focus_terms": focus,
    }


def direct_example(split: str, i: int, used: set[str]) -> dict:
    subtype = i % 4
    variant = i // 4
    system = (
        "You are Ember. Answer directly without a tool. Copy the exact code "
        "from the current request and stop at endoftext."
    )
    if subtype in {0, 1}:
        value = code(split, variant, "echo" if subtype == 0 else "say", used)
        user = f"{mention(value)} Repeat this identifier exactly once: {value}"
        answer = value
        focus = [value]
    elif subtype == 2:
        value = code(split, variant, "ttl", used)
        user = f"{mention(value)} Rewrite this title clearly while keeping {value}."
        answer = f"{value} is not working."
        focus = [value]
    else:
        value = code(split, variant, "cmp", used)
        user = f"{mention(value)} Name this component exactly: {value}."
        answer = f"{value} is the component."
        focus = [value]
    return {
        "id": f"{split}-direct-{i:05d}",
        "kind": "direct_response",
        "prompt": prompt(system, user),
        "completion": done(answer),
        "focus_terms": focus,
    }


def result_example(split: str, i: int, used: set[str]) -> dict:
    subtype = i % 4
    variant = i // 4
    system = (
        "You are Ember. A tool result is already supplied. Copy the exact code "
        "from that result into the answer. Do not call another tool."
    )
    if subtype == 0:
        value = code(split, variant, "wres", used)
        history = (
            "<|assistant|>\n<|tool|>\n"
            + compact({"name": "weather", "arguments": {"location": value}})
            + "\n<|tool_result|>\n"
            + compact({"location": value, "condition": "overcast"})
            + "\n"
        )
        user = f"{mention(value)} What is the weather in {value}?"
        answer = f"Overcast in {value}."
        focus = [value]
    elif subtype == 1:
        value = expression(split, variant + 300, used)
        history = (
            "<|assistant|>\n<|tool|>\n"
            + compact({"name": "calculator", "arguments": {"expression": value}})
            + "\n<|tool_result|>\n"
            + compact({"expression": value, "status": "ok"})
            + "\n"
        )
        user = f"{mention(value)} Calculate {value}."
        answer = f"The supplied expression is {value}."
        focus = [value]
    elif subtype == 2:
        value = code(split, variant, "ver", used)
        history = (
            "<|assistant|>\n<|tool|>\n"
            + compact({"name": "web_search", "arguments": {"query": value}})
            + "\n<|tool_result|>\n"
            + compact({"title": value, "summary": f"{value} passed checks."})
            + "\n"
        )
        user = f"{mention(value)} Find the current status of {value}."
        answer = f"{value} passed checks."
        focus = [value]
    else:
        value = code(split, variant, "svc", used)
        history = (
            "<|assistant|>\n<|tool|>\n"
            + compact({"name": "service_status", "arguments": {"service": value}})
            + "\n<|tool_result|>\n"
            + compact({"service": value, "status": "nominal"})
            + "\n"
        )
        user = f"{mention(value)} What is the status of service {value}?"
        answer = f"Service {value} is nominal."
        focus = [value]
    return {
        "id": f"{split}-result-{i:05d}",
        "kind": "tool_result_response",
        "prompt": prompt(system, user, history),
        "completion": done(answer),
        "focus_terms": focus,
    }


def build_examples(split: str, total: int) -> list[dict]:
    if split not in {"train", "validation"}:
        raise ValueError(split)
    per = total // 3
    counts = (per, per, total - 2 * per)
    used: set[str] = set()
    rows = (
        [tool_example(split, i, used) for i in range(counts[0])]
        + [direct_example(split, i, used) for i in range(counts[1])]
        + [result_example(split, i, used) for i in range(counts[2])]
    )
    seen = set()
    for row in rows:
        missing = [str(term) for term in row["focus_terms"] if str(term) not in row["completion"]]
        if missing:
            raise RuntimeError(f"semantic focus absent from target: {row['id']} {missing}")
        absent = [str(term) for term in row["focus_terms"] if str(term) not in row["prompt"]]
        if absent:
            raise RuntimeError(f"v0.0.14 requires the focus term in the prompt: {row['id']} {absent}")
        if len(row["focus_terms"]) != 1:
            raise RuntimeError(f"v0.0.14 requires exactly one focus term: {row['id']}")
        term = row["focus_terms"][0]
        if term in seen:
            raise RuntimeError(f"duplicate focus term {term}")
        seen.add(term)
        if row["prompt"].count(term) < 2:
            raise RuntimeError(f"v0.0.14 requires the focus term twice in the prompt: {row['id']}")
    return rows


def assert_held_out_clean(rows: list[dict], official_prompts: list[str] | None = None) -> None:
    blob = "\n".join(row["prompt"] + row["completion"] for row in rows)
    for term in HELD_OUT_TERMS:
        if term in blob:
            raise RuntimeError(f"held-out term leaked into v0.0.14 data: {term}")
    if official_prompts:
        for official in official_prompts:
            if official in blob:
                raise RuntimeError("official promotion prompt leaked into v0.0.14 data")


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")


def dataset_sha256(rows: list[dict]) -> str:
    return hashlib.sha256(jsonl_bytes(rows)).hexdigest()


def focus_ok(completion: str, focus_terms: list[str]) -> bool:
    low = completion.casefold()
    return all(str(term).casefold() in low for term in focus_terms)
