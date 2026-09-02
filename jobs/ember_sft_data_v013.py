"""Short unique-value copy curriculum for Ember v0.0.13.

v0.0.11 memorized a closed city/tech pool. The v0.0.12 canary then initialized
from that failed checkpoint and used long unique phrases. It copied tool JSON
shape but not the requested values, and direct responses collapsed under 8x
semantic weight.

This dataset keeps one short copy target per example so a 15M model can attend
to the current prompt. Official promotion entities stay held out.
"""
from __future__ import annotations

import hashlib
import json

KINDS = ("tool_call", "direct_response", "tool_result_response")
TOOLS = ("weather", "calculator", "web_search", "get_time")
HELD_OUT_TERMS = (
    "Detroit",
    "Python",
    "Tokyo",
    "website thing not working",
    "validation checks passed",
    "Calculate 347 multiplied by 28",
    "sunny",
    "healthy",
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


def split_offset(split: str) -> int:
    return 200_000 if split == "validation" else 20_000


def code(split: str, i: int, tag: str) -> str:
    value = split_offset(split) + i * 7919 + sum(ord(ch) for ch in tag)
    for _ in range(32):
        token = f"{tag.upper()}-{value % 999983:06d}"
        if not any(term in token for term in HELD_OUT_TERMS):
            return token
        value += 17
    raise RuntimeError(f"unable to allocate a clean identifier for {split} {tag} {i}")


def expression(split: str, i: int) -> tuple[str, str, str, str, str]:
    off = split_offset(split)
    left = 40 + ((i * 41 + off) % 850)
    right = 4 + ((i * 29 + off // 9) % 90)
    if str(left) in FORBIDDEN_NUMBERS:
        left += 5
    if str(right) in FORBIDDEN_NUMBERS:
        right += 7
    op, word = (("*", "multiplied by"), ("+", "plus"), ("-", "minus"))[i % 3]
    return f"{left}{op}{right}", str(left), str(right), word, op


def tool_example(split: str, i: int) -> dict:
    tool = TOOLS[i % 4]
    variant = i // 4
    system = (
        f"You are Ember. Call {tool} when needed. Copy the exact short identifier "
        "from the current user message into the JSON arguments."
    )
    if tool == "weather":
        value = code(split, variant, "wx")
        user = (
            f"What is the weather in {value} right now?"
            if variant % 2 == 0
            else f"Check current conditions for {value}."
        )
        args = {"location": value}
        focus = [value]
    elif tool == "calculator":
        expr, left, right, word, op = expression(split, variant)
        user = (
            f"Calculate {left} {word} {right}."
            if variant % 2 == 0
            else f"Use a calculator for {expr}."
        )
        args = {"expression": expr}
        focus = [expr]
    elif tool == "web_search":
        value = code(split, variant, "rel")
        user = (
            f"Find the latest published release of {value}."
            if variant % 2 == 0
            else f"Search for current information about {value}."
        )
        args = {"query": value}
        focus = [value]
    else:
        value = code(split, variant + 7000, "tz")
        user = (
            f"What time is it in {value}?"
            if variant % 2 == 0
            else f"Check the current time in {value}."
        )
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


def direct_example(split: str, i: int) -> dict:
    subtype = i % 4
    variant = i // 4
    system = (
        "You are Ember. Answer directly without a tool. Copy the exact short "
        "identifier from the current request and stop at endoftext."
    )
    if subtype == 0:
        value = code(split, variant, "echo")
        user = f"Repeat this identifier exactly once: {value}"
        answer = value
        focus = [value]
    elif subtype == 1:
        value = code(split, variant, "ttl")
        user = f"Rewrite this title clearly while keeping its unique identifier: {value} is not working."
        answer = f"{value} Is Not Working"
        focus = [value]
    elif subtype == 2:
        value = code(split, variant, "cmp")
        user = f"Explain this component in one short sentence and include its exact name: {value}."
        answer = f"{value} is a system component used to support reliable operation."
        focus = [value]
    else:
        value = code(split, variant, "demo")
        user = f"Give me two short steps for testing a login form in demo {value}."
        answer = (
            f"1. Submit valid and invalid credentials in demo {value}.\n"
            "2. Verify the expected success and error states."
        )
        focus = [value]
    return {
        "id": f"{split}-direct-{i:05d}",
        "kind": "direct_response",
        "prompt": prompt(system, user),
        "completion": done(answer),
        "focus_terms": focus,
    }


def result_example(split: str, i: int) -> dict:
    subtype = i % 4
    variant = i // 4
    system = (
        "You are Ember. A tool result is already supplied. Copy the exact unique "
        "identifier from that result into the answer. Do not call another tool."
    )
    if subtype == 0:
        loc = code(split, variant + 9000, "wx")
        temp = 31 + ((variant * 17 + split_offset(split)) % 61)
        if str(temp) in FORBIDDEN_NUMBERS:
            temp += 3
        history = (
            "<|assistant|>\n<|tool|>\n"
            + compact({"name": "weather", "arguments": {"location": loc}})
            + "\n<|tool_result|>\n"
            + compact({"temperature_f": temp, "condition": "overcast"})
            + "\n"
        )
        user = f"What is the weather in {loc}?"
        answer = f"It is {temp}°F and overcast in {loc}."
        focus = [loc]
    elif subtype == 1:
        expr, left, right, _word, _op = expression(split, variant + 300)
        result = int(left) * int(right) + (variant % 5)
        if str(result) in FORBIDDEN_NUMBERS:
            result += 11
        history = (
            "<|assistant|>\n<|tool|>\n"
            + compact({"name": "calculator", "arguments": {"expression": expr}})
            + "\n<|tool_result|>\n"
            + compact({"result": result})
            + "\n"
        )
        user = f"Calculate {left} multiplied by {right}."
        answer = f"The supplied result is {result}."
        focus = [str(result)]
    elif subtype == 2:
        value = code(split, variant + 5000, "rel")
        release = code(split, variant, "ver")
        history = (
            "<|assistant|>\n<|tool|>\n"
            + compact({"name": "web_search", "arguments": {"query": value}})
            + "\n<|tool_result|>\n"
            + compact({"title": value, "summary": f"{value} release {release} passed checks."})
            + "\n"
        )
        user = f"Find the current status of {value}."
        answer = f"{value} release {release} passed checks."
        focus = [release]
    else:
        svc = code(split, variant, "svc")
        latency = 21 + ((variant * 43 + split_offset(split)) % 240)
        if str(latency) in FORBIDDEN_NUMBERS:
            latency += 13
        history = (
            "<|assistant|>\n<|tool|>\n"
            + compact({"name": "service_status", "arguments": {"service": svc}})
            + "\n<|tool_result|>\n"
            + compact({"status": "nominal", "latency_ms": latency})
            + "\n"
        )
        user = f"What is the status of service {svc}?"
        answer = f"Service {svc} is nominal with {latency} ms latency."
        focus = [svc]
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
    rows = (
        [tool_example(split, i) for i in range(counts[0])]
        + [direct_example(split, i) for i in range(counts[1])]
        + [result_example(split, i) for i in range(counts[2])]
    )
    for row in rows:
        missing = [str(term) for term in row["focus_terms"] if str(term) not in row["completion"]]
        if missing:
            raise RuntimeError(f"semantic focus absent from target: {row['id']} {missing}")
        if len(row["focus_terms"]) != 1:
            raise RuntimeError(f"v0.0.13 requires exactly one focus term: {row['id']}")
    return rows


def assert_held_out_clean(rows: list[dict], official_prompts: list[str] | None = None) -> None:
    blob = "\n".join(row["prompt"] + row["completion"] for row in rows)
    for term in HELD_OUT_TERMS:
        if term in blob:
            raise RuntimeError(f"held-out term leaked into v0.0.13 data: {term}")
    if official_prompts:
        for official in official_prompts:
            if official in blob:
                raise RuntimeError("official promotion prompt leaked into v0.0.13 data")


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")


def dataset_sha256(rows: list[dict]) -> str:
    return hashlib.sha256(jsonl_bytes(rows)).hexdigest()


def focus_ok(completion: str, focus_terms: list[str]) -> bool:
    low = completion.casefold()
    return all(str(term).casefold() in low for term in focus_terms)
