"""Deterministic high-diversity copy/grounding curriculum for Ember v0.0.12.

v0.0.11 learned EOS stopping and routing structure but memorized a small value
pool. v0.0.12 makes nearly every semantic value unique so success requires
attending to the current prompt/tool result instead of recalling a training
location, expression, service, or topic.
"""
from __future__ import annotations

import json
import random
import re

KINDS = ("tool_call", "direct_response", "tool_result_response")
TOOLS = ("weather", "calculator", "web_search", "get_time")
HELD_OUT_TERMS = (
    "Detroit", "Python", "Tokyo", "website thing not working",
    "validation checks passed", "9716", "347", "28", "84", "72",
    "sunny", "healthy",
)

PREFIXES = (
    "Cedar","Maple","Silver","North","South","East","West","Lake","River","Pine",
    "Oak","Stone","Bright","Clear","Red","Blue","Green","Grand","New","Port",
    "Fort","Spring","High","Low","Ash","Birch","Copper","Iron","Golden","Quiet",
    "Rapid","Misty","Crystal","Summit","Valley","Harbor","Forest","Prairie","Ridge","Brook",
)
SUFFIXES = (
    "haven","field","ridge","point","view","brook","ford","ton","vale","crest",
    "grove","port","side","hill","bay","falls","park","wood","lake","bridge",
)
TECH_ROOTS = (
    "Atlas","Orchid","Quartz","Nimbus","Vector","Beacon","Helix","Juniper","Falcon","Mosaic",
    "Aster","Comet","Delta","Emberline","Flux","Garnet","Harbor","Ion","Kestrel","Lumen",
    "Matrix","Nova","Onyx","Pixel","Quasar","Rivet","Solace","Tangent","Umbra","Vertex",
)
NOUNS = (
    "cache","schema","webhook","queue","retry","backup","timeout","tokenizer","session","migration",
    "index","archive","worker","gateway","ledger","catalog","upload","preview","report","filter",
    "profile","notification","request","response","artifact","checkpoint","manifest","router","parser","monitor",
)
ADJECTIVES = (
    "alpha","brisk","cobalt","drift","elastic","frozen","granite","hollow","indigo","jade",
    "kinetic","lunar","modular","neon","opal","polar","quiet","rapid","solar","tidal",
    "urban","velvet","wild","xenial","young","zenith",
)
CONDITIONS = ("clear","cloudy","windy","rainy","cool","warm","dry","foggy","breezy","overcast")
STATUSES = ("ready","online","stable","available","responsive","operational","active","nominal")


def compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def prompt(system: str, user: str, history: str = "") -> str:
    return f"<|system|>\n{system}\n<|user|>\n{user}\n{history}<|assistant|>\n"


def done(text: str) -> str:
    return text.rstrip() + "\n<|endoftext|>\n"


def tool_done(name: str, arguments: dict) -> str:
    return done("<|tool|>\n" + compact({"name": name, "arguments": arguments}))


def split_offset(split: str) -> int:
    return 100_000 if split == "validation" else 10_000


def code(split: str, i: int, tag: str) -> str:
    value = split_offset(split) + i * 7919 + sum(ord(ch) for ch in tag)
    return f"{tag.upper()}-{value % 999983:06d}"


def city(split: str, i: int) -> str:
    base = split_offset(split) + i * 37
    p = PREFIXES[base % len(PREFIXES)]
    s = SUFFIXES[(base // len(PREFIXES)) % len(SUFFIXES)]
    return f"{p}{s}-{code(split, i, 'c')[-4:]}"


def topic(split: str, i: int) -> str:
    root = TECH_ROOTS[(split_offset(split) + i * 11) % len(TECH_ROOTS)]
    return f"{root} framework {code(split, i, 'r')}"


def noun_phrase(split: str, i: int) -> str:
    adj = ADJECTIVES[(split_offset(split) + i * 13) % len(ADJECTIVES)]
    noun = NOUNS[(split_offset(split) + i * 17) % len(NOUNS)]
    return f"{adj} {noun} {code(split, i, 'n')}"


def service(split: str, i: int) -> str:
    return f"{ADJECTIVES[(i * 7 + split_offset(split)) % len(ADJECTIVES)]}-{NOUNS[(i * 19) % len(NOUNS)]}-{code(split, i, 's')[-4:]}"


def tool_example(split: str, i: int) -> dict:
    tool = TOOLS[i % 4]
    variant = i // 4
    system = (
        f"You are Ember. Call {tool} when needed. Copy the exact semantic value from the current user "
        "message into the JSON arguments. Never reuse a value from another example."
    )
    if tool == "weather":
        value = city(split, variant)
        user = (
            f"What is the weather in {value} right now?" if variant % 2 == 0
            else f"Check current conditions for {value}."
        )
        args = {"location": value}
        focus = [value]
    elif tool == "calculator":
        off = split_offset(split)
        left = 100 + ((variant * 41 + off) % 880)
        right = 3 + ((variant * 29 + off // 10) % 93)
        if str(left) in {"347", "28"}: left += 5
        if str(right) in {"347", "28"}: right += 7
        op, word = (("*","multiplied by"),("+","plus"),("-","minus"))[variant % 3]
        user = (
            f"Calculate {left} {word} {right}." if variant % 2 == 0
            else f"Use a calculator for {left}{op}{right}."
        )
        args = {"expression": f"{left}{op}{right}"}
        focus = [str(left), str(right)]
    elif tool == "web_search":
        value = topic(split, variant)
        user = (
            f"Find the latest published release of {value}." if variant % 2 == 0
            else f"Search for current information about {value}."
        )
        args = {"query": value}
        focus = [value]
    else:
        value = city(split, variant + 7000)
        user = (
            f"What time is it in {value}?" if variant % 2 == 0
            else f"Check the current time in {value}."
        )
        args = {"timezone": value}
        focus = [value]
    return {
        "id": f"{split}-tool-{i:05d}", "kind": "tool_call",
        "prompt": prompt(system, user), "completion": tool_done(tool, args),
        "expected_tool": tool, "focus_terms": focus,
    }


def direct_example(split: str, i: int) -> dict:
    subtype = i % 4
    variant = i // 4
    system = (
        "You are Ember. Answer directly without a tool. Preserve the exact semantic value supplied in the current "
        "request and stop at endoftext."
    )
    if subtype == 0:
        value = code(split, variant, "echo")
        user = f"Repeat this identifier exactly once: {value}"
        answer = value
        focus = [value]
    elif subtype == 1:
        value = noun_phrase(split, variant)
        user = f"Rewrite this title clearly while keeping its unique identifier: {value} is not working."
        answer = f"{value} Is Not Working"
        focus = [value]
    elif subtype == 2:
        value = noun_phrase(split, variant + 3000)
        user = f"Explain this component in one short sentence and include its exact name: {value}."
        answer = f"{value} is a system component used to support reliable operation."
        focus = [value]
    else:
        value = code(split, variant, "demo")
        user = f"Give me two short steps for testing a login form in demo {value}."
        answer = f"1. Submit valid and invalid credentials in demo {value}.\n2. Verify the expected success and error states."
        focus = ["login", value]
    return {
        "id": f"{split}-direct-{i:05d}", "kind": "direct_response",
        "prompt": prompt(system, user), "completion": done(answer), "focus_terms": focus,
    }


def result_example(split: str, i: int) -> dict:
    subtype = i % 4
    variant = i // 4
    system = (
        "You are Ember. A tool result is already supplied. Copy the exact facts from that result into the answer. "
        "Do not reuse facts from another example, do not call another tool, and stop."
    )
    if subtype == 0:
        loc = city(split, variant + 9000)
        temp = 31 + ((variant * 17 + split_offset(split)) % 61)
        if temp in {72, 84}: temp += 3
        condition = CONDITIONS[(variant * 3 + split_offset(split)) % len(CONDITIONS)]
        if condition == "sunny": condition = "clear"
        user = f"What is the weather in {loc}?"
        history = (
            "<|assistant|>\n<|tool|>\n" + compact({"name":"weather","arguments":{"location":loc}})
            + "\n<|tool_result|>\n" + compact({"temperature_f":temp,"condition":condition}) + "\n"
        )
        answer = f"It is {temp}°F and {condition} in {loc}."
        focus = [str(temp), condition, loc]
    elif subtype == 1:
        off = split_offset(split)
        a = 40 + ((variant * 31 + off) % 500)
        b = 4 + ((variant * 23 + off // 10) % 80)
        result = a * b + (variant % 5)
        if result in {9716, 84, 72}: result += 11
        user = f"Calculate {a} multiplied by {b}."
        history = (
            "<|assistant|>\n<|tool|>\n" + compact({"name":"calculator","arguments":{"expression":f"{a}*{b}"}})
            + "\n<|tool_result|>\n" + compact({"result":result}) + "\n"
        )
        answer = f"The supplied result is {result}."
        focus = [str(result)]
    elif subtype == 2:
        value = topic(split, variant + 5000)
        release = code(split, variant, "rel")
        fact = f"{value} release {release} passed its published verification checks."
        user = f"Find the current status of {value}."
        history = (
            "<|assistant|>\n<|tool|>\n" + compact({"name":"web_search","arguments":{"query":value}})
            + "\n<|tool_result|>\n" + compact({"title":value,"summary":fact}) + "\n"
        )
        answer = fact
        focus = [value, release, "verification checks"]
    else:
        svc = service(split, variant)
        latency = 21 + ((variant * 43 + split_offset(split)) % 240)
        if latency in {84,72}: latency += 13
        status = STATUSES[(variant * 5 + split_offset(split)) % len(STATUSES)]
        if status == "healthy": status = "nominal"
        user = f"What is the status of service {svc}?"
        history = (
            "<|assistant|>\n<|tool|>\n" + compact({"name":"service_status","arguments":{"service":svc}})
            + "\n<|tool_result|>\n" + compact({"status":status,"latency_ms":latency}) + "\n"
        )
        answer = f"Service {svc} is {status} with {latency} ms latency."
        focus = [svc, status, str(latency)]
    return {
        "id": f"{split}-result-{i:05d}", "kind": "tool_result_response",
        "prompt": prompt(system, user, history), "completion": done(answer), "focus_terms": focus,
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
    random.Random(20260912 if split == "train" else 20260913).shuffle(rows)
    blob = "\n".join(r["prompt"] + r["completion"] for r in rows)
    folded = blob.casefold()
    for term in HELD_OUT_TERMS:
        pattern = r"(?<![A-Za-z0-9])" + re.escape(term.casefold()) + r"(?![A-Za-z0-9])"
        if re.search(pattern, folded):
            raise RuntimeError(f"held-out term leaked into {split}: {term}")
    signatures = [tuple(str(v) for v in row["focus_terms"]) for row in rows]
    unique_ratio = len(set(signatures)) / len(signatures)
    if unique_ratio < 0.98:
        raise RuntimeError(f"semantic focus diversity too low: {unique_ratio:.4f}")
    if any(not row.get("focus_terms") for row in rows):
        raise RuntimeError("every v0.0.12 example must identify semantic focus terms")
    return rows
