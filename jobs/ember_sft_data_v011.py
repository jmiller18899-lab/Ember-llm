"""Ember v0.0.11 semantic copy/grounding curriculum.

The v0.0.10 run learned tool selection but substituted memorized values. This
dataset makes dynamic prompt/result values the dominant supervision signal.
Official v0.0.10 promotion prompts and their distinctive values stay held out.
"""
from __future__ import annotations
import json
import random
import re

KINDS = ("tool_call", "direct_response", "tool_result_response")
TOOLS = ("weather", "calculator", "web_search", "get_time")

HELD_OUT_TERMS = (
    "Detroit", "Python", "Tokyo",
    "website thing not working", "validation checks passed",
)
SPECIAL = "<|endoftext|>"

TRAIN_CITIES = (
    "Ann Arbor","Grand Rapids","Flint","Lansing","Chicago","Boston","Seattle",
    "Miami","Denver","Phoenix","Portland","Kansas City","Omaha","Tampa",
    "Pittsburgh","Buffalo","Madison","Spokane","Tucson","Orlando","Sacramento",
    "Atlanta","Dallas","Minneapolis","Cleveland","Boise","Raleigh","Tulsa",
    "Richmond","Milwaukee","Nashville","Toledo","Baltimore","Charlotte",
)
VAL_CITIES = (
    "Akron","Fresno","Reno","Boulder","Savannah","Knoxville","Albany","Eugene",
    "Wichita","Mobile","Bend","Dayton",
)
TRAIN_TECH = (
    "Rust","Go","Ruby","Node.js","TypeScript","PostgreSQL","Kubernetes","Fedora",
    "Debian","LLVM","WebAssembly","SQLite","Linux","Firefox","Chromium","Deno",
    "Swift","Zig","Elixir","Erlang","CMake","Docker","OpenSSL","Redis",
)
VAL_TECH = ("MariaDB","FreeBSD","Bun","Gradle","Nginx","Caddy","Jenkins","Meson")

DIRECT_NOUNS = (
    "cache","schema","webhook","queue","retry","backup","timeout",
    "tokenizer","database index","session","migration","health check","rate limit",
    "log file","API response","deployment","unit test","access token","worker",
)
VAL_DIRECT_NOUNS = (
    "latency","checksum","mutex","release artifact","job queue","audit log",
    "config file","rollback",
)
SERVICES = (
    "gateway","scheduler","database","worker","dashboard","queue","mailer",
    "indexer","proxy","registry","renderer","notifier","billing","cache","auth",
)
VAL_SERVICES = ("catalog","ledger","search","ingest","reports","uploads")

WEATHER_ADJ = ("clear","cloudy","windy","rainy","cool","warm","dry","foggy")
STATUS_WORDS = ("ready","online","stable","available","responsive","operational")


def compact(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def prompt(system, user, history=""):
    return f"<|system|>\n{system}\n<|user|>\n{user}\n{history}<|assistant|>\n"


def done(text):
    return text.rstrip() + "\n<|endoftext|>\n"


def tool_done(name, arguments):
    return done("<|tool|>\n" + compact({"name": name, "arguments": arguments}))


def _pool(split, train, val):
    return train if split == "train" else val


def tool_example(split, i):
    cities = _pool(split, TRAIN_CITIES, VAL_CITIES)
    techs = _pool(split, TRAIN_TECH, VAL_TECH)
    tool = TOOLS[i % 4]
    variant = i // 4
    if tool == "weather":
        city = cities[variant % len(cities)]
        forms = (
            "What is the weather in {x} right now?",
            "Check current conditions for {x}.",
            "Use weather for {x}.",
            "Tell me today's weather in {x}.",
            "Is it raining in {x} now?",
        )
        user = forms[variant % len(forms)].format(x=city)
        args = {"location": city}
        focus = [city]
    elif tool == "calculator":
        left = 101 + ((variant * 37 + (0 if split == "train" else 17)) % 790)
        right = 11 + ((variant * 23 + (0 if split == "train" else 31)) % 80)
        if left == 347: left += 1
        if right == 28: right += 1
        op, word = (("*","multiplied by"),("+","plus"),("-","minus"))[variant % 3]
        forms = (
            "Calculate {a} {word} {b}.",
            "Use a calculator for {a} {op} {b}.",
            "Evaluate the expression {a}{op}{b}.",
        )
        user = forms[variant % len(forms)].format(a=left,b=right,op=op,word=word)
        args = {"expression": f"{left}{op}{right}"}
        focus = [str(left), str(right)]
    elif tool == "web_search":
        tech = techs[variant % len(techs)]
        forms = (
            "Find the latest published release of {x}.",
            "Search for the newest stable {x} release.",
            "Look up current release information for {x}.",
            "Find recent official news about {x}.",
        )
        user = forms[variant % len(forms)].format(x=tech)
        args = {"query": f"latest published release of {tech}"}
        focus = [tech]
    else:
        city = cities[(variant * 3 + 1) % len(cities)]
        forms = (
            "What time is it in {x}?",
            "Check the current time in {x}.",
            "Use get_time for {x}.",
            "Tell me {x}'s local time.",
        )
        user = forms[variant % len(forms)].format(x=city)
        args = {"timezone": city}
        focus = [city]
    system = (
        f"You are Ember. Call {tool} when needed. Copy the user's requested "
        "entity or numeric values exactly into the JSON arguments; never substitute."
    )
    return {
        "id": f"{split}-tool-{i:04d}", "kind": "tool_call",
        "prompt": prompt(system, user), "completion": tool_done(tool, args),
        "expected_tool": tool, "focus_terms": focus,
    }


def direct_example(split, i):
    nouns = _pool(split, DIRECT_NOUNS, VAL_DIRECT_NOUNS)
    cities = _pool(split, TRAIN_CITIES, VAL_CITIES)
    subtype = i % 4
    variant = i // 4
    if subtype == 0:
        names = ("Maya","Jordan","Sam","Avery","Riley","Morgan","Casey","Quinn")
        name = names[variant % len(names)]
        user = f"Say hello to {name} in one friendly sentence."
        answer = f"Hello, {name}! I hope you're having a great day."
        focus = ["Hello", name]
    elif subtype == 1:
        noun = nouns[variant % len(nouns)]
        user = f"Rewrite this title clearly: {noun} is not working."
        answer = f"{noun.title()} Is Not Working"
        focus = [noun.split()[0]]
    elif subtype == 2:
        noun = nouns[variant % len(nouns)]
        user = f"Explain what a {noun} is in one short sentence."
        answer = f"A {noun} is a system component or record used to support reliable operation."
        focus = [noun]
    else:
        city = cities[variant % len(cities)]
        user = f"Give me two short steps for testing a login form for the {city} demo."
        answer = (
            f"1. Submit valid and invalid credentials in the {city} demo login form.\n"
            "2. Verify the expected success and error states."
        )
        focus = ["login", city]
    return {
        "id": f"{split}-direct-{i:04d}", "kind": "direct_response",
        "prompt": prompt(
            "You are Ember. Answer directly when no external tool is required. "
            "Preserve the important words from the user's request and stop after the answer.",
            user,
        ),
        "completion": done(answer), "focus_terms": focus,
    }


def result_example(split, i):
    services = _pool(split, SERVICES, VAL_SERVICES)
    cities = _pool(split, TRAIN_CITIES, VAL_CITIES)
    subtype = i % 4
    variant = i // 4
    if subtype == 0:
        city = cities[variant % len(cities)]
        temp = 35 + ((variant * 7 + (1 if split == "train" else 3)) % 55)
        condition = WEATHER_ADJ[variant % len(WEATHER_ADJ)]
        history = (
            "<|assistant|>\n<|tool|>\n"
            + compact({"name":"weather","arguments":{"location":city}})
            + "\n<|tool_result|>\n"
            + compact({"temperature_f":temp,"condition":condition})
            + "\n"
        )
        user = f"What is the weather in {city}?"
        answer = f"It is {temp}°F and {condition} in {city}."
        focus = [str(temp), condition, city]
    elif subtype == 1:
        a = 43 + ((variant * 29) % 400)
        b = 6 + ((variant * 17) % 70)
        result = a * b
        if str(result) == "9716": result += 1
        history = (
            "<|assistant|>\n<|tool|>\n"
            + compact({"name":"calculator","arguments":{"expression":f"{a}*{b}"}})
            + "\n<|tool_result|>\n"
            + compact({"result":result})
            + "\n"
        )
        user = f"Calculate {a} multiplied by {b}."
        answer = f"The result is {result}."
        focus = [str(result)]
    elif subtype == 2:
        techs = _pool(split, TRAIN_TECH, VAL_TECH)
        tech = techs[variant % len(techs)]
        version = f"{2 + variant % 8}.{1 + (variant*3)%20}.{(variant*7)%10}"
        summary = f"{tech} release {version} is available and its release checks passed."
        history = (
            "<|assistant|>\n<|tool|>\n"
            + compact({"name":"web_search","arguments":{"query":f"{tech} release"}})
            + "\n<|tool_result|>\n"
            + compact({"title":f"{tech} release","summary":summary})
            + "\n"
        )
        user = f"Find the current {tech} release status."
        answer = summary
        focus = [tech, version, "checks passed"]
    else:
        service = services[variant % len(services)]
        latency = 31 + ((variant * 19 + (0 if split == "train" else 7)) % 180)
        status = STATUS_WORDS[variant % len(STATUS_WORDS)]
        history = (
            "<|assistant|>\n<|tool|>\n"
            + compact({"name":"service_status","arguments":{"service":service}})
            + "\n<|tool_result|>\n"
            + compact({"status":status,"latency_ms":latency})
            + "\n"
        )
        user = f"What is the status of the {service} service?"
        answer = f"The {service} service is {status} with {latency} ms latency."
        focus = [service, status, str(latency)]
    return {
        "id": f"{split}-result-{i:04d}", "kind": "tool_result_response",
        "prompt": prompt(
            "You are Ember. A tool result is already supplied. Answer only from its "
            "facts, preserve its values exactly, do not call another tool, and stop.",
            user, history,
        ),
        "completion": done(answer), "focus_terms": focus,
    }


def build_examples(split: str, total: int):
    if split not in {"train","validation"}:
        raise ValueError(split)
    per = total // 3
    counts = [per, per, total - 2*per]
    rows = (
        [tool_example(split, i) for i in range(counts[0])]
        + [direct_example(split, i) for i in range(counts[1])]
        + [result_example(split, i) for i in range(counts[2])]
    )
    random.Random(20260902 if split == "train" else 20260903).shuffle(rows)
    blob = "\n".join(r["prompt"] + r["completion"] for r in rows)
    folded = blob.casefold()
    for term in HELD_OUT_TERMS:
        pattern = r"(?<![A-Za-z0-9])" + re.escape(term.casefold()) + r"(?![A-Za-z0-9])"
        if re.search(pattern, folded):
            raise RuntimeError(f"held-out term leaked into {split}: {term}")
    if any(not row.get("focus_terms") for row in rows):
        raise RuntimeError("every v0.0.11 example must identify semantic focus terms")
    return rows
