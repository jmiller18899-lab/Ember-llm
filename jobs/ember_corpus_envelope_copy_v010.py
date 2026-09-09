#!/usr/bin/env python3
"""Synthetic envelope-copy slice for the Ember v0.1.0 re-pretraining corpus.

Why this slice exists
---------------------
Every copy phase from v0.0.15 to v0.0.31 trained the bare value at completion
position zero. The semantic-v1 verdict on v0.0.31 showed that skill never
reached the JSON argument slot, and v0.0.33 through v0.0.51 showed that a
fine-tune small enough to preserve routing cannot move placement. v0.1.0 stops
grafting: this module generates tool-call documents whose copied value sits
inside ``{"arguments":{...},"name":"..."}`` and mixes them into pretraining from
step 0 as the ``envelope_copy`` category of ``config/corpus_v0.1.0.json``.

Properties the corpus builder relies on
---------------------------------------
* Deterministic: document ``i`` depends only on the config seed and ``i``.
* Stdlib only, no network, no tokenizer, no model.
* Every document ends with ``<|endoftext|>`` and renders tool calls as
  ``<|assistant|>\\n<|tool|>\\n{json}`` with canonical (sorted-key) JSON, the
  shape the frozen semantic-v1 prompts use.
* No target or distractor value equals a frozen semantic-v1 argument, a
  configured exclusion, or a value from the 90-case copy battery in
  ``jobs/ember_sft_data_v026.py``.
* ``audit()`` checks format parity at the envelope level: every tool appears in
  every tool-call shape, and every value kind has a minimum number of documents
  and distinct values.

Nothing here trains, uploads, promotes, or launches GPU work.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from typing import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION = "ember-corpus-envelope-copy-v0.1.0"
EOT = "<|endoftext|>"
ROLE_TOKENS = {role: f"<|{role}|>" for role in ("system", "user", "assistant", "tool", "tool_result")}
ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
DEFAULT_CONFIG = REPO_ROOT / "config" / "corpus_v0.1.0.json"
SEMANTIC_V1 = REPO_ROOT / "config" / "ember_semantic_v1.json"
V026_DATA = REPO_ROOT / "jobs" / "ember_sft_data_v026.py"

TOOL_SHAPES = ("tool_call", "tool_call_choice", "tool_call_multi_arg", "tool_call_with_result", "two_turn")
DIRECT_SHAPES = ("direct_extract", "direct_sentence")
SHAPES = TOOL_SHAPES + DIRECT_SHAPES

CITIES = (
    ("Asheville", "North Carolina", "NC"), ("Eugene", "Oregon", "OR"), ("Duluth", "Minnesota", "MN"),
    ("Tacoma", "Washington", "WA"), ("Santa Fe", "New Mexico", "NM"), ("Albany", "New York", "NY"),
    ("Erie", "Pennsylvania", "PA"), ("Tucson", "Arizona", "AZ"), ("Burlington", "Vermont", "VT"),
    ("Savannah", "Georgia", "GA"), ("Fargo", "North Dakota", "ND"), ("Mobile", "Alabama", "AL"),
    ("Wichita", "Kansas", "KS"), ("Omaha", "Nebraska", "NE"), ("Reno", "Nevada", "NV"),
    ("Spokane", "Washington", "WA"), ("Flagstaff", "Arizona", "AZ"), ("Madison", "Wisconsin", "WI"),
    ("Akron", "Ohio", "OH"), ("Bend", "Oregon", "OR"), ("Chattanooga", "Tennessee", "TN"),
    ("Waco", "Texas", "TX"), ("Roanoke", "Virginia", "VA"), ("Missoula", "Montana", "MT"),
    ("Boise", "Idaho", "ID"), ("Provo", "Utah", "UT"), ("Lansing", "Michigan", "MI"),
    ("Peoria", "Illinois", "IL"), ("Topeka", "Kansas", "KS"), ("Cheyenne", "Wyoming", "WY"),
    ("Augusta", "Maine", "ME"), ("Dover", "Delaware", "DE"), ("Trenton", "New Jersey", "NJ"),
    ("Hartford", "Connecticut", "CT"), ("Columbia", "South Carolina", "SC"), ("Lincoln", "Nebraska", "NE"),
    ("Pierre", "South Dakota", "SD"), ("Helena", "Montana", "MT"), ("Juneau", "Alaska", "AK"),
    ("Hilo", "Hawaii", "HI"), ("Laredo", "Texas", "TX"), ("Macon", "Georgia", "GA"),
    ("Yuma", "Arizona", "AZ"), ("Ogden", "Utah", "UT"), ("Salem", "Oregon", "OR"),
    ("Gary", "Indiana", "IN"), ("Racine", "Wisconsin", "WI"), ("Utica", "New York", "NY"),
)
ZONES = (
    "Europe/Amsterdam", "Europe/Oslo", "Europe/Warsaw", "Europe/Lisbon", "Europe/Prague",
    "Europe/Vienna", "Europe/Zurich", "Europe/Helsinki", "Europe/Dublin", "Europe/Athens",
    "Africa/Nairobi", "Africa/Cairo", "Africa/Lagos", "Africa/Johannesburg", "Africa/Accra",
    "Asia/Bangkok", "Asia/Jakarta", "Asia/Manila", "Asia/Taipei", "Asia/Dubai", "Asia/Kathmandu",
    "Asia/Kolkata", "Asia/Seoul", "Asia/Singapore", "Asia/Tashkent", "Pacific/Fiji", "Pacific/Guam",
    "Pacific/Auckland", "Pacific/Honolulu", "America/Bogota", "America/Lima", "America/Montevideo",
    "America/Costa_Rica", "America/Denver", "America/Halifax", "America/Anchorage", "America/Phoenix",
    "America/Sao_Paulo", "Australia/Perth", "Australia/Adelaide",
)
VENDORS = ("openai", "anthropic", "meta", "mistral", "cohere", "deepmind", "ember", "aleph")
FAMILIES = ("gpt", "claude", "llama", "mixtral", "ember", "sol", "orion", "atlas")
CODENAMES = ("astra", "sol", "nimbus", "vector", "harbor", "summit", "zephyr", "orion", "quartz", "delta")
HOSTS = ("example", "docs", "status", "files", "reports", "build", "cdn", "api", "mirror", "archive")
TLDS = ("test", "example", "invalid", "localhost")
DIRS = ("ember", "runs", "cache", "exports", "logs", "jobs", "staging", "snapshots")
LEAVES = ("result", "output", "payload", "record", "summary", "trace", "manifest", "metrics")
LEFT_NAMES = (
    "Northfield", "Westhaven", "Rivergate", "Stonebridge", "Clearwater", "Pinecrest", "Fairmont",
    "Ashcroft", "Brookline", "Cedarhurst", "Eastmoor", "Glenwood", "Harborview", "Ironvale",
    "Lakeshore", "Marbleton", "Oakridge", "Redstone", "Silverton", "Thornbury",
)
RIGHT_NAMES = (
    "Zephyr", "Orion", "Harbor", "Summit", "Vector", "Nimbus", "Lantern", "Quarry", "Beacon",
    "Cinder", "Draft", "Ember", "Foundry", "Granite", "Meridian", "Torrent",
)
SERVICE_STEMS = ("worker", "indexer", "scheduler", "api", "queue", "cache", "auth", "billing", "search", "mailer")
NAMES = ("Nia", "Omar", "Elena", "Hugo", "Imani", "Luca", "Priya", "Noah", "Amara", "Felix",
         "Sofia", "Idris", "Leila", "Theo", "Zara", "Milo", "Anika", "Jonah", "Nora", "Dante")
CONDITIONS = ("clear", "rainy", "cloudy", "windy", "snowy", "sunny", "foggy", "humid")
ERRORS = ("timeout", "rate_limit", "connection_refused", "unavailable", "not_found")

# Frozen semantic-v1 prompts use this wording; several phrasings per tool keep the
# slice from teaching one rigid request template.
NATURAL_REQUESTS = {
    "weather": ("What is the weather in {v} right now?", "Is it raining in {v} today?", "Give me the current weather for {v}."),
    "calculator": ("Calculate {v}.", "What is {v}?", "Evaluate {v} for me."),
    "web_search": ("Find current information about {v}.", "Search for {v}.", "Look up {v} online."),
    "get_time": ("What time is it in {v}?", "Tell me the current time in the timezone {v}.", "Current time for {v}?"),
    "service_status": ("Is the {v} service healthy?", "Check the status of {v}.", "How is {v} doing right now?"),
    "lookup_ticket": ("Look up ticket {v}.", "Show me ticket {v}.", "What is the state of ticket {v}?"),
    "open_path": ("Open the file at {v}.", "Read {v}.", "Show the contents of {v}."),
    "fetch_url": ("Fetch {v}.", "Download the page at {v}.", "Retrieve {v} for me."),
    "get_model_info": ("Show details for the model {v}.", "Describe {v}.", "What do we know about {v}?"),
    "check_order": ("Check the status of order {v}.", "Where is order {v}?", "Track order {v}."),
}
VALUE_STYLES = (
    "Use {tool}. Set {field} to {value} exactly.",
    "Call {tool} with {field} equal to {value}.",
    "For {tool}, use this exact {field}: {value}.",
    "Use the {tool} tool with {field} {value}. Preserve the value.",
    "The requested {field} is {value}. Pass it unchanged to {tool}.",
    "Send {field} {value} to {tool} without changing it.",
)
DISTRACTOR_STYLES = (
    "Ignore old={d1} and fallback={d2}. TARGET={value}. Call {tool} for TARGET.",
    "Previous values {d1} and {d2} are stale. The current {field} is {value}; call {tool}.",
    "Candidates: {d1}; {d2}; {value}. Only the last one is correct. Use {tool} with it as {field}.",
)
CHOICE_STYLES = (
    "Use {tool} with {field} {value}, not {other}.",
    "Choose {value} for {field}; ignore {other}. Call {tool}.",
    "Call {tool}. The correct {field} is {value}; {other} is a distractor.",
    "For {tool}, pass {value} as {field} instead of {other}.",
    "Use {value} as {field} in {tool}. Do not substitute {other}.",
    "Ignore {other}; the requested {field} for {tool} is {value}.",
)
MULTI_ARG_REQUESTS = {
    "send_message": ("Send the message {text} to {recipient}.", "Message {recipient}: {text}.",
                     "Use send_message. recipient is {recipient} and text is {text}. Copy both exactly."),
    "compare_models": ("Compare {left} with {right}.", "How does {left} differ from {right}?",
                       "Use compare_models with left {left} and right {right}. Keep both identifiers unchanged."),
}
TOOL_PURPOSE = {
    "weather": "current weather is requested", "calculator": "arithmetic is requested",
    "web_search": "current information is requested", "get_time": "the current time in a timezone is requested",
    "service_status": "a service health check is requested", "lookup_ticket": "a ticket is referenced",
    "open_path": "a file path should be read", "fetch_url": "a URL should be retrieved",
    "get_model_info": "a model identifier is referenced", "check_order": "an order number is referenced",
    "send_message": "a message should be delivered", "compare_models": "two models should be compared",
}
DIRECT_LABELS = {
    "short_code": "code", "long_code": "reference", "digits": "order number", "model_id": "model",
    "url": "link", "path": "path", "entity": "site", "expression": "expression", "mixed": "account",
    "city": "city", "timezone": "timezone", "query": "topic", "service": "service",
}
DIRECT_SYSTEM = "You are Ember. Answer from the supplied facts without tools. Finish the answer cleanly."


def compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(*parts) -> bytes:
    return hashlib.sha256("|".join([VERSION, *(str(p) for p in parts)]).encode("utf-8")).digest()


def _code(seed: bytes, length: int, offset: int = 0) -> str:
    return "".join(ALPHABET[seed[(offset + i) % len(seed)] % len(ALPHABET)] for i in range(length))


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def slice_config(cfg: dict) -> dict:
    """Return the envelope_copy source block, whether given the corpus config or the block itself."""
    if "sources" in cfg:
        return cfg["sources"]["envelope_copy"]
    return cfg


class Exclusions:
    """Every frozen held-out value this slice must never emit as a target or distractor."""

    def __init__(self, exact: set[str], substrings: set[str]):
        self.exact = {v.strip().lower() for v in exact if v and v.strip()}
        self.substrings = {v.strip().lower() for v in substrings if v and v.strip()}

    @classmethod
    def from_config(cls, block: dict, *, semantic_path: Path = SEMANTIC_V1, v026_path: Path = V026_DATA) -> "Exclusions":
        held = block.get("held_out_exclusions", {})
        exact = set(held.get("exact_values", []))
        substrings = set(held.get("substrings", []))
        if semantic_path.exists():
            for case in load_json(semantic_path).get("cases", []):
                for variant in case.get("argument_variants", []) or []:
                    exact.update(str(v) for v in variant.values())
        if v026_path.exists():
            exact.update(load_module(v026_path, "ember_sft_data_v026").HELD_OUT_VALUES)
        return cls(exact, substrings)

    def blocks(self, value: str) -> bool:
        low = value.strip().lower()
        if low in self.exact:
            return True
        return any(sub in low for sub in self.substrings)


class ValueStream:
    """Deterministic value generator with format-parity variants and a global used-set."""

    POOL_KINDS = frozenset({"city", "timezone"})

    def __init__(self, seed: int, exclusions: Exclusions):
        self.seed = seed
        self.exclusions = exclusions
        self.used: set[str] = set()
        self.counter: Counter = Counter()

    def render(self, kind: str, variant: int, seed: bytes) -> str:
        n = int.from_bytes(seed[:8], "big")
        if kind == "short_code":
            return _code(seed, 4) if variant == 0 else _code(seed, 5)
        if kind == "long_code":
            return f"{_code(seed, 4)}-{_code(seed, 4, 8)}" if variant == 0 else f"{_code(seed, 3)}-{_code(seed, 5, 8)}"
        if kind == "digits":
            return (str(n % 90000000 + 10000000), str(n % 900000 + 100000), str(n % 9000000000 + 1000000000))[variant % 3]
        if kind == "model_id":
            vendor = VENDORS[seed[0] % len(VENDORS)]
            family = FAMILIES[seed[1] % len(FAMILIES)]
            if variant == 0:
                return f"{vendor}/ember-{_code(seed, 4, 3).lower()}-{10 + seed[1] % 90}b"
            if variant == 1:
                return f"{vendor}/{family}-{2 + seed[2] % 8}-{CODENAMES[seed[3] % len(CODENAMES)]}"
            return f"{vendor}/{family}-{2 + seed[2] % 8}.{seed[3] % 10}-{CODENAMES[seed[4] % len(CODENAMES)]}"
        if kind == "url":
            host = f"{HOSTS[seed[0] % len(HOSTS)]}.{TLDS[seed[1] % len(TLDS)]}"
            if variant == 0:
                return f"https://{host}/{_code(seed, 4)}"
            if variant == 1:
                return f"https://{host}/{_code(seed, 4)}/{_code(seed, 4, 8).lower()}"
            return f"https://{host}/{LEAVES[seed[2] % len(LEAVES)]}-{_code(seed, 5, 4)}"
        if kind == "path":
            directory = DIRS[seed[0] % len(DIRS)]
            leaf = LEAVES[seed[1] % len(LEAVES)]
            if variant == 0:
                return f"/tmp/{directory}/{_code(seed, 4)}/{leaf}.json"
            if variant == 1:
                return f"/var/{directory}/{leaf}-{_code(seed, 4, 5).lower()}.log"
            return f"~/{directory}/{leaf}_{_code(seed, 5, 2)}.txt"
        if kind == "entity":
            left = LEFT_NAMES[seed[0] % len(LEFT_NAMES)]
            right = RIGHT_NAMES[seed[1] % len(RIGHT_NAMES)]
            return f"{left} {right}" if variant == 0 else f"{left} {right} {_code(seed, 4, 6)}"
        if kind == "expression":
            a, b, c = 12 + n % 887, 3 + seed[3] % 61, 2 + seed[4] % 9
            return (f"{a}+{b}", f"{a}-{b}", f"{a}*{b}", f"({a}+{b})*{c}", f"{a * c}/{c}")[variant % 5]
        if kind == "mixed":
            if variant == 0:
                # Structure of the held-out acct_Q7m4-5831: mixed case in the stem.
                stem = _code(seed, 4)
                return f"acct_{stem[0]}{stem[1]}{stem[2].lower()}{stem[3]}-{n % 9000 + 1000}"
            return f"ref-{_code(seed, 3, 2).lower()}{n % 900 + 100}"
        if kind == "city":
            city, state, abbrev = CITIES[seed[0] % len(CITIES)]
            return (city, f"{city}, {abbrev}", f"{city}, {state}")[variant % 3]
        if kind == "timezone":
            return ZONES[seed[0] % len(ZONES)]
        if kind == "query":
            code = _code(seed, 4, 2)
            model = self.render("model_id", seed[5] % 3, seed[8:] + seed[:8])
            entity = f"{LEFT_NAMES[seed[6] % len(LEFT_NAMES)]} {RIGHT_NAMES[seed[7] % len(RIGHT_NAMES)]}"
            return (f"release notes for build {code}", f"ticket {code} recovery status",
                    f"changelog for {model}", f"deployment {code} rollout report",
                    f"documentation for {entity}")[variant % 5]
        if kind == "service":
            return f"{SERVICE_STEMS[seed[0] % len(SERVICE_STEMS)]}-{_code(seed, 4, 1).lower()}"
        raise ValueError(f"unknown value kind: {kind}")

    def variants(self, kind: str) -> int:
        return {"short_code": 2, "long_code": 2, "digits": 3, "model_id": 3, "url": 3, "path": 3, "entity": 2,
                "expression": 5, "mixed": 2, "city": 3, "timezone": 1, "query": 5, "service": 1}[kind]

    def next(self, kind: str, rng: random.Random, *, unique: bool = True) -> str:
        for attempt in range(64):
            index = self.counter[kind]
            self.counter[kind] += 1
            seed = _digest(self.seed, kind, index)
            value = self.render(kind, rng.randrange(self.variants(kind)), seed)
            if self.exclusions.blocks(value):
                continue
            if unique and kind not in self.POOL_KINDS and value in self.used:
                continue
            self.used.add(value)
            return value
        raise RuntimeError(f"could not draw an allowed {kind} value after 64 attempts")


def tool_schema(tool: str, fields: list[str]) -> dict:
    return {"type": "object", "properties": {f: {"type": "string"} for f in fields}, "required": list(fields),
            "additionalProperties": False}


def system_prompt(style: str, tool: str, fields: list[str], tools_cfg: dict, rng: random.Random,
                  *, also: tuple[str, ...] = ()) -> str:
    """Render the system turn. ``also`` lists extra tools a multi-turn document will call later."""
    if also and style != "available_tools":
        raise ValueError("multi-tool documents must use the available_tools system style")
    if style == "sentence":
        if len(fields) == 1:
            variants = (
                f"You are Ember. Use {tool} with JSON arguments. Its only required argument is {fields[0]}, a string.",
                f"You are Ember. When {TOOL_PURPOSE[tool]}, call the {tool} tool with JSON arguments.",
                f"You are Ember. Use {tool} for this request and provide JSON arguments.",
            )
        else:
            variants = (
                f"You are Ember. Use {tool} with JSON arguments. Its required arguments are {fields[0]} and {fields[1]}, both strings.",
                f"You are Ember. When {TOOL_PURPOSE[tool]}, call the {tool} tool with JSON arguments for {fields[0]} and {fields[1]}.",
            )
        return variants[rng.randrange(len(variants))]
    if style == "schema":
        return (f"You are Ember. Use {tool}. Return one tool call with a JSON object containing name and arguments. "
                f"Arguments schema: {compact(tool_schema(tool, fields))}")
    required = [tool, *[t for t in also if t != tool]]
    others = [t for t in tools_cfg if t not in required]
    rng.shuffle(others)
    listed = [*required, *others[: rng.randrange(0, 3)]]
    rng.shuffle(listed)
    specs = [{"name": t, "description": f"Call when {TOOL_PURPOSE[t]}.", "parameters": tool_schema(t, list(tools_cfg[t]))}
             for t in listed]
    return "You are Ember. Choose one tool from AVAILABLE_TOOLS and answer with a single tool call.\nAVAILABLE_TOOLS\n" + compact(specs)


def envelope(tool: str, arguments: dict) -> str:
    return compact({"name": tool, "arguments": arguments})


def render_messages(messages: list[tuple[str, str | None]]) -> str:
    """Render Ember chat text; an assistant turn with None content is a bare tool-call turn."""
    parts = []
    for role, content in messages:
        if role == "assistant" and content is None:
            parts.append(f"{ROLE_TOKENS['assistant']}\n")
        else:
            parts.append(f"{ROLE_TOKENS[role]}\n{content}\n")
    return "".join(parts) + EOT + "\n"


def expression_result(expression: str) -> str:
    """Evaluate the tiny arithmetic grammar this module emits, exactly."""
    expr = expression.replace(" ", "")
    if expr.startswith("("):
        inner, tail = expr[1:].split(")")
        a, b = inner.split("+")
        value = (Fraction(int(a)) + Fraction(int(b))) * Fraction(int(tail[1:]))
    else:
        for op in ("+", "-", "*", "/"):
            if op in expr:
                a, b = expr.split(op)
                fa, fb = Fraction(int(a)), Fraction(int(b))
                value = {"+": fa + fb, "-": fa - fb, "*": fa * fb, "/": fa / fb}[op]
                break
        else:
            raise ValueError(f"unsupported expression: {expression}")
    if value.denominator == 1:
        return str(value.numerator)
    return str(float(value))


def tool_result(tool: str, arguments: dict, rng: random.Random, *, error: bool) -> tuple[dict, str]:
    """Return a tool result payload and the grounded one-sentence answer for it."""
    if error:
        err = rng.choice(ERRORS)
        return {"error": err}, f"{tool} failed: {err}."
    if tool == "weather":
        temp, cond = rng.randrange(-15, 105), rng.choice(CONDITIONS)
        return {"temperature_f": temp, "condition": cond}, f"It is {temp}°F and {cond} in {arguments['location']}."
    if tool == "calculator":
        result = expression_result(arguments["expression"])
        payload = {"result": int(result) if result.lstrip("-").isdigit() else float(result)}
        return payload, f"{arguments['expression']} = {result}."
    if tool == "web_search":
        if rng.random() < 0.2:
            return {"results": []}, "No results found."
        # Never re-case the copied query; the grounded answer must quote it verbatim.
        summary = f"Top result for {arguments['query']}: {rng.choice(('published', 'updated', 'archived'))}."
        return {"results": [{"title": arguments["query"], "summary": summary}]}, summary
    if tool == "get_time":
        clock = f"{rng.randrange(24):02d}:{rng.randrange(60):02d}"
        return {"time": clock, "timezone": arguments["timezone"]}, f"It is {clock} in {arguments['timezone']}."
    if tool == "service_status":
        status, latency = rng.choice(("healthy", "unhealthy", "degraded")), rng.randrange(5, 900)
        return ({"status": status, "latency_ms": latency},
                f"The {arguments['service']} service is {status} with {latency} ms latency.")
    if tool == "lookup_ticket":
        status = rng.choice(("open", "closed", "blocked"))
        if rng.random() < 0.25:
            return {"ticket_id": arguments["ticket_id"], "status": status}, f"Ticket {arguments['ticket_id']} is {status}; the owner is unknown."
        owner = rng.choice(NAMES)
        return ({"ticket_id": arguments["ticket_id"], "status": status, "owner": owner},
                f"Ticket {arguments['ticket_id']} is {status} and owned by {owner}.")
    if tool == "open_path":
        size = rng.randrange(120, 900000)
        return {"path": arguments["path"], "bytes": size}, f"{arguments['path']} is {size} bytes."
    if tool == "fetch_url":
        code = rng.choice((200, 200, 204, 301, 404, 500))
        return {"url": arguments["url"], "status_code": code}, f"{arguments['url']} returned {code}."
    if tool == "get_model_info":
        params, ctx = rng.choice(("1B", "3B", "7B", "13B", "27M", "70B")), rng.choice((2048, 4096, 8192, 32768))
        return ({"model_id": arguments["model_id"], "parameters": params, "context": ctx},
                f"{arguments['model_id']} has {params} parameters and a {ctx}-token context.")
    if tool == "check_order":
        status = rng.choice(("shipped", "pending", "delivered", "cancelled"))
        return {"order_number": arguments["order_number"], "status": status}, f"Order {arguments['order_number']} is {status}."
    raise ValueError(f"no result template for {tool}")


class Generator:
    def __init__(self, cfg: dict, *, exclusions: Exclusions | None = None):
        self.block = slice_config(cfg)
        self.seed = int(self.block.get("seed", cfg.get("seed", 0)))
        self.tools: dict[str, dict[str, list[str]]] = self.block["tools"]
        self.single_tools = [t for t, f in self.tools.items() if len(f) == 1]
        self.multi_tools = [t for t, f in self.tools.items() if len(f) == 2]
        weights = self.block["shape_weights"]
        if abs(sum(weights.values()) - 1.0) > 1e-9:
            raise ValueError("envelope_copy shape_weights must sum to 1.0")
        self.shapes = list(weights)
        self.shape_weights = [float(weights[s]) for s in self.shapes]
        self.styles = list(self.block.get("system_prompt_styles", ["sentence", "schema", "available_tools"]))
        self.distractor_fraction = float(self.block.get("distractor_fraction", 0.5))
        self.exclusions = exclusions or Exclusions.from_config(self.block)
        self.values = ValueStream(self.seed, self.exclusions)

    # -- helpers -----------------------------------------------------------
    def _pick_tool(self, rng: random.Random, pool: list[str]) -> tuple[str, list[str]]:
        tool = pool[rng.randrange(len(pool))]
        return tool, list(self.tools[tool])

    def _draw(self, tool: str, field: str, rng: random.Random) -> tuple[str, str]:
        kind = rng.choice(self.tools[tool][field])
        return kind, self.values.next(kind, rng)

    def _user_single(self, tool: str, field: str, kind: str, value: str, rng: random.Random, distractors: bool) -> str:
        if distractors:
            d1, d2 = self.values.next(kind, rng), self.values.next(kind, rng)
            while d2 == d1 or d1 == value or d2 == value:
                d1, d2 = self.values.next(kind, rng), self.values.next(kind, rng)
            return rng.choice(DISTRACTOR_STYLES).format(tool=tool, field=field, value=value, d1=d1, d2=d2)
        if rng.random() < 0.5:
            return rng.choice(NATURAL_REQUESTS[tool]).format(v=value)
        return rng.choice(VALUE_STYLES).format(tool=tool, field=field, value=value)

    def _tool_turns(self, tool: str, args: dict) -> list[tuple[str, str | None]]:
        return [("assistant", None), ("tool", envelope(tool, args))]

    # -- shapes ------------------------------------------------------------
    def make(self, index: int) -> dict:
        rng = random.Random(int.from_bytes(_digest(self.seed, "doc", index)[:8], "big"))
        shape = rng.choices(self.shapes, weights=self.shape_weights)[0]
        style = rng.choice(self.styles)
        values: list[str] = []
        kinds: list[str] = []
        messages: list[tuple[str, str | None]] = []
        tools_used: list[str] = []

        if shape in ("tool_call", "tool_call_with_result", "two_turn", "tool_call_choice"):
            tool, fields = self._pick_tool(rng, self.single_tools)
            field = fields[0]
            kind, value = self._draw(tool, field, rng)
            values.append(value)
            kinds.append(kind)
            tools_used.append(tool)
            also: tuple[str, ...] = ()
            if shape == "two_turn":
                # Two tools are called, so the system turn must advertise both.
                style = "available_tools"
                tool2, fields2 = self._pick_tool(rng, self.single_tools)
                also = (tool2,)
            messages.append(("system", system_prompt(style, tool, fields, self.tools, rng, also=also)))
            if shape == "tool_call_choice":
                other = self.values.next(kind, rng)
                while other == value:
                    other = self.values.next(kind, rng)
                user = rng.choice(CHOICE_STYLES).format(tool=tool, field=field, value=value, other=other)
            else:
                user = self._user_single(tool, field, kind, value, rng, rng.random() < self.distractor_fraction)
            messages.append(("user", user))
            args = {field: value}
            messages.extend(self._tool_turns(tool, args))
            if shape in ("tool_call_with_result", "two_turn"):
                payload, answer = tool_result(tool, args, rng, error=rng.random() < 0.15)
                messages.append(("tool_result", compact(payload)))
                messages.append(("assistant", answer))
            if shape == "two_turn":
                kind2, value2 = self._draw(tool2, fields2[0], rng)
                values.append(value2)
                kinds.append(kind2)
                tools_used.append(tool2)
                messages.append(("user", self._user_single(tool2, fields2[0], kind2, value2, rng, False)))
                messages.extend(self._tool_turns(tool2, {fields2[0]: value2}))
        elif shape == "tool_call_multi_arg":
            tool, fields = self._pick_tool(rng, self.multi_tools)
            args = {}
            for field in fields:
                kind, value = self._draw(tool, field, rng)
                args[field] = value
                values.append(value)
                kinds.append(kind)
            tools_used.append(tool)
            messages.append(("system", system_prompt(style, tool, fields, self.tools, rng)))
            messages.append(("user", rng.choice(MULTI_ARG_REQUESTS[tool]).format(**args)))
            messages.extend(self._tool_turns(tool, args))
        elif shape in DIRECT_SHAPES:
            kind = rng.choice(list(self.block["value_kinds"]))
            value = self.values.next(kind, rng)
            label = DIRECT_LABELS[kind]
            values.append(value)
            kinds.append(kind)
            messages.append(("system", DIRECT_SYSTEM))
            if rng.random() < 0.5:
                facts = {label.replace(" ", "_"): value}
                if rng.random() < 0.5:
                    facts["owner"] = rng.choice(NAMES)
                user_facts = f"Facts: {compact(facts)}"
            else:
                user_facts = f"The {label} is {value}."
            if shape == "direct_extract":
                messages.append(("user", f"{user_facts}\nReply with only the {label}."))
                messages.append(("assistant", value))
            else:
                messages.append(("user", f"{user_facts}\nWhich {label} was requested? Answer in one sentence."))
                messages.append(("assistant", f"The requested {label} is {value}."))
        else:
            raise ValueError(f"unknown shape: {shape}")

        text = render_messages(messages)
        return {
            "id": f"envelope-copy-{index:07d}",
            "shape": shape,
            "system_style": style if shape in TOOL_SHAPES else "direct",
            "tools": tools_used,
            "kinds": kinds,
            "values": values,
            "text": text,
        }

    def iter_documents(self, count: int) -> Iterator[dict]:
        for index in range(count):
            yield self.make(index)


def iter_documents(cfg: dict, count: int | None = None) -> Iterator[dict]:
    """Corpus-builder entry point: stream deterministic envelope-copy documents."""
    generator = Generator(cfg)
    total = int(count if count is not None else slice_config(cfg)["candidate_documents"])
    return generator.iter_documents(total)


def audit(documents: Iterable[dict], cfg: dict, *, exclusions: Exclusions | None = None) -> dict:
    """Format-parity and leakage audit for a generated slice. No tokenizer or model."""
    block = slice_config(cfg)
    exclusions = exclusions or Exclusions.from_config(block)
    reqs = block["coverage_requirements"]
    tool_shapes = [s for s in block["shapes"] if s in TOOL_SHAPES]
    per_tool: Counter = Counter()
    per_shape: Counter = Counter()
    per_kind: Counter = Counter()
    per_style: Counter = Counter()
    tool_by_shape: dict[str, set[str]] = defaultdict(set)
    distinct: dict[str, set[str]] = defaultdict(set)
    failures: list[str] = []
    chars = 0
    document_count = 0

    for doc in documents:
        document_count += 1
        text = doc["text"]
        chars += len(text)
        per_shape[doc["shape"]] += 1
        per_style[doc["system_style"]] += 1
        for tool in doc["tools"]:
            per_tool[tool] += 1
            tool_by_shape[doc["shape"]].add(tool)
        for kind, value in zip(doc["kinds"], doc["values"]):
            per_kind[kind] += 1
            distinct[kind].add(value)
            if exclusions.blocks(value):
                failures.append(f"{doc['id']}: held-out value emitted: {value!r}")
            if text.count(value) < 2:
                failures.append(f"{doc['id']}: value {value!r} must appear in the prompt and in the completion")
        if "\nAVAILABLE_TOOLS\n" in text:
            advertised_line = text.split("\nAVAILABLE_TOOLS\n", 1)[1].split("\n", 1)[0]
            advertised = {spec["name"] for spec in json.loads(advertised_line)}
            missing = set(doc["tools"]) - advertised
            if missing:
                failures.append(f"{doc['id']}: called tools {sorted(missing)} are not in AVAILABLE_TOOLS")
        if not text.endswith(EOT + "\n"):
            failures.append(f"{doc['id']}: missing terminal {EOT}")
        if text.count(EOT) != 1:
            failures.append(f"{doc['id']}: exactly one {EOT} expected")
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line == ROLE_TOKENS["tool"]:
                if i == 0 or lines[i - 1] != ROLE_TOKENS["assistant"]:
                    failures.append(f"{doc['id']}: <|tool|> must directly follow a bare <|assistant|> line")
                payload = lines[i + 1] if i + 1 < len(lines) else ""
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError:
                    failures.append(f"{doc['id']}: tool envelope is not JSON: {payload!r}")
                    continue
                if compact(parsed) != payload:
                    failures.append(f"{doc['id']}: tool envelope is not canonical sorted-key JSON")
                if set(parsed) != {"arguments", "name"} or parsed["name"] not in block["tools"]:
                    failures.append(f"{doc['id']}: tool envelope keys/name invalid: {payload!r}")
                elif set(parsed["arguments"]) != set(block["tools"][parsed["name"]]):
                    failures.append(f"{doc['id']}: argument keys do not match the tool schema: {payload!r}")

    for tool in block["tools"]:
        if per_tool[tool] < int(reqs["min_documents_per_tool"]):
            failures.append(f"tool {tool} has {per_tool[tool]} documents (< {reqs['min_documents_per_tool']})")
    for shape in block["shapes"]:
        if per_shape[shape] < int(reqs["min_documents_per_shape"]):
            failures.append(f"shape {shape} has {per_shape[shape]} documents (< {reqs['min_documents_per_shape']})")
    for kind in block["value_kinds"]:
        if per_kind[kind] < int(reqs["min_documents_per_kind"]):
            failures.append(f"kind {kind} has {per_kind[kind]} documents (< {reqs['min_documents_per_kind']})")
        floor = int(reqs["min_distinct_values_per_kind"])
        pooled = kind in ValueStream.POOL_KINDS
        if not pooled and len(distinct[kind]) < floor:
            failures.append(f"kind {kind} has {len(distinct[kind])} distinct values (< {floor})")
    if reqs.get("every_tool_in_every_tool_shape"):
        single = {t for t, f in block["tools"].items() if len(f) == 1}
        multi = {t for t, f in block["tools"].items() if len(f) == 2}
        for shape in tool_shapes:
            expected = multi if shape == "tool_call_multi_arg" else single
            missing = expected - tool_by_shape[shape]
            if missing:
                failures.append(f"shape {shape} never uses tools {sorted(missing)}")

    approx_tokens = int(chars / 4.8)
    return {
        "version": VERSION,
        "status": "PASS" if not failures else "FAIL",
        "documents": document_count,
        "characters": chars,
        "approx_tokens_at_4.8_chars_per_token": approx_tokens,
        "approx_tokens_per_document": round(approx_tokens / max(1, document_count), 1),
        "documents_per_tool": dict(sorted(per_tool.items())),
        "documents_per_shape": dict(sorted(per_shape.items())),
        "documents_per_kind": dict(sorted(per_kind.items())),
        "documents_per_system_style": dict(sorted(per_style.items())),
        "distinct_values_per_kind": {k: len(v) for k, v in sorted(distinct.items())},
        "tools_by_shape": {s: sorted(t) for s, t in sorted(tool_by_shape.items())},
        "exclusion_exact_values": len(exclusions.exact),
        "exclusion_substrings": sorted(exclusions.substrings),
        "held_out_leakage": any("held-out" in f for f in failures),
        "failures": failures[:50],
        "failure_count": len(failures),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate and audit the Ember v0.1.0 envelope-copy corpus slice")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--count", type=int, default=None, help="documents to generate (default: candidate_documents)")
    ap.add_argument("--out", default=None, help="optional JSONL path for the generated documents")
    ap.add_argument("--audit-out", default=None, help="optional path for the audit report")
    ap.add_argument("--sample", type=int, default=0, help="print the first N documents")
    ap.add_argument("--assert-coverage", action="store_true", help="exit non-zero when the audit fails")
    args = ap.parse_args(argv)

    cfg = load_json(Path(args.config))
    documents = list(iter_documents(cfg, args.count))
    report = audit(documents, cfg)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for doc in documents:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
    if args.audit_out:
        Path(args.audit_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.audit_out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for doc in documents[: args.sample]:
        print(f"--- {doc['id']} [{doc['shape']} / {doc['system_style']}] tools={doc['tools']} kinds={doc['kinds']}")
        print(doc["text"], end="")
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2, ensure_ascii=False))
    if report["failures"]:
        print(json.dumps({"failures": report["failures"]}, indent=2, ensure_ascii=False))
    if args.assert_coverage and report["status"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
