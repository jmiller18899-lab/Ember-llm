"""Deterministic semantic-fidelity instruction data for Ember v0.0.10.

v0.0.9 taught Ember when to call weather, calculator, web_search, and get_time.
This curriculum keeps those routing decisions and adds three missing skills:

* copy the requested tool-argument values from the user prompt;
* answer from the supplied tool-result facts instead of inventing new ones; and
* stop at <|endoftext|> without starting another conversation.

The official promotion prompts remain held out.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable

KINDS = ("tool_call", "direct_response", "tool_result_response")
TOOLS = ("weather", "calculator", "web_search", "get_time")
HELD_OUT_FRAGMENTS = (
    "What is the weather in Detroit right now?",
    "Calculate 347 multiplied by 28.",
    "Find the latest published release of Python.",
    "What time is it in Tokyo?",
    "website thing not working",
    "Say hello in a friendly sentence.",
    "Explain what a checkpoint is in one sentence.",
    "Give me two short steps for testing a login form.",
    "Is the service healthy?",
    '"location":"Detroit"',
    '"result":9716',
    '"latency_ms":84',
)


TRAIN_VALUES = {
    "locations": [
        "Ann Arbor", "Grand Rapids", "Flint", "Lansing", "Chicago", "Boston",
        "Seattle", "Miami", "Denver", "Phoenix", "Portland", "Kansas City",
        "Omaha", "Tampa", "Pittsburgh", "Buffalo", "Madison", "Spokane",
        "Tucson", "Orlando", "Sacramento", "Atlanta", "Dallas", "Minneapolis",
    ],
    "timezones": [
        ("London", "Europe/London"), ("Sydney", "Australia/Sydney"),
        ("Berlin", "Europe/Berlin"), ("Honolulu", "Pacific/Honolulu"),
        ("Seoul", "Asia/Seoul"), ("Dublin", "Europe/Dublin"),
        ("Singapore", "Asia/Singapore"), ("Cairo", "Africa/Cairo"),
        ("Toronto", "America/Toronto"), ("Lisbon", "Europe/Lisbon"),
        ("Helsinki", "Europe/Helsinki"), ("Jakarta", "Asia/Jakarta"),
    ],
    "services": [
        "api", "scheduler", "database", "worker", "dashboard", "queue",
        "gateway", "mailer", "indexer", "proxy", "registry", "renderer",
    ],
}

VALIDATION_VALUES = {
    "locations": [
        "Toledo", "Cleveland", "Milwaukee", "Nashville", "Raleigh", "Boise",
        "Richmond", "Tulsa", "Des Moines", "Anchorage", "Austin", "Honolulu",
    ],
    "timezones": [
        ("Paris", "Europe/Paris"), ("Auckland", "Pacific/Auckland"),
        ("Kolkata", "Asia/Kolkata"), ("Vancouver", "America/Vancouver"),
        ("Rome", "Europe/Rome"), ("Reykjavik", "Atlantic/Reykjavik"),
        ("Chicago", "America/Chicago"), ("Lagos", "Africa/Lagos"),
    ],
    "services": [
        "search", "notifier", "billing", "cache", "ingest", "auth",
        "catalog", "ledger",
    ],
}

WEATHER_PHRASES = (
    "Check the current weather in {value}.",
    "What are conditions like in {value} today?",
    "Should I carry an umbrella in {value}?",
    "Look up the temperature for {value}.",
    "Is it warm outside in {value} right now?",
    "Give me today's forecast for {value}.",
    "Please check whether it is raining in {value}.",
    "Find the present weather conditions for {value}.",
)

SEARCH_PHRASES = (
    "Search the web for {value}.",
    "Find current information about {value}.",
    "Look online for {value}.",
    "Please research {value} on the web.",
    "Retrieve a current source about {value}.",
    "Check recent web results for {value}.",
    "Use web search to investigate {value}.",
    "Find a reliable recent page about {value}.",
)

TIME_PHRASES = (
    "What is the current time in {city}?",
    "Check the time for the {timezone} timezone.",
    "Tell me the local time in {city}.",
    "What time is it now in {city}?",
    "Look up the clock time for {timezone}.",
    "Give me the present time in {city}.",
    "Check {city}'s local time.",
    "Find the current hour in {city}.",
    "Use the time tool for {timezone}.",
    "What does the clock say in {city}?",
    "Retrieve the current time for {timezone}.",
    "Please check the time in {city}.",
)

SEARCH_TOPICS = (
    "the newest stable Rust release", "today's NASA mission update",
    "the current Fedora release", "recent WebAssembly announcements",
    "this week's Linux kernel news", "current browser compatibility data",
    "recent renewable energy headlines", "the newest TypeScript release",
    "today's space weather bulletin", "the current Kubernetes release",
    "recent robotics research news", "the latest LLVM release notes",
    "today's IETF standards update", "the newest Node.js LTS release",
    "current OpenStreetMap editing news", "recent RISC-V hardware announcements",
)

VALIDATION_SEARCH_TOPICS = (
    "the newest stable Go release", "today's NOAA climate update",
    "the current Debian release", "recent accessibility standard updates",
    "this week's database security news", "the latest PostgreSQL release notes",
    "current CSS specification news", "the newest Ruby release",
)

DIRECT_TOPICS = (
    "cache", "checksum", "retry", "queue", "tokenizer",
    "webhook", "database index", "unit test", "backup", "rate limit", "timeout",
    "mutex",
)

VALIDATION_DIRECT_TOPICS = (
    "latency", "schema", "session", "encryption", "migration", "health check",
)

PLAN_TASKS = (
    "test a contact form", "verify a backup", "check a broken link",
    "review an error log", "test a password reset", "validate a CSV import",
    "check an API response", "test a search box", "review a deployment",
    "verify a notification", "test file upload", "check a health endpoint",
)

VALIDATION_PLAN_TASKS = (
    "test a logout flow", "verify a scheduled report", "check an image preview",
    "review a failed request", "test account recovery", "validate a JSON export",
)

VALIDATION_NAMES = ("Casey", "Taylor", "Quinn", "Jamie", "Drew", "Reese")
VALIDATION_ROUGH_TITLES = (
    "profile settings fail to save", "billing summary wording confusing",
    "preview image stays blank", "activity feed looks delayed",
    "reminder notification repeats", "filter results are incomplete",
)

GREETING_CONTEXTS = (
    "friendly", "warm", "cheerful", "brief", "welcoming", "casual", "upbeat", "polite",
    "morning", "afternoon", "evening", "professional", "neighborly", "helpful", "simple", "kind",
)

REWRITE_CONTEXTS = (
    "for a bug report", "for a status page", "for a support ticket", "for a task list",
    "for release notes", "for a dashboard", "for an incident log", "for a test case",
    "for a project board", "for an email subject", "for a changelog", "for a help article",
    "for a pull request", "for a checklist", "for a report", "for an alert",
)

EXPLAIN_STYLES = (
    "in one plain sentence", "for a beginner", "in fewer than twenty words", "without jargon",
    "with a practical definition", "in a concise sentence", "using simple language", "briefly",
)

PLAN_CONTEXTS = (
    "before release", "in a staging environment", "with a reproducible example", "without changing production",
    "and record the result", "using a test account", "with one expected outcome", "as a quick smoke test",
)


def compact_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _system(kind: str, variant: int, tool: str | None = None) -> str:
    if kind == "tool_call":
        choices = (
            f"You are Ember. Use the {tool} tool for this request and copy the requested values into JSON arguments.",
            f"You are Ember. Call {tool} with the exact entity or expression from the user message.",
            f"You are Ember. Do not substitute a different location, number, query, or timezone when calling {tool}.",
        )
        return choices[variant % len(choices)]
    if kind == "direct_response":
        choices = (
            "You are Ember. Answer directly when no external tool is needed and stop after the answer.",
            "You are Ember. Do not call a tool for writing, explanation, or planning requests.",
            "You are Ember. Give a concise direct answer unless current external data is required.",
        )
        return choices[variant % len(choices)]
    choices = (
        "You are Ember. Use only the supplied tool-result facts and do not invent different numbers or places.",
        "You are Ember. A tool result is already available; quote its facts and do not call another tool.",
        "You are Ember. Summarize the supplied result exactly and stop at the end of the answer.",
    )
    return choices[variant % len(choices)]


def _prompt(system: str, user: str, history: str = "") -> str:
    return f"<|system|>\n{system}\n<|user|>\n{user}\n{history}<|assistant|>\n"


def _tool_completion(name: str, arguments: dict) -> str:
    return f"<|tool|>\n{compact_json({'name': name, 'arguments': arguments})}\n<|endoftext|>\n"


def _direct_completion(text: str) -> str:
    return f"{text}\n<|endoftext|>\n"


def _tool_call_example(split: str, index: int, values: dict) -> dict:
    tool = TOOLS[index % len(TOOLS)]
    variant = index // len(TOOLS)
    if tool == "weather":
        phrase_count = len(WEATHER_PHRASES) if split == "train" else 2
        location = values["locations"][(variant // phrase_count) % len(values["locations"])]
        user = WEATHER_PHRASES[variant % phrase_count].format(value=location)
        arguments = {"location": location}
        expected_arguments = {"location": location}
    elif tool == "calculator":
        split_offset = 811 if split == "validation" else 0
        left = 21 + (((variant + split_offset) * 19) % 460)
        right = 4 + (((variant + split_offset) * 13) % 86)
        operators = ("+", "-", "*", "/")
        operator = operators[variant % len(operators)]
        expression = f"{left}{operator}{right}"
        user = (
            f"Use a calculator to evaluate {left} {operator} {right}."
            if variant % 2 == 0
            else f"Please calculate the expression {expression}."
        )
        arguments = {"expression": expression}
        expected_arguments = {"expression": [str(left), str(right)]}
    elif tool == "web_search":
        topics = SEARCH_TOPICS if split == "train" else VALIDATION_SEARCH_TOPICS
        phrase_count = len(SEARCH_PHRASES) if split == "train" else 2
        topic = topics[(variant // phrase_count) % len(topics)]
        user = SEARCH_PHRASES[variant % phrase_count].format(value=topic)
        arguments = {"query": topic}
        expected_arguments = {"query": topic}
    else:
        phrase_count = len(TIME_PHRASES) if split == "train" else 2
        city, timezone = values["timezones"][(variant // phrase_count) % len(values["timezones"])]
        user = TIME_PHRASES[variant % phrase_count].format(city=city, timezone=timezone)
        arguments = {"timezone": timezone}
        expected_arguments = {"timezone": city}
    return {
        "id": f"{split}-tool-{index:04d}",
        "kind": "tool_call",
        "expected_tool": tool,
        "expected_arguments": expected_arguments,
        "prompt": _prompt(_system("tool_call", variant, tool), user),
        "completion": _tool_completion(tool, arguments),
    }


def _direct_example(split: str, index: int) -> dict:
    subtype = index % 4
    variant = index // 4
    if subtype == 0:
        names = (
            ("Maya", "Jordan", "Sam", "Avery", "Riley", "Morgan")
            if split == "train"
            else VALIDATION_NAMES
        )
        context_count = len(GREETING_CONTEXTS) if split == "train" else 2
        name = names[(variant // context_count) % len(names)]
        context = GREETING_CONTEXTS[variant % context_count]
        user = f"Write one {context} hello sentence for {name}."
        completion = f"Hello, {name}! This is a {context} note to wish you a great day."
        required_facts = ["Hello", name]
    elif subtype == 1:
        rough_titles = (
            (
                "login page fails sometimes", "reports page wording confusing",
                "upload button does nothing", "dashboard numbers look stale",
                "email alert arrives late", "search results are unclear",
            )
            if split == "train"
            else VALIDATION_ROUGH_TITLES
        )
        context_count = len(REWRITE_CONTEXTS) if split == "train" else 2
        rough = rough_titles[(variant // context_count) % len(rough_titles)]
        context = REWRITE_CONTEXTS[variant % context_count]
        user = f"Rewrite this short title clearly {context}: {rough}."
        completion = rough.capitalize().rstrip(".") + f" ({context.removeprefix('for ')})."
        required_facts = [rough.split()[0]]
    elif subtype == 2:
        topics = DIRECT_TOPICS if split == "train" else VALIDATION_DIRECT_TOPICS
        style_count = len(EXPLAIN_STYLES) if split == "train" else 2
        topic = topics[(variant // style_count) % len(topics)]
        style = EXPLAIN_STYLES[variant % style_count]
        user = f"Explain {topic} {style}."
        completion = f"A {topic} is a practical mechanism used to make a system easier to operate reliably."
        required_facts = [topic]
    else:
        tasks = PLAN_TASKS if split == "train" else VALIDATION_PLAN_TASKS
        context_count = len(PLAN_CONTEXTS) if split == "train" else 2
        task = tasks[(variant // context_count) % len(tasks)]
        context = PLAN_CONTEXTS[variant % context_count]
        user = f"Give me two short steps to {task} {context}."
        completion = f"1. Prepare a controlled example and {task}.\n2. Record the result and verify the expected behavior."
        required_facts = [task.split()[-1]]
    return {
        "id": f"{split}-direct-{index:04d}",
        "kind": "direct_response",
        "required_facts": required_facts,
        "prompt": _prompt(_system("direct_response", variant), user),
        "completion": _direct_completion(completion),
    }


def _result_example(split: str, index: int, values: dict) -> dict:
    subtype = index % 4
    variant = index // 4
    if subtype == 0:
        detail_count = 8 if split == "train" else 2
        location = values["locations"][(variant // detail_count) % len(values["locations"])]
        detail = variant % detail_count
        temperature = 41 + ((detail * 11) % 39)
        condition = ("cloudy", "clear", "windy", "light rain")[detail % 4]
        user = f"How is the weather in {location}?"
        name = "weather"
        arguments = {"location": location}
        result = {"temperature_f": temperature, "condition": condition}
        completion = f"It is {temperature}°F and {condition} in {location}."
        required_facts = [str(temperature), condition, location]
    elif subtype == 1:
        split_offset = 419 if split == "validation" else 0
        left = 23 + (((variant + split_offset) * 17) % 160)
        right = 3 + (((variant + split_offset) * 7) % 29)
        answer = left * right
        user = f"What is {left} times {right}?"
        name = "calculator"
        arguments = {"expression": f"{left}*{right}"}
        result = {"result": answer}
        completion = f"{left} multiplied by {right} equals {answer}."
        required_facts = [str(answer)]
    elif subtype == 2:
        topics = SEARCH_TOPICS if split == "train" else VALIDATION_SEARCH_TOPICS
        detail_count = 8 if split == "train" else 2
        topic = topics[(variant // detail_count) % len(topics)]
        detail = variant % detail_count
        user = f"Find update {detail + 1} about {topic}."
        name = "web_search"
        arguments = {"query": topic}
        result = {"title": f"Update {detail + 1}", "summary": f"A verified update about {topic} is available."}
        completion = f"Update {detail + 1} confirms that verified information about {topic} is available."
        required_facts = [f"Update {detail + 1}", topic]
    else:
        detail_count = 16 if split == "train" else 2
        service = values["services"][(variant // detail_count) % len(values["services"])]
        detail = variant % detail_count
        latency = 21 + ((detail * 11) % 130)
        user = f"Is the {service} service healthy?"
        name = "service_status"
        arguments = {"service": service}
        result = {"status": "healthy", "latency_ms": latency}
        completion = f"The {service} service is healthy with {latency} ms latency."
        required_facts = ["healthy", str(latency), service]
    history = (
        f"<|assistant|>\n<|tool|>\n{compact_json({'name': name, 'arguments': arguments})}\n"
        f"<|tool_result|>\n{compact_json(result)}\n"
    )
    return {
        "id": f"{split}-result-{index:04d}",
        "kind": "tool_result_response",
        "required_facts": required_facts,
        "prompt": _prompt(_system("tool_result_response", variant), user, history),
        "completion": _direct_completion(completion),
    }


def build_sft_examples(split: str, total: int, seed: int = 20260902) -> list[dict]:
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    if total < 3 or total % len(KINDS):
        raise ValueError("total examples must be divisible by three")
    values = TRAIN_VALUES if split == "train" else VALIDATION_VALUES
    per_kind = total // len(KINDS)
    examples = []
    for index in range(per_kind):
        examples.append(_tool_call_example(split, index, values))
        examples.append(_direct_example(split, index))
        examples.append(_result_example(split, index, values))
    random.Random(seed + (0 if split == "train" else 1)).shuffle(examples)
    validate_examples(examples)
    return examples


def validate_examples(examples: Iterable[dict]) -> None:
    rows = list(examples)
    if not rows:
        raise ValueError("SFT data is empty")
    ids = [row.get("id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("SFT example IDs must be unique")
    for row in rows:
        if row.get("kind") not in KINDS:
            raise ValueError(f"unsupported SFT kind: {row.get('kind')}")
        if not str(row.get("prompt", "")).endswith("<|assistant|>\n"):
            raise ValueError(f"SFT prompt must end at the assistant boundary: {row.get('id')}")
        completion = str(row.get("completion", ""))
        if not completion.endswith("<|endoftext|>\n"):
            raise ValueError(f"SFT completion must end with endoftext: {row.get('id')}")
        if completion.count("<|endoftext|>") != 1:
            raise ValueError(f"SFT completion must contain exactly one endoftext: {row.get('id')}")
        if row["kind"] == "tool_call":
            marker, payload_text = completion.split("\n", 1)
            payload = json.loads(payload_text.split("\n", 1)[0])
            if marker != "<|tool|>" or payload.get("name") != row.get("expected_tool"):
                raise ValueError(f"invalid canonical tool envelope: {row.get('id')}")
            arguments = payload.get("arguments")
            if not isinstance(arguments, dict) or not arguments:
                raise ValueError(f"tool arguments must be a non-empty object: {row.get('id')}")
            expected = row.get("expected_arguments") or {}
            blob = " ".join(str(value) for value in arguments.values())
            prompt = row["prompt"]
            for expected_value in expected.values():
                tokens = expected_value if isinstance(expected_value, list) else [expected_value]
                for token in tokens:
                    if str(token) not in blob:
                        raise ValueError(f"tool arguments dropped a required value: {row.get('id')}")
                    if str(token) not in prompt:
                        raise ValueError(f"tool argument is not grounded in the prompt: {row.get('id')}")
        else:
            if "<|tool|>" in completion:
                raise ValueError(f"non-tool completion contains a tool marker: {row.get('id')}")
            for fact in row.get("required_facts") or []:
                if str(fact).casefold() not in completion.casefold():
                    raise ValueError(f"completion omitted a required fact: {row.get('id')}")
                if row["kind"] == "tool_result_response" and str(fact).casefold() not in row["prompt"].casefold():
                    raise ValueError(f"required fact is not present in the tool result: {row.get('id')}")


def assert_held_out_clean(examples: Iterable[dict], held_out_prompts: Iterable[str]) -> None:
    rows = list(examples)
    held_out = [prompt for prompt in held_out_prompts if prompt]
    for row in rows:
        combined = row["prompt"] + row["completion"]
        for prompt in held_out:
            if row["prompt"] == prompt or prompt in combined:
                raise ValueError(f"official evaluation prompt leaked into SFT data: {row['id']}")
        for fragment in HELD_OUT_FRAGMENTS:
            if fragment in combined:
                raise ValueError(f"held-out evaluation fragment leaked into SFT data: {row['id']}")


def jsonl_bytes(examples: Iterable[dict]) -> bytes:
    return ("\n".join(compact_json(row) for row in examples) + "\n").encode("utf-8")


def dataset_sha256(examples: Iterable[dict]) -> str:
    return hashlib.sha256(jsonl_bytes(examples)).hexdigest()
