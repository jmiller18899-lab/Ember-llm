"""Deterministic semantic repair data. No model calls, benchmark labels, or training.

Historical generators remain immutable. Gold answers come from explicit task
facts and reviewed reference tables; validation independently reads the visible
request/result instead of trusting an expected-answer field in each record.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
import random
import re


VERSION = "ember-semantic-data-v1"
EOT = "<|endoftext|>"
TOOL = "<|tool|>"
KINDS = ("tool_call", "direct_response", "tool_result_response")
TOOL_FAMILIES = ("weather_literal", "weather_choice", "calculator_literal", "calculator_parentheses",
                 "search_literal", "search_choice", "time_literal", "time_choice")
DIRECT_FAMILIES = ("definition", "plan", "greeting", "rewrite", "owner", "missing_owner", "saved", "count")
RESULT_FAMILIES = ("weather_f", "weather_c", "arithmetic", "service", "error", "search", "missing_owner", "checks")
FAMILIES = tuple(f"{kind}/{family}" for kind, names in zip(KINDS, (TOOL_FAMILIES, DIRECT_FAMILIES, RESULT_FAMILIES)) for family in names)

# Concrete definitions replace the old single generic sentence. These shared
# concepts are training knowledge; the frozen benchmark's prompts are excluded.
DEFINITIONS = {
    "checkpoint": "A checkpoint stores model weights and training state so an interrupted training run can continue.",
    "cache": "A cache keeps a copy of data nearby so later requests can retrieve it faster.",
    "checksum": "A checksum is a value calculated from data to help detect accidental changes.",
    "retry": "A retry repeats an operation after an earlier attempt fails.",
    "queue": "A queue holds items until they can be processed, usually in arrival order.",
    "tokenizer": "A tokenizer converts text into token IDs that a language model can process.",
    "webhook": "A webhook sends an HTTP request to another application when an event occurs.",
    "database index": "A database index is a lookup structure that can speed up searches for matching rows.",
    "unit test": "A unit test checks that a small part of a program behaves as expected.",
    "backup": "A backup is a separate copy of data that can be used to restore the original.",
    "rate limit": "A rate limit caps how many requests a client can make within a time period.",
    "timeout": "A timeout ends a wait when an operation has not finished within its allowed time.",
    "latency": "Latency is the time between starting a request and receiving its response.",
    "schema": "A schema describes the expected structure and data types of a record or document.",
    "session": "A session links a sequence of interactions to shared application state.",
    "encryption": "Encryption transforms readable data into ciphertext that requires a key to decrypt.",
    "migration": "A migration applies a versioned change to a database schema or its data.",
    "health check": "A health check tests whether a service can respond as expected.",
}
PLANS = {
    "test a contact form": "1. Submit a valid message and verify receipt.\n2. Omit a required field and verify a clear error.",
    "verify a backup": "1. Restore the backup into an isolated test location.\n2. Compare restored files with the expected contents.",
    "check a broken link": "1. Open the link and record the response.\n2. Correct its destination and verify the page loads.",
    "test a password reset": "1. Request a reset for a test account.\n2. Use the reset link and verify the new password works.",
    "validate a CSV import": "1. Import a small CSV with known values.\n2. Compare the stored rows with those values.",
    "test a search box": "1. Search for an item that exists and verify it appears.\n2. Search for an absent item and verify the empty state.",
    "verify a notification": "1. Trigger the event with a test account.\n2. Verify exactly one notification reaches the intended account.",
    "test file upload": "1. Upload a supported file and verify its contents.\n2. Upload an unsupported file and verify rejection.",
    "test a logout flow": "1. Sign in with a test account and log out.\n2. Request a protected page and verify authentication is required.",
    "verify a scheduled report": "1. Run the schedule with known test data.\n2. Check the report time and compare its values with that data.",
    "check an image preview": "1. Select an image and verify its preview.\n2. Remove the image and verify the preview disappears.",
    "test account recovery": "1. Start recovery for a test account.\n2. Complete verification and confirm account access is restored.",
    "validate a JSON export": "1. Export a record with known values.\n2. Parse the JSON and compare its fields with that record.",
    "test a shopping cart": "1. Add a product and verify its price and quantity.\n2. Remove it and verify the total updates.",
    "test a pagination control": "1. Load enough records to require another page.\n2. Change pages and check for missing or repeated records.",
    "check a download button": "1. Download a known file.\n2. Compare its name and contents with the expected file.",
    "test a date filter": "1. Choose a range containing known records.\n2. Verify records outside that range are excluded.",
    "test a settings toggle": "1. Change the toggle and save the setting.\n2. Reload the page and verify the saved state remains.",
}
REWRITES = {
    "upload button does nothing": "Upload button does not respond.",
    "reports page wording confusing": "Reports page wording is unclear.",
    "dashboard numbers look stale": "Dashboard shows outdated numbers.",
    "email alert arrives late": "Email alerts arrive late.",
    "search results are unclear": "Search results need clearer labels.",
    "profile photo goes missing": "Profile photo disappears.",
    "download starts two times": "Download starts twice.",
    "cart total does not update": "Cart total fails to update.",
    "reset link expires too soon": "Password reset link expires too quickly.",
    "date filter includes wrong dates": "Date filter includes dates outside the selected range.",
    "message sends twice": "Message is sent twice.",
    "attachment preview is blurry": "Attachment preview is blurry.",
    "page numbers jump around": "Pagination controls change position.",
    "save button stays disabled": "Save button remains disabled.",
    "dark theme text hard to read": "Dark theme text is difficult to read.",
    "menu covers the submit button": "Menu obscures the submit button.",
    "audio keeps playing after pause": "Audio continues playing after pause.",
    "export leaves out a column": "Export omits a column.",
}
CITIES = (
    ("Asheville", "North Carolina", "NC"), ("Eugene", "Oregon", "OR"),
    ("Duluth", "Minnesota", "MN"), ("Tacoma", "Washington", "WA"),
    ("Santa Fe", "New Mexico", "NM"), ("Albany", "New York", "NY"),
    ("Erie", "Pennsylvania", "PA"), ("Tucson", "Arizona", "AZ"),
    ("Burlington", "Vermont", "VT"), ("Savannah", "Georgia", "GA"),
    ("Fargo", "North Dakota", "ND"), ("Mobile", "Alabama", "AL"),
    ("Wichita", "Kansas", "KS"), ("Omaha", "Nebraska", "NE"),
    ("Reno", "Nevada", "NV"), ("Spokane", "Washington", "WA"),
    ("Flagstaff", "Arizona", "AZ"), ("Madison", "Wisconsin", "WI"),
    ("Akron", "Ohio", "OH"), ("Bend", "Oregon", "OR"),
    ("Chattanooga", "Tennessee", "TN"), ("Waco", "Texas", "TX"),
    ("Roanoke", "Virginia", "VA"), ("Missoula", "Montana", "MT"),
)
ZONES = ("Europe/Amsterdam", "Europe/Oslo", "Europe/Warsaw", "Europe/Lisbon",
         "Europe/Prague", "Europe/Vienna", "Europe/Zurich", "Europe/Helsinki",
         "Africa/Nairobi", "Africa/Cairo", "Africa/Lagos", "Africa/Johannesburg",
         "Asia/Bangkok", "Asia/Jakarta", "Asia/Manila", "Asia/Taipei",
         "Asia/Dubai", "Asia/Kathmandu", "Pacific/Fiji", "Pacific/Guam",
         "America/Bogota", "America/Lima", "America/Montevideo", "America/Costa_Rica")
NAMES = ("Nia", "Omar", "Elena", "Hugo", "Imani", "Luca", "Priya", "Noah",
         "Amara", "Felix", "Sofia", "Idris", "Leila", "Theo", "Zara", "Milo",
         "Anika", "Jonah", "Nora", "Dante", "Iris", "Mateo", "Esme", "Rohan")

VALUE_STYLES = (
    "Use {tool}. Set {field} to {value} exactly.",
    "Call {tool} with {field} equal to {value}.",
    "For {tool}, use this exact {field}: {value}.",
    "Use the {tool} tool with {field} {value}. Preserve the value.",
    "The requested {field} is {value}. Pass it unchanged to {tool}.",
    "Send {field} {value} to {tool} without changing it.",
)
CHOICE_STYLES = (
    "Use {tool} with {field} {value}, not {other}.",
    "Choose {value} for {field}; ignore {other}. Call {tool}.",
    "Call {tool}. The correct {field} is {value}; {other} is a distractor.",
    "For {tool}, pass {value} as {field} instead of {other}.",
    "Use {value} as {field} in {tool}. Do not substitute {other}.",
    "Ignore {other}; the requested {field} for {tool} is {value}.",
)
NATURAL_REQUESTS = {
    "weather": "What is the weather in {value} right now?",
    "calculator": "Use a calculator to evaluate {value}.",
    "web_search": "Find current information about {value}.",
    "get_time": "What time is it in the timezone {value}?",
}
PLAIN_GREETINGS = {
    "Offer a brief greeting.": "Hello! It is good to meet you.",
    "Offer a brief farewell.": "Goodbye! Have a good day.",
    "Welcome me in a short sentence.": "Welcome! I am glad you are here.",
    "Wish me well as I leave.": "Take care! I hope your day goes well.",
    "Start our conversation with a warm hello.": "Hello! I am happy to help.",
    "End our conversation with a warm goodbye.": "Goodbye! It was nice talking with you.",
    "Give me a simple welcome.": "Welcome! It is nice to see you.",
    "Give me a simple parting message.": "Goodbye, and take care!",
}
CATALOG_STYLES = {
    "definition": ("What does {value} mean?", "Define {value} in plain language.", "Briefly explain {value}.",
                   "Give a useful definition of {value}.", "What is the purpose of {value}?",
                   "Describe {value} for a beginner.", "Explain the term {value} briefly.", "Tell me what {value} does."),
    "plan": ("Give two practical steps to {value}.", "How can I {value}? Use two steps.",
             "List two concrete steps to {value}.", "Help me {value} in two steps.",
             "Write a two-step checklist to {value}.", "Suggest two checks to {value}.",
             "In two steps, explain how to {value}.", "I want to {value}. Provide two steps."),
    "rewrite": ("Make this issue title clearer: {value}.", "Rewrite this issue clearly: {value}.",
                "Improve this bug title: {value}.", "Give this issue a clear title: {value}.",
                "Polish this title without changing its meaning: {value}.", "Use a clearer issue title for: {value}.",
                "Turn this into a readable bug title: {value}.", "Clarify this report title: {value}."),
    "greeting": ("Greet {value} in a friendly sentence.", "Write a short greeting for {value}.",
                 "Say hello warmly to {value}.", "Give {value} a brief welcome.",
                 "Write one welcoming sentence addressed to {value}.", "Offer a friendly hello to {value}."),
}
DIRECT_INSTRUCTIONS = {
    "owner": "Who owns the ticket? Answer in one sentence.",
    "missing_owner": "Who owns the ticket? If the owner is absent, say Owner unavailable.",
    "saved": "State whether the named file was saved in one sentence.",
    "count": "How many items are in the list? Answer with only the number.",
}
RESULT_INSTRUCTIONS = {
    "weather_f": "Describe the supplied weather result in one sentence.",
    "weather_c": "Describe the supplied weather result in one sentence.",
    "arithmetic": "State the supplied calculation result in one short sentence.",
    "service": "Report the service status and latency from the result.",
    "error": "Explain the weather lookup error without inventing a forecast.",
    "search": "Summarize the supplied search result; say No results found. if empty.",
    "missing_owner": "Who owns the service? If the result omits the owner, say Owner unavailable.",
    "checks": "Report how many checks passed or failed from the supplied counts.",
}
ROLE_TOKENS = {role: f"<|{role}|>" for role in ("system", "user", "assistant", "tool_result")}


def compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def done(answer: str) -> str:
    return answer + "\n" + EOT + "\n"


def render_prompt(messages: list[dict]) -> str:
    return "".join(f"{ROLE_TOKENS[m['role']]}\n{m['content']}\n" for m in messages) + "<|assistant|>\n"


def tool_spec(family: str) -> tuple[str, str]:
    if family.startswith("weather_"):
        return "weather", "location"
    if family.startswith("calculator_"):
        return "calculator", "expression"
    if family.startswith("search_"):
        return "web_search", "query"
    return "get_time", "timezone"


def system_for(family: str) -> str:
    kind, subtype = family.split("/")
    if kind == "tool_call":
        tool, field = tool_spec(subtype)
        return f"You are Ember. Use {tool} with JSON arguments. Its only required argument is {field}, a string."
    if kind == "direct_response":
        return "You are Ember. Answer the request directly without tools. Be specific and use only the supplied facts when present."
    return "You are Ember. Answer from the supplied tool result without another tool call. Do not invent missing values or claim success after an error."


def tool_styles(subtype: str) -> tuple:
    tool, _ = tool_spec(subtype)
    return CHOICE_STYLES if subtype.endswith("choice") else (NATURAL_REQUESTS[tool], *VALUE_STYLES[1:])


def result_request(subtype: str, arguments: dict, alternate: bool = False) -> str:
    if alternate:
        return RESULT_INSTRUCTIONS[subtype]
    if subtype in {"weather_f", "weather_c", "error"}:
        return f"What is the weather in {arguments['location']}?"
    if subtype == "arithmetic":
        return f"Calculate {arguments['expression']}."
    if subtype == "service":
        return f"Is the {arguments['service']} service healthy?"
    if subtype == "search":
        return f"Find the result for {arguments['query']}."
    if subtype == "missing_owner":
        return f"Who owns the {arguments['service']} service?"
    return f"What is the result of the {arguments['service']} checks?"


def city_value(group: int, side: int) -> str:
    city, state, abbrev = CITIES[2 * (group % 12) + side]
    return (city, f"{city}, {state}", f"{city}, {abbrev}", f"{city} {abbrev}",
            f"{city}, {state}, USA", f"{city}, {state}, United States")[group // 12]


def identifier(seed: int, group: int, tag: str) -> str:
    return hashlib.sha256(f"{VERSION}|{seed}|{tag}|{group}".encode()).hexdigest()[:6].upper()


def tool_value(subtype: str, group: int, side: int, seed: int) -> str:
    n = 2 * group + side
    if subtype.startswith("weather_"):
        return city_value(group, side)
    if subtype.startswith("time_"):
        return ZONES[2 * (group % 12) + side]
    if subtype == "calculator_literal":
        return f"{14 + n}{('+', '-', '*', '/')[group % 4]}{3 + group % 17}"
    if subtype == "calculator_parentheses":
        return f"({174 + n}+{5 + group % 13})*{2 + group % 7}"
    code = identifier(seed, n, subtype)
    return (f"release notes for build {code}", f"dependency \"optional\" change {code}",
            f"issue {code} recovery status", f"API migration {{preview}} {code}")[group % 4]


def make_record(family: str, group: int, side: int, seed: int) -> dict:
    kind, subtype = family.split("/")
    style = group // 12
    n = 2 * group + side
    history = []
    if kind == "tool_call":
        tool, field = tool_spec(subtype)
        value = tool_value(subtype, group, side, seed)
        other = tool_value(subtype, group, 1 - side, seed)
        styles = tool_styles(subtype)
        user = styles[style].format(tool=tool, field=field, value=compact(value), other=compact(other))
        answer = TOOL + "\n" + compact({"name": tool, "arguments": {field: value}})
    elif kind == "direct_response":
        if subtype in {"definition", "plan", "rewrite"}:
            catalog = {"definition": DEFINITIONS, "plan": PLANS, "rewrite": REWRITES}[subtype]
            key = list(catalog)[n % len(catalog)]
            user = CATALOG_STYLES[subtype][n // len(catalog)].format(value=key)
            answer = catalog[key]
        elif subtype == "greeting":
            if group < 4:
                user = list(PLAIN_GREETINGS)[2 * group + side]
                answer = PLAIN_GREETINGS[user]
            else:
                name = NAMES[2 * (group % 12) + side]
                user = CATALOG_STYLES[subtype][style].format(value=name)
                answer = f"Hello, {name}! It is nice to meet you."
        else:
            code = identifier(seed, group, subtype)
            name = NAMES[2 * (group % 12) + side]
            if subtype in {"owner", "missing_owner"}:
                facts = {"ticket": f"Case-{code}"}
                if subtype == "owner" or side == 0:
                    facts["owner"] = name
                answer = f"The owner is {name}." if "owner" in facts else "Owner unavailable."
            elif subtype == "saved":
                facts = {"file": f"report-{code}.json", "saved": side == 0}
                answer = f"{facts['file']} was {'saved' if facts['saved'] else 'not saved'}."
            else:
                facts = {"items": [f"item-{code}-{i}" for i in range(1, 2 + group % 5 + side)]}
                answer = str(1 + group % 5 + side)
            user = "Facts: " + compact(facts) + "\n" + DIRECT_INSTRUCTIONS[subtype]
    else:
        code = identifier(seed, group, subtype)
        if subtype in {"weather_f", "weather_c"}:
            tool, arguments = "weather", {"location": city_value(group, 0)}
            unit = "F" if subtype == "weather_f" else "C"
            temperature = -19 + group + 7 * side if unit == "F" else -13 + group // 2 + 5 * side
            condition = ("clear", "rainy", "cloudy", "windy", "snowy", "sunny")[group % 6]
            result = {"temperature_f" if unit == "F" else "temperature_c": temperature, "condition": condition}
            answer = f"It is {temperature}°{unit} and {condition} in {arguments['location']}."
        elif subtype == "arithmetic":
            left, right = 31 + group, 2 + side
            tool, arguments = "calculator", {"expression": f"{left}/{right}" if group % 2 == 0 else f"{left}-{right}"}
            # Division uses /2 or /4 to keep exact terminating decimal labels.
            if group % 2 == 0:
                right = 2 if side == 0 else 4
                arguments["expression"] = f"{left}/{right}"
                value = Decimal(left) / Decimal(right)
            else:
                value = Decimal(left - right)
            result = {"result": int(value) if value == int(value) else float(value)}
            answer = f"The result is {format_number(value)}."
        elif subtype == "service":
            tool, arguments = "service_status", {"service": f"worker-{code}"}
            result = {"status": "healthy" if side == 0 else "unhealthy", "latency_ms": 17 + 3 * group}
            answer = f"The {arguments['service']} service is {result['status']} with {result['latency_ms']} ms latency."
        elif subtype == "error":
            tool, arguments = "weather", {"location": city_value(group, 0)}
            result = {"error": ("timeout", "rate_limit", "connection_refused", "unavailable")[(group + side) % 4]}
            answer = f"Weather lookup failed: {result['error']}."
        elif subtype == "search":
            tool, arguments = "web_search", {"query": f"build report {code}"}
            summary = f"Build {code} {'completed' if group % 2 == 0 else 'failed'}."
            result = {"results": [] if side == 0 else [{"title": f"Build {code}", "summary": summary}]}
            answer = "No results found." if side == 0 else summary
        elif subtype == "missing_owner":
            tool, arguments = "service_status", {"service": f"worker-{code}"}
            result = {"status": "healthy"}
            if side == 0:
                result["owner"] = NAMES[2 * (group % 12)]
            answer = f"The owner is {result['owner']}." if side == 0 else "Owner unavailable."
        else:
            tool, arguments = "service_status", {"service": f"checks-{code}"}
            result = {"total_checks": 7 + group, "failed_checks": 0 if side == 0 else 1 + group % 4}
            answer = (f"All {result['total_checks']} checks passed." if side == 0 else
                      f"{result['failed_checks']} of {result['total_checks']} checks failed.")
        user = result_request(subtype, arguments, alternate=bool(group % 2))
        history = [{"role": "assistant", "content": TOOL + "\n" + compact({"name": tool, "arguments": arguments})},
                   {"role": "tool_result", "content": compact(result)}]
    messages = [{"role": "system", "content": system_for(family)}, {"role": "user", "content": user}, *history]
    record = {"id": f"{family.replace('/', '-')}-{group:03d}-{side}", "family": family,
              "pair_id": f"{family}-{group:03d}", "pair_side": side, "kind": kind,
              "messages": messages, "prompt": render_prompt(messages), "completion": done(answer)}
    if kind == "tool_call":
        record["expected_tool"] = tool
    return record


def build_dataset(seed: int = 20260908) -> dict[str, list[dict]]:
    # Entire counterfactual pairs stay together. City/timezone pairs are also
    # kept in one split across all their wording variants and tool families.
    pair_classes = list(range(12))
    random.Random(seed).shuffle(pair_classes)
    validation_classes = set(pair_classes[:2])
    splits = {"train": [], "validation": []}
    for family in FAMILIES:
        for group in range(72):
            split = "validation" if group % 12 in validation_classes else "train"
            for side in (0, 1):
                splits[split].append({**make_record(family, group, side, seed), "split": split})
    for offset, rows in enumerate(splits.values()):
        random.Random(seed + 100 + offset).shuffle(rows)
    return splits


def strict_json(text: str):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError(f"duplicate JSON key: {key}")
            obj[key] = value
        return obj
    def invalid(value):
        raise ValueError(f"invalid JSON constant: {value}")
    return json.loads(text, object_pairs_hook=pairs, parse_float=Decimal, parse_constant=invalid)


def format_number(value) -> str:
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError("non-finite number")
    text = format(number, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def expression_value(expression: str) -> Decimal:
    """Evaluate only the tiny arithmetic grammar, never arbitrary Python."""
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return Decimal(node.value)
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div) and right != 0:
                return left / right
        raise ValueError("unsupported arithmetic expression")
    if len(expression) > 80:
        raise ValueError("expression too long")
    return visit(ast.parse(expression, mode="eval").body)


def match_template(user: str, templates: tuple, quoted: bool = False, **fixed) -> dict:
    value_pattern = r'("(?:[^"\\]|\\.)*")' if quoted else r"(.+?)"
    for template in templates:
        pattern = template
        for key, value in fixed.items():
            pattern = pattern.replace("{" + key + "}", value)
        pattern = re.escape(pattern)
        names = []
        for placeholder in re.findall(r"\{(value|other)\}", template):
            if placeholder not in names:
                names.append(placeholder)
                pattern = pattern.replace(re.escape("{" + placeholder + "}"), f"(?P<{placeholder}>{value_pattern[1:-1]})")
        match = re.fullmatch(pattern, user)
        if match:
            return {name: strict_json(match[name]) if quoted else match[name] for name in names}
    raise ValueError("unrecognized request wording")


def fields(value, required: set[str], optional: set[str] = frozenset()) -> None:
    if not isinstance(value, dict) or not required <= set(value) <= required | optional:
        raise ValueError("unexpected fixture fields")


def nonempty(value) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected a nonempty string")


def numeric(value) -> None:
    if type(value) not in {int, Decimal} or not Decimal(value).is_finite():
        raise ValueError("expected a finite JSON number")


def expected_from_request(row: dict) -> str:
    """Independent oracle: recover requested values from the actual messages."""
    kind, subtype = row["family"].split("/")
    messages = row["messages"]
    user = messages[1]["content"]
    if kind == "tool_call":
        tool, field = tool_spec(subtype)
        styles = tool_styles(subtype)
        parsed = match_template(user, styles, quoted=True, tool=tool, field=field)
        if not isinstance(parsed["value"], str) or not parsed["value"]:
            raise ValueError("tool value must be a nonempty string")
        return done(TOOL + "\n" + compact({"name": tool, "arguments": {field: parsed["value"]}}))
    if kind == "direct_response":
        if subtype == "greeting" and user in PLAIN_GREETINGS:
            return done(PLAIN_GREETINGS[user])
        if subtype in CATALOG_STYLES:
            key = match_template(user, CATALOG_STYLES[subtype])["value"]
            if subtype == "greeting":
                answer = f"Hello, {key}! It is nice to meet you."
            else:
                answer = {"definition": DEFINITIONS, "plan": PLANS, "rewrite": REWRITES}[subtype][key]
        else:
            first, instruction = user.split("\n", 1)
            if not first.startswith("Facts: ") or instruction != DIRECT_INSTRUCTIONS[subtype]:
                raise ValueError("unrecognized fact request")
            facts = strict_json(first[len("Facts: "):])
            if subtype in {"owner", "missing_owner"}:
                fields(facts, {"ticket", "owner"} if subtype == "owner" else {"ticket"}, {"owner"})
                nonempty(facts["ticket"])
                if "owner" in facts:
                    nonempty(facts["owner"])
                answer = f"The owner is {facts['owner']}." if "owner" in facts else "Owner unavailable."
            elif subtype == "saved":
                fields(facts, {"file", "saved"})
                nonempty(facts["file"])
                if type(facts["saved"]) is not bool:
                    raise ValueError("saved must be a JSON boolean")
                answer = f"{facts['file']} was saved." if facts["saved"] else f"{facts['file']} was not saved."
            else:
                fields(facts, {"items"})
                if not isinstance(facts["items"], list):
                    raise ValueError("items must be a list")
                for item in facts["items"]:
                    nonempty(item)
                answer = str(len(facts["items"]))
        return done(answer)
    if not messages[2]["content"].startswith(TOOL + "\n"):
        raise ValueError("missing tool marker in assistant history")
    call = strict_json(messages[2]["content"].removeprefix(TOOL + "\n"))
    fields(call, {"name", "arguments"})
    tool, argument = ({"weather_f": ("weather", "location"), "weather_c": ("weather", "location"),
                       "error": ("weather", "location"), "arithmetic": ("calculator", "expression"),
                       "search": ("web_search", "query")}.get(subtype, ("service_status", "service")))
    if call["name"] != tool:
        raise ValueError("tool result is attached to the wrong tool")
    fields(call["arguments"], {argument})
    nonempty(call["arguments"][argument])
    result = strict_json(messages[3]["content"])
    if user not in {result_request(subtype, call["arguments"], alternate) for alternate in (False, True)}:
        raise ValueError("unrecognized result request")
    if subtype in {"weather_f", "weather_c"}:
        field, unit = ("temperature_f", "F") if subtype == "weather_f" else ("temperature_c", "C")
        fields(result, {field, "condition"})
        numeric(result[field])
        nonempty(result["condition"])
        answer = f"It is {format_number(result[field])}°{unit} and {result['condition']} in {call['arguments']['location']}."
    elif subtype == "arithmetic":
        fields(result, {"result"})
        numeric(result["result"])
        value = Decimal(str(result["result"]))
        if value != expression_value(call["arguments"]["expression"]):
            raise ValueError("supplied arithmetic result is false")
        answer = "The result is " + format_number(value) + "."
    elif subtype == "service":
        fields(result, {"status", "latency_ms"})
        if result["status"] not in {"healthy", "unhealthy"} or type(result["latency_ms"]) is not int or result["latency_ms"] < 0:
            raise ValueError("invalid service status fixture")
        answer = f"The {call['arguments']['service']} service is {result['status']} with {result['latency_ms']} ms latency."
    elif subtype == "error":
        fields(result, {"error"})
        nonempty(result["error"])
        answer = f"Weather lookup failed: {result['error']}."
    elif subtype == "search":
        fields(result, {"results"})
        if not isinstance(result["results"], list) or len(result["results"]) > 1:
            raise ValueError("unsupported multi-result fixture")
        for item in result["results"]:
            fields(item, {"title", "summary"})
            nonempty(item["title"])
            nonempty(item["summary"])
        answer = result["results"][0]["summary"] if result["results"] else "No results found."
    elif subtype == "missing_owner":
        fields(result, {"status"}, {"owner"})
        if result["status"] not in {"healthy", "unhealthy"}:
            raise ValueError("invalid service status")
        if "owner" in result:
            nonempty(result["owner"])
        answer = f"The owner is {result['owner']}." if "owner" in result else "Owner unavailable."
    else:
        fields(result, {"total_checks", "failed_checks"})
        total, failed = result["total_checks"], result["failed_checks"]
        if type(total) is not int or type(failed) is not int or not 0 <= failed <= total:
            raise ValueError("inconsistent check counts")
        answer = f"{failed} of {total} checks failed." if failed else f"All {total} checks passed."
    return done(answer)


def validate_row(row: dict) -> None:
    family = row["family"]
    if family not in FAMILIES or row["kind"] != family.split("/")[0]:
        raise ValueError("invalid family/kind")
    messages = row["messages"]
    expected_roles = ["system", "user"] + (["assistant", "tool_result"] if row["kind"] == KINDS[2] else [])
    if [m["role"] for m in messages] != expected_roles or messages[0]["content"] != system_for(family):
        raise ValueError("invalid message roles or system instructions")
    for message in messages:
        nonempty(message["content"])
        if any(marker in message["content"] for marker in (*ROLE_TOKENS.values(), EOT)):
            raise ValueError("unexpected control marker inside message content")
    if row["prompt"] != render_prompt(messages) or EOT in row["prompt"]:
        raise ValueError("rendered prompt differs from the source messages")
    expected = expected_from_request(row)
    if row["completion"] != expected:
        raise ValueError("target does not match the actual request or tool result")
    if row["kind"] == "tool_call":
        payload = strict_json(row["completion"].split("\n")[1])
        if payload["name"] != row["expected_tool"]:
            raise ValueError("wrong tool metadata")


def jsonl_bytes(rows: list[dict]) -> bytes:
    return ("\n".join(compact(row) for row in rows) + "\n").encode("utf-8")


def validate_dataset(splits: dict[str, list[dict]]) -> dict:
    if set(splits) != {"train", "validation"}:
        raise ValueError("both dataset splits are required")
    all_ids, all_prompts, all_pairs = set(), set(), set()
    summary = {}
    mutations_rejected = 0
    for split, expected_count in (("train", 2880), ("validation", 576)):
        rows = splits[split]
        if len(rows) != expected_count:
            raise ValueError("unexpected dataset size")
        pairs = defaultdict(list)
        for row in rows:
            validate_row(row)
            if row["split"] != split or row["id"] in all_ids or row["prompt"] in all_prompts:
                raise ValueError("duplicate or misplaced training/validation row")
            all_ids.add(row["id"])
            all_prompts.add(row["prompt"])
            pairs[row["pair_id"]].append(row)
        counts = Counter(row["family"] for row in rows)
        if counts != Counter({family: expected_count // len(FAMILIES) for family in FAMILIES}):
            raise ValueError("unbalanced dataset families")
        if all_pairs.intersection(pairs):
            raise ValueError("counterfactual pair leaked between splits")
        all_pairs.update(pairs)
        for group in pairs.values():
            if len(group) != 2 or {r["pair_side"] for r in group} != {0, 1} or group[0]["completion"] == group[1]["completion"]:
                raise ValueError("invalid counterfactual pair")
            for row, other in ((group[0], group[1]), (group[1], group[0])):
                # Wrong but well-formed answers, and post-EOS noise, must fail.
                for bad in (other["completion"], row["completion"] + " invented extra facts"):
                    try:
                        validate_row({**row, "completion": bad})
                    except ValueError:
                        mutations_rejected += 1
                    else:
                        raise ValueError("semantic checker accepted a corrupted target")
        summary[split] = {"rows": len(rows), "pairs": len(pairs), "kinds": dict(Counter(r["kind"] for r in rows)),
                          "families": dict(counts), "sha256": hashlib.sha256(jsonl_bytes(rows)).hexdigest()}
    return {"status": "PASS", "splits": summary, "corrupted_targets_rejected": mutations_rejected}
