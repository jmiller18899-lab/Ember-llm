"""Unchanged extraction-v2 functions from commit 10683d1f.
This helper is deterministic supporting code, not learned argument generation.
"""
import re

def extract_location_v2(user: str) -> str | None:
    text = user.strip()
    # Stop only at an explicit temporal suffix or sentence-ending punctuation.
    # Internal dots/apostrophes/commas remain part of the place name.
    match = re.search(
        r"\b(?:in|for)\s+(.+?)(?=\s+(?:right now|at the moment|at this moment|currently|today|now)\b|[?!]\s*$|\.\s*$|$)",
        text,
        flags=re.I,
    )
    if not match:
        return None
    value = match.group(1).strip(" ,?!")
    return value or None


def extract_expression_v2(user: str) -> str | None:
    text = user.casefold().replace(",", "").strip()
    text = re.sub(r"[?.!]+\s*$", "", text)
    num = r"([0-9]+(?:\.[0-9]+)?)"

    m = re.search(num + r"\s+percent\s+of\s+" + num, text)
    if m:
        return f"({m.group(1)}/100)*{m.group(2)}"
    m = re.search(num + r"\s+squared\b", text)
    if m:
        return f"{m.group(1)}**2"
    m = re.search(num + r"\s+to\s+the\s+power\s+of\s+" + num, text)
    if m:
        return f"{m.group(1)}**{m.group(2)}"

    normalized = text
    normalized = re.sub(r"\bmultiplied\s+by\b", " * ", normalized)
    normalized = re.sub(r"\bdivided\s+by\b", " / ", normalized)
    normalized = re.sub(r"\btimes\b", " * ", normalized)
    normalized = re.sub(r"\bplus\b", " + ", normalized)
    normalized = re.sub(r"\bminus\b", " - ", normalized)
    tokens = re.findall(r"[0-9]+(?:\.[0-9]+)?|[+*/-]", normalized)
    if len(tokens) < 3 or len(tokens) % 2 == 0:
        return None
    for index, token in enumerate(tokens):
        if index % 2 == 0 and not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", token):
            return None
        if index % 2 == 1 and token not in {"+", "-", "*", "/"}:
            return None
    return "".join(tokens)


def resolve_v2(case: dict, routed_tool: str) -> dict:
    user = case["user"]
    if routed_tool == "weather":
        key, value = "location", extract_location_v2(user)
    elif routed_tool == "get_time":
        key, value = "timezone", extract_location_v2(user)
    elif routed_tool == "calculator":
        key, value = "expression", extract_expression_v2(user)
    elif routed_tool == "web_search":
        key, value = "query", user.strip().rstrip("?").strip()
    else:
        raise ValueError(routed_tool)
    payload = {"name": routed_tool, "arguments": {key: value}} if value else None
    return {"tool": routed_tool, "key": key, "value": value, "payload": payload}
