"""Whole-input argument parsing; the evaluated v2 helper stays unchanged.

Place names are parsed, not geocoded. A live adapter must still resolve a unique
place. Unsupported expressions and ambiguous locations produce no tool call.
"""
from __future__ import annotations

import argparse
import ast
from fractions import Fraction
import json
import re
import unicodedata
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

REVISION = "argument-parser-v3"
MAX_REQUEST = 4096
MAX_EXPRESSION = 200
MAX_VALUE = 10**15
NUMBER = r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
SIGNED_NUMBER = rf"[+-]?\s*{NUMBER}"
TEMPORAL = r"(?:right now|at the moment|at this moment|currently|today|now)"
POLITE = r"(?:please|thank you|thanks|for me)"
MESSAGES = {
    "invalid_request": "Please provide one short request without conversation markers.",
    "unsupported_arithmetic": "Please provide one expression using numbers, parentheses, +, -, *, /, or **.",
    "arithmetic_limit": "Please use smaller numbers and integer exponents between -12 and 12.",
    "undefined_arithmetic": "That calculation is undefined. Please check its divisors and exponents.",
    "missing_location": "Which location should I use? Include a city and region.",
    "ambiguous_location": "Please name one location. Quote the full name if it contains words such as 'and' or 'or'.",
    "unsupported_time": "This tool handles current weather or time. Please request the current conditions.",
    "invalid_timezone": "Please provide a valid IANA time zone or a city and region.",
}


class ParseError(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(MESSAGES[reason])


def request_text(user):
    if not isinstance(user, str) or not user.strip() or len(user) > MAX_REQUEST or "<|" in user:
        raise ParseError("invalid_request")
    if any(unicodedata.category(c).startswith("C") and c not in "\n\r\t" for c in user):
        raise ParseError("invalid_request")
    return " ".join(user.split())


def _trim_sentence(text):
    text = text.rstrip("?! ")
    # Keep the last dot in initialisms such as D.C.; other final dots end a sentence.
    if text.endswith(".") and not re.search(r"(?:[^\W\d_]\.){2,}$", text):
        text = text[:-1].rstrip()
    return text


def _trim_polite(text):
    for _ in range(3):
        old = text
        text = _trim_sentence(text)
        text = re.sub(rf"(?:,\s*|\s+){POLITE}$", "", text, flags=re.I).rstrip()
        if text == old:
            break
    return text


def _expression_body(user):
    text = request_text(user)
    # An exclamation mark may mean factorial; it is not harmless punctuation.
    if "!" in text:
        raise ParseError("unsupported_arithmetic")
    text = _trim_polite(text).strip()
    text = re.sub(r"^(?:please|kindly)\s+", "", text, flags=re.I)
    text = re.sub(r"^(?:can|could|would|will)\s+you\s+(?:please\s+)?", "", text, flags=re.I)
    text = re.sub(
        r"^(?:calculate|compute|evaluate|work out|find the value of|find the result of|"
        r"what is|what are|how much is|i need|find)(?:\s*:\s*|\s+)", "", text, flags=re.I,
    )
    product = re.fullmatch(rf"the product of ({SIGNED_NUMBER}\s+(?:times|multiplied by)\s+{SIGNED_NUMBER})", text, re.I)
    if product:
        text = product.group(1)
    # The 'What do ... add up to?' frame is explicit addition, not arbitrary prose.
    frame = re.fullmatch(rf"what do ({SIGNED_NUMBER}(?:\s+plus\s+{SIGNED_NUMBER})+) add up to", text, re.I)
    if frame:
        text = frame.group(1)
    for left, right in (("`", "`"), ('"', '"'), ("“", "”")):
        if text.startswith(left) and text.endswith(right) and len(text) > 2:
            text = text[1:-1].strip()
            break
    return text


def _bounded_value(value):
    if abs(value) > MAX_VALUE or value.numerator.bit_length() > 4096 or value.denominator.bit_length() > 4096:
        raise ParseError("arithmetic_limit")
    return value


def _validate_arithmetic(expression):
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        raise ParseError("unsupported_arithmetic") from None
    if sum(1 for _ in ast.walk(tree)) > 60:
        raise ParseError("arithmetic_limit")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            literal = ast.get_source_segment(expression, node)
            if len(literal) > 48:
                raise ParseError("arithmetic_limit")
            exponent = re.search(r"[eE]([+-]?[0-9]+)$", literal)
            if exponent and abs(int(exponent.group(1))) > 12:
                raise ParseError("arithmetic_limit")
            return _bounded_value(Fraction(literal))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return _bounded_value(visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1))
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow):
                if b.denominator != 1 or abs(b) > 12:
                    raise ParseError("arithmetic_limit")
                value = a ** int(b)
            elif isinstance(node.op, ast.Div):
                value = a / b
            elif isinstance(node.op, ast.Mult):
                value = a * b
            elif isinstance(node.op, ast.Add):
                value = a + b
            else:
                value = a - b
            return _bounded_value(value)
        raise ParseError("unsupported_arithmetic")

    try:
        visit(tree.body)
    except (ZeroDivisionError, OverflowError):
        raise ParseError("undefined_arithmetic") from None


def parse_arithmetic(user):
    text = _expression_body(user)
    text = text.translate(str.maketrans({"−": "-", "×": "*", "÷": "/"}))
    # Consecutive postfix powers need explicit grouping to avoid changing their
    # order into Python's right-associative exponentiation.
    if re.search(r"\b(?:squared|cubed)\s*(?:\*\*|squared\b|cubed\b|to the power of\b)", text, re.I):
        raise ParseError("unsupported_arithmetic")
    # Remove commas only inside complete, well-formed thousands groups.
    text = re.sub(r"(?<![\w.,])[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?(?![\w.,])",
                  lambda m: m.group().replace(",", ""), text)
    for pattern, operator in ((r"\bmultiplied by\b", "*"), (r"\bdivided by\b", "/"),
                              (r"\btimes\b", "*"), (r"\bplus\b", "+"), (r"\bminus\b", "-")):
        text = re.sub(pattern, f" {operator} ", text, flags=re.I)

    def word_base(match):
        base = match.group(1).strip()
        prefix = text[:match.start()].rstrip()
        # Whitespace must not turn a binary minus into part of a signed base.
        if base[0] in "+-" and prefix and (prefix[-1].isdigit() or prefix[-1] == ")"):
            return base[0] + "(" + base[1:].strip() + ")"
        return "(" + base + ")"
    # English powers apply to their signed numeric base; symbolic ** keeps
    # conventional precedence, so -3 squared is (-3)**2 while -3**2 is -(3**2).
    text = re.sub(rf"(?<![\w.)])({SIGNED_NUMBER})\s+(squared|cubed)\b",
                  lambda m: word_base(m) + f"**{2 if m.group(2).lower() == 'squared' else 3}", text, flags=re.I)
    text = re.sub(rf"(?<![\w.)])({SIGNED_NUMBER})\s+to the power of\s+",
                  lambda m: word_base(m) + "**", text, flags=re.I)
    text = re.sub(rf"(?<![\w.)])({NUMBER})\s*(?:percent\s+of|%\s*of)\s*",
                  lambda m: f"({m.group(1)}/100)*", text, flags=re.I)
    words = ((r"\bto the power of\b", "**"), (r"\bsquared\b", "**2"), (r"\bcubed\b", "**3"))
    for pattern, operator in words:
        text = re.sub(pattern, f" {operator} ", text, flags=re.I)
    if not text or len(text) > MAX_EXPRESSION:
        raise ParseError("arithmetic_limit" if text else "unsupported_arithmetic")
    if not re.fullmatch(r"[0-9eE.+*/()\s-]+", text):
        raise ParseError("unsupported_arithmetic")
    # Parse BEFORE removing whitespace: '1 2' must not turn into '12'.
    _validate_arithmetic(text)
    return "".join(text.split())


def _mask_quotes(text):
    masked = list(text)
    pairs = {'"': '"', "“": "”", "'": "'", "‘": "’"}
    i = 0
    while i < len(text):
        char = text[i]
        if char not in pairs or (char in "'‘" and i and text[i - 1].isalnum()):
            i += 1
            continue
        end = i + 1
        while end < len(text):
            if text[end] == pairs[char]:
                if text[end] in "'’" and end + 1 < len(text) and text[end - 1].isalpha() and text[end + 1].isalpha():
                    end += 1
                    continue
                break
            end += 1
        if end == len(text):
            raise ParseError("ambiguous_location")
        masked[i:end + 1] = "~" * (end + 1 - i)
        i = end + 1
    return "".join(masked)


def _place_value(text, tool):
    quoted = False
    for left, right in (('"', '"'), ("“", "”"), ("'", "'"), ("‘", "’")):
        if text.startswith(left) and text.endswith(right) and len(text) > 2:
            text, quoted = text[1:-1].strip(), True
            break
    if not text or len(text) > 160:
        raise ParseError("ambiguous_location")
    if "/" in text or text in ("UTC", "GMT"):
        if tool != "get_time" or not re.fullmatch(r"[A-Za-z0-9_+.-]+(?:/[A-Za-z0-9_+.-]+)*", text):
            raise ParseError("invalid_timezone")
        try:
            ZoneInfo(text)
        except (ZoneInfoNotFoundError, ValueError):
            raise ParseError("invalid_timezone") from None
        return text
    if not any(c.isalpha() for c in text) or any(not (unicodedata.category(c)[0] in "LM" or c in " .,'’ʼ-‐‑") for c in text):
        raise ParseError("ambiguous_location")
    if text[0] in ".,'’-" or text.endswith((",", "-")) or re.search(r"[,.-]{2,}", text):
        raise ParseError("ambiguous_location")
    if not quoted:
        for abbreviation in re.finditer(r"([^\W\d_]+)\.(?=\s)", text):
            part = abbreviation.group(1).casefold()
            if len(part) > 1 and part not in {"st", "ste", "sta", "sto", "mt", "ft"}:
                raise ParseError("ambiguous_location")
        words = set(re.findall(r"[^\W\d_]+(?:['’ʼ‐‑-][^\W\d_]+)*", text.casefold()))
        forbidden = {
            "and", "or", "versus", "vs", "both", "here", "there", "nearby", "home", "somewhere", "anywhere",
            "my", "your", "our", "their", "this", "that", "these", "those", "a", "an", "me",
            "in", "for", "from", "to", "near", "with", "where", "which", "is", "are", "was", "be",
            "at", "on", "by", "during", "until", "after", "before", "current", "live",
            "please", "thanks", "show", "read", "display", "says", "weather", "temperature", "time",
            "forecast", "conditions", "currently", "today", "now", "tomorrow", "yesterday", "tonight",
            "morning", "afternoon", "evening", "noon", "midnight", "ago", "hour", "hours", "day", "days",
            "next", "last", "instead", "then", "also",
        }
        if words & forbidden or re.match(r"the (?:city|town|office|hotel|airport|station|store|area)\b", text, re.I):
            raise ParseError("ambiguous_location")
    return text


def parse_location(user, tool):
    if tool not in ("weather", "get_time"):
        raise ValueError(tool)
    text = request_text(user)
    masked = _mask_quotes(text)
    if re.search(r"\b(?:tomorrow|yesterday|tonight|next (?:week|month|year)|last (?:week|month|year)|in [0-9]+ (?:hours?|days?))\b", masked, re.I):
        raise ParseError("unsupported_time")
    if re.search(r"[;!?].*\w", masked) or ";" in masked:
        raise ParseError("ambiguous_location")
    text = _trim_polite(text)
    # Only a complete recognized temporal suffix may be removed.
    text = re.sub(rf"(?:,\s*|\s+){TEMPORAL}$", "", text, flags=re.I).rstrip()
    text = _trim_polite(text)
    masked = _mask_quotes(text)
    field = re.match(r"^(?:location|timezone)\s*[:=]\s*", masked, re.I)
    if field:
        return _place_value(text[field.end():].strip(), tool)
    slots = list(re.finditer(r"\b(?:in|for)\s+", masked, re.I))
    if not slots:
        raise ParseError("missing_location")
    for earlier, later in zip(slots, slots[1:]):
        between = masked[earlier.end():later.start()].strip()
        # 'a jacket for the weather in Denver' contains only one place slot.
        if not re.fullmatch(r"(?:the )?(?:(?:current|live|local) )*(?:weather|time|temperature|conditions)", between, re.I):
            raise ParseError("ambiguous_location")
    slot = slots[-1]
    place = text[slot.end():].strip()
    if re.search(r"\bclock\b", masked[:slot.start()], re.I):
        place = re.sub(r"\s+(?:show|read|display)$", "", place, flags=re.I).rstrip()
    return _place_value(place, tool)


def resolve_v3(case, routed_tool):
    keys = {"weather": "location", "get_time": "timezone", "calculator": "expression", "web_search": "query"}
    if routed_tool not in keys:
        raise ValueError(routed_tool)
    key = keys[routed_tool]
    try:
        user = case.get("user")
        if routed_tool == "calculator":
            value = parse_arithmetic(user)
        elif routed_tool in ("weather", "get_time"):
            value = parse_location(user, routed_tool)
        else:
            request_text(user)
            value = user.strip().rstrip("?").strip()
        return {"tool": routed_tool, "key": key, "value": value,
                "payload": {"name": routed_tool, "arguments": {key: value}}, "reason": None, "clarification": None}
    except ParseError as exc:
        return {"tool": routed_tool, "key": key, "value": None, "payload": None,
                "reason": exc.reason, "clarification": str(exc)}


def main():
    parser = argparse.ArgumentParser(description="Parse one tool argument without loading an Ember model")
    parser.add_argument("tool", choices=("calculator", "weather", "get_time", "web_search"))
    parser.add_argument("request")
    args = parser.parse_args()
    print(json.dumps(resolve_v3({"user": args.request}, args.tool), ensure_ascii=False))


if __name__ == "__main__":
    main()
