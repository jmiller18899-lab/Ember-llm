"""Preserve binary signs between English powers; retain prior parser limits."""
import re

from . import resolver_v3 as v3
from .resolver_v4 import resolve_v4

REVISION = "argument-parser-v7"


def parse_arithmetic(user):
    # This is the frozen v3 arithmetic grammar, with one corrected distinction:
    # a preceding word power is an operand, just like a digit or closing bracket.
    text = v3._expression_body(user)
    text = text.translate(str.maketrans({"−": "-", "×": "*", "÷": "/"}))
    if re.search(r"\b(?:squared|cubed)\s*(?:\*\*|squared\b|cubed\b|to the power of\b)", text, re.I):
        raise v3.ParseError("unsupported_arithmetic")
    text = re.sub(r"(?<![\w.,])[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?(?![\w.,])",
                  lambda m: m.group().replace(",", ""), text)
    for pattern, operator in ((r"\bmultiplied by\b", "*"), (r"\bdivided by\b", "/"),
                              (r"\btimes\b", "*"), (r"\bplus\b", "+"), (r"\bminus\b", "-")):
        text = re.sub(pattern, f" {operator} ", text, flags=re.I)

    def word_base(match):
        base = match.group(1).strip()
        prefix = text[:match.start()].rstrip()
        prior_operand = prefix and (prefix[-1].isdigit() or prefix[-1] == ")"
                                    or re.search(r"\b(?:squared|cubed)$", prefix, re.I))
        if base[0] in "+-" and prior_operand:
            return base[0] + "(" + base[1:].strip() + ")"
        return "(" + base + ")"

    text = re.sub(rf"(?<![\w.)])({v3.SIGNED_NUMBER})\s+(squared|cubed)\b",
                  lambda m: word_base(m) + f"**{2 if m.group(2).lower() == 'squared' else 3}", text, flags=re.I)
    text = re.sub(rf"(?<![\w.)])({v3.SIGNED_NUMBER})\s+to the power of\s+",
                  lambda m: word_base(m) + "**", text, flags=re.I)
    text = re.sub(rf"(?<![\w.)])({v3.NUMBER})\s*(?:percent\s+of|%\s*of)\s*",
                  lambda m: f"({m.group(1)}/100)*", text, flags=re.I)
    for pattern, operator in ((r"\bto the power of\b", "**"), (r"\bsquared\b", "**2"), (r"\bcubed\b", "**3")):
        text = re.sub(pattern, f" {operator} ", text, flags=re.I)
    if not text or len(text) > v3.MAX_EXPRESSION:
        raise v3.ParseError("arithmetic_limit" if text else "unsupported_arithmetic")
    if not re.fullmatch(r"[0-9eE.+*/()\s-]+", text):
        raise v3.ParseError("unsupported_arithmetic")
    v3._validate_arithmetic(text)
    return "".join(text.split())


def resolve_v7(case, routed_tool):
    if routed_tool != "calculator":
        return resolve_v4(case, routed_tool)
    try:
        value = parse_arithmetic(case.get("user"))
        return {"tool": routed_tool, "key": "expression", "value": value,
                "payload": {"name": routed_tool, "arguments": {"expression": value}},
                "reason": None, "clarification": None}
    except v3.ParseError as exc:
        return {"tool": routed_tool, "key": "expression", "value": None, "payload": None,
                "reason": exc.reason, "clarification": str(exc)}
