"""Opt-in support for a bounded context clause and one current-place question.

V3's whole-expression arithmetic and ordinary location parsing are retained.
Only complete, recognized context frames may be separated from the question.
"""
from __future__ import annotations

import re
from . import resolver_v3 as v3

REVISION = "argument-parser-v4"
PERSON = r"(?:i am|i'm|i’m|we are|we're|we’re)"
NEUTRAL_CONTEXT = re.compile(
    rf"{PERSON}\s+(?:about to\s+)?(?:"
    r"(?:step|stepping|go|going|head|heading)\s+(?:outside|out)|"
    r"(?:pack|packing|bring|bringing|take|taking)\s+(?:a|an|my|our|the)\s+(?:jacket|coat|umbrella)|"
    r"getting ready to (?:go|head) (?:outside|out))", re.I)
PLACE_CONTEXT = re.compile(
    rf"{PERSON}\s+(?:(?:about to|planning to|going to)\s+)?"
    r"(?:call|calling|phone|phoning|contact|contacting|visit|visiting)\s+"
    r"(?:someone|a friend|my friend|a colleague|my colleague|a relative|my family|a customer)\s+"
    r"in\s+(?P<place>.+)", re.I)
SEPARATOR = re.compile(r";|\.(?=\s+(?:what|how|is|can|could|would|please|check|tell|give|show|do)\b)", re.I)
CURRENT = r"(?:\s+(?:right now|now|at the moment|at this moment|currently|today))?"
REFERENCE_QUESTION = {
    "get_time": re.compile(
        rf"(?:what time is it there|what (?:is|would be) the (?:current |local )?time there|"
        rf"what's the (?:current |local )?time there|is it (?:morning|afternoon|evening) there){CURRENT}", re.I),
    "weather": re.compile(
        rf"(?:what is the (?:current )?(?:weather|temperature) there|what's the (?:current )?(?:weather|temperature) there|"
        rf"is it (?:raining|snowing|wet|warm|cold|windy|sunny|cloudy) there){CURRENT}", re.I),
}
QUESTION_CUES = {
    "weather": re.compile(r"\b(?:weather|temperature|rain|raining|snow|snowing|wet|damp|warm|cold|hot|chilly|windy|sunny|cloudy|humidity)\b", re.I),
    "get_time": re.compile(r"\b(?:time|clock|hour|morning|afternoon|evening)\b", re.I),
}
QUESTION_START = re.compile(
    r"^(?:(?:please|kindly)\s+)?(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
    r"(?:what|what's|what’s|how|is|are|do|does|check|tell|show|give|read|get)\b", re.I)


def parse_location(user, tool):
    if tool not in ("weather", "get_time"):
        raise ValueError(tool)
    text = v3.request_text(user)
    masked = v3._mask_quotes(text)
    separators = list(SEPARATOR.finditer(masked))
    if not separators:
        return v3.parse_location(user, tool)
    # Preserve v3's invalid-request/future-time rejections for the whole input.
    # A recognized boundary must pass the complete context grammar even if v3
    # could obtain a name by ignoring a period in an earlier clause.
    try:
        v3.parse_location(user, tool)
    except v3.ParseError as exc:
        if exc.reason != "ambiguous_location":
            raise
    if len(separators) != 1:
        raise v3.ParseError("ambiguous_location")
    split = separators[0]
    context = v3._trim_polite(text[:split.start()].strip())
    question = text[split.end():].strip()
    if not context or not question:
        raise v3.ParseError("ambiguous_location")
    if NEUTRAL_CONTEXT.fullmatch(context):
        # The prefix cannot contain a second location or an extra instruction.
        # The remaining request must still pass v3's complete location parser.
        if not QUESTION_START.search(question) or not QUESTION_CUES[tool].search(v3._mask_quotes(question)):
            raise v3.ParseError("ambiguous_location")
        return v3.parse_location(question, tool)
    antecedent = PLACE_CONTEXT.fullmatch(context)
    if antecedent and REFERENCE_QUESTION[tool].fullmatch(v3._trim_polite(question)):
        # Resolve 'there' only against one validated, explicitly named place.
        # Do not choose between two names or ignore an explicit second location.
        return v3._place_value(antecedent.group("place").strip(), tool)
    raise v3.ParseError("ambiguous_location")


def resolve_v4(case, routed_tool):
    if routed_tool not in ("weather", "get_time"):
        return v3.resolve_v3(case, routed_tool)
    key = "location" if routed_tool == "weather" else "timezone"
    try:
        value = parse_location(case.get("user"), routed_tool)
        return {"tool": routed_tool, "key": key, "value": value,
                "payload": {"name": routed_tool, "arguments": {key: value}},
                "reason": None, "clarification": None}
    except v3.ParseError as exc:
        return {"tool": routed_tool, "key": key, "value": None, "payload": None,
                "reason": exc.reason, "clarification": str(exc)}
