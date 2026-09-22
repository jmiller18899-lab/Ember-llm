"""Recognize short definition questions without naming particular subjects."""
import re

from . import resolver_v3 as v3
from .routing_v7 import ADDITIONAL_REQUEST, CURRENT_INFORMATION

REVISION = "definition-routing-v8"
QUESTION = re.compile(
    r"(?:(?:please|kindly)\s+)?(?:what (?:is|are)|what['’]s)\s+(?P<term>.+)", re.I)
WORD = r"[^\W\d_]+(?:[-'’][^\W\d_]+)*"
TERM = re.compile(rf"{WORD}(?:\s+{WORD}){{0,7}}")
# A noun-shaped suffix alone cannot distinguish definitions from tool requests.
# Abstain on arithmetic, service readings, changing facts, and extra clauses.
RESERVED = frozenset("""
    plus minus times multiplied divided squared cubed power powers percent percentage
    sum product quotient remainder factorial square cube root calculate compute evaluate
    zero one two three four five six seven eight nine ten eleven twelve hundred thousand
    weather temperature forecast conditions humidity rain raining snow snowing wind windy
    time clock date timezone utc gmt
    price prices pricing cost costs value worth stock stocks market exchange rate rates
    news headline headlines score scores results schedule timetable hours opening closing
    availability available cheapest best president ceo mayor governor leader version release
    updates update trending happening winning doing
    and or but then instead also in at on near from to for with during until after before
    here there nearby this that these those it they he she we you my our their
    search find check tell show give send delete
""".split())


def definition_question(user):
    text = v3.request_text(user)
    try:
        text = v3._trim_polite(text)
        masked = v3._mask_quotes(text)
    except v3.ParseError:
        return False
    if ADDITIONAL_REQUEST.search(masked):
        return False
    match = QUESTION.fullmatch(text)
    if not match:
        return False
    term = match["term"]
    for left, right in (("\"", "\""), ("“", "”"), ("'", "'"), ("‘", "’"), ("`", "`")):
        if term.startswith(left) and term.endswith(right):
            term = term[1:-1].strip()
            break
    if not TERM.fullmatch(term) or CURRENT_INFORMATION.search(term):
        return False
    words = set(re.findall(r"[^\W\d_]+", term.casefold()))
    return bool(words - {"a", "an", "the", "of"}) and not words & RESERVED
