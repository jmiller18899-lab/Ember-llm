"""Bounded request framing before the unchanged contextual/text fallback."""
import re

from . import resolver_v3 as v3

REVISION = "request-routing-v7"
PREFIX = r"(?:(?:please|kindly)\s+)?(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
SEARCH = re.compile(
    PREFIX + r"(?:search (?:the (?:web|internet) for|online for)|"
    r"look (?:online|on the web|on the internet) for)\s+(?P<query>.+)", re.I)
EXPLANATION = re.compile(
    r"(?:(?:please|kindly)\s+)?(?:"
    r"why (?:do|does|did|is|are|was|were|can|could|would|will)\s+.+|"
    r"how (?:do|does)\s+.+\s+(?:work|function|operate|help|reduce|increase|affect|cause|"
    r"change|form|grow|produce|generate|transfer|absorb|release|store|convert|regulate|"
    r"protect|prevent|differ|interact|keep|enable)\b.*|"
    r"what (?:is|are) the (?:role|purpose|function|meaning) of\s+.+|"
    r"what (?:is|are) the (?:difference|relationship) between\s+.+|"
    r"what is the reason (?:for|behind)\s+.+)", re.I)
CURRENT_INFORMATION = re.compile(
    r"\b(?:current(?:ly)?|latest|live|recent(?:ly)?|today|tonight|tomorrow|yesterday|now|"
    r"this (?:week|month|year)|last (?:night|week|month|year)|next (?:week|month|year))\b", re.I)
ADDITIONAL_REQUEST = re.compile(
    r";|[!?].*\w|\.\s+\w|\b(?:and|then|but|instead)\s+(?:then\s+)?"
    r"(?:please\s+)?(?:search|look|find|calculate|compute|evaluate|check|tell|show|give|send|delete)\b", re.I)
CLOCK_READING = re.compile(r"\bclock\b.*\b(?:read|show|display)\b", re.I)


def request_frame(user):
    text = v3.request_text(user)
    try:
        text = v3._trim_polite(text)
        masked = v3._mask_quotes(text)
    except v3.ParseError:
        return None
    if ADDITIONAL_REQUEST.search(masked):
        return None
    search = SEARCH.fullmatch(masked)
    if search and any(c.isalnum() for c in text[search.start("query"):search.end("query")]):
        return "web_search", "explicit_search_request"
    if (EXPLANATION.fullmatch(masked) and not CURRENT_INFORMATION.search(masked)
            and not CLOCK_READING.search(masked)):
        return "direct", "explanation_question"
    return None
