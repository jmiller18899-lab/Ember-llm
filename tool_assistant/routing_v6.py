"""Opt-in contextual routing over the unchanged v5 text helper.

Only an entire named contact/visit context and an entire current-place
reference question can select a route. The existing parser must also accept
the whole request. No keywords, locations, fitted weights, or parser rules
are added here.
"""
from . import resolver_v3 as v3
from .resolver_v4 import PLACE_CONTEXT, REFERENCE_QUESTION, SEPARATOR, resolve_v4

REVISION = "context-routing-v6"


def contextual_route(user):
    text = v3.request_text(user)
    try:
        separators = list(SEPARATOR.finditer(v3._mask_quotes(text)))
        if len(separators) != 1:
            return None
        split = separators[0]
        context = v3._trim_polite(text[:split.start()].strip())
        question = v3._trim_polite(text[split.end():].strip())
        if not PLACE_CONTEXT.fullmatch(context):
            return None
        matches = [tool for tool, pattern in REFERENCE_QUESTION.items()
                   if pattern.fullmatch(question)
                   and resolve_v4({"user": user}, tool)["payload"] is not None]
        return matches[0] if len(matches) == 1 else None
    except v3.ParseError:
        return None
