"""Bounded additional contact descriptions; retain frozen v8 as fallback."""
import re
from . import resolver_v3 as v3
from .resolver_v4 import PERSON, SEPARATOR, REFERENCE_QUESTION

REVISION = 'contact-routing-v9'
CONTACT = re.compile(
    rf"{PERSON}\s+(?:(?:about to|planning to|going to)\s+)?"
    r"(?:call|calling|phone|phoning|contact|contacting|visit|visiting)\s+"
    r"(?:a|my|our|the)\s+(?:cousin|client)\s+in\s+(?P<place>.+)", re.I)


def contact_call(user):
    text = v3.request_text(user)
    splits = list(SEPARATOR.finditer(v3._mask_quotes(text)))
    if len(splits) != 1:
        return None
    split = splits[0]
    context = v3._trim_polite(text[:split.start()].strip())
    question = v3._trim_polite(text[split.end():].strip())
    match = CONTACT.fullmatch(context)
    if not match:
        return None
    tools = [tool for tool, pattern in REFERENCE_QUESTION.items() if pattern.fullmatch(question)]
    if len(tools) != 1:
        return None
    tool = tools[0]
    try:
        # Keep whole-input rejections such as future requests and malformed text.
        try:
            v3.parse_location(user, tool)
        except v3.ParseError as exc:
            if exc.reason != 'ambiguous_location':
                raise
        value = v3._place_value(match.group('place').strip(), tool)
    except v3.ParseError:
        return None
    key = 'location' if tool == 'weather' else 'timezone'
    return {'name': tool, 'arguments': {key: value}}
