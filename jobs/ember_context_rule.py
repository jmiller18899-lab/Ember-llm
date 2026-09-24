"""Scoped runtime rule for record questions (context consistency). No model weights involved.

Evidence (jobs/ember_context_diag.py, Actions run 35952932842): under the frozen policy, all three
context_consistency misses are the item=folder records. There the model answers "Theo owns the item ..."
and writes the field name instead of its value. This rule is added to the system prompt only when the
user gives a field=value record and asks a question about it. Extraction prompts that ask to return a
single field ("Return only ...", "Give only ...") are excluded, so exact-answer behaviour there cannot change.
"""
import re

CONTEXT_RULE=("Record answers: when the user gives a record of field=value pairs and asks about it, answer with the "
"actual values from the record, never with the field names. Example: owner=Ana; item=scarf; return_day=Sunday -> "
"Ana owns the scarf, and it is returned Sunday. Never write 'the item', 'the owner' or any other field name in place "
"of its value, and do not change or add facts.")

PAIR=re.compile(r"\b[A-Za-z_][A-Za-z_ ]{0,20}?\s*=\s*[^;|,=\n]+")
QUESTION=re.compile(r"\b(who|whose|when|what|which|where)\b[^.!]*\?",re.I)
SINGLE_FIELD=re.compile(r"\b(return|give|output|reply with|extract)\s+(?:only|just)\b|\bonly the\b|\bjust the\b",re.I)
def context_gate(p):
    return len(PAIR.findall(p))>=2 and bool(QUESTION.search(p)) and not SINGLE_FIELD.search(p)
