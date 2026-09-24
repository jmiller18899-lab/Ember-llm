"""Policy v3 candidate rules (no model weights involved). v2's rule files are unchanged; this module only adds:

  drafting_gate_v3   v2's drafting gate, plus "Text/Tell X that ...", "Let X know ...", "Write a note to X" requests.
                     The drafting rule text is v2's, unchanged.
  CONTEXT_RULE_V3    v2's context rule plus one sentence for "whose item" questions. Evidence: v3 baseline context-00/18
                     answered "Nadia's item ...", dropping the item.
  missing_text_gate / MISSING_TEXT_RULE
                     A transformation request ("translate/fix/summarize ... this/my X") that includes no text.
                     Evidence: v3 baseline clarification-00/05, where the request itself was translated.
"""
import re

MESSAGE_V3=re.compile(r"^\s*(?:(?i:text|tell|remind|ping)\s+[A-Z][a-z]+\s+(?i:that|about|to)\b|(?i:let)\s+[A-Z][a-z]+\s+(?i:know)\b|(?i:write|send|draft)\s+(?i:a\s+)?(?i:quick\s+|short\s+)?(?i:note|text|message|reply)\s+(?i:to)\s+[A-Z])")
def drafting_gate_v3(prompt,v2_drafting_gate):
    return bool(v2_drafting_gate(prompt) or MESSAGE_V3.search(prompt))

WHOSE_SENTENCE=(" If asked whose item something is, name both the owner and the item itself, for example 'It is Ana's scarf', "
"never just 'Ana's item'.")
def context_rule_v3(v2_context_rule): return v2_context_rule+WHOSE_SENTENCE

MISSING_TEXT_RULE=("Missing text: if the user asks you to transform some text (translate, fix, proofread, rewrite, summarize, "
"shorten, simplify or reformat it) but the message does not include that text, do not transform the request itself. "
"Ask them to paste or share the text.")
TEXT_NOUN=r"(?:text|paragraph|email|essay|message|notes?|bio|draft|article|letter|cover letter|resume|post|sentence|intro|summary|report|story|speech|caption|tweet|review)"
TRANSFORM=re.compile(r"\b(translat\w*|proofread|rewrite|re-write|reword|summari[sz]e|shorten|simplify|polish|paraphrase|condense|edit)\b"
    r"|\bfix\b[^.?!]{0,12}\b(?:grammar|spelling|typos?|wording|punctuation)\b"
    r"|\bmake\b[^.?!]{0,25}\b(?:sound|more|less|easier|shorter|clearer|punchier)\b|\bcut\b[^.?!]{0,20}\bdown\b|\bturn\b[^.?!]{0,25}\binto\b",re.I)
DEICTIC=re.compile(rf"\b(?:this|that|it|these|those)\b|\b(?:my|the|our)\s+{TEXT_NOUN}\b",re.I)
SUPPLIED=re.compile(r"[\"“‘]|(?:^|\s)'|:\s*\S+(?:\s+\S+){2,}")   # opening quotes, or "label: several words"
def missing_text_gate(prompt):
    return len(prompt)<=160 and bool(TRANSFORM.search(prompt) and DEICTIC.search(prompt)) and not SUPPLIED.search(prompt)
