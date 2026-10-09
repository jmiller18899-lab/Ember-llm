"""Meaning-preservation-v3: experimental review checker, not a runtime gate.

v2 remains the frozen historical grader. This follow-on only normalizes the
specific equivalent timing and consent constructions identified by the
2026-10-09 probe, keeps reminder/obligation/ownership/negation/quantity
checks, and still abstains on unrecognized paraphrases.
"""
from __future__ import annotations
import re
import ember_drafting_repair_candidates_eval as v2

GRADER_VERSION = "meaning-preservation-v3"
_REMINDER_PLACEHOLDERS = (
    (re.compile(r"\bdo not forget\b|\bdon't forget\b"), " __reminder_forget__ "),
    (re.compile(r"\bplease remember\b"), " __reminder_remember__ "),
    (re.compile(r"\bplease recall\b"), " __reminder_recall__ "),
)
_REMINDER_TOKEN_RE = re.compile(r"__(?:reminder_forget|reminder_remember|reminder_recall)__")
_CONSENT_ROLE = r"(?:the )?([a-z]+)"


def _reminders(text):
    t = v2.norm(text)
    found = []
    if re.search(r"\bdo not forget\b|\bdon't forget\b", t):
        found.append("forget")
    if re.search(r"\bplease remember\b|\bremember\b", t):
        found.append("remember")
    if re.search(r"\bplease recall\b|\brecall\b", t):
        found.append("recall")
    return found


def _canonicalize(text):
    t = v2.norm(text).replace("−", "-")
    t = re.sub(r"\bprior to\b", "before", t)
    t = re.sub(r"\bhas given consent\b", "has consented", t)
    t = re.sub(r"\bhave given consent\b", "have consented", t)
    t = re.sub(r"\bgiven consent\b", "consented", t)
    t = re.sub(rf"\bunless {_CONSENT_ROLE}(?:'s)? has consented\b", r"without \1 consent", t)
    t = re.sub(rf"\bunless {_CONSENT_ROLE} consents\b", r"without \1 consent", t)
    t = re.sub(rf"\bwithout {_CONSENT_ROLE}'s consent\b", r"without \1 consent", t)
    t = re.sub(rf"\bwithout {_CONSENT_ROLE} consent\b", r"without \1 consent", t)
    for pattern, token in _REMINDER_PLACEHOLDERS:
        t = pattern.sub(token, t)
    t = re.sub(r"^please note(?: that)?\s+", "", t)
    t = re.sub(r"^it is important that\s+", "", t)
    t = re.sub(r"\bplease (?:make sure(?: that)?|ensure)\s+", "please ", t)
    t = re.sub(r"^we would appreciate it if (.+?) (?:could|would)\s+", r"please \1 ", t)
    t = re.sub(r"\bremember that\b", "remember", t)
    t = re.sub(r"\brecall that\b", "recall", t)
    return t


def _meaning_view(text):
    text = _canonicalize(text)
    forces = []

    def remove_force(match):
        kind = v2._FORCE_PATTERNS[int(match.lastgroup[1:])][0]
        forces.append(kind)
        if kind == "uncertainty":
            return " " + match.group() + " "
        return " __negative_subject__ " if match.group() in ("nobody", "no one") else " "

    body = v2._FORCE_RE.sub(remove_force, text)
    distinct = set(forces)
    if len(distinct) > 1 and "request" in distinct and not re.search(r"\basked to\b", text):
        distinct.remove("request")
    force = next(iter(distinct)) if len(distinct) == 1 else ("statement" if not distinct else "mixed")
    negation = len(re.findall(r"\b(?:not|never|neither|without)\b", body))
    qualifiers = re.findall(r"\b(?:exactly|at least|at most|only|all|each|every)\b", body)
    conditions = re.findall(r"\b(?:only if|if|unless|provided that|as long as)\b", body)
    timing = re.findall(r"\b(?:no later than|before|after|until|ahead of)\b", body)
    numbers = re.findall(r"(?<![\w.])[+-]?\d+(?:[.:]\d+)*(?![\w.])", body)
    body = re.sub(rf"\bon (?={v2._DAYS}\b)", "", body)
    tokens = [t for t in v2._TOKEN_RE.findall(body) if t not in {"a", "an", "the", "that"}]
    return {"force": force, "negation": negation, "qualifiers": qualifiers,
            "conditions": conditions, "timing": timing, "numbers": numbers,
            "tokens": tokens, "reminders": _REMINDER_TOKEN_RE.findall(text)}


def grade_case(row, output, legacy_scorer):
    """Bounded v3 shortening check. review is not a pass. Not a product gate."""
    if not isinstance(output, str) or not output.strip():
        return {"passed": False, "legacy_passed": False, "status": "fail",
                "reasons": ["empty_or_invalid_output"], "grader_version": GRADER_VERSION}
    legacy = bool(legacy_scorer(row, output))
    result = {"passed": legacy, "legacy_passed": legacy,
              "status": "pass" if legacy else "fail",
              "reasons": [] if legacy else ["legacy_requirements_failed"],
              "grader_version": GRADER_VERSION}
    if row.get("scoring") == "exact" or row.get("kind") not in ("shorten", "shortening"):
        return result
    source = row.get("source")
    if not isinstance(source, str) or not source.strip():
        return {**result, "passed": False, "status": "fail", "reasons": ["missing_source"]}
    before, after = _meaning_view(source), _meaning_view(output)
    reasons = list(result["reasons"])
    if len(v2.norm(output)) >= len(v2.norm(source)):
        reasons.append("not_shorter")
    src_reminders, out_reminders = _reminders(source), _reminders(output)
    if src_reminders and not out_reminders:
        reasons.append("reminder_dropped")
    if before["force"] != after["force"] and "mixed" not in (before["force"], after["force"]):
        reasons.append("force_changed:" + before["force"] + "->" + after["force"])
    for key in ("negation", "qualifiers", "conditions", "timing", "numbers", "reminders"):
        if before[key] != after[key]:
            reasons.append(key + "_changed")
    if reasons:
        return {**result, "passed": False, "status": "fail", "reasons": reasons}
    if before["tokens"] != after["tokens"]:
        return {**result, "passed": False, "status": "review",
                "reasons": ["unverified_paraphrase_or_scope"]}
    if before["force"] != after["force"]:
        return {**result, "passed": False, "status": "review",
                "reasons": ["unverified_paraphrase_or_scope"]}
    return {**result, "passed": True, "status": "pass", "reasons": [],
            "grader_version": GRADER_VERSION}


def accept(row, output):
    return grade_case(row, output, lambda r, o: True)["status"] == "pass"
