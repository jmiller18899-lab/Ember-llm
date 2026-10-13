"""WR13 preparation: replace weak shortening examples; no training launch.

Keep the other 320 WR12 training rows and the existing loss interface. New
shortening references save at least three words while preserving reminders,
modal force, quantities, ownership, conditions and timing. The frozen v3
checker is unchanged. Development sources are never training examples.
"""
from collections import Counter
import re

import ember_writing_repair12_data as W

VERSION = "ember-writing-repair13-clear-shortening-v1"
WRAPPERS = W.WRAPPERS
DEV_WRAPPERS = W.DEV_WRAPPERS

# Singular nouns are separate from groups; never format 'each' with a plural.
TRAIN_WORKERS = ("trainee", "baker", "cashier", "porter", "custodian", "tutor",
                 "gardener", "installer", "electrician", "receptionist", "technician", "repairer")
DEV_WORKERS = ("assistant", "cleaner", "driver", "instructor")
TRAIN_NAMES = ("Milo", "Nadia", "Felix", "Anika", "Jonas", "Leila", "Ronan", "Priya", "Tessa", "Omar", "Heidi", "Dante")
DEV_NAMES = ("Amina", "Diego", "Keiko", "Ravi")
TRAIN_OBJECTS = ("rental toolkit", "bicycle pump", "visitor register", "camera tripod",
                 "portable radio", "cable reel", "lunch cooler", "safety vest",
                 "travel bag", "repair manual", "document pouch", "battery charger")
DEV_OBJECTS = ("sample tray", "floor mat", "route map", "rain jacket")
TRAIN_PLACES = ("supply office", "dispatch booth", "locker area", "repair shed",
                "ticket booth", "packing room", "crew cabin", "break room",
                "service counter", "linen closet", "receiving area", "side office")
DEV_PLACES = ("print room", "driver lounge", "sorting cabin", "canteen")
TRAIN_EVENTS = ("shift ends", "stock check ends", "delivery arrives", "inspection starts")
DEV_EVENTS = ("class ends", "bus arrives", "practice starts", "rain stops")

PREFERRED = [
    ("timing_should", "Please note that each {worker} should check the {obj} prior to {time}.",
     "Each {worker} should check the {obj} before {time}.", "check", ("should", "before")),
    ("consent_unless", "No one may borrow {owner}'s {obj} unless the {role} has given consent.",
     "No one may borrow {owner}'s {obj} without {role} consent.", "borrow", ("No one may", "without")),
    ("reminder", "Please remember that each {worker} is required to keep the {obj} in the {place} until the {event}.",
     "Please remember each {worker} must keep the {obj} in the {place} until the {event}.", "keep", ("Please remember", "must", "until")),
    ("consent_only_if", "Please note that you may collect the {obj} only if the {role} has given consent.",
     "You may collect the {obj} only if the {role} has consented.", "collect", ("may", "only if")),
    ("reminder_forget", "Do not forget that {owner} is required to bring the {obj} to the {place} prior to {time}.",
     "Do not forget {owner} must bring the {obj} to the {place} before {time}.", "bring", ("Do not forget", "must", "before")),
    ("exactly", "Please note that each {worker} must leave exactly {q} {items} in the {place} until the {event}.",
     "Each {worker} must leave exactly {q} {items} in the {place} until the {event}.", "leave", ("must", "exactly", "until")),
    ("ownership", "Please note that you should deliver {owner}'s {obj} to the {place} prior to the {event}.",
     "You should deliver {owner}'s {obj} to the {place} before the {event}.", "deliver", ("should", "before")),
    ("only_after", "Please note that you may inspect the {obj} only after {time} at the {place}.",
     "You may inspect the {obj} only after {time} at the {place}.", "inspect", ("may", "only after")),
]
DEVELOPMENT = [
    ("timing_should", "Please note that every {worker} should test the {obj} prior to {time}.",
     "Every {worker} should test the {obj} before {time}.", "test", ("should", "before")),
    ("consent_unless", "No one may move {owner}'s {obj} unless the {role} has given consent.",
     "No one may move {owner}'s {obj} without {role} consent.", "move", ("No one may", "without")),
    ("reminder", "Please remember that every {worker} is required to hold the {obj} in the {place} until the {event}.",
     "Please remember every {worker} must hold the {obj} in the {place} until the {event}.", "hold", ("Please remember", "must", "until")),
    ("consent_only_if", "Please note that you may take the {obj} only if the {role} has given consent.",
     "You may take the {obj} only if the {role} has consented.", "take", ("may", "only if")),
    ("reminder_forget", "Do not forget that {owner} is required to carry the {obj} to the {place} prior to {time}.",
     "Do not forget {owner} must carry the {obj} to the {place} before {time}.", "carry", ("Do not forget", "must", "before")),
    ("exactly", "Please note that every {worker} must retain exactly {q} {items} in the {place} until the {event}.",
     "Every {worker} must retain exactly {q} {items} in the {place} until the {event}.", "retain", ("must", "exactly", "until")),
    ("ownership", "Please note that you should return {owner}'s {obj} to the {place} prior to the {event}.",
     "You should return {owner}'s {obj} to the {place} before the {event}.", "return", ("should", "before")),
    ("only_after", "Please note that you may pack the {obj} only after {time} at the {place}.",
     "You may pack the {obj} only after {time} at the {place}.", "pack", ("may", "only after")),
]


def values(split, i):
    train = split == "train"
    banks = (TRAIN_WORKERS, TRAIN_NAMES, TRAIN_OBJECTS, TRAIN_PLACES, TRAIN_EVENTS) if train else (
        DEV_WORKERS, DEV_NAMES, DEV_OBJECTS, DEV_PLACES, DEV_EVENTS)
    worker, owner, obj, place, event = (bank[i % len(bank)] for bank in banks)
    return dict(worker=worker, owner=owner, obj=obj, place=place, event=event,
                role=("manager", "planner", "librarian", "mechanic")[i % 4] if train else (
                    "coach", "supervisor", "organizer", "inspector")[i % 4],
                q=str(2 + i % 7), items=("forms", "copies", "envelopes", "receipts")[i % 4],
                time=f"{2 + i % 8}:{(7 + i * 13) % 60:02d} PM")


def writing(split):
    if split not in ("train", "dev"):
        raise ValueError("Only train and development splits are supported")
    templates = PREFERRED if split == "train" else DEVELOPMENT
    count = 12 if split == "train" else 4
    rows = []
    for t, (group, source, answer, verb, anchors) in enumerate(templates):
        for j in range(count):
            v = values(split, j)
            src, ans = source.format(**v), answer.format(**v)
            row = dict(id=f"wr13-{split}-short-{group}-{j:02}", split=split,
                       training_allowed=split == "train", family="shortening", shortening_group=group,
                       source=src, answer=ans, structure=source, action_verb=verb,
                       prompt=(WRAPPERS if split == "train" else DEV_WRAPPERS)[(t + j) % 4].format(s=src),
                       protected=W.protected_fields(answer, v) + list(anchors), semantic_review_required=True,
                       loss="copy_margin" if split == "train" else "eval")
            if split == "train":
                row["rejected"] = src
                if group in ("reminder", "reminder_forget"):
                    row["overedit"] = re.sub(r"^(?:Please remember |Do not forget )", "", ans)
                elif group == "consent_unless":
                    row["overedit"] = ans.replace(" without ", " with ")
                else:
                    row["overedit"] = W.overedit_for(row)
            W.validate_reference(row)
            if len(src.split()) - len(ans.split()) < 3 or len(ans) / len(src) > .90:
                raise ValueError("Preferred shortening is too slight: " + row["id"])
            if re.search(r"\b(?:each|every) (?:" + "|".join(re.escape(x + "s") for x in TRAIN_WORKERS + DEV_WORKERS) + r")\b", src):
                raise ValueError("Singular quantifier has a plural subject")
            rows.append(row)
    return rows


def build_train_rows(retention):
    parent = W.build_train_rows(retention)
    replacements = iter(writing("train"))
    rows = [next(replacements) if r["family"] == "shortening" else dict(r) for r in parent]
    W.sft_bytes(rows)
    return rows


def audit(train, dev, historical, benchmark_sources):
    short = [r for r in train if r["family"] == "shortening"]
    expected = {"shortening": 96, "recipient": 64, "clarification": 48, "arithmetic": 64,
                "extraction": 64, "direct": 32, "grounding": 48}
    if Counter(r["family"] for r in train) != expected:
        raise ValueError("WR13 training composition changed")
    # Retained families still participate in train/dev isolation. Only their
    # intentional overlap with historical training is excluded from novelty.
    checked_train = [r if r["family"] == "shortening" else dict(r, id="wr12-retain-" + r["id"])
                     for r in train]
    result = W.audit(checked_train, dev, historical, benchmark_sources)
    result.update(train=len(train), encoded_sequences=len(train), unchanged_nonshortening_rows=320,
                  intentional_nonshortening_training_reuse=True,
                  minimum_words_saved=min(len(r["source"].split()) - len(r["answer"].split()) for r in short),
                  maximum_reference_length_ratio=round(max(len(r["answer"]) / len(r["source"]) for r in short), 6),
                  scope="New shortening sources and development audited; other training families intentionally retained. Reserved final holdouts unloaded.")
    return result
