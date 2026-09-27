"""Writing Repair 2 training rows. Imports only WR1's pinned training retention rows.

No evaluation split is opened or exported by this module. The new writing
examples use training-only entities and structurally distinct source sentences.
"""
from collections import Counter
import hashlib
import json
import random

VERSION = "ember-writing-repair2-data-v1"
SEED = 431

# Source, shorter reference, semantic force, literal facts that must survive.
SHORT = [
    ("Please make sure that all {group} meet at the {place} before {time} on {day}.",
     "Please ensure all {group} meet at the {place} before {time} on {day}.", "request", ("group","place","time","day")),
    ("Please make sure all {group} gather at the {place} before {time}.",
     "Please ensure all {group} gather at the {place} before {time}.", "request", ("group","place","time")),
    ("I would appreciate it if you could ask the {group} to arrive at the {place} by {time}.",
     "Please ask the {group} to arrive at the {place} by {time}.", "request", ("group","place","time")),
    ("Would you please see to it that the {group} collect the {obj} at the {place} on {day}?",
     "Please ensure the {group} collect the {obj} at the {place} on {day}.", "request", ("group","obj","place","day")),
    ("It is a requirement that the {group} keep the {obj} inside cabinet {room} until {day}.",
     "The {group} must keep the {obj} inside cabinet {room} until {day}.", "requirement", ("group","obj","room","day")),
    ("Under the rule, the {group} are obligated to take the {obj} to the {place} by {time}.",
     "The {group} must take the {obj} to the {place} by {time}.", "requirement", ("group","obj","place","time")),
    ("The recommendation is that the {group} should bring the {obj} to the {place} on {day}.",
     "The {group} should bring the {obj} to the {place} on {day}.", "recommendation", ("group","obj","place","day")),
    ("For this task, it is advisable that the {group} should check cabinet {room} before {time}.",
     "The {group} should check cabinet {room} before {time}.", "recommendation", ("group","room","time")),
    ("The {group} are under no obligation to move the {obj} from the {place} before {day}.",
     "The {group} need not move the {obj} from the {place} before {day}.", "no_obligation", ("group","obj","place","day")),
    ("There is no requirement for the {group} to inspect the {obj} at the {place} on {day}.",
     "The {group} need not inspect the {obj} at the {place} on {day}.", "no_obligation", ("group","obj","place","day")),
    ("The {group} are prohibited from entering the {place} until {day}.",
     "The {group} must not enter the {place} until {day}.", "prohibition", ("group","place","day")),
    ("The {group} are forbidden to move the {obj} before {time}.",
     "The {group} must not move the {obj} before {time}.", "prohibition", ("group","obj","time")),
    ("The {group} have permission to collect the {obj} at the {place}, but only if the {role} is present.",
     "The {group} may collect the {obj} at the {place} only if the {role} is present.", "permission", ("group","obj","place","role")),
    ("Provided the {role} is present, the {group} are permitted to handle the {obj} at the {place}.",
     "The {group} may handle the {obj} at the {place} if the {role} is present.", "permission", ("role","group","obj","place")),
    ("It is possible that the {obj} will arrive at the {place} before {time} on {day}.",
     "The {obj} might arrive at the {place} before {time} on {day}.", "uncertainty", ("obj","place","time","day")),
    ("There is a chance that the {obj} could reach the {place} by {time} on {day}.",
     "The {obj} could reach the {place} by {time} on {day}.", "uncertainty", ("obj","place","time","day")),
    ("The {group} are required to bring a total of exactly {q} {items} to the {place} by {time}.",
     "The {group} must bring exactly {q} {items} to the {place} by {time}.", "requirement", ("group","q","items","place","time")),
    ("It is required that the {group} bring no fewer than {q} {items} to the {place} by {time}.",
     "The {group} must bring at least {q} {items} to the {place} by {time}.", "requirement", ("group","q","items","place","time")),
    ("It is required that the {group} bring no more than {q} {items} to the {place} by {time}.",
     "The {group} must bring at most {q} {items} to the {place} by {time}.", "requirement", ("group","q","items","place","time")),
    ("The {group} are required to store the {obj} inside cabinet {room} until {day}, unless the {role} approves removal.",
     "The {group} must store the {obj} inside cabinet {room} until {day}, unless the {role} approves removal.", "requirement", ("group","obj","room","day","role")),
]
WRAPPERS = (
    "Shorten this while keeping its meaning and all details: {s}",
    "Write a shorter version; keep who, what, when, and any condition: {s}",
    "Make the following sentence shorter without changing its force: {s}",
    "Condense this sentence faithfully. Return only the rewrite.\n{s}",
)
# Six paired frames; for each frame, one recipient-owned and one third-party-owned
# version uses the same entity slots. The named recipient is always the addressee.
RECIPIENT = (
    ("I picked up {pos} {obj}. It belongs to {owner}. Write a note to {n} saying I can return it on {day}.",
     "I picked up {ownership} and can return it on {day}."),
    ("My message is for {n}. I have the {obj} owned by {owner} and can bring it to the {place} on {day}. Draft it.",
     "I have {ownership} and can bring it to the {place} on {day}."),
    ("Write to {n}, the intended reader: I repaired the {obj} belonging to {owner}, and will return it before {time}.",
     "I repaired {ownership} and will return it before {time}."),
    ("Please draft words addressed to {n}. The {obj} belongs to {owner}. I might bring it to the {place} on {day} if the {role} agrees.",
     "I might bring {ownership} to the {place} on {day} if the {role} agrees."),
    ("The addressee is {n}, and the owner of the {obj} is {owner}. Write a draft saying I am keeping it inside cabinet {room} until {day}.",
     "I am keeping {ownership} inside cabinet {room} until {day}."),
    ("Prepare my reply to {n}, not a claim that it was sent: I have {owner}'s {obj} and cannot drop it at the {place} before {time}.",
     "I have {ownership} and cannot drop it at the {place} before {time}."),
)

def build_train_rows(wr1):
    rows = []
    for t, (source, answer, force, facts) in enumerate(SHORT):
        for j in range(8):
            v = wr1.values("train", t * 8 + j + 160)
            src, ans = source.format(**v), answer.format(**v)
            assert len(ans) < len(src), (t, src, ans)
            assert all(v[k].casefold() in src.casefold() and v[k].casefold() in ans.casefold() for k in facts)
            prompt = WRAPPERS[(t+j) % len(WRAPPERS)].format(s=src)
            rows.append({"id":f"wr2-train-shortening-{t:02}-{j:03}","prompt":prompt,"answer":ans,
                         "family":"shortening","source":src,"force":force})
    for pair in range(6):
        src_fmt, ans_fmt = RECIPIENT[pair]
        for j in range(8):
            v = wr1.values("train", pair * 8 + j + 336)
            for rel in ("recipient", "third_party"):
                owner = v["n"] if rel == "recipient" else v["owner"]
                w = {**v, "owner":owner, "pos":"their"}
                ownership = f"your {v['obj']}" if rel == "recipient" else f"{owner}'s {v['obj']}"
                prompt = src_fmt.format(**w)
                answer = f"Hi {v['n']}, " + ans_fmt.format(**{**w,"ownership":ownership})
                assert owner != v["n"] or rel == "recipient"
                assert (f"Hi {v['n']}, " in answer and ownership in answer)
                rows.append({"id":f"wr2-train-recipient-{pair:02}-{j:03}-{rel}",
                             "prompt":prompt,"answer":answer,"family":"recipient",
                             "recipient":v["n"],"owner":owner,"relation":rel})
    old = wr1.retention("train")
    assert len(old) == 256
    for r in old:
        rows.append({"id":"wr2-retain-"+r["id"],"prompt":r["prompt"],"answer":r["answer"],
                     "family":r["family"]})
    assert len(rows)==512
    assert Counter(r["family"] for r in rows)=={
        "shortening":160,"recipient":96,"arithmetic":64,"extraction":64,
        "grounding":48,"clarification":48,"direct":32}
    assert len({r["id"] for r in rows})==len({r["prompt"] for r in rows})==512
    random.Random(SEED+2).shuffle(rows)
    return rows

def sft_bytes(rows):
    """Only id/prompt/answer fields enter training; return deterministic bytes."""
    return ("".join(json.dumps({k:r[k] for k in ("id","prompt","answer")},
                    ensure_ascii=False,sort_keys=True)+"\n" for r in rows)).encode("utf-8")

def fingerprint(rows):
    return hashlib.sha256(sft_bytes(rows)).hexdigest()
