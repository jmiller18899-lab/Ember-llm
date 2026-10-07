"""Authored WR6 hedge-removal and addressee targets; generation grades stay manual."""
from collections import Counter
from string import Formatter
import json, random, re

VERSION = "ember-writing-repair6-data-v1"
SEED = 431
WRAPPERS = [
    'Shorten this without changing its meaning: "{s}"',
    "Give a briefer version, keeping the roles and conditions:\n{s}",
    "Make this sentence shorter without changing its force: {s}",
    'Trim this, keeping the facts, ownership and timing: "{s}"',
]
HEDGE = [
    ("Please see that the {group} meet at the {place} before {time}.",
     "Please meet at the {place} before {time}.", "meet", ("Please", "before")),
    ("Please confirm the {group} wait at the {place} until {day}.",
     "Please wait at the {place} until {day}.", "wait", ("Please", "until")),
    ("Please check that you bring my {obj} over to the {place} on {day}.",
     "Please bring my {obj} to the {place} on {day}.", "bring", ("Please", "my")),
    ("Please be certain the {group} return to the {place} after {time}.",
     "Please return to the {place} after {time}.", "return", ("Please", "after")),
    ("Please verify you inspect all {q} {items} on {day}.",
     "Please inspect all {q} {items} on {day}.", "inspect", ("Please", "all")),
    ("Please remember to keep {owner}'s {obj} stored in the {place} until {day}.",
     "Please keep {owner}'s {obj} in the {place} until {day}.", "keep", ("Please", "until")),
    ("Please confirm that you leave exactly {q} {items} waiting at the {place}.",
     "Please leave exactly {q} {items} at the {place}.", "leave", ("Please", "exactly")),
    ("Please see that you place {owner}'s {obj} back in the {place} on {day}.",
     "Please place {owner}'s {obj} in the {place} on {day}.", "place", ("Please",)),
    ("Please check the {group} stay at the {place} until {time}.",
     "Please stay at the {place} until {time}.", "stay", ("Please", "until")),
    ("Please confirm that you collect at most {q} {items} from inside the {place}.",
     "Please collect at most {q} {items} from the {place}.", "collect", ("Please", "at most")),
    ("Please remember to tell the {group} that they should wait at the {place} after {time}.",
     "Please tell the {group} to wait at the {place} after {time}.", "tell", ("Please tell", "after")),
    ("Please verify that you move {owner}'s {obj} only after {time} has passed.",
     "Please move {owner}'s {obj} only after {time}.", "move", ("Please", "only after")),
]
CONDITION = [
    ("I am setting this rule in writing: you can move the {obj} only if the {role} agrees.",
     "You can move the {obj} only if the {role} agrees.", "move", ("can", "only if", "agrees")),
    ("Here is the limit I want followed: you must not open the {obj} unless the {role} gives permission.",
     "You must not open the {obj} unless the {role} gives permission.", "open", ("must not", "unless", "permission")),
    ("This is the request I need carried out: please collect exactly {q} {items} from the {place}.",
     "Please collect exactly {q} {items} from the {place}.", "collect", ("Please", "exactly")),
    ("Hold this fact for later so it is not lost: all the {group} should inspect the {obj} on {day}.",
     "Remember: all the {group} should inspect the {obj} on {day}.", "inspect", ("Remember", "all", "should")),
    ("Carry out this cap: please carry at most {q} {items} into the {place}.",
     "Please carry at most {q} {items} into the {place}.", "carry", ("Please", "at most")),
    ("Follow this timing rule: you may use the {obj} only after {time}.",
     "You may use the {obj} only after {time}.", "use", ("may", "only after")),
    ("Keep this access rule in writing: only the {group} may enter the {place} before {time}.",
     "Remember: only the {group} may enter the {place} before {time}.", "enter", ("Remember", "only", "may", "before")),
    ("Apply this floor: please leave at least {q} {items} in the {place} until {day}.",
     "Please leave at least {q} {items} in the {place} until {day}.", "leave", ("Please", "at least", "until")),
]
COMPACT = [
    ("Kindly note for your records you should inspect the {obj} on {day}.",
     "You should inspect the {obj} on {day}.", "inspect", ("should",)),
    ("Take note in writing that my {obj} belongs in the {place} after {time}.",
     "My {obj} belongs in the {place} after {time}.", "belongs", ("my", "after")),
    ("Do not forget to return {owner}'s {obj} before {day}.",
     "Return {owner}'s {obj} before {day}.", "return", ("before",)),
    ("Be advised in this note you may carry exactly {q} {items} into the {place}.",
     "You may carry exactly {q} {items} into the {place}.", "carry", ("may", "exactly")),
    ("For the record in this note, leave {owner}'s {obj} at the {place} until {time}.",
     "Leave {owner}'s {obj} at the {place} until {time}.", "leave", ("until",)),
    ("Just so it is clear, keep all {q} {items} in the {place} on {day}.",
     "Keep all {q} {items} in the {place} on {day}.", "keep", ("all",)),
    ("As a reminder, move the {obj} only after {time}.",
     "Move the {obj} only after {time}.", "move", ("only after",)),
    ("Note well in this message: bring my {obj} to the {place} on {day}.",
     "Bring my {obj} to the {place} on {day}.", "bring", ("my",)),
]
DEV_SHORT = [
    ("Please look to it that you verify exactly {q} {items} at the {place} before {time}.",
     "Please verify exactly {q} {items} at the {place} before {time}.", "verify", ("Please", "exactly", "before")),
    ("Please hold in mind to keep {owner}'s {obj} in the {place} until {day}.",
     "Please keep {owner}'s {obj} in the {place} until {day}.", "keep", ("Please", "until")),
    ("You may review the {obj} at the {place}, with permission required from the {role}.",
     "You may review the {obj} at the {place} with the {role}'s permission.", "review", ("may", "permission")),
    ("This remains a plan, not a promise: I should carry the {obj} to the {place} on {day}.",
     "I should carry the {obj} to the {place} on {day}; no commitment.", "carry", ("should", "no commitment")),
    ("Please look to it that nobody has moved {owner}'s {obj} from the {place}.",
     "Please check: nobody has moved {owner}'s {obj} from the {place}.", "check", ("Please check", "nobody has moved")),
    ("Follow this gate: only if the {role} agrees may the {group} take exactly {q} {items}.",
     "Only if the {role} agrees may the {group} take exactly {q} {items}.", "take", ("Only if", "agrees", "may", "exactly")),
    ("Please look to it that you ask all the {group} to wait at the {place} until {time}.",
     "Please ask all the {group} to wait at the {place} until {time}.", "ask", ("Please ask", "all", "until")),
    ("Hold this rule in mind: the {obj} must not leave the {place} before {day}.",
     "Remember: the {obj} must not leave the {place} before {day}.", "leave", ("Remember", "must not", "before")),
]
TRAIN_MESSAGES = [
    "I spotted {owned} and {force} return it on {day}.",
    "I packed {owned} and {force} bring it to the {place} before {time}.",
    "I gathered {owned} and {force} leave it at the {place} on {day}.",
    "I mended {owned} and {force} hand it back after {time}.",
    "I looked over {owned} at the {place} on {day}; I have not moved it elsewhere.",
    "I put away {owned} in the {place} on {day}; nobody has moved it elsewhere.",
    "I can send {owned} back only after {time}.",
    "I will take {owned} to the {place} on {day} only if the {role} agrees.",
]
MODAL_MESSAGES = [
    "I spotted {owned}; I {force} return it on {day}.",
    "I packed {owned}; I {force} bring it to the {place} before {time}.",
    "I gathered {owned}; I {force} leave it at the {place} on {day}.",
    "I mended {owned}; I {force} hand it back after {time}.",
]
DEV_MESSAGES = [
    "I logged {owned} and {force} send it back on {day}.",
    "I looked at {owned} and {force} deliver it to the {place} before {time}.",
    "I locked {owned} at the {place} on {day}; I have not handed it to anyone.",
    "I tagged {owned} at the {place} on {day}; nobody has handed it to anyone.",
]
ADDRESSEE_WRAPPERS = (
    "Write a quick note to {n}: {s}",
    "Send {n} this update: {s}",
    "Draft a brief message for {n}: {s}",
    "Write {n} a short note: {s}",
)
OTHER_WRAPPERS = (
    "Write my message to {n}: {s}",
    "Compose this update for {n}: {s}",
)
DEV_WRAPPERS = (
    "Compose my note addressed to {n}: {s}",
    "Send this note over to {n}: {s}",
)
FORCES = ("can", "should", "might", "will")
GREETINGS = ("Hi {n}, ", "{n}, ", "Hello {n}, ")


def norm(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def prefix2(text):
    tokens = norm(text).split()
    return tuple(tokens[:2])


def values(split, i):
    if split == "train":
        people = [("Nadia", "her"), ("Oscar", "his"), ("Sloane", "their"), ("Priel", "her"),
                  ("Helmut", "his"), ("Eden", "their"), ("Clio", "her"), ("Bram", "his")]
        owners = {"her": ["Noemi", "Hester", "Odette", "Sable"], "his": ["Rafa", "Ivo", "Pavel", "Tomas"],
                  "their": ["Oakley", "Shiloh", "Wren", "Rowan"]}
        objects = ["copper whistle", "navy canteen", "ivory comb", "cedar metronome",
                   "striped scarf", "olive knapsack", "tin harmonica", "crimson vial"]
        groups = ["printers", "glaziers", "scribes", "coopers", "dyers", "fletchers", "masons", "joiners"]
        places = ["oak vestibule", "clay studio", "pine loft", "amber quay", "west cloister", "iron mezzanine", "south apse", "glass atrium"]
    else:
        people = [("Hana", "her"), ("Nils", "his"), ("Sage", "their"), ("Ottilie", "her")]
        owners = {"her": ["Ines", "Paloma"], "his": ["Joaquin", "Leif"], "their": ["Haven", "Soren"]}
        objects = ["linen satchel", "cobalt flask", "woven visor", "onyx clasp"]
        groups = ["tilers", "gilders", "luthiers", "ushers"]
        places = ["slate courtyard", "maple gallery", "harbor shed", "bronze foyer"]
    pick = lambda seq: seq[i % len(seq)]
    n, pronoun = pick(people)
    owner_pronoun = ("her", "his", "their")[i % 3]
    return dict(n=n, pronoun=pronoun, owner=owners[owner_pronoun][(i // 3) % len(owners[owner_pronoun])],
                owner_pronoun=owner_pronoun, obj=pick(objects), group=pick(groups), place=pick(places),
                day=pick(["Thursday", "Saturday", "Monday", "Wednesday", "Friday", "Sunday", "Tuesday"]),
                time=f"{1 + i % 8}:{(i * 13 + 23) % 60:02d} PM", q=str(3 + i % 11),
                items=pick(["tickets", "pins", "labels", "reels"]), role=pick(["curator", "foreman", "captain", "custodian"]))


def protected_fields(template, v):
    keys = dict.fromkeys(k for _, k, _, _ in Formatter().parse(template) if k)
    return [v[k] for k in keys]


def short_row(split, group, t, j, template, i):
    source, answer, verb, anchors = template
    v = values(split, i)
    return dict(id=f"wr6-{split}-short-{group}-{t:02}-{j:02}", split=split, training_allowed=split == "train",
                family="shortening", shortening_group=group, source=source.format(**v),
                prompt=WRAPPERS[(t + j) % 4].format(s=source.format(**v)), answer=answer.format(**v),
                structure=source, action_verb=verb, protected=protected_fields(answer, v) + list(anchors),
                semantic_review_required=True)


def message_row(split, row_id, template, v, relation, representation, group, i, wrapper=None, **meta):
    owner = v["n"] if relation == "recipient" else v["owner"]
    pronoun = v["pronoun"] if relation == "recipient" else v["owner_pronoun"]
    owned = owner + "'s " + v["obj"] if representation == "named" else pronoun + " " + v["obj"]
    ownership = "your " + v["obj"] if relation == "recipient" else owner + "'s " + v["obj"]
    context = "" if relation == "recipient" or representation == "named" else f"Owner entry: {owner} owns the {v['obj']}. "
    w = dict(v, owned=owned, ownership=ownership)
    src = template.format(**w)
    answer = template.format(**dict(w, owned=ownership))
    if wrapper is None:
        pool = ADDRESSEE_WRAPPERS if relation == "recipient" else OTHER_WRAPPERS
        if split != "train":
            pool = DEV_WRAPPERS
        wrapper = pool[i % len(pool)]
    return dict(id=row_id, split=split, training_allowed=split == "train", family="recipient",
                prompt=context + wrapper.format(n=v["n"], s=src), source=src,
                answer=GREETINGS[i % 3].format(n=v["n"]) + answer, recipient=v["n"], owner=owner,
                owner_pronoun=pronoun, object=v["obj"], relation=relation, representation=representation,
                message_group=group, structure=template + "|" + wrapper + "|" + relation + "|" + representation,
                protected=[v["n"], ownership] + [v[k] for _, k, _, _ in Formatter().parse(template) if k and k != "owned"],
                force=v.get("force") if "{force}" in template else None, semantic_review_required=True, **meta)


def writing(split):
    if split not in ("train", "dev"):
        raise ValueError("Only training and fresh development are supported")
    rows = []
    if split == "train":
        for group, templates, repeats, offset in (("hedge", HEDGE, 8, 0), ("condition", CONDITION, 2, 101),
                                                  ("compact", COMPACT, 2, 173)):
            for t, template in enumerate(templates):
                for j in range(repeats):
                    rows.append(short_row(split, group, t, j, template, offset + t * 8 + j))
        for p, pronoun in enumerate(("her", "his", "their")):
            for j in range(16):
                v = values(split, j * 3 + p)
                v["pronoun"] = pronoun
                v["force"] = FORCES[(j // 4) % 4]
                rows.append(message_row(split, f"wr6-train-addressee-{pronoun}-{j:02}", TRAIN_MESSAGES[j % 8],
                                        v, "recipient", "pronoun", "addressee_pronoun", j + p))
        for j in range(24):
            v = values(split, 211 + j)
            v["force"] = FORCES[(j // 6) % 4]
            rows.append(message_row(split, f"wr6-train-recipient-named-{j:02}", TRAIN_MESSAGES[j % 8], v,
                                    "recipient", "named", "recipient_named", j))
        for p, pronoun in enumerate(("her", "his", "their")):
            for j in range(8):
                v = values(split, 280 + j * 3 + p)
                v["owner_pronoun"] = pronoun
                v["force"] = FORCES[(j // 2) % 4]
                rows.append(message_row(split, f"wr6-train-owner-{pronoun}-{j:02}", TRAIN_MESSAGES[j % 8],
                                        v, "third_party", "pronoun", "third_party_pronoun", j + p))
        for j in range(16):
            v = values(split, 340 + j)
            v["force"] = FORCES[(j // 4) % 4]
            rows.append(message_row(split, f"wr6-train-named-{j:02}", TRAIN_MESSAGES[j % 8], v,
                                    "third_party", "named", "third_party_named", j))
        for j in range(8):
            for force in ("should", "will"):
                v = dict(values(split, 401 + j), force=force)
                rows.append(message_row(split, f"wr6-train-modal-{j:02}-{force}", MODAL_MESSAGES[j % 4], v,
                                        "recipient", "pronoun", "modal_contrast", j, pair_id=f"modal-{j:02}"))
    else:
        for t, template in enumerate(DEV_SHORT):
            for j in range(4):
                rows.append(short_row(split, "diagnostic", t, j, template, 701 + t * 4 + j))
        for j in range(16):
            v = values(split, 809 + j)
            v["force"] = FORCES[(j // 4) % 4]
            rows.append(message_row(split, f"wr6-dev-addressee-{j:02}", DEV_MESSAGES[j % 4], v,
                                    "recipient", "pronoun", "addressee_pronoun", j))
        for j in range(8):
            v = values(split, 880 + j)
            v["force"] = FORCES[j % 4]
            rows.append(message_row(split, f"wr6-dev-owner-{j:02}", DEV_MESSAGES[j % 4], v,
                                    "third_party", "pronoun", "third_party_pronoun", j))
        for j in range(4):
            for force in ("should", "will"):
                v = dict(values(split, 907 + j), force=force)
                rows.append(message_row(split, f"wr6-dev-modal-{j:02}-{force}", DEV_MESSAGES[j % 2], v,
                                        "recipient", "pronoun", "modal_contrast", j, pair_id=f"dev-modal-{j:02}"))
    for row in rows:
        validate_reference(row)
    return rows


def validate_reference(row):
    answer = row["answer"]
    source = row["source"]
    if not all(x.casefold() in answer.casefold() for x in row["protected"]):
        raise ValueError("Required fact or force was dropped: " + row["id"])
    if row["family"] == "shortening":
        if norm(answer) == norm(source) or len(answer) >= len(source) or len(answer.split()) >= len(source.split()):
            raise ValueError("Reference must actually shorten: " + row["id"])
        if row["split"] == "train" and len(answer) / len(source) > 0.8:
            raise ValueError("Reference exceeds shortening budget: " + row["id"])
        if row["split"] == "train" and prefix2(answer) == prefix2(source):
            raise ValueError("Reference shares the source prefix: " + row["id"])
        if row["action_verb"] not in norm(answer).split():
            raise ValueError("Original action verb changed")
        if "ensure" in norm(answer).split() and "ensure" not in norm(source).split():
            raise ValueError("Cannot introduce ensure")
        if re.findall(r"\d+", answer) != re.findall(r"\d+", source):
            raise ValueError("Numeric information changed")
    else:
        if not any(answer.startswith(g.format(n=row["recipient"])) for g in GREETINGS):
            raise ValueError("Wrong addressee")
        if row.get("force") and not re.search(r"\b" + row["force"] + r"\b", answer):
            raise ValueError("Modal force changed")
        if row.get("message_group") == "addressee_pronoun" and "your " + row["object"] not in answer:
            raise ValueError("Addressee object was not converted to your")
        for claim in ("I have not moved it elsewhere", "nobody has moved it elsewhere",
                      "I have not handed it to anyone", "nobody has handed it to anyone",
                      "only after", "only if"):
            if claim in source and claim not in answer:
                raise ValueError("Negative or condition scope changed")


def build_train_rows(retention):
    expected = {"arithmetic": 64, "extraction": 64, "grounding": 48, "clarification": 48, "direct": 32}
    if len(retention) != 256 or Counter(r["family"] for r in retention) != expected:
        raise ValueError("Retention composition changed")
    if any(r.get("split") != "train" or r.get("training_allowed") is not True for r in retention):
        raise ValueError("Require training-only retention")
    rows = writing("train")
    short = [r for r in rows if r["family"] == "shortening"]
    mean = sum(len(r["answer"]) / len(r["source"]) for r in short) / len(short)
    if not 0.62 <= mean <= 0.78:
        raise ValueError("Mean reference length left the 0.62-0.78 window: " + str(mean))
    rows.extend(dict(id="wr6-retain-" + r["id"], split="train", training_allowed=True,
                     family=r["family"], prompt=r["prompt"], answer=r["answer"]) for r in retention)
    random.Random(SEED + 3).shuffle(rows)
    sft_bytes(rows)
    return rows


def sft_bytes(rows):
    ids, prompts = set(), set()
    for r in rows:
        if r.get("split") != "train" or r.get("training_allowed") is not True:
            raise ValueError("Evaluation rows cannot enter training")
        if not all(isinstance(r.get(k), str) and r[k].strip() for k in ("id", "prompt", "answer")):
            raise ValueError("Invalid training text")
        if r["id"] in ids or norm(r["prompt"]) in prompts:
            raise ValueError("Duplicate training row: " + r["id"] + " | " + r["prompt"])
        ids.add(r["id"])
        prompts.add(norm(r["prompt"]))
    return "".join(json.dumps({k: r[k] for k in ("id", "prompt", "answer")}, sort_keys=True, ensure_ascii=False) + "\n" for r in rows).encode()


def fourgrams(text):
    tokens = norm(text).split()
    return {tuple(tokens[i:i + 4]) for i in range(len(tokens) - 3)}


def audit(train, dev, historical, benchmark_sources):
    for field in ("id", "prompt", "source", "structure"):
        a = {norm(r[field]) for r in train if r.get(field)}
        b = {norm(r[field]) for r in dev if r.get(field)}
        if a & b:
            raise ValueError("Cross-split overlap: " + field)
    history_prompts = {norm(r["prompt"]) for r in historical}
    history_sources = {norm(r["source"]) for r in historical if r.get("source")}
    new = [r for r in train if not r["id"].startswith("wr6-retain-")] + dev
    if any(norm(r["prompt"]) in history_prompts or (r.get("source") and norm(r["source"]) in history_sources) for r in new):
        raise ValueError("Previously observed writing prompt/source reused")
    benchmark_grams = set().union(*(fourgrams(s) for s in benchmark_sources)) if benchmark_sources else set()
    for r in train:
        if r["family"] == "shortening" and fourgrams(r["source"]) & benchmark_grams:
            raise ValueError("Training source shares a benchmark four-token frame: " + r["id"])
    short = [r for r in train if r["family"] == "shortening"]
    return dict(train=len(train), fresh_dev=len(dev), historical_rows_checked=len(historical),
                benchmark_shortening_sources_checked=len(benchmark_sources), exact_prompt_source_overlaps=0,
                cross_split_structure_overlap=0, benchmark_four_token_overlaps=0,
                mean_reference_length_ratio=round(sum(len(r["answer"]) / len(r["source"]) for r in short) / len(short), 6),
                scope="Observed history only; reserved final holdouts remain unloaded")


def shortening_source(row):
    if row.get("kind") in ("shorten", "shortening") and row.get("source"):
        return row["source"]
    if str(row.get("id", "")).startswith("draft-short-"):
        match = re.search(r"'([^']+)'", row.get("prompt", ""))
        return match.group(1) if match else None
    return None


def length_stats(pairs):
    if not pairs:
        raise ValueError("No shortening cases")
    return dict(rows=len(pairs), verbatim_copy=sum(norm(out) == norm(src) for src, out in pairs),
                not_shorter=sum(len(out.strip()) >= len(src) for src, out in pairs),
                mean_length_ratio=round(sum(len(out.strip()) / len(src) for src, out in pairs) / len(pairs), 6))
