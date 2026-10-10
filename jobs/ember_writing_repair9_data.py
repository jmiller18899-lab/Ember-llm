"""WR9 copy-margin shortening. The rejected completion is the verbatim source.

WR8 penalized meaning-wrong rewrites and the frozen benchmark still copied more.
These pairs keep a v3-passing short answer and contrast it with the unchanged source.
Generation grades stay manual. v3 is not a runtime gate.
"""
from collections import Counter
from string import Formatter
import json, math, random, re
import ember_meaning_preservation_v3 as V3

VERSION = "ember-writing-repair9-copy-margin-v1"
SEED = 431
WRAPPERS = [
    'Compress this note. Keep force, owners, counts and timing: "{s}"',
    "Write a shorter version. Do not copy the source unchanged:\n{s}",
    "Reduce the wording and preserve every obligation: {s}",
    'Brief form, same meaning and same reminders: "{s}"',
]
DEV_WRAPPERS = [
    'Make a shorter note with the same force: "{s}"',
    "Trim wording. Keep owners, counts and reminders:\n{s}",
    "Shorter equivalent, not a copy: {s}",
    'Condense without dropping obligations: "{s}"',
]
PREFERRED = [
    ("timing_should",
     "Please note that each {group} should examine the {obj} prior to {time}.",
     "Each {group} should examine the {obj} before {time}.",
     "examine", ("should", "before")),
    ("consent_unless",
     "No one may carry {owner}'s {obj} unless the {role} has given consent.",
     "No one may carry {owner}'s {obj} without {role} consent.",
     "carry", ("No one may", "without")),
    ("reminder",
     "Please remember that all of the {group} must stay in the {place} until {day}.",
     "Please remember all of the {group} must stay in the {place} until {day}.",
     "stay", ("Please remember", "must", "until")),
    ("consent_only_if",
     "Please note that you may hold the {obj} only if the {role} has given consent.",
     "You may hold the {obj} only if the {role} has consented.",
     "hold", ("may", "only if")),
    ("reminder_forget",
     "Do not forget that {owner}'s {obj} might rest in the {place} after {time}.",
     "Do not forget {owner}'s {obj} might rest in the {place} after {time}.",
     "rest", ("Do not forget", "might", "after")),
    ("exactly",
     "Please note that each {group} must store exactly {q} {items} in the {place} until {day}.",
     "Each {group} must store exactly {q} {items} in the {place} until {day}.",
     "store", ("must", "exactly", "until")),
    ("ownership",
     "Please carry {owner}'s {obj} to the {place} prior to {day}.",
     "Please carry {owner}'s {obj} to the {place} before {day}.",
     "carry", ("Please", "before")),
    ("only_after",
     "Please note that you may close the {obj} only after {time} at the {place}.",
     "You may close the {obj} only after {time} at the {place}.",
     "close", ("may", "only after")),
]
DEV_SHORT = [
    ("Please note that each {group} should mend the {obj} prior to {time}.",
     "Each {group} should mend the {obj} before {time}.", "mend", ("should", "before")),
    ("No one may shift {owner}'s {obj} unless the {role} has given consent.",
     "No one may shift {owner}'s {obj} without {role} consent.", "shift", ("No one may", "without")),
    ("Please remember that all of the {group} must pause in the {place} until {day}.",
     "Please remember all of the {group} must pause in the {place} until {day}.", "pause", ("Please remember", "must")),
    ("Please note that you may read the {obj} only if the {role} has given consent.",
     "You may read the {obj} only if the {role} has consented.", "read", ("may", "only if")),
    ("Do not forget that {owner}'s {obj} might remain in the {place} after {time}.",
     "Do not forget {owner}'s {obj} might remain in the {place} after {time}.", "remain", ("Do not forget", "might")),
    ("Please note that each {group} must count exactly {q} {items} in the {place} until {day}.",
     "Each {group} must count exactly {q} {items} in the {place} until {day}.", "count", ("must", "exactly")),
    ("Please ferry {owner}'s {obj} to the {place} prior to {day}.",
     "Please ferry {owner}'s {obj} to the {place} before {day}.", "ferry", ("Please", "before")),
    ("Please note that you may lock the {obj} only after {time} at the {place}.",
     "You may lock the {obj} only after {time} at the {place}.", "lock", ("may", "only after")),
]
TRAIN_MESSAGES = [
    "I labeled {owned} and {force} return it on {day}.",
    "I packed {owned} and {force} carry it to the {place} before {time}.",
    "I sorted {owned} and {force} leave it in the {place} on {day}.",
    "I logged {owned} and {force} hand it back after {time}.",
]
DEV_MESSAGES = [
    "I indexed {owned} and {force} send it back on {day}.",
    "I measured {owned} and {force} deliver it to the {place} before {time}.",
]
FORCES = ("can", "should", "might", "will")
GREETINGS = ("Hi {n}, ", "{n}, ", "Hello {n}, ")


def norm(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def copy_margin(nll_preferred, nll_source):
    """Positive when the verbatim source is more likely than the short answer.

    Shared fact tokens sit in both sequence NLLs, so the margin is driven by the
    extra source wording. The trainer adds this to answer-only cross-entropy.
    """
    delta = nll_preferred - nll_source
    if hasattr(delta, "detach"):
        import torch
        return torch.nn.functional.softplus(delta)
    delta = float(delta)
    return math.log1p(math.exp(-abs(delta))) + max(delta, 0.0)


def values(split, i):
    if split == "train":
        people = [("Liora", "her"), ("Soren", "his"), ("Quill", "their"), ("Maren", "her"),
                  ("Idris", "his"), ("Noor", "their"), ("Elspeth", "her"), ("Bram", "his")]
        owners = {"her": ["Nessa", "Orla", "Pilar", "Wren"], "his": ["Torin", "Edric", "Galen", "Rolf"],
                  "their": ["Shay", "Remy", "Cassian", "Hollis"]}
        objects = ["brass compass", "silk spindle", "amber inkwell", "wool satchel",
                   "copper kettle", "glass prism", "ivory comb", "oak stamp"]
        groups = ["bookbinders", "sailmakers", "goldsmiths", "weavers", "potters", "cartographers", "locksmiths", "tanners"]
        places = ["cedar gallery", "marble court", "birch workshop", "granite vault",
                  "linen hall", "copper yard", "sage cloister", "harbor loft"]
        roles = ["archivist", "notary", "curator", "clerk"]
        items = ["labels", "vials", "tiles", "keys"]
    else:
        people = [("Anwen", "her"), ("Leif", "his"), ("Sable", "their"), ("Freya", "her")]
        owners = {"her": ["Iona", "Blythe"], "his": ["Finn", "Pavel"], "their": ["Sol", "Wynn"]}
        objects = ["silver needle", "clay flute", "ebony ruler", "hemp bundle"]
        groups = ["illuminators", "ropemakers", "plasterers", "joiners"]
        places = ["maple arcade", "slate forge", "ivory scriptorium", "willow dock"]
        roles = ["scribe", "auditor", "herald", "custodian"]
        items = ["stamps", "cords", "slips", "pegs"]
    pick = lambda seq: seq[i % len(seq)]
    n, pronoun = pick(people)
    owner_pronoun = ("her", "his", "their")[i % 3]
    return dict(n=n, pronoun=pronoun, owner=owners[owner_pronoun][(i // 3) % len(owners[owner_pronoun])],
                owner_pronoun=owner_pronoun, obj=pick(objects), group=pick(groups), place=pick(places),
                day=pick(["Thursday", "Saturday", "Monday", "Wednesday", "Friday", "Sunday", "Tuesday"]),
                time=f"{1 + i % 8}:{(i * 13 + 7) % 60:02d} PM", q=str(4 + i % 9),
                items=pick(items), role=pick(roles))


def protected_fields(template, v):
    keys = dict.fromkeys(k for _, k, _, _ in Formatter().parse(template) if k)
    return [v[k] for k in keys]


def shortening_row(source, answer, verb, anchors, v, split, group, t, j, wrappers):
    src, ans = source.format(**v), answer.format(**v)
    row = dict(id=f"wr9-{split}-short-{group}-{t:02}-{j:02}", split=split,
               training_allowed=split == "train", family="shortening", shortening_group=group,
               source=src, prompt=wrappers[(t + j) % 4].format(s=src), answer=ans, structure=source,
               action_verb=verb, protected=protected_fields(answer, v) + list(anchors),
               semantic_review_required=True, loss="copy_margin" if split == "train" else "eval")
    if split == "train":
        row["rejected"] = src
    return row


def meaning_row(source, answer):
    return {"kind": "shortening", "source": source}


def message_row(split, row_id, template, v, relation, representation, group, i):
    owner = v["n"] if relation == "recipient" else v["owner"]
    pronoun = v["pronoun"] if relation == "recipient" else v["owner_pronoun"]
    owned = owner + "'s " + v["obj"] if representation == "named" else pronoun + " " + v["obj"]
    ownership = "your " + v["obj"] if relation == "recipient" else owner + "'s " + v["obj"]
    context = "" if relation == "recipient" or representation == "named" else f"Owner card: {owner} owns the {v['obj']}. "
    w = dict(v, owned=owned, ownership=ownership)
    src = template.format(**w)
    answer = template.format(**dict(w, owned=ownership))
    wrapper = "Forward this draft to {n}: {s}" if split == "train" else "Pass this note to {n}: {s}"
    return dict(id=row_id, split=split, training_allowed=split == "train", family="recipient",
                prompt=context + wrapper.format(n=v["n"], s=src), source=src,
                answer=GREETINGS[i % 3].format(n=v["n"]) + answer, recipient=v["n"], owner=owner,
                owner_pronoun=pronoun, object=v["obj"], relation=relation, representation=representation,
                message_group=group, structure=template + "|" + wrapper + "|" + relation + "|" + representation,
                protected=[v["n"], ownership] + [v[k] for _, k, _, _ in Formatter().parse(template) if k and k != "owned"],
                force=v.get("force") if "{force}" in template else None, semantic_review_required=True, loss="sft")


def writing(split):
    if split not in ("train", "dev"):
        raise ValueError("Only training and fresh development are supported")
    rows = []
    if split == "train":
        for t, (group, source, answer, verb, anchors) in enumerate(PREFERRED):
            for j in range(12):
                rows.append(shortening_row(source, answer, verb, anchors, values(split, 23 + t * 12 + j),
                                           split, group, t, j, WRAPPERS))
        for j in range(32):
            v = values(split, 451 + j)
            v["force"] = FORCES[j % 4]
            rows.append(message_row(split, f"wr9-train-recipient-{j:02}", TRAIN_MESSAGES[j % 4], v,
                                    "recipient", "named" if j % 2 == 0 else "pronoun", "recipient", j))
        for j in range(32):
            v = values(split, 551 + j)
            v["force"] = FORCES[j % 4]
            rows.append(message_row(split, f"wr9-train-named-{j:02}", TRAIN_MESSAGES[j % 4], v,
                                    "third_party", "named", "third_party_named", j))
    else:
        for t, (source, answer, verb, anchors) in enumerate(DEV_SHORT):
            for j in range(4):
                rows.append(shortening_row(source, answer, verb, anchors, values(split, 761 + t * 4 + j),
                                           split, "diagnostic", t, j, DEV_WRAPPERS))
        for j in range(16):
            v = values(split, 881 + j)
            v["force"] = FORCES[j % 4]
            rows.append(message_row(split, f"wr9-dev-recipient-{j:02}", DEV_MESSAGES[j % 2], v,
                                    "recipient" if j % 2 == 0 else "third_party",
                                    "named" if j % 3 else "pronoun", "mixed_owner", j))
        compact = [
            ("Please remember that every illuminator must wear a cuff until dusk.",
             "Please remember every illuminator must wear a cuff until dusk.", "wear",
             ["Please remember", "every illuminator", "must", "cuff", "until dusk"]),
            ("Everyone in the slate forge should gather at the maple arcade prior to dusk.",
             "Everyone in the slate forge should gather at the maple arcade before dusk.", "gather",
             ["Everyone", "slate forge", "should", "maple arcade", "before dusk"]),
            ("No one may lend Freya's hemp bundle unless the scribe has given consent.",
             "No one may lend Freya's hemp bundle without scribe consent.", "lend",
             ["No one may", "Freya's hemp bundle", "without"]),
            ("Please note that each joiner must retain exactly six slips until the count ends.",
             "Each joiner must retain exactly six slips until the count ends.", "retain",
             ["joiner", "must", "exactly six slips"]),
            ("Do not forget that Leif might collect his clay flute after rehearsal.",
             "Do not forget Leif might collect his clay flute after rehearsal.", "collect",
             ["Do not forget", "Leif", "might", "his clay flute"]),
            ("You can unlatch the silver needle only if the auditor is present in the maple arcade.",
             "You can unlatch the silver needle only if the auditor is in the maple arcade.", "unlatch",
             ["can", "silver needle", "only if", "auditor"]),
            ("Please store Anwen's ebony ruler flat in order to avoid warping it overnight.",
             "Please store Anwen's ebony ruler flat to avoid warping it overnight.", "store",
             ["Please", "Anwen's ebony ruler", "flat"]),
            ("It is important that nobody shift the hemp bundle before Friday.",
             "No one shift the hemp bundle before Friday.", "shift",
             ["No one", "hemp bundle", "before Friday"]),
        ]
        for i, (source, answer, verb, protected) in enumerate(compact):
            rows.append(dict(id=f"wr9-dev-compact-{i:02}", split="dev", training_allowed=False,
                             family="shortening", shortening_group="compact_diagnostic", source=source,
                             prompt=DEV_WRAPPERS[i % 4].format(s=source), answer=answer, structure=source,
                             action_verb=verb, protected=protected, semantic_review_required=True, loss="eval"))
    for row in rows:
        validate_reference(row)
    return rows


def validate_reference(row):
    answer, source = row["answer"], row["source"]
    if not all(x.casefold() in answer.casefold() for x in row["protected"]):
        raise ValueError("Required fact or force was dropped: " + row["id"])
    if row["family"] == "shortening":
        if norm(answer) == norm(source) or len(answer) >= len(source) or len(answer.split()) >= len(source.split()):
            raise ValueError("Reference must actually shorten: " + row["id"])
        if row["action_verb"] not in norm(answer).split():
            raise ValueError("Original action verb changed")
        if "ensure" in norm(answer).split() and "ensure" not in norm(source).split():
            raise ValueError("Cannot introduce ensure")
        if re.findall(r"\d+", answer) != re.findall(r"\d+", source):
            raise ValueError("Numeric information changed")
        grade = V3.grade_case(meaning_row(source, answer), answer, lambda r, o: True)
        if row["split"] == "train" and grade["status"] != "pass":
            raise ValueError("Preferred WR9 shortening must pass v3: " + row["id"] + " " + str(grade))
        if row.get("rejected"):
            if row["rejected"] != source:
                raise ValueError("WR9 rejected completion must be the verbatim source: " + row["id"])
            bad = V3.grade_case(meaning_row(source, answer), row["rejected"], lambda r, o: True)
            if bad["status"] == "pass":
                raise ValueError("Verbatim source must not pass v3: " + row["id"])
    else:
        if not any(answer.startswith(g.format(n=row["recipient"])) for g in GREETINGS):
            raise ValueError("Wrong addressee")
        if row.get("force") and not re.search(r"\b" + row["force"] + r"\b", answer):
            raise ValueError("Modal force changed")


def build_train_rows(retention):
    expected = {"arithmetic": 64, "extraction": 64, "grounding": 48, "clarification": 48, "direct": 32}
    if len(retention) != 256 or Counter(r["family"] for r in retention) != expected:
        raise ValueError("Retention composition changed")
    if any(r.get("split") != "train" or r.get("training_allowed") is not True for r in retention):
        raise ValueError("Require training-only retention")
    rows = writing("train")
    short = [r for r in rows if r["family"] == "shortening"]
    if len(short) != 96:
        raise ValueError("WR9 uses 96 copy-margin shortening pairs")
    if any(r.get("rejected") != r.get("source") for r in short):
        raise ValueError("Each shortening pair must contrast the verbatim source")
    rows.extend(dict(id="wr9-retain-" + r["id"], split="train", training_allowed=True,
                     family=r["family"], prompt=r["prompt"], answer=r["answer"], loss="sft") for r in retention)
    random.Random(SEED + 9).shuffle(rows)
    sft_bytes(rows)
    return rows


def encode_examples(rows):
    """One trainer example per row. Copy-margin rows carry the source as the contrast."""
    encoded = []
    for r in rows:
        item = dict(id=r["id"], prompt=r["prompt"], answer=r["answer"], family=r["family"],
                    loss_kind=1 if r.get("loss") == "copy_margin" else 0)
        if item["loss_kind"] == 1:
            item["rejected"] = r["rejected"]
        encoded.append(item)
    if len(encoded) != 416 or sum(x["loss_kind"] == 1 for x in encoded) != 96:
        raise ValueError("Expected 416 examples and 96 copy-margin pairs")
    return encoded


def sft_bytes(rows):
    ids, prompts = set(), set()
    for r in rows:
        if r.get("split") != "train" or r.get("training_allowed") is not True:
            raise ValueError("Evaluation rows cannot enter training")
        if not all(isinstance(r.get(k), str) and r[k].strip() for k in ("id", "prompt", "answer")):
            raise ValueError("Invalid training text")
        if r["id"] in ids:
            raise ValueError("Duplicate training row: " + r["id"])
        if r.get("loss") != "copy_margin" and norm(r["prompt"]) in prompts:
            raise ValueError("Duplicate training prompt: " + r["prompt"])
        ids.add(r["id"])
        prompts.add(norm(r["prompt"]))
        if r.get("loss") == "copy_margin":
            if r.get("rejected") != r.get("source"):
                raise ValueError("Copy-margin row must reject the verbatim source")
    payload = []
    for r in rows:
        item = {k: r[k] for k in ("id", "prompt", "answer") if k in r}
        if r.get("rejected"):
            item["rejected"] = r["rejected"]
            item["kind"] = "copy_margin"
        payload.append(item)
    return "".join(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n" for item in payload).encode()


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
    new = [r for r in train if not r["id"].startswith("wr9-retain-")] + dev
    if any(norm(r["prompt"]) in history_prompts or (r.get("source") and norm(r["source"]) in history_sources) for r in new):
        raise ValueError("Previously observed writing prompt/source reused")
    benchmark_grams = set().union(*(fourgrams(s) for s in benchmark_sources)) if benchmark_sources else set()
    for r in train:
        if r["family"] == "shortening" and fourgrams(r["source"]) & benchmark_grams:
            raise ValueError("Training source shares a benchmark four-token frame: " + r["id"])
    short = [r for r in train if r["family"] == "shortening"]
    return dict(train=len(train), encoded_sequences=len(train), fresh_dev=len(dev),
                historical_rows_checked=len(historical),
                benchmark_shortening_sources_checked=len(benchmark_sources), exact_prompt_source_overlaps=0,
                cross_split_structure_overlap=0, benchmark_four_token_overlaps=0,
                copy_margin_pairs=len(short),
                mean_reference_length_ratio=round(sum(len(r["answer"]) / len(r["source"]) for r in short) / len(short), 6),
                grader_version=V3.GRADER_VERSION,
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


def meaning_stats(pairs):
    counts = Counter()
    details = []
    for ident, src, out in pairs:
        grade = V3.grade_case(meaning_row(src, out), out, lambda r, o: True)
        counts[grade["status"]] += 1
        details.append({"id": ident, "status": grade["status"], "reasons": grade["reasons"]})
    return {"grader_version": V3.GRADER_VERSION, "counts": dict(counts), "rows": details}
