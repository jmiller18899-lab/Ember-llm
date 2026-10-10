"""WR10 keep-pass + copy-margin shortening.

WR9 contrasted a v3-passing short answer with the verbatim source and held
the copy cap. It lost fresh-shorten-03 by dropping a reminder while trimming.
These pairs keep that source contrast and add an over-aggressive rewrite so
the margin also punishes reminder drops and force changes. v3 is not a
runtime gate.
"""
from collections import Counter
from string import Formatter
import json, math, random, re
import ember_meaning_preservation_v3 as V3

VERSION = "ember-writing-repair10-keep-pass-v1"
SEED = 431
WRAPPERS = [
    'Shorten this note. Keep force, reminders, owners and timing: "{s}"',
    "Write a briefer version. Do not copy the source and do not drop reminders:\n{s}",
    "Reduce the wording and preserve every obligation and reminder: {s}",
    'Compact form, same force and same remember/forget cues: "{s}"',
]
DEV_WRAPPERS = [
    'Make a shorter note with the same force and reminders: "{s}"',
    "Trim wording. Keep owners, counts, reminders and timing:\n{s}",
    "Shorter equivalent, not a copy and not a force change: {s}",
    'Condense without dropping remember/forget or obligations: "{s}"',
]
PREFERRED = [
    ("timing_should",
     "Please note that each {group} should survey the {obj} prior to {time}.",
     "Each {group} should survey the {obj} before {time}.",
     "survey", ("should", "before")),
    ("consent_unless",
     "No one may hoist {owner}'s {obj} unless the {role} has given consent.",
     "No one may hoist {owner}'s {obj} without {role} consent.",
     "hoist", ("No one may", "without")),
    ("reminder",
     "Please remember that all of the {group} must linger in the {place} until {day}.",
     "Please remember all of the {group} must linger in the {place} until {day}.",
     "linger", ("Please remember", "must", "until")),
    ("consent_only_if",
     "Please note that you may clasp the {obj} only if the {role} has given consent.",
     "You may clasp the {obj} only if the {role} has consented.",
     "clasp", ("may", "only if")),
    ("reminder_forget",
     "Do not forget that {owner}'s {obj} might settle in the {place} after {time}.",
     "Do not forget {owner}'s {obj} might settle in the {place} after {time}.",
     "settle", ("Do not forget", "might", "after")),
    ("exactly",
     "Please note that each {group} must file exactly {q} {items} in the {place} until {day}.",
     "Each {group} must file exactly {q} {items} in the {place} until {day}.",
     "file", ("must", "exactly", "until")),
    ("ownership",
     "Please escort {owner}'s {obj} to the {place} prior to {day}.",
     "Please escort {owner}'s {obj} to the {place} before {day}.",
     "escort", ("Please", "before")),
    ("only_after",
     "Please note that you may latch the {obj} only after {time} at the {place}.",
     "You may latch the {obj} only after {time} at the {place}.",
     "latch", ("may", "only after")),
]
DEV_SHORT = [
    ("Please note that each {group} should review the {obj} prior to {time}.",
     "Each {group} should review the {obj} before {time}.", "review", ("should", "before")),
    ("No one may raise {owner}'s {obj} unless the {role} has given consent.",
     "No one may raise {owner}'s {obj} without {role} consent.", "raise", ("No one may", "without")),
    ("Please remember that all of the {group} must assemble in the {place} until {day}.",
     "Please remember all of the {group} must assemble in the {place} until {day}.", "assemble", ("Please remember", "must")),
    ("Please note that you may claim the {obj} only if the {role} has given consent.",
     "You may claim the {obj} only if the {role} has consented.", "claim", ("may", "only if")),
    ("Do not forget that {owner}'s {obj} might abide in the {place} after {time}.",
     "Do not forget {owner}'s {obj} might abide in the {place} after {time}.", "abide", ("Do not forget", "might")),
    ("Please note that each {group} must stack exactly {q} {items} in the {place} until {day}.",
     "Each {group} must stack exactly {q} {items} in the {place} until {day}.", "stack", ("must", "exactly")),
    ("Please guide {owner}'s {obj} to the {place} prior to {day}.",
     "Please guide {owner}'s {obj} to the {place} before {day}.", "guide", ("Please", "before")),
    ("Please note that you may bolt the {obj} only after {time} at the {place}.",
     "You may bolt the {obj} only after {time} at the {place}.", "bolt", ("may", "only after")),
]
TRAIN_MESSAGES = [
    "I tagged {owned} and {force} return it on {day}.",
    "I boxed {owned} and {force} carry it to the {place} before {time}.",
    "I stacked {owned} and {force} leave it in the {place} on {day}.",
    "I recorded {owned} and {force} hand it back after {time}.",
]
DEV_MESSAGES = [
    "I catalogued {owned} and {force} send it back on {day}.",
    "I weighed {owned} and {force} deliver it to the {place} before {time}.",
]
FORCES = ("can", "should", "might", "will")
GREETINGS = ("Hi {n}, ", "{n}, ", "Hello {n}, ")


def norm(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def copy_margin(nll_preferred, nll_other):
    """Positive when the rejected completion is more likely than the short answer."""
    delta = nll_preferred - nll_other
    if hasattr(delta, "detach"):
        import torch
        return torch.nn.functional.softplus(delta)
    delta = float(delta)
    return math.log1p(math.exp(-abs(delta))) + max(delta, 0.0)


def values(split, i):
    if split == "train":
        people = [("Yara", "her"), ("Kellan", "his"), ("Arden", "their"), ("Sable", "her"),
                  ("Oren", "his"), ("Pax", "their"), ("Lydia", "her"), ("Cyrus", "his")]
        owners = {"her": ["Tamsin", "Vesper", "Odessa", "Juniper"], "his": ["Alaric", "Bastian", "Dorian", "Leander"],
                  "their": ["Quinn", "Ellis", "Harlow", "Sloane"]}
        objects = ["jade sextant", "linen folio", "bronze chime", "slate palette",
                   "pewter clasp", "amber spool", "walnut gavel", "ivory flute"]
        groups = ["engravers", "coopers", "fletchers", "saddlers", "gilders", "fullers", "cutlers", "vintners"]
        places = ["opal atrium", "basalt quay", "cypress loft", "umber chapel",
                  "tin gallery", "pearl cloister", "hazel annex", "cobalt yard"]
        roles = ["bursar", "legate", "proctor", "castellan"]
        items = ["seals", "wafers", "chits", "tokens"]
    else:
        people = [("Isolde", "her"), ("Magnus", "his"), ("Briar", "their"), ("Odette", "her")]
        owners = {"her": ["Cleo", "Nerys"], "his": ["Ivo", "Pascal"], "their": ["Reese", "Auden"]}
        objects = ["onyx stylus", "copper reed", "velvet satchel", "marble cubit"]
        groups = ["limners", "farriers", "thatchers", "glaziers"]
        places = ["sienna arcade", "flint forge", "ivory nave", "cedar jetty"]
        roles = ["beadle", "provost", "bailiff", "prefect"]
        items = ["badges", "ribbons", "tickets", "markers"]
    pick = lambda seq: seq[i % len(seq)]
    n, pronoun = pick(people)
    owner_pronoun = ("her", "his", "their")[i % 3]
    return dict(n=n, pronoun=pronoun, owner=owners[owner_pronoun][(i // 3) % len(owners[owner_pronoun])],
                owner_pronoun=owner_pronoun, obj=pick(objects), group=pick(groups), place=pick(places),
                day=pick(["Thursday", "Saturday", "Monday", "Wednesday", "Friday", "Sunday", "Tuesday"]),
                time=f"{1 + i % 8}:{(i * 19 + 5) % 60:02d} PM", q=str(4 + i % 9),
                items=pick(items), role=pick(roles))


def protected_fields(template, v):
    keys = dict.fromkeys(k for _, k, _, _ in Formatter().parse(template) if k)
    return [v[k] for k in keys]


def overedit_for(row):
    """Meaning-wrong rewrite: drop a reminder or change force/conditions."""
    group, src, ans = row["shortening_group"], row["source"], row["answer"]
    if group in ("reminder", "reminder_forget"):
        return re.sub(r"^(?:Please remember that |Please remember |Do not forget that |Do not forget )", "", src)
    if group == "timing_should":
        return ans.replace(" should ", " must ")
    if group == "consent_unless":
        return src.replace(" unless ", " if ")
    if group == "consent_only_if":
        return ans.replace("only if", "if")
    if group == "exactly":
        return ans.replace("exactly ", "")
    if group == "ownership":
        return re.sub(r"\b\w+'s ", "your ", ans)
    if group == "only_after":
        return ans.replace("only after", "after")
    return src


def shortening_row(source, answer, verb, anchors, v, split, group, t, j, wrappers):
    src, ans = source.format(**v), answer.format(**v)
    row = dict(id=f"wr10-{split}-short-{group}-{t:02}-{j:02}", split=split,
               training_allowed=split == "train", family="shortening", shortening_group=group,
               source=src, prompt=wrappers[(t + j) % 4].format(s=src), answer=ans, structure=source,
               action_verb=verb, protected=protected_fields(answer, v) + list(anchors),
               semantic_review_required=True, loss="copy_margin" if split == "train" else "eval")
    if split == "train":
        row["rejected"] = src
        row["overedit"] = overedit_for(row)
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
    wrapper = "Relay this draft to {n}: {s}" if split == "train" else "Give this note to {n}: {s}"
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
                rows.append(shortening_row(source, answer, verb, anchors, values(split, 31 + t * 12 + j),
                                           split, group, t, j, WRAPPERS))
        for j in range(32):
            v = values(split, 451 + j)
            v["force"] = FORCES[j % 4]
            rows.append(message_row(split, f"wr10-train-recipient-{j:02}", TRAIN_MESSAGES[j % 4], v,
                                    "recipient", "named" if j % 2 == 0 else "pronoun", "recipient", j))
        for j in range(32):
            v = values(split, 551 + j)
            v["force"] = FORCES[j % 4]
            rows.append(message_row(split, f"wr10-train-named-{j:02}", TRAIN_MESSAGES[j % 4], v,
                                    "third_party", "named", "third_party_named", j))
    else:
        for t, (source, answer, verb, anchors) in enumerate(DEV_SHORT):
            for j in range(4):
                rows.append(shortening_row(source, answer, verb, anchors, values(split, 761 + t * 4 + j),
                                           split, "diagnostic", t, j, DEV_WRAPPERS))
        for j in range(16):
            v = values(split, 881 + j)
            v["force"] = FORCES[j % 4]
            rows.append(message_row(split, f"wr10-dev-recipient-{j:02}", DEV_MESSAGES[j % 2], v,
                                    "recipient" if j % 2 == 0 else "third_party",
                                    "named" if j % 3 else "pronoun", "mixed_owner", j))
        compact = [
            ("Please remember that every limner must wear a cuff until dusk.",
             "Please remember every limner must wear a cuff until dusk.", "wear",
             ["Please remember", "every limner", "must", "cuff", "until dusk"]),
            ("Everyone in the flint forge should gather at the sienna arcade prior to dusk.",
             "Everyone in the flint forge should gather at the sienna arcade before dusk.", "gather",
             ["Everyone", "flint forge", "should", "sienna arcade", "before dusk"]),
            ("No one may lend Odette's velvet satchel unless the beadle has given consent.",
             "No one may lend Odette's velvet satchel without beadle consent.", "lend",
             ["No one may", "Odette's velvet satchel", "without"]),
            ("Please note that each glazier must retain exactly six tickets until the count ends.",
             "Each glazier must retain exactly six tickets until the count ends.", "retain",
             ["glazier", "must", "exactly six tickets"]),
            ("Do not forget that Magnus might collect his copper reed after rehearsal.",
             "Do not forget Magnus might collect his copper reed after rehearsal.", "collect",
             ["Do not forget", "Magnus", "might", "his copper reed"]),
            ("You can unlatch the onyx stylus only if the provost is present in the sienna arcade.",
             "You can unlatch the onyx stylus only if the provost is in the sienna arcade.", "unlatch",
             ["can", "onyx stylus", "only if", "provost"]),
            ("Please store Isolde's marble cubit flat in order to avoid warping it overnight.",
             "Please store Isolde's marble cubit flat to avoid warping it overnight.", "store",
             ["Please", "Isolde's marble cubit", "flat"]),
            ("It is important that nobody shift the velvet satchel before Friday.",
             "No one shift the velvet satchel before Friday.", "shift",
             ["No one", "velvet satchel", "before Friday"]),
        ]
        for i, (source, answer, verb, protected) in enumerate(compact):
            rows.append(dict(id=f"wr10-dev-compact-{i:02}", split="dev", training_allowed=False,
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
            raise ValueError("Preferred WR10 shortening must pass v3: " + row["id"] + " " + str(grade))
        if row.get("rejected"):
            if row["rejected"] != source:
                raise ValueError("WR10 rejected completion must be the verbatim source: " + row["id"])
            bad = V3.grade_case(meaning_row(source, answer), row["rejected"], lambda r, o: True)
            if bad["status"] == "pass":
                raise ValueError("Verbatim source must not pass v3: " + row["id"])
        if row.get("overedit"):
            if norm(row["overedit"]) in {norm(answer), norm(source)} and row["shortening_group"] not in ("reminder", "reminder_forget"):
                if norm(row["overedit"]) == norm(answer):
                    raise ValueError("Over-edit must differ from the preferred short: " + row["id"])
            over = V3.grade_case(meaning_row(source, answer), row["overedit"], lambda r, o: True)
            if over["status"] == "pass":
                raise ValueError("Over-edit must not pass v3: " + row["id"])
            if row["shortening_group"] in ("reminder", "reminder_forget"):
                if not any(reason.startswith("reminder") for reason in over["reasons"]):
                    raise ValueError("Reminder over-edit must drop the reminder: " + row["id"])
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
        raise ValueError("WR10 uses 96 keep-pass copy-margin shortening pairs")
    if any(r.get("rejected") != r.get("source") for r in short):
        raise ValueError("Each shortening pair must contrast the verbatim source")
    if any(not r.get("overedit") for r in short):
        raise ValueError("Each shortening pair must include an over-edit")
    rows.extend(dict(id="wr10-retain-" + r["id"], split="train", training_allowed=True,
                     family=r["family"], prompt=r["prompt"], answer=r["answer"], loss="sft") for r in retention)
    random.Random(SEED + 10).shuffle(rows)
    sft_bytes(rows)
    return rows


def encode_examples(rows):
    """One trainer example per row. Copy-margin rows carry source and over-edit contrasts."""
    encoded = []
    for r in rows:
        item = dict(id=r["id"], prompt=r["prompt"], answer=r["answer"], family=r["family"],
                    loss_kind=1 if r.get("loss") == "copy_margin" else 0)
        if item["loss_kind"] == 1:
            item["rejected"] = r["rejected"]
            item["overedit"] = r["overedit"]
        encoded.append(item)
    if len(encoded) != 416 or sum(x["loss_kind"] == 1 for x in encoded) != 96:
        raise ValueError("Expected 416 examples and 96 keep-pass pairs")
    if any(x["loss_kind"] == 1 and not x.get("overedit") for x in encoded):
        raise ValueError("Copy-margin row missing over-edit")
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
            if not r.get("overedit"):
                raise ValueError("Copy-margin row must include an over-edit")
    payload = []
    for r in rows:
        item = {k: r[k] for k in ("id", "prompt", "answer") if k in r}
        if r.get("rejected"):
            item["rejected"] = r["rejected"]
            item["overedit"] = r["overedit"]
            item["kind"] = "keep_pass_copy_margin"
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
    new = [r for r in train if not r["id"].startswith("wr10-retain-")] + dev
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
                copy_margin_pairs=len(short), keep_pass_pairs=len(short),
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
