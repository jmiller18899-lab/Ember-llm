"""WR8 meaning-checked contrastive shortening; generation grades stay manual."""
from collections import Counter
from string import Formatter
import json, random, re
import ember_meaning_preservation_v3 as V3

VERSION = "ember-writing-repair8-meaning-contrast-v1"
SEED = 431
WRAPPERS = [
    'Shorten this while keeping meaning, force and owners: "{s}"',
    "Give a briefer version that a meaning checker would accept:\n{s}",
    "Make this shorter without changing obligations or reminders: {s}",
    'Condense this, keeping counts, consent and timing: "{s}"',
]
DEV_WRAPPERS = [
    'Rewrite more briefly, preserving meaning: "{s}"',
    "Cut extra words but keep force, owners and reminders:\n{s}",
    "Produce a shorter equivalent: {s}",
    'Trim this without changing obligations: "{s}"',
]
PREFERRED = [
    ("timing_should",
     "Please note that the {group} should inspect the {obj} prior to {time}.",
     "The {group} should inspect the {obj} before {time}.",
     "inspect", ("should", "before")),
    ("consent_unless",
     "No one may take {owner}'s {obj} unless the {role} has given consent.",
     "No one may take {owner}'s {obj} without {role} consent.",
     "take", ("No one may", "without")),
    ("reminder",
     "Please remember that each of the {group} must wait at the {place} until {day}.",
     "Please remember each of the {group} must wait at the {place} until {day}.",
     "wait", ("Please remember", "must", "until")),
    ("consent_only_if",
     "Please note that you may borrow the {obj} only if the {role} has given consent.",
     "You may borrow the {obj} only if the {role} has consented.",
     "borrow", ("may", "only if")),
    ("reminder_forget",
     "Do not forget that {owner}'s {obj} might arrive at the {place} after {time}.",
     "Do not forget {owner}'s {obj} might arrive at the {place} after {time}.",
     "arrive", ("Do not forget", "might", "after")),
    ("exactly",
     "Please note that the {group} must keep exactly {q} {items} in the {place} until {day}.",
     "The {group} must keep exactly {q} {items} in the {place} until {day}.",
     "keep", ("must", "exactly", "until")),
    ("ownership",
     "Please return {owner}'s {obj} to the {place} prior to {day}.",
     "Please return {owner}'s {obj} to the {place} before {day}.",
     "return", ("Please", "before")),
    ("only_after",
     "Please note that you may open the {obj} only after {time} at the {place}.",
     "You may open the {obj} only after {time} at the {place}.",
     "open", ("may", "only after")),
]
DEV_SHORT = [
    ("Please note that the {group} should polish the {obj} prior to {time}.",
     "The {group} should polish the {obj} before {time}.", "polish", ("should", "before")),
    ("No one may move {owner}'s {obj} unless the {role} has given consent.",
     "No one may move {owner}'s {obj} without {role} consent.", "move", ("No one may", "without")),
    ("Please remember that each of the {group} must remain at the {place} until {day}.",
     "Please remember each of the {group} must remain at the {place} until {day}.", "remain", ("Please remember", "must")),
    ("Please note that you may use the {obj} only if the {role} has given consent.",
     "You may use the {obj} only if the {role} has consented.", "use", ("may", "only if")),
    ("Do not forget that {owner}'s {obj} might stay in the {place} after {time}.",
     "Do not forget {owner}'s {obj} might stay in the {place} after {time}.", "stay", ("Do not forget", "might")),
    ("Please note that the {group} must leave exactly {q} {items} at the {place} until {day}.",
     "The {group} must leave exactly {q} {items} at the {place} until {day}.", "leave", ("must", "exactly")),
    ("Please bring {owner}'s {obj} to the {place} prior to {day}.",
     "Please bring {owner}'s {obj} to the {place} before {day}.", "bring", ("Please", "before")),
    ("Please note that you may seal the {obj} only after {time} at the {place}.",
     "You may seal the {obj} only after {time} at the {place}.", "seal", ("may", "only after")),
]
TRAIN_MESSAGES = [
    "I found {owned} and {force} return it on {day}.",
    "I wrapped {owned} and {force} bring it to the {place} before {time}.",
    "I collected {owned} and {force} leave it at the {place} on {day}.",
    "I repaired {owned} and {force} hand it back after {time}.",
]
DEV_MESSAGES = [
    "I catalogued {owned} and {force} send it back on {day}.",
    "I examined {owned} and {force} deliver it to the {place} before {time}.",
]
FORCES = ("can", "should", "might", "will")
GREETINGS = ("Hi {n}, ", "{n}, ", "Hello {n}, ")


def norm(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def values(split, i):
    if split == "train":
        people = [("Cosima", "her"), ("Henrik", "his"), ("Rowan", "their"), ("Paloma", "her"),
                  ("Oswald", "his"), ("Indigo", "their"), ("Beatrix", "her"), ("Casper", "his")]
        owners = {"her": ["Nysa", "Odette", "Petra", "Willa"], "his": ["Tomas", "Ewan", "Giles", "Ruben"],
                  "their": ["Sage", "Robin", "Cameron", "Harper"]}
        objects = ["ivory metronome", "cedar whistle", "coral locket", "moss satchel",
                   "pewter goblet", "linen visor", "onyx bead", "maple mallet"]
        groups = ["luthiers", "glaziers", "farriers", "coppersmiths", "milliners", "chandlers", "thatchers", "coopers"]
        places = ["amber vestibule", "slate quay", "willow terrace", "ochre cellar",
                  "tin annex", "glass atrium", "moss cloister", "ivory loft"]
        roles = ["registrar", "marshal", "warden", "steward"]
        items = ["badges", "wafers", "chits", "seals"]
    else:
        people = [("Heloise", "her"), ("Magnus", "his"), ("Callum", "their"), ("Ottilie", "her")]
        owners = {"her": ["Ines", "Bianca"], "his": ["Folke", "Pieter"], "their": ["Sloane", "Waverly"]}
        objects = ["jade thimble", "tin harmonica", "quartz bobbin", "umber cloak"]
        groups = ["dyers", "cobblers", "hatters", "sawyers"]
        places = ["coral jetty", "pewter nave", "linen loft", "oak galley"]
        roles = ["keeper", "provost", "bailiff", "prefect"]
        items = ["tokens", "ribbons", "tickets", "markers"]
    pick = lambda seq: seq[i % len(seq)]
    n, pronoun = pick(people)
    owner_pronoun = ("her", "his", "their")[i % 3]
    return dict(n=n, pronoun=pronoun, owner=owners[owner_pronoun][(i // 3) % len(owners[owner_pronoun])],
                owner_pronoun=owner_pronoun, obj=pick(objects), group=pick(groups), place=pick(places),
                day=pick(["Thursday", "Saturday", "Monday", "Wednesday", "Friday", "Sunday", "Tuesday"]),
                time=f"{1 + i % 8}:{(i * 17 + 11) % 60:02d} PM", q=str(4 + i % 9),
                items=pick(items), role=pick(roles))


def protected_fields(template, v):
    keys = dict.fromkeys(k for _, k, _, _ in Formatter().parse(template) if k)
    return [v[k] for k in keys]


def shortening_row(source, answer, verb, anchors, v, split, group, t, j, wrappers):
    src, ans = source.format(**v), answer.format(**v)
    return dict(id=f"wr8-{split}-short-{group}-{t:02}-{j:02}", split=split,
                training_allowed=split == "train", family="shortening", shortening_group=group,
                source=src, prompt=wrappers[(t + j) % 4].format(s=src), answer=ans, structure=source,
                action_verb=verb, protected=protected_fields(answer, v) + list(anchors),
                semantic_review_required=True, loss="contrastive" if split == "train" else "eval")


def rejected_for(row):
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
    wrapper = "Send this draft to {n}: {s}" if split == "train" else "Address this note to {n}: {s}"
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
                row = shortening_row(source, answer, verb, anchors, values(split, 17 + t * 12 + j),
                                     split, group, t, j, WRAPPERS)
                row["rejected"] = rejected_for(row)
                rows.append(row)
        for j in range(32):
            v = values(split, 401 + j)
            v["force"] = FORCES[j % 4]
            rows.append(message_row(split, f"wr8-train-recipient-{j:02}", TRAIN_MESSAGES[j % 4], v,
                                    "recipient", "named" if j % 2 == 0 else "pronoun", "recipient", j))
        for j in range(32):
            v = values(split, 501 + j)
            v["force"] = FORCES[j % 4]
            rows.append(message_row(split, f"wr8-train-named-{j:02}", TRAIN_MESSAGES[j % 4], v,
                                    "third_party", "named", "third_party_named", j))
    else:
        for t, (source, answer, verb, anchors) in enumerate(DEV_SHORT):
            for j in range(4):
                rows.append(shortening_row(source, answer, verb, anchors, values(split, 701 + t * 4 + j),
                                           split, "diagnostic", t, j, DEV_WRAPPERS))
        for j in range(16):
            v = values(split, 809 + j)
            v["force"] = FORCES[j % 4]
            rows.append(message_row(split, f"wr8-dev-recipient-{j:02}", DEV_MESSAGES[j % 2], v,
                                    "recipient" if j % 2 == 0 else "third_party",
                                    "named" if j % 3 else "pronoun", "mixed_owner", j))
        compact = [
            ("Please remember that every dyer must wear a sash until dusk.",
             "Please remember every dyer must wear a sash until dusk.", "wear",
             ["Please remember", "every dyer", "must", "sash", "until dusk"]),
            ("Everyone in the pewter choir should gather at the coral jetty prior to dusk.",
             "Everyone in the pewter choir should gather at the coral jetty before dusk.", "gather",
             ["Everyone", "pewter choir", "should", "coral jetty", "before dusk"]),
            ("No one may lend Ottilie's umber cloak unless the keeper has given consent.",
             "No one may lend Ottilie's umber cloak without keeper consent.", "lend",
             ["No one may", "Ottilie's umber cloak", "without"]),
            ("Please note that the hatters must retain exactly six ribbons until the inventory ends.",
             "The hatters must retain exactly six ribbons until the inventory ends.", "retain",
             ["hatters", "must", "exactly six ribbons"]),
            ("Do not forget that Magnus might collect his tin harmonica after rehearsal.",
             "Do not forget Magnus might collect his tin harmonica after rehearsal.", "collect",
             ["Do not forget", "Magnus", "might", "his tin harmonica"]),
            ("You can unlatch the jade thimble only if the provost is present in the pewter nave.",
             "You can unlatch the jade thimble only if the provost is in the pewter nave.", "unlatch",
             ["can", "jade thimble", "only if", "provost"]),
            ("Please store Heloise's quartz bobbin flat in order to avoid warping it overnight.",
             "Please store Heloise's quartz bobbin flat to avoid warping it overnight.", "store",
             ["Please", "Heloise's quartz bobbin", "flat"]),
            ("It is important that nobody shift the umber cloak before Friday.",
             "No one shift the umber cloak before Friday.", "shift",
             ["No one", "umber cloak", "before Friday"]),
        ]
        for i, (source, answer, verb, protected) in enumerate(compact):
            rows.append(dict(id=f"wr8-dev-compact-{i:02}", split="dev", training_allowed=False,
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
            raise ValueError("Preferred WR8 shortening must pass v3: " + row["id"] + " " + str(grade))
        if row.get("rejected"):
            bad = V3.grade_case(meaning_row(source, answer), row["rejected"], lambda r, o: True)
            if bad["status"] == "pass":
                raise ValueError("Rejected WR8 target must not pass v3: " + row["id"])
            if norm(row["rejected"]) == norm(answer):
                raise ValueError("Rejected target matches preferred: " + row["id"])
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
        raise ValueError("WR8 uses 96 contrastive shortening pairs")
    if any(not r.get("rejected") for r in short):
        raise ValueError("Each shortening pair needs a rejected target")
    rows.extend(dict(id="wr8-retain-" + r["id"], split="train", training_allowed=True,
                     family=r["family"], prompt=r["prompt"], answer=r["answer"], loss="sft") for r in retention)
    random.Random(SEED + 8).shuffle(rows)
    sft_bytes(rows)
    return rows


def encode_pairs(rows):
    """Expand contrastive rows into SFT + unlikelihood sequences for the 128-step budget."""
    encoded = []
    for r in rows:
        encoded.append(dict(id=r["id"], prompt=r["prompt"], answer=r["answer"], kind=0, family=r["family"]))
        if r.get("loss") == "contrastive":
            encoded.append(dict(id=r["id"] + "-ul", prompt=r["prompt"], answer=r["rejected"], kind=1, family="shortening"))
    if len(encoded) != 512:
        raise ValueError("Expected 512 encoded sequences, got " + str(len(encoded)))
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
        if r.get("loss") != "contrastive" and norm(r["prompt"]) in prompts:
            raise ValueError("Duplicate training prompt: " + r["prompt"])
        ids.add(r["id"])
        prompts.add(norm(r["prompt"]))
        if r.get("loss") == "contrastive" and (not r.get("rejected") or not str(r["rejected"]).strip()):
            raise ValueError("Contrastive row missing rejected text")
    payload = []
    for r in rows:
        item = {k: r[k] for k in ("id", "prompt", "answer") if k in r}
        if r.get("rejected"):
            item["rejected"] = r["rejected"]
            item["kind"] = "contrastive"
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
    new = [r for r in train if not r["id"].startswith("wr8-retain-")] + dev
    if any(norm(r["prompt"]) in history_prompts or (r.get("source") and norm(r["source"]) in history_sources) for r in new):
        raise ValueError("Previously observed writing prompt/source reused")
    benchmark_grams = set().union(*(fourgrams(s) for s in benchmark_sources)) if benchmark_sources else set()
    for r in train:
        if r["family"] == "shortening" and fourgrams(r["source"]) & benchmark_grams:
            raise ValueError("Training source shares a benchmark four-token frame: " + r["id"])
    short = [r for r in train if r["family"] == "shortening"]
    return dict(train=len(train), encoded_sequences=512, fresh_dev=len(dev),
                historical_rows_checked=len(historical),
                benchmark_shortening_sources_checked=len(benchmark_sources), exact_prompt_source_overlaps=0,
                cross_split_structure_overlap=0, benchmark_four_token_overlaps=0,
                contrastive_pairs=len(short),
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
