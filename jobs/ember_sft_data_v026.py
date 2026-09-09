"""Format-parity literal-copy curriculum and expanded held-out battery for Ember v0.0.26.

Why this module exists
----------------------
The v0.0.15 curriculum used one rigid template per kind. Four of the nine
held-out diagnostic values have a character-level structure that template can
never emit (``openai/gpt-6-astra`` against ``vendor/ember-xxxx-NNb``,
``.../a7Q9`` against a two-segment URL, ``result.json`` against
``result-xxxx.json``, ``acct_Q7m4-`` against a lowercase-only account stem).
v0.0.21 through v0.0.25 added more rows drawn from those same templates, so no
amount of extra data or loss shaping could reach the missing structure. The
v0.0.25 coverage summary shows the dead end directly: 320 selected ``model_id``
rows produced only 6 distinct target tokens.

This module fixes two separate things.

1. **Format parity.** Every kind now spans several templates chosen so that the
   union covers the structure of every held-out value. ``jobs/
   ember_curriculum_audit_v026.py`` asserts this property.
2. **Metric resolution.** The held-out battery grows from 9 cases to 90, so
   exact-copy moves in steps of ~1.1 points instead of 11.1. A 9-case metric
   cannot distinguish a real gain from a single lucky string, which is what the
   v0.0.16..v0.0.25 "plateau" was actually made of.

Split discipline: train and validation are drawn from one deterministic value
stream with a global used-set, so the two splits are disjoint by construction
without partitioning the leading character. v0.0.15 reserved ``2-9A-H`` for
train and ``J-Z`` for validation, which meant no training target ever began with
a letter in the second half of the alphabet -- an avoidable coverage hole.
"""
from __future__ import annotations

import hashlib

EOT = "<|endoftext|>"
ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
KINDS = ("short_code", "long_code", "digits", "model_id", "url", "path", "entity", "expression", "mixed")

VENDORS = ("openai", "anthropic", "meta", "mistral", "cohere", "deepmind", "ember", "aleph")
FAMILIES = ("gpt", "claude", "llama", "mixtral", "ember", "sol", "orion", "atlas")
CODENAMES = ("astra", "sol", "nimbus", "vector", "harbor", "summit", "zephyr", "orion", "quartz", "delta")
# The two-word entity variant has no trailing code, so its value space is just
# |LEFT| * |RIGHT|. Both lists are sized so the deduplicated stream can still
# fill that variant's share of a 3600-row curriculum.
LEFT_NAMES = (
    "Northfield", "Westhaven", "Rivergate", "Stonebridge", "Clearwater", "Pinecrest",
    "Fairmont", "Ashcroft", "Brookline", "Cedarhurst", "Eastmoor", "Glenwood",
    "Harborview", "Ironvale", "Lakeshore", "Marbleton", "Oakridge", "Redstone",
    "Silverton", "Thornbury",
)
RIGHT_NAMES = (
    "Zephyr", "Orion", "Harbor", "Summit", "Vector", "Nimbus", "Lantern", "Quarry",
    "Beacon", "Cinder", "Draft", "Ember", "Foundry", "Granite", "Meridian", "Torrent",
)
LEAF_NAMES = ("result", "output", "payload", "record", "summary", "trace")

# Variant counts per kind. Index 0 is always the legacy v0.0.15 template so the
# curriculum stays a continuation of what v0.0.20 already learned.
VARIANTS = {
    "short_code": 2,
    "long_code": 2,
    "digits": 4,
    "model_id": 3,
    "url": 3,
    "path": 3,
    "entity": 2,
    "expression": 2,
    "mixed": 3,
}

# The legacy nine. Kept verbatim so the v0.0.20 protection metric stays
# comparable across runs; they are also the four format-gap cases this
# curriculum is built to support.
LEGACY_DIAGNOSTICS = (
    ("short_code", "Q7M4", "R8N5"),
    ("long_code", "V9K2-4R7P", "W8L3-5S6Q"),
    ("digits", "58310429", "69421530"),
    ("model_id", "openai/gpt-6-astra", "openai/gpt-5.6-sol"),
    ("url", "https://example.test/a7Q9", "https://example.test/b8R2"),
    ("path", "/tmp/ember/Q7M4/result.json", "/tmp/ember/R8N5/output.json"),
    ("entity", "Northfield Zephyr", "Westhaven Orion"),
    ("expression", "53*19+7", "61*17+9"),
    ("mixed", "acct_Q7m4-5831", "acct_R8n5-6942"),
)

HELD_OUT_PER_KIND = 9


def _digest(*parts) -> bytes:
    return hashlib.sha256("|".join(["ember-v026", *(str(p) for p in parts)]).encode()).digest()


def _code(seed: bytes, length: int = 4, offset: int = 0) -> str:
    return "".join(ALPHABET[seed[(offset + i) % len(seed)] % len(ALPHABET)] for i in range(length))


def _render(kind: str, variant: int, seed: bytes) -> str:
    """Render one value. Every template here is reachable by the audit."""
    if kind == "short_code":
        return _code(seed, 4) if variant == 0 else _code(seed, 5)
    if kind == "long_code":
        if variant == 0:
            return f"{_code(seed, 4)}-{_code(seed, 4, 8)}"
        return f"{_code(seed, 3)}-{_code(seed, 5, 8)}"
    if kind == "digits":
        n = int.from_bytes(seed[:8], "big")
        if variant == 0:
            return str(n % 90000000 + 10000000)          # 8 digits
        if variant == 1:
            return str(n % 900000 + 100000)              # 6 digits
        if variant == 2:
            return str(n % 9000000000 + 1000000000)      # 10 digits
        # Segmentation stress: a fixed high-frequency prefix plus a free tail,
        # so the model cannot ride a memorised whole-number piece.
        return f"{58310 if seed[0] % 2 else 20264}{n % 10000:04d}"
    if kind == "model_id":
        vendor = VENDORS[seed[0] % len(VENDORS)]
        if variant == 0:
            return f"{vendor}/ember-{_code(seed, 4, 3).lower()}-{10 + seed[1] % 90}b"
        if variant == 1:
            # Structure of the held-out openai/gpt-6-astra.
            return f"{vendor}/{FAMILIES[seed[1] % len(FAMILIES)]}-{2 + seed[2] % 8}-{CODENAMES[seed[3] % len(CODENAMES)]}"
        return (
            f"{vendor}/{FAMILIES[seed[1] % len(FAMILIES)]}-{2 + seed[2] % 8}."
            f"{seed[3] % 10}-{CODENAMES[seed[4] % len(CODENAMES)]}"
        )
    if kind == "url":
        if variant == 0:
            return f"https://example.test/{_code(seed, 4)}/{_code(seed, 4, 8).lower()}"
        if variant == 1:
            # Structure of the held-out https://example.test/a7Q9: one segment,
            # mixed case, no trailing path component.
            body = _code(seed, 4)
            return f"https://example.test/{body[0].lower()}{body[1]}{body[2]}{body[3]}"
        return f"https://example.test/{_code(seed, 4)}"
    if kind == "path":
        if variant == 0:
            return f"/tmp/ember/{_code(seed, 4)}/result-{_code(seed, 4, 8).lower()}.json"
        if variant == 1:
            # Structure of the held-out /tmp/ember/Q7M4/result.json.
            return f"/tmp/ember/{_code(seed, 4)}/{LEAF_NAMES[seed[1] % len(LEAF_NAMES)]}.json"
        return f"/tmp/ember/{_code(seed, 4)}/{LEAF_NAMES[seed[1] % len(LEAF_NAMES)]}-{seed[2] % 100:02d}.json"
    if kind == "entity":
        left = LEFT_NAMES[seed[0] % len(LEFT_NAMES)]
        right = RIGHT_NAMES[seed[1] % len(RIGHT_NAMES)]
        if variant == 0:
            return f"{left} {right} {_code(seed, 4, 2)}"
        return f"{left} {right}"          # structure of the held-out Northfield Zephyr
    if kind == "expression":
        a, b, c = 12 + seed[0] % 80, 2 + seed[1] % 40, 1 + seed[2] % 20
        op = ("*", "+", "-")[seed[3] % 3]
        if variant == 0:
            return f"{a}{op}{b}+{c}"
        return f"{a}{op}{b}{('+', '-')[seed[4] % 2]}{c}"
    if kind == "mixed":
        tail = int.from_bytes(seed[4:6], "big") % 9000 + 1000
        body = _code(seed, 4)
        if variant == 0:
            return f"acct_{body.lower()}-{tail}"
        if variant == 1:
            # Structure of the held-out acct_Q7m4-5831: mixed case in the stem.
            return f"acct_{body[0]}{body[1]}{body[2].lower()}{body[3]}-{tail}"
        return f"acct_{body}-{tail}"
    raise ValueError(kind)


def _held_out_value(kind: str, i: int) -> str:
    variant = i % VARIANTS[kind]
    return _render(kind, variant, _digest("heldout", kind, variant, i))


def _corrupt(value: str, salt: int = 0) -> str:
    """Same-shape corruption used as the teacher-forced comparison completion."""
    chars = list(value)
    positions = [i for i, ch in enumerate(chars) if ch.isalnum()]
    pos = positions[(salt + len(chars)) % len(positions)]
    ch = chars[pos]
    if ch.isdigit():
        chars[pos] = str((int(ch) + 1 + salt % 8) % 10)
    elif ch.islower():
        source = ALPHABET.lower()
        chars[pos] = source[(source.index(ch) + 1 + salt % 8) % len(source)] if ch in source else "x"
    else:
        chars[pos] = ALPHABET[(ALPHABET.index(ch) + 1 + salt % 8) % len(ALPHABET)] if ch in ALPHABET else "X"
    out = "".join(chars)
    return out if out != value else _corrupt(value, salt + 1)


def _build_held_out() -> tuple[tuple[str, str, str], ...]:
    cases = list(LEGACY_DIAGNOSTICS)
    taken = {value for _, value, _ in cases}
    for kind in KINDS:
        made = 0
        i = 0
        while made < HELD_OUT_PER_KIND:
            value = _held_out_value(kind, i)
            i += 1
            if value in taken:
                continue
            corrupt = _corrupt(value, i)
            if corrupt in taken or corrupt == value:
                continue
            taken.add(value)
            cases.append((kind, value, corrupt))
            made += 1
    return tuple(cases)


DIAGNOSTICS = _build_held_out()
HELD_OUT_VALUES = frozenset(value for _, value, _ in DIAGNOSTICS) | frozenset(
    corrupt for _, _, corrupt in DIAGNOSTICS
)


def prompt(value: str, distractor_a: str, distractor_b: str) -> str:
    return (
        "<|system|>\nYou are Ember. Copy TARGET from the current user message exactly. "
        "Do not explain, normalize, calculate, or call a tool. Stop at endoftext.\n"
        f"<|user|>\nIgnore old={distractor_a} and fallback={distractor_b}. TARGET={value}. "
        "Reply with TARGET exactly once.\n<|assistant|>\n"
    )


def completion(value: str) -> str:
    return f"{value}\n{EOT}\n"


def prompt_for(value: str) -> str:
    """Evaluation prompt whose distractor slots match the training distribution.

    v0.0.15 evaluated with the hard-coded distractors ``old=K2P8`` and
    ``fallback=77291``. Neither is reachable by its generator: ``K`` was reserved
    for the validation split and every training ``digits`` value has eight
    digits, so all nine held-out prompts carried a slot pattern the model had
    never seen. Here the distractors are derived from the target, so the
    evaluation prompt is drawn from the training distribution.
    """
    seed = _digest("eval-distractor", value)
    kind_a = KINDS[seed[0] % len(KINDS)]
    kind_b = KINDS[seed[1] % len(KINDS)]
    a = _render(kind_a, seed[2] % VARIANTS[kind_a], _digest("eval-a", value))
    b = _render(kind_b, seed[3] % VARIANTS[kind_b], _digest("eval-b", value))
    return prompt(value, a, b)


def legacy_prompt_for(value: str) -> str:
    """The exact v0.0.15 evaluation prompt, kept so the protected v0.0.20
    metric (exact-copy 4/9, continuation 84.06%) stays comparable run to run."""
    return (
        "<|system|>\nYou are Ember. Copy TARGET from the current user message exactly. "
        "Do not explain, normalize, calculate, or call a tool. Stop at endoftext.\n"
        f"<|user|>\nIgnore old=K2P8 and fallback=77291. TARGET={value}. "
        "Reply with TARGET exactly once.\n<|assistant|>\n"
    )


def _stream(limit: int) -> list[dict]:
    """One deterministic, de-duplicated value stream shared by both splits.

    Position ``i`` belongs to validation when ``i % 9 == 0`` and to train
    otherwise, so the splits are disjoint by construction and neither is
    restricted to a sub-alphabet.
    """
    rows: list[dict] = []
    used: set[str] = set()
    i = 0
    attempts = 0
    while len(rows) < limit:
        attempts += 1
        if attempts > limit * 64 + 4096:
            raise RuntimeError("v0.0.26 value stream failed to reach the requested size")
        kind = KINDS[i % len(KINDS)]
        variant = (i // len(KINDS)) % VARIANTS[kind]
        value = _render(kind, variant, _digest("stream", kind, variant, i))
        i += 1
        if value in used or value in HELD_OUT_VALUES:
            continue
        used.add(value)
        rows.append({"kind": kind, "variant": variant, "value": value, "position": len(rows)})
    return rows


def build_examples(split: str, total: int) -> list[dict]:
    if split not in {"train", "validation"}:
        raise ValueError(split)
    wanted = 10 * total + 512
    stream = _stream(wanted)
    is_validation = [row["position"] % 9 == 0 for row in stream]
    pool = [row for row, v in zip(stream, is_validation) if v == (split == "validation")]
    if len(pool) < total:
        raise RuntimeError(f"v0.0.26 stream produced only {len(pool)} {split} rows, needed {total}")
    pool = pool[:total]
    # Distractors are drawn from the same split, so a validation target never
    # appears inside a training prompt and validation loss stays a measurement of
    # unseen values rather than of values glimpsed in a distractor slot.
    by_value = {row["value"]: row for row in pool}
    values = [row["value"] for row in pool]

    rows = []
    for n, item in enumerate(pool):
        value = item["value"]
        kind = item["kind"]
        seed = _digest("distractor", split, value)
        a = values[(n + 1 + seed[0] % 97) % len(values)]
        b = values[(n + 101 + seed[1] % 89) % len(values)]
        if a == value:
            a = values[(n + 7) % len(values)]
        if b in {value, a}:
            b = values[(n + 211) % len(values)]
        if b == value:
            b = values[(n + 313) % len(values)]
        row = {
            "id": f"{split}-{kind}-v{item['variant']}-{n:05d}",
            "kind": kind,
            "variant": item["variant"],
            "value": value,
            "prompt": prompt(value, a, b),
            "completion": completion(value),
            "focus_terms": [value],
            "distractor_kinds": [by_value[a]["kind"], by_value[b]["kind"]],
        }
        if not row["completion"].startswith(value):
            raise RuntimeError(f"completion does not start with target: {row['id']}")
        if row["prompt"].count(value) != 1:
            raise RuntimeError(f"target must appear exactly once in prompt: {row['id']}")
        rows.append(row)
    return rows


def assert_clean(train: list[dict], validation: list[dict]) -> None:
    train_values = {row["value"] for row in train}
    val_values = {row["value"] for row in validation}
    if train_values & val_values:
        raise RuntimeError(f"train/validation value overlap: {sorted(train_values & val_values)[:5]}")
    leaked = HELD_OUT_VALUES & (train_values | val_values)
    if leaked:
        raise RuntimeError(f"diagnostic held-out leakage: {sorted(leaked)}")
    for rows, name in ((train, "train"), (validation, "validation")):
        seen_kinds = {row["kind"] for row in rows}
        if seen_kinds != set(KINDS):
            raise RuntimeError(f"{name} split is missing kinds: {sorted(set(KINDS) - seen_kinds)}")
        for row in rows:
            # The stream already excludes every held-out target and its corrupted
            # twin, so distractor slots cannot carry one. A held-out value may
            # still appear as a *substring* of a longer legitimate value
            # ("Northfield Zephyr" inside "Northfield Zephyr 7K2M"); that is not
            # leakage, because the copy target of the row is a different string.
            if row["value"] in HELD_OUT_VALUES:
                raise RuntimeError(f"held-out value used as a target: {row['id']}")
            for slot in ("old=", "fallback="):
                start = row["prompt"].index(slot) + len(slot)
                end = row["prompt"].index(" ", start)
                if row["prompt"][start:end].rstrip(".") in HELD_OUT_VALUES:
                    raise RuntimeError(f"held-out value in a distractor slot: {row['id']}")
