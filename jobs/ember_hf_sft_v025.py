# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["huggingface-hub>=1.4", "sentencepiece>=0.2", "torch>=2.4"]
# ///
"""Ember v0.0.25: repair sparse token/position coverage without held-out leakage.

v0.0.24 showed that four of five remaining held-out failures are sparse at the
same-format/same-offset or previous-token transition level. This phase returns
to v0.0.20 best.pt and builds a tokenizer-aware, leakage-safe curriculum from
fresh same-format examples. Selection balances token IDs and previous->current
token transitions at the weak structural offsets without using the held-out
expected token IDs/pieces themselves. Digits additionally use confusable numeric
distractors to force attention to TARGET= rather than generic number patterns.
"""
from __future__ import annotations

import urllib.request

BASE_COMMIT = "b2bca035fa1b2d172c0f297e5ad1733e56245b90"
CONFIG_COMMIT = "05c8682a0b454c2013fabdd0f4130cffd5dfb25d"
BASE_URL = (
    "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/"
    f"{BASE_COMMIT}/jobs/ember_hf_sft_v021.py"
)

with urllib.request.urlopen(BASE_URL) as response:
    source = response.read().decode("utf-8")


def replace_required(old: str, new: str, *, count: int = -1) -> None:
    global source
    if old not in source:
        raise RuntimeError(f"v0.0.25 transform target missing: {old!r}")
    source = source.replace(old, new, count)


replace_required(
    'CONFIG_COMMIT = "8a73afd77a0e6e0cf0532ad8d0bec52cc6d25807"',
    f'CONFIG_COMMIT = "{CONFIG_COMMIT}"',
    count=1,
)
source = source.replace("0.0.21", "0.0.25")
replace_required("ember_failure_neighbor_v0.0.25.json", "ember_position_coverage_v0.0.25.json")
replace_required("failure-neighbor-continuation-hard-mining", "position-conditioned-curriculum-repair")
source = source.replace("EMBER_V021_", "EMBER_V025_")
source = source.replace("EMBER_HF_V021_", "EMBER_HF_V025_")

coverage_helpers = r'''
def _v025_digit_value(index, salt=0):
    import hashlib
    d = hashlib.sha256(f"ember-v025-digits|{index}|{salt}".encode()).digest()
    n = int.from_bytes(d[:8], "big") % 90000000 + 10000000
    return str(n)


def _v025_confusable(value, index, salt):
    chars = list(value)
    pos = (index + salt * 3) % len(chars)
    bump = 1 + ((index + salt) % 8)
    chars[pos] = str((int(chars[pos]) + bump) % 10)
    out = "".join(chars)
    if out == value:
        chars[pos] = str((int(chars[pos]) + 1) % 10)
        out = "".join(chars)
    return out


def _v025_candidate_row(data, kind, index, banned, cfg):
    if kind == "digits" and bool(cfg.get("digits_confusable_distractors", False)):
        for bump in (0, 1000000, 2000000, 3000000):
            value = _v025_digit_value(index + bump, 0)
            da = _v025_confusable(value, index + bump, 1)
            db = _v025_confusable(value, index + bump, 2)
            if value not in banned and da not in banned and db not in banned and len({value, da, db}) == 3:
                break
        else:
            raise RuntimeError("v0.0.25 could not build leakage-safe confusable digits row")
    else:
        value = _fresh_value(data, kind, index, banned)
        da = _fresh_value(data, kind, index + 100003, banned | {value})
        kinds = list(data.KINDS)
        other = kinds[(kinds.index(kind) + 3) % len(kinds)]
        db = _fresh_value(data, other, index + 200009, banned | {value, da})
    held = set(str(v) for v in data.HELD_OUT_VALUES)
    if value in held or da in held or db in held:
        raise RuntimeError("v0.0.25 held-out diagnostic leakage in coverage candidate")
    row = {
        "id": f"coverage-{kind}-{index:07d}",
        "kind": kind,
        "value": value,
        "prompt": data.prompt(value, da, db),
        "completion": data.completion(value),
        "focus_terms": [value],
        "failure_neighbor": True,
        "position_coverage": True,
    }
    if row["prompt"].count(value) != 1 or not row["completion"].startswith(value):
        raise RuntimeError(f"v0.0.25 malformed coverage row {row['id']}")
    return row


def _v025_signature(tokenizer, base, row, offset):
    prompt_ids = list(tokenizer.encode(row["prompt"]))
    full = list(tokenizer.encode(row["prompt"] + row["completion"]))
    if full[:len(prompt_ids)] != prompt_ids:
        return None
    eot_id = int(tokenizer.encode(base.EOT)[-1])
    eot_positions = [i for i in range(len(prompt_ids), len(full)) if int(full[i]) == eot_id]
    if not eot_positions:
        return None
    first_target = len(prompt_ids) - 1
    eos_target = eot_positions[0] - 1
    positions = list(range(first_target + 1, eos_target))
    if offset < 0 or offset >= len(positions):
        return None
    y = full[1:]
    pos = positions[offset]
    cur = int(y[pos])
    prev_pos = positions[offset - 1] if offset > 0 else first_target
    prev = int(y[prev_pos])
    return prev, cur, len(positions)


def _v025_balanced_select(infos, limit):
    selected = []
    token_counts = {}
    pair_counts = {}
    remaining = list(infos)
    while remaining and len(selected) < limit:
        best_i = min(
            range(len(remaining)),
            key=lambda j: (
                token_counts.get(remaining[j]["token_id"], 0),
                pair_counts.get((remaining[j]["prev_id"], remaining[j]["token_id"]), 0),
                remaining[j]["serial"],
            ),
        )
        info = remaining.pop(best_i)
        selected.append(info)
        tok = info["token_id"]
        pair = (info["prev_id"], tok)
        token_counts[tok] = token_counts.get(tok, 0) + 1
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return selected, token_counts, pair_counts


def build_position_coverage_rows(data, tokenizer, base, cfg, train_rows, val_rows):
    held = set(str(v) for v in data.HELD_OUT_VALUES)
    banned = held | {str(r["value"]) for r in train_rows} | {str(r["value"]) for r in val_rows}
    offsets = {str(k): int(v) for k, v in cfg["coverage_offsets"].items()}
    candidate_n = int(cfg["coverage_candidate_examples_per_kind"])
    selected_n = int(cfg["coverage_examples_per_kind"])
    base_index = int(cfg["coverage_index_base"])
    all_selected = []
    summary = {}
    serial = 0

    for kind_i, (kind, offset) in enumerate(offsets.items()):
        infos = []
        local_banned = set(banned)
        for j in range(candidate_n):
            index = base_index + kind_i * 100000 + j
            row = _v025_candidate_row(data, kind, index, local_banned, cfg)
            local_banned.add(str(row["value"]))
            sig = _v025_signature(tokenizer, base, row, offset)
            if sig is None:
                continue
            prev_id, token_id, continuation_tokens = sig
            infos.append({
                "row": row,
                "prev_id": prev_id,
                "token_id": token_id,
                "continuation_tokens": continuation_tokens,
                "serial": serial,
            })
            serial += 1
        if len(infos) < selected_n:
            raise RuntimeError(f"v0.0.25 {kind} produced only {len(infos)} usable coverage candidates")
        chosen, token_counts, pair_counts = _v025_balanced_select(infos, selected_n)
        distinct_tokens = len(token_counts)
        distinct_pairs = len(pair_counts)
        if distinct_tokens < int(cfg["minimum_distinct_tokens_per_offset"]):
            raise RuntimeError(f"v0.0.25 {kind} offset {offset} has only {distinct_tokens} distinct target tokens")
        if distinct_pairs < int(cfg["minimum_distinct_pairs_per_offset"]):
            raise RuntimeError(f"v0.0.25 {kind} offset {offset} has only {distinct_pairs} distinct prev->target pairs")
        rows = [item["row"] for item in chosen]
        for row in rows:
            if str(row["value"]) in held:
                raise RuntimeError("v0.0.25 held-out target leaked into selected coverage rows")
        all_selected.extend(rows)
        summary[kind] = {
            "offset": offset,
            "candidates": len(infos),
            "selected": len(rows),
            "distinct_token_ids": distinct_tokens,
            "distinct_prev_token_pairs": distinct_pairs,
            "max_selected_per_token": max(token_counts.values()) if token_counts else 0,
            "max_selected_per_pair": max(pair_counts.values()) if pair_counts else 0,
        }
        banned |= {str(r["value"]) for r in rows}

    if not all_selected:
        raise RuntimeError("v0.0.25 selected no position-conditioned coverage rows")
    return all_selected, summary

'''
replace_required("def build_hard_pools(train, cfg):", coverage_helpers + "def build_hard_pools(train, cfg):", count=1)

old_build = '''val = [base.encode_row(tokenizer, row, cfg, torch) for row in val_rows]
        baseline = diagnostic(model, tokenizer, cfg, "cpu", torch, base)
        failed_kinds = diagnostic_failure_kinds(baseline, base, data)
        neighbor_rows = build_failure_neighbor_rows(data, failed_kinds, cfg, train_rows, val_rows)
        train_rows = list(train_rows) + neighbor_rows
        train = [base.encode_row(tokenizer, row, cfg, torch) for row in train_rows]
        hard_pools = build_hard_pools(train, cfg)'''
new_build = '''val = [base.encode_row(tokenizer, row, cfg, torch) for row in val_rows]
        baseline = diagnostic(model, tokenizer, cfg, "cpu", torch, base)
        coverage_rows, coverage_summary = build_position_coverage_rows(data, tokenizer, base, cfg, train_rows, val_rows)
        train_rows = list(train_rows) + coverage_rows
        train = [base.encode_row(tokenizer, row, cfg, torch) for row in train_rows]
        hard_pools = build_hard_pools(train, cfg)'''
replace_required(old_build, new_build, count=1)

old_report = '''"failure_neighbor_curriculum": {
                "failed_kinds": failed_kinds,
                "neighbor_rows": len(neighbor_rows),
                "held_out_values_added_to_training": False,
            },'''
new_report = '''"position_conditioned_curriculum": {
                "coverage_rows": len(coverage_rows),
                "coverage_summary": coverage_summary,
                "held_out_values_added_to_training": False,
                "uses_held_out_expected_token_ids": False,
            },'''
replace_required(old_report, new_report, count=1)

# Protect the v0.0.20 baseline: an equal diagnostic score is not allowed to
# replace best.pt merely because validation loss changed.
replace_required(
    'best_score = None\n        best_step = -1\n        history = []',
    'best_score = score(baseline, -1e9)\n'
    '        best_step = -1\n'
    '        history = []\n'
    '        save_checkpoint(best_path, model=model, optimizer=optimizer, tokenizer=tokenizer, model_config=model.cfg, train_config=cfg, step=-1, best_val_loss=float("inf"), run_id=run_id)',
    count=1,
)

print("EMBER_V025_POSITION_COVERAGE_TRANSFORM=PASS", flush=True)
exec(compile(source, "ember_hf_sft_v025_wrapper_runtime.py", "exec"), {"__name__": "__main__"})
