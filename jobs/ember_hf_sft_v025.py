# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["huggingface-hub>=1.4", "sentencepiece>=0.2", "torch>=2.4"]
# ///
"""Ember v0.0.25: tokenizer-aware position-conditioned curriculum repair.

v0.0.24 found that four of five remaining failures are sparse at the exact
same-format/same-offset or previous-token transition level. This phase returns
to v0.0.20 best.pt, generates fresh leakage-safe same-format candidates, selects
a balanced tokenizer-aware curriculum at the weak structural offsets, and trains
with continuation hard-mining/margin loss. The held-out expected token IDs and
pieces are never used for selection. Digits additionally use confusable numeric
distractors to force attention to TARGET= rather than generic number patterns.
"""
from __future__ import annotations

import urllib.request

BASE_COMMIT = "8ccd8cf3cd6de9c1b92e9f9a8897772f67d14c15"
CONFIG_COMMIT = "05c8682a0b454c2013fabdd0f4130cffd5dfb25d"
BASE_URL = (
    "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/"
    f"{BASE_COMMIT}/jobs/ember_hf_sft_v016.py"
)

with urllib.request.urlopen(BASE_URL) as response:
    source = response.read().decode("utf-8")


def replace_required(old: str, new: str, *, count: int = -1) -> None:
    global source
    if old not in source:
        raise RuntimeError(f"v0.0.25 transform target missing: {old!r}")
    source = source.replace(old, new, count)


# Repoint the proven v0.0.16 scaffold at v0.0.25 and v0.0.20 best.pt.
replace_required(
    'CONFIG_PIN = "42ed91ac6e6b8ff22873fec6706c4f243f57f2fa"',
    f'CONFIG_PIN = "{CONFIG_COMMIT}"',
    count=1,
)
source = source.replace("0.0.16", "0.0.25")
replace_required("ember_sequence_copy_v0.0.25.json", "ember_position_coverage_v0.0.25.json")
replace_required("sequence-copy-consolidation", "position-conditioned-curriculum-repair")
replace_required(
    'SOURCE_REPO = "Jmiller18899/ember-v0.0.15-t4"',
    'SOURCE_REPO = "Jmiller18899/ember-v0.0.20-t4"',
)
replace_required(
    'cfg.get("source_model_name") != "ember-v0.0.15-t4"',
    'cfg.get("source_model_name") != "ember-v0.0.20-t4"',
)
replace_required("must start from v0.0.15", "must start from v0.0.20")
replace_required(
    'if str(source_cfg.get("version")) != "0.0.15":',
    'if str(source_cfg.get("version")) != "0.0.20":',
)
replace_required("expected a v0.0.15 checkpoint", "expected a v0.0.20 checkpoint")
replace_required("v0.0.15 checkpoint run_id", "v0.0.20 checkpoint run_id")
replace_required("v0.0.15 source state", "v0.0.20 source state")
replace_required("resolve_v015_source", "resolve_v020_source")
replace_required("v015_promotion", "v020_promotion")
replace_required("v015_best_step", "v020_best_step")

# Keep copy-position top-1 explicit in promotion metrics.
replace_required(
    'result["metrics"]["continuation_top1_rate"] = rate\n'
    '    result["gates"]["continuation_top1"] = rate >= float(cfg["minimum_continuation_top1_rate"])\n'
    '    result["passed"] = all(result["gates"].values())',
    'result["metrics"]["continuation_top1_rate"] = rate\n'
    '    result["metrics"]["copy_position_top1_rate"] = rate\n'
    '    result["gates"]["continuation_top1"] = rate >= float(cfg["minimum_continuation_top1_rate"])\n'
    '    result["gates"]["copy_position_top1"] = rate >= float(cfg["minimum_copy_position_top1_rate"])\n'
    '    result["passed"] = all(result["gates"].values())',
)

helpers = r'''
def _fresh_value(data, kind, index, banned):
    for bump in (0, 1000000, 2000000, 3000000):
        value = data._value("train", kind, index + bump)
        if value not in banned:
            return value
    raise RuntimeError(f"v0.0.25 could not generate leakage-safe value for {kind}")


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


def build_hard_pools(train, cfg):
    hard_kinds = set(str(k) for k in cfg["hard_kinds"])
    warm_kinds = {"short_code", "digits", "expression", "mixed", "long_code", "entity"}
    pools = {"neighbor": [], "hard": [], "warm": [], "all": list(range(len(train)))}
    for i, item in enumerate(train):
        row = item["row"]
        kind = str(row.get("kind", ""))
        if bool(row.get("failure_neighbor")):
            pools["neighbor"].append(i)
        if kind in hard_kinds:
            pools["hard"].append(i)
        if kind in warm_kinds:
            pools["warm"].append(i)
    if not pools["neighbor"] or not pools["hard"] or not pools["warm"] or not pools["all"]:
        raise RuntimeError("v0.0.25 hard-example pool construction failed")
    return pools


def _draw(pool, count, generator, torch):
    if count <= 0:
        return []
    local = torch.randint(len(pool), (count,), generator=generator).tolist()
    return [pool[j] for j in local]


def sample_hard_mined_indices(train, pools, step, total_steps, batch_size, cfg,
                              generator, model, device, torch):
    import torch.nn.functional as F
    multiplier = max(2, int(cfg["hard_candidate_multiplier"]))
    candidate_count = max(batch_size, batch_size * multiplier)
    progress = (step + 1) / max(1, total_steps)
    neighbor_start = float(cfg["neighbor_fraction_start"])
    neighbor_end = float(cfg["neighbor_fraction_end"])
    neighbor_fraction = neighbor_start + (neighbor_end - neighbor_start) * progress
    neighbor_count = min(candidate_count - 1, max(1, int(round(candidate_count * neighbor_fraction))))
    remaining = candidate_count - neighbor_count
    hard_start = float(cfg["hard_kind_fraction_start"])
    hard_end = float(cfg["hard_kind_fraction_end"])
    hard_fraction = hard_start + (hard_end - hard_start) * progress
    hard_count = min(remaining, max(0, int(round(remaining * hard_fraction))))
    general_count = remaining - hard_count
    general_pool = pools["warm"] if progress <= float(cfg["warm_general_fraction"]) else pools["all"]
    candidates = _draw(pools["neighbor"], neighbor_count, generator, torch)
    candidates += _draw(pools["hard"], hard_count, generator, torch)
    candidates += _draw(general_pool, general_count, generator, torch)
    x, y, weights = tuple(torch.stack([train[i][key] for i in candidates]).to(device) for key in ("x", "y", "w"))
    copy_weight = float(cfg["copy_token_weight"])
    target_mask = y.ne(-100) & weights.ge(copy_weight - 1e-6)
    with torch.no_grad():
        logits, _ = model(x, None)
        pred = torch.argmax(logits, dim=-1)
        wrong = pred.ne(y) & target_mask
        denom = target_mask.sum(dim=1).clamp_min(1)
        error_rate = wrong.sum(dim=1).float() / denom.float()
        safe_y = y.clamp_min(0)
        correct = logits.float().gather(-1, safe_y.unsqueeze(-1)).squeeze(-1)
        top2_values, top2_indices = torch.topk(logits.float(), k=2, dim=-1)
        best_is_correct = top2_indices[..., 0].eq(y)
        best_wrong = torch.where(best_is_correct, top2_values[..., 1], top2_values[..., 0])
        deficit = F.relu(float(cfg["margin"]) - (correct - best_wrong)) * target_mask
        deficit_mean = deficit.sum(dim=1) / denom.float()
        row_failed = wrong.any(dim=1).float()
        sequence_boost = 1.0 + row_failed * (float(cfg["sequence_failure_multiplier"]) - 1.0)
        hardness = (error_rate * 2.0 + deficit_mean) * sequence_boost
        chosen = torch.topk(hardness, k=min(batch_size, len(candidates))).indices.tolist()
    return [candidates[j] for j in chosen]


def margin_weighted_loss(model, x, y, weights, cfg, torch):
    import torch.nn.functional as F
    logits, _ = model(x, None)
    raw = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction="none").reshape_as(y)
    active = y.ne(-100)
    denom = (weights * active).sum().clamp_min(1.0)
    ce = (raw * weights * active).sum() / denom
    copy_weight = float(cfg["copy_token_weight"])
    target_mask = active & weights.ge(copy_weight - 1e-6)
    if not bool(target_mask.any().item()):
        return ce
    copy_logits = logits[target_mask].float()
    copy_targets = y[target_mask]
    correct = copy_logits.gather(1, copy_targets.unsqueeze(1)).squeeze(1)
    top2_values, top2_indices = torch.topk(copy_logits, k=2, dim=1)
    best_is_correct = top2_indices[:, 0].eq(copy_targets)
    best_wrong = torch.where(best_is_correct, top2_values[:, 1], top2_values[:, 0])
    gap = correct - best_wrong
    margin_term = F.relu(float(cfg["margin"]) - gap)
    hard = (~best_is_correct).to(margin_term.dtype)
    multiplier = 1.0 + hard * (float(cfg["hard_error_multiplier"]) - 1.0)
    margin_loss = (margin_term * multiplier).mean()
    return ce + float(cfg["margin_loss_weight"]) * margin_loss

'''
replace_required("def score(diag: dict, val_loss: float):", helpers + "def score(diag: dict, val_loss: float):")

# Build baseline first, then append only tokenizer-balanced, leakage-safe coverage rows.
replace_required(
    'train = [base.encode_row(tokenizer, row, cfg, torch) for row in train_rows]\n'
    '        val = [base.encode_row(tokenizer, row, cfg, torch) for row in val_rows]\n'
    '        baseline = diagnostic(model, tokenizer, cfg, "cpu", torch, base)',
    'val = [base.encode_row(tokenizer, row, cfg, torch) for row in val_rows]\n'
    '        baseline = diagnostic(model, tokenizer, cfg, "cpu", torch, base)\n'
    '        coverage_rows, coverage_summary = build_position_coverage_rows(data, tokenizer, base, cfg, train_rows, val_rows)\n'
    '        train_rows = list(train_rows) + coverage_rows\n'
    '        train = [base.encode_row(tokenizer, row, cfg, torch) for row in train_rows]\n'
    '        hard_pools = build_hard_pools(train, cfg)',
)

replace_required(
    '"baseline": baseline["metrics"],\n            "hf": {',
    '"baseline": baseline["metrics"],\n'
    '            "position_conditioned_curriculum": {\n'
    '                "coverage_rows": len(coverage_rows),\n'
    '                "coverage_summary": coverage_summary,\n'
    '                "held_out_values_added_to_training": False,\n'
    '                "uses_held_out_expected_token_ids": False,\n'
    '            },\n'
    '            "hf": {',
    count=1,
)

# Exercise tokenizer-aware hard mining on CPU before any paid T4 launch.
replace_required(
    'if args.preflight_only:\n'
    '            print("EMBER_HF_V016_PREFLIGHT=PASS", flush=True)\n'
    '            return',
    'if args.preflight_only:\n'
    '            probe_generator = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))\n'
    '            probe = sample_hard_mined_indices(train, hard_pools, 0, int(cfg["max_steps"]), 2, cfg, probe_generator, model, "cpu", torch)\n'
    '            if len(probe) != 2:\n'
    '                raise RuntimeError("v0.0.25 hard-mining preflight returned wrong batch size")\n'
    '            print("EMBER_V025_POSITION_COVERAGE_PREFLIGHT=PASS", flush=True)\n'
    '            print("EMBER_HF_V025_PREFLIGHT=PASS", flush=True)\n'
    '            return',
    count=1,
)

replace_required(
    'ids = torch.randint(len(train), (int(cfg["batch_size"]),), generator=generator).tolist()',
    'ids = sample_hard_mined_indices(train, hard_pools, step, max_steps, int(cfg["batch_size"]), cfg, generator, model, device, torch)',
)
replace_required(
    'loss = base.weighted_loss(model, x, y, w, torch) / accum',
    'loss = margin_weighted_loss(model, x, y, w, cfg, torch) / accum',
)

# A checkpoint with equal held-out diagnostics is not allowed to replace the protected baseline.
replace_required(
    'best_score = None\n        best_step = -1\n        history = []',
    'best_score = score(baseline, -1e9)\n'
    '        best_step = -1\n'
    '        history = []\n'
    '        save_checkpoint(best_path, model=model, optimizer=optimizer, tokenizer=tokenizer, model_config=model.cfg, train_config=cfg, step=-1, best_val_loss=float("inf"), run_id=run_id)',
    count=1,
)

source = source.replace("EMBER_HF_V016_", "EMBER_HF_V025_")
source = source.replace("EMBER_V016_", "EMBER_V025_")
print("EMBER_V025_POSITION_COVERAGE_TRANSFORM=PASS", flush=True)
exec(compile(source, "ember_hf_sft_v025_wrapper_runtime.py", "exec"), {"__name__": "__main__"})
