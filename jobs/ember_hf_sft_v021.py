# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["huggingface-hub>=1.4", "sentencepiece>=0.2", "torch>=2.4"]
# ///
"""Ember v0.0.21: train on fresh neighbors of the held-out formats still failing.

The diagnostic strings remain held out. At runtime the v0.0.20 best checkpoint is
scored on the nine diagnostics, failed cases are mapped to their copy-format
kinds, and fresh same-format training rows are generated with distant deterministic
indices. Hard mining then strongly samples those failure-neighbor rows while the
margin objective remains continuation-only.
"""
from __future__ import annotations

import urllib.request

BASE_COMMIT = "8ccd8cf3cd6de9c1b92e9f9a8897772f67d14c15"
CONFIG_COMMIT = "8a73afd77a0e6e0cf0532ad8d0bec52cc6d25807"
BASE_URL = (
    "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/"
    f"{BASE_COMMIT}/jobs/ember_hf_sft_v016.py"
)

with urllib.request.urlopen(BASE_URL) as response:
    source = response.read().decode("utf-8")


def replace_required(old: str, new: str, *, count: int = -1) -> None:
    global source
    if old not in source:
        raise RuntimeError(f"v0.0.21 transform target missing: {old!r}")
    source = source.replace(old, new, count)


# Repoint the tested v0.0.16 scaffold at v0.0.21 and v0.0.20 best.pt.
replace_required(
    'CONFIG_PIN = "42ed91ac6e6b8ff22873fec6706c4f243f57f2fa"',
    f'CONFIG_PIN = "{CONFIG_COMMIT}"',
    count=1,
)
source = source.replace("0.0.16", "0.0.21")
replace_required("ember_sequence_copy_v0.0.21.json", "ember_failure_neighbor_v0.0.21.json")
replace_required("sequence-copy-consolidation", "failure-neighbor-continuation-hard-mining")
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
source = source.replace("EMBER_HF_V016_", "EMBER_HF_V021_")
source = source.replace("EMBER_V016_", "EMBER_V021_")

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
def diagnostic_failure_kinds(baseline, base, data):
    cases = baseline.get("cases") or []
    if len(cases) != len(base.DIAGNOSTICS) or len(data.KINDS) != len(base.DIAGNOSTICS):
        raise RuntimeError("v0.0.21 diagnostic/kind cardinality changed")
    exact_by_value = {str(row.get("value")): bool(row.get("exact")) for row in cases}
    failed = []
    for idx, pair in enumerate(base.DIAGNOSTICS):
        value = str(pair[0])
        if value not in exact_by_value:
            raise RuntimeError(f"v0.0.21 diagnostic result missing {value!r}")
        if not exact_by_value[value]:
            failed.append(str(data.KINDS[idx]))
    if not failed:
        failed = [str(k) for k in data.KINDS]
    return failed


def _fresh_value(data, kind, index, banned):
    for bump in (0, 1000000, 2000000, 3000000):
        value = data._value("train", kind, index + bump)
        if value not in banned:
            return value
    raise RuntimeError(f"v0.0.21 could not generate leakage-safe value for {kind}")


def build_failure_neighbor_rows(data, failed_kinds, cfg, train_rows, val_rows):
    held = set(str(v) for v in data.HELD_OUT_VALUES)
    banned = held | {str(r["value"]) for r in train_rows} | {str(r["value"]) for r in val_rows}
    rows = []
    per_kind = int(cfg["neighbor_examples_per_failed_kind"])
    base_index = int(cfg["neighbor_index_base"])
    kinds = list(data.KINDS)
    for kind_offset, kind in enumerate(failed_kinds):
        if kind not in kinds:
            raise RuntimeError(f"v0.0.21 unknown failed kind {kind!r}")
        other_kind = kinds[(kinds.index(kind) + 3) % len(kinds)]
        for j in range(per_kind):
            i = base_index + kind_offset * 10000 + j
            value = _fresh_value(data, kind, i, banned)
            banned.add(value)
            distractor_a = _fresh_value(data, kind, i + per_kind + 17, banned)
            banned.add(distractor_a)
            distractor_b = _fresh_value(data, other_kind, i + per_kind + 31, banned)
            banned.add(distractor_b)
            if value in held or distractor_a in held or distractor_b in held:
                raise RuntimeError("v0.0.21 held-out diagnostic leakage in neighbor row")
            row = {
                "id": f"neighbor-{kind}-{j:05d}",
                "kind": kind,
                "value": value,
                "prompt": data.prompt(value, distractor_a, distractor_b),
                "completion": data.completion(value),
                "focus_terms": [value],
                "failure_neighbor": True,
            }
            if row["prompt"].count(value) != 1 or not row["completion"].startswith(value):
                raise RuntimeError(f"v0.0.21 malformed neighbor row {row['id']}")
            rows.append(row)
    if not rows:
        raise RuntimeError("v0.0.21 generated no failure-neighbor rows")
    return rows


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
        raise RuntimeError("v0.0.21 hard-example pool construction failed")
    return pools


def _draw(pool, count, generator, torch):
    if count <= 0:
        return []
    local = torch.randint(len(pool), (count,), generator=generator).tolist()
    return [pool[j] for j in local]


def sample_hard_mined_indices(train, pools, step, total_steps, batch_size, cfg,
                              generator, model, device, torch):
    """Mine continuation mistakes, heavily sampling neighbors of failed formats."""
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

    x, y, weights = tuple(
        torch.stack([train[i][key] for i in candidates]).to(device)
        for key in ("x", "y", "w")
    )
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
    """Weighted CE plus continuation-only correct-vs-best-wrong margin."""
    import torch.nn.functional as F

    logits, _ = model(x, None)
    raw = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(y)
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

# Build baseline first, derive failure kinds, then append only fresh same-format neighbors.
replace_required(
    'train = [base.encode_row(tokenizer, row, cfg, torch) for row in train_rows]\n'
    '        val = [base.encode_row(tokenizer, row, cfg, torch) for row in val_rows]\n'
    '        baseline = diagnostic(model, tokenizer, cfg, "cpu", torch, base)',
    'val = [base.encode_row(tokenizer, row, cfg, torch) for row in val_rows]\n'
    '        baseline = diagnostic(model, tokenizer, cfg, "cpu", torch, base)\n'
    '        failed_kinds = diagnostic_failure_kinds(baseline, base, data)\n'
    '        neighbor_rows = build_failure_neighbor_rows(data, failed_kinds, cfg, train_rows, val_rows)\n'
    '        train_rows = list(train_rows) + neighbor_rows\n'
    '        train = [base.encode_row(tokenizer, row, cfg, torch) for row in train_rows]\n'
    '        hard_pools = build_hard_pools(train, cfg)',
)

# Surface the dynamically derived curriculum in preflight output without exposing held-out targets.
replace_required(
    '"baseline": baseline["metrics"],\n            "hf": {',
    '"baseline": baseline["metrics"],\n'
    '            "failure_neighbor_curriculum": {\n'
    '                "failed_kinds": failed_kinds,\n'
    '                "neighbor_rows": len(neighbor_rows),\n'
    '                "held_out_values_added_to_training": False,\n'
    '            },\n'
    '            "hf": {',
    count=1,
)

# Exercise dynamic mining during CPU preflight before any paid T4 launch.
replace_required(
    'if args.preflight_only:\n'
    '            print("EMBER_HF_V021_PREFLIGHT=PASS", flush=True)\n'
    '            return',
    'if args.preflight_only:\n'
    '            probe_generator = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))\n'
    '            probe = sample_hard_mined_indices(train, hard_pools, 0, int(cfg["max_steps"]), 2, cfg, probe_generator, model, "cpu", torch)\n'
    '            if len(probe) != 2:\n'
    '                raise RuntimeError(f"v0.0.21 hard-mining preflight returned {len(probe)} rows")\n'
    '            if not failed_kinds or not neighbor_rows:\n'
    '                raise RuntimeError("v0.0.21 failure-neighbor curriculum is empty")\n'
    '            print("EMBER_V021_FAILURE_NEIGHBOR_PREFLIGHT=PASS", flush=True)\n'
    '            print("EMBER_HF_V021_PREFLIGHT=PASS", flush=True)\n'
    '            return',
)

replace_required(
    'ids = torch.randint(len(train), (int(cfg["batch_size"]),), generator=generator).tolist()',
    'ids = sample_hard_mined_indices(train, hard_pools, step, max_steps, int(cfg["batch_size"]), cfg, generator, model, device, torch)',
)
replace_required(
    'loss = base.weighted_loss(model, x, y, w, torch) / accum',
    'loss = margin_weighted_loss(model, x, y, w, cfg, torch) / accum',
)

exec(compile(source, "ember_hf_sft_v021_runtime.py", "exec"), {"__name__": "__main__"})
