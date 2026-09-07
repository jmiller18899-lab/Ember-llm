# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["huggingface-hub>=1.4", "sentencepiece>=0.2", "torch>=2.4"]
# ///
"""Ember v0.0.19: dynamically mine current copy mistakes and push them past argmax.

This phase resumes from the v0.0.18 best checkpoint. Each optimizer micro-step
samples a larger candidate set, scores current TARGET-token argmax errors without
gradients, and trains on the hardest examples. Harder literal formats are
progressively oversampled while the held-out diagnostic set remains unchanged.
"""
from __future__ import annotations

import urllib.request

BASE_COMMIT = "8ccd8cf3cd6de9c1b92e9f9a8897772f67d14c15"
CONFIG_COMMIT = "c29d2a2f15e3bbbf478e0643dcad9151c3ede78f"
BASE_URL = (
    "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/"
    f"{BASE_COMMIT}/jobs/ember_hf_sft_v016.py"
)

with urllib.request.urlopen(BASE_URL) as response:
    source = response.read().decode("utf-8")


def replace_required(old: str, new: str, *, count: int = -1) -> None:
    global source
    if old not in source:
        raise RuntimeError(f"v0.0.19 transform target missing: {old!r}")
    source = source.replace(old, new, count)


# Repoint the tested scaffold at v0.0.19 and the v0.0.18 best checkpoint.
replace_required(
    'CONFIG_PIN = "42ed91ac6e6b8ff22873fec6706c4f243f57f2fa"',
    f'CONFIG_PIN = "{CONFIG_COMMIT}"',
    count=1,
)
source = source.replace("0.0.16", "0.0.19")
replace_required("ember_sequence_copy_v0.0.19.json", "ember_margin_copy_v0.0.19.json")
replace_required("sequence-copy-consolidation", "hard-example-margin-copy")
replace_required(
    'SOURCE_REPO = "Jmiller18899/ember-v0.0.15-t4"',
    'SOURCE_REPO = "Jmiller18899/ember-v0.0.18-t4"',
)
replace_required(
    'cfg.get("source_model_name") != "ember-v0.0.15-t4"',
    'cfg.get("source_model_name") != "ember-v0.0.18-t4"',
)
replace_required("must start from v0.0.15", "must start from v0.0.18")
replace_required(
    'if str(source_cfg.get("version")) != "0.0.15":',
    'if str(source_cfg.get("version")) != "0.0.18":',
)
replace_required("expected a v0.0.15 checkpoint", "expected a v0.0.18 checkpoint")
replace_required("v0.0.15 checkpoint run_id", "v0.0.18 checkpoint run_id")
replace_required("v0.0.15 source state", "v0.0.18 source state")
replace_required("resolve_v015_source", "resolve_v018_source")
replace_required("v015_promotion", "v018_promotion")
replace_required("v015_best_step", "v018_best_step")
source = source.replace("EMBER_HF_V016_", "EMBER_HF_V019_")
source = source.replace("EMBER_V016_", "EMBER_V019_")

# Surface held-out TARGET-position top-1 as an explicit promotion metric.
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
def build_hard_pools(train, cfg):
    hard_kinds = set(str(k) for k in cfg["hard_kinds"])
    warm_kinds = {"short_code", "digits", "expression", "mixed", "long_code", "entity"}
    pools = {"hard": [], "warm": [], "all": list(range(len(train)))}
    for i, item in enumerate(train):
        kind = str(item["row"].get("kind", ""))
        if kind in hard_kinds:
            pools["hard"].append(i)
        if kind in warm_kinds:
            pools["warm"].append(i)
    if not pools["hard"] or not pools["warm"] or not pools["all"]:
        raise RuntimeError("v0.0.19 hard-example pool construction failed")
    return pools


def _draw(pool, count, generator, torch):
    if count <= 0:
        return []
    local = torch.randint(len(pool), (count,), generator=generator).tolist()
    return [pool[j] for j in local]


def sample_hard_mined_indices(train, pools, step, total_steps, batch_size, cfg,
                              generator, model, device, torch):
    """Choose the currently hardest TARGET-copy rows from an oversampled candidate set."""
    import torch.nn.functional as F

    multiplier = max(2, int(cfg["hard_candidate_multiplier"]))
    candidate_count = max(batch_size, batch_size * multiplier)
    progress = (step + 1) / max(1, total_steps)
    hard_start = float(cfg["hard_kind_fraction_start"])
    hard_end = float(cfg["hard_kind_fraction_end"])
    hard_fraction = hard_start + (hard_end - hard_start) * progress
    hard_count = min(candidate_count, max(1, int(round(candidate_count * hard_fraction))))
    general_count = candidate_count - hard_count
    general_pool = pools["warm"] if progress <= float(cfg["warm_general_fraction"]) else pools["all"]
    candidates = _draw(pools["hard"], hard_count, generator, torch)
    candidates += _draw(general_pool, general_count, generator, torch)

    x, y, weights = tuple(
        torch.stack([train[i][key] for i in candidates]).to(device)
        for key in ("x", "y", "w")
    )
    copy_weight = float(cfg["copy_token_weight"])
    first_weight = float(cfg["first_token_weight"])
    target_mask = y.ne(-100) & (
        weights.ge(copy_weight - 1e-6) | (weights - first_weight).abs().lt(1e-6)
    )
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
        hardness = error_rate * 2.0 + deficit_mean
        chosen = torch.topk(hardness, k=min(batch_size, len(candidates))).indices.tolist()
    return [candidates[j] for j in chosen]


def margin_weighted_loss(model, x, y, weights, cfg, torch):
    """Weighted CE plus a correct-vs-best-wrong margin on TARGET positions."""
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
    first_weight = float(cfg["first_token_weight"])
    target_mask = active & (
        weights.ge(copy_weight - 1e-6) | (weights - first_weight).abs().lt(1e-6)
    )
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

replace_required(
    'train = [base.encode_row(tokenizer, row, cfg, torch) for row in train_rows]\n'
    '        val = [base.encode_row(tokenizer, row, cfg, torch) for row in val_rows]\n'
    '        baseline = diagnostic(model, tokenizer, cfg, "cpu", torch, base)',
    'train = [base.encode_row(tokenizer, row, cfg, torch) for row in train_rows]\n'
    '        val = [base.encode_row(tokenizer, row, cfg, torch) for row in val_rows]\n'
    '        hard_pools = build_hard_pools(train, cfg)\n'
    '        baseline = diagnostic(model, tokenizer, cfg, "cpu", torch, base)',
)

# Exercise the hard-mining forward pass during CPU preflight using a small batch.
replace_required(
    'if args.preflight_only:\n'
    '            print("EMBER_HF_V019_PREFLIGHT=PASS", flush=True)\n'
    '            return',
    'if args.preflight_only:\n'
    '            probe_generator = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))\n'
    '            probe = sample_hard_mined_indices(train, hard_pools, 0, int(cfg["max_steps"]), 2, cfg, probe_generator, model, "cpu", torch)\n'
    '            if len(probe) != 2:\n'
    '                raise RuntimeError(f"v0.0.19 hard-mining preflight returned {len(probe)} rows")\n'
    '            print("EMBER_V019_HARD_MINING_PREFLIGHT=PASS", flush=True)\n'
    '            print("EMBER_HF_V019_PREFLIGHT=PASS", flush=True)\n'
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

exec(compile(source, "ember_hf_sft_v019_runtime.py", "exec"), {"__name__": "__main__"})
