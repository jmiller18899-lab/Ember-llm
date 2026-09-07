# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["huggingface-hub>=1.4", "sentencepiece>=0.2", "torch>=2.4"]
# ///
"""Ember v0.0.18: convert lower copy NLL into greedy top-1 exact copying.

This phase resumes from v0.0.17 latest.pt (not best.pt), keeps the tested
v0.0.16 training scaffold, and adds an argmax-margin objective on TARGET token
positions. Positions whose current argmax is wrong receive extra weight. A
short-to-long curriculum is used before the full mixed copy distribution.
"""
from __future__ import annotations

import urllib.request

BASE_COMMIT = "8ccd8cf3cd6de9c1b92e9f9a8897772f67d14c15"
CONFIG_COMMIT = "020cbefdd6caae75cb5842b9a94e4eb66e90eb47"
BASE_URL = (
    "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/"
    f"{BASE_COMMIT}/jobs/ember_hf_sft_v016.py"
)

with urllib.request.urlopen(BASE_URL) as response:
    source = response.read().decode("utf-8")


def replace_required(old: str, new: str, *, count: int = -1) -> None:
    global source
    if old not in source:
        raise RuntimeError(f"v0.0.18 transform target missing: {old!r}")
    source = source.replace(old, new, count)


# Repoint the tested scaffold at v0.0.18 and the v0.0.17 latest checkpoint.
replace_required(
    'CONFIG_PIN = "42ed91ac6e6b8ff22873fec6706c4f243f57f2fa"',
    f'CONFIG_PIN = "{CONFIG_COMMIT}"',
    count=1,
)
source = source.replace("0.0.16", "0.0.18")
replace_required("ember_sequence_copy_v0.0.18.json", "ember_margin_copy_v0.0.18.json")
replace_required("sequence-copy-consolidation", "argmax-margin-copy")
replace_required(
    'SOURCE_REPO = "Jmiller18899/ember-v0.0.15-t4"',
    'SOURCE_REPO = "Jmiller18899/ember-v0.0.17-t4"',
)
replace_required(
    'cfg.get("source_model_name") != "ember-v0.0.15-t4"',
    'cfg.get("source_model_name") != "ember-v0.0.17-t4"',
)
replace_required("must start from v0.0.15", "must start from v0.0.17")
replace_required(
    'if str(source_cfg.get("version")) != "0.0.15":',
    'if str(source_cfg.get("version")) != "0.0.17":',
)
replace_required("expected a v0.0.15 checkpoint", "expected a v0.0.17 checkpoint")
replace_required("v0.0.15 checkpoint run_id", "v0.0.17 checkpoint run_id")
replace_required("v0.0.15 source state", "v0.0.17 source state")
replace_required("resolve_v015_source", "resolve_v017_source")
replace_required("v015_promotion", "v017_promotion")
replace_required("v015_best_step", "v017_best_step")
replace_required(
    'remote_path = f"checkpoints/{run_id}/best.pt"',
    'remote_path = f"checkpoints/{run_id}/latest.pt"',
)
source = source.replace("EMBER_HF_V016_", "EMBER_HF_V018_")
source = source.replace("EMBER_V016_", "EMBER_V018_")

# The existing continuation diagnostic is already a held-out TARGET-position
# top-1 measurement. Surface it explicitly under the v0.0.18 metric name and
# add a stricter gate without changing the held-out diagnostic cases.
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
def build_curriculum_pools(train):
    short_kinds = {"short_code", "digits", "expression", "mixed"}
    medium_kinds = short_kinds | {"long_code", "entity"}
    pools = {"short": [], "medium": [], "all": list(range(len(train)))}
    for i, item in enumerate(train):
        kind = str(item["row"].get("kind", ""))
        if kind in short_kinds:
            pools["short"].append(i)
        if kind in medium_kinds:
            pools["medium"].append(i)
    if not pools["short"] or not pools["medium"] or not pools["all"]:
        raise RuntimeError("v0.0.18 curriculum pool construction failed")
    return pools


def sample_curriculum_indices(pools, step, total_steps, batch_size, cfg, generator, torch):
    progress = (step + 1) / max(1, total_steps)
    if progress <= float(cfg["curriculum_short_fraction"]):
        pool = pools["short"]
    elif progress <= float(cfg["curriculum_medium_fraction"]):
        pool = pools["medium"]
    else:
        pool = pools["all"]
    local = torch.randint(len(pool), (batch_size,), generator=generator).tolist()
    return [pool[j] for j in local]


def margin_weighted_loss(model, x, y, weights, cfg, torch):
    """Weighted CE plus a correct-vs-best-wrong logit margin on TARGET positions."""
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

    # Margin applies to the first TARGET token and continuation TARGET tokens,
    # but not EOS / post-EOS formatting positions.
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
    '        curriculum_pools = build_curriculum_pools(train)\n'
    '        baseline = diagnostic(model, tokenizer, cfg, "cpu", torch, base)',
)
replace_required(
    'ids = torch.randint(len(train), (int(cfg["batch_size"]),), generator=generator).tolist()',
    'ids = sample_curriculum_indices(curriculum_pools, step, max_steps, int(cfg["batch_size"]), cfg, generator, torch)',
)
replace_required(
    'loss = base.weighted_loss(model, x, y, w, torch) / accum',
    'loss = margin_weighted_loss(model, x, y, w, cfg, torch) / accum',
)

exec(compile(source, "ember_hf_sft_v018_runtime.py", "exec"), {"__name__": "__main__"})
