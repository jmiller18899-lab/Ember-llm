# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["huggingface-hub>=1.4", "sentencepiece>=0.2", "torch>=2.4"]
# ///
"""Ember v0.0.22: token-level continuation repair plus self-error recovery.

This phase resumes from the v0.0.21 best checkpoint. It no longer targets whole
copy formats. Each update mines rows with current continuation-token mistakes,
upweights only the wrong TARGET positions, and adds a recovery pass where a
small schedule of prefix tokens is replaced with the model's own best-wrong
alternative. The recovery loss then teaches the correct next token after that
plausible self-generated mistake.
"""
from __future__ import annotations

import urllib.request

BASE_COMMIT = "8ccd8cf3cd6de9c1b92e9f9a8897772f67d14c15"
CONFIG_COMMIT = "9e2b1c2ad6ead7e5d4f1b47c3c7d9bd354e47145"
BASE_URL = (
    "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/"
    f"{BASE_COMMIT}/jobs/ember_hf_sft_v016.py"
)

with urllib.request.urlopen(BASE_URL) as response:
    source = response.read().decode("utf-8")


def replace_required(old: str, new: str, *, count: int = -1) -> None:
    global source
    if old not in source:
        raise RuntimeError(f"v0.0.22 transform target missing: {old!r}")
    source = source.replace(old, new, count)


# Repoint the stable v0.0.16 scaffold at v0.0.22 and v0.0.21 best.pt.
replace_required(
    'CONFIG_PIN = "42ed91ac6e6b8ff22873fec6706c4f243f57f2fa"',
    f'CONFIG_PIN = "{CONFIG_COMMIT}"',
    count=1,
)
source = source.replace("0.0.16", "0.0.22")
replace_required("ember_sequence_copy_v0.0.22.json", "ember_token_recovery_v0.0.22.json")
replace_required("sequence-copy-consolidation", "token-error-recovery-scheduled-sampling")
replace_required(
    'SOURCE_REPO = "Jmiller18899/ember-v0.0.15-t4"',
    'SOURCE_REPO = "Jmiller18899/ember-v0.0.21-t4"',
)
replace_required(
    'cfg.get("source_model_name") != "ember-v0.0.15-t4"',
    'cfg.get("source_model_name") != "ember-v0.0.21-t4"',
)
replace_required("must start from v0.0.15", "must start from v0.0.21")
replace_required(
    'if str(source_cfg.get("version")) != "0.0.15":',
    'if str(source_cfg.get("version")) != "0.0.21":',
)
replace_required("expected a v0.0.15 checkpoint", "expected a v0.0.21 checkpoint")
replace_required("v0.0.15 checkpoint run_id", "v0.0.21 checkpoint run_id")
replace_required("v0.0.15 source state", "v0.0.21 source state")
replace_required("resolve_v015_source", "resolve_v021_source")
replace_required("v015_promotion", "v021_promotion")
replace_required("v015_best_step", "v021_best_step")
source = source.replace("EMBER_HF_V016_", "EMBER_HF_V022_")
source = source.replace("EMBER_V016_", "EMBER_V022_")

# Keep held-out continuation top-1 explicit as copy-position top-1.
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
def sample_token_hard_indices(train, step, total_steps, batch_size, cfg,
                              generator, model, device, torch):
    """Select rows by current continuation-token errors, never by format label."""
    import torch.nn.functional as F

    multiplier = max(2, int(cfg["hard_candidate_multiplier"]))
    candidate_count = max(batch_size, batch_size * multiplier)
    candidates = torch.randint(len(train), (candidate_count,), generator=generator).tolist()
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
        wrong_count = wrong.sum(dim=1).float()
        error_rate = wrong_count / denom.float()

        safe_y = y.clamp_min(0)
        correct = logits.float().gather(-1, safe_y.unsqueeze(-1)).squeeze(-1)
        top2_values, top2_indices = torch.topk(logits.float(), k=2, dim=-1)
        best_is_correct = top2_indices[..., 0].eq(y)
        best_wrong = torch.where(best_is_correct, top2_values[..., 1], top2_values[..., 0])
        deficit = F.relu(float(cfg["margin"]) - (correct - best_wrong)) * target_mask
        deficit_mean = deficit.sum(dim=1) / denom.float()
        # Wrong-token count dominates, margin deficit breaks ties.
        hardness = wrong_count * 2.0 + error_rate + deficit_mean
        chosen = torch.topk(hardness, k=min(batch_size, len(candidates))).indices.tolist()
    return [candidates[j] for j in chosen]


def token_recovery_loss(model, x, y, weights, cfg, torch, step, total_steps,
                        return_stats=False):
    """Teacher CE + wrong-position margin + self-error recovery CE.

    The recovery pass replaces selected prefix tokens with the model's own
    best-wrong alternative from the teacher-forced pass. Loss is then measured
    on the correct next continuation token, teaching local recovery from a
    plausible generated mistake.
    """
    import torch.nn.functional as F

    logits, _ = model(x, None)
    active = y.ne(-100)
    copy_weight = float(cfg["copy_token_weight"])
    copy_mask = active & weights.ge(copy_weight - 1e-6)

    raw = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(y)

    with torch.no_grad():
        pred = torch.argmax(logits, dim=-1)
        wrong_positions = pred.ne(y) & copy_mask
    position_multiplier = torch.ones_like(weights)
    position_multiplier = torch.where(
        wrong_positions,
        torch.full_like(position_multiplier, float(cfg["error_position_multiplier"])),
        position_multiplier,
    )
    effective_weights = weights * position_multiplier
    teacher_denom = (effective_weights * active).sum().clamp_min(1.0)
    teacher_ce = (raw * effective_weights * active).sum() / teacher_denom

    margin_loss = logits.sum() * 0.0
    if bool(copy_mask.any().item()):
        copy_logits = logits[copy_mask].float()
        copy_targets = y[copy_mask]
        correct_logits = copy_logits.gather(1, copy_targets.unsqueeze(1)).squeeze(1)
        top2_values, top2_indices = torch.topk(copy_logits, k=2, dim=1)
        best_is_correct = top2_indices[:, 0].eq(copy_targets)
        best_wrong_logits = torch.where(best_is_correct, top2_values[:, 1], top2_values[:, 0])
        margin_term = F.relu(float(cfg["margin"]) - (correct_logits - best_wrong_logits))
        margin_multiplier = 1.0 + (~best_is_correct).to(margin_term.dtype) * (
            float(cfg["error_position_multiplier"]) - 1.0
        )
        margin_loss = (margin_term * margin_multiplier).mean()

    # Construct deterministic scheduled sampling slots. x[:, p] is the token
    # predicted by logits[:, p-1], so corrupting x[:, p] teaches recovery at y[:, p].
    progress = (step + 1) / max(1, total_steps)
    rate = float(cfg["scheduled_sampling_rate_start"]) + (
        float(cfg["scheduled_sampling_rate_end"]) - float(cfg["scheduled_sampling_rate_start"])
    ) * progress
    rate = max(0.0, min(1.0, rate))

    top2_values_all, top2_indices_all = torch.topk(logits.detach().float(), k=2, dim=-1)
    top1_all = top2_indices_all[..., 0]
    top2_all = top2_indices_all[..., 1]
    best_wrong_all = torch.where(top1_all.eq(y), top2_all, top1_all)

    eligible = copy_mask[:, 1:] & active[:, :-1]
    flat = torch.arange(eligible.numel(), device=x.device).reshape_as(eligible)
    period = max(1, int(round(1.0 / max(rate, 1e-6))))
    slots = eligible & ((flat + int(step)) % period == 0)
    if bool(eligible.any().item()) and not bool(slots.any().item()):
        first = torch.nonzero(eligible, as_tuple=False)[0]
        slots[first[0], first[1]] = True

    recovery_loss = logits.sum() * 0.0
    recovery_slots = int(slots.sum().item())
    if recovery_slots:
        recovery_x = x.clone()
        recovery_view = recovery_x[:, 1:]
        recovery_view[slots] = best_wrong_all[:, :-1][slots]
        recovery_logits, _ = model(recovery_x, None)
        recovery_raw = F.cross_entropy(
            recovery_logits.reshape(-1, recovery_logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(y)
        recovery_mask = torch.zeros_like(active)
        recovery_mask[:, 1:] = slots
        recovery_loss = recovery_raw[recovery_mask].mean()

    total = (
        teacher_ce
        + float(cfg["margin_loss_weight"]) * margin_loss
        + float(cfg["recovery_loss_weight"]) * recovery_loss
    )
    if return_stats:
        return total, {
            "copy_positions": int(copy_mask.sum().item()),
            "wrong_positions": int(wrong_positions.sum().item()),
            "recovery_slots": recovery_slots,
            "scheduled_sampling_rate": rate,
        }
    return total

'''
replace_required("def score(diag: dict, val_loss: float):", helpers + "def score(diag: dict, val_loss: float):")

# CPU preflight exercises hard-row selection and the two-forward recovery loss.
replace_required(
    'if args.preflight_only:\n'
    '            print("EMBER_HF_V022_PREFLIGHT=PASS", flush=True)\n'
    '            return',
    'if args.preflight_only:\n'
    '            probe_generator = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))\n'
    '            probe_ids = sample_token_hard_indices(train, 0, int(cfg["max_steps"]), 2, cfg, probe_generator, model, "cpu", torch)\n'
    '            if len(probe_ids) != 2:\n'
    '                raise RuntimeError(f"v0.0.22 token-hard preflight returned {len(probe_ids)} rows")\n'
    '            px, py, pw = base.batch(train, probe_ids, "cpu", torch)\n'
    '            probe_loss, probe_stats = token_recovery_loss(model, px, py, pw, cfg, torch, 0, int(cfg["max_steps"]), return_stats=True)\n'
    '            if not bool(torch.isfinite(probe_loss).item()):\n'
    '                raise RuntimeError("v0.0.22 recovery preflight produced non-finite loss")\n'
    '            if probe_stats["copy_positions"] <= 0 or probe_stats["recovery_slots"] <= 0:\n'
    '                raise RuntimeError(f"v0.0.22 recovery preflight did not exercise token recovery: {probe_stats}")\n'
    '            print("EMBER_V022_TOKEN_RECOVERY_STATS=" + json.dumps(probe_stats, sort_keys=True), flush=True)\n'
    '            print("EMBER_V022_TOKEN_RECOVERY_PREFLIGHT=PASS", flush=True)\n'
    '            print("EMBER_HF_V022_PREFLIGHT=PASS", flush=True)\n'
    '            return',
    count=1,
)

replace_required(
    'ids = torch.randint(len(train), (int(cfg["batch_size"]),), generator=generator).tolist()',
    'ids = sample_token_hard_indices(train, step, max_steps, int(cfg["batch_size"]), cfg, generator, model, device, torch)',
    count=1,
)
replace_required(
    'loss = base.weighted_loss(model, x, y, w, torch) / accum',
    'loss = token_recovery_loss(model, x, y, w, cfg, torch, step, max_steps) / accum',
    count=1,
)

exec(compile(source, "ember_hf_sft_v022_runtime.py", "exec"), {"__name__": "__main__"})
