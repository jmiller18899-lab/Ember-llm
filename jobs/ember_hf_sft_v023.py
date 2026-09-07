# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["huggingface-hub>=1.4", "sentencepiece>=0.2", "torch>=2.4"]
# ///
"""Ember v0.0.23: prevent the first continuation error before it starts a cascade.

This phase deliberately returns to the v0.0.20 best checkpoint. It mines rows by
their earliest wrong continuation argmax, applies a large correct-vs-best-wrong
margin at that exact first-error boundary, and lightly guards the already-correct
prefix before the boundary. Structured synthetic candidates provide punctuation-
and digit-dense opportunities without using any held-out diagnostic value.
"""
from __future__ import annotations

import urllib.request

BASE_COMMIT = "8ccd8cf3cd6de9c1b92e9f9a8897772f67d14c15"
CONFIG_COMMIT = "c434a2e2e346b4cb73f5adec608561d15e704608"
BASE_URL = (
    "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/"
    f"{BASE_COMMIT}/jobs/ember_hf_sft_v016.py"
)

with urllib.request.urlopen(BASE_URL) as response:
    source = response.read().decode("utf-8")


def replace_required(old: str, new: str, *, count: int = -1) -> None:
    global source
    if old not in source:
        raise RuntimeError(f"v0.0.23 transform target missing: {old!r}")
    source = source.replace(old, new, count)


# Repoint the proven scaffold at v0.0.23 and the v0.0.20 best checkpoint.
replace_required(
    'CONFIG_PIN = "42ed91ac6e6b8ff22873fec6706c4f243f57f2fa"',
    f'CONFIG_PIN = "{CONFIG_COMMIT}"',
    count=1,
)
source = source.replace("0.0.16", "0.0.23")
replace_required("ember_sequence_copy_v0.0.23.json", "ember_first_error_v0.0.23.json")
replace_required("sequence-copy-consolidation", "first-error-prevention")
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
source = source.replace("EMBER_HF_V016_", "EMBER_HF_V023_")
source = source.replace("EMBER_V016_", "EMBER_V023_")

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
    raise RuntimeError(f"v0.0.23 could not generate leakage-safe value for {kind}")


def build_structured_rows(data, cfg, train_rows, val_rows):
    held = set(str(v) for v in data.HELD_OUT_VALUES)
    banned = held | {str(r["value"]) for r in train_rows} | {str(r["value"]) for r in val_rows}
    kinds = [str(k) for k in cfg["structured_kinds"]]
    all_kinds = list(data.KINDS)
    rows = []
    per_kind = int(cfg["structured_neighbor_examples_per_kind"])
    base_index = int(cfg["structured_neighbor_index_base"])
    for kind_offset, kind in enumerate(kinds):
        if kind not in all_kinds:
            raise RuntimeError(f"v0.0.23 unknown structured kind {kind!r}")
        other = all_kinds[(all_kinds.index(kind) + 3) % len(all_kinds)]
        for j in range(per_kind):
            i = base_index + kind_offset * 10000 + j
            value = _fresh_value(data, kind, i, banned); banned.add(value)
            distractor_a = _fresh_value(data, kind, i + per_kind + 17, banned); banned.add(distractor_a)
            distractor_b = _fresh_value(data, other, i + per_kind + 31, banned); banned.add(distractor_b)
            if value in held or distractor_a in held or distractor_b in held:
                raise RuntimeError("v0.0.23 held-out leakage in structured row")
            row = {
                "id": f"first-error-{kind}-{j:05d}",
                "kind": kind,
                "value": value,
                "prompt": data.prompt(value, distractor_a, distractor_b),
                "completion": data.completion(value),
                "focus_terms": [value],
                "structured_candidate": True,
            }
            if row["prompt"].count(value) != 1 or not row["completion"].startswith(value):
                raise RuntimeError(f"v0.0.23 malformed structured row {row['id']}")
            rows.append(row)
    if not rows:
        raise RuntimeError("v0.0.23 generated no structured rows")
    return rows


def build_candidate_pools(train):
    pools = {"structured": [], "all": list(range(len(train)))}
    for i, item in enumerate(train):
        if bool(item["row"].get("structured_candidate")):
            pools["structured"].append(i)
    if not pools["structured"] or not pools["all"]:
        raise RuntimeError("v0.0.23 candidate pools are empty")
    return pools


def _draw(pool, count, generator, torch):
    if count <= 0:
        return []
    local = torch.randint(len(pool), (count,), generator=generator).tolist()
    return [pool[j] for j in local]


def _gap_fields(logits, y, copy_mask, torch):
    safe_y = y.clamp_min(0)
    correct = logits.float().gather(-1, safe_y.unsqueeze(-1)).squeeze(-1)
    top2_values, top2_indices = torch.topk(logits.float(), k=2, dim=-1)
    best_is_correct = top2_indices[..., 0].eq(y)
    best_wrong = torch.where(best_is_correct, top2_values[..., 1], top2_values[..., 0])
    gap = correct - best_wrong
    huge = torch.full_like(gap, 1e6)
    gap_masked = torch.where(copy_mask, gap, huge)
    return gap, gap_masked, best_is_correct


def _first_wrong_positions(pred, y, copy_mask, torch):
    wrong = pred.ne(y) & copy_mask
    seq_len = y.size(1)
    positions = torch.arange(seq_len, device=y.device).unsqueeze(0).expand_as(y)
    sentinel = torch.full_like(positions, seq_len)
    first = torch.where(wrong, positions, sentinel).min(dim=1).values
    has_error = first.lt(seq_len)
    return wrong, first, has_error, positions


def sample_first_error_indices(train, pools, step, total_steps, batch_size, cfg,
                               generator, model, device, torch):
    multiplier = max(2, int(cfg["candidate_multiplier"]))
    candidate_count = max(batch_size, batch_size * multiplier)
    progress = (step + 1) / max(1, total_steps)
    start = float(cfg["structured_candidate_fraction_start"])
    end = float(cfg["structured_candidate_fraction_end"])
    structured_fraction = start + (end - start) * progress
    structured_count = min(candidate_count - 1, max(1, int(round(candidate_count * structured_fraction))))
    general_count = candidate_count - structured_count
    candidates = _draw(pools["structured"], structured_count, generator, torch)
    candidates += _draw(pools["all"], general_count, generator, torch)

    x, y, weights = tuple(
        torch.stack([train[i][key] for i in candidates]).to(device)
        for key in ("x", "y", "w")
    )
    copy_mask = y.ne(-100) & weights.ge(float(cfg["copy_token_weight"]) - 1e-6)
    with torch.no_grad():
        logits, _ = model(x, None)
        pred = torch.argmax(logits, dim=-1)
        wrong, first, has_error, _positions = _first_wrong_positions(pred, y, copy_mask, torch)
        _gap, gap_masked, _best = _gap_fields(logits, y, copy_mask, torch)
        weakest_gap = gap_masked.min(dim=1).values
        copy_count = copy_mask.sum(dim=1).clamp_min(1)
        first_rank = first.float() / y.size(1)
        early_bonus = torch.where(has_error, 1.0 - first_rank, torch.zeros_like(first_rank))
        deficit = torch.relu(float(cfg["first_error_margin"]) - weakest_gap)
        error_rate = wrong.sum(dim=1).float() / copy_count.float()
        hardness = has_error.float() * 5.0 + early_bonus * 2.0 + error_rate + deficit
        chosen = torch.topk(hardness, k=min(batch_size, len(candidates))).indices.tolist()
    return [candidates[j] for j in chosen]


def first_error_prevention_loss(model, x, y, weights, cfg, torch, *, return_stats=False):
    import torch.nn.functional as F

    logits, _ = model(x, None)
    raw = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), y.reshape(-1),
        ignore_index=-100, reduction="none",
    ).reshape_as(y)
    active = y.ne(-100)
    ce_denom = (weights * active).sum().clamp_min(1.0)
    ce = (raw * weights * active).sum() / ce_denom

    copy_mask = active & weights.ge(float(cfg["copy_token_weight"]) - 1e-6)
    if not bool(copy_mask.any().item()):
        stats = {"rows": int(y.size(0)), "rows_with_first_error": 0, "anchors": 0, "prefix_guard_positions": 0}
        return (ce, stats) if return_stats else (ce, None)

    pred = torch.argmax(logits.detach(), dim=-1)
    wrong, first, has_error, positions = _first_wrong_positions(pred, y, copy_mask, torch)
    gap, gap_masked, best_is_correct = _gap_fields(logits, y, copy_mask, torch)

    # One anchor per row: true first error when present, otherwise the weakest
    # still-correct continuation position so near-boundary rows remain guarded.
    weakest = gap_masked.argmin(dim=1)
    anchor_pos = torch.where(has_error, first, weakest)
    anchor_mask = torch.zeros_like(copy_mask)
    anchor_mask.scatter_(1, anchor_pos.unsqueeze(1), True)
    anchor_mask &= copy_mask

    anchor_gap = gap[anchor_mask]
    anchor_correct = best_is_correct[anchor_mask]
    margin_term = F.relu(float(cfg["first_error_margin"]) - anchor_gap)
    wrong_mult = torch.where(
        anchor_correct,
        torch.ones_like(margin_term),
        torch.full_like(margin_term, float(cfg["first_error_wrong_multiplier"])),
    )
    anchor_loss = (margin_term * wrong_mult).mean() if margin_term.numel() else ce * 0.0

    # Preserve the correct teacher-forced prefix before the first mistake.
    prefix_mask = copy_mask & has_error.unsqueeze(1) & positions.lt(first.unsqueeze(1))
    guard_term = F.relu(float(cfg["prefix_guard_margin"]) - gap[prefix_mask])
    guard_loss = guard_term.mean() if guard_term.numel() else ce * 0.0

    total = ce + float(cfg["first_error_margin_weight"]) * anchor_loss + float(cfg["prefix_guard_weight"]) * guard_loss
    stats = {
        "rows": int(y.size(0)),
        "rows_with_first_error": int(has_error.sum().item()),
        "anchors": int(anchor_mask.sum().item()),
        "prefix_guard_positions": int(prefix_mask.sum().item()),
    }
    return (total, stats) if return_stats else (total, None)


def first_error_diagnostic(model, tokenizer, cfg, device, torch, base, data):
    rows = []
    for idx, pair in enumerate(base.DIAGNOSTICS):
        value = str(pair[0])
        prompt = base.prompt_for(value)
        prompt_ids = list(tokenizer.encode(prompt))
        full = list(tokenizer.encode(prompt + base.completion_for(value)))
        eot_id = int(tokenizer.encode(base.EOT)[-1])
        eot_positions = [i for i in range(len(prompt_ids), len(full)) if int(full[i]) == eot_id]
        if not eot_positions:
            raise RuntimeError("v0.0.23 first-error diagnostic missing EOS")
        first_target = len(prompt_ids) - 1
        eos_target = eot_positions[0] - 1
        positions = list(range(first_target + 1, eos_target))
        x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
        y = full[1:]
        with torch.inference_mode():
            logits, _ = model(x, None)
            pred = torch.argmax(logits[0], dim=-1)
        first_error_offset = None
        for offset, pos in enumerate(positions):
            if int(pred[pos].item()) != int(y[pos]):
                first_error_offset = offset
                break
        rows.append({
            "kind": str(data.KINDS[idx]),
            "continuation_tokens": len(positions),
            "first_error_offset": first_error_offset,
            "teacher_forced_continuation_exact": first_error_offset is None,
        })
    return rows

'''
replace_required("def score(diag: dict, val_loss: float):", helpers + "def score(diag: dict, val_loss: float):")

# Build baseline first, then add leakage-safe structured candidates only to train.
replace_required(
    'train = [base.encode_row(tokenizer, row, cfg, torch) for row in train_rows]\n'
    '        val = [base.encode_row(tokenizer, row, cfg, torch) for row in val_rows]\n'
    '        baseline = diagnostic(model, tokenizer, cfg, "cpu", torch, base)',
    'val = [base.encode_row(tokenizer, row, cfg, torch) for row in val_rows]\n'
    '        baseline = diagnostic(model, tokenizer, cfg, "cpu", torch, base)\n'
    '        first_error_cases = first_error_diagnostic(model, tokenizer, cfg, "cpu", torch, base, data)\n'
    '        structured_rows = build_structured_rows(data, cfg, train_rows, val_rows)\n'
    '        train_rows = list(train_rows) + structured_rows\n'
    '        train = [base.encode_row(tokenizer, row, cfg, torch) for row in train_rows]\n'
    '        candidate_pools = build_candidate_pools(train)',
)

replace_required(
    '"baseline": baseline["metrics"],\n            "hf": {',
    '"baseline": baseline["metrics"],\n'
    '            "first_error_diagnostic": first_error_cases,\n'
    '            "structured_candidates": {\n'
    '                "rows": len(structured_rows),\n'
    '                "kinds": list(cfg["structured_kinds"]),\n'
    '                "held_out_values_added_to_training": False,\n'
    '            },\n'
    '            "hf": {',
    count=1,
)

# Exercise mining and the actual first-error loss during CPU preflight.
replace_required(
    'if args.preflight_only:\n'
    '            print("EMBER_HF_V023_PREFLIGHT=PASS", flush=True)\n'
    '            return',
    'if args.preflight_only:\n'
    '            probe_generator = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))\n'
    '            probe = sample_first_error_indices(train, candidate_pools, 0, int(cfg["max_steps"]), 4, cfg, probe_generator, model, "cpu", torch)\n'
    '            x, y, w = base.batch(train, probe, "cpu", torch)\n'
    '            probe_loss, probe_stats = first_error_prevention_loss(model, x, y, w, cfg, torch, return_stats=True)\n'
    '            if not bool(torch.isfinite(probe_loss).item()) or int(probe_stats["anchors"]) != len(probe):\n'
    '                raise RuntimeError(f"v0.0.23 first-error preflight failed: stats={probe_stats}")\n'
    '            print("EMBER_V023_FIRST_ERROR_DIAGNOSTIC=" + json.dumps(first_error_cases, separators=(",", ":")), flush=True)\n'
    '            print("EMBER_V023_FIRST_ERROR_PROBE=" + json.dumps(probe_stats, separators=(",", ":")), flush=True)\n'
    '            print("EMBER_V023_FIRST_ERROR_PREFLIGHT=PASS", flush=True)\n'
    '            print("EMBER_HF_V023_PREFLIGHT=PASS", flush=True)\n'
    '            return',
)

replace_required(
    'ids = torch.randint(len(train), (int(cfg["batch_size"]),), generator=generator).tolist()',
    'ids = sample_first_error_indices(train, candidate_pools, step, max_steps, int(cfg["batch_size"]), cfg, generator, model, device, torch)',
)
replace_required(
    'with amp():\n                    loss = base.weighted_loss(model, x, y, w, torch) / accum',
    'with amp():\n                    raw_loss, _loss_stats = first_error_prevention_loss(model, x, y, w, cfg, torch)\n                    loss = raw_loss / accum',
)

# Seed best.pt with the v0.0.20 source itself. Later checkpoints must beat its
# diagnostic tuple; lower validation loss alone cannot compensate for worse
# exact-copy or continuation metrics.
replace_required(
    'best_score = None\n        best_step = -1\n        history = []',
    'best_score = score(baseline, float("inf"))\n'
    '        best_step = -1\n'
    '        history = []\n'
    '        save_checkpoint(\n'
    '            best_path, model=model, optimizer=optimizer, tokenizer=tokenizer,\n'
    '            model_config=model.cfg, train_config=cfg, step=-1,\n'
    '            best_val_loss=float("inf"), run_id=run_id,\n'
    '        )',
    count=1,
)

exec(compile(source, "ember_hf_sft_v023_runtime.py", "exec"), {"__name__": "__main__"})
