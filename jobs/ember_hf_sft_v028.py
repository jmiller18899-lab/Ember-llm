# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["huggingface-hub>=1.4", "sentencepiece>=0.2", "torch>=2.4"]
# ///
"""Ember v0.0.28: boundary focus, from the v0.0.27 checkpoint.

v0.0.27 moved ``sequence_margin_health`` from -0.8871 to -0.6846 without moving
exact copy at all. The mechanism works; the allocation is wrong. See
``reports/ember-v0.0.27-result.json``.

Bounding the arithmetic: health is the mean over 90 cases of
``clamp(min_gap, -2.0, +1.2)``, and the 20 exact cases have positive min_gap, so
their clamped contribution is at most 24. The 70 failing cases therefore sit
between -1.223 and -0.880 on average after v0.0.27, up from between -1.483 and
-1.141. They are still about a logit short of the decision boundary, and the
whole run was worth +0.260 each. Waiting for the average to reach zero is four
more identical runs, optimistically.

The count does not need the average to cross zero. It needs the sequences
*nearest* zero to cross. Four changes, all the same idea in different places:

1. **Mine near the boundary.** ``relu(sequence_margin - worst_gap)`` grows
   without bound as a row gets worse, so v0.0.27's mining put a row at -3.0 ahead
   of a row at -0.05 for every mined slot -- exactly backwards for maximising the
   number of rows that flip. Rows outside the reachable band now take a constant
   low priority instead of the highest one.
2. **Down-weight out-of-band rows in the loss** rather than dropping them, so
   hopeless sequences keep learning slowly without owning the batch.
3. **Lower the sequence margin** from 1.20 to 0.60, so pressure is not spent
   widening margins on sequences that are already comfortably correct.
4. **Span every decision an exact copy requires.** ``encode_row`` overwrites the
   first token's weight with ``first_token_weight`` and the EOS decision's with
   ``eos_token_weight``, so both fall outside the copy-weight mask that the loss
   and the diagnostic select on. A sequence whose weakest decision is its first
   token has been invisible to the entire v0.0.27 apparatus. Independence
   arithmetic says the present magnitude is probably small, but it has never been
   measured, and the fix is free.

The point of this phase is as much to *learn* as to train. The free CPU preflight
now reports the per-case worst-margin distribution and ``sequences_within_reach``.
If that band is near-empty, boundary focus cannot work and the next lever is
capacity or representation -- and that is worth knowing before a paid session,
not after.
"""
from __future__ import annotations

import urllib.request

# The proven v0.0.16 scaffold, transformed at run time.
BASE_COMMIT = "8ccd8cf3cd6de9c1b92e9f9a8897772f67d14c15"
# The v0.0.26 curriculum, carried unchanged through v0.0.27 and v0.0.28.
DATA_COMMIT = "36a213d9fecaf9ab8348132d6036e9c99a17e18d"
# The commit carrying config/ember_boundary_focus_v0.0.28.json.
CONFIG_COMMIT = "862cfe8acdfe0e5e1fd96677973c37339bb1a2e0"
BASE_URL = (
    "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/"
    f"{BASE_COMMIT}/jobs/ember_hf_sft_v016.py"
)

HELPERS = r'''
def _v028_field_signature(value):
    """Template signature: separators literal, alphanumeric fields reduced to
    (length, class set). Mirrors jobs/ember_curriculum_audit_v026.py."""
    parts = []
    i = 0
    while i < len(value):
        if value[i].isalnum():
            j = i
            classes = set()
            while j < len(value) and value[j].isalnum():
                classes.add("d" if value[j].isdigit() else "a" if value[j].islower() else "A")
                j += 1
            parts.append("[%d%s]" % (j - i, "".join(sorted(classes))))
            i = j
        else:
            j = i
            while j < len(value) and not value[j].isalnum():
                j += 1
            parts.append(value[i:j])
            i = j
    return "".join(parts)


def assert_format_parity(data, train_rows):
    """Refuse to train on a curriculum that cannot emit a held-out template."""
    supported = {}
    for row in train_rows:
        signature = _v028_field_signature(str(row["value"]))
        bucket = supported.setdefault(str(row["kind"]), {})
        bucket[signature] = bucket.get(signature, 0) + 1
    missing = []
    per_case = []
    for kind, value, _ in data.DIAGNOSTICS:
        signature = _v028_field_signature(value)
        rows = int(supported.get(str(kind), {}).get(signature, 0))
        per_case.append({"kind": kind, "value": value, "template": signature, "train_rows": rows})
        if rows == 0:
            missing.append({"kind": kind, "value": value, "template": signature})
    if missing:
        raise RuntimeError(
            "v0.0.28 format parity failed; the curriculum cannot emit these held-out "
            "templates: %s" % missing
        )
    return {
        "held_out_cases": len(per_case),
        "templates_supported": len(per_case),
        "min_train_rows_for_a_held_out_template": min(c["train_rows"] for c in per_case),
    }


def v028_supervised_span(y, weights, cfg, torch):
    """Mask over every decision an exact copy requires.

    encode_row lays out copy_token_weight across [first, eos_target), then
    overwrites w[first] with first_token_weight and w[eos_target] with
    eos_token_weight. Selecting on the copy weight therefore omits the first
    token and the EOS decision, both of which greedy exact copy tests. The copy
    positions are contiguous, so the row's full span is that block plus the
    position immediately before it and the one immediately after.
    """
    active = y.ne(-100)
    copy_mask = active & weights.ge(float(cfg["copy_token_weight"]) - 1e-6)
    before = torch.zeros_like(copy_mask)
    before[:, :-1] = copy_mask[:, 1:]
    after = torch.zeros_like(copy_mask)
    after[:, 1:] = copy_mask[:, :-1]
    span = (copy_mask | before | after) & active
    return copy_mask, span


def v028_gaps(logits, y, mask, torch):
    """Per-position margin between the correct token and its best competitor."""
    selected = logits[mask].float()
    targets = y[mask]
    correct = selected.gather(1, targets.unsqueeze(1)).squeeze(1)
    top2_values, top2_indices = torch.topk(selected, k=2, dim=1)
    best_is_correct = top2_indices[:, 0].eq(targets)
    best_wrong = torch.where(best_is_correct, top2_values[:, 1], top2_values[:, 0])
    return correct - best_wrong, best_is_correct


def v028_row_worst(gap, mask, cfg, torch):
    """Each row's worst-position hinge, and whether that row is within reach.

    Returns (worst_hinge, worst_gap, has_targets). A row outside the reachable
    band is not dropped; the caller down-weights it, so hopeless sequences keep
    learning slowly instead of taking every mined slot.
    """
    import torch.nn.functional as F
    hinge_full = torch.zeros(mask.shape, dtype=gap.dtype, device=gap.device)
    hinge_full[mask] = F.relu(float(cfg["sequence_margin"]) - gap)
    gap_full = torch.full(mask.shape, float("inf"), dtype=gap.dtype, device=gap.device)
    gap_full[mask] = gap
    return hinge_full.max(dim=1).values, gap_full.min(dim=1).values, mask.any(dim=1)


def v028_boundary_weight(worst_gap, cfg, torch):
    """1.0 inside the reachable band, out_of_band_weight below it."""
    band_low = float(cfg["boundary_band_low"])
    far = float(cfg["out_of_band_weight"])
    in_band = worst_gap.ge(band_low)
    return torch.where(in_band, torch.ones_like(worst_gap), torch.full_like(worst_gap, far)), in_band


def boundary_focus_loss(model, x, y, weights, cfg, torch):
    """CE, plus the per-token hinge, plus a boundary-focused worst-position hinge."""
    import torch.nn.functional as F
    logits, _ = model(x, None)
    raw = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction="none"
    ).reshape_as(y)
    active = y.ne(-100)
    denom = (weights * active).sum().clamp_min(1.0)
    ce = (raw * weights * active).sum() / denom

    copy_mask, span_mask = v028_supervised_span(y, weights, cfg, torch)
    if not bool(copy_mask.any().item()):
        return ce

    copy_gap, copy_best_is_correct = v028_gaps(logits, y, copy_mask, torch)
    token_hinge = F.relu(float(cfg["margin"]) - copy_gap)
    hard = (~copy_best_is_correct).to(token_hinge.dtype)
    multiplier = 1.0 + hard * (float(cfg["hard_error_multiplier"]) - 1.0)
    token_term = (token_hinge * multiplier).mean()

    span_gap, _ = v028_gaps(logits, y, span_mask, torch)
    worst_hinge, worst_gap, has_targets = v028_row_worst(span_gap, span_mask, cfg, torch)
    focus, _ = v028_boundary_weight(worst_gap, cfg, torch)
    rows = has_targets.to(worst_hinge.dtype)
    sequence_term = (worst_hinge * focus * rows).sum() / rows.sum().clamp_min(1.0)

    return (
        ce
        + float(cfg["margin_loss_weight"]) * token_term
        + float(cfg["sequence_margin_loss_weight"]) * sequence_term
    )


def sample_boundary_focus_indices(train, batch_size, cfg, generator, model, device, torch):
    """Half the batch uniform, half the rows closest to flipping.

    v0.0.27 ranked candidates by relu(margin - worst_gap), which is monotone in
    how badly a row fails, so the mined half filled with rows that had no chance
    of becoming exact copies. Here a row outside the reachable band takes the
    constant priority it would have at the band edge, scaled down, so
    near-boundary rows outrank it while it still appears sometimes.
    """
    uniform = torch.randint(len(train), (int(batch_size),), generator=generator).tolist()
    hard_count = int(round(int(batch_size) * float(cfg["hard_batch_fraction"])))
    if hard_count <= 0:
        return uniform
    multiplier = max(2, int(cfg["hard_candidate_multiplier"]))
    candidates = torch.randint(
        len(train), (hard_count * multiplier,), generator=generator
    ).tolist()
    x, y, w = tuple(
        torch.stack([train[i][key] for i in candidates]).to(device) for key in ("x", "y", "w")
    )
    with torch.no_grad():
        logits, _ = model(x, None)
        copy_mask, span_mask = v028_supervised_span(y, w, cfg, torch)
        if not bool(span_mask.any().item()):
            return uniform
        span_gap, _ = v028_gaps(logits, y, span_mask, torch)
        worst_hinge, worst_gap, has_targets = v028_row_worst(span_gap, span_mask, cfg, torch)
        far_priority = (
            (float(cfg["sequence_margin"]) - float(cfg["boundary_band_low"]))
            * float(cfg["out_of_band_weight"])
        )
        in_band = worst_gap.ge(float(cfg["boundary_band_low"]))
        priority = torch.where(
            in_band, worst_hinge, torch.full_like(worst_hinge, far_priority)
        )
        priority = priority * has_targets.to(priority.dtype)
        chosen = torch.topk(priority, k=min(hard_count, len(candidates))).indices.tolist()
    return [candidates[j] for j in chosen] + uniform[: int(batch_size) - hard_count]


def v028_sequence_stats(model, tokenizer, prompt, expected, device, torch, base):
    """Teacher-forced statistics over the full exact-copy conjunction.

    Returns continuation top-1 counts (comparable to every run since v0.0.16),
    the continuation-only worst margin (comparable to v0.0.27's
    sequence_margin_health), and the worst margin across every supervised
    decision including the first token and EOS.
    """
    prompt_ids = list(tokenizer.encode(prompt))
    full = list(tokenizer.encode(prompt + base.completion_for(expected)))
    if full[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError("continuation diagnostic boundary changed")
    eot_id = int(tokenizer.encode(base.EOT)[-1])
    eot_positions = [i for i in range(len(prompt_ids), len(full)) if int(full[i]) == eot_id]
    if not eot_positions:
        raise RuntimeError("continuation diagnostic has no EOS")
    first_target = len(prompt_ids) - 1
    eos_target = eot_positions[0] - 1
    continuation = list(range(first_target + 1, eos_target))
    span = list(range(first_target, eos_target + 1))
    x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    y = full[1:]
    with torch.inference_mode():
        logits, _ = model(x, None)
        rows = logits[0].float()
        pred = torch.argmax(rows, dim=-1)

    def gap_at(pos):
        target = int(y[pos])
        vector = rows[pos]
        target_logit = float(vector[target].item())
        competitors = vector.clone()
        competitors[target] = float("-inf")
        return target_logit - float(competitors.max().item())

    correct = sum(int(pred[pos].item()) == int(y[pos]) for pos in continuation)
    continuation_gaps = [gap_at(pos) for pos in continuation]
    span_gaps = [gap_at(pos) for pos in span]
    return {
        "continuation_correct": correct,
        "continuation_tokens": len(continuation),
        "continuation_min_gap": min(continuation_gaps) if continuation_gaps else None,
        "span_min_gap": min(span_gaps) if span_gaps else None,
        "first_token_gap": gap_at(first_target),
        "eos_gap": gap_at(eos_target),
        "first_token_top1": int(pred[first_target].item()) == int(y[first_target]),
    }


def _v028_bucket(tokens, edges):
    for edge in edges:
        if tokens <= edge:
            return "<=%d" % edge
    return ">%d" % edges[-1]


def _v028_percentiles(values, points):
    if not values:
        return {}
    ordered = sorted(values)
    out = {}
    for p in points:
        index = min(len(ordered) - 1, max(0, int(round((float(p) / 100.0) * (len(ordered) - 1)))))
        out["p%d" % int(p)] = ordered[index]
    return out


def _v028_histogram(values, edges):
    counts = {}
    for i in range(len(edges) - 1):
        counts["[%.2f,%.2f)" % (edges[i], edges[i + 1])] = 0
    counts["<%.2f" % edges[0]] = 0
    counts[">=%.2f" % edges[-1]] = 0
    for v in values:
        if v < edges[0]:
            counts["<%.2f" % edges[0]] += 1
            continue
        if v >= edges[-1]:
            counts[">=%.2f" % edges[-1]] += 1
            continue
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1]:
                counts["[%.2f,%.2f)" % (edges[i], edges[i + 1])] += 1
                break
    return counts


def v028_battery(model, tokenizer, cases, prompt_fn, cfg, device, torch, base):
    model.eval()
    rows = []
    correct_tokens = 0
    total_tokens = 0
    for kind, value, corrupt in cases:
        prompt = prompt_fn(value)
        text, stopped = base.greedy(model, tokenizer, prompt, device, torch)
        visible = text.split(base.EOT, 1)[0].strip()
        first = base.first_token_stats(model, tokenizer, prompt, value, device, torch)
        expected_nll = base.completion_nll(model, tokenizer, prompt, value, device, torch)
        corrupt_nll = base.completion_nll(model, tokenizer, prompt, corrupt, device, torch)
        stats = v028_sequence_stats(model, tokenizer, prompt, value, device, torch, base)
        correct_tokens += stats["continuation_correct"]
        total_tokens += stats["continuation_tokens"]
        row = {
            "kind": kind,
            "value": value,
            "generated": visible,
            "exact": visible == value,
            "clean_stop": stopped,
            "first_token": first,
            "expected_nll": expected_nll,
            "corrupt_nll": corrupt_nll,
            "expected_wins": expected_nll < corrupt_nll,
        }
        row.update(stats)
        rows.append(row)
    model.train()

    total = len(rows)
    floor = float(cfg["margin_health_floor"])
    ceiling = float(cfg["margin_health_ceiling"])
    band_low = float(cfg["boundary_band_low"])

    def health(key):
        values = [r[key] for r in rows if r[key] is not None]
        if not values:
            return floor
        return sum(min(max(v, floor), ceiling) for v in values) / len(values)

    span_gaps = [r["span_min_gap"] for r in rows if r["span_min_gap"] is not None]
    within_reach = [g for g in span_gaps if band_low <= g < 0.0]
    metrics = {
        "exact_copy_rate": sum(r["exact"] for r in rows) / total,
        "continuation_top1_rate": (correct_tokens / total_tokens) if total_tokens else 0.0,
        "clean_stop_rate": sum(r["clean_stop"] for r in rows) / total,
        "first_token_top1_rate": sum(r["first_token_top1"] for r in rows) / total,
        "first_token_top5_rate": sum(r["first_token"]["top5"] for r in rows) / total,
        "first_token_top20_rate": sum(r["first_token"]["top20"] for r in rows) / total,
        "teacher_forced_expected_win_rate": sum(r["expected_wins"] for r in rows) / total,
        "mean_first_token_rank": sum(r["first_token"]["rank"] for r in rows) / total,
        # Continuation-only, on the v0.0.27 scale, so the series stays readable.
        "sequence_margin_health": health("continuation_min_gap"),
        # Every decision exact copy requires. This is what selection uses.
        "full_sequence_margin_health": health("span_min_gap"),
        "sequences_all_positions_positive": sum(1 for g in span_gaps if g > 0) / total,
        "sequences_within_reach": len(within_reach) / total,
        "continuation_tokens": total_tokens,
        "mean_continuation_tokens": (total_tokens / total) if total else 0.0,
    }
    distribution = {
        "span_min_gap_percentiles": _v028_percentiles(
            span_gaps, cfg["margin_report_percentiles"]
        ),
        "span_min_gap_histogram": _v028_histogram(span_gaps, cfg["margin_histogram_edges"]),
        "within_reach_cases": sorted(
            (
                {"kind": r["kind"], "value": r["value"], "span_min_gap": r["span_min_gap"]}
                for r in rows
                if r["span_min_gap"] is not None and band_low <= r["span_min_gap"] < 0.0
            ),
            key=lambda item: -item["span_min_gap"],
        ),
        "weakest_decision": {
            "first_token": sum(
                1 for r in rows
                if r["span_min_gap"] is not None and r["first_token_gap"] <= r["span_min_gap"] + 1e-9
            ),
            "eos": sum(
                1 for r in rows
                if r["span_min_gap"] is not None and r["eos_gap"] <= r["span_min_gap"] + 1e-9
            ),
        },
    }
    by_kind = {}
    by_length = {}
    edges = [int(e) for e in cfg["length_buckets"]]
    for row in rows:
        for label, table in (
            (row["kind"], by_kind),
            (_v028_bucket(row["continuation_tokens"], edges), by_length),
        ):
            bucket = table.setdefault(label, {"cases": 0, "exact": 0})
            bucket["cases"] += 1
            bucket["exact"] += int(row["exact"])
    return {
        "metrics": metrics,
        "cases": rows,
        "distribution": distribution,
        "by_kind": by_kind,
        "by_length": by_length,
    }


def v028_diagnostic(model, tokenizer, data, cfg, device, torch, base):
    legacy = diagnostic(model, tokenizer, cfg, device, torch, base)
    expanded = v028_battery(
        model, tokenizer, data.DIAGNOSTICS, data.prompt_for, cfg, device, torch, base
    )

    metrics = dict(legacy["metrics"])
    metrics["copy_position_top1_rate"] = metrics["continuation_top1_rate"]
    for key, value in expanded["metrics"].items():
        metrics["expanded_" + key] = value

    gates = dict(legacy["gates"])
    gates["copy_position_top1"] = (
        metrics["copy_position_top1_rate"] >= float(cfg["minimum_copy_position_top1_rate"])
    )
    gates["expanded_exact_copy"] = (
        metrics["expanded_exact_copy_rate"] >= float(cfg["minimum_expanded_exact_copy_rate"])
    )
    gates["expanded_continuation_top1"] = (
        metrics["expanded_continuation_top1_rate"]
        >= float(cfg["minimum_expanded_continuation_top1_rate"])
    )
    protection = {
        "legacy_exact_copy": metrics["exact_copy_rate"]
        >= float(cfg["protected_legacy_exact_copy_rate"]) - 1e-9,
        "legacy_continuation": metrics["continuation_top1_rate"]
        >= float(cfg["protected_legacy_continuation_top1_rate"]) - 1e-9,
        "expanded_exact_copy": metrics["expanded_exact_copy_rate"]
        >= float(cfg["protected_expanded_exact_copy_rate"]) - 1e-9,
        "expanded_continuation": metrics["expanded_continuation_top1_rate"]
        >= float(cfg["protected_expanded_continuation_top1_rate"]) - 1e-9,
        "expanded_sequence_margin_health": metrics["expanded_sequence_margin_health"]
        >= float(cfg["protected_expanded_sequence_margin_health"]) - 1e-9,
    }
    gates["not_regressed"] = all(protection.values())
    return {
        "passed": all(gates.values()),
        "metrics": metrics,
        "gates": gates,
        "protection": protection,
        "distribution": expanded["distribution"],
        "legacy_cases": legacy["cases"],
        "expanded_by_kind": expanded["by_kind"],
        "expanded_by_length": expanded["by_length"],
        "expanded_cases": expanded["cases"],
    }


def v028_score(diag, val_loss):
    """Selection leads on the full-span margin health.

    v0.0.27 selected on the continuation-only statistic, which omits the first
    token and the EOS decision even though exact copy tests both.
    """
    m = diag["metrics"]
    return (
        1.0 if diag["gates"].get("not_regressed") else 0.0,
        m["expanded_full_sequence_margin_health"],
        m["expanded_exact_copy_rate"],
        m["expanded_continuation_top1_rate"],
        m["exact_copy_rate"],
        m["continuation_top1_rate"],
        -val_loss,
    )


def v028_progress(baseline, final, cfg):
    """Did this run advance the model?

    v0.0.27 gained 0.2025 of margin health and was reported FLAT, because
    v0.0.27's own progress test looked only at exact copy and continuation. A
    large move in the continuous statistic now counts as an advance on its own,
    which is what that run deserved.
    """
    b = baseline["metrics"]
    f = final["metrics"]
    deltas = {}
    for key in (
        "expanded_exact_copy_rate",
        "expanded_continuation_top1_rate",
        "expanded_sequence_margin_health",
        "expanded_full_sequence_margin_health",
        "expanded_sequences_within_reach",
        "expanded_first_token_top1_rate",
        "exact_copy_rate",
        "continuation_top1_rate",
    ):
        if key in b and key in f:
            deltas[key] = {"baseline": b[key], "final": f[key], "gain": f[key] - b[key]}
    exact_gain = deltas["expanded_exact_copy_rate"]["gain"]
    continuation_gain = deltas["expanded_continuation_top1_rate"]["gain"]
    health_gain = deltas["expanded_sequence_margin_health"]["gain"]
    advanced = (
        exact_gain >= float(cfg["progress_expanded_exact_copy_gain"]) - 1e-9
        or continuation_gain >= float(cfg["progress_expanded_continuation_gain"]) - 1e-9
        or health_gain >= float(cfg["progress_sequence_margin_health_gain"]) - 1e-9
    )
    regressed = exact_gain < -1e-9 and continuation_gain < -1e-9 and health_gain < -1e-9
    return {
        "deltas": deltas,
        "verdict": "REGRESSED" if regressed else ("ADVANCED" if advanced else "FLAT"),
    }

'''

TRANSFORMS = (
    ('CONFIG_PIN = "42ed91ac6e6b8ff22873fec6706c4f243f57f2fa"',
     f'CONFIG_PIN = "{CONFIG_COMMIT}"', 1),
    ('DATA_PIN = "14995a94a4d1594463e266c4c5fed0ecec329da9"',
     f'DATA_PIN = "{DATA_COMMIT}"', 1),
    ("0.0.16", "0.0.28", -1),
    ("ember_sequence_copy_v0.0.28.json", "ember_boundary_focus_v0.0.28.json", -1),
    ("ember_sft_data_v015.py", "ember_sft_data_v026.py", -1),
    ("sequence-copy-consolidation", "boundary-focus-repair", -1),
    ('SOURCE_REPO = "Jmiller18899/ember-v0.0.15-t4"',
     'SOURCE_REPO = "Jmiller18899/ember-v0.0.27-t4"', -1),
    ('cfg.get("source_model_name") != "ember-v0.0.15-t4"',
     'cfg.get("source_model_name") != "ember-v0.0.27-t4"', -1),
    ("must start from v0.0.15", "must start from v0.0.27", -1),
    ('if str(source_cfg.get("version")) != "0.0.15":',
     'if str(source_cfg.get("version")) != "0.0.27":', -1),
    ("expected a v0.0.15 checkpoint", "expected a v0.0.27 checkpoint", -1),
    ("v0.0.15 checkpoint run_id", "v0.0.27 checkpoint run_id", -1),
    ("v0.0.15 source state", "v0.0.27 source state", -1),
    ("resolve_v015_source", "resolve_v027_source", -1),
    ("v015_promotion", "v027_promotion", -1),
    ("v015_best_step", "v027_best_step", -1),

    ("def score(diag: dict, val_loss: float):", HELPERS + "def score(diag: dict, val_loss: float):", 1),

    ("        data.assert_clean(train_rows, val_rows)\n"
     "        owner, repo = verify_hf_output_access(api, cfg, work)",
     "        data.assert_clean(train_rows, val_rows)\n"
     "        parity = assert_format_parity(data, train_rows)\n"
     "        owner, repo = verify_hf_output_access(api, cfg, work)", 1),

    ('baseline = diagnostic(model, tokenizer, cfg, "cpu", torch, base)',
     'baseline = v028_diagnostic(model, tokenizer, data, cfg, "cpu", torch, base)', 1),
    ("diag = diagnostic(model, tokenizer, cfg, device, torch, base)",
     "diag = v028_diagnostic(model, tokenizer, data, cfg, device, torch, base)", 1),
    ("final = diagnostic(model, tokenizer, cfg, device, torch, base)",
     "final = v028_diagnostic(model, tokenizer, data, cfg, device, torch, base)", 1),
    ("current = score(diag, val_loss)", "current = v028_score(diag, val_loss)", 1),

    # The free preflight now answers whether a paid session is worth launching.
    ('"baseline": baseline["metrics"],\n            "hf": {',
     '"baseline": baseline["metrics"],\n'
     '            "baseline_distribution": baseline["distribution"],\n'
     '            "baseline_by_kind": baseline["expanded_by_kind"],\n'
     '            "baseline_by_length": baseline["expanded_by_length"],\n'
     '            "format_parity": parity,\n'
     '            "hf": {', 1),

    ('if args.preflight_only:\n'
     '            print("EMBER_HF_V016_PREFLIGHT=PASS", flush=True)\n'
     '            return',
     'if args.preflight_only:\n'
     '            probe_generator = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))\n'
     '            probe = sample_boundary_focus_indices(train, 2, cfg, probe_generator, model, "cpu", torch)\n'
     '            if len(probe) != 2:\n'
     '                raise RuntimeError("v0.0.28 boundary mining preflight returned wrong batch size")\n'
     '            reach = baseline["metrics"]["expanded_sequences_within_reach"]\n'
     '            print("EMBER_V028_FORMAT_PARITY=PASS", flush=True)\n'
     '            print("EMBER_V028_BOUNDARY_MINING=PASS", flush=True)\n'
     '            print(f"EMBER_V028_SEQUENCES_WITHIN_REACH={reach:.4f}", flush=True)\n'
     '            print(json.dumps(baseline["distribution"], indent=2, sort_keys=True), flush=True)\n'
     '            if reach <= 0.0:\n'
     '                print("EMBER_V028_ADVICE=NO_REACHABLE_SEQUENCES_DO_NOT_LAUNCH_T4", flush=True)\n'
     '            print("EMBER_HF_V028_PREFLIGHT=PASS", flush=True)\n'
     '            return', 1),

    ("best_score = None\n        best_step = -1\n        history = []",
     "best_score = v028_score(baseline, -1e9)\n"
     "        best_step = -1\n"
     "        history = []\n"
     "        save_checkpoint(best_path, model=model, optimizer=optimizer, tokenizer=tokenizer, "
     "model_config=model.cfg, train_config=cfg, step=-1, best_val_loss=float(\"inf\"), run_id=run_id)", 1),

    ('ids = torch.randint(len(train), (int(cfg["batch_size"]),), generator=generator).tolist()',
     'ids = sample_boundary_focus_indices(train, int(cfg["batch_size"]), cfg, generator, model, device, torch)', 1),
    ("loss = base.weighted_loss(model, x, y, w, torch) / accum",
     "loss = boundary_focus_loss(model, x, y, w, cfg, torch) / accum", 1),

    ('"promotion": "PASS" if final["passed"] else "FAIL",',
     '"format_parity": parity,\n'
     '            "progress": v028_progress(baseline, final, cfg),\n'
     '            "baseline_distribution": baseline["distribution"],\n'
     '            "final_distribution": final["distribution"],\n'
     '            "final_by_kind": final["expanded_by_kind"],\n'
     '            "final_by_length": final["expanded_by_length"],\n'
     '            "promotion": "PASS" if final["passed"] else "FAIL",', 1),

    ('print(f"EMBER_HF_V016_PROMOTION={report[\'promotion\']}", flush=True)',
     'print(f"EMBER_HF_V028_PROMOTION={report[\'promotion\']}", flush=True)\n'
     '        print(f"EMBER_V028_PROGRESS={report[\'progress\'][\'verdict\']}", flush=True)\n'
     '        print(json.dumps(report["progress"]["deltas"], indent=2, sort_keys=True), flush=True)\n'
     '        print(json.dumps(report["final_distribution"], indent=2, sort_keys=True), flush=True)', 1),

    ("EMBER_HF_V016_", "EMBER_HF_V028_", -1),
    ("EMBER_V016_", "EMBER_V028_", -1),
)

OPTIONAL_TARGETS = frozenset({"0.0.16", "EMBER_HF_V016_", "EMBER_V016_"})


def apply_transforms(source: str) -> str:
    for old, new, count in TRANSFORMS:
        if old not in source and old not in OPTIONAL_TARGETS:
            raise RuntimeError(f"v0.0.28 transform target missing: {old!r}")
        source = source.replace(old, new) if count < 0 else source.replace(old, new, count)
    return source


def unmatched_transform_targets(source: str) -> list[str]:
    missing = []
    for old, new, count in TRANSFORMS:
        if old not in source:
            if old not in OPTIONAL_TARGETS:
                missing.append(old)
            continue
        source = source.replace(old, new) if count < 0 else source.replace(old, new, count)
    return missing


def main() -> None:
    with urllib.request.urlopen(BASE_URL) as response:
        scaffold = response.read().decode("utf-8")
    source = apply_transforms(scaffold)
    print("EMBER_V028_BOUNDARY_FOCUS_TRANSFORM=PASS", flush=True)
    exec(compile(source, "ember_hf_sft_v028_wrapper_runtime.py", "exec"), {"__name__": "__main__"})


if __name__ == "__main__":
    main()
