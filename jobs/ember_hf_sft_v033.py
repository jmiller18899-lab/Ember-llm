# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["huggingface-hub>=1.4", "sentencepiece>=0.2", "torch>=2.4"]
# ///
"""Ember v0.0.33: boundary focus, from the v0.0.27 checkpoint.

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
# The v0.0.26 curriculum, carried unchanged through v0.0.27 and v0.0.33.
DATA_COMMIT = "36a213d9fecaf9ab8348132d6036e9c99a17e18d"
# The commit carrying config/ember_envelope_copy_v0.0.33.json.
CONFIG_COMMIT = "d0f2081bfea80b3f755d0006cb5d371f06fca2cd"
BASE_URL = (
    "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/"
    f"{BASE_COMMIT}/jobs/ember_hf_sft_v016.py"
)

HELPERS = r'''
def _v033_field_signature(value):
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
        signature = _v033_field_signature(str(row["value"]))
        bucket = supported.setdefault(str(row["kind"]), {})
        bucket[signature] = bucket.get(signature, 0) + 1
    missing = []
    per_case = []
    for kind, value, _ in data.DIAGNOSTICS:
        signature = _v033_field_signature(value)
        rows = int(supported.get(str(kind), {}).get(signature, 0))
        per_case.append({"kind": kind, "value": value, "template": signature, "train_rows": rows})
        if rows == 0:
            missing.append({"kind": kind, "value": value, "template": signature})
    if missing:
        raise RuntimeError(
            "v0.0.33 format parity failed; the curriculum cannot emit these held-out "
            "templates: %s" % missing
        )
    return {
        "held_out_cases": len(per_case),
        "templates_supported": len(per_case),
        "min_train_rows_for_a_held_out_template": min(c["train_rows"] for c in per_case),
    }


def v033_supervised_span(y, weights, cfg, torch):
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
    # The envelope encoder puts eos_token_weight on the EOS decision, which is
    # outside the value's copy block, so it is added explicitly. Without this the
    # sequence hinge would stop caring whether the envelope ever closes.
    eos_mask = active & weights.eq(float(cfg["eos_token_weight"]))
    span = (copy_mask | before | after | eos_mask) & active
    return copy_mask, span


def v033_gaps(logits, y, mask, torch):
    """Per-position margin between the correct token and its best competitor."""
    selected = logits[mask].float()
    targets = y[mask]
    correct = selected.gather(1, targets.unsqueeze(1)).squeeze(1)
    top2_values, top2_indices = torch.topk(selected, k=2, dim=1)
    best_is_correct = top2_indices[:, 0].eq(targets)
    best_wrong = torch.where(best_is_correct, top2_values[:, 1], top2_values[:, 0])
    return correct - best_wrong, best_is_correct


def v033_row_worst(gap, mask, cfg, torch):
    """The k worst positions per row, summed, plus the row's worst margin.

    v0.0.28 reduced each row to the maximum of its hinges, so one pass could
    retire at most one bad decision per row. The v0.0.28 result says the average
    failing row carries about 1.66 wrong decisions: 730 x (1 - 0.8658) = 98
    wrong continuation decisions over 62 failing rows, plus five first-token
    failures. Summing the k worst hinges pushes that many at full strength, and
    at k = 1 it is exactly the v0.0.28 statistic.

    Returns (worst_k_sum, worst_gap, has_targets). Rows outside the reachable
    band are not dropped; the caller down-weights them, so hopeless sequences
    keep learning slowly instead of taking every mined slot.
    """
    import torch.nn.functional as F
    hinge_full = torch.zeros(mask.shape, dtype=gap.dtype, device=gap.device)
    hinge_full[mask] = F.relu(float(cfg["sequence_margin"]) - gap)
    gap_full = torch.full(mask.shape, float("inf"), dtype=gap.dtype, device=gap.device)
    gap_full[mask] = gap

    width = hinge_full.shape[1]
    k = max(1, min(int(cfg["sequence_worst_k"]), width))
    top = hinge_full.topk(k=k, dim=1).values
    # Non-target positions hold a zero hinge, so a row with fewer than k
    # supervised decisions contributes only the ones it actually has.
    counts = mask.sum(dim=1)
    ranks = torch.arange(k, device=hinge_full.device).unsqueeze(0)
    valid = ranks < counts.unsqueeze(1).clamp_max(k)
    worst_k_sum = (top * valid).sum(dim=1)
    return worst_k_sum, gap_full.min(dim=1).values, mask.any(dim=1)


def v033_boundary_weight(worst_gap, cfg, torch):
    """1.0 inside the reachable band, out_of_band_weight below it."""
    band_low = float(cfg["boundary_band_low"])
    far = float(cfg["out_of_band_weight"])
    in_band = worst_gap.ge(band_low)
    return torch.where(in_band, torch.ones_like(worst_gap), torch.full_like(worst_gap, far)), in_band


def multi_position_loss(model, x, y, weights, cfg, torch):
    """CE, plus the per-token hinge, plus a boundary-focused worst-position hinge."""
    import torch.nn.functional as F
    logits, _ = model(x, None)
    raw = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction="none"
    ).reshape_as(y)
    active = y.ne(-100)
    denom = (weights * active).sum().clamp_min(1.0)
    ce = (raw * weights * active).sum() / denom

    copy_mask, span_mask = v033_supervised_span(y, weights, cfg, torch)
    if not bool(copy_mask.any().item()):
        return ce

    copy_gap, copy_best_is_correct = v033_gaps(logits, y, copy_mask, torch)
    token_hinge = F.relu(float(cfg["margin"]) - copy_gap)
    hard = (~copy_best_is_correct).to(token_hinge.dtype)
    multiplier = 1.0 + hard * (float(cfg["hard_error_multiplier"]) - 1.0)
    token_term = (token_hinge * multiplier).mean()

    span_gap, _ = v033_gaps(logits, y, span_mask, torch)
    worst_k, worst_gap, has_targets = v033_row_worst(span_gap, span_mask, cfg, torch)
    focus, _ = v033_boundary_weight(worst_gap, cfg, torch)
    rows = has_targets.to(worst_k.dtype)
    sequence_term = (worst_k * focus * rows).sum() / rows.sum().clamp_min(1.0)

    return (
        ce
        + float(cfg["margin_loss_weight"]) * token_term
        + float(cfg["sequence_margin_loss_weight"]) * sequence_term
    )


def sample_multi_position_indices(train, batch_size, cfg, generator, model, device, torch):
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
        copy_mask, span_mask = v033_supervised_span(y, w, cfg, torch)
        if not bool(span_mask.any().item()):
            return uniform
        span_gap, _ = v033_gaps(logits, y, span_mask, torch)
        worst_k, worst_gap, has_targets = v033_row_worst(span_gap, span_mask, cfg, torch)
        # An out-of-band row takes the priority a single band-edge decision would
        # have, scaled down, so near-boundary rows lead while hopeless rows still
        # appear. Ranking in-band rows by the summed statistic keeps mining and
        # the loss measuring the same thing.
        far_priority = (
            (float(cfg["sequence_margin"]) - float(cfg["boundary_band_low"]))
            * float(cfg["out_of_band_weight"])
        )
        in_band = worst_gap.ge(float(cfg["boundary_band_low"]))
        priority = torch.where(in_band, worst_k, torch.full_like(worst_k, far_priority))
        priority = priority * has_targets.to(priority.dtype)
        chosen = torch.topk(priority, k=min(hard_count, len(candidates))).indices.tolist()
    return [candidates[j] for j in chosen] + uniform[: int(batch_size) - hard_count]


def v033_sequence_stats(model, tokenizer, prompt, expected, device, torch, base):
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


def _v033_bucket(tokens, edges):
    for edge in edges:
        if tokens <= edge:
            return "<=%d" % edge
    return ">%d" % edges[-1]


def _v033_percentiles(values, points):
    if not values:
        return {}
    ordered = sorted(values)
    out = {}
    for p in points:
        index = min(len(ordered) - 1, max(0, int(round((float(p) / 100.0) * (len(ordered) - 1)))))
        out["p%d" % int(p)] = ordered[index]
    return out


def _v033_histogram(values, edges):
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


def v033_battery(model, tokenizer, cases, prompt_fn, cfg, device, torch, base):
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
        stats = v033_sequence_stats(model, tokenizer, prompt, value, device, torch, base)
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
        "span_min_gap_percentiles": _v033_percentiles(
            span_gaps, cfg["margin_report_percentiles"]
        ),
        "span_min_gap_histogram": _v033_histogram(span_gaps, cfg["margin_histogram_edges"]),
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
            (_v033_bucket(row["continuation_tokens"], edges), by_length),
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


def v033_meets(value, threshold, cfg):
    """Threshold comparison that tolerates the stored decimal's truncation.

    The configs store 5/9 as 0.5555555556, which is larger than 5/9 in double
    precision by 4.4e-11, and 7/9 and 8/9 the same way. The scaffold compares
    with a raw >=, so a rate that exactly equals its stated bar is recorded as
    unmet. v0.0.28 hit exactly 5/9 on legacy exact copy and tripped it.
    """
    return float(value) >= float(threshold) - float(cfg.get("gate_tolerance", 1e-9))


def v033_legacy_continuation_tokens(model, tokenizer, cfg, device, torch, base):
    """Total supervised continuation tokens across the legacy nine.

    The scaffold reports legacy continuation as a rate only, which hides that a
    0.90 gate on 69 tokens is really a 63/69 = 0.9130 gate.
    """
    correct = 0
    total = 0
    for value, _ in base.DIAGNOSTICS:
        c, n = continuation_top1_counts(
            model, tokenizer, base.prompt_for(value), value, device, torch, base
        )
        correct += c
        total += n
    return correct, total


def v033_band_flow(baseline, final, cfg):
    """Per-case transition between below-band, within-reach and correct.

    v0.0.28 ended with more cases in the reachable band than it started with
    while eight crossed out of it, which means the band was refilling from
    below. Whether that keeps happening is the difference between a pipeline
    that is still flowing and one that has saturated, and a pair of counts
    cannot tell them apart.
    """
    band_low = float(cfg["boundary_band_low"])

    def state(gap):
        if gap is None:
            return "unknown"
        if gap >= 0.0:
            return "correct"
        if gap >= band_low:
            return "within_reach"
        return "below_band"

    before = {row["value"]: row.get("span_min_gap") for row in baseline.get("expanded_cases", [])}
    after = {row["value"]: row.get("span_min_gap") for row in final.get("expanded_cases", [])}
    labels = ("below_band", "within_reach", "correct", "unknown")
    matrix = {a: {b: 0 for b in labels} for a in labels}
    crossed = []
    for value, gap in after.items():
        if value not in before:
            continue
        start, end = state(before[value]), state(gap)
        matrix[start][end] += 1
        if start != "correct" and end == "correct":
            crossed.append({"value": value, "from": before[value], "to": gap})
    return {
        "transitions": {a: {b: n for b, n in row.items() if n} for a, row in matrix.items() if any(row.values())},
        "crossed_into_correct": sorted(crossed, key=lambda item: -item["to"]),
        "entered_band_from_below": matrix["below_band"]["within_reach"],
        "fell_out_of_correct": sum(matrix["correct"][b] for b in ("below_band", "within_reach")),
    }


def v033_gate_distance(final, cfg):
    """How many discrete units each unmet gate is short.

    A 0.90 gate on a 69-token denominator is a different requirement from a 0.90
    gate on 730, and the rate alone does not say so.
    """
    m = final["metrics"]
    cases = len(final.get("expanded_cases", [])) or int(cfg["expanded_diagnostic_cases"])
    checks = [
        ("expanded_exact_copy_rate", cases, float(cfg["minimum_expanded_exact_copy_rate"]), "cases"),
        ("expanded_continuation_top1_rate", int(m.get("expanded_continuation_tokens", 0)),
         float(cfg["minimum_expanded_continuation_top1_rate"]), "tokens"),
        ("exact_copy_rate", int(cfg["legacy_diagnostic_cases"]),
         float(cfg["minimum_exact_copy_rate"]), "cases"),
        ("continuation_top1_rate", int(m.get("legacy_continuation_tokens", 0)),
         float(cfg["minimum_continuation_top1_rate"]), "tokens"),
    ]
    out = {}
    for key, denominator, gate, unit in checks:
        if key not in m or denominator <= 0:
            continue
        have = int(round(m[key] * denominator))
        need = have
        while need <= denominator and not v033_meets(need / denominator, gate, cfg):
            need += 1
        out[key] = {
            "measured": "%d/%d" % (have, denominator),
            "rate": m[key],
            "gate": gate,
            "needs": "%d/%d" % (need, denominator),
            "effective_gate": need / denominator if denominator else None,
            "short_by": max(0, need - have),
            "unit": unit,
        }
    return out


def v033_diagnostic(model, tokenizer, data, cfg, device, torch, base):
    legacy = diagnostic(model, tokenizer, cfg, device, torch, base)
    legacy_correct, legacy_total = v033_legacy_continuation_tokens(
        model, tokenizer, cfg, device, torch, base
    )
    envelope = v033_envelope_battery(model, tokenizer, data, cfg, device, torch, base)
    expanded = v033_battery(
        model, tokenizer, data.DIAGNOSTICS, data.prompt_for, cfg, device, torch, base
    )

    metrics = dict(legacy["metrics"])
    metrics["copy_position_top1_rate"] = metrics["continuation_top1_rate"]
    metrics["legacy_continuation_tokens"] = legacy_total
    metrics["legacy_continuation_correct"] = legacy_correct
    for key, value in expanded["metrics"].items():
        metrics["expanded_" + key] = value
    for key, value in envelope["metrics"].items():
        metrics["envelope_" + key] = value

    # The scaffold's gates use a raw >=; recompute them with the tolerance the
    # protection checks already use, so a rate that exactly equals its bar counts.
    gates = dict(legacy["gates"])
    gates["top20"] = v033_meets(
        metrics["first_token_top20_rate"], cfg["minimum_first_token_top20_rate"], cfg
    )
    gates["top5"] = v033_meets(
        metrics["first_token_top5_rate"], cfg["minimum_first_token_top5_rate"], cfg
    )
    gates["teacher_forced"] = v033_meets(
        metrics["teacher_forced_expected_win_rate"],
        cfg["minimum_teacher_forced_expected_win_rate"], cfg
    )
    gates["clean_stop"] = v033_meets(
        metrics["clean_stop_rate"], cfg["minimum_clean_stop_rate"], cfg
    )
    gates["exact_copy"] = v033_meets(
        metrics["exact_copy_rate"], cfg["minimum_exact_copy_rate"], cfg
    )
    gates["continuation_top1"] = v033_meets(
        metrics["continuation_top1_rate"], cfg["minimum_continuation_top1_rate"], cfg
    )
    gates["copy_position_top1"] = v033_meets(
        metrics["copy_position_top1_rate"], cfg["minimum_copy_position_top1_rate"], cfg
    )
    gates["expanded_exact_copy"] = v033_meets(
        metrics["expanded_exact_copy_rate"], cfg["minimum_expanded_exact_copy_rate"], cfg
    )
    gates["expanded_continuation_top1"] = v033_meets(
        metrics["expanded_continuation_top1_rate"],
        cfg["minimum_expanded_continuation_top1_rate"], cfg
    )
    # The metric this phase exists to move: the v0.0.32 arguments_grounded
    # check, measured on the whole held-out battery instead of four cases.
    gates["slot_exact"] = v033_meets(
        metrics["envelope_slot_exact_rate"], cfg["minimum_slot_exact_rate"], cfg
    )
    protection = {
        key: v033_meets(metrics[metric], cfg[key], cfg)
        for key, metric in (
            ("protected_legacy_exact_copy_rate", "exact_copy_rate"),
            ("protected_legacy_continuation_top1_rate", "continuation_top1_rate"),
            ("protected_expanded_exact_copy_rate", "expanded_exact_copy_rate"),
            ("protected_expanded_continuation_top1_rate", "expanded_continuation_top1_rate"),
            ("protected_expanded_sequence_margin_health", "expanded_sequence_margin_health"),
            ("protected_expanded_full_sequence_margin_health",
             "expanded_full_sequence_margin_health"),
            ("protected_expanded_first_token_top1_rate", "expanded_first_token_top1_rate"),
        )
        if key in cfg
    }
    gates["not_regressed"] = all(protection.values())
    return {
        "passed": all(gates.values()),
        "metrics": metrics,
        "gates": gates,
        "protection": protection,
        "distribution": expanded["distribution"],
        "envelope_cases": envelope["cases"],
        "envelope_by_kind": envelope["by_kind"],
        "legacy_cases": legacy["cases"],
        "expanded_by_kind": expanded["by_kind"],
        "expanded_by_length": expanded["by_length"],
        "expanded_cases": expanded["cases"],
    }


def v033_score(diag, val_loss):
    """Selection leads on the full-span margin health.

    v0.0.27 selected on the continuation-only statistic, which omits the first
    token and the EOS decision even though exact copy tests both.
    """
    m = diag["metrics"]
    return (
        # Protection first: a checkpoint that trades away the 43/90 bare-value
        # capability to win at slot copying is not an improvement.
        1.0 if diag["gates"].get("not_regressed") else 0.0,
        # Then the continuous statistic for the slot, for the same reason
        # v0.0.28 stopped leading on a 90-case count.
        m["envelope_slot_margin_health"],
        m["envelope_slot_exact_rate"],
        m["expanded_full_sequence_margin_health"],
        m["expanded_exact_copy_rate"],
        m["expanded_continuation_top1_rate"],
        m["exact_copy_rate"],
        -val_loss,
    )


def v033_progress(baseline, final, cfg):
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
        "envelope_slot_exact_rate",
        "envelope_slot_margin_health",
        "envelope_slot_continuation_top1_rate",
        "envelope_json_valid_rate",
        "envelope_tool_name_rate",
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
    slot_gain = deltas["envelope_slot_exact_rate"]["gain"] if "envelope_slot_exact_rate" in deltas else 0.0
    advanced = (
        slot_gain >= float(cfg["progress_slot_exact_gain"]) - 1e-9
        or exact_gain >= float(cfg["progress_expanded_exact_copy_gain"]) - 1e-9
        or continuation_gain >= float(cfg["progress_expanded_continuation_gain"]) - 1e-9
        or health_gain >= float(cfg["progress_sequence_margin_health_gain"]) - 1e-9
    )
    regressed = exact_gain < -1e-9 and continuation_gain < -1e-9 and health_gain < -1e-9
    # band_flow and gate_distance are read by the final reporter and by the
    # preflight. v0.0.29 defined both helpers and printed both fields but never
    # attached them here, which raised KeyError('gate_distance') after every
    # artifact had already uploaded. tests/test_ember_v033_report_contract.py
    # derives the required keys from the transform text so it cannot recur.
    return {
        "deltas": deltas,
        "band_flow": v033_band_flow(baseline, final, cfg),
        "gate_distance": v033_gate_distance(final, cfg),
        "verdict": "REGRESSED" if regressed else ("ADVANCED" if advanced else "FLAT"),
    }

# --------------------------------------------------------------------------
# v0.0.33: the copy target moves inside a tool-call envelope.
#
# Every copy phase since v0.0.15 trained completions whose first token is the
# target. The v0.0.32 gate showed the consequence on the promoted checkpoint:
# all four tool calls produced a correct envelope, a correct tool name and a
# correct argument key, and failed exactly one check, arguments_grounded --
# Detroit became Austin, 347 and 28 became 337, Tokyo became America/Anchorage.
# The same checkpoint reproduces 43 of 90 held-out literal strings exactly.
#
# So the copy skill exists and does not reach the argument slot, where the
# copied span begins about fifteen tokens into the completion. This phase moves
# the target there and changes nothing else.
# --------------------------------------------------------------------------

V033_EOT = "<|endoftext|>"

# Semantically plausible pairings, so routing -- which the checkpoint already
# gets right -- is reinforced rather than confused. The first four tools are the
# ones the v0.0.8 evaluation battery actually uses.
ENVELOPE_TOOLS = {
    "entity": ("weather", "location"),
    "expression": ("calculator", "expression"),
    "digits": ("calculator", "expression"),
    "model_id": ("web_search", "query"),
    "short_code": ("web_search", "query"),
    "long_code": ("lookup", "id"),
    "mixed": ("lookup", "id"),
    "url": ("fetch_url", "url"),
    "path": ("read_file", "path"),
}


def envelope_for(kind):
    return ENVELOPE_TOOLS.get(str(kind), ("lookup", "id"))


def envelope_prompt(value, old, fallback, tool, key):
    return (
        "<|system|>\nYou are Ember. Call the %s tool with JSON arguments. "
        "Use TARGET exactly as the %s. Do not explain, normalize, or call another "
        "tool. Stop at endoftext.\n"
        "<|user|>\nIgnore old=%s and fallback=%s. TARGET=%s. Call %s for TARGET.\n"
        "<|assistant|>\n" % (tool, key, old, fallback, value, tool)
    )


def envelope_parts(tool, key):
    """(head, tail); the value sits between them, inside the JSON string.

    Key order matches what the checkpoint already emits, so this phase teaches
    the envelope it already produces and only changes what goes in the slot.
    """
    return (
        '<|tool|>\n{"arguments":{"%s":"' % key,
        '"},"name":"%s"}\n%s\n' % (tool, V033_EOT),
    )


def envelope_distractors(prompt):
    """Reuse the distractors the bare-value curriculum already chose."""
    old = prompt.split("old=", 1)[1].split(" and fallback=", 1)[0]
    fallback = prompt.split(" and fallback=", 1)[1].split(". TARGET=", 1)[0]
    return old, fallback


def to_envelope_row(row):
    tool, key = envelope_for(row.get("kind"))
    value = str(row["value"])
    old, fallback = envelope_distractors(row["prompt"])
    head, tail = envelope_parts(tool, key)
    envelope = {
        "id": "envelope-" + str(row["id"]),
        "kind": row.get("kind"),
        "value": value,
        "tool": tool,
        "argument_key": key,
        "prompt": envelope_prompt(value, old, fallback, tool, key),
        "completion_head": head,
        "completion": head + value + tail,
    }
    if envelope["prompt"].count(value) != 1:
        raise RuntimeError("v0.0.33 target must appear once in the prompt: %s" % row["id"])
    if envelope["completion"].count(value) != 1:
        raise RuntimeError("v0.0.33 target must appear once in the completion: %s" % row["id"])
    return envelope


def encode_envelope_row(tokenizer, row, cfg, torch):
    """Weight the value's tokens, not the envelope's punctuation.

    Returns (None, reason) when the tokenizer merges across a boundary of the
    value. That cannot be checked without the tokenizer, so the caller counts
    rejections and the preflight fails if too many are unusable -- an untestable
    assumption surfaces on CPU rather than inside a paid run.
    """
    prompt, head, value = row["prompt"], row["completion_head"], row["value"]
    prompt_ids = list(tokenizer.encode(prompt))
    head_ids = list(tokenizer.encode(prompt + head))
    value_ids = list(tokenizer.encode(prompt + head + value))
    full_ids = list(tokenizer.encode(prompt + row["completion"]))
    if head_ids[:len(prompt_ids)] != prompt_ids:
        return None, "prompt_head_boundary"
    if value_ids[:len(head_ids)] != head_ids:
        return None, "head_value_boundary"
    if full_ids[:len(value_ids)] != value_ids:
        return None, "value_tail_boundary"
    if len(full_ids) > int(cfg["block_size"]) + 1:
        return None, "too_long"
    if len(value_ids) - len(head_ids) < int(cfg["minimum_value_tokens"]):
        return None, "value_too_short"
    eot_id = int(tokenizer.encode(V033_EOT)[-1])
    eot_positions = [i for i in range(len(value_ids), len(full_ids)) if int(full_ids[i]) == eot_id]
    if not eot_positions:
        return None, "missing_eos"

    block = int(cfg["block_size"])
    x = torch.full((block,), eot_id, dtype=torch.long)
    y = torch.full((block,), -100, dtype=torch.long)
    w = torch.zeros((block,), dtype=torch.float32)
    seq_x = torch.tensor(full_ids[:-1], dtype=torch.long)
    seq_y = torch.tensor(full_ids[1:], dtype=torch.long)
    x[:len(seq_x)] = seq_x

    first = len(prompt_ids) - 1          # predicts the envelope's first token
    eos_target = eot_positions[0] - 1    # predicts EOS
    value_start = len(head_ids) - 1      # predicts the value's first token
    value_end = len(value_ids) - 1       # one past the last value-predicting index

    y[first:eos_target + 1] = seq_y[first:eos_target + 1]
    w[first:eos_target + 1] = float(cfg["envelope_token_weight"])
    for pos in range(value_start, value_end):
        w[pos] = float(cfg["copy_token_weight"])
    w[value_start] = float(cfg["first_token_weight"])
    w[eos_target] = float(cfg["eos_token_weight"])
    return {
        "x": x, "y": y, "w": w, "row": row,
        "value_span": (value_start, value_end), "eos_target": eos_target,
    }, None


def encode_envelope_rows(tokenizer, rows, cfg, torch, label):
    encoded = []
    rejected = {}
    for row in rows:
        item, reason = encode_envelope_row(tokenizer, to_envelope_row(row), cfg, torch)
        if item is None:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        encoded.append(item)
    usable = len(encoded) / len(rows) if rows else 0.0
    floor = float(cfg["minimum_envelope_encodable_fraction"])
    if usable < floor:
        raise RuntimeError(
            "v0.0.33 %s: only %.3f of rows encode cleanly inside the envelope "
            "(floor %.3f); rejections: %s" % (label, usable, floor, rejected)
        )
    return encoded, {"rows": len(rows), "encoded": len(encoded),
                     "encodable_fraction": usable, "rejected": rejected}


def _v033_first_json_object(text):
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start:index + 1])
                except Exception:
                    return None
                return value if isinstance(value, dict) else None
    return None


def v033_envelope_stats(model, tokenizer, row, device, torch):
    """Teacher-forced accuracy and worst margin over the value span only."""
    prompt, head, value = row["prompt"], row["completion_head"], row["value"]
    head_ids = list(tokenizer.encode(prompt + head))
    value_ids = list(tokenizer.encode(prompt + head + value))
    full_ids = list(tokenizer.encode(prompt + row["completion"]))
    if value_ids[:len(head_ids)] != head_ids or full_ids[:len(value_ids)] != value_ids:
        return None
    positions = list(range(len(head_ids) - 1, len(value_ids) - 1))
    if not positions:
        return None
    x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    y = full_ids[1:]
    with torch.inference_mode():
        logits, _ = model(x, None)
        row_logits = logits[0].float()
        pred = torch.argmax(row_logits, dim=-1)
    correct = sum(int(pred[pos].item()) == int(y[pos]) for pos in positions)
    worst = None
    for pos in positions:
        target = int(y[pos])
        vector = row_logits[pos]
        competitors = vector.clone()
        competitors[target] = float("-inf")
        gap = float(vector[target].item()) - float(competitors.max().item())
        worst = gap if worst is None else min(worst, gap)
    return {"correct": correct, "tokens": len(positions), "slot_min_gap": worst}


def v033_envelope_battery(model, tokenizer, data, cfg, device, torch, base):
    """The held-out values, placed inside a tool-call envelope.

    slot_exact_rate is the metric this phase exists to move. It is the
    arguments_grounded check from the v0.0.32 gate, measured on 90 cases
    instead of 4.
    """
    model.eval()
    rows = []
    correct_tokens = 0
    total_tokens = 0
    floor = float(cfg["margin_health_floor"])
    ceiling = float(cfg["margin_health_ceiling"])
    for kind, value, _ in data.DIAGNOSTICS[:int(cfg["envelope_battery_cases"])]:
        row = to_envelope_row({"id": "diagnostic", "kind": kind, "value": value,
                               "prompt": data.prompt_for(value)})
        text, stopped = base.greedy(model, tokenizer, row["prompt"], device, torch)
        visible = text.split(V033_EOT, 1)[0]
        payload = _v033_first_json_object(visible)
        arguments = (payload or {}).get("arguments")
        slot = arguments.get(row["argument_key"]) if isinstance(arguments, dict) else None
        stats = v033_envelope_stats(model, tokenizer, row, device, torch)
        if stats:
            correct_tokens += stats["correct"]
            total_tokens += stats["tokens"]
        rows.append({
            "kind": kind, "value": value, "tool": row["tool"], "key": row["argument_key"],
            "generated": visible.strip()[:160],
            "json_valid": payload is not None,
            "tool_name_matches": bool(payload) and payload.get("name") == row["tool"],
            "slot_present": slot is not None,
            "slot_exact": slot == value,
            "envelope_exact": visible.strip() == row["completion"].split(V033_EOT, 1)[0].strip(),
            "clean_stop": stopped,
            "slot_min_gap": stats["slot_min_gap"] if stats else None,
        })
    model.train()
    total = len(rows)
    gaps = [r["slot_min_gap"] for r in rows if r["slot_min_gap"] is not None]
    by_kind = {}
    for row in rows:
        bucket = by_kind.setdefault(row["kind"], {"cases": 0, "slot_exact": 0})
        bucket["cases"] += 1
        bucket["slot_exact"] += int(row["slot_exact"])
    metrics = {
        "slot_exact_rate": sum(r["slot_exact"] for r in rows) / total,
        "envelope_exact_rate": sum(r["envelope_exact"] for r in rows) / total,
        "json_valid_rate": sum(r["json_valid"] for r in rows) / total,
        "tool_name_rate": sum(r["tool_name_matches"] for r in rows) / total,
        "slot_present_rate": sum(r["slot_present"] for r in rows) / total,
        "clean_stop_rate": sum(r["clean_stop"] for r in rows) / total,
        "slot_continuation_top1_rate": (correct_tokens / total_tokens) if total_tokens else 0.0,
        "slot_margin_health": (
            sum(min(max(g, floor), ceiling) for g in gaps) / len(gaps) if gaps else floor
        ),
        "cases": total,
    }
    return {"metrics": metrics, "cases": rows, "by_kind": by_kind}

'''

TRANSFORMS = (
    ('CONFIG_PIN = "42ed91ac6e6b8ff22873fec6706c4f243f57f2fa"',
     f'CONFIG_PIN = "{CONFIG_COMMIT}"', 1),
    ('DATA_PIN = "14995a94a4d1594463e266c4c5fed0ecec329da9"',
     f'DATA_PIN = "{DATA_COMMIT}"', 1),
    ("0.0.16", "0.0.33", -1),
    ("ember_sequence_copy_v0.0.33.json", "ember_envelope_copy_v0.0.33.json", -1),
    ("ember_sft_data_v015.py", "ember_sft_data_v026.py", -1),
    ("sequence-copy-consolidation", "envelope-placed-copy", -1),
    ('SOURCE_REPO = "Jmiller18899/ember-v0.0.15-t4"',
     'SOURCE_REPO = "Jmiller18899/ember-v0.0.31-t4"', -1),
    ('cfg.get("source_model_name") != "ember-v0.0.15-t4"',
     'cfg.get("source_model_name") != "ember-v0.0.31-t4"', -1),
    ("must start from v0.0.15", "must start from v0.0.31", -1),
    ('if str(source_cfg.get("version")) != "0.0.15":',
     'if str(source_cfg.get("version")) != "0.0.31":', -1),
    ("expected a v0.0.15 checkpoint", "expected a v0.0.31 checkpoint", -1),
    ("v0.0.15 checkpoint run_id", "v0.0.31 checkpoint run_id", -1),
    ("v0.0.15 source state", "v0.0.31 source state", -1),
    ("resolve_v015_source", "resolve_v031_source", -1),
    ("v015_promotion", "v031_promotion", -1),
    ("v015_best_step", "v031_best_step", -1),

    ("def score(diag: dict, val_loss: float):", HELPERS + "def score(diag: dict, val_loss: float):", 1),

    ("        data.assert_clean(train_rows, val_rows)\n"
     "        owner, repo = verify_hf_output_access(api, cfg, work)",
     "        data.assert_clean(train_rows, val_rows)\n"
     "        parity = assert_format_parity(data, train_rows)\n"
     "        owner, repo = verify_hf_output_access(api, cfg, work)", 1),

    # Train on envelope-placed copies instead of bare values. This is the one
    # change of the phase; every weight, the objective and the schedule are the
    # v0.0.31 ones.
    ("        train = [base.encode_row(tokenizer, row, cfg, torch) for row in train_rows]\n"
     "        val = [base.encode_row(tokenizer, row, cfg, torch) for row in val_rows]",
     "        train, train_encoding = encode_envelope_rows(tokenizer, train_rows, cfg, torch, \"train\")\n"
     "        val, val_encoding = encode_envelope_rows(tokenizer, val_rows, cfg, torch, \"validation\")\n"
     "        print(f\"EMBER_V033_TRAIN_ENCODING={json.dumps(train_encoding, sort_keys=True)}\", flush=True)\n"
     "        print(f\"EMBER_V033_VAL_ENCODING={json.dumps(val_encoding, sort_keys=True)}\", flush=True)", 1),

    ('baseline = diagnostic(model, tokenizer, cfg, "cpu", torch, base)',
     'baseline = v033_diagnostic(model, tokenizer, data, cfg, "cpu", torch, base)', 1),
    ("diag = diagnostic(model, tokenizer, cfg, device, torch, base)",
     "diag = v033_diagnostic(model, tokenizer, data, cfg, device, torch, base)", 1),
    ("final = diagnostic(model, tokenizer, cfg, device, torch, base)",
     "final = v033_diagnostic(model, tokenizer, data, cfg, device, torch, base)", 1),
    ("current = score(diag, val_loss)", "current = v033_score(diag, val_loss)", 1),

    # The free preflight now answers whether a paid session is worth launching.
    ('"baseline": baseline["metrics"],\n            "hf": {',
     '"baseline": baseline["metrics"],\n'
     '            "baseline_distribution": baseline["distribution"],\n'
     '            "baseline_by_kind": baseline["expanded_by_kind"],\n'
     '            "baseline_by_length": baseline["expanded_by_length"],\n'
     '            "format_parity": parity,\n'
     '            "envelope": {\n'
     '                "train_encoding": train_encoding,\n'
     '                "validation_encoding": val_encoding,\n'
     '                "tools": sorted(set(ENVELOPE_TOOLS.values())),\n'
     '                "example_prompt": to_envelope_row(train_rows[0])["prompt"],\n'
     '                "example_completion": to_envelope_row(train_rows[0])["completion"],\n'
     '                "baseline_slot_exact_rate": baseline["metrics"]["envelope_slot_exact_rate"],\n'
     '                "baseline_slot_margin_health": baseline["metrics"]["envelope_slot_margin_health"],\n'
     '            },\n'
     '            "hf": {', 1),

    ('if args.preflight_only:\n'
     '            print("EMBER_HF_V016_PREFLIGHT=PASS", flush=True)\n'
     '            return',
     'if args.preflight_only:\n'
     '            probe_generator = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))\n'
     '            probe = sample_multi_position_indices(train, 2, cfg, probe_generator, model, "cpu", torch)\n'
     '            if len(probe) != 2:\n'
     '                raise RuntimeError("v0.0.33 boundary mining preflight returned wrong batch size")\n'
     '            reach = baseline["metrics"]["expanded_sequences_within_reach"]\n'
     '            print("EMBER_V033_FORMAT_PARITY=PASS", flush=True)\n'
     '            print("EMBER_V033_BOUNDARY_MINING=PASS", flush=True)\n'
     '            print(f"EMBER_V033_SEQUENCES_WITHIN_REACH={reach:.4f}", flush=True)\n'
     '            print(json.dumps(baseline["distribution"], indent=2, sort_keys=True), flush=True)\n'
     '            print("EMBER_V033_GATE_DISTANCE", flush=True)\n'
     '            print(json.dumps(v033_gate_distance(baseline, cfg), indent=2, sort_keys=True), flush=True)\n'
     "            print('EMBER_V033_BASELINE_SLOT_EXACT=%.4f' % baseline['metrics']['envelope_slot_exact_rate'], flush=True)\n"
     "            print('EMBER_V033_BASELINE_SLOT_MARGIN_HEALTH=%.4f' % baseline['metrics']['envelope_slot_margin_health'], flush=True)\n"
     '            if reach <= 0.0:\n'
     '                print("EMBER_V033_ADVICE=NO_REACHABLE_SEQUENCES_DO_NOT_LAUNCH_T4", flush=True)\n'
     '            preflight_path = work / "v0.0.33-preflight.json"\n'
     '            preflight_path.write_text(json.dumps(preflight, indent=2) + "\\n")\n'
     '            base.upload(api, repo, preflight_path, "preflight/v0.0.33-preflight-latest.json", "Ember v0.0.33 CPU preflight report")\n'
     '            print("EMBER_V033_PREFLIGHT_REPORT=preflight/v0.0.33-preflight-latest.json", flush=True)\n'
     '            print("EMBER_HF_V033_PREFLIGHT=PASS", flush=True)\n'
     '            return', 1),

    ("best_score = None\n        best_step = -1\n        history = []",
     "best_score = v033_score(baseline, -1e9)\n"
     "        best_step = -1\n"
     "        history = []\n"
     "        save_checkpoint(best_path, model=model, optimizer=optimizer, tokenizer=tokenizer, "
     "model_config=model.cfg, train_config=cfg, step=-1, best_val_loss=float(\"inf\"), run_id=run_id)", 1),

    ('ids = torch.randint(len(train), (int(cfg["batch_size"]),), generator=generator).tolist()',
     'ids = sample_multi_position_indices(train, int(cfg["batch_size"]), cfg, generator, model, device, torch)', 1),
    ("loss = base.weighted_loss(model, x, y, w, torch) / accum",
     "loss = multi_position_loss(model, x, y, w, cfg, torch) / accum", 1),

    ('"promotion": "PASS" if final["passed"] else "FAIL",',
     '"format_parity": parity,\n'
     '            "progress": v033_progress(baseline, final, cfg),\n'
     '            "baseline_distribution": baseline["distribution"],\n'
     '            "final_distribution": final["distribution"],\n'
     '            "final_by_kind": final["expanded_by_kind"],\n'
     '            "final_by_length": final["expanded_by_length"],\n'
     '            "promotion": "PASS" if final["passed"] else "FAIL",', 1),

    ('print(f"EMBER_HF_V016_PROMOTION={report[\'promotion\']}", flush=True)',
     'print(f"EMBER_HF_V033_PROMOTION={report[\'promotion\']}", flush=True)\n'
     '        print(f"EMBER_V033_PROGRESS={report[\'progress\'][\'verdict\']}", flush=True)\n'
     '        print(json.dumps(report["progress"]["deltas"], indent=2, sort_keys=True), flush=True)\n'
     '        print("EMBER_V033_GATE_DISTANCE", flush=True)\n'
     '        print(json.dumps(report["progress"].get("gate_distance", {}), indent=2, sort_keys=True), flush=True)\n'
     '        print("EMBER_V033_BAND_FLOW", flush=True)\n'
     '        print(json.dumps(report["progress"].get("band_flow", {}), indent=2, sort_keys=True), flush=True)\n'
     '        print(json.dumps(report.get("final_distribution", {}), indent=2, sort_keys=True), flush=True)', 1),

    ("EMBER_HF_V016_", "EMBER_HF_V033_", -1),
    ("EMBER_V016_", "EMBER_V033_", -1),
)

OPTIONAL_TARGETS = frozenset({"0.0.16", "EMBER_HF_V016_", "EMBER_V016_"})


def apply_transforms(source: str) -> str:
    for old, new, count in TRANSFORMS:
        if old not in source and old not in OPTIONAL_TARGETS:
            raise RuntimeError(f"v0.0.33 transform target missing: {old!r}")
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
    print("EMBER_V033_BOUNDARY_FOCUS_TRANSFORM=PASS", flush=True)
    exec(compile(source, "ember_hf_sft_v033_wrapper_runtime.py", "exec"), {"__name__": "__main__"})


if __name__ == "__main__":
    main()
