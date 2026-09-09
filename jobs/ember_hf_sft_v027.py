# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["huggingface-hub>=1.4", "sentencepiece>=0.2", "torch>=2.4"]
# ///
"""Ember v0.0.27: sequence-completion repair from the v0.0.26 best checkpoint.

v0.0.26 broke the plateau (see ``reports/ember-v0.0.26-result.json``) and left a
specific, different problem behind. Copy errors are now approximately
independent across token positions: at 0.8151 per-token accuracy over a mean
7.67-token continuation, independent errors predict 0.2086 exact copy against a
measured 0.2222. There is no single broken structure left; exact copy is being
crushed multiplicatively by sequence length.

Two changes follow from that, and nothing else moves. The curriculum, the
learning rate, the batch size and the weighting scheme are all exactly v0.0.26's,
so this run is a clean test of the objective.

1. **A worst-position hinge.** The margin term inherited from v0.0.20 averages
   over every copy token in the batch, so a sequence with one bad position out of
   eight gets an eighth of the pressure that position needs. v0.0.27 adds a term
   on each row's *weakest* copy token. Under independent errors this is where
   almost all the remaining value is.
2. **A continuous selection statistic.** A 90-case exact-copy count has a
   standard deviation near four cases, and v0.0.26's selector appears to have
   used that noise to pick the worse of two checkpoints. Teacher-forced
   completion is not a higher-resolution alternative -- it is the same event as
   greedy exact copy -- so selection leads on ``sequence_margin_health``, the
   mean clamped worst-position margin across the held-out battery, with exact
   copy as a tiebreak.

Hard mining returns, now scored by whole-sequence failure. It was removed for
v0.0.26 because it oversampled rows from templates that could not contain the
answer; with format parity established that objection no longer applies, and
sequence-level hardness is aimed at exactly the quantity this phase targets.

The v0.0.26 weights are the protected baseline: ``best.pt`` is seeded with them,
so a checkpoint that fails to improve cannot replace them.
"""
from __future__ import annotations

import urllib.request

# The proven v0.0.16 scaffold, transformed at run time.
BASE_COMMIT = "8ccd8cf3cd6de9c1b92e9f9a8897772f67d14c15"
# The v0.0.26 curriculum, carried into this phase unchanged.
DATA_COMMIT = "36a213d9fecaf9ab8348132d6036e9c99a17e18d"
# The commit carrying config/ember_sequence_completion_v0.0.27.json.
CONFIG_COMMIT = "ce5d69828e0b4833a2d4bf125b49848931fff812"
BASE_URL = (
    "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/"
    f"{BASE_COMMIT}/jobs/ember_hf_sft_v016.py"
)

HELPERS = r'''
def _v027_field_signature(value):
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
    """Refuse to train on a curriculum that cannot emit a held-out template.

    The curriculum is unchanged from v0.0.26, but this stays because it is the
    check whose absence cost v0.0.21 through v0.0.25 five paid sessions.
    """
    supported = {}
    for row in train_rows:
        signature = _v027_field_signature(str(row["value"]))
        bucket = supported.setdefault(str(row["kind"]), {})
        bucket[signature] = bucket.get(signature, 0) + 1
    missing = []
    per_case = []
    for kind, value, _ in data.DIAGNOSTICS:
        signature = _v027_field_signature(value)
        rows = int(supported.get(str(kind), {}).get(signature, 0))
        per_case.append({"kind": kind, "value": value, "template": signature, "train_rows": rows})
        if rows == 0:
            missing.append({"kind": kind, "value": value, "template": signature})
    if missing:
        raise RuntimeError(
            "v0.0.27 format parity failed; the curriculum cannot emit these held-out "
            "templates: %s" % missing
        )
    return {
        "held_out_cases": len(per_case),
        "templates_supported": len(per_case),
        "min_train_rows_for_a_held_out_template": min(c["train_rows"] for c in per_case),
        "distinct_train_templates": {k: len(v) for k, v in sorted(supported.items())},
    }


def v027_sequence_stats(model, tokenizer, prompt, expected, device, torch, base):
    """Teacher-forced top-1 count plus the sequence's worst-position margin.

    The margin is the gap between the correct token's logit and the best
    competing token's, minimised over the sequence. Exact copy under greedy
    decoding happens exactly when that minimum is positive at every position, so
    this is a continuous surrogate for the metric rather than a proxy for it.
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
    positions = list(range(first_target + 1, eos_target))
    if not positions:
        return 0, 0, None
    x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    y = full[1:]
    with torch.inference_mode():
        logits, _ = model(x, None)
        rows = logits[0].float()
        pred = torch.argmax(rows, dim=-1)
    correct = sum(int(pred[pos].item()) == int(y[pos]) for pos in positions)
    min_gap = None
    for pos in positions:
        target = int(y[pos])
        vector = rows[pos]
        target_logit = float(vector[target].item())
        competitors = vector.clone()
        competitors[target] = float("-inf")
        gap = target_logit - float(competitors.max().item())
        min_gap = gap if min_gap is None else min(min_gap, gap)
    return correct, len(positions), min_gap


def _v027_bucket(tokens, edges):
    for edge in edges:
        if tokens <= edge:
            return "<=%d" % edge
    return ">%d" % edges[-1]


def v027_battery(model, tokenizer, cases, prompt_fn, cfg, device, torch, base):
    """Score one held-out battery. Case tuples are (kind, value, corrupted)."""
    model.eval()
    rows = []
    correct_tokens = 0
    total_tokens = 0
    floor = float(cfg["sequence_margin_floor"])
    ceiling = float(cfg["sequence_margin"])
    edges = [int(e) for e in cfg["length_buckets"]]
    for kind, value, corrupt in cases:
        prompt = prompt_fn(value)
        text, stopped = base.greedy(model, tokenizer, prompt, device, torch)
        visible = text.split(base.EOT, 1)[0].strip()
        first = base.first_token_stats(model, tokenizer, prompt, value, device, torch)
        expected_nll = base.completion_nll(model, tokenizer, prompt, value, device, torch)
        corrupt_nll = base.completion_nll(model, tokenizer, prompt, corrupt, device, torch)
        c, n, min_gap = v027_sequence_stats(model, tokenizer, prompt, value, device, torch, base)
        correct_tokens += c
        total_tokens += n
        rows.append({
            "kind": kind,
            "value": value,
            "generated": visible,
            "exact": visible == value,
            "clean_stop": stopped,
            "first_token": first,
            "continuation_correct": c,
            "continuation_tokens": n,
            "min_margin": min_gap,
            "expected_nll": expected_nll,
            "corrupt_nll": corrupt_nll,
            "expected_wins": expected_nll < corrupt_nll,
        })
    model.train()
    total = len(rows)
    gaps = [r["min_margin"] for r in rows if r["min_margin"] is not None]
    health = sum(min(max(g, floor), ceiling) for g in gaps) / len(gaps) if gaps else floor
    metrics = {
        "exact_copy_rate": sum(r["exact"] for r in rows) / total,
        "continuation_top1_rate": (correct_tokens / total_tokens) if total_tokens else 0.0,
        "clean_stop_rate": sum(r["clean_stop"] for r in rows) / total,
        "first_token_top5_rate": sum(r["first_token"]["top5"] for r in rows) / total,
        "first_token_top20_rate": sum(r["first_token"]["top20"] for r in rows) / total,
        "teacher_forced_expected_win_rate": sum(r["expected_wins"] for r in rows) / total,
        "mean_first_token_rank": sum(r["first_token"]["rank"] for r in rows) / total,
        "sequence_margin_health": health,
        "sequences_all_positions_positive": sum(
            1 for r in rows if r["min_margin"] is not None and r["min_margin"] > 0
        ) / total,
        "continuation_tokens": total_tokens,
        "mean_continuation_tokens": (total_tokens / total) if total else 0.0,
    }
    by_kind = {}
    by_length = {}
    for row in rows:
        for label, table in (
            (row["kind"], by_kind),
            (_v027_bucket(row["continuation_tokens"], edges), by_length),
        ):
            bucket = table.setdefault(label, {"cases": 0, "exact": 0})
            bucket["cases"] += 1
            bucket["exact"] += int(row["exact"])
    return {"metrics": metrics, "cases": rows, "by_kind": by_kind, "by_length": by_length}


def v027_diagnostic(model, tokenizer, data, cfg, device, torch, base):
    """Legacy nine (comparability) plus the ninety-case battery (signal)."""
    legacy = diagnostic(model, tokenizer, cfg, device, torch, base)
    expanded = v027_battery(
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

    # Protection is separate from promotion. It asks only whether this
    # checkpoint may replace the v0.0.26 weights in best.pt. The high-resolution
    # metrics carry no slack; the low-resolution counts carry one case, because a
    # 90-case rate near 0.22 has a standard deviation of about four cases.
    protection = {
        "legacy_exact_copy": metrics["exact_copy_rate"]
        >= float(cfg["protected_legacy_exact_copy_rate"]) - 1e-9,
        "legacy_continuation": metrics["continuation_top1_rate"]
        >= float(cfg["protected_legacy_continuation_top1_rate"]) - 1e-9,
        "expanded_exact_copy": metrics["expanded_exact_copy_rate"]
        >= float(cfg["protected_expanded_exact_copy_rate"]) - 1e-9,
        "expanded_continuation": metrics["expanded_continuation_top1_rate"]
        >= float(cfg["protected_expanded_continuation_top1_rate"]) - 1e-9,
    }
    gates["not_regressed"] = all(protection.values())
    return {
        "passed": all(gates.values()),
        "metrics": metrics,
        "gates": gates,
        "protection": protection,
        "legacy_cases": legacy["cases"],
        "expanded_by_kind": expanded["by_kind"],
        "expanded_by_length": expanded["by_length"],
        "expanded_cases": expanded["cases"],
    }


def v027_score(diag, val_loss):
    """Selection order: never regress, then the continuous margin statistic.

    v0.0.26 led on the 90-case exact-copy count and appears to have taken a
    one-case fluctuation over a genuinely better checkpoint. sequence_margin_health
    measures the same thing -- how far each held-out sequence is from having every
    position correct -- without quantising to 1/90.
    """
    m = diag["metrics"]
    return (
        1.0 if diag["gates"].get("not_regressed") else 0.0,
        m["expanded_sequence_margin_health"],
        m["expanded_exact_copy_rate"],
        m["expanded_continuation_top1_rate"],
        m["exact_copy_rate"],
        m["continuation_top1_rate"],
        -val_loss,
    )


def v027_progress(baseline, final, cfg):
    """Did this run advance the model? Reported separately from promotion.

    v0.0.26 improved four metrics and reported a bare FAIL, which made a real
    advance look like the flat runs that preceded it.
    """
    b = baseline["metrics"]
    f = final["metrics"]
    deltas = {}
    for key in (
        "expanded_exact_copy_rate",
        "expanded_continuation_top1_rate",
        "expanded_sequence_margin_health",
        "exact_copy_rate",
        "continuation_top1_rate",
    ):
        deltas[key] = {"baseline": b[key], "final": f[key], "gain": f[key] - b[key]}
    exact_gain = deltas["expanded_exact_copy_rate"]["gain"]
    continuation_gain = deltas["expanded_continuation_top1_rate"]["gain"]
    advanced = (
        exact_gain >= float(cfg["progress_expanded_exact_copy_gain"]) - 1e-9
        or continuation_gain >= float(cfg["progress_expanded_continuation_gain"]) - 1e-9
    )
    regressed = exact_gain < -1e-9 and continuation_gain < -1e-9
    return {
        "deltas": deltas,
        "verdict": "REGRESSED" if regressed else ("ADVANCED" if advanced else "FLAT"),
    }


def _v027_copy_gaps(logits, y, weights, cfg, torch):
    """Per-copy-token margin between the correct token and its best competitor."""
    active = y.ne(-100)
    target_mask = active & weights.ge(float(cfg["copy_token_weight"]) - 1e-6)
    if not bool(target_mask.any().item()):
        return None, None, None, active
    copy_logits = logits[target_mask].float()
    copy_targets = y[target_mask]
    correct = copy_logits.gather(1, copy_targets.unsqueeze(1)).squeeze(1)
    top2_values, top2_indices = torch.topk(copy_logits, k=2, dim=1)
    best_is_correct = top2_indices[:, 0].eq(copy_targets)
    best_wrong = torch.where(best_is_correct, top2_values[:, 1], top2_values[:, 0])
    return correct - best_wrong, best_is_correct, target_mask, active


def sequence_completion_loss(model, x, y, weights, cfg, torch):
    """v0.0.26's loss plus a hinge on each row's weakest copy token.

    The per-token term keeps every position honest. The sequence term is the new
    part: it puts the whole row's gradient on its worst position, which is the
    only position that decides whether the row produces an exact copy.
    """
    import torch.nn.functional as F
    logits, _ = model(x, None)
    raw = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction="none"
    ).reshape_as(y)
    active = y.ne(-100)
    denom = (weights * active).sum().clamp_min(1.0)
    ce = (raw * weights * active).sum() / denom

    gap, best_is_correct, target_mask, _ = _v027_copy_gaps(logits, y, weights, cfg, torch)
    if gap is None:
        return ce

    token_hinge = F.relu(float(cfg["margin"]) - gap)
    hard = (~best_is_correct).to(token_hinge.dtype)
    multiplier = 1.0 + hard * (float(cfg["hard_error_multiplier"]) - 1.0)
    token_term = (token_hinge * multiplier).mean()

    sequence_hinge = torch.zeros_like(y, dtype=gap.dtype)
    sequence_hinge[target_mask] = F.relu(float(cfg["sequence_margin"]) - gap)
    worst = sequence_hinge.max(dim=1).values
    rows = target_mask.any(dim=1).to(worst.dtype)
    sequence_term = (worst * rows).sum() / rows.sum().clamp_min(1.0)

    return (
        ce
        + float(cfg["margin_loss_weight"]) * token_term
        + float(cfg["sequence_margin_loss_weight"]) * sequence_term
    )


def sample_sequence_hard_indices(train, batch_size, cfg, generator, model, device, torch):
    """Half the batch uniform, half the rows whose worst position is worst.

    Hard mining was removed for v0.0.26 because the pool it drew from could not
    contain the answer. With format parity established, and with hardness scored
    per sequence rather than per token, it now targets exactly the rows that fail
    to produce an exact copy.
    """
    import torch.nn.functional as F
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
        gap, _, target_mask, _ = _v027_copy_gaps(logits, y, w, cfg, torch)
        if gap is None:
            return uniform
        pred = torch.argmax(logits, dim=-1)
        row_failed = (pred.ne(y) & target_mask).any(dim=1).to(gap.dtype)
        sequence_hinge = torch.zeros_like(y, dtype=gap.dtype)
        sequence_hinge[target_mask] = F.relu(float(cfg["sequence_margin"]) - gap)
        hardness = sequence_hinge.max(dim=1).values + row_failed * float(
            cfg["hard_error_multiplier"]
        )
        chosen = torch.topk(hardness, k=min(hard_count, len(candidates))).indices.tolist()
    return [candidates[j] for j in chosen] + uniform[: int(batch_size) - hard_count]

'''

# (old, new, count). Applied in order against the pinned v0.0.16 scaffold; every
# target is checked in CPU CI by tests/test_ember_v027_sequence_completion.py.
TRANSFORMS = (
    ('CONFIG_PIN = "42ed91ac6e6b8ff22873fec6706c4f243f57f2fa"',
     f'CONFIG_PIN = "{CONFIG_COMMIT}"', 1),
    ('DATA_PIN = "14995a94a4d1594463e266c4c5fed0ecec329da9"',
     f'DATA_PIN = "{DATA_COMMIT}"', 1),
    ("0.0.16", "0.0.27", -1),
    ("ember_sequence_copy_v0.0.27.json", "ember_sequence_completion_v0.0.27.json", -1),
    ("ember_sft_data_v015.py", "ember_sft_data_v026.py", -1),
    ("sequence-copy-consolidation", "sequence-completion-repair", -1),
    ('SOURCE_REPO = "Jmiller18899/ember-v0.0.15-t4"',
     'SOURCE_REPO = "Jmiller18899/ember-v0.0.26-t4"', -1),
    ('cfg.get("source_model_name") != "ember-v0.0.15-t4"',
     'cfg.get("source_model_name") != "ember-v0.0.26-t4"', -1),
    ("must start from v0.0.15", "must start from v0.0.26", -1),
    ('if str(source_cfg.get("version")) != "0.0.15":',
     'if str(source_cfg.get("version")) != "0.0.26":', -1),
    ("expected a v0.0.15 checkpoint", "expected a v0.0.26 checkpoint", -1),
    ("v0.0.15 checkpoint run_id", "v0.0.26 checkpoint run_id", -1),
    ("v0.0.15 source state", "v0.0.26 source state", -1),
    ("resolve_v015_source", "resolve_v026_source", -1),
    ("v015_promotion", "v026_promotion", -1),
    ("v015_best_step", "v026_best_step", -1),

    ("def score(diag: dict, val_loss: float):", HELPERS + "def score(diag: dict, val_loss: float):", 1),

    ("        data.assert_clean(train_rows, val_rows)\n"
     "        owner, repo = verify_hf_output_access(api, cfg, work)",
     "        data.assert_clean(train_rows, val_rows)\n"
     "        parity = assert_format_parity(data, train_rows)\n"
     "        owner, repo = verify_hf_output_access(api, cfg, work)", 1),

    ('baseline = diagnostic(model, tokenizer, cfg, "cpu", torch, base)',
     'baseline = v027_diagnostic(model, tokenizer, data, cfg, "cpu", torch, base)', 1),
    ("diag = diagnostic(model, tokenizer, cfg, device, torch, base)",
     "diag = v027_diagnostic(model, tokenizer, data, cfg, device, torch, base)", 1),
    ("final = diagnostic(model, tokenizer, cfg, device, torch, base)",
     "final = v027_diagnostic(model, tokenizer, data, cfg, device, torch, base)", 1),
    ("current = score(diag, val_loss)", "current = v027_score(diag, val_loss)", 1),

    ('"baseline": baseline["metrics"],\n            "hf": {',
     '"baseline": baseline["metrics"],\n'
     '            "baseline_by_kind": baseline["expanded_by_kind"],\n'
     '            "baseline_by_length": baseline["expanded_by_length"],\n'
     '            "format_parity": parity,\n'
     '            "batteries": {\n'
     '                "legacy_cases": len(base.DIAGNOSTICS),\n'
     '                "expanded_cases": len(data.DIAGNOSTICS),\n'
     '            },\n'
     '            "hf": {', 1),

    ('if args.preflight_only:\n'
     '            print("EMBER_HF_V016_PREFLIGHT=PASS", flush=True)\n'
     '            return',
     'if args.preflight_only:\n'
     '            probe_generator = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))\n'
     '            probe = sample_sequence_hard_indices(train, 2, cfg, probe_generator, model, "cpu", torch)\n'
     '            if len(probe) != 2:\n'
     '                raise RuntimeError("v0.0.27 sequence hard-mining preflight returned wrong batch size")\n'
     '            print("EMBER_V027_FORMAT_PARITY=PASS", flush=True)\n'
     '            print("EMBER_V027_SEQUENCE_MINING=PASS", flush=True)\n'
     '            print("EMBER_HF_V027_PREFLIGHT=PASS", flush=True)\n'
     '            return', 1),

    # Seed best.pt with the protected v0.0.26 weights. The seeded score carries a
    # +1e9 validation-loss tiebreaker, so a tie cannot replace them.
    ("best_score = None\n        best_step = -1\n        history = []",
     "best_score = v027_score(baseline, -1e9)\n"
     "        best_step = -1\n"
     "        history = []\n"
     "        save_checkpoint(best_path, model=model, optimizer=optimizer, tokenizer=tokenizer, "
     "model_config=model.cfg, train_config=cfg, step=-1, best_val_loss=float(\"inf\"), run_id=run_id)", 1),

    ('ids = torch.randint(len(train), (int(cfg["batch_size"]),), generator=generator).tolist()',
     'ids = sample_sequence_hard_indices(train, int(cfg["batch_size"]), cfg, generator, model, device, torch)', 1),
    ("loss = base.weighted_loss(model, x, y, w, torch) / accum",
     "loss = sequence_completion_loss(model, x, y, w, cfg, torch) / accum", 1),

    ('"promotion": "PASS" if final["passed"] else "FAIL",',
     '"format_parity": parity,\n'
     '            "progress": v027_progress(baseline, final, cfg),\n'
     '            "final_by_kind": final["expanded_by_kind"],\n'
     '            "final_by_length": final["expanded_by_length"],\n'
     '            "promotion": "PASS" if final["passed"] else "FAIL",', 1),

    ('print(f"EMBER_HF_V016_PROMOTION={report[\'promotion\']}", flush=True)',
     'print(f"EMBER_HF_V027_PROMOTION={report[\'promotion\']}", flush=True)\n'
     '        print(f"EMBER_V027_PROGRESS={report[\'progress\'][\'verdict\']}", flush=True)\n'
     '        print(json.dumps(report["progress"]["deltas"], indent=2, sort_keys=True), flush=True)', 1),

    ("EMBER_HF_V016_", "EMBER_HF_V027_", -1),
    ("EMBER_V016_", "EMBER_V027_", -1),
)

# Plain renames that an earlier transform may already have rewritten.
OPTIONAL_TARGETS = frozenset({"0.0.16", "EMBER_HF_V016_", "EMBER_V016_"})


def apply_transforms(source: str) -> str:
    for old, new, count in TRANSFORMS:
        if old not in source and old not in OPTIONAL_TARGETS:
            raise RuntimeError(f"v0.0.27 transform target missing: {old!r}")
        source = source.replace(old, new) if count < 0 else source.replace(old, new, count)
    return source


def unmatched_transform_targets(source: str) -> list[str]:
    """Transform targets that do not match the given scaffold, for CPU CI."""
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
    print("EMBER_V027_SEQUENCE_COMPLETION_TRANSFORM=PASS", flush=True)
    exec(compile(source, "ember_hf_sft_v027_wrapper_runtime.py", "exec"), {"__name__": "__main__"})


if __name__ == "__main__":
    main()
