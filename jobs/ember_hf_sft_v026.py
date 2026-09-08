# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["huggingface-hub>=1.4", "sentencepiece>=0.2", "torch>=2.4"]
# ///
"""Ember v0.0.26: format-parity copy repair from the protected v0.0.20 baseline.

v0.0.21 through v0.0.25 added curriculum volume, hard mining, margin loss and
position-conditioned coverage while the underlying generator could not emit the
character-level template of four of the five failing held-out values, and while
the learning rate was annealed to a point where 420 AdamW steps could not move
the model. See ``reports/ember-v0.0.25-plateau-analysis.json``.

This phase changes the three things those runs could not:

* the curriculum spans every held-out template, asserted before training starts;
* the learning rate is restored to 1.2e-6, between the last run that moved the
  model (v0.0.16 at 2.5e-6) and the frozen regime; and
* the held-out battery grows from 9 cases to 90, so a real change is
  distinguishable from a single lucky string.

Batch composition is deliberately uniform again. Neighbour and hard-kind mining
were introduced to compensate for the missing structure; with parity restored, a
run with fewer interacting mechanisms is interpretable whichever way it lands.
The v0.0.20 margin loss is kept, because it targets exactly the quantity that
decides a copy: the gap between the correct token and the runner-up.

The v0.0.20 weights remain the protected baseline. As in v0.0.25, ``best.pt`` is
seeded with them, so a checkpoint that fails to improve cannot replace v0.0.20.
"""
from __future__ import annotations

import urllib.request

# The proven v0.0.16 scaffold, transformed at run time.
BASE_COMMIT = "8ccd8cf3cd6de9c1b92e9f9a8897772f67d14c15"
# The commit carrying config/ember_format_parity_v0.0.26.json and
# jobs/ember_sft_data_v026.py.
ASSET_COMMIT = "36a213d9fecaf9ab8348132d6036e9c99a17e18d"
BASE_URL = (
    "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/"
    f"{BASE_COMMIT}/jobs/ember_hf_sft_v016.py"
)

HELPERS = r'''
def _v026_field_signature(value):
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

    This is the check the v0.0.21..v0.0.25 runs never made. Each of those runs
    spent a paid T4 session on a curriculum whose model_id generator could only
    produce vendor/ember-xxxx-NNb while the held-out value was
    openai/gpt-6-astra.
    """
    supported = {}
    for row in train_rows:
        supported.setdefault(str(row["kind"]), {}).setdefault(
            _v026_field_signature(str(row["value"])), 0
        )
        supported[str(row["kind"])][_v026_field_signature(str(row["value"]))] += 1
    missing = []
    per_case = []
    for kind, value, _ in data.DIAGNOSTICS:
        signature = _v026_field_signature(value)
        rows = int(supported.get(str(kind), {}).get(signature, 0))
        per_case.append({"kind": kind, "value": value, "template": signature, "train_rows": rows})
        if rows == 0:
            missing.append({"kind": kind, "value": value, "template": signature})
    if missing:
        raise RuntimeError(
            "v0.0.26 format parity failed; the curriculum cannot emit these held-out "
            "templates: %s" % missing
        )
    return {
        "held_out_cases": len(per_case),
        "templates_supported": len(per_case),
        "min_train_rows_for_a_held_out_template": min(c["train_rows"] for c in per_case),
        "distinct_train_templates": {k: len(v) for k, v in sorted(supported.items())},
    }


def v026_battery(model, tokenizer, cases, prompt_fn, device, torch, base):
    """Score one held-out battery. Case tuples are (kind, value, corrupted)."""
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
        c, n = continuation_top1_counts(model, tokenizer, prompt, value, device, torch, base)
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
            "expected_nll": expected_nll,
            "corrupt_nll": corrupt_nll,
            "expected_wins": expected_nll < corrupt_nll,
        })
    model.train()
    n = len(rows)
    metrics = {
        "exact_copy_rate": sum(r["exact"] for r in rows) / n,
        "continuation_top1_rate": (correct_tokens / total_tokens) if total_tokens else 0.0,
        "clean_stop_rate": sum(r["clean_stop"] for r in rows) / n,
        "first_token_top5_rate": sum(r["first_token"]["top5"] for r in rows) / n,
        "first_token_top20_rate": sum(r["first_token"]["top20"] for r in rows) / n,
        "teacher_forced_expected_win_rate": sum(r["expected_wins"] for r in rows) / n,
        "mean_first_token_rank": sum(r["first_token"]["rank"] for r in rows) / n,
        "continuation_tokens": total_tokens,
    }
    by_kind = {}
    for row in rows:
        bucket = by_kind.setdefault(row["kind"], {"cases": 0, "exact": 0})
        bucket["cases"] += 1
        bucket["exact"] += int(row["exact"])
    return {"metrics": metrics, "cases": rows, "by_kind": by_kind}


def v026_diagnostic(model, tokenizer, data, cfg, device, torch, base):
    """Legacy nine (protection) plus the ninety-case battery (signal).

    The legacy battery keeps the v0.0.15 prompt and values verbatim, so its
    numbers stay directly comparable to the protected v0.0.20 result of
    exact copy 0.4444 and continuation 0.8406.
    """
    legacy = diagnostic(model, tokenizer, cfg, device, torch, base)
    expanded = v026_battery(model, tokenizer, data.DIAGNOSTICS, data.prompt_for, device, torch, base)

    metrics = dict(legacy["metrics"])
    metrics["copy_position_top1_rate"] = metrics["continuation_top1_rate"]
    for key, value in expanded["metrics"].items():
        metrics["expanded_" + key] = value

    gates = dict(legacy["gates"])
    gates["copy_position_top1"] = (
        metrics["copy_position_top1_rate"] >= float(cfg["minimum_copy_position_top1_rate"])
    )
    gates["legacy_not_regressed"] = (
        metrics["exact_copy_rate"] >= float(cfg["protected_legacy_exact_copy_rate"]) - 1e-9
        and metrics["continuation_top1_rate"]
        >= float(cfg["protected_legacy_continuation_top1_rate"]) - 1e-9
    )
    return {
        "passed": all(gates.values()),
        "metrics": metrics,
        "gates": gates,
        "legacy_cases": legacy["cases"],
        "expanded_by_kind": expanded["by_kind"],
        "expanded_cases": expanded["cases"],
    }


def v026_score(diag, val_loss):
    """Selection order: never regress the protected v0.0.20 metric, then improve
    the ninety-case battery, then the legacy nine, then validation loss.

    The ninety-case battery leads because the nine-case one moves in steps of
    11.1 points and cannot separate a real gain from one lucky string.
    """
    m = diag["metrics"]
    return (
        1.0 if diag["gates"].get("legacy_not_regressed") else 0.0,
        m["expanded_exact_copy_rate"],
        m["expanded_continuation_top1_rate"],
        m["exact_copy_rate"],
        m["continuation_top1_rate"],
        m["clean_stop_rate"],
        m["expanded_teacher_forced_expected_win_rate"],
        -val_loss,
    )


def v026_expanded_gain(baseline, final, cfg):
    b = baseline["metrics"]
    f = final["metrics"]
    exact = f["expanded_exact_copy_rate"] - b["expanded_exact_copy_rate"]
    continuation = f["expanded_continuation_top1_rate"] - b["expanded_continuation_top1_rate"]
    return {
        "baseline_expanded_exact_copy_rate": b["expanded_exact_copy_rate"],
        "final_expanded_exact_copy_rate": f["expanded_exact_copy_rate"],
        "expanded_exact_copy_gain": exact,
        "baseline_expanded_continuation_top1_rate": b["expanded_continuation_top1_rate"],
        "final_expanded_continuation_top1_rate": f["expanded_continuation_top1_rate"],
        "expanded_continuation_gain": continuation,
        "exact_copy_gain_met": exact >= float(cfg["minimum_expanded_exact_copy_gain"]) - 1e-9,
        "continuation_gain_met": continuation
        >= float(cfg["minimum_expanded_continuation_gain"]) - 1e-9,
    }


def margin_weighted_loss(model, x, y, weights, cfg, torch):
    """v0.0.20's loss: weighted cross entropy plus a hinge on the gap between the
    correct copy token and the best competing token."""
    import torch.nn.functional as F
    logits, _ = model(x, None)
    raw = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction="none"
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
    margin_term = F.relu(float(cfg["margin"]) - (correct - best_wrong))
    hard = (~best_is_correct).to(margin_term.dtype)
    multiplier = 1.0 + hard * (float(cfg["hard_error_multiplier"]) - 1.0)
    return ce + float(cfg["margin_loss_weight"]) * (margin_term * multiplier).mean()

'''

# (old, new, count). Applied in order; a target that no longer matches the
# pinned scaffold is a hard error, and tests/test_ember_v026_format_parity.py
# checks every one of them against the scaffold in CPU CI.
TRANSFORMS = (
    # Repoint the scaffold at the v0.0.26 assets and the v0.0.20 source model.
    ('CONFIG_PIN = "42ed91ac6e6b8ff22873fec6706c4f243f57f2fa"',
     f'CONFIG_PIN = "{ASSET_COMMIT}"', 1),
    ('DATA_PIN = "14995a94a4d1594463e266c4c5fed0ecec329da9"',
     f'DATA_PIN = "{ASSET_COMMIT}"', 1),
    ("0.0.16", "0.0.26", -1),
    ("ember_sequence_copy_v0.0.26.json", "ember_format_parity_v0.0.26.json", -1),
    ("ember_sft_data_v015.py", "ember_sft_data_v026.py", -1),
    ("sequence-copy-consolidation", "format-parity-copy-repair", -1),
    ('SOURCE_REPO = "Jmiller18899/ember-v0.0.15-t4"',
     'SOURCE_REPO = "Jmiller18899/ember-v0.0.20-t4"', -1),
    ('cfg.get("source_model_name") != "ember-v0.0.15-t4"',
     'cfg.get("source_model_name") != "ember-v0.0.20-t4"', -1),
    ("must start from v0.0.15", "must start from v0.0.20", -1),
    ('if str(source_cfg.get("version")) != "0.0.15":',
     'if str(source_cfg.get("version")) != "0.0.20":', -1),
    ("expected a v0.0.15 checkpoint", "expected a v0.0.20 checkpoint", -1),
    ("v0.0.15 checkpoint run_id", "v0.0.20 checkpoint run_id", -1),
    ("v0.0.15 source state", "v0.0.20 source state", -1),
    ("resolve_v015_source", "resolve_v020_source", -1),
    ("v015_promotion", "v020_promotion", -1),
    ("v015_best_step", "v020_best_step", -1),

    # Install the v0.0.26 helpers ahead of the scaffold's scoring function.
    ("def score(diag: dict, val_loss: float):", HELPERS + "def score(diag: dict, val_loss: float):", 1),

    # Refuse to train on a curriculum that cannot emit a held-out template.
    ("        data.assert_clean(train_rows, val_rows)\n"
     "        owner, repo = verify_hf_output_access(api, cfg, work)",
     "        data.assert_clean(train_rows, val_rows)\n"
     "        parity = assert_format_parity(data, train_rows)\n"
     "        owner, repo = verify_hf_output_access(api, cfg, work)", 1),

    # Both batteries, everywhere the scaffold takes a measurement.
    ('baseline = diagnostic(model, tokenizer, cfg, "cpu", torch, base)',
     'baseline = v026_diagnostic(model, tokenizer, data, cfg, "cpu", torch, base)', 1),
    ("diag = diagnostic(model, tokenizer, cfg, device, torch, base)",
     "diag = v026_diagnostic(model, tokenizer, data, cfg, device, torch, base)", 1),
    ("final = diagnostic(model, tokenizer, cfg, device, torch, base)",
     "final = v026_diagnostic(model, tokenizer, data, cfg, device, torch, base)", 1),
    ("current = score(diag, val_loss)", "current = v026_score(diag, val_loss)", 1),

    # Report the parity result and the size of each battery in the preflight.
    ('"baseline": baseline["metrics"],\n            "hf": {',
     '"baseline": baseline["metrics"],\n'
     '            "format_parity": parity,\n'
     '            "batteries": {\n'
     '                "legacy_cases": len(base.DIAGNOSTICS),\n'
     '                "expanded_cases": len(data.DIAGNOSTICS),\n'
     '                "held_out_leakage": False,\n'
     '            },\n'
     '            "hf": {', 1),

    ('if args.preflight_only:\n'
     '            print("EMBER_HF_V016_PREFLIGHT=PASS", flush=True)\n'
     '            return',
     'if args.preflight_only:\n'
     '            print("EMBER_V026_FORMAT_PARITY=PASS", flush=True)\n'
     '            print("EMBER_HF_V026_PREFLIGHT=PASS", flush=True)\n'
     '            return', 1),

    # Seed best.pt with the protected v0.0.20 weights. A checkpoint that only
    # ties the baseline can never replace it, because the seeded score carries a
    # +1e9 validation-loss tiebreaker.
    ("best_score = None\n        best_step = -1\n        history = []",
     "best_score = v026_score(baseline, -1e9)\n"
     "        best_step = -1\n"
     "        history = []\n"
     "        save_checkpoint(best_path, model=model, optimizer=optimizer, tokenizer=tokenizer, "
     "model_config=model.cfg, train_config=cfg, step=-1, best_val_loss=float(\"inf\"), run_id=run_id)", 1),

    ("loss = base.weighted_loss(model, x, y, w, torch) / accum",
     "loss = margin_weighted_loss(model, x, y, w, cfg, torch) / accum", 1),

    # Promotion needs a measured gain on the ninety-case battery, not just the
    # legacy absolute gates.
    ('"promotion": "PASS" if final["passed"] else "FAIL",',
     '"format_parity": parity,\n'
     '            "expanded_gain": v026_expanded_gain(baseline, final, cfg),\n'
     '            "promotion": "PASS" if (\n'
     '                final["passed"]\n'
     '                and v026_expanded_gain(baseline, final, cfg)["exact_copy_gain_met"]\n'
     '            ) else "FAIL",', 1),

    ("EMBER_HF_V016_", "EMBER_HF_V026_", -1),
    ("EMBER_V016_", "EMBER_V026_", -1),
)

# Targets that are plain renames rather than structural anchors: they may
# legitimately be absent once an earlier transform has already rewritten them.
OPTIONAL_TARGETS = frozenset({"0.0.16", "EMBER_HF_V016_", "EMBER_V016_"})


def apply_transforms(source: str) -> str:
    """Apply every transform in order, raising on a target that no longer matches."""
    for old, new, count in TRANSFORMS:
        if old not in source and old not in OPTIONAL_TARGETS:
            raise RuntimeError(f"v0.0.26 transform target missing: {old!r}")
        source = source.replace(old, new) if count < 0 else source.replace(old, new, count)
    return source


def unmatched_transform_targets(source: str) -> list[str]:
    """Return the transform targets that do not match the given scaffold.

    Used by the CPU test suite so a scaffold drift fails in CI rather than
    inside a paid Hugging Face job.
    """
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
    print("EMBER_V026_FORMAT_PARITY_TRANSFORM=PASS", flush=True)
    exec(compile(source, "ember_hf_sft_v026_wrapper_runtime.py", "exec"), {"__name__": "__main__"})


if __name__ == "__main__":
    main()
