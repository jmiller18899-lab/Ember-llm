"""Weather-vs-time specialist for the expanded Ember v0.0.54 frozen router.

The expanded hierarchical router is retained unchanged:
  1) binary direct vs tool gate;
  2) four-way tool-family head.

A third tiny binary head is trained only on the existing 64 weather + 64
get_time training examples. It is consulted only when the four-way family head
predicts weather or get_time. This targets the sole remaining full-precision
development miss without changing Ember weights, direct/tool behavior, or the
calculator/web_search boundary.

All hyperparameters are selected by training-only cross-validation. The prior
20, 50, and 100 prompt suites are development evidence only. No new final
confirmation is consumed here. Precision-specific full and INT4 heads are
assessed separately; no shared cross-precision head is required.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

from huggingface_hub import HfApi
import torch

from jobs import ember_hf_eval as ev
from jobs import ember_v053_heldout_routing as held
from jobs import ember_v054_frozen_router_probe as probe
from jobs import ember_v054_router_confirmation as confirm
from jobs import ember_v054_router_third_confirmation as third
from jobs import ember_v054_hierarchical_router as hier
from jobs import ember_v054_expanded_router as expanded

OUT = Path("v054-weather-time-specialist")
REP = hier.REPRESENTATION
PAIR = ("weather", "get_time")


def fit_specialist(all_rows, family_cases):
    by_id = {r["id"]: r for r in all_rows}
    cases = [c for c in family_cases if probe.label_of(c) in PAIR]
    rows = [by_id[c["id"]] for c in cases]
    X, labels = hier.stack(rows, REP)
    y = torch.tensor([PAIR.index(label) for label in labels], dtype=torch.long)
    counts = [int((y == i).sum().item()) for i in range(2)]
    if counts != [64, 64]:
        raise RuntimeError(f"weather/time specialist must be 64/64, got {counts}")
    cv, records = hier.cv_ridge(X, y, 2)
    state = hier.fit_ridge(X, y, 2, cv["ridge"])
    return {"state": state, "cv": cv, "cv_records": records, "counts": counts}


def prepare_precision(model, tokenizer, all_train, binary_cases, family_cases):
    train_rows = hier.extract(model, tokenizer, all_train)
    heads = expanded.fit_expanded(train_rows, binary_cases, family_cases)
    specialist = fit_specialist(train_rows, family_cases)
    return heads, specialist


def specialist_predict(heads, specialist, X):
    binary_pred, _, binary_margin = hier.predict(heads["binary"], X)
    family_pred, _, family_margin = hier.predict(heads["family"], X)
    pair_pred, _, pair_margin = hier.predict(specialist["state"], X)
    labels, margins, used = [], [], []
    for i in range(X.shape[0]):
        if int(binary_pred[i]) == 0:
            labels.append("direct")
            margins.append(float(binary_margin[i]))
            used.append(False)
            continue
        family = hier.FAMILY_LABELS[int(family_pred[i])]
        if family in PAIR:
            family = PAIR[int(pair_pred[i])]
            margins.append(float(min(binary_margin[i], pair_margin[i])))
            used.append(True)
        else:
            margins.append(float(min(binary_margin[i], family_margin[i])))
            used.append(False)
        labels.append(family)
    return labels, margins, used


def evaluate_suite(model, tokenizer, heads, specialist, cases):
    rows = hier.extract(model, tokenizer, cases)
    X, _ = hier.stack(rows, REP)
    labels, margins, used = specialist_predict(heads, specialist, X)
    scored = hier.score(labels, rows, cases, margins)
    for detail, flag in zip(scored["rows"], used):
        detail["specialist_used"] = bool(flag)
    scored["specialist_invocations"] = sum(int(x) for x in used)
    return scored


def exact(m):
    return (
        m["five_way_correct"] == m["total"]
        and m["direct_vs_tool_correct"] == m["total"]
        and m["tool_family_correct"] == m["tool_family_total"]
    )


def compact(m):
    return {
        "five_way": f"{m['five_way_correct']}/{m['total']}",
        "direct_tool": f"{m['direct_vs_tool_correct']}/{m['total']}",
        "tool_family": f"{m['tool_family_correct']}/{m['tool_family_total']}",
        "specialist_invocations": m["specialist_invocations"],
        "min_correct_margin": m["min_correct_margin"],
        "max_wrong_margin": m["max_wrong_margin"],
        "misses": [r for r in m["rows"] if not r["correct"]],
    }


def evaluate_precision(model, tokenizer, all_train, binary_cases, family_cases):
    heads, specialist = prepare_precision(model, tokenizer, all_train, binary_cases, family_cases)
    suites = {
        "old20": evaluate_suite(model, tokenizer, heads, specialist, held.CASES),
        "confirm50": evaluate_suite(model, tokenizer, heads, specialist, confirm.CONFIRM),
        "third100": evaluate_suite(model, tokenizer, heads, specialist, third.THIRD),
    }
    return heads, specialist, suites


def summary_md(report):
    lines = [
        "# Ember v0.0.54 weather/time specialist router",
        "",
        "Ember weights remain frozen. Expanded binary + family heads are unchanged; a 64/64 weather-vs-time specialist is added.",
        "The specialist is used only when the family head predicts weather or get_time.",
        "No new final confirmation was consumed.",
        "",
        "| Mode | Specialist CV | Old20 | Confirm50 | Third100 | Exact all 170 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in ("full", "int4"):
        row = report["routers"][key]
        s = row["suites"]
        lines.append(
            f"| {key} | {row['specialist_cv']['accuracy']:.1%} | "
            f"{s['old20']['five_way_correct']}/20 | {s['confirm50']['five_way_correct']}/50 | "
            f"{s['third100']['five_way_correct']}/100 | {row['exact_all_dev']} |"
        )
    lines += [
        "",
        f"Strict precision-specific development pass: **{report['strict_dev_pass']}**",
        "",
        f"Interpretation: {report['interpretation']}",
        "",
        "No checkpoint, router integration, or production state changed.",
        "",
    ]
    return "\n".join(lines)


def main():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN required")
    torch.set_num_threads(2)
    torch.manual_seed(20260910)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    all_train, binary_cases, family_cases = expanded.build_training()

    with tempfile.TemporaryDirectory(prefix="ember-weather-time-router-") as td:
        work = Path(td)
        archive = ev.download_verified(ev.PACKAGE_URL, work / "ember.zip", ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        import sys
        sys.path.insert(0, str(work / "src" / "ember"))

        api = HfApi(token=token)
        owner = api.whoami()["name"]
        repo = f"{owner}/{held.MODEL_NAME}"

        full_model, full_tok, _ = held.load_full(repo, work / "full", token)
        full_model.eval()
        full_heads, full_specialist, full_suites = evaluate_precision(
            full_model, full_tok, all_train, binary_cases, family_cases
        )

        int4_model, int4_tok = held.load_int4(repo, work / "int4", token)
        int4_model.eval()
        int4_heads, int4_specialist, int4_suites = evaluate_precision(
            int4_model, int4_tok, all_train, binary_cases, family_cases
        )

        full_exact = all(exact(x) for x in full_suites.values())
        int4_exact = all(exact(x) for x in int4_suites.values())
        strict = full_exact and int4_exact
        if strict:
            interpretation = (
                "Precision-specific expanded hierarchical routers plus the weather/time specialist clear all 170 known development prompts exactly. "
                "Freeze this architecture and training set, then run a fourth untouched confirmation before any router save or integration."
            )
        else:
            interpretation = (
                "Known development misses remain even with the weather/time specialist. Do not consume a fourth confirmation; inspect only the remaining misses."
            )

        report = {
            "schema_version": 1,
            "diagnostic": "ember-v054-weather-time-specialist-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_repo": repo,
            "representation": REP,
            "specialist_pair": list(PAIR),
            "training_count": len(all_train),
            "binary_training_count": len(binary_cases),
            "family_training_count": len(family_cases),
            "routers": {
                "full": {
                    "binary_cv": full_heads["binary_cv"],
                    "family_cv": full_heads["family_cv"],
                    "specialist_cv": full_specialist["cv"],
                    "suites": full_suites,
                    "exact_all_dev": full_exact,
                },
                "int4": {
                    "binary_cv": int4_heads["binary_cv"],
                    "family_cv": int4_heads["family_cv"],
                    "specialist_cv": int4_specialist["cv"],
                    "suites": int4_suites,
                    "exact_all_dev": int4_exact,
                },
            },
            "strict_dev_pass": strict,
            "precision_specific_heads_required": True,
            "ember_weights_changed": False,
            "router_integrated": False,
            "production_changed": False,
            "fresh_final_confirmation_consumed": False,
            "interpretation": interpretation,
            "elapsed_seconds": time.monotonic() - started,
        }
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (OUT / "summary.md").write_text(summary_md(report), encoding="utf-8")
        print(json.dumps({
            "event": "weather_time_specialist_complete",
            "full": {k: compact(v) for k, v in full_suites.items()},
            "int4": {k: compact(v) for k, v in int4_suites.items()},
            "full_specialist_cv": full_specialist["cv"],
            "int4_specialist_cv": int4_specialist["cv"],
            "strict_dev_pass": strict,
            "interpretation": interpretation,
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
