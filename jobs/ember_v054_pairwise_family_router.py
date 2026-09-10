"""Generic pairwise family router for Ember v0.0.54.

Ember weights are frozen. The expanded binary direct/tool gate is unchanged.
Tool-family routing is replaced by six one-vs-one block-3 ridge heads covering
every pair among weather/calculator/web_search/get_time. Each pair uses 64+64
existing family-training prompts and chooses ridge only by training CV. A unique
majority vote wins; only a tournament tie falls back to the already-fit four-way
family head.

The failed 150-case final confirmation is intentionally not imported or scored.
Only the established 20+50+100 development suites are evaluated here. A new
untouched confirmation is required if this architecture clears development.
"""
from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
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

OUT = Path("v054-pairwise-family-router")
REP = hier.REPRESENTATION
FAMILIES = tuple(hier.FAMILY_LABELS)
PAIRS = tuple(combinations(FAMILIES, 2))


def fit_pairwise(all_rows, family_cases):
    by_id = {r["id"]: r for r in all_rows}
    states = {}
    for a, b in PAIRS:
        cases = [c for c in family_cases if probe.label_of(c) in {a, b}]
        rows = [by_id[c["id"]] for c in cases]
        X, labels = hier.stack(rows, REP)
        y = torch.tensor([0 if label == a else 1 for label in labels], dtype=torch.long)
        counts = [int((y == i).sum()) for i in range(2)]
        if counts != [64, 64]:
            raise RuntimeError(f"pair {a}/{b} must be 64/64, got {counts}")
        cv, records = hier.cv_ridge(X, y, 2)
        states[(a, b)] = {
            "state": hier.fit_ridge(X, y, 2, cv["ridge"]),
            "cv": cv,
            "cv_records": records,
            "counts": counts,
        }
    return states


def prepare(model, tokenizer, all_train, binary_cases, family_cases):
    rows = hier.extract(model, tokenizer, all_train)
    base_heads = expanded.fit_expanded(rows, binary_cases, family_cases)
    pairwise = fit_pairwise(rows, family_cases)
    return base_heads, pairwise


def predict(base_heads, pairwise, X):
    binary_pred, _, binary_margin = hier.predict(base_heads["binary"], X)
    base_family_pred, base_family_logits, base_family_margin = hier.predict(base_heads["family"], X)
    pair_outputs = {}
    for pair, item in pairwise.items():
        pair_outputs[pair] = hier.predict(item["state"], X)

    labels, margins, tie_count = [], [], 0
    for i in range(X.shape[0]):
        if int(binary_pred[i]) == 0:
            labels.append("direct")
            margins.append(float(binary_margin[i]))
            continue

        votes = {family: 0 for family in FAMILIES}
        pair_margins = []
        for (a, b), (pred, _logits, pmargin) in pair_outputs.items():
            winner = a if int(pred[i]) == 0 else b
            votes[winner] += 1
            pair_margins.append(float(pmargin[i]))
        max_votes = max(votes.values())
        tied = [family for family, count in votes.items() if count == max_votes]
        if len(tied) == 1:
            winner = tied[0]
            sorted_votes = sorted(votes.values(), reverse=True)
            fam_margin = float(sorted_votes[0] - sorted_votes[1])
        else:
            tie_count += 1
            indices = [FAMILIES.index(x) for x in tied]
            best_index = max(indices, key=lambda j: float(base_family_logits[i, j]))
            winner = FAMILIES[best_index]
            fam_margin = float(base_family_margin[i])
        labels.append(winner)
        margins.append(float(min(float(binary_margin[i]), fam_margin)))
    return labels, margins, tie_count


def evaluate(model, tokenizer, base_heads, pairwise, cases):
    rows = hier.extract(model, tokenizer, cases)
    X, _ = hier.stack(rows, REP)
    labels, margins, ties = predict(base_heads, pairwise, X)
    scored = hier.score(labels, rows, cases, margins)
    scored["pairwise_ties"] = ties
    return scored


def exact(m):
    return m["five_way_correct"] == m["total"] and m["direct_vs_tool_correct"] == m["total"] and m["tool_family_correct"] == m["tool_family_total"]


def compact(m):
    return {
        "five_way": f"{m['five_way_correct']}/{m['total']}",
        "direct_tool": f"{m['direct_vs_tool_correct']}/{m['total']}",
        "tool_family": f"{m['tool_family_correct']}/{m['tool_family_total']}",
        "pairwise_ties": m["pairwise_ties"],
        "misses": [r for r in m["rows"] if not r["correct"]],
    }


def evaluate_precision(model, tokenizer, all_train, binary_cases, family_cases):
    base_heads, pairwise = prepare(model, tokenizer, all_train, binary_cases, family_cases)
    suites = {
        "old20": evaluate(model, tokenizer, base_heads, pairwise, held.CASES),
        "confirm50": evaluate(model, tokenizer, base_heads, pairwise, confirm.CONFIRM),
        "third100": evaluate(model, tokenizer, base_heads, pairwise, third.THIRD),
    }
    return base_heads, pairwise, suites


def summary_md(report):
    lines = [
        "# Ember v0.0.54 pairwise family router",
        "",
        "Frozen Ember. Expanded direct/tool gate plus six train-CV-selected one-vs-one family heads.",
        "The failed 150-case final confirmation was not imported or evaluated.",
        "",
        "| Mode | Old20 | Confirm50 | Third100 | Exact all 170 | Pairwise CV range |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for key in ("full", "int4"):
        row = report["routers"][key]
        s = row["suites"]
        cvs = [x["accuracy"] for x in row["pairwise_cv"].values()]
        lines.append(
            f"| {key} | {s['old20']['five_way_correct']}/20 | {s['confirm50']['five_way_correct']}/50 | "
            f"{s['third100']['five_way_correct']}/100 | {row['exact_all_dev']} | {min(cvs):.1%}–{max(cvs):.1%} |"
        )
    lines += ["", f"Strict development pass: **{report['strict_dev_pass']}**", "", f"Interpretation: {report['interpretation']}", "", "No Ember weights, router artifacts, or production state changed.", ""]
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

    with tempfile.TemporaryDirectory(prefix="ember-pairwise-router-") as td:
        work = Path(td)
        archive = ev.download_verified(ev.PACKAGE_URL, work / "ember.zip", ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        import sys
        sys.path.insert(0, str(work / "src" / "ember"))
        api = HfApi(token=token)
        owner = api.whoami()["name"]
        repo = f"{owner}/{held.MODEL_NAME}"

        fm, ftok, _ = held.load_full(repo, work / "full", token)
        fm.eval()
        fh, fp, fs = evaluate_precision(fm, ftok, all_train, binary_cases, family_cases)

        im, itok = held.load_int4(repo, work / "int4", token)
        im.eval()
        ih, ip, ins = evaluate_precision(im, itok, all_train, binary_cases, family_cases)

        fexact = all(exact(x) for x in fs.values())
        iexact = all(exact(x) for x in ins.values())
        strict = fexact and iexact
        interpretation = (
            "Pairwise precision-specific family routing clears all 170 known development prompts in full and INT4. Freeze this architecture and run a new untouched confirmation; do not reuse the failed prior final."
            if strict else
            "Pairwise family routing still has known development misses. Do not consume another untouched confirmation yet."
        )
        def pair_cv(items):
            return {f"{a}__{b}": item["cv"] for (a, b), item in items.items()}
        report = {
            "schema_version": 1,
            "diagnostic": "ember-v054-pairwise-family-router-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_repo": repo,
            "representation": REP,
            "family_pairs": [list(x) for x in PAIRS],
            "failed_prior_final_imported": False,
            "routers": {
                "full": {"binary_cv": fh["binary_cv"], "pairwise_cv": pair_cv(fp), "suites": fs, "exact_all_dev": fexact},
                "int4": {"binary_cv": ih["binary_cv"], "pairwise_cv": pair_cv(ip), "suites": ins, "exact_all_dev": iexact},
            },
            "strict_dev_pass": strict,
            "precision_specific_heads_required": True,
            "ember_weights_changed": False,
            "router_integrated": False,
            "production_changed": False,
            "interpretation": interpretation,
            "elapsed_seconds": time.monotonic() - started,
        }
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (OUT / "summary.md").write_text(summary_md(report), encoding="utf-8")
        print(json.dumps({
            "event": "pairwise_family_router_complete",
            "full": {k: compact(v) for k, v in fs.items()},
            "int4": {k: compact(v) for k, v in ins.items()},
            "full_pairwise_cv": pair_cv(fp),
            "int4_pairwise_cv": pair_cv(ip),
            "strict_dev_pass": strict,
            "interpretation": interpretation,
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
