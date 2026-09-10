"""Frozen representation sweep for Ember v0.0.54 routing.

The failed 150-case final confirmation is not imported or scored. Ember weights
remain frozen. This diagnostic asks whether tool-family semantics separate more
robustly when the router sees neighboring hidden layers instead of block_03
alone.

Candidate representations are fixed combinations of blocks 2-5 and the final
hidden state. For each candidate and precision, ridge strengths are selected by
training-only CV for:
  * direct vs tool;
  * four-way tool family;
  * weather vs get_time specialist;
  * weather vs web_search diagnostic pair;
  * web_search vs get_time diagnostic pair.

A single representation is selected jointly across full and INT4 using only
training-CV scores, subject to clearing the existing 20+50+100 development
suites in both precisions. A brand-new confirmation is required afterward.
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

OUT = Path("v054-representation-sweep")
FAMILIES = tuple(hier.FAMILY_LABELS)
PAIR_DIAGNOSTICS = (("weather", "web_search"), ("web_search", "get_time"))


def unique_cases(*groups):
    out, seen = [], set()
    for group in groups:
        for c in group:
            if c["id"] not in seen:
                seen.add(c["id"])
                out.append(c)
    return out


def candidate_specs(reps):
    available = set(reps)
    wanted = [
        ("block_02",), ("block_03",), ("block_04",), ("block_05",), ("final_hidden",),
        ("block_02","block_03"), ("block_03","block_04"), ("block_04","block_05"),
        ("block_03","final_hidden"), ("block_04","final_hidden"),
        ("block_02","block_03","block_04"), ("block_03","block_04","block_05"),
        ("block_02","block_03","final_hidden"), ("block_03","block_04","final_hidden"),
        ("block_02","block_03","block_04","block_05"),
        ("block_02","block_03","block_04","block_05","final_hidden"),
    ]
    specs = [x for x in wanted if all(rep in available for rep in x)]
    if ("block_03",) not in specs:
        raise RuntimeError(f"block_03 unavailable: {reps}")
    return specs


def spec_name(spec):
    return "+".join(spec)


def feature_matrix(rows_by_id, cases, spec):
    vectors = []
    for c in cases:
        row = rows_by_id[c["id"]]
        vectors.append(torch.cat([row["features"][rep] for rep in spec]).double())
    return torch.stack(vectors)


def fit_cv_state(X, y, classes):
    cv, records = hier.cv_ridge(X, y, classes)
    return hier.fit_ridge(X, y, classes, cv["ridge"]), cv, records


def fit_candidate(rows_by_id, spec, binary_cases, family_cases):
    Xb = feature_matrix(rows_by_id, binary_cases, spec)
    yb = torch.tensor([0 if probe.label_of(c) == "direct" else 1 for c in binary_cases], dtype=torch.long)
    binary, binary_cv, _ = fit_cv_state(Xb, yb, 2)

    Xf = feature_matrix(rows_by_id, family_cases, spec)
    yf = torch.tensor([FAMILIES.index(probe.label_of(c)) for c in family_cases], dtype=torch.long)
    family, family_cv, _ = fit_cv_state(Xf, yf, 4)

    wt_cases = [c for c in family_cases if probe.label_of(c) in {"weather", "get_time"}]
    Xwt = feature_matrix(rows_by_id, wt_cases, spec)
    ywt = torch.tensor([0 if probe.label_of(c) == "weather" else 1 for c in wt_cases], dtype=torch.long)
    wt, wt_cv, _ = fit_cv_state(Xwt, ywt, 2)

    pair_cv = {}
    for a, b in PAIR_DIAGNOSTICS:
        cases = [c for c in family_cases if probe.label_of(c) in {a, b}]
        X = feature_matrix(rows_by_id, cases, spec)
        y = torch.tensor([0 if probe.label_of(c) == a else 1 for c in cases], dtype=torch.long)
        _state, cv, _records = fit_cv_state(X, y, 2)
        pair_cv[f"{a}__{b}"] = cv

    return {
        "binary": binary,
        "family": family,
        "weather_time": wt,
        "binary_cv": binary_cv,
        "family_cv": family_cv,
        "weather_time_cv": wt_cv,
        "pair_cv": pair_cv,
    }


def predict_heads(heads, X):
    bp, _bl, bm = hier.predict(heads["binary"], X)
    fp, _fl, fm = hier.predict(heads["family"], X)
    wp, _wl, wm = hier.predict(heads["weather_time"], X)
    labels, margins = [], []
    for i in range(X.shape[0]):
        if int(bp[i]) == 0:
            labels.append("direct")
            margins.append(float(bm[i]))
            continue
        family = FAMILIES[int(fp[i])]
        if family in {"weather", "get_time"}:
            family = "weather" if int(wp[i]) == 0 else "get_time"
            margins.append(float(min(bm[i], wm[i])))
        else:
            margins.append(float(min(bm[i], fm[i])))
        labels.append(family)
    return labels, margins


def eval_suite(rows_by_id, spec, heads, cases):
    X = feature_matrix(rows_by_id, cases, spec)
    pred, margins = predict_heads(heads, X)
    pseudo_rows = [{"id": c["id"], "label": probe.label_of(c)} for c in cases]
    return hier.score(pred, pseudo_rows, cases, margins)


def exact(m):
    return m["five_way_correct"] == m["total"] and m["direct_vs_tool_correct"] == m["total"] and m["tool_family_correct"] == m["tool_family_total"]


def evaluate_precision(model, tokenizer, all_cases, specs, binary_cases, family_cases):
    rows, reps, _blocks, _head = probe.extract_representations(model, tokenizer, all_cases)
    rows_by_id = {r["id"]: r for r in rows}
    results = {}
    for spec in specs:
        name = spec_name(spec)
        heads = fit_candidate(rows_by_id, spec, binary_cases, family_cases)
        suites = {
            "old20": eval_suite(rows_by_id, spec, heads, held.CASES),
            "confirm50": eval_suite(rows_by_id, spec, heads, confirm.CONFIRM),
            "third100": eval_suite(rows_by_id, spec, heads, third.THIRD),
        }
        pair_floor = min(cv["accuracy"] for cv in heads["pair_cv"].values())
        results[name] = {
            "spec": list(spec),
            "dimension": int(feature_matrix(rows_by_id, [binary_cases[0]], spec).shape[1]),
            "binary_cv": heads["binary_cv"],
            "family_cv": heads["family_cv"],
            "weather_time_cv": heads["weather_time_cv"],
            "pair_cv": heads["pair_cv"],
            "pair_cv_floor": pair_floor,
            "suites": suites,
            "exact_all_dev": all(exact(x) for x in suites.values()),
        }
        print(json.dumps({
            "event":"representation_candidate",
            "precision":"pending-label",
            "representation":name,
            "binary_cv":heads["binary_cv"]["accuracy"],
            "family_cv":heads["family_cv"]["accuracy"],
            "weather_time_cv":heads["weather_time_cv"]["accuracy"],
            "pair_cv_floor":pair_floor,
            "dev":[suites["old20"]["five_way_correct"], suites["confirm50"]["five_way_correct"], suites["third100"]["five_way_correct"]],
        }), flush=True)
    return results, reps


def joint_score(full_row, int4_row):
    # Train-only selection score. Development is a hard eligibility constraint,
    # never part of this ranking value.
    pair_floor = min(full_row["pair_cv_floor"], int4_row["pair_cv_floor"])
    family_floor = min(full_row["family_cv"]["accuracy"], int4_row["family_cv"]["accuracy"])
    binary_floor = min(full_row["binary_cv"]["accuracy"], int4_row["binary_cv"]["accuracy"])
    wt_floor = min(full_row["weather_time_cv"]["accuracy"], int4_row["weather_time_cv"]["accuracy"])
    dim = max(full_row["dimension"], int4_row["dimension"])
    return (pair_floor, family_floor, binary_floor, wt_floor, -dim)


def summary_md(report):
    lines = [
        "# Ember v0.0.54 frozen representation sweep", "",
        "No failed-final data was imported. Ember weights stayed frozen.",
        "Selection ranking uses training-only CV; the 170 known cases are development eligibility only.", "",
        "| Representation | Full family CV | INT4 family CV | Joint critical-pair floor | Full dev | INT4 dev | Eligible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, row in report["joint"].items():
        f, i = row["full"], row["int4"]
        fdev = sum(f["suites"][x]["five_way_correct"] for x in ("old20","confirm50","third100"))
        idev = sum(i["suites"][x]["five_way_correct"] for x in ("old20","confirm50","third100"))
        lines.append(
            f"| {name} | {f['family_cv']['accuracy']:.1%} | {i['family_cv']['accuracy']:.1%} | "
            f"{min(f['pair_cv_floor'], i['pair_cv_floor']):.1%} | {fdev}/170 | {idev}/170 | {row['eligible']} |"
        )
    lines += ["", f"Selected representation: **{report.get('selected_representation')}**", f"Strict development eligible: **{report.get('strict_dev_pass')}**", "", f"Interpretation: {report['interpretation']}", "", "No router artifact, checkpoint, or production state changed.", ""]
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
    dev_cases = unique_cases(held.CASES, confirm.CONFIRM, third.THIRD)
    combined = unique_cases(all_train, dev_cases)

    with tempfile.TemporaryDirectory(prefix="ember-representation-sweep-") as td:
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
        # Discover feature names from one extraction before fixing candidate set.
        _probe_rows, available, _blocks, _head = probe.extract_representations(fm, ftok, [combined[0]])
        specs = candidate_specs(available)
        full_results, full_available = evaluate_precision(fm, ftok, combined, specs, binary_cases, family_cases)

        im, itok = held.load_int4(repo, work / "int4", token)
        im.eval()
        int4_results, int4_available = evaluate_precision(im, itok, combined, specs, binary_cases, family_cases)

        joint = {}
        eligible = []
        for spec in specs:
            name = spec_name(spec)
            f, i = full_results[name], int4_results[name]
            ok = bool(f["exact_all_dev"] and i["exact_all_dev"])
            score = joint_score(f, i)
            joint[name] = {"full": f, "int4": i, "eligible": ok, "train_only_score": list(score)}
            if ok:
                eligible.append((score, name))

        if eligible:
            eligible.sort(reverse=True)
            selected = eligible[0][1]
            strict = True
            interpretation = (
                "A common frozen representation clears all 170 development prompts in full and INT4 and was selected by training-only CV. "
                "Freeze it and run a brand-new untouched confirmation; do not reuse the failed prior final."
            )
        else:
            selected = None
            strict = False
            interpretation = (
                "No swept frozen representation simultaneously clears development in both precisions. Do not consume another final confirmation; inspect the highest-CV near miss."
            )

        report = {
            "schema_version":1,
            "diagnostic":"ember-v054-frozen-representation-sweep-v1",
            "created_at":datetime.now(timezone.utc).isoformat(),
            "model_repo":repo,
            "failed_prior_final_imported":False,
            "candidate_representations":[list(x) for x in specs],
            "full_available":full_available,
            "int4_available":int4_available,
            "joint":joint,
            "selected_representation":selected,
            "strict_dev_pass":strict,
            "ember_weights_changed":False,
            "router_integrated":False,
            "production_changed":False,
            "interpretation":interpretation,
            "elapsed_seconds":time.monotonic()-started,
        }
        (OUT/"report.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        (OUT/"summary.md").write_text(summary_md(report), encoding="utf-8")
        print(json.dumps({
            "event":"representation_sweep_complete",
            "selected_representation":selected,
            "strict_dev_pass":strict,
            "eligible":[name for _score,name in sorted(eligible, reverse=True)],
            "top_train_only":[{"name":name,"score":row["train_only_score"],"eligible":row["eligible"]} for name,row in sorted(joint.items(), key=lambda kv: tuple(kv[1]["train_only_score"]), reverse=True)[:8]],
            "interpretation":interpretation,
            "elapsed_seconds":report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
