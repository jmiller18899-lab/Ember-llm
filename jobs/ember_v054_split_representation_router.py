"""Split-representation frozen router for Ember v0.0.54.

Selected only from training-CV + established 170-case development evidence:
  * direct/tool gate uses block_04;
  * tool-family + weather/time specialist use concat(block_02, block_03).

The failed 150- and 200-case finals are not imported or scored. Ember weights
stay frozen. No router artifact is saved or integrated. If full and INT4 both
clear all 170 development prompts, freeze this architecture and consume a new
untouched confirmation set.
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
from jobs import ember_v054_representation_sweep as sweep

OUT = Path("v054-split-representation-router")
BINARY_REP = ("block_04",)
FAMILY_REP = ("block_02", "block_03")
FAMILIES = tuple(hier.FAMILY_LABELS)


def unique_cases(*groups):
    out, seen = [], set()
    for group in groups:
        for c in group:
            if c["id"] not in seen:
                seen.add(c["id"])
                out.append(c)
    return out


def fit_heads(rows_by_id, binary_cases, family_cases):
    Xb = sweep.feature_matrix(rows_by_id, binary_cases, BINARY_REP)
    yb = torch.tensor([0 if probe.label_of(c) == "direct" else 1 for c in binary_cases], dtype=torch.long)
    binary, binary_cv, binary_cv_records = sweep.fit_cv_state(Xb, yb, 2)

    # Use the already-defined family training/selection contract on block2+3.
    family_bundle = sweep.fit_candidate(rows_by_id, FAMILY_REP, binary_cases, family_cases)
    return {
        "binary": binary,
        "binary_cv": binary_cv,
        "binary_cv_records": binary_cv_records,
        "family": family_bundle["family"],
        "family_cv": family_bundle["family_cv"],
        "weather_time": family_bundle["weather_time"],
        "weather_time_cv": family_bundle["weather_time_cv"],
        "pair_cv": family_bundle["pair_cv"],
    }


def predict_split(rows_by_id, cases, heads):
    Xb = sweep.feature_matrix(rows_by_id, cases, BINARY_REP)
    Xf = sweep.feature_matrix(rows_by_id, cases, FAMILY_REP)
    bp, _bl, bm = hier.predict(heads["binary"], Xb)
    fp, _fl, fm = hier.predict(heads["family"], Xf)
    wp, _wl, wm = hier.predict(heads["weather_time"], Xf)
    labels, margins = [], []
    for i in range(len(cases)):
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


def eval_suite(rows_by_id, cases, heads):
    labels, margins = predict_split(rows_by_id, cases, heads)
    pseudo = [{"id": c["id"], "label": probe.label_of(c)} for c in cases]
    return hier.score(labels, pseudo, cases, margins)


def exact(m):
    return m["five_way_correct"] == m["total"] and m["direct_vs_tool_correct"] == m["total"] and m["tool_family_correct"] == m["tool_family_total"]


def evaluate_precision(model, tokenizer, all_train, binary_cases, family_cases):
    dev = unique_cases(held.CASES, confirm.CONFIRM, third.THIRD)
    combined = unique_cases(all_train, dev)
    rows, reps, _blocks, _head = probe.extract_representations(model, tokenizer, combined)
    needed = set(BINARY_REP + FAMILY_REP)
    if not needed.issubset(set(reps)):
        raise RuntimeError(f"required reps {sorted(needed)} unavailable: {reps}")
    by_id = {r["id"]: r for r in rows}
    heads = fit_heads(by_id, binary_cases, family_cases)
    suites = {
        "old20": eval_suite(by_id, held.CASES, heads),
        "confirm50": eval_suite(by_id, confirm.CONFIRM, heads),
        "third100": eval_suite(by_id, third.THIRD, heads),
    }
    return heads, suites


def summary_md(report):
    lines = [
        "# Ember v0.0.54 split-representation frozen router", "",
        "Binary direct/tool: `block_04`. Tool-family + weather/time: `block_02+block_03`.",
        "Both failed final suites were excluded. Ember weights stayed frozen.", "",
        "| Mode | Binary CV | Family CV | Old20 | Confirm50 | Third100 | Exact 170 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in ("full", "int4"):
        r = report["routers"][key]
        s = r["suites"]
        lines.append(
            f"| {key} | {r['binary_cv']['accuracy']:.1%} | {r['family_cv']['accuracy']:.1%} | "
            f"{s['old20']['five_way_correct']}/20 | {s['confirm50']['five_way_correct']}/50 | "
            f"{s['third100']['five_way_correct']}/100 | {r['exact_all_dev']} |"
        )
    lines += ["", f"Strict development pass: **{report['strict_dev_pass']}**", "", f"Interpretation: {report['interpretation']}", "", "No router artifact, checkpoint, or production state changed.", ""]
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

    with tempfile.TemporaryDirectory(prefix="ember-split-router-") as td:
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
        fh, fs = evaluate_precision(fm, ftok, all_train, binary_cases, family_cases)

        im, itok = held.load_int4(repo, work / "int4", token)
        im.eval()
        ih, ins = evaluate_precision(im, itok, all_train, binary_cases, family_cases)

        fexact = all(exact(x) for x in fs.values())
        iexact = all(exact(x) for x in ins.values())
        strict = fexact and iexact
        interpretation = (
            "The split-representation precision-specific router clears all 170 established development prompts in full and INT4. Freeze it and run a new untouched final confirmation."
            if strict else
            "The split representation does not clear established development in both precisions; do not consume another final set."
        )
        def meta(h):
            return {
                "binary_cv": h["binary_cv"], "family_cv": h["family_cv"],
                "weather_time_cv": h["weather_time_cv"], "pair_cv": h["pair_cv"],
            }
        report = {
            "schema_version":1,
            "diagnostic":"ember-v054-split-representation-router-v1",
            "created_at":datetime.now(timezone.utc).isoformat(),
            "model_repo":repo,
            "binary_representation":list(BINARY_REP),
            "family_representation":list(FAMILY_REP),
            "failed_final_150_imported":False,
            "failed_final_200_imported":False,
            "routers":{
                "full":{**meta(fh),"suites":fs,"exact_all_dev":fexact},
                "int4":{**meta(ih),"suites":ins,"exact_all_dev":iexact},
            },
            "strict_dev_pass":strict,
            "precision_specific_heads_required":True,
            "ember_weights_changed":False,
            "router_integrated":False,
            "production_changed":False,
            "interpretation":interpretation,
            "elapsed_seconds":time.monotonic()-started,
        }
        (OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        (OUT/"summary.md").write_text(summary_md(report),encoding="utf-8")
        print(json.dumps({
            "event":"split_representation_router_complete",
            "full":{k:{"five_way":f"{v['five_way_correct']}/{v['total']}","direct_tool":f"{v['direct_vs_tool_correct']}/{v['total']}"} for k,v in fs.items()},
            "int4":{k:{"five_way":f"{v['five_way_correct']}/{v['total']}","direct_tool":f"{v['direct_vs_tool_correct']}/{v['total']}"} for k,v in ins.items()},
            "full_binary_cv":fh["binary_cv"],"int4_binary_cv":ih["binary_cv"],
            "strict_dev_pass":strict,"interpretation":interpretation,
            "elapsed_seconds":report["elapsed_seconds"],
        }),flush=True)


if __name__ == "__main__":
    main()
