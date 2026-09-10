"""Frozen tool-triad specialist for Ember v0.0.54.

This diagnostic is driven only by training-CV and the established 170-case
development evidence. The sealed failed 150-, 200-, and 250-case final suites
are not imported or scored.

Architecture:
  * direct/tool gate remains block_04 from the split router;
  * calculator remains the base four-way family decision;
  * whenever the base family head predicts weather, web_search, or get_time,
    a 3-way specialist trained on 64 examples per family using block_02+block_03
    makes the final choice among that triad.

Ridge strength is selected by training-only CV. Ember language weights remain
frozen. No router artifact, checkpoint, integration, or production pointer is
changed. If both full and INT4 clear all 170 known development prompts, freeze
this architecture and use a new untouched confirmation later.
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
from jobs import ember_v054_split_representation_router as split

OUT = Path("v054-tool-triad-router")
TRIAD = ("weather", "web_search", "get_time")


def unique_cases(*groups):
    out, seen = [], set()
    for group in groups:
        for c in group:
            if c["id"] not in seen:
                seen.add(c["id"])
                out.append(c)
    return out


def fit_triad(rows_by_id, family_cases):
    cases = [c for c in family_cases if probe.label_of(c) in TRIAD]
    X = sweep.feature_matrix(rows_by_id, cases, split.FAMILY_REP)
    labels = [probe.label_of(c) for c in cases]
    y = torch.tensor([TRIAD.index(label) for label in labels], dtype=torch.long)
    counts = [int((y == i).sum().item()) for i in range(len(TRIAD))]
    if counts != [64, 64, 64]:
        raise RuntimeError(f"triad training must be 64/class, got {counts}")
    state, cv, records = sweep.fit_cv_state(X, y, len(TRIAD))
    return {"state": state, "cv": cv, "cv_records": records, "counts": counts}


def fit_heads(rows_by_id, binary_cases, family_cases):
    base = split.fit_heads(rows_by_id, binary_cases, family_cases)
    triad = fit_triad(rows_by_id, family_cases)
    return base, triad


def predict(rows_by_id, cases, base, triad):
    Xb = sweep.feature_matrix(rows_by_id, cases, split.BINARY_REP)
    Xf = sweep.feature_matrix(rows_by_id, cases, split.FAMILY_REP)
    bp, _bl, bm = hier.predict(base["binary"], Xb)
    fp, _fl, fm = hier.predict(base["family"], Xf)
    tp, _tl, tm = hier.predict(triad["state"], Xf)

    labels, margins, triad_used = [], [], []
    for i in range(len(cases)):
        if int(bp[i]) == 0:
            labels.append("direct")
            margins.append(float(bm[i]))
            triad_used.append(False)
            continue
        family = tuple(hier.FAMILY_LABELS)[int(fp[i])]
        if family in TRIAD:
            family = TRIAD[int(tp[i])]
            margins.append(float(min(bm[i], tm[i])))
            triad_used.append(True)
        else:
            margins.append(float(min(bm[i], fm[i])))
            triad_used.append(False)
        labels.append(family)
    return labels, margins, triad_used


def eval_suite(rows_by_id, cases, base, triad):
    labels, margins, used = predict(rows_by_id, cases, base, triad)
    pseudo = [{"id": c["id"], "label": probe.label_of(c)} for c in cases]
    scored = hier.score(labels, pseudo, cases, margins)
    scored["triad_invocations"] = sum(int(x) for x in used)
    for row, flag in zip(scored["rows"], used):
        row["triad_used"] = bool(flag)
    return scored


def exact(m):
    return (
        int(m["five_way_correct"]) == int(m["total"])
        and int(m["direct_vs_tool_correct"]) == int(m["total"])
        and int(m["tool_family_correct"]) == int(m["tool_family_total"])
    )


def evaluate_precision(model, tokenizer, all_train, binary_cases, family_cases):
    dev = unique_cases(held.CASES, confirm.CONFIRM, third.THIRD)
    combined = unique_cases(all_train, dev)
    rows, reps, _blocks, _head = probe.extract_representations(model, tokenizer, combined)
    needed = set(split.BINARY_REP + split.FAMILY_REP)
    if not needed.issubset(set(reps)):
        raise RuntimeError(f"required representations unavailable: {sorted(needed)} vs {reps}")
    by_id = {r["id"]: r for r in rows}
    base, triad = fit_heads(by_id, binary_cases, family_cases)
    suites = {
        "old20": eval_suite(by_id, held.CASES, base, triad),
        "confirm50": eval_suite(by_id, confirm.CONFIRM, base, triad),
        "third100": eval_suite(by_id, third.THIRD, base, triad),
    }
    return base, triad, suites


def summary_md(report):
    lines = [
        "# Ember v0.0.54 frozen tool-triad router", "",
        "Binary direct/tool: `block_04`. Base family + triad: `block_02+block_03`.",
        "Triad specialist: weather / web_search / get_time, 64 training examples per class.",
        "All three failed final suites were excluded. Ember weights stayed frozen.", "",
        "| Mode | Binary CV | Base family CV | Triad CV | Old20 | Confirm50 | Third100 | Exact 170 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in ("full", "int4"):
        r = report["routers"][key]
        s = r["suites"]
        lines.append(
            f"| {key} | {r['binary_cv']['accuracy']:.1%} | {r['family_cv']['accuracy']:.1%} | "
            f"{r['triad_cv']['accuracy']:.1%} | {s['old20']['five_way_correct']}/20 | "
            f"{s['confirm50']['five_way_correct']}/50 | {s['third100']['five_way_correct']}/100 | {r['exact_all_dev']} |"
        )
    lines += ["", f"Strict development pass: **{report['strict_dev_pass']}**", "", f"Interpretation: {report['interpretation']}", "", "No router artifact, checkpoint, integration, or production state changed.", ""]
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

    with tempfile.TemporaryDirectory(prefix="ember-tool-triad-") as td:
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
        fb, ft, fs = evaluate_precision(fm, ftok, all_train, binary_cases, family_cases)

        im, itok = held.load_int4(repo, work / "int4", token)
        im.eval()
        ib, it, ins = evaluate_precision(im, itok, all_train, binary_cases, family_cases)

        fexact = all(exact(x) for x in fs.values())
        iexact = all(exact(x) for x in ins.values())
        strict = fexact and iexact
        interpretation = (
            "The precision-specific split router plus three-way weather/search/time specialist clears all 170 established development prompts in full and INT4. Freeze this architecture and use a brand-new untouched confirmation; do not reuse any failed final."
            if strict else
            "The triad specialist still has established development misses. Do not consume another final set; improve using training/development evidence only."
        )
        def meta(base, triad):
            return {
                "binary_cv": base["binary_cv"],
                "family_cv": base["family_cv"],
                "weather_time_cv": base["weather_time_cv"],
                "triad_cv": triad["cv"],
                "triad_counts": triad["counts"],
            }
        report = {
            "schema_version":1,
            "diagnostic":"ember-v054-tool-triad-router-v1",
            "created_at":datetime.now(timezone.utc).isoformat(),
            "model_repo":repo,
            "binary_representation":list(split.BINARY_REP),
            "family_representation":list(split.FAMILY_REP),
            "triad":list(TRIAD),
            "failed_final_150_imported":False,
            "failed_final_200_imported":False,
            "failed_final_250_imported":False,
            "routers":{
                "full":{**meta(fb,ft),"suites":fs,"exact_all_dev":fexact},
                "int4":{**meta(ib,it),"suites":ins,"exact_all_dev":iexact},
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
            "event":"tool_triad_router_complete",
            "full":{k:{"five_way":f"{v['five_way_correct']}/{v['total']}","direct_tool":f"{v['direct_vs_tool_correct']}/{v['total']}","tool_family":f"{v['tool_family_correct']}/{v['tool_family_total']}","triad_uses":v['triad_invocations']} for k,v in fs.items()},
            "int4":{k:{"five_way":f"{v['five_way_correct']}/{v['total']}","direct_tool":f"{v['direct_vs_tool_correct']}/{v['total']}","tool_family":f"{v['tool_family_correct']}/{v['tool_family_total']}","triad_uses":v['triad_invocations']} for k,v in ins.items()},
            "full_triad_cv":ft["cv"],"int4_triad_cv":it["cv"],
            "strict_dev_pass":strict,"interpretation":interpretation,
            "elapsed_seconds":report["elapsed_seconds"],
        }),flush=True)


if __name__ == "__main__":
    main()
