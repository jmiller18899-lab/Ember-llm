"""Frozen split router with a block_02 weather/search/time specialist.

Training/dev only. The sealed 150-, 200-, and 250-case finals are not imported.
Ember weights remain unchanged.

Representations:
- direct/tool gate: block_04
- base 4-way family head: block_02 + block_03
- 3-way weather/web_search/get_time specialist: block_02

The triad representation is chosen from the pre-existing training-only
representation sweep because block_02 had the best joint weather-vs-web_search
CV among tested representations. Ridge selection remains training-CV only.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, tempfile, time, zipfile
from pathlib import Path
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

OUT = Path("v054-block2-triad-router")
TRIAD = ("weather", "web_search", "get_time")
TRIAD_REP = ("block_02",)


def unique_cases(*groups):
    out, seen = [], set()
    for group in groups:
        for c in group:
            if c["id"] not in seen:
                seen.add(c["id"]); out.append(c)
    return out


def fit_triad(by_id, family_cases):
    cases = [c for c in family_cases if probe.label_of(c) in TRIAD]
    X = sweep.feature_matrix(by_id, cases, TRIAD_REP)
    y = torch.tensor([TRIAD.index(probe.label_of(c)) for c in cases], dtype=torch.long)
    counts = [int((y == i).sum().item()) for i in range(3)]
    if counts != [64,64,64]:
        raise RuntimeError(f"triad counts drifted: {counts}")
    state, cv, records = sweep.fit_cv_state(X, y, 3)
    return {"state":state,"cv":cv,"records":records,"counts":counts}


def predict(by_id, cases, base, triad):
    xb = sweep.feature_matrix(by_id, cases, split.BINARY_REP)
    xf = sweep.feature_matrix(by_id, cases, split.FAMILY_REP)
    xt = sweep.feature_matrix(by_id, cases, TRIAD_REP)
    bp, _bl, bm = hier.predict(base["binary"], xb)
    fp, _fl, fm = hier.predict(base["family"], xf)
    tp, _tl, tm = hier.predict(triad["state"], xt)
    labels, margins, used = [], [], []
    families = tuple(hier.FAMILY_LABELS)
    for i in range(len(cases)):
        if int(bp[i]) == 0:
            labels.append("direct"); margins.append(float(bm[i])); used.append(False); continue
        family = families[int(fp[i])]
        if family in TRIAD:
            family = TRIAD[int(tp[i])]
            margins.append(float(min(bm[i], tm[i]))); used.append(True)
        else:
            margins.append(float(min(bm[i], fm[i]))); used.append(False)
        labels.append(family)
    return labels, margins, used


def eval_suite(by_id, cases, base, triad):
    labels, margins, used = predict(by_id, cases, base, triad)
    pseudo = [{"id":c["id"],"label":probe.label_of(c)} for c in cases]
    scored = hier.score(labels, pseudo, cases, margins)
    scored["triad_invocations"] = sum(int(x) for x in used)
    return scored


def exact(m):
    return m["five_way_correct"]==m["total"] and m["direct_vs_tool_correct"]==m["total"] and m["tool_family_correct"]==m["tool_family_total"]


def evaluate(model, tok, all_train, binary_cases, family_cases):
    dev = unique_cases(held.CASES, confirm.CONFIRM, third.THIRD)
    combined = unique_cases(all_train, dev)
    rows, reps, _blocks, _head = probe.extract_representations(model, tok, combined)
    needed = set(split.BINARY_REP + split.FAMILY_REP + TRIAD_REP)
    if not needed.issubset(set(reps)):
        raise RuntimeError(f"missing reps {sorted(needed)} from {reps}")
    by_id = {r["id"]:r for r in rows}
    base = split.fit_heads(by_id, binary_cases, family_cases)
    triad = fit_triad(by_id, family_cases)
    suites = {
        "old20":eval_suite(by_id, held.CASES, base, triad),
        "confirm50":eval_suite(by_id, confirm.CONFIRM, base, triad),
        "third100":eval_suite(by_id, third.THIRD, base, triad),
    }
    return base, triad, suites


def summary_md(report):
    lines=["# Ember v0.0.54 block2 triad router","",
      "Frozen Ember; binary=`block_04`, base family=`block_02+block_03`, triad=`block_02`.",
      "All sealed finals excluded.","",
      "| Mode | Triad CV | Old20 | Confirm50 | Third100 | Exact170 |",
      "| --- | ---: | ---: | ---: | ---: | --- |"]
    for key in ("full","int4"):
        r=report["routers"][key]; s=r["suites"]
        lines.append(f"| {key} | {r['triad_cv']['accuracy']:.1%} | {s['old20']['five_way_correct']}/20 | {s['confirm50']['five_way_correct']}/50 | {s['third100']['five_way_correct']}/100 | {r['exact_all_dev']} |")
    lines += ["",f"Strict development pass: **{report['strict_dev_pass']}**","",f"Interpretation: {report['interpretation']}","","No router artifact, checkpoint, integration, or production state changed.",""]
    return "\n".join(lines)


def main():
    token=os.environ.get("HF_TOKEN","").strip()
    if not token: raise RuntimeError("HF_TOKEN required")
    torch.set_num_threads(2); torch.manual_seed(20260910); torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True,exist_ok=True); started=time.monotonic()
    all_train,binary_cases,family_cases=expanded.build_training()
    with tempfile.TemporaryDirectory(prefix="ember-block2-triad-") as td:
        work=Path(td); archive=ev.download_verified(ev.PACKAGE_URL,work/"ember.zip",ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package: package.extractall(work/"src")
        import sys; sys.path.insert(0,str(work/"src"/"ember"))
        api=HfApi(token=token); repo=f"{api.whoami()['name']}/{held.MODEL_NAME}"
        fm,ftok,_=held.load_full(repo,work/"full",token); fm.eval(); fb,ft,fs=evaluate(fm,ftok,all_train,binary_cases,family_cases)
        im,itok=held.load_int4(repo,work/"int4",token); im.eval(); ib,it,ins=evaluate(im,itok,all_train,binary_cases,family_cases)
        fexact=all(exact(x) for x in fs.values()); iexact=all(exact(x) for x in ins.values()); strict=fexact and iexact
        interpretation=("Block_02 triad specialist clears all 170 established development prompts in full and INT4. Freeze this architecture and use a brand-new untouched confirmation." if strict else "Block_02 triad specialist still has established development misses. Do not consume another final set.")
        report={"schema_version":1,"diagnostic":"ember-v054-block2-triad-router-v1","created_at":datetime.now(timezone.utc).isoformat(),"model_repo":repo,"binary_representation":list(split.BINARY_REP),"family_representation":list(split.FAMILY_REP),"triad_representation":list(TRIAD_REP),"triad":list(TRIAD),"failed_final_150_imported":False,"failed_final_200_imported":False,"failed_final_250_imported":False,"routers":{"full":{"binary_cv":fb["binary_cv"],"family_cv":fb["family_cv"],"triad_cv":ft["cv"],"suites":fs,"exact_all_dev":fexact},"int4":{"binary_cv":ib["binary_cv"],"family_cv":ib["family_cv"],"triad_cv":it["cv"],"suites":ins,"exact_all_dev":iexact}},"strict_dev_pass":strict,"ember_weights_changed":False,"router_integrated":False,"production_changed":False,"interpretation":interpretation,"elapsed_seconds":time.monotonic()-started}
        (OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8"); (OUT/"summary.md").write_text(summary_md(report),encoding="utf-8")
        print(json.dumps({"event":"block2_triad_router_complete","full":{k:f"{v['five_way_correct']}/{v['total']}" for k,v in fs.items()},"int4":{k:f"{v['five_way_correct']}/{v['total']}" for k,v in ins.items()},"full_triad_cv":ft["cv"],"int4_triad_cv":it["cv"],"strict_dev_pass":strict,"interpretation":interpretation,"elapsed_seconds":report["elapsed_seconds"]}),flush=True)

if __name__=="__main__": main()
