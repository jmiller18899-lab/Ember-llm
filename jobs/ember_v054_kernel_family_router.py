"""Frozen nonlinear kernel family router for Ember v0.0.54.

Training/development only. The sealed failed final suites are not imported or
scored. Ember language weights remain unchanged.

Architecture:
- direct/tool gate: existing ridge on block_04;
- tool-family representation: block_02 + block_03;
- four-way family classifier: RBF kernel ridge, hyperparameters selected only
  by deterministic stratified training CV;
- weather/get_time specialist: existing train-CV ridge on block_02+block_03.

This tests a nonlinear decision surface without changing Ember weights. No
router artifact is saved or integrated. A new untouched confirmation is allowed
only if both full and INT4 clear all 170 established development prompts.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, tempfile, time, zipfile
from pathlib import Path
from huggingface_hub import HfApi
import torch
import torch.nn.functional as F

from jobs import ember_hf_eval as ev
from jobs import ember_v053_heldout_routing as held
from jobs import ember_v054_frozen_router_probe as probe
from jobs import ember_v054_router_confirmation as confirm
from jobs import ember_v054_router_third_confirmation as third
from jobs import ember_v054_hierarchical_router as hier
from jobs import ember_v054_expanded_router as expanded
from jobs import ember_v054_representation_sweep as sweep
from jobs import ember_v054_split_representation_router as split

OUT=Path("v054-kernel-family-router")
FAMILIES=tuple(hier.FAMILY_LABELS)
GAMMAS=(0.5,1.0,2.0,4.0)
RIDGES=(1e-3,1e-2,1e-1)
FOLDS=4


def unique_cases(*groups):
    out=[]; seen=set()
    for group in groups:
        for c in group:
            if c["id"] not in seen:
                seen.add(c["id"]); out.append(c)
    return out


def family_xy(by_id,cases):
    X=sweep.feature_matrix(by_id,cases,split.FAMILY_REP).double()
    X=F.normalize(X,p=2,dim=1)
    y=torch.tensor([FAMILIES.index(probe.label_of(c)) for c in cases],dtype=torch.long)
    return X,y


def rbf(A,B,gamma):
    d2=torch.cdist(A,B,p=2).pow(2)
    return torch.exp(-float(gamma)*d2)


def onehot(y):
    return F.one_hot(y,num_classes=len(FAMILIES)).double()


def fit_krr(X,y,gamma,ridge):
    K=rbf(X,X,gamma)
    n=K.shape[0]
    alpha=torch.linalg.solve(K+float(ridge)*torch.eye(n,dtype=torch.double),onehot(y))
    return {"X":X,"alpha":alpha,"gamma":float(gamma),"ridge":float(ridge)}


def predict_krr(state,X):
    logits=rbf(X,state["X"],state["gamma"])@state["alpha"]
    pred=logits.argmax(dim=1)
    top2=torch.topk(logits,k=2,dim=1).values
    margin=top2[:,0]-top2[:,1]
    return pred,logits,margin


def stratified_folds(y):
    folds=[[] for _ in range(FOLDS)]
    for cls in range(len(FAMILIES)):
        idx=torch.where(y==cls)[0].tolist()
        for j,i in enumerate(idx): folds[j%FOLDS].append(i)
    return [sorted(x) for x in folds]


def cv_krr(X,y):
    folds=stratified_folds(y); records=[]; best=None
    all_idx=set(range(X.shape[0]))
    for gamma in GAMMAS:
        for ridge in RIDGES:
            correct=0
            for val in folds:
                train=sorted(all_idx-set(val))
                state=fit_krr(X[train],y[train],gamma,ridge)
                pred,_logits,_margin=predict_krr(state,X[val])
                correct+=int((pred==y[val]).sum().item())
            rec={"gamma":float(gamma),"ridge":float(ridge),"correct":correct,"total":int(X.shape[0]),"accuracy":correct/int(X.shape[0])}
            records.append(rec)
            key=(rec["accuracy"],-gamma,-ridge)
            if best is None or key>best[0]: best=(key,rec)
    return best[1],records


def fit_precision(by_id,binary_cases,family_cases):
    base=split.fit_heads(by_id,binary_cases,family_cases)
    X,y=family_xy(by_id,family_cases)
    cv,records=cv_krr(X,y)
    kernel=fit_krr(X,y,cv["gamma"],cv["ridge"])
    return base,kernel,cv,records


def predict(by_id,cases,base,kernel):
    xb=sweep.feature_matrix(by_id,cases,split.BINARY_REP)
    xf_raw=sweep.feature_matrix(by_id,cases,split.FAMILY_REP).double()
    xf=F.normalize(xf_raw,p=2,dim=1)
    bp,_bl,bm=hier.predict(base["binary"],xb)
    fp,_fl,fm=predict_krr(kernel,xf)
    wp,_wl,wm=hier.predict(base["weather_time"],xf_raw)
    labels=[]; margins=[]
    for i in range(len(cases)):
        if int(bp[i])==0:
            labels.append("direct"); margins.append(float(bm[i])); continue
        family=FAMILIES[int(fp[i])]
        if family in {"weather","get_time"}:
            family="weather" if int(wp[i])==0 else "get_time"
            margins.append(float(min(bm[i],wm[i])))
        else:
            margins.append(float(min(bm[i],fm[i])))
        labels.append(family)
    return labels,margins


def eval_suite(by_id,cases,base,kernel):
    labels,margins=predict(by_id,cases,base,kernel)
    pseudo=[{"id":c["id"],"label":probe.label_of(c)} for c in cases]
    return hier.score(labels,pseudo,cases,margins)


def exact(m):
    return m["five_way_correct"]==m["total"] and m["direct_vs_tool_correct"]==m["total"] and m["tool_family_correct"]==m["tool_family_total"]


def evaluate(model,tok,all_train,binary_cases,family_cases):
    dev=unique_cases(held.CASES,confirm.CONFIRM,third.THIRD); combined=unique_cases(all_train,dev)
    rows,reps,_b,_h=probe.extract_representations(model,tok,combined)
    needed=set(split.BINARY_REP+split.FAMILY_REP)
    if not needed.issubset(set(reps)): raise RuntimeError(f"missing reps: {needed} vs {reps}")
    by_id={r["id"]:r for r in rows}
    base,kernel,cv,records=fit_precision(by_id,binary_cases,family_cases)
    suites={"old20":eval_suite(by_id,held.CASES,base,kernel),"confirm50":eval_suite(by_id,confirm.CONFIRM,base,kernel),"third100":eval_suite(by_id,third.THIRD,base,kernel)}
    return base,cv,records,suites


def summary_md(report):
    lines=["# Ember v0.0.54 RBF-kernel family router","","Frozen Ember. Binary=block_04; family=block_02+block_03 RBF kernel ridge.","Sealed finals excluded; hyperparameters training-CV only.","","| Mode | Kernel CV | gamma | ridge | Old20 | Confirm50 | Third100 | Exact170 |","| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for key in ("full","int4"):
        r=report["routers"][key]; s=r["suites"]
        lines.append(f"| {key} | {r['kernel_cv']['accuracy']:.1%} | {r['kernel_cv']['gamma']} | {r['kernel_cv']['ridge']} | {s['old20']['five_way_correct']}/20 | {s['confirm50']['five_way_correct']}/50 | {s['third100']['five_way_correct']}/100 | {r['exact_all_dev']} |")
    lines += ["",f"Strict development pass: **{report['strict_dev_pass']}**","",f"Interpretation: {report['interpretation']}","","No router artifact, checkpoint, integration, or production state changed.",""]
    return "\n".join(lines)


def main():
    token=os.environ.get("HF_TOKEN","").strip()
    if not token: raise RuntimeError("HF_TOKEN required")
    torch.set_num_threads(2); torch.manual_seed(20260910); torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True,exist_ok=True); started=time.monotonic(); all_train,binary_cases,family_cases=expanded.build_training()
    with tempfile.TemporaryDirectory(prefix="ember-kernel-family-") as td:
        work=Path(td); archive=ev.download_verified(ev.PACKAGE_URL,work/"ember.zip",ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package: package.extractall(work/"src")
        import sys; sys.path.insert(0,str(work/"src"/"ember"))
        api=HfApi(token=token); repo=f"{api.whoami()['name']}/{held.MODEL_NAME}"
        fm,ftok,_=held.load_full(repo,work/"full",token); fm.eval(); fb,fcv,frecs,fs=evaluate(fm,ftok,all_train,binary_cases,family_cases)
        im,itok=held.load_int4(repo,work/"int4",token); im.eval(); ib,icv,irecs,ins=evaluate(im,itok,all_train,binary_cases,family_cases)
        fexact=all(exact(x) for x in fs.values()); iexact=all(exact(x) for x in ins.values()); strict=fexact and iexact
        interpretation=("The frozen RBF family router clears all 170 development prompts in full and INT4. Freeze it and run a new untouched confirmation." if strict else "The RBF family router still has established development misses. Do not consume another final set; the remaining error is not solved by this nonlinear kernel class.")
        report={"schema_version":1,"diagnostic":"ember-v054-kernel-family-router-v1","created_at":datetime.now(timezone.utc).isoformat(),"model_repo":repo,"binary_representation":list(split.BINARY_REP),"family_representation":list(split.FAMILY_REP),"sealed_finals_imported":False,"routers":{"full":{"binary_cv":fb['binary_cv'],"weather_time_cv":fb['weather_time_cv'],"kernel_cv":fcv,"kernel_cv_records":frecs,"suites":fs,"exact_all_dev":fexact},"int4":{"binary_cv":ib['binary_cv'],"weather_time_cv":ib['weather_time_cv'],"kernel_cv":icv,"kernel_cv_records":irecs,"suites":ins,"exact_all_dev":iexact}},"strict_dev_pass":strict,"ember_weights_changed":False,"router_integrated":False,"production_changed":False,"interpretation":interpretation,"elapsed_seconds":time.monotonic()-started}
        (OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8"); (OUT/"summary.md").write_text(summary_md(report),encoding="utf-8")
        print(json.dumps({"event":"kernel_family_router_complete","full_cv":fcv,"int4_cv":icv,"full":{k:f"{v['five_way_correct']}/{v['total']}" for k,v in fs.items()},"int4":{k:f"{v['five_way_correct']}/{v['total']}" for k,v in ins.items()},"strict_dev_pass":strict,"interpretation":interpretation,"elapsed_seconds":report['elapsed_seconds']}),flush=True)

if __name__=="__main__": main()
