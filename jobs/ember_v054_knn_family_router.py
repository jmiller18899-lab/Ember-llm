"""Nonlinear frozen family-router sweep for Ember v0.0.54.

Uses only the established expanded training corpus and 170-case development
suites. The sealed failed final suites are not imported or scored. Ember weights
stay frozen.

Architecture held fixed:
  * direct/tool gate: ridge on block_04;
  * family representation: concat(block_02, block_03);
  * weather/get_time specialist: existing train-CV ridge on block_02+block_03.

Only the four-way family classifier is varied. Candidate kNN rules are selected
jointly across full and INT4 using leave-one-out training accuracy only. The
170-case suites are a hard development eligibility gate, never part of the
ranking score. No new final confirmation is consumed here.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, math, os, tempfile, time, zipfile
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

OUT=Path("v054-knn-family-router")
FAMILIES=tuple(hier.FAMILY_LABELS)
KS=(1,3,5,7,9,15,21,31)
METHODS=tuple((metric,weighted,k) for metric in ("cosine","z_euclidean") for weighted in (False,True) for k in KS)


def unique_cases(*groups):
    out=[]; seen=set()
    for group in groups:
        for c in group:
            if c["id"] not in seen:
                seen.add(c["id"]); out.append(c)
    return out


def family_train(rows_by_id,family_cases):
    X=sweep.feature_matrix(rows_by_id,family_cases,split.FAMILY_REP).double()
    y=torch.tensor([FAMILIES.index(probe.label_of(c)) for c in family_cases],dtype=torch.long)
    return X,y


def transform_train(X,metric):
    if metric=="cosine":
        return F.normalize(X,p=2,dim=1), {"metric":"cosine"}
    if metric=="z_euclidean":
        mean=X.mean(dim=0); std=X.std(dim=0,unbiased=False).clamp_min(1e-8)
        return (X-mean)/std, {"metric":"z_euclidean","mean":mean,"std":std}
    raise ValueError(metric)


def transform_test(X,state):
    if state["metric"]=="cosine": return F.normalize(X.double(),p=2,dim=1)
    return (X.double()-state["mean"])/state["std"]


def neighbor_values(Q,T,metric):
    if metric=="cosine": return Q@T.T, True
    return torch.cdist(Q,T,p=2), False


def vote(neighbor_labels,neighbor_vals,weighted,higher):
    scores=torch.zeros((neighbor_labels.shape[0],len(FAMILIES)),dtype=torch.double)
    if weighted:
        if higher:
            weights=(neighbor_vals-neighbor_vals.min(dim=1,keepdim=True).values+1e-6)
        else:
            weights=1.0/(neighbor_vals+1e-6)
    else:
        weights=torch.ones_like(neighbor_vals,dtype=torch.double)
    for c in range(len(FAMILIES)):
        scores[:,c]=(weights*(neighbor_labels==c)).sum(dim=1)
    pred=scores.argmax(dim=1)
    top2=torch.topk(scores,k=2,dim=1).values
    margin=top2[:,0]-top2[:,1]
    return pred,margin


def loo_cv(X,y,method):
    metric,weighted,k=method
    T,state=transform_train(X,metric)
    vals,higher=neighbor_values(T,T,metric)
    n=T.shape[0]
    diag=torch.arange(n)
    vals[diag,diag]=-float("inf") if higher else float("inf")
    idx=torch.topk(vals,k=k,dim=1,largest=higher).indices
    nv=vals.gather(1,idx); nl=y[idx]
    pred,_=vote(nl,nv,weighted,higher)
    correct=int((pred==y).sum().item())
    return {"correct":correct,"total":n,"accuracy":correct/n,"metric":metric,"weighted":weighted,"k":k},state


def fit_knn(X,y,method):
    metric,weighted,k=method
    T,state=transform_train(X,metric)
    return {"T":T,"y":y.clone(),"metric":metric,"weighted":weighted,"k":k,"transform":state}


def knn_predict(model,X):
    Q=transform_test(X,model["transform"])
    vals,higher=neighbor_values(Q,model["T"],model["metric"])
    idx=torch.topk(vals,k=model["k"],dim=1,largest=higher).indices
    nv=vals.gather(1,idx); nl=model["y"][idx]
    return vote(nl,nv,model["weighted"],higher)


def fit_precision(rows_by_id,binary_cases,family_cases):
    base=split.fit_heads(rows_by_id,binary_cases,family_cases)
    X,y=family_train(rows_by_id,family_cases)
    candidates={}
    for method in METHODS:
        cv,_=loo_cv(X,y,method)
        key=f"{method[0]}__{'weighted' if method[1] else 'plain'}__k{method[2]}"
        candidates[key]={"method":method,"cv":cv,"model":fit_knn(X,y,method)}
    return base,candidates


def predict(rows_by_id,cases,base,family_model):
    xb=sweep.feature_matrix(rows_by_id,cases,split.BINARY_REP)
    xf=sweep.feature_matrix(rows_by_id,cases,split.FAMILY_REP)
    bp,_bl,bm=hier.predict(base["binary"],xb)
    fp,fm=knn_predict(family_model,xf)
    wp,_wl,wm=hier.predict(base["weather_time"],xf)
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


def eval_suite(rows_by_id,cases,base,family_model):
    labels,margins=predict(rows_by_id,cases,base,family_model)
    pseudo=[{"id":c["id"],"label":probe.label_of(c)} for c in cases]
    return hier.score(labels,pseudo,cases,margins)


def exact(m):
    return m["five_way_correct"]==m["total"] and m["direct_vs_tool_correct"]==m["total"] and m["tool_family_correct"]==m["tool_family_total"]


def prepare_precision(model,tok,all_train,dev,binary_cases,family_cases):
    combined=unique_cases(all_train,dev)
    rows,reps,_b,_h=probe.extract_representations(model,tok,combined)
    needed=set(split.BINARY_REP+split.FAMILY_REP)
    if not needed.issubset(set(reps)): raise RuntimeError(f"missing reps {needed} from {reps}")
    by_id={r["id"]:r for r in rows}
    base,candidates=fit_precision(by_id,binary_cases,family_cases)
    return by_id,base,candidates


def score_dev(by_id,base,cand):
    suites={"old20":eval_suite(by_id,held.CASES,base,cand["model"]),"confirm50":eval_suite(by_id,confirm.CONFIRM,base,cand["model"]),"third100":eval_suite(by_id,third.THIRD,base,cand["model"])}
    return suites,all(exact(x) for x in suites.values())


def summary_md(report):
    lines=["# Ember v0.0.54 nonlinear kNN family-router sweep","","Frozen Ember; binary=block_04, family=block_02+block_03. Sealed finals excluded.","Candidate family rules selected by leave-one-out training CV only.","",f"Selected method: **{report.get('selected_method')}**",f"Strict development pass: **{report['strict_dev_pass']}**","",f"Interpretation: {report['interpretation']}","","No router artifact, checkpoint, integration, or production state changed.",""]
    return "\n".join(lines)


def main():
    token=os.environ.get("HF_TOKEN","").strip()
    if not token: raise RuntimeError("HF_TOKEN required")
    torch.set_num_threads(2); torch.manual_seed(20260910); torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True,exist_ok=True); started=time.monotonic()
    all_train,binary_cases,family_cases=expanded.build_training(); dev=unique_cases(held.CASES,confirm.CONFIRM,third.THIRD)
    with tempfile.TemporaryDirectory(prefix="ember-knn-family-") as td:
        work=Path(td); archive=ev.download_verified(ev.PACKAGE_URL,work/"ember.zip",ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package: package.extractall(work/"src")
        import sys; sys.path.insert(0,str(work/"src"/"ember"))
        api=HfApi(token=token); repo=f"{api.whoami()['name']}/{held.MODEL_NAME}"
        fm,ftok,_=held.load_full(repo,work/"full",token); fm.eval(); fby,fb,fc=prepare_precision(fm,ftok,all_train,dev,binary_cases,family_cases)
        im,itok=held.load_int4(repo,work/"int4",token); im.eval(); iby,ib,ic=prepare_precision(im,itok,all_train,dev,binary_cases,family_cases)
        common=sorted(set(fc).intersection(ic))
        ranked=[]; joint={}
        for key in common:
            fs,fok=score_dev(fby,fb,fc[key]); ins,iok=score_dev(iby,ib,ic[key])
            train_floor=min(fc[key]["cv"]["accuracy"],ic[key]["cv"]["accuracy"])
            train_mean=(fc[key]["cv"]["accuracy"]+ic[key]["cv"]["accuracy"])/2
            eligible=fok and iok
            joint[key]={"full_cv":fc[key]["cv"],"int4_cv":ic[key]["cv"],"train_floor":train_floor,"train_mean":train_mean,"full_suites":fs,"int4_suites":ins,"eligible":eligible}
            if eligible: ranked.append(((train_floor,train_mean),key))
            print(json.dumps({"event":"knn_family_candidate","method":key,"full_cv":fc[key]["cv"]["accuracy"],"int4_cv":ic[key]["cv"]["accuracy"],"full_dev":[fs['old20']['five_way_correct'],fs['confirm50']['five_way_correct'],fs['third100']['five_way_correct']],"int4_dev":[ins['old20']['five_way_correct'],ins['confirm50']['five_way_correct'],ins['third100']['five_way_correct']],"eligible":eligible}),flush=True)
        ranked.sort(reverse=True)
        selected=ranked[0][1] if ranked else None; strict=selected is not None
        interpretation=("A nonlinear frozen family rule clears all 170 development prompts in both precisions and was selected by training-only CV. Freeze it and run a new untouched confirmation." if strict else "No tested nonlinear frozen family rule clears development in both precisions. Do not consume another final set; reconsider representation or router model class.")
        report={"schema_version":1,"diagnostic":"ember-v054-knn-family-router-v1","created_at":datetime.now(timezone.utc).isoformat(),"model_repo":repo,"binary_representation":list(split.BINARY_REP),"family_representation":list(split.FAMILY_REP),"methods":list(joint.keys()),"joint":joint,"selected_method":selected,"strict_dev_pass":strict,"failed_finals_imported":False,"ember_weights_changed":False,"router_integrated":False,"production_changed":False,"interpretation":interpretation,"elapsed_seconds":time.monotonic()-started}
        (OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8"); (OUT/"summary.md").write_text(summary_md(report),encoding="utf-8")
        print(json.dumps({"event":"knn_family_router_complete","selected_method":selected,"strict_dev_pass":strict,"top_eligible":[{"method":k,"score":list(s)} for s,k in ranked[:8]],"top_training":[{"method":k,"floor":v['train_floor'],"mean":v['train_mean'],"eligible":v['eligible']} for k,v in sorted(joint.items(),key=lambda kv:(kv[1]['train_floor'],kv[1]['train_mean']),reverse=True)[:8]],"interpretation":interpretation,"elapsed_seconds":report['elapsed_seconds']}),flush=True)

if __name__=="__main__": main()
