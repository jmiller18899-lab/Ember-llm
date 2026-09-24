# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","accelerate==1.15.0","huggingface-hub==1.31.0"]
# ///
"""Evaluation only: frozen policy (v1) vs v1 + scoped context rule (v2), same Repair2 weights.

v1 = arithmetic tool if it fires, else Repair2 with system prompt + scoped temporal/drafting rules (the frozen baseline).
v2 = v1, plus CONTEXT_RULE appended to the system prompt when ember_context_rule.context_gate fires.
Every prompt is generated under v1. Prompts where the context gate fires are generated a second time under v2;
elsewhere v2 is identical to v1 by construction. Suites: original 72 exact (corrected benchmark), temporal 8,
drafting 8, promotion v2 200, time held-out v1 180, and 24 fresh record questions (jobs/ember_context_holdout.py).
Usage:  python jobs/ember_4b_policy_v2_context_eval.py --preflight   (CPU: pinned sources + gate audit)
        uv run jobs/ember_4b_policy_v2_context_eval.py                (GPU)
"""
import argparse,hashlib,json,os,types,urllib.request
from pathlib import Path

SRC_COMMIT="9640efe59a1b41cab32504866fb27e991661832f"
RAW=f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{SRC_COMMIT}/"
PINNED={
 "jobs/ember_arith_tool.py":"d4dd1d38f4e34d3f7f3262d2bb1d3bfff2aecfc6d5af23b02ae39fd46d31bac1",
 "jobs/ember_context_rule.py":"239a1d29e55be77d1f46867dc8cc6d94b82e39823b33f9ffd8ab1b56ffc8cb70",
 "jobs/ember_context_holdout.py":"17cb850e6ecf54fa88ee90acff6368ae0f962a07a6447fae6af41b6425d72a2b",
 "jobs/ember_time_heldout_v1.py":"2023870c819feceb84fa9635fb00f2728bc2c81e2099759015bf605b0e4af65a",
 "jobs/ember_4b_promotion_v2_frozen_eval.py":"d78917ed19a5d8160b632b6c553f5b0d9ff62b8bc1ba49700a35c70d023907b7",
 "jobs/ember_4b_repair2_original_promotion_eval.py":"48272629067857fc873dfab6c71c61e82c90eb4bf5cab7d85a7f5c9108b29ffc",
}
MODEL="Jmiller18899/ember-qwen3.5-4b-repair2"; MODEL_REV="daf938bba5d4e6b650ec9d34a2d3ac56706cf549"
BENCH="Jmiller18899/ember-generalization-benchmark-v1"; BENCH_REV="835243e16cf19e6ee34f27a19cba14242c14969f"
FROZEN={"exact":72,"temporal":8,"drafting":7,"promo":197,"heldout_strict":180}   # Actions run 35952086798

def fetch(rel):
    local=Path(__file__).resolve().parents[1]/rel
    data=local.read_bytes() if local.exists() else urllib.request.urlopen(RAW+rel,timeout=60).read()
    got=hashlib.sha256(data).hexdigest()
    if got!=PINNED[rel]: raise RuntimeError(f"{rel}: sha256 {got} != pinned {PINNED[rel]}")
    return data
def module(rel,name):
    m=types.ModuleType(name); exec(compile(fetch(rel).decode(),rel,"exec"),m.__dict__); return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--preflight",action="store_true"); a=ap.parse_args()
    T=module("jobs/ember_arith_tool.py","arith"); C=module("jobs/ember_context_rule.py","ctx"); HO=module("jobs/ember_context_holdout.py","ctxho")
    H=module("jobs/ember_time_heldout_v1.py","heldout"); P=module("jobs/ember_4b_promotion_v2_frozen_eval.py","promo")
    O=module("jobs/ember_4b_repair2_original_promotion_eval.py","rules")
    promo,held,ho=P.build(),H.build(),HO.build()
    audit={"promo_fired_families":sorted({r["family"] for r in promo if C.context_gate(r["prompt"])}),
           "promo_fired":sum(C.context_gate(r["prompt"]) for r in promo),"heldout_fired":sum(C.context_gate(r["prompt"]) for r in held),
           "temporal_drafting_fired":sum(C.context_gate(c["prompt"]) for c in O.TEMP+O.DRAFT),"holdout_fired":sum(C.context_gate(r["prompt"]) for r in ho),
           "tool_fires_on_context":sum(T.solve(r["prompt"]) is not None for r in ho+[r for r in promo if r["family"]=="context_consistency"])}
    ok=(audit["promo_fired_families"]==["context_consistency"] and audit["promo_fired"]==15 and not audit["heldout_fired"]
        and not audit["temporal_drafting_fired"] and audit["holdout_fired"]==len(ho) and not audit["tool_fires_on_context"])
    if not ok: raise RuntimeError(f"gate audit failed: {audit}")
    if a.preflight: print("POLICY_V2_PREFLIGHT_PASS "+json.dumps(audit),flush=True); return

    import torch
    from huggingface_hub import hf_hub_download
    from peft import PeftModel
    from transformers import AutoTokenizer,Qwen3_5ForCausalLM
    token=os.environ["HF_TOKEN"]
    bench=json.loads(Path(hf_hub_download(BENCH,"candidate.json",revision=BENCH_REV,token=token)).read_text())
    exact=[r for r in bench if r.get("scoring")=="exact"]; assert len(exact)==72
    leak={"temporal":[r["id"] for r in exact if O.temporal_gate(r["prompt"])],"drafting":[r["id"] for r in exact if O.drafting_gate(r["prompt"])],
          "context":[r["id"] for r in bench if C.context_gate(r["prompt"])]}
    print("POLICY_V2_AUDIT "+json.dumps({**audit,"rule_leakage_on_benchmark":leak}),flush=True)

    tok=AutoTokenizer.from_pretrained(P.BASE,revision=P.BASE_REV); tok.pad_token=tok.eos_token
    base,li=Qwen3_5ForCausalLM.from_pretrained(P.BASE,revision=P.BASE_REV,dtype=torch.bfloat16,device_map={"":0},output_loading_info=True,key_mapping={r"^model.language_model\.":"model."})
    if li["missing_keys"] or li.get("mismatched_keys") or li.get("error_msgs"): raise RuntimeError("base load mismatch")
    model=PeftModel.from_pretrained(base,MODEL,revision=MODEL_REV).eval()
    def gen(system,p):
        ids=tok.apply_chat_template([{"role":"system","content":system},{"role":"user","content":p}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt",return_dict=False).to(model.device)
        with torch.inference_mode(): z=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
        return tok.decode(z[0,ids.shape[-1]:],skip_special_tokens=True).strip()
    def both(p):
        if (s:=T.solve(p)): return s[1],s[1]
        v1=gen(O.system_for(p),p)
        v2=gen(O.system_for(p)+" "+C.CONTEXT_RULE,p) if C.context_gate(p) else v1
        return v1,v2

    names=("v1_frozen","v2_context"); res={n:{} for n in names}; changed=[]
    ex={}
    for r in exact:
        o=both(r["prompt"]); ex[r["id"]]=[x==r["answer"] for x in o]
        if o[0]!=o[1]: changed.append({"suite":"exact72","id":r["id"],"v1":o[0],"v2":o[1]})
    for i,n in enumerate(names): res[n]["exact"]=sum(v[i] for v in ex.values())
    regress=[k for k,v in ex.items() if v[0] and not v[1]]
    for suite,cases,fn in (("temporal",O.TEMP,O.temp_pass),("drafting",O.DRAFT,O.draft_pass)):
        s=[0,0]
        for c in cases:
            o=both(c["prompt"]); s[0]+=fn(c,o[0]); s[1]+=fn(c,o[1])
            if o[0]!=o[1]: changed.append({"suite":suite,"id":c["id"],"v1":o[0],"v2":o[1]})
        for i,n in enumerate(names): res[n][suite]=s[i]
    fam={n:{} for n in names}; ctx_rows=[]
    for r in promo:
        o=both(r["prompt"]); oks=[(x==r["answer"]) if r["scoring"]=="exact" else P.rubric_pass(r,x) for x in o]
        for n,v in zip(names,oks): f=fam[n].setdefault(r["family"],[0,0]); f[0]+=int(v); f[1]+=1
        if r["family"]=="context_consistency": ctx_rows.append({"id":r["id"],"v1":o[0],"v1_pass":oks[0],"v2":o[1],"v2_pass":oks[1]})
        elif o[0]!=o[1]: changed.append({"suite":"promo_v2","id":r["id"],"v1":o[0],"v2":o[1]})
    for n in names: res[n]["promo"]=sum(v[0] for v in fam[n].values())
    hs={n:[] for n in names}
    for r in held:
        o=both(r["prompt"])
        for n,x in zip(names,o): hs[n].append({**r,**H.score(r,x)})
        if o[0]!=o[1]: changed.append({"suite":"heldout","id":r["id"],"v1":o[0],"v2":o[1]})
    for n in names: res[n]["heldout_strict"]=H.summarize(hs[n])["overall"]["strict"]
    ho_rows=[]
    for r in ho:
        o=both(r["prompt"]); ho_rows.append({"id":r["id"],"v1":o[0],"v1_pass":HO.score(r,o[0]),"v2":o[1],"v2_pass":HO.score(r,o[1])})
    for i,n in enumerate(names): res[n]["context_holdout"]=sum(x[f"{n[:2]}_pass"] for x in ho_rows)
    for x in ctx_rows+ho_rows: print("POLICY_V2_CONTEXT "+json.dumps(x),flush=True)

    v1,v2=res["v1_frozen"],res["v2_context"]; f1,f2=fam["v1_frozen"],fam["v2_context"]
    checks={"frozen_reproduced":{k:v1[k] for k in FROZEN}==FROZEN,"exact_72_of_72":v2["exact"]==72,"no_exact_regressions":not regress,
            "temporal_8_of_8":v2["temporal"]==8,"drafting_at_least_7":v2["drafting"]>=7,"promo_at_least_frozen":v2["promo"]>=FROZEN["promo"],
            **{f"promo_{f}_not_lower":f2[f][0]>=f1[f][0] for f in f1},"context_consistency_15_of_15":f2["context_consistency"][0]==15,
            "heldout_not_lower":v2["heldout_strict"]>=v1["heldout_strict"],"context_holdout_not_lower":v2["context_holdout"]>=v1["context_holdout"],
            "no_change_outside_context_prompts":not changed,"no_rule_leakage":not any(leak.values())}
    summary={"model":f"{MODEL}@{MODEL_REV}","bench":f"{BENCH}@{BENCH_REV}","src":SRC_COMMIT,"v1_frozen":v1,"v2_context":v2,
             "promo_by_family":{"v1_frozen":f1,"v2_context":f2},"context_holdout_n":len(ho),"exact_regressions":regress,
             "changed_outside_context":changed[:10],"checks":checks,"passed":all(checks.values()),"weights_changed":False,"production_ready":False}
    print("POLICY_V2_SUMMARY "+json.dumps(summary),flush=True)
if __name__=="__main__": main()
