# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","accelerate==1.15.0","huggingface-hub==1.31.0"]
# ///
"""Evaluation only: Repair2 (weights unchanged) + scoped rules + the deterministic arithmetic tool.

Runtime policy under test: if jobs/ember_arith_tool.solve(prompt) fires, its exact answer is returned;
otherwise Repair2 answers with the promotion system prompt and scoped temporal/drafting rules.
The model is generated once per prompt; "repair2" scores that output alone and "repair2+tool" substitutes
the tool's answer where it fires. Both configurations therefore come from one run with identical model outputs.

Suites: original 72 exact (benchmark v1 at the corrected revision), temporal 8, drafting 8, promotion v2 200,
time held-out v1 180. Nothing is trained or uploaded.
Usage:  python jobs/ember_4b_repair2_arith_tool_eval.py --preflight   (CPU: pinned sources + tool on local suites)
        uv run jobs/ember_4b_repair2_arith_tool_eval.py                (GPU)
"""
import argparse,hashlib,json,os,types,urllib.request
from pathlib import Path

SRC_COMMIT="aa69888c0b6a90bf9b527c14dc96b6008c669ebb"
RAW=f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{SRC_COMMIT}/"
PINNED={
 "jobs/ember_arith_tool.py":"d4dd1d38f4e34d3f7f3262d2bb1d3bfff2aecfc6d5af23b02ae39fd46d31bac1",
 "jobs/ember_time_heldout_v1.py":"2023870c819feceb84fa9635fb00f2728bc2c81e2099759015bf605b0e4af65a",
 "jobs/ember_4b_promotion_v2_frozen_eval.py":"d78917ed19a5d8160b632b6c553f5b0d9ff62b8bc1ba49700a35c70d023907b7",
 "jobs/ember_4b_repair2_original_promotion_eval.py":"48272629067857fc873dfab6c71c61e82c90eb4bf5cab7d85a7f5c9108b29ffc",
}
MODEL="Jmiller18899/ember-qwen3.5-4b-repair2"; MODEL_REV="daf938bba5d4e6b650ec9d34a2d3ac56706cf549"
BENCH="Jmiller18899/ember-generalization-benchmark-v1"; BENCH_REV="835243e16cf19e6ee34f27a19cba14242c14969f"
REPAIR2={"exact":71,"temporal":8,"drafting":7,"promo":178,"heldout_strict":148}   # without the tool, corrected benchmark
ARITH_FAMILIES={"arithmetic","time_reasoning"}

def fetch(rel):
    local=Path(__file__).resolve().parents[1]/rel
    data=local.read_bytes() if local.exists() else urllib.request.urlopen(RAW+rel,timeout=60).read()
    got=hashlib.sha256(data).hexdigest()
    if got!=PINNED[rel]: raise RuntimeError(f"{rel}: sha256 {got} != pinned {PINNED[rel]}")
    return data
def module(rel,name):
    m=types.ModuleType(name); exec(compile(fetch(rel).decode(),rel,"exec"),m.__dict__); return m

def tool_audit(T,rows,family_of,answer_of):
    """Tool behaviour on a suite, independent of the model: fires, wrong-when-fired, fires outside arithmetic."""
    fired=wrong=0; outside=[]; wrong_ids=[]
    for r in rows:
        s=T.solve(r["prompt"])
        if s is None: continue
        fired+=1
        if family_of(r) not in ARITH_FAMILIES: outside.append(r["id"])
        elif (a:=answer_of(r)) is not None and s[1]!=a: wrong+=1; wrong_ids.append(r["id"])
    return {"fired":fired,"wrong":wrong,"wrong_ids":wrong_ids,"fired_outside_arithmetic":outside}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--preflight",action="store_true"); a=ap.parse_args()
    T=module("jobs/ember_arith_tool.py","arith"); H=module("jobs/ember_time_heldout_v1.py","heldout")
    P=module("jobs/ember_4b_promotion_v2_frozen_eval.py","promo"); O=module("jobs/ember_4b_repair2_original_promotion_eval.py","original")
    assert P.SYSTEM==O.SYSTEM and P.TEMPORAL_RULE==O.TEMPORAL_RULE and P.DRAFT_RULE==O.DRAFT_RULE
    promo,held=P.build(),H.build()
    local={"promo_v2":tool_audit(T,promo,lambda r:r["family"],lambda r:r.get("answer") if r["scoring"]=="exact" else None),
           "heldout_v1":tool_audit(T,held,lambda r:"time_reasoning",lambda r:r["answer"]),
           "temporal8":tool_audit(T,O.TEMP,lambda r:"temporal",lambda r:None),
           "drafting8":tool_audit(T,O.DRAFT,lambda r:"drafting",lambda r:None)}
    bad={k:v for k,v in local.items() if v["wrong"] or v["fired_outside_arithmetic"]} | ({"temporal/drafting":"fired"} if local["temporal8"]["fired"] or local["drafting8"]["fired"] else {})
    if bad: raise RuntimeError(f"tool audit failed: {bad}")
    if a.preflight: print("ARITH_TOOL_PREFLIGHT_PASS "+json.dumps(local),flush=True); return

    import torch
    from huggingface_hub import HfApi,hf_hub_download
    from peft import PeftModel
    from transformers import AutoTokenizer,Qwen3_5ForCausalLM
    token=os.environ["HF_TOKEN"]; api=HfApi(token=token)
    bench=json.loads(Path(hf_hub_download(BENCH,"candidate.json",revision=BENCH_REV,token=token)).read_text())
    exact=[r for r in bench if r.get("scoring")=="exact"]; assert len(exact)==72
    local["bench72"]=tool_audit(T,exact,lambda r:r["family"],lambda r:r["answer"])
    rubric_bench=[r for r in bench if r.get("scoring")!="exact"]
    local["bench_rubric48"]=tool_audit(T,rubric_bench,lambda r:r["family"],lambda r:None)
    if local["bench72"]["wrong"] or local["bench_rubric48"]["fired"] or local["bench72"]["fired_outside_arithmetic"]: raise RuntimeError(f"tool audit failed on benchmark: {local}")
    leak=[r["id"] for r in exact if O.system_for(r["prompt"])!=O.SYSTEM]
    print("ARITH_TOOL_AUDIT "+json.dumps({**local,"rule_leakage_on_exact":leak}),flush=True)

    tok=AutoTokenizer.from_pretrained(P.BASE,revision=P.BASE_REV); tok.pad_token=tok.eos_token
    base,li=Qwen3_5ForCausalLM.from_pretrained(P.BASE,revision=P.BASE_REV,dtype=torch.bfloat16,device_map={"":0},output_loading_info=True,key_mapping={r"^model.language_model\.":"model."})
    if li["missing_keys"] or li.get("mismatched_keys") or li.get("error_msgs"): raise RuntimeError("base load mismatch")
    model=PeftModel.from_pretrained(base,MODEL,revision=MODEL_REV).eval()
    def gen(p):
        ids=tok.apply_chat_template([{"role":"system","content":O.system_for(p)},{"role":"user","content":p}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt",return_dict=False).to(model.device)
        with torch.inference_mode(): z=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
        return tok.decode(z[0,ids.shape[-1]:],skip_special_tokens=True).strip()
    def both(p):
        m=gen(p); s=T.solve(p)
        return {"model":m,"tool":s[1] if s else None,"kind":s[0] if s else None,"policy":s[1] if s else m}

    res={"repair2":{},"repair2+tool":{}}; changes=[]
    def record(suite,rid,o,ok_model,ok_policy):
        if o["tool"] is not None and o["tool"]!=o["model"]:
            changes.append({"suite":suite,"id":rid,"kind":o["kind"],"model":o["model"],"tool":o["tool"],"model_ok":ok_model,"policy_ok":ok_policy})
    # original exact
    ex={}
    for r in exact:
        o=both(r["prompt"]); m_ok,p_ok=o["model"]==r["answer"],o["policy"]==r["answer"]; ex[r["id"]]=(m_ok,p_ok); record("exact72",r["id"],o,m_ok,p_ok)
    for i,name in enumerate(res): res[name]["exact"]=sum(v[i] for v in ex.values())
    exact_regressions=[k for k,(m,p) in ex.items() if m and not p]
    # temporal / drafting (tool is expected never to fire here)
    for suite,cases,passfn in (("temporal",O.TEMP,O.temp_pass),("drafting",O.DRAFT,O.draft_pass)):
        s=[0,0]
        for c in cases:
            o=both(c["prompt"]); m_ok,p_ok=passfn(c,o["model"]),passfn(c,o["policy"]); s[0]+=m_ok; s[1]+=p_ok; record(suite,c["id"],o,m_ok,p_ok)
        res["repair2"][suite],res["repair2+tool"][suite]=s
    # promotion v2
    fam={"repair2":{},"repair2+tool":{}}
    for r in promo:
        o=both(r["prompt"])
        ok=[(x==r["answer"]) if r["scoring"]=="exact" else P.rubric_pass(r,x) for x in (o["model"],o["policy"])]
        for name,v in zip(fam,ok): f=fam[name].setdefault(r["family"],[0,0]); f[0]+=int(v); f[1]+=1
        record("promo_v2",r["id"],o,*ok)
    for name in fam: res[name]["promo"]=sum(v[0] for v in fam[name].values()); res[name]["promo_by_family"]=fam[name]
    # time held-out
    hs={"repair2":[],"repair2+tool":[]}
    for r in held:
        o=both(r["prompt"]); sc=[H.score(r,x) for x in (o["model"],o["policy"])]
        for name,v in zip(hs,sc): hs[name].append({**r,**v})
        record("heldout_v1",r["id"],o,sc[0]["strict"],sc[1]["strict"])
    for name in hs:
        s=H.summarize(hs[name]); res[name]["heldout_strict"]=s["overall"]["strict"]; res[name]["heldout_content"]=s["overall"]["content"]
        res[name]["heldout_by_slice"]={k:v["strict"] for k,v in s["by_slice"].items()}

    r2,rt=res["repair2"],res["repair2+tool"]
    reproduced={k:r2[k] for k in REPAIR2}
    fb,fa=r2["promo_by_family"],rt["promo_by_family"]
    checks={"exact_72_of_72":rt["exact"]==72,"no_exact_regressions":not exact_regressions,"temporal_8_of_8":rt["temporal"]==8,
            "drafting_at_least_7":rt["drafting"]>=7,"promo_at_least_178":rt["promo"]>=178,
            **{f"promo_{f}_not_lower":fa[f][0]>=fb[f][0] for f in fb},
            "heldout_strict_not_lower":rt["heldout_strict"]>=r2["heldout_strict"],
            "tool_never_worsens_an_output":not any(c["model_ok"] and not c["policy_ok"] for c in changes),
            "tool_silent_on_temporal_drafting":not any(c["suite"] in ("temporal","drafting") for c in changes),
            "no_rule_leakage":not leak}
    print("ARITH_TOOL_CHANGES "+json.dumps({"changed":len(changes),"fixed":sum(not c["model_ok"] and c["policy_ok"] for c in changes),
          "worsened":[c for c in changes if c["model_ok"] and not c["policy_ok"]],"examples":[c for c in changes if not c["model_ok"]][:12]}),flush=True)
    summary={"model":f"{MODEL}@{MODEL_REV}","bench":f"{BENCH}@{BENCH_REV}","tool":f"jobs/ember_arith_tool.py@{SRC_COMMIT}",
             "repair2":{k:v for k,v in r2.items() if k!="promo_by_family"},"repair2+tool":{k:v for k,v in rt.items() if k!="promo_by_family"},
             "promo_by_family":{"repair2":fb,"repair2+tool":fa},"repair2_reproduced":reproduced==REPAIR2,"reproduced":reproduced,
             "exact_regressions":exact_regressions,"checks":checks,"passed":all(checks.values()),"weights_changed":False,"production_ready":False}
    print("ARITH_TOOL_SUMMARY "+json.dumps(summary),flush=True)
if __name__=="__main__": main()
