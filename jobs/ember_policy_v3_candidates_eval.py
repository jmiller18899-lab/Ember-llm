# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","accelerate==1.15.0","huggingface-hub==1.31.0"]
# ///
"""Evaluation only: policy v3 candidates vs frozen policy v2, all on Repair2 weights.

Components (each optional; v2 = none of them):
  tool3    jobs/ember_arith_tool_v3.py instead of the v2 tool (label numbers masked; leave/quit; goes into/needs)
  draft3   drafting gate widened to "Text/Tell X that ...", "Let X know", "Write a note to X" (same drafting rule text)
  ctx3     context rule plus the "whose item" sentence (same context gate)
  missing  missing-text rule for transformation requests that include no text
Configurations: v2, full (all four), and full minus each component. Outputs are generated once per distinct
(system prompt, user prompt) pair and shared across configurations. Tool answers need no generation.

Suites: original 72 exact (benchmark v1 @835243e1), temporal 8, drafting 8, promotion v2 200, time held-out 180,
context held-out 24, promotion suite v3 192 (development: the fixes were designed on its failures), and 36 fresh probes.
Gate for a candidate: every v2 freeze check still holds; suite v3 above v2's score with no family lower; probes not
lower than v2; the tool never wrong when it fires; no new gate firing on the 72 exact cases.
The best passing configuration is reported. Nothing is frozen or uploaded by this job.
Usage:  python jobs/ember_policy_v3_candidates_eval.py --preflight   (CPU)
        uv run jobs/ember_policy_v3_candidates_eval.py                (GPU)
"""
import argparse,hashlib,json,os,types,urllib.request
from pathlib import Path

SRC_COMMIT="76f2b4c900fc550171f63c32f3382f8ad5916d45"
RAW=f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{SRC_COMMIT}/"
PINNED={
 "jobs/ember_arith_tool.py":"d4dd1d38f4e34d3f7f3262d2bb1d3bfff2aecfc6d5af23b02ae39fd46d31bac1",
 "jobs/ember_4b_repair2_original_promotion_eval.py":"48272629067857fc873dfab6c71c61e82c90eb4bf5cab7d85a7f5c9108b29ffc",
 "jobs/ember_context_rule.py":"239a1d29e55be77d1f46867dc8cc6d94b82e39823b33f9ffd8ab1b56ffc8cb70",
 "jobs/ember_context_holdout.py":"17cb850e6ecf54fa88ee90acff6368ae0f962a07a6447fae6af41b6425d72a2b",
 "jobs/ember_time_heldout_v1.py":"2023870c819feceb84fa9635fb00f2728bc2c81e2099759015bf605b0e4af65a",
 "jobs/ember_4b_promotion_v2_frozen_eval.py":"d78917ed19a5d8160b632b6c553f5b0d9ff62b8bc1ba49700a35c70d023907b7",
 "jobs/ember_arith_tool_v3.py":"d29beee78404367401e91b3cd1e34b38597204d95582e4b20ea643720f99900b",
 "jobs/ember_policy_v3_rules.py":"114df189683fcc4b976421f84f586145f25434782ba8636044081085119fcb7c",
 "jobs/ember_promotion_suite_v3.py":"a31e14acae58594cfca60e3de28c19cf567a00d5fdc57fa137f899695e6ca87f",
 "jobs/ember_policy_v3_probes.py":"d0da0960a22fee1a06d487179a1fd18b3d24e756e0eb9a60ee49458ad32fe3d1",
}
MODEL="Jmiller18899/ember-qwen3.5-4b-repair2"; MODEL_REV="daf938bba5d4e6b650ec9d34a2d3ac56706cf549"
BENCH="Jmiller18899/ember-generalization-benchmark-v1"; BENCH_REV="835243e16cf19e6ee34f27a19cba14242c14969f"
BASE,BASE_REV="Qwen/Qwen3.5-4B","851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
COMPONENTS=("tool3","draft3","ctx3","missing")
CONFIGS={"v2":set(),"full":set(COMPONENTS),**{f"full-minus-{c}":set(COMPONENTS)-{c} for c in COMPONENTS}}
V2_FROZEN={"exact":72,"temporal":8,"drafting":7,"promo":200,"heldout":180,"ctx_holdout":20}

def module(rel,name):
    local=Path(__file__).resolve().parents[1]/rel
    data=local.read_bytes() if local.exists() else urllib.request.urlopen(RAW+rel,timeout=60).read()
    got=hashlib.sha256(data).hexdigest()
    if got!=PINNED[rel]: raise RuntimeError(f"{rel}: sha256 {got} != pinned {PINNED[rel]}")
    m=types.ModuleType(name); exec(compile(data.decode(),rel,"exec"),m.__dict__); return m

def load():
    M=types.SimpleNamespace()
    M.T2=module("jobs/ember_arith_tool.py","t2"); M.T3=module("jobs/ember_arith_tool_v3.py","t3")
    M.O=module("jobs/ember_4b_repair2_original_promotion_eval.py","rules"); M.C=module("jobs/ember_context_rule.py","ctx")
    M.R=module("jobs/ember_policy_v3_rules.py","r3"); M.CH=module("jobs/ember_context_holdout.py","ctxho")
    M.H=module("jobs/ember_time_heldout_v1.py","held"); M.P=module("jobs/ember_4b_promotion_v2_frozen_eval.py","promo")
    M.V3=module("jobs/ember_promotion_suite_v3.py","v3"); M.PR=module("jobs/ember_policy_v3_probes.py","probes")
    return M

def policy(M,comps):
    """Returns route(prompt) -> ("tool", answer) or ("model", system_prompt)."""
    tool=M.T3 if "tool3" in comps else M.T2
    ctx_rule=M.R.context_rule_v3(M.C.CONTEXT_RULE) if "ctx3" in comps else M.C.CONTEXT_RULE
    def route(p):
        if (s:=tool.solve(p)): return ("tool",s[1])
        rules=[]
        if M.O.temporal_gate(p): rules.append(M.O.TEMPORAL_RULE)
        if (M.R.drafting_gate_v3(p,M.O.drafting_gate) if "draft3" in comps else M.O.drafting_gate(p)): rules.append(M.O.DRAFT_RULE)
        if M.C.context_gate(p): rules.append(ctx_rule)
        if "missing" in comps and M.R.missing_text_gate(p): rules.append(M.R.MISSING_TEXT_RULE)
        return ("model",M.O.SYSTEM+((" "+" ".join(rules)) if rules else ""))
    return route

def suites(M,exact):
    """name -> list of (row, scorer(row, output) -> bool, family)."""
    S={}
    S["exact72"]=[(r,lambda r,o:o==r["answer"],r["family"]) for r in exact]
    S["temporal8"]=[(c,M.O.temp_pass,"temporal") for c in M.O.TEMP]
    S["drafting8"]=[(c,M.O.draft_pass,"drafting") for c in M.O.DRAFT]
    S["promo_v2"]=[(r,(lambda r,o:o==r["answer"]) if r["scoring"]=="exact" else M.P.rubric_pass,r["family"]) for r in M.P.build()]
    S["heldout"]=[(r,lambda r,o:M.H.score(r,o)["strict"],"time") for r in M.H.build()]
    S["ctx_holdout"]=[(r,M.CH.score,"context") for r in M.CH.build()]
    S["suite_v3"]=[(r,M.V3.score,r["family"]) for r in M.V3.build()]
    S["probes"]=[(r,M.V3.score,r["family"]) for r in M.PR.build()]
    return S

def audit(M,exact_prompts):
    """CPU checks: tool v3 never wrong when it fires; new gates silent on the 72 exact prompts."""
    bad=[]
    for rows in (M.P.build(),M.H.build(),M.V3.build(),M.PR.build()):
        for r in rows:
            s=M.T3.solve(r["prompt"])
            if s and (r.get("scoring","exact")!="exact" or s[1]!=r["answer"]): bad.append(r["id"])
    leak=[p[:50] for p in exact_prompts if M.R.drafting_gate_v3(p,M.O.drafting_gate) or M.R.missing_text_gate(p) or M.C.context_gate(p) or M.O.temporal_gate(p)]
    return {"tool3_wrong_or_on_rubric":bad,"gate_leakage_on_exact":leak}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--preflight",action="store_true"); a=ap.parse_args()
    M=load()
    if a.preflight:
        au=audit(M,[])
        if au["tool3_wrong_or_on_rubric"]: raise RuntimeError(au)
        print("POLICY_V3_PREFLIGHT_PASS "+json.dumps({"configs":{k:sorted(v) for k,v in CONFIGS.items()},**au}),flush=True); return

    import torch
    from huggingface_hub import hf_hub_download
    from peft import PeftModel
    from transformers import AutoTokenizer,Qwen3_5ForCausalLM
    bench=json.loads(Path(hf_hub_download(BENCH,"candidate.json",revision=BENCH_REV,token=os.environ["HF_TOKEN"])).read_text())
    exact=[r for r in bench if r.get("scoring")=="exact"]; assert len(exact)==72
    au=audit(M,[r["prompt"] for r in exact]); print("POLICY_V3_AUDIT "+json.dumps(au),flush=True)
    tok=AutoTokenizer.from_pretrained(BASE,revision=BASE_REV); tok.pad_token=tok.eos_token
    base,li=Qwen3_5ForCausalLM.from_pretrained(BASE,revision=BASE_REV,dtype=torch.bfloat16,device_map={"":0},output_loading_info=True,key_mapping={r"^model.language_model\.":"model."})
    if li["missing_keys"] or li.get("mismatched_keys") or li.get("error_msgs"): raise RuntimeError("base load mismatch")
    model=PeftModel.from_pretrained(base,MODEL,revision=MODEL_REV).eval()
    cache={}
    def gen(system,p):
        if (system,p) not in cache:
            ids=tok.apply_chat_template([{"role":"system","content":system},{"role":"user","content":p}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt",return_dict=False).to(model.device)
            with torch.inference_mode(): z=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
            cache[(system,p)]=tok.decode(z[0,ids.shape[-1]:],skip_special_tokens=True).strip()
        return cache[(system,p)]

    S=suites(M,exact); routes={k:policy(M,v) for k,v in CONFIGS.items()}
    res={k:{} for k in CONFIGS}; outs={k:{} for k in CONFIGS}
    for sname,items in S.items():
        for cfg,route in routes.items():
            tot=[0,0]; fam={}
            for row,scorer,f in items:
                kind,val=route(row["prompt"]); o=val if kind=="tool" else gen(val,row["prompt"]); ok=bool(scorer(row,o))
                tot[0]+=ok; tot[1]+=1; ff=fam.setdefault(f,[0,0]); ff[0]+=ok; ff[1]+=1
                outs[cfg][(sname,row.get("id",row["prompt"][:40]))]=(ok,o)
            res[cfg][sname]={"score":tot,"by_family":fam}

    def checks(cfg):
        r,b=res[cfg],res["v2"]
        c={"exact_72":r["exact72"]["score"][0]==72,"temporal_8":r["temporal8"]["score"][0]==8,"drafting_ge_7":r["drafting8"]["score"][0]>=7,
           "promo_v2_200":r["promo_v2"]["score"][0]==200,"heldout_180":r["heldout"]["score"][0]==180,"ctx_holdout_ge_20":r["ctx_holdout"]["score"][0]>=20,
           "no_exact_regression":not [k for k,(ok,_) in outs["v2"].items() if k[0]=="exact72" and ok and not outs[cfg][k][0]],
           "suite_v3_above_v2":r["suite_v3"]["score"][0]>b["suite_v3"]["score"][0],
           **{f"v3_{f}_not_lower":r["suite_v3"]["by_family"][f][0]>=b["suite_v3"]["by_family"][f][0] for f in b["suite_v3"]["by_family"]},
           "probes_not_lower":r["probes"]["score"][0]>=b["probes"]["score"][0],
           "tool3_never_wrong":not au["tool3_wrong_or_on_rubric"],"no_gate_leakage_on_exact":not au["gate_leakage_on_exact"]}
        return c
    table={k:{s:res[k][s]["score"] for s in S} for k in CONFIGS}
    verdict={k:{"passed":all(c.values()),"failed":[n for n,v in checks(k).items() if not v]} for k in CONFIGS if k!="v2" for c in [checks(k)]}
    v2_ok={s:table["v2"][s][0]==V2_FROZEN[m] for s,m in (("exact72","exact"),("temporal8","temporal"),("drafting8","drafting"),("promo_v2","promo"),("heldout","heldout"))}
    passing=[k for k,v in verdict.items() if v["passed"]]
    best=max(passing,key=lambda k:(res[k]["suite_v3"]["score"][0]+res[k]["probes"]["score"][0],-len(CONFIGS[k]))) if passing else None
    # what changed for the best (or full) configuration vs v2, on the development and fresh suites
    focus=best or "full"
    diffs=[{"suite":k[0],"id":k[1],"v2_ok":outs["v2"][k][0],"cand_ok":ok,"v2":outs["v2"][k][1][:120],"cand":o[:120]}
           for k,(ok,o) in outs[focus].items() if outs["v2"][k][0]!=ok]
    for d in diffs: print("POLICY_V3_DIFF "+json.dumps(d),flush=True)
    for k,(ok,o) in outs[focus].items():
        if k[0] in("suite_v3","probes") and not ok: print("POLICY_V3_FAIL "+json.dumps({"config":focus,"suite":k[0],"id":k[1],"output":o[:160]}),flush=True)
    print("POLICY_V3_SUMMARY "+json.dumps({"scores":table,"v3_by_family":{k:res[k]["suite_v3"]["by_family"] for k in ("v2",focus)},
          "probes_by_family":{k:res[k]["probes"]["by_family"] for k in ("v2",focus)},"v2_frozen_reproduced":v2_ok,"verdict":verdict,
          "best":best,"best_components":sorted(CONFIGS[best]) if best else None,"generations":len(cache),"audit":au,"weights_changed":False}),flush=True)
if __name__=="__main__": main()
