# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","accelerate==1.15.0","huggingface-hub==1.31.0"]
# ///
"""Evaluation only: promotion suite v3 on Repair2 weights under three runtime configurations.

  model_only  Repair2 with the base system prompt; no scoped rules, no tool
  v1          frozen v1 policy (tool, then scoped temporal/drafting rules)
  v2          frozen v2 policy (v1 + scoped context rule): the promoted baseline
Generation is shared: a prompt is regenerated only when its system prompt differs from model_only's, and
tool answers need no generation. Prints per-family scores, routes, and every v2 failure with its output.
This run sets v2's baseline on v3. It applies no promotion gate and changes nothing.
Usage:  python jobs/ember_promotion_suite_v3_eval.py --preflight   (CPU)
        uv run jobs/ember_promotion_suite_v3_eval.py                (GPU)
"""
import argparse,hashlib,json,types,urllib.request
from pathlib import Path

SRC_COMMIT="f381fee812d7528b7f33eb7ba06e1b52fe854cb0"
RAW=f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{SRC_COMMIT}/"
PINNED={
 "jobs/ember_promotion_suite_v3.py":"a31e14acae58594cfca60e3de28c19cf567a00d5fdc57fa137f899695e6ca87f",
 "jobs/ember_arith_tool.py":"d4dd1d38f4e34d3f7f3262d2bb1d3bfff2aecfc6d5af23b02ae39fd46d31bac1",
 "jobs/ember_context_rule.py":"239a1d29e55be77d1f46867dc8cc6d94b82e39823b33f9ffd8ab1b56ffc8cb70",
 "jobs/ember_4b_repair2_original_promotion_eval.py":"48272629067857fc873dfab6c71c61e82c90eb4bf5cab7d85a7f5c9108b29ffc",
}
MODEL="Jmiller18899/ember-qwen3.5-4b-repair2"; MODEL_REV="daf938bba5d4e6b650ec9d34a2d3ac56706cf549"
CONFIGS=("model_only","v1","v2")

def module(rel,name):
    local=Path(__file__).resolve().parents[1]/rel
    data=local.read_bytes() if local.exists() else urllib.request.urlopen(RAW+rel,timeout=60).read()
    got=hashlib.sha256(data).hexdigest()
    if got!=PINNED[rel]: raise RuntimeError(f"{rel}: sha256 {got} != pinned {PINNED[rel]}")
    m=types.ModuleType(name); exec(compile(data.decode(),rel,"exec"),m.__dict__); return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--preflight",action="store_true"); a=ap.parse_args()
    V=module("jobs/ember_promotion_suite_v3.py","v3"); T=module("jobs/ember_arith_tool.py","tool")
    C=module("jobs/ember_context_rule.py","ctx"); O=module("jobs/ember_4b_repair2_original_promotion_eval.py","rules")
    rows=V.build(); assert len(rows)==192
    def systems(p):
        v1=O.system_for(p); v2=v1+" "+C.CONTEXT_RULE if C.context_gate(p) else v1
        return {"model_only":O.SYSTEM,"v1":v1,"v2":v2}
    tool_wrong=[r["id"] for r in rows if (s:=T.solve(r["prompt"])) and (r["scoring"]!="exact" or s[1]!=r["answer"])]
    if tool_wrong: raise RuntimeError(f"tool wrong or fired on rubric: {tool_wrong}")
    routes={}
    for r in rows:
        f=routes.setdefault(r["family"],{"tool":0,"temporal":0,"drafting":0,"context":0})
        f["tool"]+=T.solve(r["prompt"]) is not None; f["temporal"]+=O.temporal_gate(r["prompt"]); f["drafting"]+=O.drafting_gate(r["prompt"]); f["context"]+=C.context_gate(r["prompt"])
    if a.preflight: print("SUITE_V3_PREFLIGHT_PASS "+json.dumps({"cases":len(rows),"routes":routes}),flush=True); return

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer,Qwen3_5ForCausalLM
    BASE,BASE_REV="Qwen/Qwen3.5-4B","851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
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

    fam={c:{} for c in CONFIGS}; fails=[]; rescued={"by_tool":0,"by_rules":0}
    for r in rows:
        sysm=systems(r["prompt"]); hit=T.solve(r["prompt"])
        outs={"model_only":gen(sysm["model_only"],r["prompt"])}
        for c in ("v1","v2"): outs[c]=hit[1] if hit else gen(sysm[c],r["prompt"])
        ok={c:V.score(r,o) for c,o in outs.items()}
        for c in CONFIGS: f=fam[c].setdefault(r["family"],[0,0]); f[0]+=int(ok[c]); f[1]+=1
        if ok["v2"] and not ok["model_only"]: rescued["by_tool" if hit else "by_rules"]+=1
        if not ok["v2"]: fails.append({"id":r["id"],"family":r["family"],"route":"tool" if hit else "model","prompt":r["prompt"],
                                       "expected":r.get("answer") or r.get("values") or r.get("facts") or r.get("expect"),"v2_output":outs["v2"],"model_only_ok":ok["model_only"]})
    for x in fails: print("SUITE_V3_FAIL "+json.dumps(x),flush=True)
    totals={c:sum(v[0] for v in fam[c].values()) for c in CONFIGS}
    summary={"suite":"promotion-v3","cases":len(rows),"model":f"{MODEL}@{MODEL_REV}","src":SRC_COMMIT,"totals":totals,"by_family":fam,
             "routes":routes,"v2_rescues_over_model_only":rescued,"v2_failures":len(fails),"generations":len(cache),"weights_changed":False,"gated":False}
    print("SUITE_V3_SUMMARY "+json.dumps(summary),flush=True)
if __name__=="__main__": main()
