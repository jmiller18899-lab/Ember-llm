# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","accelerate==1.15.0","huggingface-hub==1.31.0"]
# ///
"""Evaluation only: the 15 promotion-v2 context_consistency cases through the frozen runtime policy.

Prints every output, pass/fail, and which rubric fact (owner / item / day) is missing, so the three misses
can be diagnosed. No weights are trained or uploaded.
"""
import hashlib,json,os,re,types,urllib.request
from pathlib import Path
COMMIT="b19ffe2bfaf6b95c7e590ad3db4b67e0e7bd3840"   # freeze/ember-4b-repair2-arith-tool-promoted-2026-09-24
RAW=f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{COMMIT}/"
PINNED={"jobs/ember_4b_promotion_v2_frozen_eval.py":"d78917ed19a5d8160b632b6c553f5b0d9ff62b8bc1ba49700a35c70d023907b7",
        "jobs/ember_4b_repair2_original_promotion_eval.py":"48272629067857fc873dfab6c71c61e82c90eb4bf5cab7d85a7f5c9108b29ffc",
        "jobs/ember_arith_tool.py":"d4dd1d38f4e34d3f7f3262d2bb1d3bfff2aecfc6d5af23b02ae39fd46d31bac1"}
MODEL="Jmiller18899/ember-qwen3.5-4b-repair2"; MODEL_REV="daf938bba5d4e6b650ec9d34a2d3ac56706cf549"
def module(rel,name):
    local=Path(__file__).resolve().parents[1]/rel
    data=local.read_bytes() if local.exists() else urllib.request.urlopen(RAW+rel,timeout=60).read()
    assert hashlib.sha256(data).hexdigest()==PINNED[rel],rel
    m=types.ModuleType(name); exec(compile(data.decode(),rel,"exec"),m.__dict__); return m
def main():
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer,Qwen3_5ForCausalLM
    P=module("jobs/ember_4b_promotion_v2_frozen_eval.py","promo"); O=module("jobs/ember_4b_repair2_original_promotion_eval.py","rules"); T=module("jobs/ember_arith_tool.py","tool")
    rows=[r for r in P.build() if r["family"]=="context_consistency"]; assert len(rows)==15
    tok=AutoTokenizer.from_pretrained(P.BASE,revision=P.BASE_REV); tok.pad_token=tok.eos_token
    base=Qwen3_5ForCausalLM.from_pretrained(P.BASE,revision=P.BASE_REV,dtype=torch.bfloat16,device_map={"":0},key_mapping={r"^model.language_model\.":"model."})
    model=PeftModel.from_pretrained(base,MODEL,revision=MODEL_REV).eval()
    def gen(p):
        if (s:=T.solve(p)): return s[1]
        ids=tok.apply_chat_template([{"role":"system","content":O.system_for(p)},{"role":"user","content":p}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt",return_dict=False).to(model.device)
        with torch.inference_mode(): z=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
        return tok.decode(z[0,ids.shape[-1]:],skip_special_tokens=True).strip()
    npass=0
    for r in rows:
        o=gen(r["prompt"]); ok=P.rubric_pass(r,o); npass+=ok
        owner,item,day=re.search(r"Answer that (.+?) owns the (.+?) and it is returned (.+?);",r["rubric"]).groups()
        missing=[k for k,v in (("owner",owner),("item",item),("day",day)) if P.norm(v) not in P.norm(o)]
        print("CONTEXT_DIAG_CASE "+json.dumps({"id":r["id"],"pass":ok,"missing":missing,"expected":[owner,item,day],"output":o}),flush=True)
    print("CONTEXT_DIAG_SUMMARY "+json.dumps({"pass":[npass,15],"weights_changed":False}),flush=True)
if __name__=="__main__": main()
