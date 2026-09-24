# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","accelerate==1.15.0","huggingface-hub==1.31.0"]
# ///
"""Evaluation-only: which of the 72 original exact cases does Repair2 fail, versus frozen Consolidation1?

Consolidation1 scored 71/72 and Repair2 70/72 under the same promotion runtime policy, but the
promotion evaluator printed only totals. This job replays both adapters (same base, same prompts,
same greedy decoding, same combined system prompt) and prints, for every case either model fails:
the prompt, gold answer, both outputs, and a teacher-forced margin on the gold answer
(first divergent token, gold-token rank and log-prob there, total gold log-prob). That shows
whether each Repair2 miss is a near miss a tiny targeted update can fix.

No weights are written and nothing is uploaded; results are only printed.
Usage:  python jobs/ember_4b_repair2_exact_diff.py --preflight   (CPU: policy/gate sanity)
        uv run jobs/ember_4b_repair2_exact_diff.py                (GPU: replay both adapters)
"""
import argparse,json,os,types,urllib.request
from pathlib import Path

BASE="Qwen/Qwen3.5-4B"; BASE_REV="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
BENCH="Jmiller18899/ember-generalization-benchmark-v1"
FROZEN=("c1","Jmiller18899/ember-qwen3.5-4b-consolidation1","62e5b58b78f823a6cd720a4ff53d0adda1624210")
REPAIR2=("r2","Jmiller18899/ember-qwen3.5-4b-repair2",None)  # latest revision; printed below
# Expected scores per benchmark revision (835243e1 corrects the time-14 gold from 12:05 AM to 12:05 PM).
EXPECTED_BY_BENCH={"9b080364c0b4d4d005cc376a4f45daa1a2edcd77":{"c1":71,"r2":70},
                   "835243e16cf19e6ee34f27a19cba14242c14969f":{"c1":72,"r2":71}}
# Reuse the exact runtime policy (system prompt + scoped rules) of the promotion evaluator.
POLICY_SRC="jobs/ember_4b_repair2_original_promotion_eval.py"
POLICY_COMMIT="3d559086a5a581bb0bb5b7c1adf8d7a0e65d8db0"

def policy():
    local=Path(__file__).resolve().parents[1]/POLICY_SRC
    src=local.read_text() if local.exists() else urllib.request.urlopen(
        f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{POLICY_COMMIT}/{POLICY_SRC}",timeout=60).read().decode()
    m=types.ModuleType("promo"); exec(compile(src,POLICY_SRC,"exec"),m.__dict__); return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--preflight",action="store_true"); a=ap.parse_args()
    P=policy()
    assert P.system_for("What is 7 times 8?")==P.SYSTEM
    assert P.temporal_gate(P.TEMP[0]["prompt"]) and P.drafting_gate(P.DRAFT[0]["prompt"])
    if a.preflight:
        print("EXACT_DIFF_PREFLIGHT_PASS",flush=True); return
    import torch
    from huggingface_hub import HfApi,hf_hub_download
    from peft import PeftModel
    from transformers import AutoTokenizer,Qwen3_5ForCausalLM
    token=os.environ["HF_TOKEN"]; api=HfApi(token=token)
    bench_rev=api.model_info(BENCH).sha
    rows=json.loads(Path(hf_hub_download(BENCH,"candidate.json",revision=bench_rev,token=token)).read_text())
    exact=[r for r in rows if r.get("scoring")=="exact"]
    if len(exact)!=72: raise RuntimeError(f"expected 72 exact, got {len(exact)}")
    leak=[r["id"] for r in exact if P.system_for(r["prompt"])!=P.SYSTEM]
    r2_rev=api.model_info(REPAIR2[1]).sha
    print("EXACT_DIFF_REVISIONS "+json.dumps({"bench":bench_rev,"c1":FROZEN[2],"r2":r2_rev,"rule_leakage":leak}),flush=True)

    tok=AutoTokenizer.from_pretrained(BASE,revision=BASE_REV); tok.pad_token=tok.eos_token
    base,info=Qwen3_5ForCausalLM.from_pretrained(BASE,revision=BASE_REV,dtype=torch.bfloat16,device_map={"":0},output_loading_info=True,key_mapping={r"^model.language_model\.":"model."})
    if info["missing_keys"] or info.get("mismatched_keys") or info.get("error_msgs"): raise RuntimeError("base load mismatch")
    model=PeftModel.from_pretrained(base,FROZEN[1],adapter_name="c1",revision=FROZEN[2]).eval()
    model.load_adapter(REPAIR2[1],adapter_name="r2",revision=r2_rev); model.eval()

    def prompt_ids(prompt):
        return tok.apply_chat_template([{"role":"system","content":P.system_for(prompt)},{"role":"user","content":prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt",return_dict=False).to(model.device)
    def gen(prompt):
        ids=prompt_ids(prompt)
        with torch.inference_mode(): out=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
        return tok.decode(out[0,ids.shape[-1]:],skip_special_tokens=True).strip()
    def margin(prompt,answer):
        """Teacher-forced score of the gold answer (+EOS): where greedy first leaves it, and by how much."""
        ids=prompt_ids(prompt); gold=tok(answer,add_special_tokens=False,return_tensors="pt").input_ids.to(model.device)
        gold=torch.cat([gold,torch.tensor([[tok.eos_token_id]],device=model.device)],1)
        with torch.inference_mode(): logits=model(input_ids=torch.cat([ids,gold],1)).logits[0,ids.shape[-1]-1:-1].float()
        lp=logits.log_softmax(-1); g=gold[0]; glp=lp.gather(1,g[:,None])[:,0]; top=lp.argmax(-1)
        miss=(top!=g).nonzero()
        res={"gold_logprob":round(glp.sum().item(),3),"gold_tokens":int(g.numel())}
        if len(miss):
            i=int(miss[0]); rank=int((lp[i]>lp[i,g[i]]).sum())+1
            res.update({"first_divergence":i,"gold_token":tok.decode(g[i:i+1]),"model_token":tok.decode(top[i:i+1]),
                        "gold_rank":rank,"gold_tok_logprob":round(glp[i].item(),3),"gap_to_top":round((lp[i,top[i]]-lp[i,g[i]]).item(),3),
                        "prefix":tok.decode(g[:i])})
        return res

    res={}
    for name in ("c1","r2"):
        model.set_adapter(name); res[name]={}
        for r in exact:
            o=gen(r["prompt"]); res[name][r["id"]]={"output":o,"pass":o==r["answer"]}
        print(f"EXACT_DIFF_SCORE {name} {sum(v['pass'] for v in res[name].values())}/72",flush=True)
    by_id={r["id"]:r for r in exact}
    fails={n:sorted(i for i,v in res[n].items() if not v["pass"]) for n in res}
    for i in sorted(set(fails["c1"])|set(fails["r2"])):
        r=by_id[i]; case={"id":i,"family":r.get("family") or r.get("category") or r.get("type"),
              "prompt":r["prompt"],"answer":r["answer"],
              "c1_output":res["c1"][i]["output"],"c1_pass":res["c1"][i]["pass"],
              "r2_output":res["r2"][i]["output"],"r2_pass":res["r2"][i]["pass"],"margin":{}}
        for n in ("c1","r2"):
            model.set_adapter(n); case["margin"][n]=margin(r["prompt"],r["answer"])
        print("EXACT_DIFF_CASE "+json.dumps(case),flush=True)
    summary={"c1":[72-len(fails["c1"]),72],"r2":[72-len(fails["r2"]),72],"c1_failures":fails["c1"],"r2_failures":fails["r2"],
             "r2_new_failures":sorted(set(fails["r2"])-set(fails["c1"])),"r2_fixed":sorted(set(fails["c1"])-set(fails["r2"])),
             "shared_failures":sorted(set(fails["c1"])&set(fails["r2"])),
             "bench_revision":bench_rev,"reproduces_expected_scores":(all(72-len(fails[n])==exp[n] for n in exp) if (exp:=EXPECTED_BY_BENCH.get(bench_rev)) else None),"weights_changed":False}
    print("EXACT_DIFF_SUMMARY "+json.dumps(summary),flush=True)
if __name__=="__main__": main()
