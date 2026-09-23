# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","accelerate==1.15.0","huggingface-hub==1.31.0"]
# ///
"""Inference-only test of a *scoped* temporal-grounding rule on Consolidation1.

The global rule fixed temporal grounding but moved the 72-case exact benchmark
from 71/72 to 67/72. Here the rule is appended to the system prompt only when
the prompt asks a when/arrival/delivery question AND carries a speaker/context
time anchor ("Context: tonight", "I'm asking this afternoon"). Any benchmark
row the gate does not fire on sees a byte-identical prompt, so under greedy
decoding its output cannot change. No weights are touched.
"""
import argparse,json,os,re
from pathlib import Path
BASE="Qwen/Qwen3.5-4B"; BASE_REV="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MODEL="Jmiller18899/ember-qwen3.5-4b-consolidation1"
EVIDENCE_SOURCE="Jmiller18899/ember-qwen3.5-4b-sft-v1-repair1b"
SYSTEM="You are Ember. Answer the current request directly and concisely. Preserve supplied facts and follow the requested format."
TEMPORAL_RULE=("Time grounding: a time that only says when the user is asking (for example 'Context: tonight' or "
    "'I'm asking this afternoon') is not the time of the event. State an event time only if it is given explicitly "
    "(tracking, logs) or follows from a stated start time and duration. Otherwise say there is not enough information.")

QUESTION_RE=re.compile(r"\b(when\b|what time\b|arriv\w*|deliver\w*)",re.I)
ANCHOR_RE=re.compile(r"(\bcontext\s*:|\bi'?m asking\b|\bi am asking\b|\basking (this|at|on)\b|\bit'?s now\b|\bit is now\b|"
    r"\bright now\b|\bas of now\b|\b(tonight|this (morning|afternoon|evening))\b)",re.I)

def temporal_gate(prompt):
    return bool(QUESTION_RE.search(prompt) and ANCHOR_RE.search(prompt))

def system_for(prompt,scoped=True):
    return SYSTEM+" "+TEMPORAL_RULE if scoped and temporal_gate(prompt) else SYSTEM

ABSTAIN_RE=re.compile(r"(not enough|insufficient|(don'?t|do not|can'?t|cannot|can not) (know|tell|determine|say|be sure)|"
    r"no way to (know|tell)|unknown|(not|isn'?t|wasn'?t) (stated|given|specified|provided)|(i|you)(\s+would)? need|need (the|a|more|tracking)|"
    r"tracking|delivery estimate|journey duration|travel time)",re.I)

def _norm(t): return re.sub(r"\s+"," ",t.lower().replace("’","'")).strip()

def rubric_pass(case,output):
    """Semantic rubric: abstain cases accept any clear abstention; answer cases need every required token."""
    o=_norm(output)
    if any(_norm(x) in o for x in case.get("reject_any",[])): return False
    if case["expect"]=="abstain": return bool(ABSTAIN_RE.search(o))
    return all(any(_norm(v) in o for v in alts) for alts in case["require_all"])

# Reconstructed from the targeted run's description; replace with the original 8 prompts if they differ.
TEMPORAL_CASES=[
 {"id":"temporal-ctx-tonight-delivery","prompt":"Context: tonight. My package shipped Monday. When was it delivered?","expect":"abstain","reject_any":["delivered tonight","arrived tonight","was delivered this evening"]},
 {"id":"temporal-asking-afternoon-bus","prompt":"I'm asking this afternoon. The bus left the depot at 9:40 AM. When did it arrive?","expect":"abstain","reject_any":["arrived this afternoon","arrived in the afternoon"]},
 {"id":"temporal-asking-evening-order","prompt":"I'm asking this evening. The order was placed Friday. When was it delivered?","expect":"abstain","reject_any":["delivered this evening","delivered tonight"]},
 {"id":"temporal-now-friend","prompt":"It's now 3 PM. My friend left home this morning. When will they arrive?","expect":"abstain","reject_any":["at 3 pm","3:00 pm"]},
 {"id":"temporal-tracking-explicit","prompt":"Context: tonight. Tracking says: delivered Tuesday at 2:14 PM. When was it delivered?","expect":"answer","require_all":[["tuesday"],["2:14"]],"reject_any":["tonight"]},
 {"id":"temporal-log-explicit","prompt":"Context: this afternoon. Log: 07:02 build started; 07:19 build finished. When did the build finish?","expect":"answer","require_all":[["07:19","7:19"]],"reject_any":["this afternoon"]},
 {"id":"temporal-duration-train","prompt":"Context: tonight. A train left at 6:05 PM and traveled for 45 minutes. When did it arrive?","expect":"answer","require_all":[["6:50"]]},
 {"id":"temporal-duration-ferry","prompt":"Right now it's 11:30 AM. The ferry left at 10:50 AM and the crossing takes 25 minutes. When did it arrive?","expect":"answer","require_all":[["11:15"]],"reject_any":["11:30"]},
]

def load_exact(api,token):
    from huggingface_hub import hf_hub_download
    rev=api.model_info(EVIDENCE_SOURCE).sha
    rows=json.loads(Path(hf_hub_download(EVIDENCE_SOURCE,"repair-evidence/after.json",revision=rev,token=token)).read_text())
    return [r for r in rows if r.get("scoring")=="exact"]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--preflight",action="store_true"); a=p.parse_args()
    assert len(TEMPORAL_CASES)==8 and all(temporal_gate(c["prompt"]) for c in TEMPORAL_CASES)
    if a.preflight: print("TEMPORAL_SCOPED_PREFLIGHT_PASS",len(TEMPORAL_CASES)); return
    import torch
    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoTokenizer,Qwen3_5ForCausalLM
    token=os.environ["HF_TOKEN"]; api=HfApi(token=token)
    exact=load_exact(api,token)
    fired=[r["id"] for r in exact if temporal_gate(r["prompt"])]
    tok=AutoTokenizer.from_pretrained(BASE,revision=BASE_REV); tok.pad_token=tok.eos_token
    base,info=Qwen3_5ForCausalLM.from_pretrained(BASE,revision=BASE_REV,dtype=torch.bfloat16,device_map={"":0},output_loading_info=True,key_mapping={r"^model.language_model\.":"model."})
    if info["missing_keys"] or info.get("mismatched_keys") or info.get("error_msgs"): raise RuntimeError("base load mismatch")
    model_rev=api.model_info(MODEL).sha
    model=PeftModel.from_pretrained(base,MODEL,revision=model_rev).eval()
    def gen(prompt,scoped):
        ids=tok.apply_chat_template([{"role":"system","content":system_for(prompt,scoped)},{"role":"user","content":prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt",return_dict=False).to(model.device)
        with torch.inference_mode(): o=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
        return tok.decode(o[0,ids.shape[-1]:],skip_special_tokens=True).strip()
    def run_exact(scoped): return [{**r,"output":(t:=gen(r["prompt"],scoped)),"exact_match":t==r["answer"]} for r in exact]
    def run_temporal(scoped): return [{**c,"output":(t:=gen(c["prompt"],scoped)),"pass":rubric_pass(c,t)} for c in TEMPORAL_CASES]
    base_exact,scoped_exact=run_exact(False),run_exact(True)
    base_temp,scoped_temp=run_temporal(False),run_temporal(True)
    B={r["id"]:r for r in base_exact}; S={r["id"]:r for r in scoped_exact}
    regress=[i for i in B if B[i]["exact_match"] and not S[i]["exact_match"]]
    by_family={}
    for r in scoped_exact:
        f=by_family.setdefault(r.get("family","unknown"),[0,0,0]); f[0]+=B[r["id"]]["exact_match"]; f[1]+=r["exact_match"]; f[2]+=1
    s0=sum(r["exact_match"] for r in base_exact); s1=sum(r["exact_match"] for r in scoped_exact)
    t0=sum(r["pass"] for r in base_temp); t1=sum(r["pass"] for r in scoped_temp)
    gate={"no_exact_regressions":not regress,"exact_not_lower":s1>=s0,"temporal_at_least_7_of_8":t1>=7,"temporal_improved":t1>t0}
    summary={"model":MODEL,"model_revision":model_rev,"gate_fired_on_benchmark_ids":fired,"baseline_exact":[s0,len(exact)],"scoped_exact":[s1,len(exact)],
        "family_baseline_scoped_total":by_family,"exact_regressions":regress,"baseline_temporal":[t0,8],"scoped_temporal":[t1,8],
        "gate_conditions":gate,"accepted":all(gate.values()),"weights_changed":False,"production_ready":False}
    out=Path("temporal-scoped-results"); out.mkdir(exist_ok=True)
    for name,data in [("baseline_exact",base_exact),("scoped_exact",scoped_exact),("baseline_temporal",base_temp),("scoped_temporal",scoped_temp),("summary",summary)]:
        (out/f"{name}.json").write_text(json.dumps(data,indent=2))
    api.upload_folder(repo_id=MODEL,folder_path=str(out),path_in_repo="temporal-scoped-rule",allow_patterns=["*.json"],commit_message="Save scoped temporal-rule evaluation (no weight change)")
    print("TEMPORAL_SCOPED_SUMMARY "+json.dumps(summary),flush=True)
if __name__=="__main__": main()
