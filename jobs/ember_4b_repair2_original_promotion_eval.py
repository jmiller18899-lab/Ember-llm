# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","accelerate==1.15.0","huggingface-hub==1.31.0"]
# ///
"""Regression evaluation for Repair2 against original Ember promotion gates."""
import argparse,json,os,re
from pathlib import Path
BASE="Qwen/Qwen3.5-4B"; BASE_REV="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MODEL="Jmiller18899/ember-qwen3.5-4b-repair2"
BENCH="Jmiller18899/ember-generalization-benchmark-v1"
SYSTEM="You are Ember. Answer the current request directly and concisely. Preserve supplied facts and follow the requested format."
TEMPORAL_RULE=("Time grounding: a time that only says when the user is asking (for example 'Context: tonight' or "
"'I'm asking this afternoon') is not the time of the event. State an event time only if it is given explicitly "
"(tracking, logs) or follows from a stated start time and duration. Otherwise say there is not enough information.")
DRAFT_RULE=("Drafting transformation: when the user asks you to message or draft for a named recipient, output only the actual sendable message, not a restatement of the instruction. Address the named recipient directly. Any third-person reference to that recipient must become second person: their -> your, they -> you, them -> you. Example: Message Alex: I found their badge -> Hi Alex, I found your badge. Never copy the word their when it refers to the person you are addressing. Preserve every supplied fact. When asked to shorten text, actually make it shorter while preserving explicitly protected facts such as place and time.")

QUESTION_RE=re.compile(r"\b(when\b|what time\b|arriv\w*|deliver\w*)",re.I)
ANCHOR_RE=re.compile(r"(\bcontext\s*:|\bi'?m asking\b|\bi am asking\b|\basking (this|at|on)\b|\bit'?s now\b|\bit is now\b|\bright now\b|\bas of now\b|\b(tonight|this (morning|afternoon|evening))\b)",re.I)
MESSAGE_RE=re.compile(r"^\s*(message|draft (?:a )?(?:message|text|reply)|write (?:a )?(?:message|text|reply))\b",re.I)
SHORTEN_RE=re.compile(r"\b(shorten|make (?:this|it) shorter|condense)\b",re.I)
def temporal_gate(p): return bool(QUESTION_RE.search(p) and ANCHOR_RE.search(p))
def drafting_gate(p): return bool(MESSAGE_RE.search(p) or SHORTEN_RE.search(p))
def system_for(p):
    rules=[]
    if temporal_gate(p): rules.append(TEMPORAL_RULE)
    if drafting_gate(p): rules.append(DRAFT_RULE)
    return SYSTEM + ((" "+" ".join(rules)) if rules else "")

ABSTAIN_RE=re.compile(r"(not enough|insufficient|(don'?t|do not|can'?t|cannot|can not) (know|tell|determine|say|be sure)|no way to (know|tell)|unknown|(not|isn'?t|wasn'?t) (stated|given|specified|provided)|(i|you)(\s+would)? need|need (the|a|more|tracking)|tracking|delivery estimate|journey duration|travel time)",re.I)
def norm(s): return re.sub(r"\s+"," ",s.lower().replace("’","'")).strip()
TEMP=[
 {"id":"rule-01","prompt":"My shipment was collected Wednesday. When was it delivered? Context: tonight.","expect":"abstain","reject_any":["delivered tonight","arrived tonight","tonight."]},
 {"id":"rule-02","prompt":"My shipment was collected Wednesday. Tracking says delivered tonight at 7:26 PM. When was it delivered?","expect":"answer","require_all":[["7:26"]],"reject_any":[]},
 {"id":"rule-03","prompt":"My bus left at 9:10 AM. I'm asking this afternoon. When did it arrive?","expect":"abstain","reject_any":["arrived this afternoon"]},
 {"id":"rule-04","prompt":"My bus left at 9:10 AM. The log says it arrived this afternoon at 1:05 PM. When did it arrive?","expect":"answer","require_all":[["1:05"]],"reject_any":[]},
 {"id":"rule-05","prompt":"My package was picked up Tuesday. When did it arrive? Context: this afternoon.","expect":"abstain","reject_any":["arrived this afternoon"]},
 {"id":"rule-06","prompt":"The delivery record says it arrived this afternoon at 3:17 PM. What time did it arrive?","expect":"answer","require_all":[["3:17"]],"reject_any":[]},
 {"id":"rule-07","prompt":"A ferry departed at 1:30 PM and traveled for 45 minutes. What time did it arrive?","expect":"answer","require_all":[["2:15"]],"reject_any":[]},
 {"id":"rule-08","prompt":"Tracking: delivered earlier at 10:18 AM. What time was delivery?","expect":"answer","require_all":[["10:18"]],"reject_any":[]},
]
DRAFT=[
 {"id":"draft-msg-01","kind":"message","prompt":"Message Nina: I found their notebook and can return it Monday.","name":"Nina","facts":["notebook","Monday"]},
 {"id":"draft-msg-02","kind":"message","prompt":"Message Priya: I found their keycard and can return it Tuesday.","name":"Priya","facts":["keycard","Tuesday"]},
 {"id":"draft-msg-03","kind":"message","prompt":"Message Rosa: I found their umbrella and can return it Friday.","name":"Rosa","facts":["umbrella","Friday"]},
 {"id":"draft-msg-04","kind":"message","prompt":"Message Tara: I found their folder and can return it Saturday.","name":"Tara","facts":["folder","Saturday"]},
 {"id":"draft-short-01","kind":"shorten","prompt":"Shorten this without dropping place or time: 'Please ensure all visitors meet at the front desk before 9.'","facts":["front desk","before 9"],"source":"Please ensure all visitors meet at the front desk before 9."},
 {"id":"draft-short-02","kind":"shorten","prompt":"Shorten this without dropping place or time: 'Please ensure all guests meet at the conference room before 10.'","facts":["conference room","before 10"],"source":"Please ensure all guests meet at the conference room before 10."},
 {"id":"draft-short-03","kind":"shorten","prompt":"Shorten this without dropping place or time: 'Please make sure everyone arrives at the lab before 8.'","facts":["lab","before 8"],"source":"Please make sure everyone arrives at the lab before 8."},
 {"id":"draft-short-04","kind":"shorten","prompt":"Condense this but keep location and time: 'All team members should gather at the garage before 7.'","facts":["garage","before 7"],"source":"All team members should gather at the garage before 7."},
]
def temp_pass(c,o):
    n=norm(o)
    if any(norm(x) in n for x in c.get("reject_any",[])): return False
    if c["expect"]=="abstain": return bool(ABSTAIN_RE.search(n))
    return all(any(norm(v) in n for v in alts) for alts in c["require_all"])
def draft_pass(c,o):
    n=norm(o)
    if not all(norm(x) in n for x in c["facts"]): return False
    if c["kind"]=="message":
        return norm(c["name"]) in n and not n.startswith("message "+norm(c["name"])+":") and (" your " in " "+n+" " or " you " in " "+n+" ")
    return len(o.strip()) < len(c["source"].strip())

def main():
    p=argparse.ArgumentParser(); p.add_argument("--preflight",action="store_true"); a=p.parse_args()
    assert len(TEMP)==8 and len(DRAFT)==8
    if a.preflight:
        print("PROMOTION_PREFLIGHT_PASS",len(TEMP),len(DRAFT)); return
    import torch
    from huggingface_hub import HfApi,hf_hub_download
    from peft import PeftModel
    from transformers import AutoTokenizer,Qwen3_5ForCausalLM
    token=os.environ["HF_TOKEN"]; api=HfApi(token=token)
    rev=api.model_info(BENCH).sha
    rows=json.loads(Path(hf_hub_download(BENCH,"candidate.json",revision=rev,token=token)).read_text())
    exact=[r for r in rows if r.get("scoring")=="exact"]
    if len(exact)!=72: raise RuntimeError(f"expected 72 exact, got {len(exact)}")
    fired_temp=[r["id"] for r in exact if temporal_gate(r["prompt"])]
    fired_draft=[r["id"] for r in exact if drafting_gate(r["prompt"])]
    tok=AutoTokenizer.from_pretrained(BASE,revision=BASE_REV); tok.pad_token=tok.eos_token
    base,info=Qwen3_5ForCausalLM.from_pretrained(BASE,revision=BASE_REV,dtype=torch.bfloat16,device_map={"":0},output_loading_info=True,key_mapping={r"^model.language_model\.":"model."})
    if info["missing_keys"] or info.get("mismatched_keys") or info.get("error_msgs"): raise RuntimeError("base load mismatch")
    model=PeftModel.from_pretrained(base,MODEL,revision=api.model_info(MODEL).sha).eval()
    def gen(prompt,combined):
        sys=system_for(prompt) if combined else SYSTEM
        ids=tok.apply_chat_template([{"role":"system","content":sys},{"role":"user","content":prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt",return_dict=False).to(model.device)
        with torch.inference_mode(): out=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
        return tok.decode(out[0,ids.shape[-1]:],skip_special_tokens=True).strip()
    b=[]; s=[]
    for r in exact:
        x=gen(r["prompt"],False); y=gen(r["prompt"],True)
        b.append({**r,"output":x,"exact_match":x==r["answer"]}); s.append({**r,"output":y,"exact_match":y==r["answer"]})
    temp=[{**c,"output":(o:=gen(c["prompt"],True)),"pass":temp_pass(c,o)} for c in TEMP]
    draft=[{**c,"output":(o:=gen(c["prompt"],True)),"pass":draft_pass(c,o)} for c in DRAFT]
    B={r["id"]:r for r in b}; S={r["id"]:r for r in s}
    regress=[i for i in B if B[i]["exact_match"] and not S[i]["exact_match"]]
    score0=sum(r["exact_match"] for r in b); score1=sum(r["exact_match"] for r in s)
    ts=sum(r["pass"] for r in temp); ds=sum(r["pass"] for r in draft)
    accepted=(score1>=71 and not regress and ts==8 and ds>=7 and not fired_temp and not fired_draft)
    summary={"model":MODEL,"baseline_exact":[score0,72],"combined_exact":[score1,72],"exact_regressions":regress,"temporal":[ts,8],"drafting":[ds,8],"temporal_fired_on_exact":fired_temp,"drafting_fired_on_exact":fired_draft,"weights_changed":False,"promotion_freeze_pass":accepted,"production_ready":False}
    print("PROMOTION_FREEZE_SUMMARY "+json.dumps(summary),flush=True)
    print("PROMOTION_FREEZE_TARGETS "+json.dumps({"temporal":temp,"drafting":draft}),flush=True)
if __name__=="__main__": main()
