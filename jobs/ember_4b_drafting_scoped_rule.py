# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","accelerate==1.15.0","huggingface-hub==1.31.0"]
# ///
"""Inference-only scoped drafting/perspective rule on untouched Consolidation1."""
import argparse,json,os,re
from pathlib import Path
BASE="Qwen/Qwen3.5-4B"; BASE_REV="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MODEL="Jmiller18899/ember-qwen3.5-4b-consolidation1"
BENCH="Jmiller18899/ember-generalization-benchmark-v1"
SYSTEM="You are Ember. Answer the current request directly and concisely. Preserve supplied facts and follow the requested format."
DRAFT_RULE=("Drafting transformation: when the user asks you to message or draft for a named recipient, output only the actual sendable message, not a restatement of the instruction. Address the named recipient directly. Any third-person reference to that recipient must become second person: their -> your, they -> you, them -> you. Example: Message Alex: I found their badge -> Hi Alex, I found your badge. Never copy the word their when it refers to the person you are addressing. Preserve every supplied fact. When asked to shorten text, actually make it shorter while preserving explicitly protected facts such as place and time.")

MESSAGE_RE=re.compile(r"^\s*(message|draft (?:a )?(?:message|text|reply)|write (?:a )?(?:message|text|reply))\b",re.I)
SHORTEN_RE=re.compile(r"\b(shorten|make (?:this|it) shorter|condense)\b",re.I)
def drafting_gate(prompt): return bool(MESSAGE_RE.search(prompt) or SHORTEN_RE.search(prompt))
def system_for(prompt,scoped=True): return SYSTEM+" "+DRAFT_RULE if scoped and drafting_gate(prompt) else SYSTEM

CASES=[
 {"id":"draft-msg-01","trigger":True,"kind":"message","prompt":"Message Nina: I found their notebook and can return it Monday.","name":"Nina","facts":["notebook","Monday"]},
 {"id":"draft-msg-02","trigger":True,"kind":"message","prompt":"Message Priya: I found their keycard and can return it Tuesday.","name":"Priya","facts":["keycard","Tuesday"]},
 {"id":"draft-msg-03","trigger":True,"kind":"message","prompt":"Message Rosa: I found their umbrella and can return it Friday.","name":"Rosa","facts":["umbrella","Friday"]},
 {"id":"draft-msg-04","trigger":True,"kind":"message","prompt":"Message Tara: I found their folder and can return it Saturday.","name":"Tara","facts":["folder","Saturday"]},
 {"id":"draft-short-01","trigger":True,"kind":"shorten","prompt":"Shorten this without dropping place or time: 'Please ensure all visitors meet at the front desk before 9.'","facts":["front desk","before 9"],"source":"Please ensure all visitors meet at the front desk before 9."},
 {"id":"draft-short-02","trigger":True,"kind":"shorten","prompt":"Shorten this without dropping place or time: 'Please ensure all guests meet at the conference room before 10.'","facts":["conference room","before 10"],"source":"Please ensure all guests meet at the conference room before 10."},
 {"id":"draft-short-03","trigger":True,"kind":"shorten","prompt":"Shorten this without dropping place or time: 'Please make sure everyone arrives at the lab before 8.'","facts":["lab","before 8"],"source":"Please make sure everyone arrives at the lab before 8."},
 {"id":"draft-short-04","trigger":True,"kind":"shorten","prompt":"Condense this but keep location and time: 'All team members should gather at the garage before 7.'","facts":["garage","before 7"],"source":"All team members should gather at the garage before 7."},
]
def norm(s): return re.sub(r"\s+"," ",s.lower().strip())
def case_pass(c,out):
 o=norm(out)
 if not all(norm(x) in o for x in c["facts"]): return False
 if c["kind"]=="message":
  # Must be transformed: address/name retained, not echo "Message NAME:", and use second-person recipient perspective.
  return norm(c["name"]) in o and not o.startswith("message "+norm(c["name"])+":") and (" your " in " "+o+" " or " you " in " "+o+" ")
 # Shortening must preserve protected facts and reduce source character count after trimming.
 return len(out.strip()) < len(c["source"].strip())

def main():
 p=argparse.ArgumentParser(); p.add_argument("--preflight",action="store_true"); a=p.parse_args()
 mismatches=[c["id"] for c in CASES if drafting_gate(c["prompt"])!=c["trigger"]]
 if mismatches: raise AssertionError("drafting trigger mismatches: "+json.dumps(mismatches))
 if a.preflight: print("DRAFT_SCOPED_PREFLIGHT_PASS",len(CASES)); return
 import torch
 from huggingface_hub import HfApi,hf_hub_download
 from peft import PeftModel
 from transformers import AutoTokenizer,Qwen3_5ForCausalLM
 token=os.environ["HF_TOKEN"]; api=HfApi(token=token)
 rev=api.model_info(BENCH).sha
 rows=json.loads(Path(hf_hub_download(BENCH,"candidate.json",revision=rev,token=token)).read_text())
 exact=[r for r in rows if r.get("scoring")=="exact"]
 if len(exact)!=72: raise RuntimeError(f"expected 72 exact, got {len(exact)}")
 fired=[r["id"] for r in exact if drafting_gate(r["prompt"])]
 tok=AutoTokenizer.from_pretrained(BASE,revision=BASE_REV); tok.pad_token=tok.eos_token
 base,info=Qwen3_5ForCausalLM.from_pretrained(BASE,revision=BASE_REV,dtype=torch.bfloat16,device_map={"":0},output_loading_info=True,key_mapping={r"^model.language_model\.":"model."})
 if info["missing_keys"] or info.get("mismatched_keys") or info.get("error_msgs"): raise RuntimeError("base load mismatch")
 model=PeftModel.from_pretrained(base,MODEL,revision=api.model_info(MODEL).sha).eval()
 def gen(prompt,scoped):
  ids=tok.apply_chat_template([{"role":"system","content":system_for(prompt,scoped)},{"role":"user","content":prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt",return_dict=False).to(model.device)
  with torch.inference_mode(): o=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
  return tok.decode(o[0,ids.shape[-1]:],skip_special_tokens=True).strip()
 def run_exact(scoped):
  out=[]
  for r in exact:
   t=gen(r["prompt"],scoped); out.append({**r,"output":t,"exact_match":t==r["answer"]})
  return out
 def run_cases(scoped):
  return [{**c,"output":(t:=gen(c["prompt"],scoped)),"pass":case_pass(c,t)} for c in CASES]
 b,s=run_exact(False),run_exact(True); bt,st=run_cases(False),run_cases(True)
 B={r["id"]:r for r in b}; S={r["id"]:r for r in s}
 regress=[i for i in B if B[i]["exact_match"] and not S[i]["exact_match"]]
 s0=sum(r["exact_match"] for r in b); s1=sum(r["exact_match"] for r in s); t0=sum(r["pass"] for r in bt); t1=sum(r["pass"] for r in st)
 summary={"model":MODEL,"gate_fired_on_benchmark_ids":fired,"baseline_exact":[s0,72],"scoped_exact":[s1,72],"exact_regressions":regress,"baseline_drafting":[t0,8],"scoped_drafting":[t1,8],"accepted":not regress and s1>=s0 and t1>=7 and t1>t0,"weights_changed":False,"production_ready":False}
 print("DRAFT_SCOPED_SUMMARY "+json.dumps(summary),flush=True)
 print("DRAFT_SCOPED_RESULTS "+json.dumps({"baseline":bt,"scoped":st}),flush=True)
if __name__=="__main__": main()
