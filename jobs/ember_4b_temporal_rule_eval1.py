# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","accelerate==1.15.0","huggingface-hub==1.31.0"]
# ///
"""Inference-only temporal grounding rule test on untouched Consolidation1."""
import json,os
from pathlib import Path
BASE="Qwen/Qwen3.5-4B"; BASE_REV="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MODEL="Jmiller18899/ember-qwen3.5-4b-consolidation1"
BENCH="Jmiller18899/ember-generalization-benchmark-v1"
SYSTEM_BASE="You are Ember. Answer the current request directly and concisely. Preserve supplied facts and follow the requested format."
RULE=" Temporal words that describe when the user asks, writes, or provides context are not evidence of when an event occurred. Infer an event time only from explicit event timestamps, durations, tracking, logs, records, or other event evidence. If event timing cannot be determined, state what information is missing and ask for it."
SYSTEM=SYSTEM_BASE+RULE
TARGETS=[
 {"id":"rule-01","prompt":"My shipment was collected Wednesday. When was it delivered? Context: tonight.","must":["not enough","tracking"],"forbid":["delivered tonight","Tonight."]},
 {"id":"rule-02","prompt":"My shipment was collected Wednesday. Tracking says delivered tonight at 7:26 PM. When was it delivered?","answer":"7:26 PM"},
 {"id":"rule-03","prompt":"My bus left at 9:10 AM. I'm asking this afternoon. When did it arrive?","must":["duration"],"forbid":["arrived this afternoon"]},
 {"id":"rule-04","prompt":"My bus left at 9:10 AM. The log says it arrived this afternoon at 1:05 PM. When did it arrive?","answer":"1:05 PM"},
 {"id":"rule-05","prompt":"My package was picked up Tuesday. When did it arrive? Context: this afternoon.","must":["not enough"],"forbid":["arrived this afternoon"]},
 {"id":"rule-06","prompt":"The delivery record says it arrived this afternoon at 3:17 PM. What time did it arrive?","answer":"3:17 PM"},
 {"id":"rule-07","prompt":"A ferry departed at 1:30 PM and traveled for 45 minutes. What time did it arrive?","answer":"2:15 PM"},
 {"id":"rule-08","prompt":"Tracking: delivered earlier at 10:18 AM. What time was delivery?","answer":"10:18 AM"},
]
def gen(model,tok,prompt):
 import torch
 ids=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt",return_dict=False).to(model.device)
 with torch.inference_mode(): o=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
 return tok.decode(o[0,ids.shape[-1]:],skip_special_tokens=True).strip()
def norm(s): return s.strip().rstrip(".").lower()
def main():
 import torch
 from huggingface_hub import HfApi,hf_hub_download
 from peft import PeftModel
 from transformers import AutoTokenizer,Qwen3_5ForCausalLM
 token=os.environ["HF_TOKEN"]; api=HfApi(token=token)
 rows=json.load(open(hf_hub_download(BENCH,"candidate.json",token=token)))
 exact=[r for r in rows if r["scoring"]=="exact"]
 tok=AutoTokenizer.from_pretrained(BASE,revision=BASE_REV); tok.pad_token=tok.eos_token
 base,info=Qwen3_5ForCausalLM.from_pretrained(BASE,revision=BASE_REV,dtype=torch.bfloat16,device_map={"":0},output_loading_info=True,key_mapping={r"^model.language_model\.":"model."})
 if info["missing_keys"] or info.get("mismatched_keys") or info.get("error_msgs"): raise RuntimeError("base load mismatch")
 model=PeftModel.from_pretrained(base,MODEL,is_trainable=False); model.eval()
 scored=[]
 for r in exact:
  out=gen(model,tok,r["prompt"]); scored.append({**r,"rule_output":out,"rule_exact":norm(out)==norm(r["answer"])})
 fam={}
 for r in scored:
  fam.setdefault(r["family"],[0,0]); fam[r["family"]][1]+=1; fam[r["family"]][0]+=int(r["rule_exact"])
 targets=[]
 for r in TARGETS:
  out=gen(model,tok,r["prompt"]); low=out.lower()
  if "answer" in r: passed=norm(r["answer"]) in norm(out)
  else: passed=all(x.lower() in low for x in r.get("must",[])) and not any(x.lower() in low for x in r.get("forbid",[]))
  targets.append({**r,"output":out,"passed":passed})
 summary={"method":"inference_only_temporal_rule","model":MODEL,"weight_changes":0,"exact_pass":sum(r["rule_exact"] for r in scored),"exact_total":len(scored),"by_family":fam,"targeted_pass":sum(r["passed"] for r in targets),"targeted_total":len(targets),"production_ready":False}
 out=Path("temporal-rule-eval"); out.mkdir(exist_ok=True)
 (out/"summary.json").write_text(json.dumps(summary,indent=2)); (out/"exact.json").write_text(json.dumps(scored,indent=2)); (out/"targeted.json").write_text(json.dumps(targets,indent=2))
 repo="Jmiller18899/ember-temporal-rule-eval1"; api.create_repo(repo,private=True,exist_ok=True); api.upload_folder(repo_id=repo,folder_path=str(out),path_in_repo="",commit_message="Save inference-only temporal rule evaluation")
 print("RULE_SUMMARY "+json.dumps(summary),flush=True); print("RULE_TARGETS "+json.dumps(targets),flush=True)
if __name__=="__main__": main()
