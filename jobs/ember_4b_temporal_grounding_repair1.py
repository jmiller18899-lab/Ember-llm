# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","trl==1.13.0","accelerate==1.15.0","huggingface-hub==1.31.0","datasets==5.0.1"]
# ///
"""Small temporal-context grounding repair from accepted consolidation1."""
import argparse,json,os
from pathlib import Path
BASE="Qwen/Qwen3.5-4B"; BASE_REV="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SOURCE="Jmiller18899/ember-qwen3.5-4b-consolidation1"
OUTPUT="Jmiller18899/ember-qwen3.5-4b-temporal-grounding1"
BENCH="Jmiller18899/ember-generalization-benchmark-v1"
SYSTEM="You are Ember. Answer the current request directly and concisely. Preserve supplied facts and follow the requested format."

def train_rows():
    contexts=["this morning","just now","today","earlier","for this task","in this chat"]
    rows=[]
    for i,ctx in enumerate(contexts):
        rows += [
          {"prompt":f"My parcel was collected Thursday. When was it delivered? Context: {ctx}.","answer":"The collection time and context do not tell me when it was delivered. Please provide tracking, transit duration, or delivery information."},
          {"prompt":f"My flight departed at 4:40 PM. When did it land? Context: {ctx}.","answer":"The departure time and context are not enough to determine landing time. Please provide the flight duration, flight details, or arrival information."},
          {"prompt":f"A courier picked up my box Monday. What time did it arrive? Note written {ctx}.","answer":"The pickup information and note timing do not determine the arrival time. Please provide tracking or delivery information."},
          {"prompt":f"My train left at 7:15 AM. What time did it arrive? Asked {ctx}.","answer":"The departure time and when you asked do not determine arrival time. Please provide the journey duration or arrival information."},
        ]
    # Positive controls: temporal phrases ARE evidence when explicitly attached to the event.
    rows += [
      {"prompt":"The tracking page says: Delivered today at 9:12 AM. What time was it delivered?","answer":"9:12 AM"},
      {"prompt":"The carrier says the package was delivered this morning at 8:45. What time was delivery?","answer":"8:45"},
      {"prompt":"A shuttle left at 2:10 PM and traveled for 35 minutes. When did it arrive?","answer":"2:45 PM"},
      {"prompt":"The flight log says it landed just now at 6:22 PM. What time did it land?","answer":"6:22 PM"},
      {"prompt":"Tracking status: delivered earlier at 11:03 AM. What time was delivery?","answer":"11:03 AM"},
      {"prompt":"A bus left at 9:25 AM and the trip took 50 minutes. Arrival time?","answer":"10:15 AM"},
    ]
    assert len(rows)==30 and len({r["prompt"] for r in rows})==30
    return rows

FRESH=[
 {"id":"temp-fresh-01","prompt":"My package was picked up Tuesday. When did it arrive? Context: this afternoon.","rubric":"Must say the supplied pickup/context is insufficient; must not claim this afternoon is the arrival time."},
 {"id":"temp-fresh-02","prompt":"My plane took off at 5:20 PM. When did it land? I'm asking just now.","rubric":"Must request duration/arrival/flight information; must not use 'just now' as landing evidence."},
 {"id":"temp-fresh-03","prompt":"The delivery record says it arrived this afternoon at 3:17 PM. What time did it arrive?","answer":"3:17 PM"},
 {"id":"temp-fresh-04","prompt":"A ferry departed at 1:30 PM and traveled for 45 minutes. What time did it arrive?","answer":"2:15 PM"},
]

def encode(tok,r):
    p=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":r["prompt"]}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
    a=tok.encode(r["answer"],add_special_tokens=False)+[tok.convert_tokens_to_ids("<|im_end|>")]
    return {"input_ids":p+a,"labels":[-100]*len(p)+a}

def gen(model,tok,prompt):
    import torch
    ids=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt",return_dict=False).to(model.device)
    with torch.inference_mode(): out=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
    return tok.decode(out[0,ids.shape[-1]:],skip_special_tokens=True).strip()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--preflight",action="store_true"); a=p.parse_args()
    rows=train_rows()
    if a.preflight:
        assert not {r["prompt"] for r in rows}&{r["prompt"] for r in FRESH}
        print("TEMPORAL_PREFLIGHT_PASS",len(rows),len(FRESH)); return
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi,hf_hub_download
    from peft import PeftModel
    from transformers import AutoTokenizer,Qwen3_5ForCausalLM,TrainingArguments,Trainer,set_seed
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    token=os.environ["HF_TOKEN"]; api=HfApi(token=token); set_seed(431)
    tok=AutoTokenizer.from_pretrained(BASE,revision=BASE_REV); tok.pad_token=tok.eos_token
    ds=Dataset.from_list([encode(tok,r) for r in rows]); coll=DataCollatorForLanguageModeling(pad_token_id=tok.pad_token_id)
    # Historical exact suite from Repair1b remains a hard regression gate.
    hist=json.load(open(hf_hub_download("Jmiller18899/ember-qwen3.5-4b-sft-v1-repair1b","repair-evidence/after.json",token=token)))
    hist=[r for r in hist if r.get("scoring")=="exact"]
    # Fresh exact benchmark provides the second hard gate.
    fresh=json.load(open(hf_hub_download(BENCH,"candidate.json",token=token)))
    fresh_exact=[r for r in fresh if r.get("scoring")=="exact"]
    base,info=Qwen3_5ForCausalLM.from_pretrained(BASE,revision=BASE_REV,dtype=torch.bfloat16,device_map={"":0},output_loading_info=True,key_mapping={r"^model.language_model\.":"model."})
    if info["missing_keys"] or info.get("mismatched_keys") or info.get("error_msgs"): raise RuntimeError("base load mismatch")
    model=PeftModel.from_pretrained(base,SOURCE,is_trainable=True); model.config.use_cache=False
    def eval_rows(rs):
        model.eval(); return [{**r,"new_output":gen(model,tok,r["prompt"]),"new_exact":None if r.get("scoring")=="rubric" else gen(model,tok,r["prompt"])==r.get("answer")} for r in rs]
    before_hist=eval_rows(hist); before_fresh=eval_rows(fresh_exact)
    ta=TrainingArguments(output_dir="temporal-run",max_steps=24,per_device_train_batch_size=1,gradient_accumulation_steps=4,learning_rate=8e-7,warmup_steps=3,bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},save_strategy="no",report_to=[],seed=431,data_seed=431)
    tr=Trainer(model=model,args=ta,train_dataset=ds,data_collator=coll); tr.train()
    out=Path("temporal-candidate"); tr.save_model(str(out)); tok.save_pretrained(str(out))
    after_hist=eval_rows(hist); after_fresh=eval_rows(fresh_exact); targeted=eval_rows(FRESH)
    bh={r["id"]:r for r in before_hist}; ah={r["id"]:r for r in after_hist}; bf={r["id"]:r for r in before_fresh}; af={r["id"]:r for r in after_fresh}
    regress_hist=[i for i in bh if bh[i]["new_exact"] and not ah[i]["new_exact"]]
    regress_fresh=[i for i in bf if bf[i]["new_exact"] and not af[i]["new_exact"]]
    score_hist=sum(r["new_exact"] for r in after_hist); score_fresh=sum(r["new_exact"] for r in after_fresh)
    accepted=not regress_hist and not regress_fresh and score_hist>=68 and score_fresh>=71
    summary={"source":SOURCE,"steps":tr.state.global_step,"historical_exact":[score_hist,len(after_hist)],"fresh_exact":[score_fresh,len(after_fresh)],"historical_regressions":regress_hist,"fresh_regressions":regress_fresh,"accepted":accepted,"targeted_review_required":True,"production_ready":False}
    Path("temporal-summary.json").write_text(json.dumps(summary,indent=2)); Path("temporal-targeted.json").write_text(json.dumps(targeted,indent=2))
    api.create_repo(OUTPUT,private=True,exist_ok=True)
    api.upload_folder(repo_id=OUTPUT,folder_path=str(out),path_in_repo="candidate",commit_message="Preserve temporal grounding candidate")
    api.upload_file(repo_id=OUTPUT,path_in_repo="summary.json",path_or_fileobj="temporal-summary.json",commit_message="Save temporal repair gate")
    api.upload_file(repo_id=OUTPUT,path_in_repo="targeted.json",path_or_fileobj="temporal-targeted.json",commit_message="Save fresh temporal review cases")
    if accepted: api.upload_folder(repo_id=OUTPUT,folder_path=str(out),path_in_repo="",commit_message="Promote gated temporal grounding candidate")
    print("TEMPORAL_SUMMARY "+json.dumps(summary),flush=True)
    print("TEMPORAL_TARGETED "+json.dumps(targeted),flush=True)
if __name__=="__main__": main()
