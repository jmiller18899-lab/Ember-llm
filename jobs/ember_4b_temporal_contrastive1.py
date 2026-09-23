# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","trl==1.13.0","accelerate==1.15.0","huggingface-hub==1.31.0","datasets==5.0.1"]
# ///
"""Contrastive temporal grounding repair from accepted Consolidation1."""
import argparse,json,os
from pathlib import Path
BASE="Qwen/Qwen3.5-4B"; BASE_REV="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SOURCE="Jmiller18899/ember-qwen3.5-4b-consolidation1"
OUTPUT="Jmiller18899/ember-qwen3.5-4b-temporal-contrastive1"
BENCH="Jmiller18899/ember-generalization-benchmark-v1"
SYSTEM="You are Ember. Answer the current request directly and concisely. Preserve supplied facts and follow the requested format."

def train_rows():
    rows=[]
    # Near-identical A/B pairs: metadata time vs event time.
    specs=[
      ("package","picked up Tuesday","delivered","this afternoon","3:14 PM"),
      ("parcel","collected Friday","delivered","this morning","8:42 AM"),
      ("flight","departed at 5:20 PM","landed","just now","6:31 PM"),
      ("train","left at 7:15 AM","arrived","earlier","8:03 AM"),
      ("courier box","picked up Monday","arrived","today","11:27 AM"),
      ("ferry","departed at 1:30 PM","arrived","this evening","2:15 PM"),
    ]
    for thing,start,event,ctx,clock in specs:
        rows.append({"prompt":f"My {thing} was {start}. When was it {event}? Context: {ctx}.",
          "answer":f"The {start} information and context '{ctx}' do not determine when it was {event}. Please provide actual {event} information or the duration."})
        rows.append({"prompt":f"My {thing} was {start}. The record says it was {event} {ctx} at {clock}. When was it {event}?",
          "answer":clock})
        rows.append({"prompt":f"My {thing} was {start}. I'm asking {ctx}. When was it {event}?",
          "answer":f"When you're asking ({ctx}) does not tell me when it was {event}. Please provide actual {event} information or the duration."})
        rows.append({"prompt":f"My {thing} was {start}. Status: {event} at {clock} {ctx}. When was it {event}?",
          "answer":clock})
    assert len(rows)==24 and len({r["prompt"] for r in rows})==24
    return rows

FRESH=[
 {"id":"contrast-01","prompt":"My shipment was collected Wednesday. When was it delivered? Context: tonight.","rubric":"Must say context 'tonight' is not delivery evidence and request delivery/tracking/duration information."},
 {"id":"contrast-02","prompt":"My shipment was collected Wednesday. Tracking says delivered tonight at 7:26 PM. When was it delivered?","answer":"7:26 PM"},
 {"id":"contrast-03","prompt":"My bus left at 9:10 AM. I'm asking this afternoon. When did it arrive?","rubric":"Must not infer this afternoon as arrival; request trip duration or arrival information."},
 {"id":"contrast-04","prompt":"My bus left at 9:10 AM. The log says it arrived this afternoon at 1:05 PM. When did it arrive?","answer":"1:05 PM"},
 {"id":"contrast-05","prompt":"A boat departed at 4:20 PM and traveled for 35 minutes. What time did it arrive?","answer":"4:55 PM"},
 {"id":"contrast-06","prompt":"Tracking: delivered earlier at 10:18 AM. What time was delivery?","answer":"10:18 AM"},
]

def encode(tok,r):
    p=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":r["prompt"]}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
    a=tok.encode(r["answer"],add_special_tokens=False)+[tok.convert_tokens_to_ids("<|im_end|>")]
    return {"input_ids":p+a,"labels":[-100]*len(p)+a}

def gen(model,tok,prompt):
    import torch
    ids=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt",return_dict=False).to(model.device)
    with torch.inference_mode(): o=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
    return tok.decode(o[0,ids.shape[-1]:],skip_special_tokens=True).strip()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--preflight",action="store_true"); a=p.parse_args()
    rows=train_rows()
    if a.preflight:
        assert not {r["prompt"] for r in rows}&{r["prompt"] for r in FRESH}
        print("CONTRASTIVE_PREFLIGHT_PASS",len(rows),len(FRESH)); return
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi,hf_hub_download
    from peft import PeftModel
    from transformers import AutoTokenizer,Qwen3_5ForCausalLM,TrainingArguments,Trainer,set_seed
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    token=os.environ["HF_TOKEN"]; api=HfApi(token=token); set_seed(431)
    tok=AutoTokenizer.from_pretrained(BASE,revision=BASE_REV); tok.pad_token=tok.eos_token
    ds=Dataset.from_list([encode(tok,r) for r in rows]); coll=DataCollatorForLanguageModeling(pad_token_id=tok.pad_token_id)
    hist=json.load(open(hf_hub_download("Jmiller18899/ember-qwen3.5-4b-sft-v1-repair1b","repair-evidence/after.json",token=token)))
    hist=[r for r in hist if r.get("scoring")=="exact"]
    fresh=json.load(open(hf_hub_download(BENCH,"candidate.json",token=token)))
    fresh_exact=[r for r in fresh if r.get("scoring")=="exact"]
    base,info=Qwen3_5ForCausalLM.from_pretrained(BASE,revision=BASE_REV,dtype=torch.bfloat16,device_map={"":0},output_loading_info=True,key_mapping={r"^model.language_model\.":"model."})
    if info["missing_keys"] or info.get("mismatched_keys") or info.get("error_msgs"): raise RuntimeError("base load mismatch")
    model=PeftModel.from_pretrained(base,SOURCE,is_trainable=True); model.config.use_cache=False
    def eval_rows(rs):
        model.eval(); out=[]
        for r in rs:
            text=gen(model,tok,r["prompt"])
            out.append({**r,"new_output":text,"new_exact":None if "answer" not in r else text.rstrip(".")==r["answer"].rstrip(".")})
        return out
    before_hist=eval_rows(hist); before_fresh=eval_rows(fresh_exact)
    ta=TrainingArguments(output_dir="contrastive-run",max_steps=12,per_device_train_batch_size=1,gradient_accumulation_steps=4,learning_rate=4e-7,warmup_steps=2,bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},save_strategy="no",report_to=[],seed=431,data_seed=431)
    tr=Trainer(model=model,args=ta,train_dataset=ds,data_collator=coll); tr.train()
    out=Path("contrastive-candidate"); tr.save_model(str(out)); tok.save_pretrained(str(out))
    after_hist=eval_rows(hist); after_fresh=eval_rows(fresh_exact); targeted=eval_rows(FRESH)
    bh={r["id"]:r for r in before_hist}; ah={r["id"]:r for r in after_hist}; bf={r["id"]:r for r in before_fresh}; af={r["id"]:r for r in after_fresh}
    regress_hist=[i for i in bh if bh[i]["new_exact"] and not ah[i]["new_exact"]]
    regress_fresh=[i for i in bf if bf[i]["new_exact"] and not af[i]["new_exact"]]
    score_hist=sum(bool(r["new_exact"]) for r in after_hist); score_fresh=sum(bool(r["new_exact"]) for r in after_fresh)
    targeted_exact=[r for r in targeted if "answer" in r]
    accepted=not regress_hist and not regress_fresh and score_hist>=68 and score_fresh>=71 and all(r["new_exact"] for r in targeted_exact)
    summary={"source":SOURCE,"method":"contrastive_temporal_pairs","steps":tr.state.global_step,"lr":4e-7,"historical_exact":[score_hist,len(after_hist)],"fresh_exact":[score_fresh,len(after_fresh)],"historical_regressions":regress_hist,"fresh_regressions":regress_fresh,"targeted_exact":[sum(bool(r["new_exact"]) for r in targeted_exact),len(targeted_exact)],"accepted":accepted,"targeted_rubric_review_required":True,"production_ready":False}
    Path("contrastive-summary.json").write_text(json.dumps(summary,indent=2)); Path("contrastive-targeted.json").write_text(json.dumps(targeted,indent=2))
    api.create_repo(OUTPUT,private=True,exist_ok=True)
    api.upload_folder(repo_id=OUTPUT,folder_path=str(out),path_in_repo="candidate",commit_message="Preserve contrastive temporal candidate")
    api.upload_file(repo_id=OUTPUT,path_in_repo="summary.json",path_or_fileobj="contrastive-summary.json",commit_message="Save contrastive temporal gate")
    api.upload_file(repo_id=OUTPUT,path_in_repo="targeted.json",path_or_fileobj="contrastive-targeted.json",commit_message="Save fresh contrastive temporal review")
    if accepted: api.upload_folder(repo_id=OUTPUT,folder_path=str(out),path_in_repo="",commit_message="Promote gated contrastive temporal candidate")
    print("CONTRASTIVE_SUMMARY "+json.dumps(summary),flush=True)
    print("CONTRASTIVE_TARGETED "+json.dumps(targeted),flush=True)
if __name__=="__main__": main()
