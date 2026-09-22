# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","trl==1.13.0","accelerate==1.15.0","huggingface-hub==1.31.0","datasets==5.0.1"]
# ///
"""Aggressive arithmetic+grounding consolidation from accepted Repair1b."""
import argparse,json,os
from pathlib import Path
BASE="Qwen/Qwen3.5-4B"; BASE_REV="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SOURCE="Jmiller18899/ember-qwen3.5-4b-sft-v1-repair1b"
SYSTEM="You are Ember. Answer the current request directly and concisely. Preserve supplied facts and follow the requested format."
GAIN_IDS={"v2-confirmation-classification-0011","v2-confirmation-classification-0035","v2-confirmation-classification-0059","fresh-natural-v2-16","v3-confirm-00","v2-confirmation-extraction-0026"}

def rows(lane):
    out=[]
    def add(f,p,a): out.append({"family":f,"prompt":p,"answer":a})
    if lane=="arithmetic":
        for a,b,c in [(18,7,4),(23,11,6),(31,8,9),(16,13,5),(28,6,12),(35,9,14),(22,15,8),(41,7,16),(19,12,3),(27,10,11),(32,5,7),(24,14,9)]:
            add("arithmetic",f"A bin has {a} red items and {b} blue items. Remove {c} red items. Number only.",str(a+b-c))
        for n,k,loose in [(5,7,2),(6,8,3),(4,9,5),(7,6,4),(8,5,7),(3,12,6),(9,4,8),(5,11,3),(6,9,5),(4,13,2),(7,8,6),(8,7,4)]:
            add("arithmetic",f"{n} packets contain {k} cards each and there are {loose} loose cards. Total? Number only.",str(n*k+loose))
    elif lane=="grounding":
        pairs=[
        ("Can you simplify that?","Please share the text or topic you want simplified."),
        ("Can you shorten it?","Please paste the text you want shortened."),
        ("My train left at 3:10. When did it arrive?","I need the journey duration or actual arrival time to determine when it arrived."),
        ("The parcel was collected Wednesday. When was it delivered?","The collection day alone doesn't determine delivery. Do you have tracking or a delivery estimate?"),
        ("Did you send the update? There is no messaging tool here.","No, I haven't sent it. I can draft the update for you to send."),
        ("Did you publish the notice? There is no publishing tool here.","No, I haven't published it. I can draft the notice for you to post."),
        ("A bus left at 8:15 AM and traveled for 50 minutes. When did it arrive?","9:05 AM"),
        ("A ferry left at 2:20 PM and traveled for 35 minutes. When did it arrive?","2:55 PM"),
        ]
        for cycle in range(3):
            for p,a in pairs: add("clarification",p.replace("the ",f"the {['first','next','final'][cycle]} ",1) if cycle else p,a)
    elif lane=="drafting":
        names=["Ava","Ben","Cora","Drew","Esme","Finn","Gia","Hugo"]
        for i,n in enumerate(names):
            day=["Monday","Tuesday","Wednesday","Thursday"][i%4]
            add("drafting",f"Message {n}: I found their badge and can return it {day}.",f"Hi {n}, I found your badge and can return it {day}.")
            place=["office","studio","warehouse","lobby"][i%4]; hour=8+i%3
            add("drafting",f"Shorten without dropping the location: 'Please make sure everyone arrives at the {place} before {hour}.'",f"Please arrive at the {place} before {hour}.")
            add("drafting",f"{n} said, 'Thanks for returning my cable.' Draft my reply.",f"You're welcome, {n}! Glad I could help.")
    else: raise ValueError(lane)
    assert len(out)==24
    return out

def encode(tok,r):
    p=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":r["prompt"]}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
    a=tok.encode(r["answer"],add_special_tokens=False)+[tok.convert_tokens_to_ids("<|im_end|>")]
    if len(p)+len(a)>256: raise ValueError("row too long")
    return {"input_ids":p+a,"labels":[-100]*len(p)+a}

def gen(model,tok,r):
    import torch
    ids=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":r["prompt"]}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt",return_dict=False).to(model.device)
    with torch.inference_mode(): o=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
    return tok.decode(o[0,ids.shape[-1]:],skip_special_tokens=True).strip()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--lane",required=True,choices=["arithmetic","grounding","drafting"]); p.add_argument("--preflight",action="store_true"); a=p.parse_args()
    lane=a.lane; train=rows(lane)
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi,hf_hub_download
    from peft import PeftModel
    from transformers import AutoTokenizer,Qwen3_5ForCausalLM,TrainingArguments,Trainer,set_seed
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    set_seed(431); tok=AutoTokenizer.from_pretrained(BASE,revision=BASE_REV); tok.pad_token=tok.eos_token
    ds=Dataset.from_list([encode(tok,r) for r in train]); coll=DataCollatorForLanguageModeling(pad_token_id=tok.pad_token_id)
    b=coll([ds[0],ds[1]]); assert (b["labels"]==-100).any() and (b["labels"]!=-100).any()
    if a.preflight: print("LANE_PREFLIGHT_PASS",lane,len(train)); return
    token=os.environ["HF_TOKEN"]; api=HfApi(token=token); source_rev=api.model_info(SOURCE).sha
    # Repair1b evidence is the immutable comparison source.
    ev=Path(hf_hub_download(SOURCE,"repair-evidence/after.json",revision=source_rev,token=token))
    historical=json.loads(ev.read_text()); exact=[r for r in historical if r.get("scoring")=="exact"]
    base,info=Qwen3_5ForCausalLM.from_pretrained(BASE,revision=BASE_REV,dtype=torch.bfloat16,device_map={"":0},output_loading_info=True,key_mapping={r"^model.language_model\.":"model."})
    if info["missing_keys"] or info.get("mismatched_keys") or info.get("error_msgs"): raise RuntimeError("base load mismatch")
    model=PeftModel.from_pretrained(base,SOURCE,revision=source_rev,is_trainable=True); model.config.use_cache=False
    def evaluate():
        model.eval(); out=[]
        for r in exact:
            t=gen(model,tok,r); out.append({**r,"output":t,"exact_match":t==r["answer"]})
        return out
    before=evaluate()
    ta=TrainingArguments(output_dir="consolidation-run",max_steps=48,per_device_train_batch_size=1,gradient_accumulation_steps=4,learning_rate=1.25e-6,warmup_steps=4,bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},save_strategy="no",report_to=[],seed=431,data_seed=431)
    tr=Trainer(model=model,args=ta,train_dataset=ds,data_collator=coll); tr.train()
    outdir=Path("consolidation-candidate"); tr.save_model(str(outdir)); tok.save_pretrained(str(outdir))
    after=evaluate(); B={r["id"]:r for r in before}; A={r["id"]:r for r in after}
    regress=[i for i in B if B[i]["exact_match"] and not A[i]["exact_match"]]
    preserved=all(A[i]["exact_match"] for i in GAIN_IDS)
    score0=sum(r["exact_match"] for r in before); score1=sum(r["exact_match"] for r in after)
    summary={"lane":lane,"before_exact":[score0,len(before)],"after_exact":[score1,len(after)],"regressions":regress,"protected_gains_preserved":preserved,"steps":tr.state.global_step,"accepted":not regress and preserved and score1>=score0,"production_ready":False}
    Path("consolidation-summary.json").write_text(json.dumps(summary,indent=2))
    repo="Jmiller18899/ember-qwen3.5-4b-consolidation1"; api.create_repo(repo,private=True,exist_ok=True)
    api.upload_folder(repo_id=repo,folder_path=str(outdir),path_in_repo="candidate",commit_message="Preserve sprint candidate before gate")
    api.upload_file(repo_id=repo,path_in_repo="summary.json",path_or_fileobj="consolidation-summary.json",commit_message="Save sprint lane gate evidence")
    if summary["accepted"]: api.upload_folder(repo_id=repo,folder_path=str(outdir),path_in_repo="",commit_message="Promote accepted sprint lane candidate")
    print("CONSOLIDATION_SUMMARY "+json.dumps(summary),flush=True)
if __name__=="__main__": main()
