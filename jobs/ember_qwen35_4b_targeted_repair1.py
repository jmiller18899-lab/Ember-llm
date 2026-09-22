# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","trl==1.13.0","accelerate==1.15.0","huggingface-hub==1.31.0","datasets==5.0.1"]
# ///
"""Small continued-LoRA repair for Ember 4B. Experimental only; no deployment."""
import argparse, hashlib, json, os, time
from pathlib import Path

BASE="Qwen/Qwen3.5-4B"
BASE_REV="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SOURCE="Jmiller18899/ember-qwen3.5-4b-sft-v1"
OUTPUT="Jmiller18899/ember-qwen3.5-4b-sft-v1-repair1"
SYSTEM="You are Ember. Answer the current request directly and concisely. Preserve supplied facts and follow the requested format."
SEED=431
GAIN_IDS={"v2-confirmation-classification-0011","v2-confirmation-classification-0035","v2-confirmation-classification-0059","fresh-natural-v2-16","v3-confirm-00"}
FAIL_IDS={"v2-confirmation-extraction-0026","v3-confirm-02","v3-confirm-03"}

def repair_rows():
    rows=[]
    def add(f,p,a): rows.append({"family":f,"prompt":p,"answer":a})
    # Extraction: return the requested field only, never the whole record.
    for who,code,state in [("Ari","K-381_z","ready"),("Bea","M-204_q","open"),("Cleo","R-915_x","done"),("Dax","P-067_v","hold"),("Eli","N-442_k","ready"),("Fia","S-730_j","open")]:
        add("extraction",f"Extract only the code from 'owner={who}; code={code}; state={state}'.",code)
    # Arithmetic composition: mixed add/subtract and multiplication-plus-loose-item patterns.
    for a,b,c in [(21,8,5),(33,6,14),(18,11,7),(29,9,12),(24,13,8),(37,5,16)]:
        add("arithmetic",f"A box has {a} red pieces and {b} blue pieces. Remove {c} red pieces. Number only.",str(a+b-c))
    for packs,each,loose in [(4,7,2),(6,5,4),(3,11,5),(7,4,3),(8,3,6),(5,8,7)]:
        add("arithmetic",f"{packs} packets contain {each} cards each, plus {loose} loose cards. Total cards? Number only.",str(packs*each+loose))
    # Clarification: identify the missing input and offer the useful next action.
    clar=[
      ("Can you make that simpler?","Please share the text or topic you want simplified."),
      ("Could you tighten the wording?","Please paste the wording you want tightened."),
      ("When did my ferry arrive? I only know it left at 9:20.","I need the trip duration or actual arrival time to determine when it arrived."),
      ("A parcel was collected Tuesday. When was it delivered?","The collection day alone doesn't determine delivery. Do you have tracking or a delivery estimate?"),
      ("Did you publish my update? There is no publishing tool here.","No, I haven't published it. I can draft the update for you to post."),
      ("Did you send the reminder? There is no messaging tool here.","No, I haven't sent it. I can draft the reminder for you to send."),
    ]
    for p,a in clar: add("clarification",p,a)
    # Fact preservation and drafting perspective.
    writing=[
      ("Shorten this without dropping the location: 'Please make sure everyone arrives at the warehouse before eight.'","Please arrive at the warehouse before eight."),
      ("Shorten this without dropping the location: 'We ask all visitors to meet at the lobby before noon.'","Meet at the lobby before noon."),
      ("Message Lena: I found her charger and can return it Friday.","Hi Lena, I found your charger and can return it Friday."),
      ("Message Omar: I received his folder and will bring it tomorrow.","Hi Omar, I received your folder and will bring it tomorrow."),
      ("Rina said, 'Thanks for fixing the shelf.' Draft my reply.","You're welcome, Rina! Glad I could help."),
      ("Theo said, 'Thanks for returning my keys.' Draft my reply.","You're welcome, Theo! Happy to help."),
    ]
    for p,a in writing: add("drafting",p,a)
    assert len(rows)==30
    return rows

FRESH=[
 {"id":"repair-fresh-extract","family":"extraction","prompt":"From 'owner=Uma; code=Q-518_r; state=ready', give only the code.","answer":"Q-518_r","scoring":"exact"},
 {"id":"repair-fresh-math1","family":"arithmetic","prompt":"A tray has 26 red tokens and 9 white tokens. Remove 7 red tokens. Number only.","answer":"28","scoring":"exact"},
 {"id":"repair-fresh-math2","family":"arithmetic","prompt":"Seven packs have 6 cards each, with 5 loose cards. Total? Number only.","answer":"47","scoring":"exact"},
 {"id":"repair-fresh-clarify","family":"clarification","prompt":"Could you rewrite that for a beginner?","answer":None,"scoring":"rubric","rubric":"Ask for the missing text or topic; do not refuse or invent source content."},
 {"id":"repair-fresh-location","family":"drafting","prompt":"Shorten without losing the place: 'Everyone needs to be at the workshop before ten.'","answer":None,"scoring":"rubric","rubric":"Retain workshop and before ten."},
 {"id":"repair-fresh-perspective","family":"drafting","prompt":"Message Nia: I found her badge and can bring it Monday.","answer":None,"scoring":"rubric","rubric":"Draft from sender to Nia; retain badge and Monday."},
]

def encode(tok,row):
    prompt=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":row["prompt"]}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
    ans=tok.encode(row["answer"],add_special_tokens=False)+[tok.convert_tokens_to_ids("<|im_end|>")]
    if len(prompt)+len(ans)>256: raise ValueError("repair row too long")
    return {"input_ids":prompt+ans,"labels":[-100]*len(prompt)+ans}

def load_eval(api, revision):
    from huggingface_hub import hf_hub_download
    p=hf_hub_download(SOURCE,"evaluation/final.json",revision=revision,token=api.token)
    rows=json.loads(Path(p).read_text())
    byid={r["id"]:r for r in rows}
    if not (GAIN_IDS|FAIL_IDS)<=set(byid): raise ValueError("required historical gates missing")
    return rows

def generate(model,tok,row):
    import torch
    ids=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":row["prompt"]}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt",return_dict=False).to(model.device)
    with torch.inference_mode():
        out=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
    return tok.decode(out[0,ids.shape[-1]:],skip_special_tokens=True).strip()

def exact_score(rows): return sum(r.get("exact_match") is True for r in rows if r.get("scoring")=="exact"),sum(r.get("scoring")=="exact" for r in rows)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--preflight",action="store_true"); args=p.parse_args()
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoTokenizer,Qwen3_5ForCausalLM,Qwen3_5TextConfig,TrainingArguments,Trainer,set_seed
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    set_seed(SEED); torch.set_num_threads(2)
    rows=repair_rows()
    # Training prompts must stay disjoint from the fresh holdout.
    assert not {r["prompt"] for r in rows}&{r["prompt"] for r in FRESH}
    tok=AutoTokenizer.from_pretrained(BASE,revision=BASE_REV); tok.pad_token=tok.eos_token
    ds=Dataset.from_list([encode(tok,r) for r in rows])
    collator=DataCollatorForLanguageModeling(pad_token_id=tok.pad_token_id)
    batch=collator([ds[0],ds[1]]); assert (batch["labels"]==-100).any() and (batch["labels"]!=-100).any()
    if args.preflight:
        cfg=Qwen3_5TextConfig(vocab_size=len(tok),hidden_size=32,intermediate_size=64,num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,head_dim=16,linear_num_key_heads=2,linear_num_value_heads=2,linear_key_head_dim=16,linear_value_head_dim=16,layer_types=["linear_attention"]*3+["full_attention"],rope_parameters={"rope_type":"default","rope_theta":10000.,"partial_rotary_factor":1.,"mrope_section":[2,3,3]})
        m=Qwen3_5ForCausalLM(cfg)
        # Validate encoded repair corpus only; continued-adapter mechanics require real checkpoint.
        o=m(input_ids=torch.tensor([ds[0]["input_ids"]]),labels=torch.tensor([ds[0]["labels"]]))
        assert torch.isfinite(o.loss); print("REPAIR_PREFLIGHT_PASS",len(rows),len(FRESH),float(o.loss)); return
    if not torch.cuda.is_available(): raise RuntimeError("GPU required")
    token=os.environ.get("HF_TOKEN")
    if not token: raise RuntimeError("HF_TOKEN required")
    api=HfApi(token=token)
    if api.whoami()["name"].lower()!="jmiller18899": raise RuntimeError("unexpected HF account")
    source_info=api.model_info(SOURCE); source_rev=source_info.sha
    if not source_info.private: raise RuntimeError("source adapter must remain private")
    api.create_repo(OUTPUT,private=True,exist_ok=True)
    if not api.model_info(OUTPUT).private: raise RuntimeError("output must remain private")
    historical=load_eval(api,source_rev)
    base,loading=Qwen3_5ForCausalLM.from_pretrained(BASE,revision=BASE_REV,dtype=torch.bfloat16,device_map={"":0},output_loading_info=True,key_mapping={r"^model.language_model\.":"model."})
    if loading["missing_keys"] or loading.get("mismatched_keys") or loading.get("error_msgs"): raise RuntimeError("base loading mismatch")
    model=PeftModel.from_pretrained(base,SOURCE,revision=source_rev,is_trainable=True)
    model.config.use_cache=False
    if not all("lora_" in n for n,p in model.named_parameters() if p.requires_grad): raise RuntimeError("non-LoRA trainable parameter")
    out=Path("ember-repair1-results"); out.mkdir(exist_ok=True)
    manifest={"source_adapter":SOURCE,"source_revision":source_rev,"base":BASE,"base_revision":BASE_REV,"seed":SEED,"training_examples":len(rows),"max_steps":24,"learning_rate":1e-6,"production_ready":False,"fresh_holdout_count":len(FRESH),"hard_gain_ids":sorted(GAIN_IDS),"known_failure_ids":sorted(FAIL_IDS)}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    # Pre-repair replay of all exact historical cases plus focused rubrics/fresh holdout.
    selected=[r for r in historical if r["scoring"]=="exact" or r["id"] in {"v3-confirm-08","v3-confirm-11","v3-confirm-12","v3-confirm-15"}]+FRESH
    def evaluate(label):
        model.eval(); result=[]
        for r in selected:
            text=generate(model,tok,r)
            result.append({**r,"output":text,"exact_match":text==r.get("answer") if r.get("scoring")=="exact" else None})
        (out/f"{label}.json").write_text(json.dumps(result,indent=2)); return result
    before=evaluate("before")
    args2=TrainingArguments(output_dir=str(out/"checkpoints"),max_steps=24,per_device_train_batch_size=1,gradient_accumulation_steps=4,learning_rate=1e-6,warmup_steps=2,bf16=True,fp16=False,gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},logging_steps=2,save_strategy="no",report_to=[],seed=SEED,data_seed=SEED)
    trainer=Trainer(model=model,args=args2,train_dataset=ds,data_collator=collator)
    trainer.train(); trainer.save_model(str(out/"adapter")); tok.save_pretrained(str(out/"adapter"))
    after=evaluate("after")
    b={r["id"]:r for r in before}; a={r["id"]:r for r in after}
    before_score=exact_score(before); after_score=exact_score(after)
    preserved=[i for i in GAIN_IDS if a[i]["exact_match"]]
    repaired=[i for i in FAIL_IDS if a[i]["exact_match"]]
    exact_regressions=[i for i in b if b[i].get("exact_match") is True and a[i].get("exact_match") is False]
    fresh_exact=[r for r in after if r["id"].startswith("repair-fresh") and r["scoring"]=="exact"]
    gate_conditions={
        "no_exact_regressions": len(exact_regressions)==0,
        "all_five_gains_preserved": set(preserved)==GAIN_IDS,
        "overall_exact_not_lower": after_score[0]>=before_score[0],
        "overall_exact_improved": after_score[0]>before_score[0],
    }
    accepted=gate_conditions["no_exact_regressions"] and gate_conditions["all_five_gains_preserved"] and gate_conditions["overall_exact_not_lower"]
    summary={"before_exact":before_score,"after_exact":after_score,"preserved_historical_gains":sorted(preserved),"repaired_known_failures":sorted(repaired),"exact_regressions":exact_regressions,"fresh_exact_pass":sum(r["exact_match"] for r in fresh_exact),"fresh_exact_total":len(fresh_exact),"human_review_pending":True,"production_ready":False,"steps":trainer.state.global_step,"gate_conditions":gate_conditions,"accepted_for_candidate_repo":accepted}
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print("GATE_CONDITIONS "+json.dumps(gate_conditions,sort_keys=True),flush=True)
    quarantine="quarantine/candidate-before-gate"
    api.upload_folder(repo_id=OUTPUT,folder_path=str(out/"adapter"),path_in_repo=quarantine+"/adapter",commit_message="Quarantine Ember repair candidate before quality gate")
    api.upload_folder(repo_id=OUTPUT,folder_path=str(out),path_in_repo=quarantine+"/evidence",allow_patterns=["*.json"],commit_message="Preserve Ember repair evidence before quality gate")
    print("CANDIDATE_PRESERVED "+json.dumps({"path":quarantine,"accepted":accepted}),flush=True)
    if not accepted:
        raise RuntimeError("repair preserved but rejected by exact regression gate: "+json.dumps(summary))
    api.upload_folder(repo_id=OUTPUT,folder_path=str(out/"adapter"),path_in_repo="",commit_message="Promote gated Ember 4B targeted repair candidate")
    api.upload_folder(repo_id=OUTPUT,folder_path=str(out),path_in_repo="repair-evidence",allow_patterns=["*.json"],commit_message="Save accepted targeted repair evidence")
    print("REPAIR_SUMMARY "+json.dumps(summary),flush=True)

if __name__=="__main__": main()
