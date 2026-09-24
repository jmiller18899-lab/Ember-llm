# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","trl==1.13.0","accelerate==1.15.0","huggingface-hub==1.31.0","datasets==5.0.1"]
# ///
"""Repair2: train an experimental successor to frozen promoted Ember.

Starts from the frozen promoted adapter (read-only, pinned revision), trains
on the 444-row repair2 curriculum, and writes ONLY to a new repo. The frozen
model repo is never written. Before and after training, the same run scores
time-heldout-v1 and promotion v2, then applies the acceptance gate.

The curriculum and both evaluators are fetched at a pinned commit and must
match pinned SHA-256 hashes, so the job cannot silently train or grade on
different files.
Usage:  python jobs/ember_4b_repair2_train.py --preflight   (CPU: data + tokenization)
        uv run jobs/ember_4b_repair2_train.py                (GPU: train + gate + upload)
"""
import argparse,hashlib,json,os,types,urllib.request
from pathlib import Path

SRC_COMMIT="ffa073b77155b1693f7c7fef87b27739e32e1596"
RAW=f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{SRC_COMMIT}/"
PINNED={
 "data/ember_4b_repair2_curriculum.jsonl":"135e2ba7925e927369431ea435db7132eb7fef8f21f10a9d82b20080057fce25",
 "jobs/ember_time_heldout_v1.py":"2023870c819feceb84fa9635fb00f2728bc2c81e2099759015bf605b0e4af65a",
 "jobs/ember_4b_promotion_v2_frozen_eval.py":"d78917ed19a5d8160b632b6c553f5b0d9ff62b8bc1ba49700a35c70d023907b7",
}
FROZEN_MODEL="Jmiller18899/ember-qwen3.5-4b-consolidation1"; FROZEN_REV="62e5b58b78f823a6cd720a4ff53d0adda1624210"
OUTPUT_REPO="Jmiller18899/ember-qwen3.5-4b-repair2"
assert OUTPUT_REPO!=FROZEN_MODEL
TRAIN={"epochs":2,"learning_rate":1.5e-6,"grad_accum":4,"warmup_steps":10,"seed":431,"max_len":256}
PROTECTED=("extraction","grounding","drafting","action_honesty")   # promotion v2 families that must not drop
TARGETED=("time_reasoning","arithmetic","context_consistency","clarification")
FROZEN_BASELINE={"time_heldout_strict":125,"promo_overall":151}  # reports/time_heldout_v1_frozen_baseline.md, promo v2 job

def fetch(rel):
 """Pinned file bytes: local checkout if present, otherwise GitHub at SRC_COMMIT."""
 local=Path(__file__).resolve().parents[1]/rel
 data=local.read_bytes() if local.exists() else urllib.request.urlopen(RAW+rel,timeout=60).read()
 got=hashlib.sha256(data).hexdigest()
 if got!=PINNED[rel]: raise RuntimeError(f"{rel}: sha256 {got} != pinned {PINNED[rel]}")
 return data
def module(rel,name):
 m=types.ModuleType(name); exec(compile(fetch(rel).decode(),rel,"exec"),m.__dict__); return m
def load_inputs():
 rows=[json.loads(l) for l in fetch("data/ember_4b_repair2_curriculum.jsonl").decode().splitlines()]
 assert len(rows)==444 and sum(x["slice"]=="replay" for x in rows)==44
 return rows,module("jobs/ember_time_heldout_v1.py","heldout"),module("jobs/ember_4b_promotion_v2_frozen_eval.py","promo")

def gate(before,after):
 """Acceptance rule. before/after: {"heldout":summary,"promo":summary} as produced by evaluate()."""
 hb,ha=before["heldout"]["overall"],after["heldout"]["overall"]
 fb,fa=before["promo"]["by_family"],after["promo"]["by_family"]
 checks={
  "time_heldout_strict_improves":ha["strict"]>hb["strict"],
  "time_heldout_content_not_lower":ha["content"]>=hb["content"],
  "promo_overall_not_lower":after["promo"]["overall"][0]>=before["promo"]["overall"][0],
  **{f"promo_{f}_not_lower":fa[f][0]>=fb[f][0] for f in PROTECTED+TARGETED},
 }
 return {"accepted":all(checks.values()),"checks":checks}

def encode(tok,system_for,r):
 p=tok.apply_chat_template([{"role":"system","content":system_for(r["prompt"])},{"role":"user","content":r["prompt"]}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
 a=tok.encode(r["answer"],add_special_tokens=False)+[tok.convert_tokens_to_ids("<|im_end|>")]
 if len(p)+len(a)>TRAIN["max_len"]: raise ValueError(f"row too long: {r['id']}")
 return {"input_ids":p+a,"labels":[-100]*len(p)+a}

def evaluate(gen,H,P,label):
 hres=[{**r,"output":(o:=gen(r["prompt"])),**H.score(r,o)} for r in H.build()]
 hs=H.summarize(hres); hs["configuration"]=label; hs["model"]=label; hs.pop("model_revision",None)
 pres=[]; fam={}
 for r in P.build():
  o=gen(r["prompt"]); ok=(o==r["answer"]) if r["scoring"]=="exact" else P.rubric_pass(r,o)
  pres.append({**r,"output":o,"pass":ok}); f=fam.setdefault(r["family"],[0,0]); f[0]+=int(ok); f[1]+=1
 ps={"configuration":label,"overall":[sum(x["pass"] for x in pres),len(pres)],"by_family":fam}
 return {"heldout":hs,"promo":ps},{"heldout":hres,"promo":pres}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--preflight",action="store_true"); a=ap.parse_args()
 rows,H,P=load_inputs()
 from transformers import AutoTokenizer,set_seed
 set_seed(TRAIN["seed"]); base_id,base_rev=P.BASE,P.BASE_REV
 tok=AutoTokenizer.from_pretrained(base_id,revision=base_rev); tok.pad_token=tok.eos_token
 enc=[encode(tok,P.system_for,r) for r in rows]
 from datasets import Dataset
 from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
 ds=Dataset.from_list(enc); coll=DataCollatorForLanguageModeling(pad_token_id=tok.pad_token_id)
 b=coll([ds[0],ds[1]]); assert (b["labels"]==-100).any() and (b["labels"]!=-100).any()
 steps=-(-len(ds)*TRAIN["epochs"]//TRAIN["grad_accum"])
 info={"rows":len(ds),"max_tokens":max(len(e["input_ids"]) for e in enc),"optimizer_steps":steps,"rules_fired":sum(P.system_for(r["prompt"])!=P.SYSTEM for r in rows),"train":TRAIN,"source":f"{FROZEN_MODEL}@{FROZEN_REV}","output":OUTPUT_REPO}
 if a.preflight: print("REPAIR2_TRAIN_PREFLIGHT_PASS "+json.dumps(info),flush=True); return

 import torch
 from huggingface_hub import HfApi
 from peft import PeftModel
 from transformers import Qwen3_5ForCausalLM,TrainingArguments,Trainer
 token=os.environ["HF_TOKEN"]; api=HfApi(token=token)
 base,li=Qwen3_5ForCausalLM.from_pretrained(base_id,revision=base_rev,dtype=torch.bfloat16,device_map={"":0},output_loading_info=True,key_mapping={r"^model.language_model\.":"model."})
 if li["missing_keys"] or li.get("mismatched_keys") or li.get("error_msgs"): raise RuntimeError("base load mismatch")
 model=PeftModel.from_pretrained(base,FROZEN_MODEL,revision=FROZEN_REV,is_trainable=True)
 def gen(p):
  model.eval()
  ids=tok.apply_chat_template([{"role":"system","content":P.system_for(p)},{"role":"user","content":p}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt",return_dict=False).to(model.device)
  with torch.inference_mode(): z=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
  return tok.decode(z[0,ids.shape[-1]:],skip_special_tokens=True).strip()
 before,before_rows=evaluate(gen,H,P,"frozen-promoted-ember")
 print("REPAIR2_BEFORE "+json.dumps(before),flush=True)

 model.train(); model.config.use_cache=False
 ta=TrainingArguments(output_dir="repair2-run",num_train_epochs=TRAIN["epochs"],per_device_train_batch_size=1,gradient_accumulation_steps=TRAIN["grad_accum"],learning_rate=TRAIN["learning_rate"],warmup_steps=TRAIN["warmup_steps"],bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},save_strategy="no",logging_steps=10,report_to=[],seed=TRAIN["seed"],data_seed=TRAIN["seed"])
 tr=Trainer(model=model,args=ta,train_dataset=ds,data_collator=coll); tr.train()
 outdir=Path("repair2-candidate"); tr.save_model(str(outdir)); tok.save_pretrained(str(outdir))

 after,after_rows=evaluate(gen,H,P,"repair2-candidate")
 print("REPAIR2_AFTER "+json.dumps(after),flush=True)
 verdict=gate(before,after)
 reproduced={"time_heldout_strict":before["heldout"]["overall"]["strict"],"promo_overall":before["promo"]["overall"][0]}
 summary={**info,"frozen_baseline_reproduced":reproduced==FROZEN_BASELINE,"before_scores":reproduced,"steps_run":tr.state.global_step,"src_commit":SRC_COMMIT,"pinned":PINNED,"before":before,"after":after,**verdict,"production_ready":False}
 Path("repair2-summary.json").write_text(json.dumps(summary,indent=1))
 Path("repair2-outputs.json").write_text(json.dumps({"before":before_rows,"after":after_rows}))
 api.create_repo(OUTPUT_REPO,private=True,exist_ok=True)
 api.upload_folder(repo_id=OUTPUT_REPO,folder_path=str(outdir),path_in_repo="candidate",commit_message="Save repair2 candidate before gate")
 for f in ("repair2-summary.json","repair2-outputs.json"):
  api.upload_file(repo_id=OUTPUT_REPO,path_in_repo=f"evidence/{f}",path_or_fileobj=f,commit_message="Save repair2 gate evidence")
 if verdict["accepted"]: api.upload_folder(repo_id=OUTPUT_REPO,folder_path=str(outdir),path_in_repo="",commit_message="Accept repair2 candidate (experimental successor)")
 print("REPAIR2_SUMMARY "+json.dumps({k:summary[k] for k in ("accepted","checks","frozen_baseline_reproduced","steps_run","output")}),flush=True)
if __name__=="__main__": main()
