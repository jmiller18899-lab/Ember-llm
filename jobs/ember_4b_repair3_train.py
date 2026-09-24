# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","trl==1.13.0","accelerate==1.15.0","huggingface-hub==1.31.0","datasets==5.0.1"]
# ///
"""Repair3: smallest targeted fix on top of Repair2 for the multiply-then-add confusion.

Evidence (jobs/ember_4b_repair2_exact_diff.py, HF job 6ab4926b52d0dbd7f1d88d6c): Repair2's only lost
original exact case is arith-pack-13, "4 bundles x 9 screws + 4 extra" -> 45 (gold 40). The frozen model's one
promotion-v2 multiply miss was the same error: 4x11+9 -> 55. Both answers are n*k+k = (n+1)*k: the "extra"
items are read as one more full bundle. This run touches only that pattern (plus replay from the Repair2
curriculum) and does not train time reasoning.

Starts from Repair2 at a pinned revision (read-only) and writes only to a new repo. Trains one short epoch at
a time (at most MAX_EPOCHS). After each epoch a fast gate replays the 72 original exact cases, temporal 8 and
drafting 8. The run stops at the first epoch that passes; only then does it run the full gate
(promotion v2 200 + time held-out v1 180). No benchmark prompt, and no operand triple from any benchmark,
is used for training.
Usage:  python jobs/ember_4b_repair3_train.py --preflight   (CPU: data, contamination, tokenization)
        uv run jobs/ember_4b_repair3_train.py                (GPU: train + gate + upload)
"""
import argparse,hashlib,json,os,random,re,types,urllib.request
from pathlib import Path

SRC_COMMIT="3d559086a5a581bb0bb5b7c1adf8d7a0e65d8db0"
RAW=f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{SRC_COMMIT}/"
PINNED={
 "data/ember_4b_repair2_curriculum.jsonl":"135e2ba7925e927369431ea435db7132eb7fef8f21f10a9d82b20080057fce25",
 "jobs/ember_time_heldout_v1.py":"2023870c819feceb84fa9635fb00f2728bc2c81e2099759015bf605b0e4af65a",
 "jobs/ember_4b_promotion_v2_frozen_eval.py":"d78917ed19a5d8160b632b6c553f5b0d9ff62b8bc1ba49700a35c70d023907b7",
 "jobs/ember_4b_repair2_original_promotion_eval.py":"48272629067857fc873dfab6c71c61e82c90eb4bf5cab7d85a7f5c9108b29ffc",
}
SOURCE_MODEL="Jmiller18899/ember-qwen3.5-4b-repair2"; SOURCE_REV="daf938bba5d4e6b650ec9d34a2d3ac56706cf549"
FROZEN_MODEL="Jmiller18899/ember-qwen3.5-4b-consolidation1"
OUTPUT_REPO="Jmiller18899/ember-qwen3.5-4b-repair3"
assert OUTPUT_REPO not in (SOURCE_MODEL,FROZEN_MODEL)
BENCH="Jmiller18899/ember-generalization-benchmark-v1"
SEED=733
TRAIN={"max_epochs":3,"learning_rate":1.5e-6,"grad_accum":4,"warmup_steps":4,"max_len":256}
COUNTS={"target":64,"replay":64}
# Repair2 scores reproduced by HF jobs 6ab4926b52d0dbd7f1d88d6c (exact) and 6ab481106b030d633f68cf7a (promo/held-out).
REPAIR2_BASELINE={"exact":70,"promo_overall":178,"heldout_strict":148}
KNOWN_BENCH_TRIPLES={(4,9,4),(4,11,9)}  # arith-pack-13 and promotion-v2 mult-06: never train on these

# ---- targeted multiply-then-add rows: n groups of k, plus x loose items (x != k) ----
TARGET_TEMPLATES=[
 "There are {n} boxes with {k} pens in each box and {x} extra pens. Total pens? Number only.",
 "A shelf holds {n} crates of {k} jars each, plus {x} loose jars. How many jars in all? Number only.",
 "{n} packs contain {k} batteries each. There are also {x} spare batteries. Total batteries? Number only.",
 "A coach has {n} bags with {k} balls in each bag and {x} extra balls. How many balls? Number only.",
 "There are {n} cartons of {k} eggs and {x} more single eggs. Total eggs? Number only.",
 "We have {n} rolls with {k} tickets per roll plus {x} extra tickets. Total tickets? Number only.",
 "{n} racks hold {k} cups each and {x} cups sit outside the racks. How many cups altogether? Number only.",
 "There are {n} kits with {k} nails in each kit and {x} additional nails. Total nails? Number only.",
]
REPLAY_QUOTA={"placeholder":8,"rollover_60":6,"long_duration":5,"midnight":5,"add_subtract":10,"multiply_add":4,
 "restate_all_facts":6,"ask_for_text":6,"maintenance":2,"replay":12}
assert sum(REPLAY_QUOTA.values())==COUNTS["replay"]

def fetch(rel):
 local=Path(__file__).resolve().parents[1]/rel
 data=local.read_bytes() if local.exists() else urllib.request.urlopen(RAW+rel,timeout=60).read()
 got=hashlib.sha256(data).hexdigest()
 if got!=PINNED[rel]: raise RuntimeError(f"{rel}: sha256 {got} != pinned {PINNED[rel]}")
 return data
def module(rel,name):
 m=types.ModuleType(name); exec(compile(fetch(rel).decode(),rel,"exec"),m.__dict__); return m

def numbers(prompt): return tuple(int(x) for x in re.findall(r"\d+",prompt))
def promo_mult_triples(P):
 return {numbers(r["prompt"]) for r in P.build() if r["id"].startswith("mult-")}

def target_rows(excluded):
 r=random.Random(SEED); rows=[]; used=set()
 while len(rows)<COUNTS["target"]:
  i=len(rows); n=r.randint(2,9); k=r.randint(3,15)
  # a third of rows reuse the group count as the loose count (the arith-pack-13 shape), the rest vary freely
  x=n if i%3==0 else r.randint(1,12)
  if x==k or (n,k,x) in excluded or (n,k,x) in used: continue
  used.add((n,k,x))
  rows.append({"id":f"repair3-target-{i:02d}","family":"arithmetic","slice":"multiply_add_target",
   "prompt":TARGET_TEMPLATES[i%len(TARGET_TEMPLATES)].format(n=n,k=k,x=x),"answer":str(n*k+x),"operands":[n,k,x]})
 return rows
def replay_rows(curriculum):
 r=random.Random(SEED+1); rows=[]
 for sl,q in REPLAY_QUOTA.items():
  pool=[x for x in curriculum if x["slice"]==sl]; assert len(pool)>=q,sl
  rows+= [{**x,"id":"repair3-replay-"+x["id"]} for x in r.sample(pool,q)]
 return rows
def build(P,excluded):
 rows=target_rows(excluded|KNOWN_BENCH_TRIPLES)
 curriculum=[json.loads(l) for l in fetch("data/ember_4b_repair2_curriculum.jsonl").decode().splitlines()]
 rows+=replay_rows(curriculum)
 random.Random(SEED+2).shuffle(rows)
 return rows
def rows_digest(rows): return hashlib.sha256("".join(json.dumps(x,sort_keys=True)+"\n" for x in rows).encode()).hexdigest()

MULT_SHAPE=re.compile(r"\b(each|per|extra|loose|spare|additional)\b",re.I)
def contamination(rows,bench_prompts):
 """Training prompts must not equal a benchmark prompt, and no targeted row may reuse a benchmark's numbers."""
 bp={re.sub(r"\s+"," ",p.strip().lower()) for p in bench_prompts}
 bn={numbers(p) for p in bench_prompts if MULT_SHAPE.search(p)}
 same=[x["id"] for x in rows if re.sub(r"\s+"," ",x["prompt"].strip().lower()) in bp]
 nums=[x["id"] for x in rows if x["slice"]=="multiply_add_target" and numbers(x["prompt"]) in bn]
 return {"same_prompt":same,"same_numbers":nums}

def fresh_mult_holdout(excluded):
 """40 unseen multiply-then-add probes in the benchmark's own bundle wording (never trained on)."""
 r=random.Random(SEED+99); rows=[]; used=set()
 while len(rows)<40:
  n=r.randint(2,9); k=r.randint(3,15); x=n if len(rows)%2==0 else r.randint(1,12)
  if x==k or (n,k,x) in excluded or (n,k,x) in used: continue
  used.add((n,k,x))
  rows.append({"id":f"mult-holdout-{len(rows):02d}","prompt":f"There are {n} sealed packs with {k} clips in each pack and {x} extra clips. Total clips? Number only.","answer":str(n*k+x)})
 return rows

def encode(tok,system_for,r):
 p=tok.apply_chat_template([{"role":"system","content":system_for(r["prompt"])},{"role":"user","content":r["prompt"]}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
 a=tok.encode(r["answer"],add_special_tokens=False)+[tok.convert_tokens_to_ids("<|im_end|>")]
 if len(p)+len(a)>TRAIN["max_len"]: raise ValueError(f"row too long: {r['id']}")
 return {"input_ids":p+a,"labels":[-100]*len(p)+a}

def fast_eval(gen,O,exact,holdout):
 out={r["id"]:gen(r["prompt"]) for r in exact}
 ex={r["id"]:out[r["id"]]==r["answer"] for r in exact}
 outs={i:o for i,o in out.items() if not ex[i]}
 temp=[{"id":c["id"],"output":(o:=gen(c["prompt"])),"pass":O.temp_pass(c,o)} for c in O.TEMP]
 draft=[{"id":c["id"],"output":(o:=gen(c["prompt"])),"pass":O.draft_pass(c,o)} for c in O.DRAFT]
 hold=sum(gen(r["prompt"])==r["answer"] for r in holdout)
 return {"exact":sum(ex.values()),"exact_pass":ex,"exact_fail_outputs":outs,"temporal":sum(c["pass"] for c in temp),
  "drafting":sum(c["pass"] for c in draft),"temporal_rows":temp,"drafting_rows":draft,"mult_holdout":[hold,len(holdout)]}
def full_eval(gen,H,P):
 hres=[{**r,"output":(o:=gen(r["prompt"])),**H.score(r,o)} for r in H.build()]
 hs=H.summarize(hres); hs.pop("model",None); hs.pop("model_revision",None)
 fam={}
 for r in P.build():
  o=gen(r["prompt"]); ok=(o==r["answer"]) if r["scoring"]=="exact" else P.rubric_pass(r,o)
  f=fam.setdefault(r["family"],[0,0]); f[0]+=int(ok); f[1]+=1
 return {"heldout":{"overall":hs["overall"],"by_slice":hs["by_slice"]},"promo":{"overall":[sum(v[0] for v in fam.values()),sum(v[1] for v in fam.values())],"by_family":fam}}

def fast_gate(before,after):
 regress=[i for i,ok in before["exact_pass"].items() if ok and not after["exact_pass"][i]]
 checks={"exact_at_least_71":after["exact"]>=71,"no_exact_regressions_vs_repair2":not regress,
  "arith_pack_13_fixed":after["exact_pass"].get("arith-pack-13",False),
  "temporal_8_of_8":after["temporal"]==8,"drafting_at_least_7":after["drafting"]>=7,
  "mult_holdout_not_lower":after["mult_holdout"][0]>=before["mult_holdout"][0]}
 return {"passed":all(checks.values()),"checks":checks,"exact_regressions":regress}
def full_gate(before,after):
 fb,fa=before["promo"]["by_family"],after["promo"]["by_family"]
 hb,ha=before["heldout"]["overall"],after["heldout"]["overall"]
 checks={"promo_at_least_178":after["promo"]["overall"][0]>=178,
  **{f"promo_{f}_not_lower":fa[f][0]>=fb[f][0] for f in fb},
  "heldout_strict_not_lower":ha["strict"]>=hb["strict"],"heldout_content_not_lower":ha["content"]>=hb["content"]}
 return {"passed":all(checks.values()),"checks":checks}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--preflight",action="store_true"); a=ap.parse_args()
 H=module("jobs/ember_time_heldout_v1.py","heldout"); P=module("jobs/ember_4b_promotion_v2_frozen_eval.py","promo")
 O=module("jobs/ember_4b_repair2_original_promotion_eval.py","original")
 assert P.SYSTEM==O.SYSTEM and P.TEMPORAL_RULE==O.TEMPORAL_RULE and P.DRAFT_RULE==O.DRAFT_RULE
 excluded=promo_mult_triples(P)
 rows=build(P,excluded); holdout=fresh_mult_holdout(excluded|KNOWN_BENCH_TRIPLES|{tuple(x["operands"]) for x in rows if x["slice"]=="multiply_add_target"})
 local_bench=[r["prompt"] for r in P.build()]+[r["prompt"] for r in H.build()]+[c["prompt"] for c in O.TEMP+O.DRAFT]
 cont=contamination(rows,local_bench+[r["prompt"] for r in holdout])
 if cont["same_prompt"] or cont["same_numbers"]: raise RuntimeError(f"contamination: {cont}")
 from transformers import AutoTokenizer,set_seed
 set_seed(SEED); tok=AutoTokenizer.from_pretrained(P.BASE,revision=P.BASE_REV); tok.pad_token=tok.eos_token
 enc=[encode(tok,P.system_for,r) for r in rows]
 steps=-(-len(enc)//TRAIN["grad_accum"])
 info={"rows":len(rows),"by_slice":{s:sum(x["slice"]==s for x in rows) for s in sorted({x["slice"] for x in rows})},
  "rows_sha256":rows_digest(rows),"max_tokens":max(len(e["input_ids"]) for e in enc),"steps_per_epoch":steps,
  "rules_fired":sum(P.system_for(r["prompt"])!=P.SYSTEM for r in rows),"train":TRAIN,"source":f"{SOURCE_MODEL}@{SOURCE_REV}","output":OUTPUT_REPO}
 if a.preflight: print("REPAIR3_TRAIN_PREFLIGHT_PASS "+json.dumps(info),flush=True); return

 import torch
 from datasets import Dataset
 from huggingface_hub import HfApi,hf_hub_download
 from peft import PeftModel
 from transformers import Qwen3_5ForCausalLM,TrainingArguments,Trainer
 from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
 token=os.environ["HF_TOKEN"]; api=HfApi(token=token)
 bench=json.loads(Path(hf_hub_download(BENCH,"candidate.json",revision=api.model_info(BENCH).sha,token=token)).read_text())
 exact=[r for r in bench if r.get("scoring")=="exact"]; assert len(exact)==72
 cont=contamination(rows,[r["prompt"] for r in bench])
 if cont["same_prompt"] or cont["same_numbers"]: raise RuntimeError(f"benchmark contamination: {cont}")
 leak=[r["id"] for r in exact if O.system_for(r["prompt"])!=O.SYSTEM]
 print("REPAIR3_PLAN "+json.dumps({**info,"benchmark_contamination":cont,"rule_leakage":leak}),flush=True)

 base,li=Qwen3_5ForCausalLM.from_pretrained(P.BASE,revision=P.BASE_REV,dtype=torch.bfloat16,device_map={"":0},output_loading_info=True,key_mapping={r"^model.language_model\.":"model."})
 if li["missing_keys"] or li.get("mismatched_keys") or li.get("error_msgs"): raise RuntimeError("base load mismatch")
 model=PeftModel.from_pretrained(base,SOURCE_MODEL,revision=SOURCE_REV,is_trainable=True)
 def gen(p):
  model.eval()
  ids=tok.apply_chat_template([{"role":"system","content":O.system_for(p)},{"role":"user","content":p}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt",return_dict=False).to(model.device)
  with torch.inference_mode(): z=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
  return tok.decode(z[0,ids.shape[-1]:],skip_special_tokens=True).strip()

 fb=fast_eval(gen,O,exact,holdout); full_b=full_eval(gen,H,P)
 reproduced={"exact":fb["exact"],"promo_overall":full_b["promo"]["overall"][0],"heldout_strict":full_b["heldout"]["overall"]["strict"]}
 print("REPAIR3_BEFORE "+json.dumps({"fast":{k:v for k,v in fb.items() if k!="exact_pass"},"full":full_b,"repair2_reproduced":reproduced==REPAIR2_BASELINE,"reproduced":reproduced}),flush=True)

 ds=Dataset.from_list(enc); coll=DataCollatorForLanguageModeling(pad_token_id=tok.pad_token_id)
 history=[]; chosen=None; total_steps=0
 for ep in range(1,TRAIN["max_epochs"]+1):
  model.train(); model.config.use_cache=False
  ta=TrainingArguments(output_dir=f"repair3-run-e{ep}",num_train_epochs=1,per_device_train_batch_size=1,gradient_accumulation_steps=TRAIN["grad_accum"],
   learning_rate=TRAIN["learning_rate"],lr_scheduler_type="constant_with_warmup",warmup_steps=TRAIN["warmup_steps"] if ep==1 else 0,
   bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},save_strategy="no",logging_steps=8,report_to=[],seed=SEED+ep,data_seed=SEED+ep)
  tr=Trainer(model=model,args=ta,train_dataset=ds,data_collator=coll); tr.train(); total_steps+=tr.state.global_step
  model.config.use_cache=True
  fa=fast_eval(gen,O,exact,holdout); g=fast_gate(fb,fa)
  history.append({"epoch":ep,"steps":total_steps,"fast":{k:v for k,v in fa.items() if k!="exact_pass"},**g})
  print("REPAIR3_EPOCH "+json.dumps(history[-1]),flush=True)
  if g["passed"]: chosen=ep; break

 full_a=full_eval(gen,H,P) if chosen else None
 fg=full_gate(full_b,full_a) if chosen else {"passed":False,"checks":{},"reason":"no epoch passed the fast gate"}
 accepted=bool(chosen) and fg["passed"] and not leak
 print("REPAIR3_AFTER "+json.dumps({"epoch":chosen,"full":full_a}),flush=True)
 outdir=Path("repair3-candidate"); model.save_pretrained(str(outdir)); tok.save_pretrained(str(outdir))
 summary={**info,"repair2_reproduced":reproduced==REPAIR2_BASELINE,"before_reproduced":reproduced,"epochs":history,"chosen_epoch":chosen,
  "steps_run":total_steps,"full_before":full_b,"full_after":full_a,"full_gate":fg,"rule_leakage":leak,"accepted":accepted,"src_commit":SRC_COMMIT,"production_ready":False}
 Path("repair3-summary.json").write_text(json.dumps(summary,indent=1))
 api.create_repo(OUTPUT_REPO,private=True,exist_ok=True)
 api.upload_folder(repo_id=OUTPUT_REPO,folder_path=str(outdir),path_in_repo="candidate",commit_message="Save repair3 candidate before gate")
 api.upload_file(repo_id=OUTPUT_REPO,path_in_repo="evidence/repair3-summary.json",path_or_fileobj="repair3-summary.json",commit_message="Save repair3 gate evidence")
 if accepted: api.upload_folder(repo_id=OUTPUT_REPO,folder_path=str(outdir),path_in_repo="",commit_message="Accept repair3 candidate (experimental successor)")
 print("REPAIR3_SUMMARY "+json.dumps({"accepted":accepted,"chosen_epoch":chosen,"steps_run":total_steps,"repair2_reproduced":reproduced==REPAIR2_BASELINE,
  "full_gate":fg,"last_fast":history[-1]["fast"] if history else None,"output":OUTPUT_REPO}),flush=True)
if __name__=="__main__": main()
