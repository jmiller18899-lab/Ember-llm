# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","accelerate==1.15.0"]
# ///
"""Held-out time-reasoning eval v1 for the frozen promoted Ember configuration.

Inference only; never trains or modifies weights. Nine 20-case slices isolate
*why* time answers fail: placeholder-format echo vs. hour-carry arithmetic vs.
noon/midnight AM/PM handling. Each case is scored twice:
  strict  - the answer is exactly the time (e.g. "8:30 AM")
  content - the last time in the output is correct (format ignored)
Usage:  python jobs/ember_time_heldout_v1.py --preflight   (CPU: build + validate)
        uv run jobs/ember_time_heldout_v1.py                (GPU: evaluate)
"""
import json,os,re,random,sys
# ---- Frozen promoted configuration: copied verbatim from ember_4b_promotion_v2_frozen_eval.py ----
BASE="Qwen/Qwen3.5-4B"; BASE_REV="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MODEL="Jmiller18899/ember-qwen3.5-4b-consolidation1"; MODEL_REV="62e5b58b78f823a6cd720a4ff53d0adda1624210"
SYSTEM="You are Ember. Answer the current request directly and concisely. Preserve supplied facts and follow the requested format."
TEMPORAL_RULE=("Time grounding: a time that only says when the user is asking (for example 'Context: tonight' or 'I'm asking this afternoon') is not the time of the event. State an event time only if it is given explicitly (tracking, logs) or follows from a stated start time and duration. Otherwise say there is not enough information.")
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
 return SYSTEM+((" "+" ".join(rules)) if rules else "")
# ---- end frozen configuration ----
SUITE="time-heldout-v1"; SUITE_SEED=20260924
PROMO_V2_SEED=20260923  # promotion v2 suite seed; its time cases are excluded below
NATURAL_EXAMPLE="6:15 PM"; NATURAL_SUFFIX=f"Reply with only the time, formatted like {NATURAL_EXAMPLE}."
TEMPLATES=[
 "A bus departs at {t}. The ride takes {d} minutes. What time does it arrive?",
 "A meeting starts at {t} and runs for {d} minutes. What time does it end?",
 "A work shift begins at {t} and lasts {d} minutes. What time does it finish?",
 "A train leaves at {t}. The trip is {d} minutes long. What is the arrival time?",
]
SLICES=[  # (name, format, constraint) -- 20 cases each
 ("same_hour_placeholder","placeholder","same_hour"),
 ("same_hour_natural","natural","same_hour"),
 ("carry_placeholder","placeholder","carry"),
 ("carry_natural","natural","carry"),
 ("on_the_hour","natural","on_hour"),
 ("long_duration","natural","long"),
 ("noon_crossing","natural","noon"),
 ("midnight_crossing","natural","midnight"),
 ("twelve_to_one","natural","twelve_to_one"),
]
PER_SLICE=20
def fmt(mins):
 """Minutes since midnight -> 12-hour 'H:MM AM/PM'."""
 mins%=1440; h,m=divmod(mins,60)
 return f"{(h-1)%12+1}:{m:02d} {'AM' if h<12 else 'PM'}"
def parse12(s):
 h,m,ap=re.fullmatch(r"(\d{1,2}):(\d{2}) (AM|PM)",s).groups(); h,m=int(h),int(m)
 assert 1<=h<=12 and 0<=m<60
 return (h%12+(12 if ap=="PM" else 0))*60+m
def classify(start,d):
 """Which constraint a (start minute-of-day, duration) pair satisfies."""
 end=start+d; sh,sm=divmod(start%1440,60); eh=(end//60)%24
 crosses_noon=start<720<=end; crosses_mid=start<1440<=end
 twelve=lambda h:h in (0,12)
 plain=not crosses_noon and not crosses_mid and not twelve(sh) and not twelve(eh)
 tags=set()
 if plain and d<60 and sm+d<60: tags.add("same_hour")
 if plain and d<60 and sm+d>=60 and (end%60)!=0: tags.add("carry")
 if plain and d<60 and (end%60)==0: tags.add("on_hour")
 if plain and d>=75 and (end%60)!=0: tags.add("long")
 if crosses_noon and not crosses_mid: tags.add("noon")
 if crosses_mid: tags.add("midnight")
 if twelve(sh) and not crosses_noon and not crosses_mid and eh%12==1: tags.add("twelve_to_one")
 return tags
def promo_v2_time_keys():
 """(start, duration) pairs used by promotion v2 time cases -- replicates its RNG stream."""
 r=random.Random(PROMO_V2_SEED); keys=set()
 for _ in range(20): r.randint(20,90),r.randint(3,25),r.randint(2,18)
 for _ in range(20): r.randint(3,12),r.randint(3,14),r.randint(1,9)
 for i in range(30): r.randint(1000,9999)
 for i in range(30):
  h=r.randint(1,10); m=r.choice([0,5,10,15,20,25,30,35,40,45,50,55]); d=r.choice([15,20,25,30,35,40,45,50,55,60,65,70])
  ap="AM" if i%2==0 else "PM"; keys.add((parse12(f"{h}:{m:02d} {ap}"),d))
 return keys
def build():
 r=random.Random(SUITE_SEED); rows=[]; used=set(); excluded=promo_v2_time_keys()
 for name,style,need in SLICES:
  n=0; tries=0
  while n<PER_SLICE:
   tries+=1; assert tries<200000,name
   start=r.randrange(0,1440,5)
   d=r.randrange(75,181,5) if need=="long" else r.randrange(5,60,5) if need!="noon" and need!="midnight" else r.randrange(5,121,5)
   if need not in classify(start,d) or (start,d) in used or (start,d) in excluded: continue
   used.add((start,d)); t=fmt(start); ans=fmt(start+d)
   if style=="natural" and ans==NATURAL_EXAMPLE: continue
   q=TEMPLATES[n%len(TEMPLATES)].format(t=t,d=d)
   q+=" "+(f"Answer H:MM {t[-2:]}." if style=="placeholder" else NATURAL_SUFFIX)
   rows.append({"id":f"{name}-{n:02d}","slice":name,"format":style,"prompt":q,"start":t,"duration":d,"answer":ans}); n+=1
 return rows
TIME_RE=re.compile(r"(\d{1,2}):(\d{2})\s*([AaPp])\.?\s*[Mm]\.?")
def canon(out):
 """Last 12-hour time in the output, canonicalised, or None."""
 ms=TIME_RE.findall(out)
 if not ms: return None
 h,m,ap=ms[-1]; h=int(h)
 if not 1<=h<=12 or int(m)>59: return None
 return f"{h}:{m} {'AM' if ap.upper()=='A' else 'PM'}"
def score(row,out):
 o=out.strip(); strict_form=re.sub(r"\s+"," ",o).rstrip(".").strip()
 strict_form=re.sub(r"\s*([AaPp])\.?\s*[Mm]\.?$",lambda m:" "+m.group(1).upper()+"M",strict_form)
 got=canon(o); ans=row["answer"]
 return {"strict":strict_form==ans,"content":got==ans,"prefix_echo":o.startswith("H:"),"no_time":got is None,
  "copied_example":row["format"]=="natural" and got==NATURAL_EXAMPLE,
  "meridiem_ok":got is not None and got[-2:]==ans[-2:],"hour_ok":got is not None and got.split(":")[0]==ans.split(":")[0],
  "minute_ok":got is not None and got.split(":")[1][:2]==ans.split(":")[1][:2]}
def preflight():
 rows=build(); ids=[x["id"] for x in rows]
 assert len(rows)==PER_SLICE*len(SLICES)==180 and len(set(ids))==180
 assert len({x["prompt"] for x in rows})==180, "duplicate prompts"
 excl=promo_v2_time_keys(); assert len(excl)==30
 for x in rows:
  s=parse12(x["start"]); need=dict((a,c) for a,_,c in SLICES)[x["slice"]]
  assert need in classify(s,x["duration"]), x
  assert (s,x["duration"]) not in excl, x
  # independent answer check via datetime arithmetic
  import datetime as dt
  e=(dt.datetime(2026,1,1)+dt.timedelta(minutes=s+x["duration"])).strftime("%I:%M %p").lstrip("0")
  assert e==x["answer"],(x,e)
  assert not temporal_gate(x["prompt"]) and not drafting_gate(x["prompt"]), x  # frozen rules must not fire
 by={}
 for x in rows: by.setdefault(x["slice"],[]).append(x)
 assert all(x["answer"].endswith("PM") and x["start"].endswith("AM") for x in by["noon_crossing"])
 assert all(x["answer"].endswith("AM") and x["start"].endswith("PM") for x in by["midnight_crossing"])
 assert all(x["start"].startswith("12:") and x["answer"].startswith("1:") and x["start"][-2:]==x["answer"][-2:] for x in by["twelve_to_one"])
 assert all(x["answer"].endswith(":00 AM") or x["answer"].endswith(":00 PM") for x in by["on_the_hour"])
 # scorer self-tests (patterns observed on promotion v2)
 r0={"answer":"8:30 AM","format":"placeholder"}
 cases=[("8:30 AM",1,1),("8:30 am.",1,1),("H:8:30 AM",0,1),("H:MM AM",0,0),("H:00 AM",0,0),("It arrives at 8:30 a.m.",0,1),("8:30 PM",0,0),("7:15 AM, so 8:30 AM",0,1)]
 for out,st,ct in cases:
  sc=score(r0,out); assert (sc["strict"],sc["content"])==(bool(st),bool(ct)),(out,sc)
 assert score(r0,"H:8:30 AM")["prefix_echo"] and score(r0,"H:MM AM")["no_time"]
 assert score({"answer":"1:10 PM","format":"natural"},"6:15 PM")["copied_example"]
 counts={k:len(v) for k,v in by.items()}
 print("TIME_HELDOUT_PREFLIGHT_OK "+json.dumps({"suite":SUITE,"cases":len(rows),"slices":counts,"excluded_promo_v2_pairs":len(excl),"sample":[by[k][0]["prompt"]+" -> "+by[k][0]["answer"] for k in by]}),flush=True)
 return rows
def summarize(results):
 def agg(xs):
  return {"n":len(xs),"strict":sum(x["strict"] for x in xs),"content":sum(x["content"] for x in xs),"prefix_echo":sum(x["prefix_echo"] for x in xs),
   "no_time":sum(x["no_time"] for x in xs),"copied_example":sum(x["copied_example"] for x in xs),
   "meridiem_wrong":sum(not x["meridiem_ok"] and not x["no_time"] for x in xs),"hour_wrong":sum(not x["hour_ok"] and not x["no_time"] for x in xs),
   "minute_wrong":sum(not x["minute_ok"] and not x["no_time"] for x in xs)}
 by={}
 for x in results: by.setdefault(x["slice"],[]).append(x)
 return {"configuration":"frozen-promoted-ember","model":MODEL,"model_revision":MODEL_REV,"suite":SUITE,"suite_seed":SUITE_SEED,
  "overall":agg(results),"by_slice":{k:agg(v) for k,v in by.items()},"weights_changed":False}
def main():
 rows=preflight()
 import torch
 from peft import PeftModel
 from transformers import AutoTokenizer,Qwen3_5ForCausalLM
 tok=AutoTokenizer.from_pretrained(BASE,revision=BASE_REV); tok.pad_token=tok.eos_token
 base,info=Qwen3_5ForCausalLM.from_pretrained(BASE,revision=BASE_REV,dtype=torch.bfloat16,device_map={"":0},output_loading_info=True,key_mapping={r"^model.language_model\.":"model."})
 if info["missing_keys"] or info.get("mismatched_keys") or info.get("error_msgs"): raise RuntimeError("base load mismatch")
 model=PeftModel.from_pretrained(base,MODEL,revision=MODEL_REV).eval()
 def gen(p):
  ids=tok.apply_chat_template([{"role":"system","content":system_for(p)},{"role":"user","content":p}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt",return_dict=False).to(model.device)
  with torch.inference_mode(): z=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
  return tok.decode(z[0,ids.shape[-1]:],skip_special_tokens=True).strip()
 results=[{**row,"output":(o:=gen(row["prompt"])),**score(row,o)} for row in rows]
 print("TIME_HELDOUT_SUMMARY "+json.dumps(summarize(results)),flush=True)
 fails=[{k:x[k] for k in ("id","slice","prompt","answer","output")} for x in results if not x["strict"]]
 print("TIME_HELDOUT_FAILURES "+json.dumps(fails),flush=True)
if __name__=="__main__":
 preflight() if "--preflight" in sys.argv else main()
