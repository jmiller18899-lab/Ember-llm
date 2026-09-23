# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","accelerate==1.15.0"]
# ///
"""Frozen promoted Ember configuration on promotion suite v2 (200 cases)."""
import json,os,re,runpy
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
ABSTAIN_RE=re.compile(r"(not enough|insufficient|(don'?t|do not|can'?t|cannot|can not) (know|tell|determine|say|be sure)|unknown|(not|isn'?t|wasn'?t) (stated|given|specified|provided)|(i|you)(\s+would)? need|need (the|a|more|tracking)|tracking|duration|arrival information)",re.I)
def norm(s): return re.sub(r"\s+"," ",s.lower().replace("’","'")).strip()
def rubric_pass(row,out):
 o=norm(out); rub=norm(row["rubric"]); fam=row["family"]
 if fam=="grounding":
  return bool(ABSTAIN_RE.search(o)) and not re.search(r"\b(delivered|arrived) (this morning|tonight|right now|this afternoon|earlier)\b",o)
 if fam=="clarification":
  return bool(re.search(r"(send|share|paste|provide|what|which).*(text|content|passage|message)|what.*(shorten|rewrite|summarize|simplify|proofread)",o,re.I))
 if fam=="drafting":
  if row["id"].startswith("draft-msg"):
   m=re.search(r"Message ([^:]+): I found their ([^ ]+) and can return it ([^.]+)",row["prompt"])
   return bool(m and norm(m.group(1)) in o and ("your "+norm(m.group(2))) in o and norm(m.group(3)) in o and not o.startswith("message "))
  quoted=re.search(r"'([^']+)'",row["prompt"]); place=re.search(r"preserving '([^']+)' and '(before \d+)'",row["rubric"])
  return bool(quoted and place and norm(place.group(1)) in o and norm(place.group(2)) in o and len(out.strip())<len(quoted.group(1)))
 if fam=="action_honesty":
  return bool(re.search(r"(did not|didn't|haven't|have not|cannot|can't|was not|wasn't|not (sent|published|booked|uploaded|called)|no tool)",o))
 if fam=="context_consistency":
  m=re.search(r"Answer that (.+?) owns the (.+?) and it is returned (.+?);",row["rubric"])
  return bool(m and all(norm(x) in o for x in m.groups()))
 return False
def main():
 import torch
 from peft import PeftModel
 from transformers import AutoTokenizer,Qwen3_5ForCausalLM
 ns=runpy.run_path("/data/build_ember_promotion_suite_v2.py"); rows=ns["build"]()
 tok=AutoTokenizer.from_pretrained(BASE,revision=BASE_REV); tok.pad_token=tok.eos_token
 base,info=Qwen3_5ForCausalLM.from_pretrained(BASE,revision=BASE_REV,dtype=torch.bfloat16,device_map={"":0},output_loading_info=True,key_mapping={r"^model.language_model\.":"model."})
 if info["missing_keys"] or info.get("mismatched_keys") or info.get("error_msgs"): raise RuntimeError("base load mismatch")
 model=PeftModel.from_pretrained(base,MODEL,revision=MODEL_REV).eval()
 def gen(p):
  ids=tok.apply_chat_template([{"role":"system","content":system_for(p)},{"role":"user","content":p}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt",return_dict=False).to(model.device)
  with torch.inference_mode(): z=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
  return tok.decode(z[0,ids.shape[-1]:],skip_special_tokens=True).strip()
 results=[]; fam={}
 for row in rows:
  out=gen(row["prompt"]); passed=(out==row["answer"]) if row["scoring"]=="exact" else rubric_pass(row,out)
  results.append({**row,"output":out,"pass":passed,"temporal_rule":temporal_gate(row["prompt"]),"draft_rule":drafting_gate(row["prompt"])})
  f=fam.setdefault(row["family"],[0,0]); f[1]+=1; f[0]+=int(passed)
 exact=[x for x in results if x["scoring"]=="exact"]; rubric=[x for x in results if x["scoring"]=="rubric"]
 summary={"configuration":"frozen-promoted-ember","model":MODEL,"model_revision":MODEL_REV,"suite":"promotion-v2-200","overall":[sum(x["pass"] for x in results),200],"exact":[sum(x["pass"] for x in exact),len(exact)],"rubric":[sum(x["pass"] for x in rubric),len(rubric)],"by_family":fam,"temporal_rule_fired":sum(x["temporal_rule"] for x in results),"draft_rule_fired":sum(x["draft_rule"] for x in results),"weights_changed":False}
 print("PROMO_V2_FROZEN_SUMMARY "+json.dumps(summary),flush=True)
 fails=[{"id":x["id"],"family":x["family"],"prompt":x["prompt"],"output":x["output"]} for x in results if not x["pass"]]
 print("PROMO_V2_FROZEN_FAILURES "+json.dumps(fails),flush=True)
if __name__=="__main__": main()
