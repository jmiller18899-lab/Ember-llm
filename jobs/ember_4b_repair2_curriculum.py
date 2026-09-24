"""Repair2 curriculum for an experimental successor to frozen promoted Ember.

Data only; no model code. Priorities come from promotion v2 and the time
held-out v1 frozen baseline (reports/time_heldout_v1_frozen_baseline.md):

  time 50%            long durations 30 | midnight 25 | :60 rollover 20 |
                      placeholder format 20 | maintenance 5  (% of time)
  arithmetic 25%      add-then-subtract with two-digit addends, multiply-then-add
  context 12.5%       restate owner, item AND day from a context record
  clarification 12.5% ask for the missing text instead of the missing "request"

Every time (start, duration) pair used by time-heldout-v1 or promotion v2 is
excluded, arithmetic operand tuples from promotion v2 are excluded, and no
prompt uses an evaluation template. Write the frozen corpus with:
  python jobs/ember_4b_repair2_curriculum.py --write
"""
import datetime as dt
import hashlib,json,random,re,sys
from pathlib import Path

SEED=20260925
CORPUS=Path(__file__).resolve().parents[1]/"data"/"ember_4b_repair2_curriculum.jsonl"
COUNTS={  # family/slice -> rows (400 total)
 "time/long_duration":60,"time/midnight":50,"time/rollover_60":40,"time/placeholder":40,"time/maintenance":10,
 "arithmetic/add_subtract":80,"arithmetic/multiply_add":20,
 "context_consistency":50,"clarification":50,
}

def fmt(mins):
 mins%=1440; h,m=divmod(mins,60)
 return f"{(h-1)%12+1}:{m:02d} {'AM' if h<12 else 'PM'}"
def check_time(start,d,ans):
 e=(dt.datetime(2026,1,1)+dt.timedelta(minutes=start+d)).strftime("%I:%M %p").lstrip("0")
 assert e==ans,(start,d,ans,e)

# ---- time ----
TIME_TEMPLATES=[  # deliberately different wording from both evaluation suites
 "The ferry sails at {t} and the crossing takes {d}. What time does it dock?",
 "A movie begins at {t} and runs {d}. What time does it end?",
 "I put bread in the oven at {t}. It bakes for {d}. When is it done?",
 "A flight takes off at {t}. The flight time is {d}. When does it land?",
 "A class starts at {t} and lasts {d}. What time is it over?",
 "Parking starts at {t} and I paid for {d}. When does it expire?",
 "A hike begins at {t} and takes {d}. What time will we finish?",
 "The delivery van leaves the depot at {t}; the drive is {d}. What time does it get there?",
]
NATURAL_EXAMPLE="3:40 PM"  # never used as an answer, so copying it is never rewarded
NATURAL_SUFFIXES=["Answer with the time only.","Just give the time.",f"Reply with the time, like {NATURAL_EXAMPLE}.","What's the time? Give only the time."]
def placeholder_suffix(r,start_meridiem,same_meridiem):
 opts=["Answer H:MM AM/PM.","Use the format H:MM AM or H:MM PM.","Format: h:mm AM/PM."]
 if same_meridiem: opts+= [f"Answer H:MM {start_meridiem}.",f"Reply as H:MM {start_meridiem}."]
 return r.choice(opts)
def duration_text(r,d,allow_hours):
 if not allow_hours or d<60 or r.random()<0.7: return f"{d} minutes"
 h,m=divmod(d,60); hs=f"{h} hour{'s' if h>1 else ''}"
 return hs if m==0 else f"{hs} {m} minutes"

def time_rows(r,excluded):
 rows=[]; used=set(excluded)
 def pick(cond,drange):
  for _ in range(100000):
   start=r.randrange(0,1440,5); d=r.choice(drange)
   if (start,d) in used or not cond(start,d) or fmt(start+d)==NATURAL_EXAMPLE: continue
   used.add((start,d)); return start,d
  raise RuntimeError("no pair")
 def add(slice_,start,d,style,allow_hours=True):
  ans=fmt(start+d); check_time(start,d,ans)
  q=r.choice(TIME_TEMPLATES).format(t=fmt(start),d=duration_text(r,d,allow_hours))
  same=(start+d)//720==start//720
  q+=" "+(placeholder_suffix(r,fmt(start)[-2:],same) if style=="placeholder" else r.choice(NATURAL_SUFFIXES))
  rows.append({"family":"time_reasoning","slice":slice_,"prompt":q,"answer":ans,"start":fmt(start),"duration_min":d})
 mid=lambda s,d:s<1440<=s+d
 for i in range(COUNTS["time/long_duration"]):  # 65-240 min, noon crossings allowed, never midnight
  s,d=pick(lambda s,d:not mid(s,d) and (s+d)%60!=0,range(65,241,5)); add("long_duration",s,d,"natural")
 for i in range(COUNTS["time/midnight"]):  # PM -> AM, including landing on 12:xx AM and exactly 12:00 AM
  s,d=pick(lambda s,d:s>=1080 and mid(s,d) and s+d<1440+180,range(5,241,5)); add("midnight",s,d,"natural")
 for i in range(COUNTS["time/rollover_60"]):  # lands exactly on the hour (the X:60 failure)
  s,d=pick(lambda s,d:not mid(s,d) and (s+d)%60==0 and s%60!=0,range(5,181,5)); add("rollover_60",s,d,"natural")
 for i in range(COUNTS["time/placeholder"]):  # H:MM wording; answer fills in the real time
  s,d=pick(lambda s,d:not mid(s,d),range(5,121,5)); add("placeholder",s,d,"placeholder",allow_hours=False)
 for i in range(COUNTS["time/maintenance"]):  # already-solid skills: plain carry, noon, 12->1
  kind=i%3
  cond=[lambda s,d:(s%60)+d>=60 and (s+d)%60!=0 and (s+d)//720==s//720 and (s//60)%12!=11,
        lambda s,d:s<720<=s+d and (s+d)%60!=0,
        lambda s,d:(s//60)%12==0 and ((s+d)//60)%12==1 and (s+d)//720==s//720][kind]
  s,d=pick(cond,range(5,121,5)); add("maintenance",s,d,"natural",allow_hours=False)
 return rows

# ---- arithmetic ----
ADD_SUB=[
 "A shelf holds {a} books. {b} more books are added, then {c} are taken away. How many books are on the shelf? Number only.",
 "Start with {a} tickets. Add {b}, then subtract {c}. How many tickets now? Number only.",
 "A tank has {a} liters. {b} liters are poured in and {c} liters are drained. How many liters remain? Number only.",
 "A farmer has {a} sheep, buys {b} more, and sells {c}. How many sheep does the farmer have? Number only.",
 "There are {a} chairs in a hall. {b} are brought in and {c} are carried out. How many chairs are in the hall? Number only.",
]
MUL_ADD=[
 "{n} crates hold {k} apples each, and {x} apples are loose. How many apples in total? Number only.",
 "A printer makes {n} stacks of {k} pages, plus {x} extra pages. How many pages? Number only.",
 "{n} teams have {k} players each, plus {x} substitutes. How many people? Number only.",
]
def arithmetic_rows(r,excluded):
 rows=[]; used=set()
 while len(rows)<COUNTS["arithmetic/add_subtract"]:
  i=len(rows)
  a=r.randint(20,95); b=r.randint(11,39) if i%10<8 else r.randint(3,10); c=r.randint(2,25)
  if (a,b,c) in excluded or (a,b,c) in used or a+b-c<0: continue
  used.add((a,b,c)); rows.append({"family":"arithmetic","slice":"add_subtract","prompt":ADD_SUB[i%len(ADD_SUB)].format(a=a,b=b,c=c),"answer":str(a+b-c),"operands":[a,b,c]})
 while len(rows)<COUNTS["arithmetic/add_subtract"]+COUNTS["arithmetic/multiply_add"]:
  i=len(rows); n,k,x=r.randint(3,12),r.randint(3,15),r.randint(1,9)
  if ("mul",n,k,x) in excluded or ("mul",n,k,x) in used: continue
  used.add(("mul",n,k,x)); rows.append({"family":"arithmetic","slice":"multiply_add","prompt":MUL_ADD[i%len(MUL_ADD)].format(n=n,k=k,x=x),"answer":str(n*k+x),"operands":[n,k,x]})
 return rows

# ---- context consistency ----
OWNERS=["Priya","Marco","Hana","Luis","Zoe","Emeka","Sofia","Tariq","Nell","Oskar","Rhea","Dev"]
ITEMS=["laptop","scarf","key card","water bottle","jacket","tablet","backpack","camera","headphones","lunchbox"]
PLURAL_ITEMS={"headphones"}
DAYS=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
CTX_FORMS=[
 ("Record: owner={o}; item={i}; return_day={d}. Who owns the item and when is it returned?","{o} owns the {i}, and it is returned {d}."),
 ("Lost-and-found note: the {i} belongs to {o} and goes back on {d}. Whose item is it and when is it returned?","{o} owns the {i}, and it is returned {d}."),
 ("Log entry {{owner: {o}, item: {i}, return: {d}}}. Now answer: who owns the item and when does it go back?","{o} owns the {i}, and it goes back {d}."),
 ("Facts: {o} lent us a {i}. We return it on {d}. Who is the owner, and what day is it returned?","{o} is the owner of the {i}, and it is returned {d}."),
]
def context_rows(r):
 rows=[]; seen=set()
 while len(rows)<COUNTS["context_consistency"]:
  o,i,d=r.choice(OWNERS),r.choice(ITEMS),r.choice(DAYS); f=len(rows)%len(CTX_FORMS)
  if (o,i,d,f) in seen: continue
  seen.add((o,i,d,f)); q,a=CTX_FORMS[f]
  if i in PLURAL_ITEMS: a=a.replace("it is","they are").replace("it goes","they go")
  rows.append({"family":"context_consistency","slice":"restate_all_facts","prompt":q.format(o=o,i=i,d=d),"answer":a.format(o=o,i=i,d=d)})
 return rows

# ---- clarification ----
CLARIFY_VERBS={"summarize":"summarized","simplify":"simplified","shorten":"shortened","rewrite":"rewritten","proofread":"proofread","translate":"translated","reformat":"reformatted","polish":"polished"}
CLARIFY_FORMS=["Please {v} it.","{V} this one for me, thanks.","Could you {v} the above?","Need you to {v} that. Ticket {n}.","{V} item {n} please.","Quick favor: {v} the last one.","Can you {v} what I mentioned? Case {n}."]
def clarification_rows(r):
 rows=[]; seen=set()
 while len(rows)<COUNTS["clarification"]:
  v=r.choice(list(CLARIFY_VERBS)); f=r.choice(CLARIFY_FORMS); n=r.randint(2,99)
  q=f.format(v=v,V=v.capitalize(),n=n)
  if q in seen: continue
  seen.add(q); rows.append({"family":"clarification","slice":"ask_for_text","prompt":q,"answer":f"Please paste the text you want {CLARIFY_VERBS[v]}."})
 return rows

# ---- exclusions from the evaluation suites ----
def load_job(name):
 import importlib.util
 s=importlib.util.spec_from_file_location(name,Path(__file__).with_name(f"{name}.py")); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def eval_exclusions():
 h=load_job("ember_time_heldout_v1"); promo=load_job("build_ember_promotion_suite_v2").build()
 time_pairs={(h.parse12(x["start"]),x["duration"]) for x in h.build()}|h.promo_v2_time_keys()
 arith=set()
 for x in promo:
  if x["id"].startswith("arith"): arith.add(tuple(map(int,re.findall(r"\d+",x["prompt"])[:3])))
  if x["id"].startswith("mult"): arith.add(("mul",*map(int,re.findall(r"\d+",x["prompt"])[:3])))
 eval_prompts={x["prompt"] for x in h.build()}|{x["prompt"] for x in promo}
 return time_pairs,arith,eval_prompts

def build():
 time_pairs,arith,_=eval_exclusions(); r=random.Random(SEED)
 rows=time_rows(r,time_pairs)+arithmetic_rows(r,arith)+context_rows(r)+clarification_rows(r)
 for n,x in enumerate(rows): x["id"]=f"repair2-{n:03d}"
 return rows
def serialize(rows): return "".join(json.dumps(x,sort_keys=True)+"\n" for x in rows)
def main():
 rows=build(); text=serialize(rows)
 if "--write" in sys.argv:
  CORPUS.parent.mkdir(exist_ok=True); CORPUS.write_text(text)
 fam={}
 for x in rows: k=x["family"]+("/"+x["slice"] if x["family"] in ("time_reasoning","arithmetic") else ""); fam[k]=fam.get(k,0)+1
 print("REPAIR2_CURRICULUM "+json.dumps({"rows":len(rows),"sha256":hashlib.sha256(text.encode()).hexdigest(),"counts":fam}),flush=True)
if __name__=="__main__": main()
