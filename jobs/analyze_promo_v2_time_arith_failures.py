"""Cluster the frozen promoted Ember failures on promotion v2 (time + arithmetic).

Outputs come from HF job 6ab46f596b030d633f68cd4b (PROMO_V2_FROZEN_FAILURES).
CPU only; rebuilds expected answers from the suite. Run: python jobs/analyze_promo_v2_time_arith_failures.py
"""
import importlib.util,json,re
from collections import Counter
from pathlib import Path
OUT={"arith-03":"15","arith-04":"64","arith-05":"62","arith-06":"17","arith-07":"10","arith-08":"63","arith-09":"66","arith-10":"95","arith-12":"81","arith-13":"16","arith-14":"33","arith-17":"37","mult-06":"55",
"time-00":"H:8:30 AM","time-01":"H:7:45 PM","time-02":"H:00 AM","time-03":"H:2:30 PM","time-04":"H:10 AM","time-05":"H:MM PM","time-06":"H:4:30 AM","time-07":"H:4:40 PM","time-08":"H:00 AM","time-09":"H:MM PM","time-10":"H:1:45 AM","time-11":"H:11 PM","time-12":"H:10:45 AM","time-13":"H:11:50 PM","time-14":"H:10:55 AM","time-15":"H:10 PM","time-16":"H:00 AM","time-17":"H:5:55 PM","time-18":"H:45 AM","time-19":"H:9:35 PM","time-20":"H:02 AM","time-21":"H:30 PM","time-22":"H:MM AM","time-23":"H:MM PM","time-24":"H:00 AM","time-26":"H:8:25 AM","time-27":"H:5:40 PM","time-28":"H:00 AM","time-29":"H:3:40 PM"}
def load_rows():
 p=Path(__file__).with_name("build_ember_promotion_suite_v2.py")
 s=importlib.util.spec_from_file_location("suite",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
 return {r["id"]:r for r in m.build()}
def time_cluster(o,exp):
 if not o.startswith("H:"): return "correct" if o==exp else "wrong_unprefixed"
 body=o[2:]
 if body==exp: return "prefix_echo_correct_time"
 if body.startswith("MM"): return "prefix_echo_literal_HMM"
 if body.startswith("00 "): return "prefix_echo_00"
 return "prefix_echo_single_field"
def main():
 rows=load_rows(); report={"time":[],"arithmetic":[]}
 for i in range(30):
  k=f"time-{i:02d}"; r=rows[k]
  h,mm,ap,d=re.search(r"at (\d+):(\d+) (AM|PM)\. Travel time is (\d+)",r["prompt"]).groups(); h,mm,d=int(h),int(mm),int(d)
  o=OUT.get(k,r["answer"]); c=time_cluster(o,r["answer"])
  report["time"].append({"id":k,"expected":r["answer"],"output":o,"cluster":c,"hour_carry":mm+d>=60,"crosses_noon_or_midnight":h<12 and (h*60+mm+d)>=720,"meridiem_kept":o.endswith(ap)})
 for i in range(20):
  k=f"arith-{i:02d}"; r=rows[k]; a,b,c=map(int,re.findall(r"\d+",r["prompt"])[:3])
  if k not in OUT: continue
  o=int(OUT[k]); ops={"a-b+c":a-b+c,"a-b":a-b,"a-c":a-c,"a+b":a+b,"a":a}
  hit=[n for n,v in ops.items() if v==o]; err=o-int(r["answer"])
  cl="operation_confusion" if hit else ("tens_digit_slip" if err%10==0 else "other")
  report["arithmetic"].append({"id":k,"a_b_c":[a,b,c],"expected":int(r["answer"]),"output":o,"error":err,"cluster":cl,"matches":hit})
 t=report["time"]; carry=[x for x in t if x["hour_carry"]]; same=[x for x in t if not x["hour_carry"]]
 ok=lambda x:x["cluster"] in ("correct","prefix_echo_correct_time")
 summary={"time_clusters":Counter(x["cluster"] for x in t),"time_content_correct_same_hour":[sum(map(ok,same)),len(same)],
  "time_content_correct_hour_carry":[sum(map(ok,carry)),len(carry)],"time_meridiem_kept":[sum(x["meridiem_kept"] for x in t),30],
  "time_cases_crossing_noon_or_midnight":sum(x["crosses_noon_or_midnight"] for x in t),"arith_clusters":Counter(x["cluster"] for x in report["arithmetic"])}
 print(json.dumps(summary,indent=1)); print(json.dumps(report,indent=1))
if __name__=="__main__": main()
