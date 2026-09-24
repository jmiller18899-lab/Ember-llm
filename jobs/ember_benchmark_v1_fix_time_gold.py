# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["huggingface-hub==1.31.0"]
# ///
"""Correct the time-14 gold answer in Jmiller18899/ember-generalization-benchmark-v1.

The generator (jobs/ember_4b_generalization_benchmark_v1.py on codex/ember-4b-acceleration-sprint1) copied the
start time's AM/PM onto the answer, even when the trip crosses noon. Only time-14 crosses:
"10:50 AM + 75 minutes" was scored as "12:05 AM"; the true arrival is 12:05 PM.

This job recomputes every time_reasoning gold with 24-hour arithmetic and aborts unless exactly the expected
change is found. It then rewrites the answer (prompts unchanged) in cases.json, baseline.json and candidate.json,
recomputes exact_match from each file's saved outputs, refreshes summary.json scores, and adds CORRECTIONS.md.
The previous benchmark revision stays in the repo history for reproducing earlier results.
Idempotent: if the gold is already corrected, it changes nothing.
Usage:  python jobs/ember_benchmark_v1_fix_time_gold.py --self-test   (no network)
        HF_TOKEN=... python jobs/ember_benchmark_v1_fix_time_gold.py [--dry-run]
"""
import argparse,json,os,re,tempfile
from pathlib import Path

REPO="Jmiller18899/ember-generalization-benchmark-v1"
FROM_REV="9b080364c0b4d4d005cc376a4f45daa1a2edcd77"   # revision every promotion/Repair2/Repair3 run scored against
EXPECTED={"time-14":("12:05 AM","12:05 PM")}
ROW_FILES=("cases.json","baseline.json","candidate.json")
PROMPT_RE=re.compile(r"leaves at (\d{1,2}):(\d{2}) (AM|PM) and the trip takes (\d+) minutes")

def true_arrival(prompt):
    h,m,ap,d=PROMPT_RE.search(prompt).groups()
    t=((int(h)%12+(12 if ap=="PM" else 0))*60+int(m)+int(d))%1440
    return f"{(t//60-1)%12+1}:{t%60:02d} {'AM' if t<720 else 'PM'}"

def corrections(rows):
    out={}
    for r in rows:
        if r.get("family")=="time_reasoning" and r.get("scoring")=="exact":
            t=true_arrival(r["prompt"])
            if t!=r["answer"]: out[r["id"]]=(r["answer"],t)
    return out

def apply(rows,fix):
    for r in rows:
        if r["id"] in fix:
            r["answer"]=fix[r["id"]]
            if "output" in r and r.get("scoring")=="exact": r["exact_match"]=r["output"]==r["answer"]
    return rows

def score(rows):
    exact=[r for r in rows if r["scoring"]=="exact"]; fam={}
    for r in exact:
        f=fam.setdefault(r["family"],[0,0]); f[1]+=1; f[0]+=int(r["exact_match"])
    return {"exact_pass":sum(r["exact_match"] for r in exact),"exact_total":len(exact),"by_family":fam}

def correct(files):
    """files: {name: parsed json}. Returns (new files, applied changes); raises on anything unexpected."""
    found=corrections(files["cases.json"])
    if not found: return files,{}   # already corrected
    if found!=EXPECTED: raise RuntimeError(f"unexpected gold differences: {found} (expected {EXPECTED})")
    fix={i:new for i,(_,new) in found.items()}
    prompts_before={n:[r["prompt"] for r in files[n]] for n in ROW_FILES}
    for n in ROW_FILES:
        rows=files[n]
        for i,(old,_) in found.items():
            hit=[r for r in rows if r["id"]==i]
            if len(hit)!=1 or hit[0]["answer"]!=old: raise RuntimeError(f"{n}: {i} not in expected state")
        apply(rows,fix)
        assert [r["prompt"] for r in rows]==prompts_before[n]
        assert corrections(rows)=={}
    s=files["summary.json"]
    s["baseline"]=score(files["baseline.json"]); s["candidate"]=score(files["candidate.json"])
    s["exact_delta"]=s["candidate"]["exact_pass"]-s["baseline"]["exact_pass"]
    s["gold_corrections"]=[{"id":i,"old":o,"new":n_,"reason":"answer kept the start's AM/PM across noon"} for i,(o,n_) in found.items()]
    s["corrected_from_revision"]=FROM_REV
    return files,found

def self_test():
    p="A shuttle leaves at 10:50 AM and the trip takes 75 minutes. What time does it arrive? Answer as H:MM AM."
    assert true_arrival(p)=="12:05 PM"
    assert true_arrival("A shuttle leaves at 11:30 PM and the trip takes 45 minutes.")=="12:15 AM"
    assert true_arrival("A shuttle leaves at 5:10 AM and the trip takes 50 minutes.")=="6:00 AM"
    ok={"id":"time-00","family":"time_reasoning","scoring":"exact","prompt":"A shuttle leaves at 2:00 AM and the trip takes 35 minutes.","answer":"2:35 AM"}
    bad={"id":"time-14","family":"time_reasoning","scoring":"exact","prompt":p,"answer":"12:05 AM"}
    ar={"id":"arith-pack-13","family":"arithmetic","scoring":"exact","prompt":"x","answer":"40"}
    def run(outs):
        return [{**r,"output":o,"exact_match":o==r["answer"]} for r,o in zip([ok,bad,ar],outs)]
    files={"cases.json":[dict(ok),dict(bad),dict(ar)],"baseline.json":run(["2:35 AM","12:05 PM","41"]),
           "candidate.json":run(["2:35 AM","12:05 PM","40"]),"summary.json":{}}
    new,found=correct(json.loads(json.dumps(files)))
    assert found=={"time-14":("12:05 AM","12:05 PM")}
    assert new["summary.json"]["candidate"]["exact_pass"]==3 and new["summary.json"]["baseline"]["exact_pass"]==2
    assert all(r["answer"]=="12:05 PM" for n in ROW_FILES for r in new[n] if r["id"]=="time-14")
    again,found2=correct(json.loads(json.dumps(new))); assert found2=={} and again==new   # idempotent
    wrong=json.loads(json.dumps(files)); wrong["cases.json"][0]["answer"]="2:36 AM"
    try: correct(wrong); raise AssertionError("should refuse unexpected differences")
    except RuntimeError: pass
    print("TIME_GOLD_FIX_SELF_TEST_PASS",flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); ap.add_argument("--dry-run",action="store_true"); a=ap.parse_args()
    if a.self_test: return self_test()
    from huggingface_hub import HfApi,hf_hub_download
    token=os.environ["HF_TOKEN"]; api=HfApi(token=token)
    head=api.model_info(REPO).sha
    names=api.list_repo_files(REPO,revision=head)
    need=[*ROW_FILES,"summary.json"]
    if missing:=[n for n in need if n not in names]: raise RuntimeError(f"missing files: {missing}")
    files={n:json.loads(Path(hf_hub_download(REPO,n,revision=head,token=token)).read_text()) for n in need}
    new,found=correct(files)
    if not found:
        print("TIME_GOLD_FIX_RESULT "+json.dumps({"status":"already_corrected","revision":head}),flush=True); return
    if head!=FROM_REV: raise RuntimeError(f"benchmark head {head} != expected {FROM_REV}; review before correcting")
    info={"status":"dry_run" if a.dry_run else "corrected","from_revision":head,"changes":found,
          "baseline":new["summary.json"]["baseline"],"candidate":new["summary.json"]["candidate"]}
    if a.dry_run:
        print("TIME_GOLD_FIX_RESULT "+json.dumps(info),flush=True); return
    note=("# Gold corrections\n\n"
          f"## time-14 (corrected from revision `{FROM_REV}`)\n"
          "- Prompt: \"A shuttle leaves at 10:50 AM and the trip takes 75 minutes. What time does it arrive? Answer as H:MM AM.\" (unchanged)\n"
          "- Old gold: `12:05 AM`. New gold: `12:05 PM`.\n"
          "- Cause: the generator kept the start's AM/PM on the answer even when the trip crosses noon. No other time case crosses noon.\n"
          "- `exact_match` in baseline.json/candidate.json was recomputed from the saved outputs; summary.json scores were refreshed.\n"
          f"- Earlier results scored against `{FROM_REV}` stay reproducible at that revision.\n")
    with tempfile.TemporaryDirectory() as d:
        for n,v in new.items(): Path(d,n).write_text(json.dumps(v,indent=2))
        Path(d,"CORRECTIONS.md").write_text(note)
        c=api.upload_folder(repo_id=REPO,folder_path=d,path_in_repo="",parent_commit=head,
            commit_message="Correct time-14 gold: 10:50 AM + 75 min is 12:05 PM, not 12:05 AM")
    info["new_revision"]=api.model_info(REPO).sha; info["commit"]=getattr(c,"oid",None)
    print("TIME_GOLD_FIX_RESULT "+json.dumps(info),flush=True)
if __name__=="__main__": main()
