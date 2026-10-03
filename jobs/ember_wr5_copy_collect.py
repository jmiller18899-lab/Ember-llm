# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["huggingface-hub==1.31.0"]
# ///
"""Stream the explicitly requested diagnostic to completion and retrieve saved JSON; no new compute."""
import argparse, hashlib, inspect, json, os, time
from dataclasses import asdict,is_dataclass
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
OUT="Jmiller18899/ember-wr5-copying-diagnostic-20261003"
FILES=["reservation.json","launch-submission.json","data-audit.json","progress.json","final-report.json"]+["checkpoint-"+str(s)+".json" for s in (0,32,64,96,128)]
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--job-id",required=True);parser.add_argument("--snapshot-only",action="store_true");args=parser.parse_args()
    api=HfApi(token=os.environ["HF_TOKEN"])
    if api.whoami().get("name")!="Jmiller18899":raise ValueError("Wrong account")
    root=Path("copying-saved-evidence");root.mkdir(exist_ok=True)
    kwargs={"job_id":args.job_id,"namespace":"Jmiller18899"}
    if "follow" in inspect.signature(api.fetch_job_logs).parameters:kwargs["follow"]=True
    if not args.snapshot_only:
        with (root/"job-logs.txt").open("w") as handle:
            for line in api.fetch_job_logs(**kwargs):
                text=str(line);handle.write(text+"\n");handle.flush()
                if "COPY_" in text:print(text,flush=True)
    job=api.inspect_job(job_id=args.job_id,namespace="Jmiller18899")
    # The log stream can close before the final job-state update becomes visible.
    for _ in range(4):
        if getattr(job.status,"stage",None) not in ("RUNNING","STARTING","PENDING"):break
        time.sleep(5)
        job=api.inspect_job(job_id=args.job_id,namespace="Jmiller18899")
    info=api.model_info(OUT);revision=info.sha
    status=asdict(job.status) if is_dataclass(job.status) else str(job.status)
    result={"job_id":args.job_id,"url":job.url,"job_status":status,"revision":revision,"private":info.private,"files":[],"missing":[]}
    available=set(api.list_repo_files(OUT,revision=revision))
    for name in FILES:
        if name not in available:result["missing"].append(name);continue
        data=Path(hf_hub_download(OUT,name,revision=revision)).read_bytes();json.loads(data)
        (root/name).write_bytes(data)
        result["files"].append({"path":name,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()})
    for key in ("created_at","started_at","finished_at"):
        value=getattr(job,key,None)
        if value is not None:result[key]=str(value)
    (root/"inspection.json").write_text(json.dumps(result,indent=2,default=str))
    print("COPY_SAVED_EVIDENCE",json.dumps(result,default=str),flush=True)
    if getattr(job.status,"stage",None)!="COMPLETED" or result["missing"]:raise ValueError("Diagnostic did not complete with all saved evidence")
if __name__=="__main__":main()
