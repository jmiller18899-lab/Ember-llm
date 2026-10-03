# /// script
# dependencies = ["huggingface-hub==1.31.0"]
# ///
"""Read-only job failure inspection for the user's requested next Ember training."""
from huggingface_hub import HfApi
from itertools import islice
from collections import deque
import os, json, re

def main():
    token = [REDACTED]"HF_TOKEN")
    if not token:
        [REDACTED] RuntimeError("Existing EMBER_HF_TOKEN is unavailable")
    api = HfApi(token=[REDACTED]
    if api.whoami().get("name") != "Jmiller18899":
        raise RuntimeError("Unexpected HF account")
    jobs = list(islice(api.list_jobs(), 50))
    summary = [
        dict(id=j.id, url=j.url, created_at=str(j.created_at),
             stage=j.status.stage, message=j.status.message,
             flavor=j.flavor, labels=getattr(j, "labels", {}),
             code_commits={k:v for k,v in j.environment.items()
                           if k.endswith("CODE_COMMIT")})
        for j in jobs
    ]
    print("EMBER_HF_JOBS", json.dumps(summary), flush=True)
    for job in [j for j in jobs if j.status.stage in ("ERROR", "FAILED")][:3]:
        logs = deque((str(line) for line in api.fetch_job_logs(job_id=job.id)), maxlen=90)
        raw = "\n".join(logs).replace(token, "[REDACTED]")
        raw = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", raw)
        print("EMBER_FAILED_JOB_LOG", job.id, raw, flush=True)
    source = api.model_info("Jmiller18899/ember-qwen3.5-4b-repair2",
                           revision="daf938bba5d4e6b650ec9d34a2d3ac56706cf549")
    print("EMBER_PARENT_VERIFIED", source.id, source.sha, flush=True)

if __name__ == "__main__":
    main()

