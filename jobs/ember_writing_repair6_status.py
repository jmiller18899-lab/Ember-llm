# /// script
# dependencies = ["huggingface-hub==1.31.0"]
# ///
"""Verify private WR6 storage with the existing training credential. No GPU job."""
from huggingface_hub import HfApi
from itertools import islice
from collections import deque
import os, json, re

OUTPUT = "Jmiller18899/ember-qwen3.5-4b-writing-repair6-20261007"
SOURCE = "Jmiller18899/ember-qwen3.5-4b-repair2"
SOURCE_REV = "daf938bba5d4e6b650ec9d34a2d3ac56706cf549"


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Existing EMBER_HF_TOKEN is unavailable")
    api = HfApi(token=token)
    if api.whoami().get("name") != "Jmiller18899":
        raise RuntimeError("Unexpected HF account")
    jobs = list(islice(api.list_jobs(), 50))
    summary = [
        dict(id=j.id, url=j.url, created_at=str(j.created_at),
             stage=j.status.stage, message=j.status.message,
             flavor=j.flavor, labels=getattr(j, "labels", {}),
             code_commits={k: v for k, v in j.environment.items()
                           if k.endswith("CODE_COMMIT")})
        for j in jobs
    ]
    print("EMBER_HF_JOBS", json.dumps(summary), flush=True)
    for job in [j for j in jobs if j.status.stage in ("ERROR", "FAILED")][:3]:
        logs = deque((str(line) for line in api.fetch_job_logs(job_id=job.id)), maxlen=90)
        raw = "\n".join(logs).replace(token, "[REDACTED]")
        raw = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", raw)
        print("EMBER_FAILED_JOB_LOG", job.id, raw, flush=True)
    source = api.model_info(SOURCE, revision=SOURCE_REV)
    print("EMBER_PARENT_VERIFIED", source.id, source.sha, flush=True)
    auth = api.whoami().get("auth", {})
    print("EMBER_TRAINING_CREDENTIAL", json.dumps({"type": auth.get("type"),
          "role": auth.get("accessToken", {}).get("role")}), flush=True)
    try:
        api.create_repo(OUTPUT, repo_type="model", private=True, exist_ok=False)
        receipt = dict(status="PASS", operation="private_candidate_storage", output_repo=OUTPUT,
                       source_model=source.id, source_revision=source.sha, training_started=False)
        api.upload_file(repo_id=OUTPUT, path_in_repo="storage-preflight.json",
                        path_or_fileobj=json.dumps(receipt, indent=2).encode(),
                        commit_message="Verify private storage before requested WR6 training")
        print("EMBER_STORAGE_PREFLIGHT", json.dumps(receipt), flush=True)
    except Exception as error:
        response = getattr(error, "response", None)
        detail = str(error).replace(token, "[REDACTED]")
        detail = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", detail)
        print("EMBER_STORAGE_PREFLIGHT", json.dumps({"status": "BLOCKED", "error_type": type(error).__name__,
              "http_status": getattr(response, "status_code", None), "detail": detail, "training_started": False}),
              flush=True)
        raise


if __name__ == "__main__":
    main()
