"""Use the existing writable HF_TOKEN without exposing it; reserve one paid launch."""
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
from ember_qwen_sft_v1 import REPO


def main():
    if os.environ.get("GITHUB_RUN_ATTEMPT", "1") != "1":
        raise RuntimeError("Workflow retries cannot launch a second paid experiment")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(REPO, private=True, exist_ok=True)
    info = api.repo_info(REPO)
    if not info.private:
        raise RuntimeError("Output repository must be private")
    guard = "launches/sft-v1.json"
    if api.file_exists(REPO, guard):
        raise RuntimeError("Experiment already reserved; refusing duplicate GPU billing")
    source = Path("jobs/ember_qwen_sft_v1.py").read_bytes()
    receipt = {"status":"reserved", "source_commit":os.environ["GITHUB_SHA"],
        "source_sha256":hashlib.sha256(source).hexdigest(),
        "github_run_id":os.environ["GITHUB_RUN_ID"], "hardware":"l4x1",
        "timeout_seconds":7200, "listed_hourly_usd":0.80, "maximum_compute_usd":1.60}
    # Atomic parent-commit guard verifies write permission BEFORE any GPU request.
    api.create_commit(REPO, parent_commit=info.sha, operations=[
        CommitOperationAdd(path_in_repo=guard, path_or_fileobj=json.dumps(receipt).encode()),
        CommitOperationAdd(path_in_repo="source/train.py", path_or_fileobj=source)],
        commit_message="Reserve one capped Qwen adapter experiment after CPU preflight")
    try:
        job = api.run_uv_job("jobs/ember_qwen_sft_v1.py", python="3.11", flavor="l4x1", timeout=7200,
            name="ember-qwen35-2b-sft-v1", secrets={"HF_TOKEN":os.environ["HF_TOKEN"]},
            env={"GITHUB_SHA":os.environ["GITHUB_SHA"],"TOKENIZERS_PARALLELISM":"false"})
    except Exception:
        # Never automatically retry an ambiguous submission; retain the reservation.
        receipt["status"] = "submission_failed_or_unconfirmed"
        Path("gpu-submission.json").write_text(json.dumps(receipt, indent=2))
        raise
    receipt.update(status="submitted", job_id=job.id, job_url=job.url)
    Path("gpu-submission.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
    api.upload_file(repo_id=REPO, path_in_repo=guard,
        path_or_fileobj=json.dumps(receipt, indent=2).encode(), commit_message="Record paid training job receipt")


if __name__ == "__main__":
    main()
