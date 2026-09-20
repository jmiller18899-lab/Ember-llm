"""Reserve and submit one persisted Qwen3.5-4B Ember benchmark."""
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = "Jmiller18899/ember-qwen3.5-2b-sft-v3"
GUARD = "benchmarks/qwen35-4b-base/launch.json"
RESULT = "benchmarks/qwen35-4b-base/summary.json"


def main():
    if os.environ.get("GITHUB_RUN_ATTEMPT", "1") != "1":
        raise RuntimeError("Workflow retries cannot launch a second paid benchmark")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Write-enabled HF token is unavailable")
    api = HfApi(token=token)
    if api.whoami()["name"].lower() != "jmiller18899":
        raise RuntimeError("Unexpected Hugging Face account")
    info = api.repo_info(REPO)
    if not info.private:
        raise RuntimeError("Evidence repository must remain private")
    if api.file_exists(REPO, RESULT) or api.file_exists(REPO, GUARD):
        raise RuntimeError("Benchmark already reserved or completed; refusing duplicate GPU billing")

    source_path = Path("jobs/ember_qwen35_4b_benchmark.py")
    source = source_path.read_bytes()
    receipt = {
        "status": "reserved",
        "source_commit": os.environ["GITHUB_SHA"],
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "github_run_id": os.environ["GITHUB_RUN_ID"],
        "hardware": "l4x1",
        "timeout_seconds": 3600,
        "listed_hourly_usd": 0.80,
        "maximum_compute_usd": 0.80,
        "training_started": False,
    }
    api.create_commit(
        REPO,
        parent_commit=info.sha,
        operations=[
            CommitOperationAdd(path_in_repo=GUARD, path_or_fileobj=json.dumps(receipt, indent=2).encode()),
            CommitOperationAdd(path_in_repo="benchmarks/qwen35-4b-base/source.py", path_or_fileobj=source),
        ],
        commit_message="Reserve one persisted Qwen3.5-4B baseline benchmark",
    )
    job = api.run_uv_job(
        str(source_path),
        python="3.11",
        flavor="l4x1",
        timeout=3600,
        name="ember-qwen35-4b-review",
        secrets={"HF_TOKEN": token},
        env={"GITHUB_SHA": os.environ["GITHUB_SHA"], "TOKENIZERS_PARALLELISM": "false"},
    )
    receipt.update(status="submitted", job_id=job.id, job_url=job.url)
    Path("gpu-submission-4b.json").write_text(json.dumps(receipt, indent=2))
    api.upload_file(
        repo_id=REPO,
        path_in_repo=GUARD,
        path_or_fileobj=json.dumps(receipt, indent=2).encode(),
        commit_message="Record persisted Qwen3.5-4B benchmark receipt",
    )
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
