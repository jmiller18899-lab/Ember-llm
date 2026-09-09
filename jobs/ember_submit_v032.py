"""Submit at most one bounded T4 run after the matching CPU learning PASS."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jobs.ember_semantic_train_v032 import DEFAULT_CONFIG, file_hashes, validate_canary, write_json


def job_script(commit: str, publication: dict, files: dict) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("the job must run an immutable GitHub commit")
    metadata = {"commit": commit, "canary_revision": publication["revision"], "canary_report": publication["report_path"], "files": files}
    return '''# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.6.0", "huggingface-hub==1.30.0", "sentencepiece==0.2.0", "jsonschema==4.23.0", "trackio==0.37.1", "numpy==2.2.6"]
# ///
import hashlib, json, os, pathlib, subprocess, sys, tempfile, urllib.request
META = ''' + repr(metadata) + '''
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["GITHUB_SHA"] = META["commit"]
with tempfile.TemporaryDirectory(prefix="ember-v032-code-") as temp:
    root = pathlib.Path(temp)
    base = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/" + META["commit"] + "/"
    for name, expected in META["files"].items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(base + name, path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("job source hash mismatch: " + name)
    subprocess.run([sys.executable, str(root / "jobs/ember_semantic_train_v032.py"),
                    "--mode", "train", "--output-dir", str(root / "results"),
                    "--canary-revision", META["canary_revision"], "--canary-report", META["canary_report"]], check=True)
'''


def main():
    from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("GITHUB_RUN_ATTEMPT", "1") != "1":
        raise ValueError("workflow retries do not automatically submit another paid job")
    config = json.loads(DEFAULT_CONFIG.read_text())
    identity = {"files": file_hashes(config), "dataset": config["dataset"], "version": config["version"]}
    publication = json.loads(args.publication.read_text())
    if publication["repo_id"] != config["canary"]["output_repo"] or not re.fullmatch(r"[a-f0-9]{40}", publication["revision"]):
        raise ValueError("invalid canary publication identity")
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    report_path = hf_hub_download(publication["repo_id"], revision=publication["revision"], filename=publication["report_path"], token=token)
    validate_canary(json.loads(Path(report_path).read_text()), identity, config["canary"]["gate"])
    files = {**identity["files"], "ember-v0.0.7-hf-ready.zip": "27e8f7c80317652a22b3d58a0bd474724491a685dfe9e20c0b997b7c5907a289"}
    script = job_script(os.environ["GITHUB_SHA"], publication, files)
    guard_path = "launches/v0.0.32-t4.json"
    info = api.repo_info(publication["repo_id"])
    if api.file_exists(publication["repo_id"], guard_path):
        raise ValueError("v0.0.32 already has a launch reservation; refusing a duplicate paid job")
    reservation = {"status": "reserved", "canary_revision": publication["revision"], "code_commit": os.environ["GITHUB_SHA"],
                   "created_at": datetime.now(timezone.utc).isoformat(), "github_run_id": os.environ.get("GITHUB_RUN_ID")}
    api.create_commit(publication["repo_id"], parent_commit=info.sha,
                      operations=[CommitOperationAdd(path_in_repo=guard_path, path_or_fileobj=(json.dumps(reservation) + "\n").encode())],
                      commit_message="Reserve the single approved v0.0.32 T4 experiment")
    with tempfile.TemporaryDirectory(prefix="ember-v032-submit-") as td:
        wrapper = Path(td) / "ember-v032-job.py"
        wrapper.write_text(script)
        job = api.run_uv_job(str(wrapper), python="3.11", flavor=config["train"]["hardware"],
                             timeout=config["train"]["job_timeout_seconds"],
                             name="ember-v0-0-32-semantic-repair", labels={"ember_version": "0.0.32", "phase": "semantic-repair"},
                             secrets={"HF_TOKEN": token}, env={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    receipt = {**reservation, "status": "submitted", "job_id": job.id, "job_url": job.url,
               "flavor": config["train"]["hardware"], "timeout_seconds": config["train"]["job_timeout_seconds"]}
    write_json(args.publication.parent / "gpu-submission.json", receipt)
    api.upload_file(repo_id=publication["repo_id"], path_in_repo=guard_path,
                    path_or_fileobj=(json.dumps(receipt, indent=2) + "\n").encode(), commit_message="Record the v0.0.32 T4 job receipt")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
