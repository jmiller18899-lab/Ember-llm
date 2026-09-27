# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","accelerate==1.15.0","huggingface-hub==1.31.0","jinja2==3.1.6"]
# ///
"""Validate and submit Ember WR2 using a GitHub Actions repository secret.

This script runs only on the pinned WR2 experiment branch. It does not train
locally, inspect final holdouts, log the secret, or promote a candidate.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from huggingface_hub import CommitOperationAdd, HfApi
from transformers import AutoTokenizer

import ember_writing_repair2_train as T
import ember_writing_repair2_data as W
import ember_writing_repair1_data as D

BRANCH = "experiment/ember-writing-repair2-train-20260927"

def main():
    token = os.environ.get("HF_TOKEN")
    revision = os.environ.get("WR2_CODE_COMMIT", "")
    if not token or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Require an Actions HF_TOKEN secret and pinned source commit")
    if os.environ.get("GITHUB_REF") != f"refs/heads/{BRANCH}":
        raise RuntimeError("Refuse to launch outside the WR2 experiment branch")
    if not (Path(__file__).parent / "ember_writing_repair2_train.py").is_file():
        raise RuntimeError("Trainer missing from pinned checkout")
    if hashlib.sha256(Path(W.__file__).read_bytes()).hexdigest() != T.WR2_DATA_SHA:
        raise RuntimeError("WR2 generator checksum mismatch")
    rows = W.build_train_rows(D)
    sft = W.sft_bytes(rows)
    if hashlib.sha256(sft).hexdigest() != T.SFT_SHA:
        raise RuntimeError("Training export checksum mismatch")
    train_rows = T.validate_rows([json.loads(line) for line in sft.splitlines()])
    work = Path(tempfile.mkdtemp(prefix="ember-wr2-launch-"))
    _, dev, G, E, M, suites = T.load_inputs(work)
    tok = AutoTokenizer.from_pretrained(T.BASE, revision=T.BASE_REV)
    tok.pad_token = tok.eos_token
    route = G.build_route(E, M, "v3_baseline")
    encoded = T.encoded_rows(train_rows, tok, route, M.O.SYSTEM)
    lengths = [len(x["input_ids"]) for x in encoded]
    if len(train_rows) != 512 or len(dev) != 64 or max(lengths) > T.MAX_LEN:
        raise RuntimeError("WR2 training coverage or length check failed")
    if {suite: len(items) for suite, items in suites.items()} != T.COUNTS:
        raise RuntimeError("Frozen benchmark coverage changed")
    api = HfApi(token=token)
    if api.whoami().get("name") != "Jmiller18899":
        raise RuntimeError("Wrong Hugging Face account")
    if api.repo_exists(T.OUTPUT_REPO):
        raise RuntimeError("WR2 output repository already exists; refusing duplicate launch")
    api.create_repo(T.OUTPUT_REPO, repo_type="model", private=True, exist_ok=False)
    trainer_bytes = Path(T.__file__).read_bytes()
    launch = {**T.run_spec(), "data_sha256": T.SFT_SHA,
              "trainer_sha256": hashlib.sha256(trainer_bytes).hexdigest(),
              "max_training_tokens": max(lengths),
              "families": dict(Counter(r["family"] for r in rows)),
              "baseline_manifest_sha256": T.SUITE_SHA,
              "code_commit": revision, "reserved_for_training": True}
    api.create_commit(
        repo_id=T.OUTPUT_REPO,
        operations=[
            CommitOperationAdd(path_in_repo="launch.json",
                               path_or_fileobj=json.dumps(launch, indent=2).encode()),
            CommitOperationAdd(path_in_repo="train/sft_train_only.jsonl",
                               path_or_fileobj=sft),
            CommitOperationAdd(path_in_repo="source/ember_writing_repair2_data.py",
                               path_or_fileobj=Path(W.__file__).read_bytes()),
            CommitOperationAdd(path_in_repo="source/ember_writing_repair2_train.py",
                               path_or_fileobj=trainer_bytes),
        ],
        commit_message="Reserve audited WR2 training artifacts",
    )
    print("WR2_PREFLIGHT_PASS", json.dumps({"repo": T.OUTPUT_REPO,
          "rows": len(train_rows), "dev": len(dev), "max_tokens": max(lengths),
          "code_commit": revision}), flush=True)
    job = api.run_uv_job(
        script=str(Path(T.__file__).resolve()),
        python="3.11",
        flavor="l4x1",
        timeout="90m",
        env={"WR2_CODE_COMMIT": revision},
        secrets={"HF_TOKEN": token},
    )
    print("WR2_GPU_JOB_SUBMITTED", json.dumps({"job_id": job.id, "url": job.url,
          "repo": T.OUTPUT_REPO}), flush=True)
    try:
        api.upload_file(
            repo_id=T.OUTPUT_REPO,
            path_in_repo="evidence/launch-submission.json",
            path_or_fileobj=json.dumps({"job_id":job.id,"url":job.url,
                                        "code_commit":revision,
                                        "training_started":False},indent=2).encode(),
            commit_message="Record WR2 GPU job submission",
        )
    except Exception as exc:
        print("WR2_SUBMISSION_RECORD_WARNING", type(exc).__name__, flush=True)

if __name__ == "__main__":
    main()
