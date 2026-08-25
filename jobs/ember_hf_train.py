# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "huggingface-hub>=1.4",
#   "sentencepiece>=0.2",
#   "torch>=2.4",
#   "trackio",
# ]
# ///
"""Run Ember's gated T4 validation training with Trackio and Hub persistence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

from huggingface_hub import HfApi, snapshot_download
import trackio


PACKAGE_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/main/ember-v0.0.6-hf-ready.zip"
PACKAGE_SHA256 = "1705c2baaac0a51d4bed9cd9b1d7afe0a149aae4138ba351257aaec9d476635f"


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN was not injected as a Job secret")
    api = HfApi(token=token)
    owner = api.whoami()["name"]
    corpus_repo = f"{owner}/ember-corpus-v0.0.6"
    model_repo = f"{owner}/ember-v0.0.6-t4"
    trackio_space = f"{owner}/trackio"
    run_name = "ember-v0.0.6-t4-500-step-validation"

    with tempfile.TemporaryDirectory(prefix="ember-train-") as temporary:
        work = Path(temporary)
        archive = work / "ember.zip"
        urllib.request.urlretrieve(PACKAGE_URL, archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != PACKAGE_SHA256:
            raise RuntimeError(f"Ember package checksum mismatch: {digest}")
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        root = work / "src" / "ember"

        corpus = work / "corpus"
        snapshot_download(
            repo_id=corpus_repo,
            repo_type="dataset",
            local_dir=corpus,
            token=token,
        )
        stats = json.loads((corpus / "data" / "corpus_stats.json").read_text(encoding="utf-8"))
        if stats.get("status") != "PASS" or not 10_000_000 <= int(stats.get("actual_ember_tokens", 0)) <= 20_000_000:
            raise RuntimeError("refusing GPU training: verified corpus gate is not PASS")

        api.create_repo(model_repo, repo_type="model", private=True, exist_ok=True)
        trackio.init(
            project="ember",
            name=run_name,
            space_id=trackio_space,
            config={
                "version": "0.0.6",
                "hardware": "t4-small",
                "max_steps": 500,
                "block_size": 512,
                "batch_size": 8,
                "gradient_accumulation_steps": 4,
                "corpus_tokens": int(stats["actual_ember_tokens"]),
            },
        )

        env = os.environ.copy()
        env.update({
            "EMBER_CONFIG": "config/ember_agent_t4_validation.json",
            "EMBER_DATA": str(corpus / "data" / "train.txt"),
            "EMBER_VAL_DATA": str(corpus / "data" / "val.txt"),
            "EMBER_OUTPUT_DIR": "checkpoints",
            "EMBER_HF_REPO": model_repo,
            "EMBER_REQUIRE_CUDA_PREFLIGHT": "1",
            "EMBER_EXPORT_INT4": "1",
        })
        command = [sys.executable, "scripts/hf_train.py"]
        print("+", " ".join(command), flush=True)
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "step" in payload:
                    trackio.log({
                        "train/micro_loss": payload.get("micro_train_loss"),
                        "train/loss": payload.get("train_loss"),
                        "validation/loss": payload.get("val_loss"),
                        "learning_rate": payload.get("lr"),
                        "elapsed_seconds": payload.get("elapsed_s"),
                        "training_step": payload.get("step"),
                    })
            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
        finally:
            trackio.finish()

        files = api.list_repo_files(model_repo, repo_type="model")
        if not any(path.endswith("best.pt") for path in files):
            raise RuntimeError("training finished but best.pt was not persisted")
        if not any(path.endswith("best.int4.pt") for path in files):
            raise RuntimeError("training finished but INT4 export was not persisted")
        print("EMBER_HF_T4_VALIDATION=PASS")
        print(f"MODEL_REPO={model_repo}")
        print(f"TRACKIO_SPACE={trackio_space}")


if __name__ == "__main__":
    main()
