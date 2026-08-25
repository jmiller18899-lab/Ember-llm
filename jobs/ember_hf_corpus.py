# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "datasets==5.0.1",
#   "datasketch>=2.0",
#   "huggingface-hub>=1.4",
#   "pytest>=8.0",
#   "sentencepiece>=0.2",
# ]
# ///
"""Build Ember's verified corpus on HF Jobs and persist it to a private dataset."""
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

from huggingface_hub import HfApi


PACKAGE_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/main/ember-v0.0.6-hf-ready.zip"
PACKAGE_SHA256 = "1705c2baaac0a51d4bed9cd9b1d7afe0a149aae4138ba351257aaec9d476635f"


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN was not injected as a Job secret")
    api = HfApi(token=token)
    owner = api.whoami()["name"]
    repo_id = f"{owner}/ember-corpus-v0.0.6"
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ember-corpus-") as temporary:
        work = Path(temporary)
        archive = work / "ember.zip"
        urllib.request.urlretrieve(PACKAGE_URL, archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != PACKAGE_SHA256:
            raise RuntimeError(f"Ember package checksum mismatch: {digest}")
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        root = work / "src" / "ember"
        if (root / "VERSION").read_text(encoding="utf-8").strip() != "0.0.6":
            raise RuntimeError("unexpected Ember package version")

        run([
            sys.executable, "-m", "pytest", "-q",
            "tests/test_corpus.py", "tests/test_corpus_pipeline.py",
            "tests/test_data.py", "tests/test_tokenizer.py",
        ], root)
        run([
            sys.executable, "scripts/build_training_corpus.py",
            "--config", "config/corpus_v0.0.6.json",
        ], root)

        processed = root / "data" / "processed"
        stats = json.loads((processed / "corpus_stats.json").read_text(encoding="utf-8"))
        if stats.get("status") != "PASS":
            raise RuntimeError("corpus gate did not report PASS")
        tokens = int(stats.get("actual_ember_tokens", 0))
        if not 10_000_000 <= tokens <= 20_000_000:
            raise RuntimeError(f"corpus token gate failed: {tokens}")
        if int(stats.get("vocab_size", 0)) != 16_384:
            raise RuntimeError("tokenizer vocabulary gate failed")

        card = work / "README.md"
        card.write_text(
            "---\npretty_name: Ember v0.0.6 verified training corpus\n"
            "license: other\nprivate: true\n---\n\n"
            "Private staging corpus for Ember v0.0.6. Source revisions, licenses, "
            "document hashes, and attribution metadata are recorded in `data/provenance.json`.\n",
            encoding="utf-8",
        )
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(processed),
            path_in_repo="data",
        )
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=str(card),
            path_in_repo="README.md",
        )

        print("EMBER_HF_CORPUS=PASS")
        print(f"CORPUS_REPO={repo_id}")
        print(f"ACTUAL_EMBER_TOKENS={tokens}")
        print(f"PACKAGE_SHA256={digest}")
        print("GPU_TRAINING_LAUNCHED=false")


if __name__ == "__main__":
    main()
