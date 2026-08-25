#!/usr/bin/env python3
"""Verify Ember's GitHub-held HF token and create durable Hub repos."""
from __future__ import annotations

import os

from huggingface_hub import HfApi


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is unavailable to the launcher")

    api = HfApi(token=token)
    owner = api.whoami()["name"]
    corpus_repo = f"{owner}/ember-corpus-v0.0.7"
    model_repo = f"{owner}/ember-v0.0.7-t4"
    trackio_space = f"{owner}/ember-trackio"

    api.create_repo(corpus_repo, repo_type="dataset", private=True, exist_ok=True)
    api.create_repo(model_repo, repo_type="model", private=True, exist_ok=True)
    api.create_repo(
        trackio_space,
        repo_type="space",
        private=False,
        space_sdk="static",
        exist_ok=True,
    )

    print("EMBER_HF_BOOTSTRAP=PASS")
    print(f"HF_OWNER={owner}")
    print(f"CORPUS_REPO={corpus_repo}")
    print(f"MODEL_REPO={model_repo}")
    print(f"TRACKIO_SPACE={trackio_space}")
    print("TRACKIO_MODE=static-snapshot-after-training")


if __name__ == "__main__":
    main()
