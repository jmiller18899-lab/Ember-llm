# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["huggingface-hub>=1.4", "sentencepiece>=0.2", "torch>=2.4"]
# ///
"""Export and verify an INT4 checkpoint for the promoted Ember v0.0.31 model.

CPU-only. Refuses to export unless run-state.json says v0.0.31 is evaluation_complete,
promotion PASS, and best_step 479. The resulting artifact is written beside best.pt.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request
import zipfile

from huggingface_hub import HfApi, hf_hub_download
import torch

REPO = "Jmiller18899/ember-v0.0.31-t4"
PACKAGE_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/main/ember-v0.0.7-hf-ready.zip"
PACKAGE_SHA256 = "27e8f7c80317652a22b3d58a0bd474724491a685dfe9e20c0b997b7c5907a289"
EXPECTED_BEST_STEP = 479


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    api = HfApi(token=token)

    with tempfile.TemporaryDirectory(prefix="ember-v031-int4-") as td:
        work = Path(td)
        state_path = Path(hf_hub_download(
            repo_id=REPO,
            repo_type="model",
            filename="run-state.json",
            token=token,
            local_dir=work / "state",
        ))
        state = json.loads(state_path.read_text())
        if state.get("status") != "evaluation_complete":
            raise RuntimeError(f"v0.0.31 is not evaluation_complete: {state}")
        if state.get("promotion") != "PASS":
            raise RuntimeError(f"v0.0.31 copy promotion is not PASS: {state}")
        if int(state.get("best_step", -1)) != EXPECTED_BEST_STEP:
            raise RuntimeError(f"unexpected best_step: {state.get('best_step')}")

        run_id = str(state["run_id"])
        best_repo_path = f"checkpoints/{run_id}/best.pt"
        best_path = Path(hf_hub_download(
            repo_id=REPO,
            repo_type="model",
            filename=best_repo_path,
            token=token,
            local_dir=work / "model",
        ))

        archive = work / "ember.zip"
        urllib.request.urlretrieve(PACKAGE_URL, archive)
        actual = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual != PACKAGE_SHA256:
            raise RuntimeError(f"Ember package checksum mismatch: {actual}")
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        root = work / "src" / "ember"
        sys.path.insert(0, str(root))

        from src.quantize_int4 import export_int4_checkpoint, dequantize_tensor
        from src.model import EmberGPT, ModelConfig
        from src.tokenizer import tokenizer_from_state_dict

        out = work / "best.int4.pt"
        export_int4_checkpoint(str(best_path), str(out))
        payload = torch.load(out, map_location="cpu", weights_only=False)
        state_dict = {
            name: dequantize_tensor(value)
            for name, value in payload["quantized_state"].items()
        }
        state_dict.update(payload.get("passthrough_state", {}))
        model = EmberGPT(ModelConfig(**payload["model_config"]))
        model.load_state_dict(state_dict)
        model.eval()
        tokenizer = tokenizer_from_state_dict(payload["tokenizer"])
        prompt_ids = tokenizer.encode("<|system|>You are Ember.<|user|>Ready?<|assistant|>")
        x = torch.tensor([prompt_ids], dtype=torch.long)
        with torch.inference_mode():
            logits, _ = model(x, None)
        if not bool(torch.isfinite(logits).all().item()):
            raise RuntimeError("INT4 reconstruction produced non-finite logits")

        remote = f"checkpoints/{run_id}/best.int4.pt"
        api.upload_file(
            repo_id=REPO,
            repo_type="model",
            path_or_fileobj=str(out),
            path_in_repo=remote,
            commit_message="Export promoted Ember v0.0.31 INT4 checkpoint",
        )
        print("EMBER_V031_INT4_EXPORT=PASS", flush=True)
        print(f"RUN_ID={run_id}", flush=True)
        print(f"BEST_STEP={state['best_step']}", flush=True)
        print(f"INT4_PATH={remote}", flush=True)
        print(f"INT4_BYTES={out.stat().st_size}", flush=True)
        print("INT4_RECONSTRUCTION=PASS", flush=True)


if __name__ == "__main__":
    main()
