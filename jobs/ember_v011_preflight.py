# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "huggingface-hub>=1.4",
#   "sentencepiece>=0.2",
#   "torch>=2.4",
# ]
# ///
"""CPU-only Ember v0.0.11 corrective preflight."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile

from huggingface_hub import HfApi, hf_hub_download

PIN = "12645f8166c7ae583eb6f645f672a4352efb4f5d"
CONFIG_URL = f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{PIN}/config/ember_agent_semantic_sft_v0.0.11.json"
DATA_URL = f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{PIN}/jobs/ember_sft_data_v011.py"
PACKAGE_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/main/ember-v0.0.7-hf-ready.zip"
PACKAGE_SHA256 = "27e8f7c80317652a22b3d58a0bd474724491a685dfe9e20c0b997b7c5907a289"
SOURCE_REPO = "Jmiller18899/ember-v0.0.9-t4"
SOURCE_CHECKPOINT = "checkpoints/ember-agent-v0.0.9-tool-sft-20260828T142047Z/best.pt"
SOURCE_SHA256 = "8299d52e8a852b9bd3e8403e086b48fd42d2babfd51c74eb94af29bd87ef2d13"

def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def fetch(url, path):
    urllib.request.urlretrieve(url, path)
    return path

def load_module(path):
    spec = importlib.util.spec_from_file_location("ember_sft_data_v011", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load v0.0.11 data module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def kind_counts(rows):
    return {kind: sum(row["kind"] == kind for row in rows) for kind in ("tool_call","direct_response","tool_result_response")}

def main():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is required")
    api = HfApi(token=token)
    with tempfile.TemporaryDirectory(prefix="ember-v011-preflight-") as td:
        work = Path(td)
        config_path = fetch(CONFIG_URL, work / "config.json")
        data_path = fetch(DATA_URL, work / "data.py")
        config = json.loads(config_path.read_text())
        data = load_module(data_path)
        train = data.build_examples("train", int(config["train_examples"]))
        val = data.build_examples("validation", int(config["validation_examples"]))
        assert kind_counts(train) == {"tool_call":600,"direct_response":600,"tool_result_response":600}
        assert kind_counts(val) == {"tool_call":80,"direct_response":80,"tool_result_response":80}

        archive = fetch(PACKAGE_URL, work / "ember.zip")
        if sha(archive) != PACKAGE_SHA256:
            raise RuntimeError("authoritative package checksum mismatch")
        with zipfile.ZipFile(archive) as z:
            z.extractall(work / "src")
        root = work / "src" / "ember"
        import sys
        sys.path.insert(0, str(root))
        from src.checkpoint import load_checkpoint
        from src.model import EmberGPT, ModelConfig
        from src.tokenizer import tokenizer_from_state_dict

        files = set(api.list_repo_files(SOURCE_REPO, repo_type="model"))
        for required in (SOURCE_CHECKPOINT, "evaluations/latest.json"):
            if required not in files:
                raise RuntimeError(f"source repo missing {required}")
        ckpt_path = Path(hf_hub_download(
            repo_id=SOURCE_REPO, repo_type="model", filename=SOURCE_CHECKPOINT,
            token=token, local_dir=work / "source-model",
        ))
        if sha(ckpt_path) != SOURCE_SHA256:
            raise RuntimeError("accepted v0.0.9 checkpoint checksum mismatch")
        eval_path = Path(hf_hub_download(
            repo_id=SOURCE_REPO, repo_type="model", filename="evaluations/latest.json",
            token=token, local_dir=work / "source-model",
        ))
        source_eval = json.loads(eval_path.read_text())
        if not source_eval.get("technical_pass", False):
            raise RuntimeError("v0.0.9 source evaluation is not technical_pass")

        source = load_checkpoint(ckpt_path, device="cpu")
        tok = tokenizer_from_state_dict(source["tokenizer"])
        if int(config["block_size"]) > int(source["model_config"]["block_size"]):
            raise RuntimeError("v0.0.11 block_size exceeds source context")

        max_tokens = 0
        min_completion_tokens = 10**9
        focus_terms = 0
        for row in train + val:
            p = tok.encode(row["prompt"])
            full = tok.encode(row["prompt"] + row["completion"])
            if full[:len(p)] != p:
                raise RuntimeError(f"tokenization boundary changed: {row['id']}")
            if len(full) > int(config["block_size"]) + 1:
                raise RuntimeError(f"sequence too long: {row['id']}={len(full)}")
            completion_ids = full[len(p):]
            if len(completion_ids) < 2:
                raise RuntimeError(f"empty completion: {row['id']}")
            max_tokens = max(max_tokens, len(full))
            min_completion_tokens = min(min_completion_tokens, len(completion_ids))
            text = row["completion"].casefold()
            for term in row["focus_terms"]:
                focus_terms += 1
                if str(term).casefold() not in text:
                    raise RuntimeError(f"semantic focus absent from target: {row['id']} {term!r}")

        eot_ids = tok.encode("<|endoftext|>")
        if not eot_ids:
            raise RuntimeError("tokenizer cannot encode endoftext")

        import torch
        model = EmberGPT(ModelConfig(**source["model_config"]))
        model.load_state_dict(source["model_state"])
        model.eval()
        losses = []
        with torch.inference_mode():
            for row in val[:12]:
                p = tok.encode(row["prompt"])
                full = tok.encode(row["prompt"] + row["completion"])
                x = torch.tensor([full[:-1]], dtype=torch.long)
                y = torch.tensor([full[1:]], dtype=torch.long)
                y[:, :max(0, len(p)-1)] = -100
                _, loss = model(x, y)
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError("non-finite completion loss in CPU preflight")
                losses.append(float(loss.item()))

        result = {
            "status":"PASS",
            "version":"0.0.11",
            "source_repo":SOURCE_REPO,
            "source_checkpoint_sha256":SOURCE_SHA256,
            "train_examples":len(train),
            "validation_examples":len(val),
            "train_kind_counts":kind_counts(train),
            "validation_kind_counts":kind_counts(val),
            "semantic_focus_terms_verified":focus_terms,
            "max_sequence_tokens":max_tokens,
            "min_completion_tokens":min_completion_tokens,
            "endoftext_ids":eot_ids,
            "source_completion_loss_sample":sum(losses)/len(losses),
            "semantic_token_weight":config["semantic_token_weight"],
            "eos_token_weight":config["eos_token_weight"],
            "next_gate":"implement weighted-loss trainer + EOS-aware generation smoke; do not launch T4 unless CPU checks pass",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        print("EMBER_V011_PREFLIGHT=PASS")

if __name__ == "__main__":
    main()
