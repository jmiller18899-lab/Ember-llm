"""Fit on the original training data once, then export an immutable candidate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

from huggingface_hub import HfApi, hf_hub_download
import torch

from . import binary, family
from .runtime import hidden, load_model, sha256

RUN = "ember-v053-step9-20260910T165742Z"
SOURCE = {
    "full": {"path": f"checkpoints/{RUN}/best.pt", "sha256": "700e4259b2e724c8fa384e23ba1a6e67636c4f110863e887499f06951622c9ad"},
    "int4": {"path": f"checkpoints/{RUN}/best.int4.pt", "sha256": "d77c466fb8a9e602796aed3dffa4070b55cae8b492057b8f59a672ab6a1c1c78"},
}
PACKAGE_SHA256 = "27e8f7c80317652a22b3d58a0bd474724491a685dfe9e20c0b997b7c5907a289"


def save_manifest(out, metadata):
    files = {p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob("*"))
             if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts}
    (out / "manifest.json").write_text(json.dumps({**metadata, "files": files}, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if out.exists():
        raise RuntimeError("Refusing to overwrite an existing frozen candidate")
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(20260910)
    torch.use_deterministic_algorithms(True)
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    repo = f"{api.whoami()['name']}/ember-v0.0.53-t4"
    revision = api.model_info(repo).sha
    root = Path(__file__).resolve().parent.parent
    archive = root / "ember-v0.0.7-hf-ready.zip"
    if sha256(archive) != PACKAGE_SHA256:
        raise RuntimeError("Authoritative Ember package checksum mismatch")
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            if entry.is_dir() or not entry.filename.startswith("ember/src/"):
                continue
            target = (out / entry.filename).resolve()
            if not target.is_relative_to(out.resolve()):
                raise RuntimeError("Unsafe package entry")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(entry))
    metadata = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "model_repo": repo, "model_revision": revision, "model_sources": SOURCE,
        "package_sha256": PACKAGE_SHA256, "router_source_commit": "13bcf0eb645c5b1f1fd5423982f6b89ab8c923f2",
        "resolver_source_commit": "10683d1f62fb9c2eff04f9f522c4f8ffb3907414",
        "build_commit": os.environ.get("GITHUB_SHA"), "runtime_system": __import__("tool_assistant.runtime", fromlist=["SYSTEM"]).SYSTEM,
        "status": "UNTESTED_CANDIDATE", "production_ready": False,
        "int4_execution": "INT4 checkpoint dequantized to float32 on CPU; not a packed INT4 kernel",
        "dependencies": {"torch": torch.__version__, "numpy": __import__("numpy").__version__, "sentencepiece": __import__("sentencepiece").__version__},
    }
    with tempfile.TemporaryDirectory() as td:
        for precision, source in SOURCE.items():
            path = Path(hf_hub_download(repo, source["path"], revision=revision, token=token, local_dir=td))
            if sha256(path) != source["sha256"]:
                raise RuntimeError(f"Source checkpoint hash mismatch: {precision}")
            # Legacy checkpoint is trusted only after matching its pinned digest.
            payload = torch.load(path, map_location="cpu", weights_only=False)
            keys = ("model_config", "tokenizer", "model_state") if precision == "full" else ("model_config", "tokenizer", "quantized_state", "passthrough_state")
            if precision == "full":
                cfg = payload.get("train_config", {})
                if cfg.get("version") != "0.0.53" or int(cfg.get("anchor_steps", -1)) != 9:
                    raise RuntimeError("Unexpected source checkpoint metadata")
            torch.save({k: payload[k] for k in keys if k in payload}, out / f"model-{precision}.pt")
            del payload
    training_path = root / "tool_assistant/data/router-training.json"
    training = json.loads(training_path.read_text())
    by_id = {c["id"]: c for c in training["cases"]}
    binary_cases = [by_id[i] for i in training["binary_ids"]]
    family_cases = [by_id[i] for i in training["family_ids"]]
    text_head = family.select_family(family_cases)
    if text_head["cv"]["mode"] != "hybrid" or text_head["cv"]["ridge"] != 0.01:
        raise RuntimeError("Original family configuration did not reproduce")
    torch.save(text_head, out / "family.pt")
    metadata.update(training_sha256=sha256(training_path), family_cv=text_head["cv"], binary_cv={})
    for precision in ("full", "int4"):
        save_manifest(out, metadata)
        model, tokenizer, _ = load_model(out, precision)
        x = torch.stack([hidden(model, tokenizer, c["prompt"]) for c in binary_cases])
        y = torch.tensor([0 if c["kind"] == "direct_response" else 1 for c in binary_cases])
        cv, records = binary.cv_ridge(x, y, 2)
        state = binary.fit_ridge(x, y, 2, cv["ridge"])
        torch.save(state, out / f"router-{precision}.pt")
        metadata["binary_cv"][precision] = {"selected": cv, "records": records}
        del model
    for name in ("__init__.py", "runtime.py", "family.py", "binary.py", "resolver.py", "__main__.py"):
        dest = out / "tool_assistant" / name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(root / "tool_assistant" / name, dest)
    save_manifest(out, metadata)
    print(json.dumps({"event": "candidate_frozen", "manifest_sha256": sha256(out / "manifest.json"), "family_cv": metadata["family_cv"], "binary_cv": metadata["binary_cv"]}), flush=True)


if __name__ == "__main__":
    main()
