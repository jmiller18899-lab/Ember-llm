"""Fit the v5 text helper and preserve both previously frozen checkpoints."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import torch

from .build import save_manifest
from .evidence_v5 import emit_json
from .routing_data_v5 import load_training
from .routing_v5 import REVISION, select
from .runtime import sha256


def build(base, out):
    base, out = Path(base), Path(out)
    if out.exists():
        raise ValueError("Refusing to overwrite a v5 candidate")
    root = Path(__file__).resolve().parent.parent
    before = sha256(base / "manifest.json")
    manifest = json.loads((base / "manifest.json").read_text())
    if manifest.get("routing_revision") != "routing-head-v4":
        raise ValueError("Expected the frozen v4 control bundle")
    for name, digest in manifest["files"].items():
        target = (base / name).resolve()
        if not target.is_relative_to(base.resolve()) or sha256(target) != digest:
            raise ValueError(f"Original candidate checksum mismatch: {name}")
    historical, development = load_training(root)
    cases = historical + development
    state, report = select(cases, len(historical))
    shutil.copytree(base, out, ignore=shutil.ignore_patterns("__pycache__"))
    torch.save(state, out / "routing-v5.pt")
    (out / "routing-v5-training.json").write_text(json.dumps(cases, indent=2) + "\n")
    (out / "routing-v5-selection.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for name in ("resolver_v4.py", "routing_data_v5.py", "routing_v5.py", "runtime_v5.py"):
        shutil.copyfile(root / "tool_assistant" / name, out / "tool_assistant" / name)
    metadata = {k: v for k, v in manifest.items() if k != "files"}
    metadata.update(created_at=datetime.now(timezone.utc).isoformat(),
                    status="UNTESTED_ROUTING_V5_CANDIDATE", routing_revision=REVISION,
                    argument_parser_revision="argument-parser-v4", base_manifest_sha256=before,
                    base_model_trained=False, uses_ember_features=False, production_ready=False,
                    routing_selection=report["selection"],
                    routing_training_sha256=sha256(out / "routing-v5-training.json"))
    save_manifest(out, metadata)
    if sha256(base / "manifest.json") != before:
        raise RuntimeError("The v4 control bundle changed")
    for name, digest in manifest["files"].items():
        if sha256(out / name) != digest:
            raise RuntimeError(f"A preserved candidate file changed: {name}")
    print(json.dumps({"event": "routing_v5_frozen", "manifest_sha256": sha256(out / "manifest.json"),
                      "selection": report["selection"], "base_model_trained": False,
                      "uses_ember_features": False}), flush=True)
    emit_json(out / "manifest.json", "routing-v5-manifest.json")
    emit_json(out / "routing-v5-selection.json", "routing-v5-selection.json")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20260912)
    torch.use_deterministic_algorithms(True)
    build(args.base, args.out)


if __name__ == "__main__":
    main()
