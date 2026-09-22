"""Freeze a routing-head experiment without altering the original bundle."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import torch

from .build import save_manifest
from .routing_data_v4 import load_training
from .routing_v4 import REVISION, select
from .runtime import hidden, load_model, prompt, sha256


def build(base, out):
    base, out = Path(base), Path(out)
    if out.exists():
        raise ValueError("Refusing to overwrite a routing candidate")
    root = Path(__file__).resolve().parent.parent
    historical, development = load_training(root)
    cases = historical + development
    before = sha256(base / "manifest.json")
    metadata = json.loads((base / "manifest.json").read_text())
    metadata.pop("files")
    metadata.update(created_at=datetime.now(timezone.utc).isoformat(),
                    status="UNTESTED_ROUTING_CANDIDATE", routing_revision=REVISION,
                    base_manifest_sha256=before, production_ready=False,
                    base_model_trained=False, argument_parser_revision="argument-parser-v3")
    shutil.copytree(base, out, ignore=shutil.ignore_patterns("__pycache__"))
    (out / "routing-development.json").write_text(json.dumps(cases, indent=2) + "\n")
    reports = {}
    for precision in ("full", "int4"):
        model, tokenizer, _ = load_model(base, precision)
        # Historical fitting prompts varied; use the exact inference prompt
        # for every frozen representation in this new candidate.
        vectors = []
        for index, case in enumerate(cases):
            vectors.append(hidden(model, tokenizer, prompt(case["user"])))
            if (index + 1) % 100 == 0:
                print(json.dumps({"event": "routing_features", "precision": precision,
                                  "completed": index + 1, "total": len(cases)}), flush=True)
        state, report = select(cases, torch.stack(vectors), len(historical))
        torch.save(state, out / f"routing-v4-{precision}.pt")
        reports[precision] = report
        print(json.dumps({"event": "routing_head_selected", "precision": precision,
                          "selection": report["selection"]}), flush=True)
        del model, state, vectors
    for name in ("routing_data_v4.py", "routing_v4.py", "runtime_v4.py", "runtime_v3.py", "resolver_v3.py"):
        shutil.copyfile(root / "tool_assistant" / name, out / "tool_assistant" / name)
    metadata["routing_development_sha256"] = sha256(out / "routing-development.json")
    metadata["routing_selection"] = {key: value["selection"] for key, value in reports.items()}
    (out / "routing-development-report.json").write_text(json.dumps(reports, indent=2) + "\n")
    save_manifest(out, metadata)
    if sha256(base / "manifest.json") != before:
        raise RuntimeError("Original bundle changed during the routing experiment")
    for precision in ("full", "int4"):
        if sha256(base / f"model-{precision}.pt") != sha256(out / f"model-{precision}.pt"):
            raise RuntimeError("A frozen model changed")
    return reports


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
