# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "huggingface-hub>=1.4",
# ]
# ///
"""Inspect the accepted v0.0.9 and failed v0.0.10 evaluation artifacts.

This is deliberately CPU-only and read-only. It prints the promotion metrics and
held-out completions needed to design the next Ember corrective SFT without
relaunching a paid GPU job.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not TOKEN:
    raise SystemExit("HF_TOKEN is required")

REPOS = (
    "Jmiller18899/ember-v0.0.9-t4",
    "Jmiller18899/ember-v0.0.10-t4",
)


def load_json(repo_id: str, filename: str, work: Path):
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=filename,
        token=TOKEN,
        local_dir=work / repo_id.replace("/", "__"),
    )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def first_present(obj: dict, *keys):
    for key in keys:
        if key in obj:
            return obj[key]
    return None


def summarize(repo_id: str, payload: dict) -> None:
    print(f"\n=== {repo_id} ===")
    for key in (
        "version",
        "label",
        "technical_pass",
        "promotion_eligible",
        "held_out_loss",
        "validation_loss",
        "tool_name_correct",
        "tool_argument_correct",
        "tool_result_grounded",
        "direct_response_correct",
        "clean_stop_correct",
        "int4_pass",
    ):
        if key in payload:
            print(f"{key}={payload[key]}")

    metrics = first_present(payload, "metrics", "promotion_metrics", "scores")
    if isinstance(metrics, dict):
        print("metrics=" + json.dumps(metrics, sort_keys=True))

    cases = first_present(payload, "cases", "evaluation_cases", "results")
    if isinstance(cases, list):
        print(f"cases={len(cases)}")
        for row in cases:
            if not isinstance(row, dict):
                continue
            slim = {}
            for key in (
                "id",
                "kind",
                "expected_tool",
                "expected_arguments",
                "required_facts",
                "tool_name_correct",
                "arguments_correct",
                "facts_grounded",
                "direct_response_correct",
                "clean_stop",
                "passed",
                "completion",
            ):
                if key in row:
                    slim[key] = row[key]
            print(json.dumps(slim, ensure_ascii=False, sort_keys=True))
    else:
        # The artifact schema may nest the per-case data. Printing the JSON is
        # still safe: evaluation artifacts contain model outputs/metrics, not secrets.
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    api = HfApi(token=TOKEN)
    with tempfile.TemporaryDirectory(prefix="ember-v011-diag-") as td:
        work = Path(td)
        for repo_id in REPOS:
            files = set(api.list_repo_files(repo_id, repo_type="model"))
            print(f"{repo_id}: {len(files)} files")
            candidates = [
                "evaluations/latest.json",
                "run-state.json",
            ]
            found = [name for name in candidates if name in files]
            print("found=" + ",".join(found))
            if "evaluations/latest.json" in files:
                summarize(repo_id, load_json(repo_id, "evaluations/latest.json", work))
            elif "run-state.json" in files:
                summarize(repo_id, load_json(repo_id, "run-state.json", work))


if __name__ == "__main__":
    main()
