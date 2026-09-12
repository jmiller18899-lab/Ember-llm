#!/usr/bin/env python3
# /// script
# requires-python = "==3.11.*"
# dependencies = [
#   "torch==2.6.0",
#   "huggingface-hub==1.31.0",
#   "sentencepiece==0.2.0",
#   "numpy==2.2.6",
#   "trackio",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""A single bounded HF learning trial; all outputs stay in a separate private repo."""
from __future__ import annotations

import argparse
import builtins
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

SOURCE_COMMIT = "04e879ae479dab166daef32febe9fbd6f5507f37"
SOURCE_FILES = (
    "direct_answers/__init__.py",
    "direct_answers/learn.py",
    "direct_answers/quality.py",
    "direct_answers/canary-data.json",
    "tool_assistant/__init__.py",
    "tool_assistant/runtime.py",
    "tool_assistant/binary.py",
    "tool_assistant/family.py",
    "tool_assistant/resolver.py",
    "tool_assistant/data/router-training.json",
)
OVERRIDES = {
    "steps": 600,
    "checkpoints": [40, 80, 120, 200, 300, 400, 500, 600],
    "maximum_training_seconds": 2400,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def authenticated_api(expected_repo):
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("The existing Hugging Face write secret is unavailable")
    api = HfApi(token=token)
    if api.whoami()["name"].casefold() != expected_repo.split("/")[0].casefold():
        raise RuntimeError("Unexpected Hugging Face account")
    return api


def prepare(args):
    from tool_assistant import build
    from tool_assistant.runtime import load_model
    from direct_answers import learn
    import torch
    root = Path(__file__).resolve().parent.parent
    api = authenticated_api(args.repo)
    output = Path(args.prepared).resolve()
    if output.exists():
        raise RuntimeError("Preparation directory already exists")
    output.mkdir(parents=True)
    bundle = output / "bundle"
    old_argv = sys.argv
    try:
        sys.argv = ["tool_assistant.build", "--out", str(bundle)]
        build.main()
    finally:
        sys.argv = old_argv

    data = json.loads((root / "direct_answers/canary-data.json").read_text())
    tool_rows = json.loads((root / "tool_assistant/data/router-training.json").read_text())["cases"]
    learn.validate_data(data, tool_rows)
    model, tokenizer, manifest = load_model(bundle, "full")
    selected = learn.configure_training(model)
    before = learn.frozen_hash(model)
    train = [learn.encode_row(tokenizer, r, model.cfg.block_size) for r in data["train"]]
    dev = [learn.encode_row(tokenizer, r, model.cfg.block_size) for r in data["development"]]
    losses = {
        "training": learn.mean_loss(model, train),
        "development": learn.mean_loss(model, dev),
    }
    if not all(torch.isfinite(torch.tensor(v)).item() and v > 0 for v in losses.values()):
        raise RuntimeError("Invalid baseline answer-token loss")
    if learn.frozen_hash(model) != before:
        raise RuntimeError("Preflight changed protected parameters")
    del model

    for relative in SOURCE_FILES:
        dest = output / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, dest)
    shutil.copyfile(__file__, output / "runner.py")
    original_rows = {key: data[key] for key in ("train", "development", "confirmation")}
    data["training_config"] = {**data["training_config"], **OVERRIDES}
    assert original_rows == {key: data[key] for key in original_rows}
    write_json(output / "effective-data.json", data)
    recipe = {
        "status": "PREFLIGHT_PASS",
        "source_commit": SOURCE_COMMIT,
        "run_name": args.run_name,
        "target_repo": args.repo,
        "source_model_repo": manifest["model_repo"],
        "source_model_revision": manifest["model_revision"],
        "baseline_losses": losses,
        "train_examples": len(train),
        "development_examples": len(dev),
        "confirmation_examples": len(data["confirmation"]),
        "confirmation_evaluated": False,
        "training_config": data["training_config"],
        "trainable_names": list(selected),
        "trainable_parameters": sum(p.numel() for p in selected.values()),
        "frozen_parameter_sha256": before,
        "flavor": "cpu-upgrade",
        "job_timeout_seconds": 3600,
        "production_ready": False,
        "files": {p.relative_to(output).as_posix(): digest(p)
                  for p in sorted(output.rglob("*"))
                  if p.is_file() and "__pycache__" not in p.parts},
    }
    write_json(output / "preflight.json", recipe)
    api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
    if not api.model_info(args.repo).private:
        raise RuntimeError("Candidate output repository must be private")
    api.upload_folder(
        repo_id=args.repo, repo_type="model", folder_path=str(output),
        path_in_repo="input", ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
        commit_message="Freeze verified inputs for one 600-step Ember direct-answer trial",
    )
    print(json.dumps({"event": "hf_direct_preflight_pass", "repo": args.repo,
                      "train_examples": len(train), "steps": 600,
                      "baseline_losses": losses}), flush=True)


def train(args):
    from huggingface_hub import snapshot_download
    api = authenticated_api(args.repo)
    if not api.model_info(args.repo).private:
        raise RuntimeError("Candidate output repository must be private")
    with tempfile.TemporaryDirectory(prefix="ember-direct-live-") as temp:
        work = Path(temp)
        snapshot = Path(snapshot_download(
            repo_id=args.repo, revision=args.input_revision,
            allow_patterns=["input/**"], local_dir=work / "snapshot",
        )) / "input"
        recipe = json.loads((snapshot / "preflight.json").read_text())
        if recipe["status"] != "PREFLIGHT_PASS" or recipe["run_name"] != args.run_name:
            raise RuntimeError("Unexpected or failed preflight")
        for relative, expected in recipe["files"].items():
            path = (snapshot / relative).resolve()
            if not path.is_relative_to(snapshot.resolve()) or digest(path) != expected:
                raise RuntimeError(f"Frozen input checksum mismatch: {relative}")
        if digest(__file__) != recipe["files"]["runner.py"]:
            raise RuntimeError("Runner does not match the verified input")
        data = json.loads((snapshot / "effective-data.json").read_text())
        original_data = json.loads((snapshot / "source/direct_answers/canary-data.json").read_text())
        for key in ("train", "development", "confirmation"):
            if data[key] != original_data[key]:
                raise RuntimeError(f"Unexpected example change in {key}")
        if data["training_config"] != {**original_data["training_config"], **OVERRIDES}:
            raise RuntimeError("Unexpected training recipe change")
        shutil.copyfile(snapshot / "effective-data.json", snapshot / "source/direct_answers/canary-data.json")
        sys.path.insert(0, str(snapshot / "source"))
        from direct_answers import learn

        artifacts = work / "artifacts"
        artifacts.mkdir()
        run_output = artifacts / "learning"
        os.environ["TRACKIO_DIR"] = str(artifacts / "trackio")
        import trackio
        trackio.init(project="ember-direct-answer", name=args.run_name, config={
            "steps": 600,
            "learning_rate": data["training_config"]["learning_rate"],
            "train_examples": 72,
            "trainable_scope": "blocks.5 and ln_f",
        })
        write_json(artifacts / "run.json", {
            "run_name": args.run_name, "input_revision": args.input_revision,
            "input_manifest_sha256": digest(snapshot / "preflight.json"),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "TRAINING_STARTED", "production_ready": False,
        })
        api.upload_file(path_or_fileobj=str(artifacts / "run.json"),
                        path_in_repo="results/run.json", repo_id=args.repo)
        print(json.dumps({"event": "training_started", "steps": 600,
                          "flavor": "cpu-upgrade", "repo": args.repo}), flush=True)

        def observed_print(*values, **kwargs):
            builtins.print(*values, **kwargs)
            if not values or not isinstance(values[0], str):
                return
            try:
                event = json.loads(values[0])
            except (TypeError, ValueError):
                return
            if event.get("event") == "direct_learning_checkpoint":
                step = event["step"]
                trackio.log({"step": step, "train_loss": event["train_loss"],
                             "development_loss": event["development_loss"]})
                checkpoint = run_output / f"checkpoint-{step}.pt"
                api.upload_file(path_or_fileobj=str(checkpoint),
                                path_in_repo=f"results/learning/checkpoint-{step}.pt",
                                repo_id=args.repo, commit_message=f"Persist learning checkpoint {step}")
                write_json(artifacts / "progress.json", event)
                api.upload_file(path_or_fileobj=str(artifacts / "progress.json"),
                                path_in_repo="results/progress.json", repo_id=args.repo)
            elif event.get("event") == "direct_learning_complete":
                trackio.log({k: v for k, v in event.items()
                             if isinstance(v, (int, float)) and not isinstance(v, bool)})

        learn.print = observed_print
        failure = None
        old_argv = sys.argv
        try:
            sys.argv = ["direct_answers.learn", "--bundle", str(snapshot / "bundle"),
                        "--out", str(run_output)]
            learn.main()
            report = json.loads((run_output / "report.json").read_text())
            if not report["block04_features_identical"]:
                raise RuntimeError("Routing representation preservation failed")
            if report["frozen_parameter_sha256_before"] != report["frozen_parameter_sha256_after"]:
                raise RuntimeError("Frozen parameter preservation failed")
            if [r["step"] for r in report["checkpoints"]] != OVERRIDES["checkpoints"]:
                raise RuntimeError("The requested 600 steps did not finish")
            status = {"status": "TRAINING_COMPLETED", "steps_completed": 600,
                      "selected_step": report["selected_step"],
                      "development_progress_gate": report["development_progress_gate"],
                      "candidate_quality": report["candidate_quality"]["passed"],
                      "baseline_quality": report["baseline_quality"]["passed"],
                      "production_ready": False}
        except BaseException as exc:
            failure = exc
            status = {"status": "TRAINING_FAILED", "error_type": type(exc).__name__,
                      "production_ready": False}
        finally:
            sys.argv = old_argv
            try:
                trackio.finish()
            except Exception as exc:
                print(json.dumps({"event": "trackio_finish_error", "error_type": type(exc).__name__}), flush=True)
            write_json(artifacts / "status.json", status)
            api.upload_folder(repo_id=args.repo, folder_path=str(artifacts), path_in_repo="results",
                              ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
                              commit_message="Persist direct-answer training results and metrics")
            print(json.dumps(status), flush=True)
        if failure is not None:
            raise RuntimeError(f"Training failed: {type(failure).__name__}") from failure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "train"), required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--prepared", default="hf-direct-prepared")
    parser.add_argument("--input-revision")
    args = parser.parse_args()
    if args.mode == "train" and not args.input_revision:
        parser.error("Training requires a pinned input revision")
    (prepare if args.mode == "prepare" else train)(args)


if __name__ == "__main__":
    main()
