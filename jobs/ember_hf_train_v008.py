# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "huggingface-hub>=1.4",
#   "sentencepiece>=0.2",
#   "torch>=2.4",
#   "trackio[spaces]",
# ]
# ///
"""Run Ember v0.0.8's cost-gated 3,000-step T4 training milestone.

The long run starts from scratch so its cosine schedule is internally coherent.
If the same v0.0.8 run is interrupted, it resumes only from the durable
`resume/latest.pt` checkpoint in its private model repository.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


PACKAGE_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/main/ember-v0.0.7-hf-ready.zip"
PACKAGE_SHA256 = "27e8f7c80317652a22b3d58a0bd474724491a685dfe9e20c0b997b7c5907a289"
TRAIN_CONFIG_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/main/config/ember_agent_t4_long_v0.0.8.json"
TRAIN_CONFIG_SHA256 = "6e50aa1eb8be9a0db597451bcdd5b2a8df26c72dbc99b692ea1c279c40c4bef1"
RUN_STATE_PATH = "run-state.json"
RESUME_CHECKPOINT_PATH = "resume/latest.pt"
RESUME_MANIFEST_PATH = "resume/manifest.json"
RUN_STATE_STALE_SECONDS = 4 * 60 * 60


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, destination: Path, expected_sha256: str) -> Path:
    urllib.request.urlretrieve(url, destination)
    actual = sha256_file(destination)
    if actual != expected_sha256:
        raise RuntimeError(f"download checksum mismatch for {url}: {actual}")
    return destination


def run(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def load_repo_json(api: HfApi, repo_id: str, filename: str, token: str, local_dir: Path) -> dict | None:
    files = set(api.list_repo_files(repo_id, repo_type="model"))
    if filename not in files:
        return None
    downloaded = hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=filename,
        token=token,
        local_dir=local_dir,
    )
    return json.loads(Path(downloaded).read_text(encoding="utf-8"))


def assert_no_live_duplicate(previous_state: dict | None) -> None:
    if not previous_state:
        return
    status = previous_state.get("status")
    if status in {"training_complete_pending_eval", "evaluation_complete", "complete"}:
        raise RuntimeError("v0.0.8 training is already complete; refusing a duplicate paid run")
    if status != "running":
        return
    updated_at = str(previous_state.get("updated_at", ""))
    if not updated_at:
        raise RuntimeError("existing v0.0.8 run lock has no timestamp; manual review required")
    age = (utc_now() - parse_timestamp(updated_at)).total_seconds()
    if age < RUN_STATE_STALE_SECONDS:
        raise RuntimeError(f"another v0.0.8 run appears active (state age {age:.0f}s); refusing duplicate")


def upload_state(api: HfApi, repo_id: str, scratch: Path, state: dict, message: str) -> None:
    state = {**state, "updated_at": utc_now().isoformat()}
    path = write_json(scratch / "run-state.json", state)
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=str(path),
        path_in_repo=RUN_STATE_PATH,
        commit_message=message,
    )


def wait_for_checkpoint(path: Path, previous_mtime_ns: int, timeout_seconds: int = 180) -> int:
    deadline = time.monotonic() + timeout_seconds
    stable_size = -1
    stable_checks = 0
    while time.monotonic() < deadline:
        if path.is_file():
            stat = path.stat()
            if stat.st_size > 0 and stat.st_mtime_ns > previous_mtime_ns:
                if stat.st_size == stable_size:
                    stable_checks += 1
                else:
                    stable_size = stat.st_size
                    stable_checks = 0
                if stable_checks >= 2:
                    return stat.st_mtime_ns
        time.sleep(1)
    raise TimeoutError(f"checkpoint did not become durable within {timeout_seconds}s: {path}")


def upload_resume_checkpoint(
    api: HfApi,
    repo_id: str,
    scratch: Path,
    checkpoint: Path,
    log_path: Path,
    *,
    step: int,
    best_val_loss: float,
    run_id: str,
    corpus_tokens: int,
    launch_id: str,
) -> None:
    # Training continues while the Hub upload runs. Copy the checkpoint to an
    # immutable staging path so a later local save cannot mutate the file being
    # hashed or uploaded.
    staged_checkpoint = scratch / "resume-latest.pt"
    shutil.copy2(checkpoint, staged_checkpoint)
    if staged_checkpoint.stat().st_size != checkpoint.stat().st_size:
        raise RuntimeError("failed to stage a complete Ember resume checkpoint")
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=str(staged_checkpoint),
        path_in_repo=RESUME_CHECKPOINT_PATH,
        commit_message=f"Persist Ember v0.0.8 resume checkpoint at step {step}",
    )
    if log_path.is_file():
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=str(log_path),
            path_in_repo="resume/train.jsonl",
            commit_message=f"Persist Ember v0.0.8 training log at step {step}",
        )
    manifest = {
        "schema_version": 1,
        "status": "running",
        "version": "0.0.8",
        "launch_id": launch_id,
        "run_id": run_id,
        "step": step,
        "best_validation_loss": best_val_loss,
        "max_steps": 3000,
        "checkpoint_path": RESUME_CHECKPOINT_PATH,
        "checkpoint_sha256": sha256_file(staged_checkpoint),
        "checkpoint_bytes": staged_checkpoint.stat().st_size,
        "config_sha256": TRAIN_CONFIG_SHA256,
        "package_sha256": PACKAGE_SHA256,
        "corpus_tokens": corpus_tokens,
        "updated_at": utc_now().isoformat(),
    }
    manifest_path = write_json(scratch / "resume-manifest.json", manifest)
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=str(manifest_path),
        path_in_repo=RESUME_MANIFEST_PATH,
        commit_message=f"Update Ember v0.0.8 resume manifest at step {step}",
    )
    upload_state(
        api,
        repo_id,
        scratch,
        manifest,
        message=f"Update Ember v0.0.8 run state at step {step}",
    )


def resolve_resume_checkpoint(
    api: HfApi,
    repo_id: str,
    token: str,
    local_dir: Path,
) -> tuple[Path | None, dict | None]:
    files = set(api.list_repo_files(repo_id, repo_type="model"))
    has_checkpoint = RESUME_CHECKPOINT_PATH in files
    has_manifest = RESUME_MANIFEST_PATH in files
    if has_checkpoint != has_manifest:
        raise RuntimeError("partial v0.0.8 resume state found; manual review required")
    if not has_checkpoint:
        return None, None
    manifest_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=RESUME_MANIFEST_PATH,
        token=token,
        local_dir=local_dir,
    )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("config_sha256") != TRAIN_CONFIG_SHA256:
        raise RuntimeError("resume checkpoint was created by a different v0.0.8 training config")
    if int(manifest.get("step", -1)) >= 2999:
        raise RuntimeError("resume checkpoint already reached the final step; refusing duplicate training")
    checkpoint_path = Path(hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=RESUME_CHECKPOINT_PATH,
        token=token,
        local_dir=local_dir,
    ))
    if sha256_file(checkpoint_path) != manifest.get("checkpoint_sha256"):
        raise RuntimeError("downloaded resume checkpoint checksum does not match its manifest")
    return checkpoint_path, manifest


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN was not injected as a Job secret")

    api = HfApi(token=token)
    owner = api.whoami()["name"]
    corpus_repo = f"{owner}/ember-corpus-v0.0.7"
    model_repo = f"{owner}/ember-v0.0.8-t4"
    trackio_space = f"{owner}/ember-trackio"
    launch_id = f"ember-v0.0.8-{utc_now().strftime('%Y%m%dT%H%M%SZ')}"

    api.create_repo(model_repo, repo_type="model", private=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ember-v008-train-") as temporary:
        work = Path(temporary)
        previous_state = load_repo_json(api, model_repo, RUN_STATE_PATH, token, work / "prior-state")
        assert_no_live_duplicate(previous_state)

        archive = download_verified(PACKAGE_URL, work / "ember.zip", PACKAGE_SHA256)
        config_download = download_verified(
            TRAIN_CONFIG_URL,
            work / "ember_agent_t4_long_v0.0.8.json",
            TRAIN_CONFIG_SHA256,
        )
        config = json.loads(config_download.read_text(encoding="utf-8"))
        if config.get("version") != "0.0.8" or int(config.get("max_steps", 0)) != 3000:
            raise RuntimeError("unexpected Ember v0.0.8 long-run configuration")
        if config.get("initialization") != "from_scratch":
            raise RuntimeError("v0.0.8 must use a coherent fresh 3,000-step learning-rate schedule")

        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        root = work / "src" / "ember"
        config_target = root / "config" / "ember_agent_t4_long_v0.0.8.json"
        shutil.copy2(config_download, config_target)

        corpus = work / "corpus"
        snapshot_download(
            repo_id=corpus_repo,
            repo_type="dataset",
            local_dir=corpus,
            token=token,
        )
        stats = json.loads((corpus / "data" / "corpus_stats.json").read_text(encoding="utf-8"))
        corpus_tokens = int(stats.get("actual_ember_tokens", 0))
        if stats.get("status") != "PASS" or not 10_000_000 <= corpus_tokens <= 20_000_000:
            raise RuntimeError("refusing GPU training: verified corpus gate is not PASS")
        if int(stats.get("vocab_size", 0)) != 16_384:
            raise RuntimeError("refusing GPU training: verified tokenizer vocabulary is not 16,384")

        tokenizer_source = corpus / "data" / "ember_tokenizer.model"
        tokenizer_target = root / "data" / "processed" / "ember_tokenizer.model"
        if not tokenizer_source.is_file() or tokenizer_source.stat().st_size == 0:
            raise RuntimeError("verified corpus tokenizer is missing or empty")
        tokenizer_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tokenizer_source, tokenizer_target)
        if tokenizer_target.stat().st_size != tokenizer_source.stat().st_size:
            raise RuntimeError("failed to materialize the verified corpus tokenizer")

        resume_checkpoint, resume_manifest = resolve_resume_checkpoint(
            api,
            model_repo,
            token,
            work / "resume",
        )
        resume_step = int(resume_manifest["step"]) if resume_manifest else None

        trackio_dir = work / "trackio"
        os.environ["TRACKIO_DIR"] = str(trackio_dir)
        import trackio

        trackio.init(
            project="ember",
            name="ember-v0.0.8-t4-3000-step-long-run",
            embed=False,
            config={
                "version": "0.0.8",
                "hardware": "t4-small",
                "max_steps": 3000,
                "block_size": 512,
                "batch_size": 8,
                "gradient_accumulation_steps": 4,
                "corpus_tokens": corpus_tokens,
                "initialization": "from_scratch" if resume_checkpoint is None else "resume_v0.0.8",
                "resume_step": resume_step,
                "config_sha256": TRAIN_CONFIG_SHA256,
            },
        )

        state = {
            "schema_version": 1,
            "status": "running",
            "version": "0.0.8",
            "launch_id": launch_id,
            "run_id": resume_manifest.get("run_id") if resume_manifest else None,
            "step": resume_step,
            "max_steps": 3000,
            "config_sha256": TRAIN_CONFIG_SHA256,
            "package_sha256": PACKAGE_SHA256,
            "corpus_tokens": corpus_tokens,
            "resumed": resume_checkpoint is not None,
        }
        upload_state(api, model_repo, work, state, "Start Ember v0.0.8 long-run state")

        preflight = [
            sys.executable,
            "scripts/preflight.py",
            "--config",
            str(config_target.relative_to(root)),
            "--data",
            str(corpus / "data" / "train.txt"),
            "--val-data",
            str(corpus / "data" / "val.txt"),
            "--require-cuda",
        ]
        run(preflight, root)

        output_root = root / "checkpoints"
        command = [
            sys.executable,
            "-m",
            "src.train",
            "--config",
            str(config_target.relative_to(root)),
            "--data",
            str(corpus / "data" / "train.txt"),
            "--val-data",
            str(corpus / "data" / "val.txt"),
            "--output-dir",
            str(output_root),
        ]
        if resume_checkpoint is not None:
            command += ["--resume", str(resume_checkpoint)]

        print("+", " ".join(command), flush=True)
        process = subprocess.Popen(
            command,
            cwd=root,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        run_id = str(resume_manifest.get("run_id")) if resume_manifest else ""
        checkpoint_mtime_ns = 0
        last_synced_step = resume_step
        best_val_loss = (
            float(resume_manifest.get("best_validation_loss", float("inf")))
            if resume_manifest
            else float("inf")
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("event") == "start":
                    run_id = str(payload["run_id"])
                if "step" in payload:
                    step = int(payload["step"])
                    val_loss = float(payload["val_loss"])
                    best_val_loss = min(best_val_loss, val_loss)
                    trackio.log({
                        "train/micro_loss": payload.get("micro_train_loss"),
                        "train/loss": payload.get("train_loss"),
                        "validation/loss": val_loss,
                        "learning_rate": payload.get("lr"),
                        "elapsed_seconds": payload.get("elapsed_s"),
                        "training_step": step,
                    })
                    if (step + 1) % int(config["hub_checkpoint_interval"]) == 0:
                        if not run_id:
                            raise RuntimeError("training emitted a metric before its run id")
                        latest = output_root / run_id / "latest.pt"
                        checkpoint_mtime_ns = wait_for_checkpoint(
                            latest,
                            checkpoint_mtime_ns,
                        )
                        log_path = root / "logs" / f"{run_id}.jsonl"
                        upload_resume_checkpoint(
                            api,
                            model_repo,
                            work,
                            latest,
                            log_path,
                            step=step,
                            best_val_loss=best_val_loss,
                            run_id=run_id,
                            corpus_tokens=corpus_tokens,
                            launch_id=launch_id,
                        )
                        last_synced_step = step
            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
        except Exception as exc:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            upload_state(
                api,
                model_repo,
                work,
                {
                    **state,
                    "status": "error",
                    "run_id": run_id or None,
                    "step": last_synced_step,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:1000],
                },
                "Record interrupted Ember v0.0.8 run",
            )
            raise
        finally:
            trackio.finish()

        if not run_id:
            raise RuntimeError("training completed without reporting a run id")
        run_dir = output_root / run_id
        best = run_dir / "best.pt"
        latest = run_dir / "latest.pt"
        if not best.is_file() or not latest.is_file():
            raise RuntimeError("training completed without best.pt and latest.pt")
        run([sys.executable, "-m", "src.quantize_int4", str(best)], root)
        int4 = best.with_suffix(".int4.pt")
        if not int4.is_file() or int4.stat().st_size == 0:
            raise RuntimeError("INT4 export did not produce a durable checkpoint")

        api.upload_folder(
            repo_id=model_repo,
            repo_type="model",
            folder_path=str(run_dir),
            path_in_repo=f"checkpoints/{run_id}",
            commit_message="Persist completed Ember v0.0.8 checkpoints",
        )
        log_path = root / "logs" / f"{run_id}.jsonl"
        if log_path.is_file():
            api.upload_file(
                repo_id=model_repo,
                repo_type="model",
                path_or_fileobj=str(log_path),
                path_in_repo=f"logs/{log_path.name}",
                commit_message="Persist completed Ember v0.0.8 training log",
            )
        api.upload_file(
            repo_id=model_repo,
            repo_type="model",
            path_or_fileobj=str(config_target),
            path_in_repo=f"config/{config_target.name}",
            commit_message="Persist Ember v0.0.8 training configuration",
        )
        if trackio_dir.exists():
            api.upload_folder(
                repo_id=model_repo,
                repo_type="model",
                folder_path=str(trackio_dir),
                path_in_repo="trackio",
                commit_message="Persist Ember v0.0.8 Trackio metrics",
            )

        remote_files = api.list_repo_files(model_repo, repo_type="model")
        remote_best = f"checkpoints/{run_id}/best.pt"
        remote_int4 = f"checkpoints/{run_id}/best.int4.pt"
        if remote_best not in remote_files or remote_int4 not in remote_files:
            raise RuntimeError("completed v0.0.8 checkpoints were not persisted to the Hub")

        try:
            trackio.sync(project="ember", space_id=trackio_space, force=True, sdk="static")
            print(f"TRACKIO_SPACE={trackio_space}")
        except Exception as exc:
            print(f"TRACKIO_SPACE_SYNC_SKIPPED={type(exc).__name__}: {exc}")

        final_state = {
            **state,
            "status": "training_complete_pending_eval",
            "run_id": run_id,
            "step": 2999,
            "best_validation_loss": best_val_loss,
            "best_checkpoint_path": remote_best,
            "int4_checkpoint_path": remote_int4,
            "best_checkpoint_bytes": best.stat().st_size,
            "int4_checkpoint_bytes": int4.stat().st_size,
            "evaluation_required": True,
        }
        resume_manifest_path = work / "resume-manifest.json"
        if resume_manifest_path.is_file():
            completed_manifest = json.loads(resume_manifest_path.read_text(encoding="utf-8"))
            completed_manifest.update({
                "status": "training_complete_pending_eval",
                "best_checkpoint_path": remote_best,
                "int4_checkpoint_path": remote_int4,
                "updated_at": utc_now().isoformat(),
            })
            write_json(resume_manifest_path, completed_manifest)
            api.upload_file(
                repo_id=model_repo,
                repo_type="model",
                path_or_fileobj=str(resume_manifest_path),
                path_in_repo=RESUME_MANIFEST_PATH,
                commit_message="Complete Ember v0.0.8 resume manifest",
            )
        upload_state(api, model_repo, work, final_state, "Complete Ember v0.0.8 training state")

        print("EMBER_HF_V008_TRAINING=PASS")
        print(f"MODEL_REPO={model_repo}")
        print(f"RUN_ID={run_id}")
        print(f"BEST_VALIDATION_LOSS={best_val_loss}")
        print(f"NEXT_GATE=run eval-v008 on CPU")


if __name__ == "__main__":
    main()
