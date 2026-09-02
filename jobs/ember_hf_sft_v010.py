# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "huggingface-hub>=1.4",
#   "sentencepiece>=0.2",
#   "torch>=2.4",
#   "trackio[spaces]",
# ]
# ///
"""Run Ember v0.0.10's semantic-fidelity supervised fine-tune.

This phase starts from the accepted v0.0.9 checkpoint, which learned tool
routing but not argument or fact fidelity.  It uses completion-only loss over a
deterministic curriculum that copies requested tool arguments, answers from
supplied tool-result facts, and stops at <|endoftext|>.  Official promotion
prompts stay held out.  Promotion uses the stricter v0.0.10 evaluation spec.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

PACKAGE_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/main/ember-v0.0.7-hf-ready.zip"
PACKAGE_SHA256 = "27e8f7c80317652a22b3d58a0bd474724491a685dfe9e20c0b997b7c5907a289"
CONFIG_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/main/config/ember_agent_tool_sft_v0.0.10.json"
CONFIG_SHA256 = "fb2d14567b1f749ff8d3d5e386d89c6c64fb0c79d4d4b10bb93953596b05ba37"
DATA_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/main/jobs/ember_sft_data_v010.py"
DATA_SHA256 = "d507966aa9fcb4011bcfb383930ddb7b371850e480942ae9b7ab7fd8a4b77220"
EVAL_SPEC_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/main/config/ember_v0.0.10_eval.json"
EVAL_SPEC_SHA256 = "545bd49ea901b5ec60e8df017107dbc8f88b54fe5d7f1af929f7da3dc4ecf1e0"
SOURCE_EVAL_SPEC_SHA256 = "e006aa0f7c50797e0466a87fa3f1e35a1f00a63baf1f5113cf6e574844079bd4"
RUN_STATE_PATH = "run-state.json"
RESUME_CHECKPOINT_PATH = "resume/latest.pt"
RESUME_BEST_PATH = "resume/best.pt"
RESUME_MANIFEST_PATH = "resume/manifest.json"
RUN_STATE_STALE_SECONDS = 2 * 60 * 60


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


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def load_remote_json(api: HfApi, repo_id: str, filename: str, token: str, local_dir: Path) -> dict | None:
    files = set(api.list_repo_files(repo_id, repo_type="model"))
    if filename not in files:
        return None
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=filename,
        token=token,
        local_dir=local_dir,
    )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_no_live_duplicate(state: dict | None) -> None:
    if not state:
        return
    status = state.get("status")
    if status in {"training_complete_pending_eval", "evaluation_complete", "complete"}:
        raise RuntimeError("v0.0.10 SFT is already complete; refusing a duplicate paid run")
    if status != "running":
        return
    updated_at = str(state.get("updated_at", ""))
    if not updated_at:
        raise RuntimeError("existing v0.0.10 run lock has no timestamp; manual review required")
    age = (utc_now() - parse_timestamp(updated_at)).total_seconds()
    if age < RUN_STATE_STALE_SECONDS:
        raise RuntimeError(f"another v0.0.10 SFT run appears active (state age {age:.0f}s)")


def upload_state(api: HfApi, repo_id: str, scratch: Path, state: dict, message: str) -> dict:
    state = {**state, "updated_at": utc_now().isoformat()}
    path = write_json(scratch / "run-state.json", state)
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=str(path),
        path_in_repo=RUN_STATE_PATH,
        commit_message=message,
    )
    return state


def load_data_module(path: Path):
    spec = importlib.util.spec_from_file_location("ember_sft_data_v010", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load the pinned Ember SFT data module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cosine_lr(step: int, *, base_lr: float, warmup: int, max_steps: int, min_lr_ratio: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def set_seed(seed: int, torch) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_examples(tokenizer, examples: list[dict], block_size: int, torch) -> list[tuple]:
    fill_ids = tokenizer.encode("<|endoftext|>")
    if not fill_ids:
        raise RuntimeError("tokenizer cannot encode Ember's endoftext marker")
    fill_id = int(fill_ids[-1])
    encoded = []
    for row in examples:
        prompt_ids = tokenizer.encode(row["prompt"])
        full_ids = tokenizer.encode(row["prompt"] + row["completion"])
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(f"tokenization changed across the completion boundary: {row['id']}")
        if len(full_ids) < 2 or len(full_ids) > block_size + 1:
            raise RuntimeError(
                f"SFT sequence {row['id']} has {len(full_ids)} tokens; expected 2..{block_size + 1}"
            )
        inputs = torch.full((block_size,), fill_id, dtype=torch.long)
        targets = torch.full((block_size,), -100, dtype=torch.long)
        sequence_inputs = torch.tensor(full_ids[:-1], dtype=torch.long)
        sequence_targets = torch.tensor(full_ids[1:], dtype=torch.long)
        inputs[: len(sequence_inputs)] = sequence_inputs
        first_completion_prediction = max(0, len(prompt_ids) - 1)
        targets[first_completion_prediction : len(sequence_targets)] = sequence_targets[
            first_completion_prediction:
        ]
        if int((targets != -100).sum().item()) < 2:
            raise RuntimeError(f"SFT sequence has no trainable completion: {row['id']}")
        encoded.append((inputs, targets))
    return encoded


def sample_batch(dataset: list[tuple], batch_size: int, generator, device: str, torch):
    indices = torch.randint(len(dataset), (batch_size,), generator=generator).tolist()
    x = torch.stack([dataset[index][0] for index in indices]).to(device)
    y = torch.stack([dataset[index][1] for index in indices]).to(device)
    return x, y


def all_batches(dataset: list[tuple], batch_size: int, device: str, torch):
    for start in range(0, len(dataset), batch_size):
        rows = dataset[start : start + batch_size]
        yield (
            torch.stack([row[0] for row in rows]).to(device),
            torch.stack([row[1] for row in rows]).to(device),
        )


def evaluate_loss(model, dataset: list[tuple], batch_size: int, device: str, amp_ctx, torch) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for x, y in all_batches(dataset, batch_size, device, torch):
            with amp_ctx():
                _, loss = model(x, y)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("SFT validation produced a non-finite loss")
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def extract_json_object(text: str) -> dict | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return payload if isinstance(payload, dict) else None
    return None


def visible_text(text: str) -> str:
    for marker in (
        "<|system|>", "<|user|>", "<|assistant|>", "<|tool|>",
        "<|tool_result|>", "<|endoftext|>",
    ):
        text = text.replace(marker, " ")
    return " ".join(text.split())


def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int, device: str, torch) -> str:
    prompt_ids = tokenizer.encode(prompt)
    available = int(model.cfg.block_size) - len(prompt_ids)
    if available < 1:
        raise RuntimeError("internal SFT smoke prompt exceeds the checkpoint context")
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        y = model.generate(x, max_new_tokens=min(max_new_tokens, available), temperature=1.0, top_k=1)
    return tokenizer.decode(y[0].tolist()[len(prompt_ids) :])


def arguments_needles(expected_value) -> list[str]:
    values = expected_value if isinstance(expected_value, list) else [expected_value]
    return [str(value) for value in values if str(value)]


def arguments_match(actual, expected: dict | None) -> bool:
    if not expected:
        return isinstance(actual, (dict, str)) and bool(actual)
    if not isinstance(actual, dict) or not actual:
        return False
    blob = " ".join(str(value) for value in actual.values())
    compact = blob.casefold().replace(" ", "")
    for expected_value in expected.values():
        for token in arguments_needles(expected_value):
            if token not in blob and token.casefold().replace(" ", "") not in compact:
                return False
    return True


def facts_match(completion: str, facts) -> bool:
    if not facts:
        return True
    readable = visible_text(completion)
    compact = readable.casefold().replace(" ", "")
    for fact in facts:
        token = str(fact)
        if token not in readable and token.casefold().replace(" ", "") not in compact:
            return False
    return True


def clean_stop(completion: str) -> bool:
    if "<|endoftext|>" not in completion:
        return "<|user|>" not in completion and "<|system|>" not in completion
    leftover = visible_text(completion.split("<|endoftext|>", 1)[1])
    after = completion.split("<|endoftext|>", 1)[1]
    return leftover == "" and "<|user|>" not in after and "<|system|>" not in after


def internal_smoke(model, tokenizer, examples: list[dict], per_kind: int, device: str, torch) -> dict:
    selected = []
    for kind in ("tool_call", "direct_response", "tool_result_response"):
        selected.extend([row for row in examples if row["kind"] == kind][:per_kind])
    results = []
    torch.manual_seed(20260902)
    for row in selected:
        completion = generate_completion(model, tokenizer, row["prompt"], 64, device, torch)
        stopped = clean_stop(completion)
        if row["kind"] == "tool_call":
            marker = "<|tool|>" in completion
            payload = extract_json_object(completion.split("<|tool|>", 1)[1] if marker else completion)
            passed = (
                marker
                and bool(payload)
                and payload.get("name") == row["expected_tool"]
                and arguments_match(payload.get("arguments"), row.get("expected_arguments"))
                and stopped
            )
        else:
            passed = (
                "<|tool|>" not in completion
                and len(visible_text(completion)) >= 3
                and facts_match(completion, row.get("required_facts"))
                and stopped
            )
        results.append({
            "id": row["id"],
            "kind": row["kind"],
            "passed": bool(passed),
            "clean_stop": stopped,
            "completion": completion,
        })
    rates = {}
    for kind in ("tool_call", "direct_response", "tool_result_response"):
        rows = [row for row in results if row["kind"] == kind]
        rates[kind] = sum(row["passed"] for row in rows) / len(rows)
    rates["clean_stop"] = sum(row["clean_stop"] for row in results) / len(results)
    return {"rates": rates, "cases": results}


def resolve_resume(
    api: HfApi,
    repo_id: str,
    token: str,
    local_dir: Path,
    *,
    data_sha256: str,
    source_checkpoint_sha256: str,
):
    files = set(api.list_repo_files(repo_id, repo_type="model"))
    required = {RESUME_CHECKPOINT_PATH, RESUME_BEST_PATH, RESUME_MANIFEST_PATH}
    present = required.intersection(files)
    if present and present != required:
        raise RuntimeError("partial v0.0.10 resume state found; manual review required")
    if not present:
        return None, None, None
    manifest_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=RESUME_MANIFEST_PATH,
        token=token,
        local_dir=local_dir,
    )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    expected = {
        "config_sha256": CONFIG_SHA256,
        "data_sha256": data_sha256,
        "source_checkpoint_sha256": source_checkpoint_sha256,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"resume manifest {key} does not match this v0.0.10 run")
    latest = Path(hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=RESUME_CHECKPOINT_PATH,
        token=token,
        local_dir=local_dir,
    ))
    best = Path(hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=RESUME_BEST_PATH,
        token=token,
        local_dir=local_dir,
    ))
    if sha256_file(latest) != manifest.get("latest_sha256"):
        raise RuntimeError("resume latest checkpoint checksum mismatch")
    if sha256_file(best) != manifest.get("best_sha256"):
        raise RuntimeError("resume best checkpoint checksum mismatch")
    return latest, best, manifest


def upload_resume(
    api: HfApi,
    repo_id: str,
    scratch: Path,
    latest: Path,
    best: Path,
    metrics_path: Path,
    *,
    step: int,
    run_id: str,
    best_val_loss: float,
    data_sha256: str,
    source_checkpoint_sha256: str,
) -> dict:
    staged_latest = scratch / "resume-latest.pt"
    staged_best = scratch / "resume-best.pt"
    shutil.copy2(latest, staged_latest)
    shutil.copy2(best, staged_best)
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=str(staged_latest),
        path_in_repo=RESUME_CHECKPOINT_PATH,
        commit_message=f"Persist Ember v0.0.10 SFT resume checkpoint at step {step}",
    )
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=str(staged_best),
        path_in_repo=RESUME_BEST_PATH,
        commit_message=f"Persist Ember v0.0.10 SFT best checkpoint at step {step}",
    )
    if metrics_path.is_file():
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=str(metrics_path),
            path_in_repo="resume/train.jsonl",
            commit_message=f"Persist Ember v0.0.10 SFT metrics at step {step}",
        )
    manifest = {
        "schema_version": 1,
        "status": "running",
        "version": "0.0.10",
        "step": step,
        "run_id": run_id,
        "best_validation_loss": best_val_loss,
        "max_steps": 600,
        "config_sha256": CONFIG_SHA256,
        "data_sha256": data_sha256,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "latest_sha256": sha256_file(staged_latest),
        "best_sha256": sha256_file(staged_best),
        "updated_at": utc_now().isoformat(),
    }
    manifest_path = write_json(scratch / "resume-manifest.json", manifest)
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=str(manifest_path),
        path_in_repo=RESUME_MANIFEST_PATH,
        commit_message=f"Update Ember v0.0.10 SFT resume manifest at step {step}",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or preflight Ember v0.0.10 semantic-fidelity SFT")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN was not injected as a Job secret")

    import torch
    import trackio

    if not args.preflight_only and not torch.cuda.is_available():
        raise RuntimeError("Ember v0.0.10 SFT requires the explicitly approved T4 GPU")

    api = HfApi(token=token)
    owner = api.whoami()["name"]
    source_repo = f"{owner}/ember-v0.0.9-t4"
    target_repo = f"{owner}/ember-v0.0.10-t4"
    trackio_space = f"{owner}/ember-trackio"

    with tempfile.TemporaryDirectory(prefix="ember-v010-sft-") as temporary:
        work = Path(temporary)

        archive = download_verified(PACKAGE_URL, work / "ember.zip", PACKAGE_SHA256)
        config_path = download_verified(CONFIG_URL, work / "sft-config.json", CONFIG_SHA256)
        data_path = download_verified(DATA_URL, work / "sft-data.py", DATA_SHA256)
        eval_spec_path = download_verified(EVAL_SPEC_URL, work / "eval-spec.json", EVAL_SPEC_SHA256)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        eval_spec = json.loads(eval_spec_path.read_text(encoding="utf-8"))
        if config.get("version") != "0.0.10" or config.get("phase") != "semantic-fidelity-sft":
            raise RuntimeError("unexpected Ember v0.0.10 SFT configuration")
        if config.get("source_model_name") != "ember-v0.0.9-t4":
            raise RuntimeError("v0.0.10 SFT must initialize from the accepted v0.0.9 checkpoint")
        if not bool(config.get("completion_only_loss")):
            raise RuntimeError("v0.0.10 SFT requires completion-only loss")

        source_evaluation_path = hf_hub_download(
            repo_id=source_repo,
            repo_type="model",
            filename="evaluations/latest.json",
            token=token,
            local_dir=work / "source-evaluation",
        )
        source_evaluation = json.loads(Path(source_evaluation_path).read_text(encoding="utf-8"))
        if source_evaluation.get("label") != config["source_evaluation_label"]:
            raise RuntimeError("source checkpoint was not evaluated under the expected v0.0.9 label")
        if source_evaluation.get("evaluation_spec_sha256") != SOURCE_EVAL_SPEC_SHA256:
            raise RuntimeError("source v0.0.9 was not evaluated with the structural v0.0.8 specification")
        if not bool(source_evaluation.get("technical_pass")):
            raise RuntimeError("source v0.0.9 checkpoint did not pass the technical gate")
        if not bool(source_evaluation.get("promotion", {}).get("promotion_eligible")):
            raise RuntimeError("v0.0.10 requires the accepted v0.0.9 promotion-eligible checkpoint")
        source_checkpoint_repo_path = source_evaluation["checkpoint"]["best_path"]
        source_checkpoint = Path(hf_hub_download(
            repo_id=source_repo,
            repo_type="model",
            filename=source_checkpoint_repo_path,
            token=token,
            local_dir=work / "source-model",
        ))
        source_checkpoint_sha256 = sha256_file(source_checkpoint)

        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        root = work / "src" / "ember"
        sys.path.insert(0, str(root))
        from src.checkpoint import load_checkpoint, save_checkpoint
        from src.model import EmberGPT, ModelConfig
        from src.quantize_int4 import export_int4_checkpoint
        from src.tokenizer import tokenizer_from_state_dict

        data_module = load_data_module(data_path)
        train_examples = data_module.build_sft_examples(
            "train", int(config["train_examples"]), int(config["seed"])
        )
        validation_examples = data_module.build_sft_examples(
            "validation", int(config["validation_examples"]), int(config["seed"])
        )
        data_module.assert_held_out_clean(
            train_examples + validation_examples,
            [case["prompt"] for case in eval_spec["cases"]],
        )
        train_bytes = data_module.jsonl_bytes(train_examples)
        validation_bytes = data_module.jsonl_bytes(validation_examples)
        train_data_path = work / "data" / "train.jsonl"
        validation_data_path = work / "data" / "validation.jsonl"
        train_data_path.parent.mkdir(parents=True, exist_ok=True)
        train_data_path.write_bytes(train_bytes)
        validation_data_path.write_bytes(validation_bytes)
        data_sha256 = hashlib.sha256(train_bytes + validation_bytes).hexdigest()
        dataset_manifest = {
            "schema_version": 1,
            "generator_sha256": DATA_SHA256,
            "evaluation_spec_sha256": EVAL_SPEC_SHA256,
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
            "train_sha256": hashlib.sha256(train_bytes).hexdigest(),
            "validation_sha256": hashlib.sha256(validation_bytes).hexdigest(),
            "combined_sha256": data_sha256,
            "official_cases_used_for_training": 0,
            "kinds": {
                kind: sum(row["kind"] == kind for row in train_examples)
                for kind in data_module.KINDS
            },
        }
        dataset_manifest_path = write_json(work / "data" / "manifest.json", dataset_manifest)

        source = load_checkpoint(source_checkpoint, device="cpu")
        tokenizer = tokenizer_from_state_dict(source["tokenizer"])
        block_size = int(config["block_size"])
        if block_size > int(source["model_config"]["block_size"]):
            raise RuntimeError("SFT block size exceeds the source checkpoint context")
        train_dataset = encode_examples(tokenizer, train_examples, block_size, torch)
        validation_dataset = encode_examples(tokenizer, validation_examples, block_size, torch)

        if args.preflight_only:
            model_config = ModelConfig(**source["model_config"])
            model = EmberGPT(model_config)
            model.load_state_dict(source["model_state"])
            model.eval()
            x = torch.stack([row[0] for row in validation_dataset[:2]])
            y = torch.stack([row[1] for row in validation_dataset[:2]])
            with torch.inference_mode():
                _, loss = model(x, y)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("CPU preflight produced a non-finite completion loss")
            print(json.dumps({
                "event": "v0.0.10_sft_preflight",
                "source_model_repo": source_repo,
                "source_checkpoint_path": source_checkpoint_repo_path,
                "source_checkpoint_sha256": source_checkpoint_sha256,
                "train_examples": len(train_examples),
                "validation_examples": len(validation_examples),
                "data_sha256": data_sha256,
                "completion_only_loss": float(loss.item()),
                "official_cases_used_for_training": 0,
            }, sort_keys=True), flush=True)
            print("EMBER_HF_V010_SFT_PREFLIGHT=PASS", flush=True)
            return

        api.create_repo(target_repo, repo_type="model", private=True, exist_ok=True)
        prior_state = load_remote_json(api, target_repo, RUN_STATE_PATH, token, work / "prior-state")
        assert_no_live_duplicate(prior_state)

        resume_checkpoint, resume_best, _resume_manifest = resolve_resume(
            api,
            target_repo,
            token,
            work / "resume",
            data_sha256=data_sha256,
            source_checkpoint_sha256=source_checkpoint_sha256,
        )

        device = "cuda"
        dtype = torch.float16
        amp_ctx = lambda: torch.autocast(device_type="cuda", dtype=dtype)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        set_seed(int(config["seed"]), torch)

        checkpoint_to_load = load_checkpoint(resume_checkpoint, device="cpu") if resume_checkpoint else source
        model_config = ModelConfig(**checkpoint_to_load["model_config"])
        model = EmberGPT(model_config)
        model.load_state_dict(checkpoint_to_load["model_state"])
        model.to(device)
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            betas=tuple(config["betas"]),
            weight_decay=float(config["weight_decay"]),
        )
        if resume_checkpoint and checkpoint_to_load.get("optimizer_state"):
            optimizer.load_state_dict(checkpoint_to_load["optimizer_state"])

        run_id = (
            str(checkpoint_to_load["run_id"])
            if resume_checkpoint
            else f"{config['run_name']}-{utc_now().strftime('%Y%m%dT%H%M%SZ')}"
        )
        start_step = int(checkpoint_to_load["step"]) + 1 if resume_checkpoint else 0
        best_val_loss = (
            float(checkpoint_to_load.get("best_val_loss", float("inf")))
            if resume_checkpoint
            else float("inf")
        )
        run_dir = root / "checkpoints" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        best_path = run_dir / "best.pt"
        latest_path = run_dir / "latest.pt"
        if resume_best:
            shutil.copy2(resume_best, best_path)
        metrics_path = root / "logs" / f"{run_id}.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_config = {
            **config,
            "source_model_repo": source_repo,
            "source_checkpoint_path": source_checkpoint_repo_path,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "dataset_sha256": data_sha256,
            "dataset_generator_sha256": DATA_SHA256,
            "evaluation_spec_sha256": EVAL_SPEC_SHA256,
        }

        if not resume_checkpoint:
            best_val_loss = evaluate_loss(
                model, validation_dataset, int(config["batch_size"]), device, amp_ctx, torch
            )
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                tokenizer=tokenizer,
                model_config=model_config,
                train_config=runtime_config,
                step=-1,
                best_val_loss=best_val_loss,
                run_id=run_id,
            )

        trackio_dir = work / "trackio"
        os.environ["TRACKIO_DIR"] = str(trackio_dir)
        trackio.init(
            project="ember",
            name="ember-v0.0.10-semantic-sft",
            embed=False,
            config={
                "version": "0.0.10",
                "phase": "semantic-fidelity-sft",
                "hardware": "t4-small",
                "source_checkpoint": source_checkpoint_repo_path,
                "train_examples": len(train_examples),
                "validation_examples": len(validation_examples),
                "max_steps": int(config["max_steps"]),
                "completion_only_loss": True,
                "resumed": bool(resume_checkpoint),
            },
        )

        state = upload_state(
            api,
            target_repo,
            work,
            {
                "schema_version": 1,
                "status": "running",
                "version": "0.0.10",
                "phase": "semantic-fidelity-sft",
                "run_id": run_id,
                "step": start_step - 1,
                "max_steps": int(config["max_steps"]),
                "source_model_repo": source_repo,
                "source_checkpoint_path": source_checkpoint_repo_path,
                "source_checkpoint_sha256": source_checkpoint_sha256,
                "config_sha256": CONFIG_SHA256,
                "data_sha256": data_sha256,
                "resumed": bool(resume_checkpoint),
            },
                "Start Ember v0.0.10 semantic-fidelity SFT",
        )

        batch_size = int(config["batch_size"])
        grad_accum = int(config["gradient_accumulation_steps"])
        max_steps = int(config["max_steps"])
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        batch_generator = torch.Generator(device="cpu")
        batch_generator.manual_seed(int(config["seed"]) + start_step)
        started = time.time()

        try:
            for step in range(start_step, max_steps):
                lr = cosine_lr(
                    step,
                    base_lr=float(config["learning_rate"]),
                    warmup=int(config["warmup_steps"]),
                    max_steps=max_steps,
                    min_lr_ratio=float(config["min_lr_ratio"]),
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                micro_loss = 0.0
                for _ in range(grad_accum):
                    x, y = sample_batch(train_dataset, batch_size, batch_generator, device, torch)
                    with amp_ctx():
                        _, loss = model(x, y)
                        scaled_loss = loss / grad_accum
                    if not bool(torch.isfinite(loss).item()):
                        raise RuntimeError(f"non-finite SFT loss at step {step}")
                    scaler.scale(scaled_loss).backward()
                    micro_loss += float(loss.item()) / grad_accum
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()

                should_eval = (
                    step == start_step
                    or (step + 1) % int(config["eval_interval"]) == 0
                    or step == max_steps - 1
                )
                if should_eval:
                    val_loss = evaluate_loss(model, validation_dataset, batch_size, device, amp_ctx, torch)
                    row = {
                        "step": step,
                        "train_completion_loss": micro_loss,
                        "validation_completion_loss": val_loss,
                        "learning_rate": lr,
                        "elapsed_seconds": round(time.time() - started, 2),
                    }
                    print(json.dumps(row), flush=True)
                    with metrics_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row) + "\n")
                    trackio.log({
                        "sft/train_completion_loss": micro_loss,
                        "sft/validation_completion_loss": val_loss,
                        "learning_rate": lr,
                        "training_step": step,
                    })
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            best_path,
                            model=model,
                            optimizer=optimizer,
                            tokenizer=tokenizer,
                            model_config=model_config,
                            train_config=runtime_config,
                            step=step,
                            best_val_loss=best_val_loss,
                            run_id=run_id,
                        )

                should_save = (
                    (step + 1) % int(config["save_interval"]) == 0
                    or step == max_steps - 1
                )
                if should_save:
                    save_checkpoint(
                        latest_path,
                        model=model,
                        optimizer=optimizer,
                        tokenizer=tokenizer,
                        model_config=model_config,
                        train_config=runtime_config,
                        step=step,
                        best_val_loss=best_val_loss,
                        run_id=run_id,
                    )

                if (
                    (step + 1) % int(config["hub_checkpoint_interval"]) == 0
                    or step == max_steps - 1
                ):
                    manifest = upload_resume(
                        api,
                        target_repo,
                        work,
                        latest_path,
                        best_path,
                        metrics_path,
                        step=step,
                        run_id=run_id,
                        best_val_loss=best_val_loss,
                        data_sha256=data_sha256,
                        source_checkpoint_sha256=source_checkpoint_sha256,
                    )
                    state = upload_state(
                        api,
                        target_repo,
                        work,
                        {**state, **manifest, "phase": "semantic-fidelity-sft"},
                        f"Update Ember v0.0.10 SFT state at step {step}",
                    )
        except Exception as exc:
            upload_state(
                api,
                target_repo,
                work,
                {
                    **state,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:1000],
                },
                "Record interrupted Ember v0.0.10 SFT",
            )
            raise
        finally:
            trackio.finish()

        best_checkpoint = load_checkpoint(best_path, device="cpu")
        model.load_state_dict(best_checkpoint["model_state"])
        model.to(device)
        model.eval()
        smoke = internal_smoke(
            model,
            tokenizer,
            validation_examples,
            int(config["internal_smoke_cases_per_kind"]),
            device,
            torch,
        )
        smoke_rates = smoke["rates"]
        smoke_pass = (
            smoke_rates["tool_call"] >= float(config["internal_minimum_valid_tool_call_rate"])
            and smoke_rates["direct_response"] >= float(config["internal_minimum_direct_response_rate"])
            and smoke_rates["tool_result_response"]
            >= float(config["internal_minimum_tool_result_response_rate"])
            and smoke_rates["clean_stop"] >= float(config["internal_minimum_clean_stop_rate"])
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

        int4_path = export_int4_checkpoint(best_path)
        if not int4_path.is_file() or int4_path.stat().st_size == 0:
            raise RuntimeError("v0.0.10 INT4 export did not produce a durable checkpoint")
        summary = {
            "schema_version": 1,
            "created_at": utc_now().isoformat(),
            "version": "0.0.10",
            "phase": "semantic-fidelity-sft",
            "run_id": run_id,
            "source_model_repo": source_repo,
            "source_checkpoint_path": source_checkpoint_repo_path,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "best_validation_completion_loss": best_val_loss,
            "best_step": int(best_checkpoint["step"]),
            "dataset": dataset_manifest,
            "internal_smoke": smoke,
            "internal_smoke_pass": smoke_pass,
            "next_gate": "run candidate-v0.0.10 with the semantic v0.0.10 CPU promotion evaluation",
        }
        summary_path = write_json(work / "training-summary.json", summary)

        api.upload_folder(
            repo_id=target_repo,
            repo_type="model",
            folder_path=str(run_dir),
            path_in_repo=f"checkpoints/{run_id}",
            commit_message="Persist completed Ember v0.0.10 SFT checkpoints",
        )
        for local_path, remote_path, message in (
            (config_path, "config/ember_agent_tool_sft_v0.0.10.json", "Persist Ember v0.0.10 SFT config"),
            (data_path, "data/ember_sft_data_v010.py", "Persist Ember v0.0.10 data generator"),
            (train_data_path, "data/train.jsonl", "Persist Ember v0.0.10 SFT training data"),
            (validation_data_path, "data/validation.jsonl", "Persist Ember v0.0.10 SFT validation data"),
            (dataset_manifest_path, "data/manifest.json", "Persist Ember v0.0.10 SFT data manifest"),
            (summary_path, "training/summary.json", "Persist Ember v0.0.10 SFT summary"),
        ):
            api.upload_file(
                repo_id=target_repo,
                repo_type="model",
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                commit_message=message,
            )
        if metrics_path.is_file():
            api.upload_file(
                repo_id=target_repo,
                repo_type="model",
                path_or_fileobj=str(metrics_path),
                path_in_repo=f"logs/{metrics_path.name}",
                commit_message="Persist Ember v0.0.10 SFT metrics",
            )
        if trackio_dir.exists():
            api.upload_folder(
                repo_id=target_repo,
                repo_type="model",
                folder_path=str(trackio_dir),
                path_in_repo="trackio",
                commit_message="Persist Ember v0.0.10 Trackio metrics",
            )

        remote_files = set(api.list_repo_files(target_repo, repo_type="model"))
        remote_best = f"checkpoints/{run_id}/best.pt"
        remote_int4 = f"checkpoints/{run_id}/best.int4.pt"
        if remote_best not in remote_files or remote_int4 not in remote_files:
            raise RuntimeError("completed v0.0.10 checkpoints were not persisted to the Hub")

        upload_state(
            api,
            target_repo,
            work,
            {
                **state,
                "status": "training_complete_pending_eval",
                "step": max_steps - 1,
                "best_step": int(best_checkpoint["step"]),
                "best_validation_completion_loss": best_val_loss,
                "best_checkpoint_path": remote_best,
                "int4_checkpoint_path": remote_int4,
                "internal_smoke_pass": smoke_pass,
                "training_summary_path": "training/summary.json",
            },
            "Complete Ember v0.0.10 semantic-fidelity SFT",
        )

        try:
            trackio.sync(project="ember", space_id=trackio_space, force=True, sdk="static")
        except Exception as exc:  # noqa: BLE001 - metrics sync must not invalidate durable checkpoints
            print(f"TRACKIO_SYNC_WARNING={type(exc).__name__}: {exc}", flush=True)

        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        print("EMBER_HF_V010_SFT=PASS", flush=True)
        print(f"MODEL_REPO={target_repo}", flush=True)
        print(f"RUN_ID={run_id}", flush=True)
        print(f"BEST_VALIDATION_COMPLETION_LOSS={best_val_loss}", flush=True)
        print(f"INTERNAL_SMOKE_PASS={str(smoke_pass).lower()}", flush=True)
        print("NEXT_GATE=run eval-v010 on CPU", flush=True)


if __name__ == "__main__":
    main()
