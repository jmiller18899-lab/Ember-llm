# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "huggingface-hub>=1.4",
#   "sentencepiece>=0.2",
#   "torch>=2.4",
# ]
# ///
"""Run Ember v0.0.14's literal unique-copy canary from the accepted v0.0.9 checkpoint.

Do not initialize from the failed v0.0.11, v0.0.12, or v0.0.13 runs. Those
copied tool JSON shape but invented nearby identifiers. This job requires every
focus term to appear in the prompt, trains for multiple epochs, and uses
EOS-aware greedy decoding for its internal smoke gate.
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
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

PACKAGE_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/main/ember-v0.0.7-hf-ready.zip"
PACKAGE_SHA256 = "27e8f7c80317652a22b3d58a0bd474724491a685dfe9e20c0b997b7c5907a289"
ASSET_PIN = "0000000000000000000000000000000000000000"
CONFIG_URL = f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{ASSET_PIN}/config/ember_agent_copy_canary_v0.0.14.json"
CONFIG_SHA256 = "fba5a2e9a58b0eb204638bfc9099581f5867a255f829eb44b55df90640252621"
DATA_URL = f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{ASSET_PIN}/jobs/ember_sft_data_v014.py"
DATA_SHA256 = "01b61b79287f799545c38220592e0cadc6ea013585ac61fb5b9ea0e2dae2a3b8"
SOURCE_REPO = "Jmiller18899/ember-v0.0.9-t4"
SOURCE_CHECKPOINT = "checkpoints/ember-agent-v0.0.9-tool-sft-20260828T142047Z/best.pt"
SOURCE_SHA256 = "8299d52e8a852b9bd3e8403e086b48fd42d2babfd51c74eb94af29bd87ef2d13"
RUN_STATE_STALE_SECONDS = 2 * 60 * 60


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, destination: Path, expected: str | None = None) -> Path:
    urllib.request.urlretrieve(url, destination)
    actual = sha(destination)
    if expected and actual != expected:
        raise RuntimeError(f"download checksum mismatch for {url}: {actual}")
    return destination


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("ember_sft_data_v014", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load the v0.0.14 data module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cosine_lr(step: int, base_lr: float, warmup: int, max_steps: int, min_lr_ratio: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def encode_row(tokenizer, row: dict, block_size: int, semantic_weight: float, eos_weight: float, torch):
    prompt_ids = list(tokenizer.encode(row["prompt"]))
    full_ids = list(tokenizer.encode(row["prompt"] + row["completion"]))
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(f"tokenization changed at completion boundary: {row['id']}")
    if len(full_ids) < 2 or len(full_ids) > block_size + 1:
        raise RuntimeError(f"sequence length out of range: {row['id']}={len(full_ids)}")

    eot = list(tokenizer.encode("<|endoftext|>"))
    eot_id = int(eot[-1])
    x = torch.full((block_size,), eot_id, dtype=torch.long)
    y = torch.full((block_size,), -100, dtype=torch.long)
    w = torch.zeros((block_size,), dtype=torch.float32)
    seq_x = torch.tensor(full_ids[:-1], dtype=torch.long)
    seq_y = torch.tensor(full_ids[1:], dtype=torch.long)
    x[: len(seq_x)] = seq_x
    first_target = max(0, len(prompt_ids) - 1)
    y[first_target : len(seq_y)] = seq_y[first_target:]
    w[first_target : len(seq_y)] = 1.0

    completion = row["completion"]
    semantic_positions = set()
    for term in row["focus_terms"]:
        term_text = str(term)
        char_start = completion.find(term_text)
        if char_start < 0:
            raise RuntimeError(f"semantic text absent from target: {row['id']} {term_text}")
        char_end = char_start + len(term_text)
        before_ids = list(tokenizer.encode(row["prompt"] + completion[:char_start]))
        through_ids = list(tokenizer.encode(row["prompt"] + completion[:char_end]))
        token_start = max(len(prompt_ids), len(before_ids) - 1)
        token_end = min(len(full_ids), max(token_start + 1, len(through_ids) + 1))
        for token_pos in range(token_start, token_end):
            target_pos = token_pos - 1
            if first_target <= target_pos < len(seq_y):
                semantic_positions.add(target_pos)
                w[target_pos] = max(float(w[target_pos]), float(semantic_weight))
    if not semantic_positions:
        raise RuntimeError(f"semantic token span not found: {row['id']}")

    eos_positions = 0
    for token_pos in range(len(prompt_ids), len(full_ids)):
        if int(full_ids[token_pos]) == eot_id:
            target_pos = token_pos - 1
            if first_target <= target_pos < len(seq_y):
                w[target_pos] = max(float(w[target_pos]), float(eos_weight))
                eos_positions += 1
    if eos_positions < 1:
        raise RuntimeError(f"no trainable EOS target: {row['id']}")
    return {"x": x, "y": y, "w": w, "row": row, "semantic_positions": len(semantic_positions)}


def encode_dataset(tokenizer, rows, config, torch):
    encoded = [
        encode_row(
            tokenizer,
            row,
            int(config["block_size"]),
            float(config["semantic_token_weight"]),
            float(config["eos_token_weight"]),
            torch,
        )
        for row in rows
    ]
    if not encoded or min(item["semantic_positions"] for item in encoded) < 1:
        raise RuntimeError("semantic weighting produced an empty focus mask")
    return encoded


def batch(dataset, indices, device, torch):
    return (
        torch.stack([dataset[i]["x"] for i in indices]).to(device),
        torch.stack([dataset[i]["y"] for i in indices]).to(device),
        torch.stack([dataset[i]["w"] for i in indices]).to(device),
    )


def weighted_loss(model, x, y, weights, torch):
    import torch.nn.functional as F

    logits, _ = model(x, None)
    raw = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(y)
    active = y.ne(-100)
    denom = (weights * active).sum().clamp_min(1.0)
    return (raw * weights * active).sum() / denom


def evaluate(model, dataset, batch_size, device, amp_ctx, torch, limit=120):
    model.eval()
    losses = []
    subset = list(range(min(len(dataset), limit)))
    with torch.inference_mode():
        for start in range(0, len(subset), batch_size):
            x, y, w = batch(dataset, subset[start : start + batch_size], device, torch)
            with amp_ctx():
                loss = weighted_loss(model, x, y, w, torch)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("non-finite validation loss")
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def extract_json(text):
    start = text.find("{")
    if start < 0:
        return None
    depth, quoted, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if quoted:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quoted = False
            continue
        if ch == '"':
            quoted = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def eos_greedy(model, tokenizer, prompt, device, torch, max_new=80):
    prompt_ids = list(tokenizer.encode(prompt))
    eot_id = int(tokenizer.encode("<|endoftext|>")[-1])
    seq = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = []
    stopped = False
    with torch.inference_mode():
        for _ in range(max_new):
            x = seq[:, -int(model.cfg.block_size) :]
            logits, _ = model(x, None)
            nxt = int(torch.argmax(logits[0, -1]).item())
            generated.append(nxt)
            seq = torch.cat([seq, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
            if nxt == eot_id:
                stopped = True
                break
    return tokenizer.decode(generated), stopped


def internal_smoke(model, tokenizer, rows, config, device, torch):
    selected = []
    n = int(config["internal_smoke_cases_per_kind"])
    for kind in ("tool_call", "direct_response", "tool_result_response"):
        selected += [row for row in rows if row["kind"] == kind][:n]
    results = []
    for row in selected:
        completion, stopped = eos_greedy(model, tokenizer, row["prompt"], device, torch)
        focus = all(str(term).casefold() in completion.casefold() for term in row["focus_terms"])
        if row["kind"] == "tool_call":
            payload = extract_json(completion)
            tool_ok = (
                "<|tool|>" in completion
                and isinstance(payload, dict)
                and payload.get("name") == row["expected_tool"]
            )
            passed = bool(tool_ok and focus and stopped)
        else:
            passed = bool("<|tool|>" not in completion and focus and stopped)
        results.append(
            {
                "id": row["id"],
                "kind": row["kind"],
                "passed": passed,
                "stopped": stopped,
                "focus_ok": focus,
                "completion": completion,
            }
        )
    rates = {}
    for kind in ("tool_call", "direct_response", "tool_result_response"):
        group = [row for row in results if row["kind"] == kind]
        rates[kind] = sum(row["passed"] for row in group) / len(group)
    rates["clean_stop"] = sum(row["stopped"] for row in results) / len(results)
    passed = (
        rates["tool_call"] >= float(config["internal_minimum_valid_tool_call_rate"])
        and rates["direct_response"] >= float(config["internal_minimum_direct_response_rate"])
        and rates["tool_result_response"] >= float(config["internal_minimum_tool_result_response_rate"])
        and rates["clean_stop"] >= float(config["internal_minimum_clean_stop_rate"])
    )
    return {"passed": passed, "rates": rates, "cases": results}


def upload(api, repo, local, remote, message):
    api.upload_file(
        repo_id=repo,
        repo_type="model",
        path_or_fileobj=str(local),
        path_in_repo=remote,
        commit_message=message,
    )


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def assert_no_live_duplicate(state: dict | None) -> None:
    if not state:
        return
    status = state.get("status")
    if status in {"training_complete_pending_eval", "evaluation_complete", "complete"}:
        raise RuntimeError("v0.0.14 canary is already complete; refusing a duplicate paid run")
    if status != "running":
        return
    updated_at = str(state.get("updated_at") or state.get("started_at") or "")
    if not updated_at:
        raise RuntimeError("existing v0.0.14 run lock has no timestamp; manual review required")
    age = (utc_now() - parse_timestamp(updated_at)).total_seconds()
    if age < RUN_STATE_STALE_SECONDS:
        raise RuntimeError(f"another v0.0.14 canary appears active (state age {age:.0f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or preflight Ember v0.0.14 literal-copy canary")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN was not injected as a Job secret")
    if ASSET_PIN == "CURSOR_PIN_PLACEHOLDER" or set(ASSET_PIN) == {"0"}:
        raise RuntimeError("v0.0.14 asset pin is unset; refuse to launch an unpinned job")

    import torch

    api = HfApi(token=token)
    with tempfile.TemporaryDirectory(prefix="ember-v014-") as temporary:
        work = Path(temporary)
        config_path = fetch(CONFIG_URL, work / "config.json", CONFIG_SHA256)
        data_path = fetch(DATA_URL, work / "data.py", DATA_SHA256)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("version") != "0.0.14" or config.get("phase") != "literal-copy-canary":
            raise RuntimeError("unexpected Ember v0.0.14 configuration")
        if config.get("source_model_name") != "ember-v0.0.9-t4":
            raise RuntimeError("v0.0.14 must initialize from the accepted v0.0.9 checkpoint")
        if float(config["semantic_token_weight"]) > 6.0:
            raise RuntimeError("v0.0.14 refuses the 8x semantic weight that collapsed v0.0.12")
        if int(config["max_steps"]) < 400:
            raise RuntimeError("v0.0.14 refuses another one-epoch canary")
        seen = int(config["max_steps"]) * int(config["batch_size"]) * int(config["gradient_accumulation_steps"])
        if seen < 3 * int(config["train_examples"]):
            raise RuntimeError("v0.0.14 requires at least three passes over the training set")

        data = load_module(data_path)
        train_rows = data.build_examples("train", int(config["train_examples"]))
        val_rows = data.build_examples("validation", int(config["validation_examples"]))
        data.assert_held_out_clean(train_rows + val_rows)
        for row in train_rows + val_rows:
            for term in row["focus_terms"]:
                if str(term) not in row["prompt"]:
                    raise RuntimeError(f"v0.0.14 focus term missing from prompt: {row['id']} {term}")
                if row["prompt"].count(str(term)) < 2:
                    raise RuntimeError(f"v0.0.14 focus term is not repeated in prompt: {row['id']} {term}")

        archive = fetch(PACKAGE_URL, work / "ember.zip", PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        root = work / "src" / "ember"
        sys.path.insert(0, str(root))
        from src.checkpoint import load_checkpoint, save_checkpoint
        from src.model import EmberGPT, ModelConfig
        from src.quantize_int4 import export_int4_checkpoint
        from src.tokenizer import tokenizer_from_state_dict

        source_path = Path(
            hf_hub_download(
                repo_id=SOURCE_REPO,
                repo_type="model",
                filename=SOURCE_CHECKPOINT,
                token=token,
                local_dir=work / "source",
            )
        )
        if sha(source_path) != SOURCE_SHA256:
            raise RuntimeError("source checkpoint checksum mismatch")
        source = load_checkpoint(source_path, device="cpu")
        tokenizer = tokenizer_from_state_dict(source["tokenizer"])
        train = encode_dataset(tokenizer, train_rows, config, torch)
        val = encode_dataset(tokenizer, val_rows, config, torch)

        model = EmberGPT(ModelConfig(**source["model_config"]))
        model.load_state_dict(source["model_state"])
        model.eval()
        x, y, w = batch(val, list(range(min(4, len(val)))), "cpu", torch)
        with torch.inference_mode():
            probe = weighted_loss(model, x, y, w, torch)
        if not bool(torch.isfinite(probe).item()):
            raise RuntimeError("weighted-loss preflight is non-finite")
        preflight = {
            "status": "PASS",
            "version": "0.0.14",
            "weighted_probe_loss": float(probe.item()),
            "train_examples": len(train),
            "validation_examples": len(val),
            "min_semantic_positions": min(item["semantic_positions"] for item in train + val),
            "source_checkpoint_sha256": SOURCE_SHA256,
            "source_model": "ember-v0.0.9-t4",
            "refuses_failed_v012_init": True,
            "refuses_failed_v013_init": True,
        }
        print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
        if args.preflight_only:
            print("EMBER_HF_V014_SFT_PREFLIGHT=PASS", flush=True)
            return

        if not torch.cuda.is_available():
            raise RuntimeError("Ember v0.0.14 canary requires the explicitly approved T4 GPU")

        device = "cuda"
        amp_ctx = lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        random.seed(int(config["seed"]))
        torch.manual_seed(int(config["seed"]))
        torch.cuda.manual_seed_all(int(config["seed"]))

        owner = api.whoami()["name"]
        repo = f"{owner}/{config['output_model_name']}"
        api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
        remote_files = set(api.list_repo_files(repo, repo_type="model"))
        existing = None
        if "run-state.json" in remote_files:
            state_path = hf_hub_download(
                repo_id=repo,
                repo_type="model",
                filename="run-state.json",
                token=token,
                local_dir=work / "existing-state",
            )
            existing = json.loads(Path(state_path).read_text(encoding="utf-8"))
        assert_no_live_duplicate(existing)

        run_id = f"{config['run_name']}-{utc_now().strftime('%Y%m%dT%H%M%SZ')}"
        state = {
            "schema_version": 1,
            "version": "0.0.14",
            "status": "running",
            "run_id": run_id,
            "started_at": utc_now().isoformat(),
            "source_repo": SOURCE_REPO,
            "source_checkpoint_sha256": SOURCE_SHA256,
        }
        upload(api, repo, write_json(work / "run-state.json", state), "run-state.json", "Start Ember v0.0.14 literal-copy canary")

        model.to(device)
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )
        generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
        best_val = float("inf")
        best_step = -1
        ckpt_dir = work / "checkpoints" / run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_path = ckpt_dir / "best.pt"
        latest_path = ckpt_dir / "latest.pt"
        history = []
        max_steps = int(config["max_steps"])

        try:
            for step in range(max_steps):
                lr = cosine_lr(
                    step,
                    float(config["learning_rate"]),
                    int(config["warmup_steps"]),
                    max_steps,
                    float(config["min_lr_ratio"]),
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                accum = int(config["gradient_accumulation_steps"])
                train_loss = 0.0
                for _ in range(accum):
                    ids = torch.randint(len(train), (int(config["batch_size"]),), generator=generator).tolist()
                    bx, by, bw = batch(train, ids, device, torch)
                    with amp_ctx():
                        loss = weighted_loss(model, bx, by, bw, torch) / accum
                    if not bool(torch.isfinite(loss).item()):
                        raise RuntimeError(f"non-finite train loss at step {step}")
                    loss.backward()
                    train_loss += float(loss.item())
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
                optimizer.step()

                do_eval = step == 0 or (step + 1) % int(config["eval_interval"]) == 0 or step + 1 == max_steps
                if do_eval:
                    val_loss = evaluate(model, val, int(config["batch_size"]), device, amp_ctx, torch)
                    record = {"step": step, "train_loss": train_loss, "val_weighted_loss": val_loss, "lr": lr}
                    history.append(record)
                    print(json.dumps(record), flush=True)
                    if val_loss < best_val:
                        best_val, best_step = val_loss, step
                        save_checkpoint(
                            best_path,
                            model=model,
                            optimizer=optimizer,
                            tokenizer=tokenizer,
                            model_config=model.cfg,
                            train_config=config,
                            step=step,
                            best_val_loss=best_val,
                            run_id=run_id,
                        )
                if (step + 1) % int(config["save_interval"]) == 0 or step + 1 == max_steps:
                    save_checkpoint(
                        latest_path,
                        model=model,
                        optimizer=optimizer,
                        tokenizer=tokenizer,
                        model_config=model.cfg,
                        train_config=config,
                        step=step,
                        best_val_loss=best_val,
                        run_id=run_id,
                    )
                if (step + 1) % int(config["hub_checkpoint_interval"]) == 0:
                    upload(api, repo, latest_path, "resume/latest.pt", f"Save v0.0.14 resume step {step}")
        except Exception as exc:
            state.update(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:1000],
                    "updated_at": utc_now().isoformat(),
                }
            )
            upload(api, repo, write_json(work / "run-state.json", state), "run-state.json", "Record interrupted Ember v0.0.14 canary")
            raise

        best = load_checkpoint(best_path, device="cpu")
        model.load_state_dict(best["model_state"])
        model.to(device)
        model.eval()
        smoke = internal_smoke(model, tokenizer, val_rows, config, device, torch)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        int4_path = export_int4_checkpoint(best_path)
        if not Path(int4_path).is_file() or Path(int4_path).stat().st_size == 0:
            raise RuntimeError("v0.0.14 INT4 export failed")

        summary = {
            "schema_version": 1,
            "version": "0.0.14",
            "run_id": run_id,
            "source_model": "ember-v0.0.9-t4",
            "best_step": best_step,
            "best_weighted_validation_loss": best_val,
            "history": history,
            "internal_smoke": smoke,
            "eos_aware_generation": True,
            "semantic_token_weight": config["semantic_token_weight"],
            "status": "training_complete_pending_eval" if smoke["passed"] else "failed_internal_smoke",
        }
        summary_path = write_json(work / "training-summary.json", summary)
        frozen_config = work / "ember_agent_copy_canary_v0.0.14.json"
        frozen_config.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        upload(api, repo, best_path, f"checkpoints/{run_id}/best.pt", "Upload v0.0.14 best checkpoint")
        upload(api, repo, Path(int4_path), f"checkpoints/{run_id}/best.int4.pt", "Upload v0.0.14 INT4 checkpoint")
        upload(api, repo, latest_path, f"checkpoints/{run_id}/latest.pt", "Upload v0.0.14 latest checkpoint")
        upload(api, repo, frozen_config, "config/ember_agent_copy_canary_v0.0.14.json", "Record v0.0.14 config")
        upload(api, repo, summary_path, "training-summary.json", "Record v0.0.14 training summary")

        state.update(
            {
                "status": summary["status"],
                "updated_at": utc_now().isoformat(),
                "best_step": best_step,
                "best_weighted_validation_loss": best_val,
                "internal_smoke_pass": bool(smoke["passed"]),
            }
        )
        upload(api, repo, write_json(work / "run-state.json", state), "run-state.json", "Update Ember v0.0.14 run state")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        print(f"MODEL_REPO={repo}", flush=True)
        print(f"EMBER_HF_V014_SFT={'PASS' if smoke['passed'] else 'FAIL'}", flush=True)
        if not smoke["passed"]:
            raise RuntimeError("v0.0.14 internal copy smoke failed; do not run promotion evaluation")


if __name__ == "__main__":
    main()
