# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "huggingface-hub>=1.4",
#   "sentencepiece>=0.2",
#   "torch>=2.4",
# ]
# ///
"""Ember v0.0.11 corrective semantic-copy SFT.

Starts from accepted v0.0.9, upweights dynamic semantic tokens, and uses
EOS-aware greedy smoke generation. A failed internal smoke is a failed run,
not a promotable candidate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import tempfile
import urllib.request
import zipfile

from huggingface_hub import HfApi, hf_hub_download

ASSET_PIN = "12645f8166c7ae583eb6f645f672a4352efb4f5d"
CONFIG_URL = f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{ASSET_PIN}/config/ember_agent_semantic_sft_v0.0.11.json"
DATA_URL = f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{ASSET_PIN}/jobs/ember_sft_data_v011.py"
PACKAGE_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/main/ember-v0.0.7-hf-ready.zip"
PACKAGE_SHA256 = "27e8f7c80317652a22b3d58a0bd474724491a685dfe9e20c0b997b7c5907a289"
SOURCE_REPO = "Jmiller18899/ember-v0.0.9-t4"
SOURCE_CHECKPOINT = "checkpoints/ember-agent-v0.0.9-tool-sft-20260828T142047Z/best.pt"
SOURCE_SHA256 = "8299d52e8a852b9bd3e8403e086b48fd42d2babfd51c74eb94af29bd87ef2d13"

def utc_now():
    return datetime.now(timezone.utc)

def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def fetch(url, path):
    urllib.request.urlretrieve(url, path)
    return path

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

def load_module(path):
    spec = importlib.util.spec_from_file_location("ember_sft_data_v011", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load v0.0.11 data module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def cosine_lr(step, base, warmup, max_steps, minimum):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return base * (minimum + (1.0 - minimum) * cosine)

def subseq_hits(haystack, needle, start):
    if not needle:
        return []
    out = []
    n = len(needle)
    for i in range(max(0, start), len(haystack) - n + 1):
        if haystack[i:i+n] == needle:
            out.append((i, i+n))
    return out

def term_sequences(tokenizer, term):
    seqs = []
    for value in (str(term), " " + str(term)):
        ids = list(tokenizer.encode(value))
        if ids and ids not in seqs:
            seqs.append(ids)
        if len(ids) > 1 and ids[1:] not in seqs:
            seqs.append(ids[1:])
    return sorted(seqs, key=len, reverse=True)

def encode_row(tokenizer, row, block_size, semantic_weight, eos_weight, torch):
    prompt_ids = list(tokenizer.encode(row["prompt"]))
    full_ids = list(tokenizer.encode(row["prompt"] + row["completion"]))
    if full_ids[:len(prompt_ids)] != prompt_ids:
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
    x[:len(seq_x)] = seq_x
    first_target = max(0, len(prompt_ids) - 1)
    y[first_target:len(seq_y)] = seq_y[first_target:]
    w[first_target:len(seq_y)] = 1.0

    semantic_positions = set()
    missing = []
    for term in row["focus_terms"]:
        found = False
        for needle in term_sequences(tokenizer, term):
            for a, b in subseq_hits(full_ids, needle, len(prompt_ids)):
                found = True
                for token_pos in range(a, b):
                    target_pos = token_pos - 1
                    if first_target <= target_pos < len(seq_y):
                        semantic_positions.add(target_pos)
        if not found:
            missing.append(str(term))
    if missing:
        raise RuntimeError(f"semantic token span not found: {row['id']} {missing}")
    for pos in semantic_positions:
        w[pos] = max(float(w[pos]), float(semantic_weight))

    eos_positions = 0
    for token_pos in range(len(prompt_ids), len(full_ids)):
        if int(full_ids[token_pos]) == eot_id:
            target_pos = token_pos - 1
            if first_target <= target_pos < len(seq_y):
                w[target_pos] = max(float(w[target_pos]), float(eos_weight))
                eos_positions += 1
    if eos_positions < 1:
        raise RuntimeError(f"no trainable EOS target: {row['id']}")
    return {"x":x, "y":y, "w":w, "row":row, "semantic_positions":len(semantic_positions)}

def encode_dataset(tokenizer, rows, config, torch):
    encoded = [
        encode_row(
            tokenizer, row, int(config["block_size"]),
            float(config["semantic_token_weight"]), float(config["eos_token_weight"]), torch
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
        logits.reshape(-1, logits.size(-1)), y.reshape(-1),
        ignore_index=-100, reduction="none"
    ).reshape_as(y)
    active = y.ne(-100)
    denom = (weights * active).sum().clamp_min(1.0)
    loss = (raw * weights * active).sum() / denom
    return loss

def evaluate(model, dataset, batch_size, device, amp_ctx, torch, limit=120):
    model.eval()
    losses = []
    subset = list(range(min(len(dataset), limit)))
    with torch.inference_mode():
        for start in range(0, len(subset), batch_size):
            x, y, w = batch(dataset, subset[start:start+batch_size], device, torch)
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
            if escaped: escaped = False
            elif ch == "\\": escaped = True
            elif ch == '"': quoted = False
            continue
        if ch == '"': quoted = True
        elif ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try: return json.loads(text[start:i+1])
                except json.JSONDecodeError: return None
    return None

def eos_greedy(model, tokenizer, prompt, device, torch, max_new=80):
    prompt_ids = list(tokenizer.encode(prompt))
    eot_id = int(tokenizer.encode("<|endoftext|>")[-1])
    seq = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = []
    stopped = False
    with torch.inference_mode():
        for _ in range(max_new):
            x = seq[:, -int(model.cfg.block_size):]
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
    for kind in ("tool_call","direct_response","tool_result_response"):
        selected += [row for row in rows if row["kind"] == kind][:n]
    results = []
    for row in selected:
        completion, stopped = eos_greedy(model, tokenizer, row["prompt"], device, torch)
        low = completion.casefold()
        focus_ok = all(str(term).casefold() in low for term in row["focus_terms"])
        if row["kind"] == "tool_call":
            payload = extract_json(completion)
            tool_ok = (
                "<|tool|>" in completion
                and isinstance(payload, dict)
                and payload.get("name") == row["expected_tool"]
            )
            passed = bool(tool_ok and focus_ok and stopped)
        else:
            passed = bool("<|tool|>" not in completion and focus_ok and stopped)
        results.append({
            "id":row["id"], "kind":row["kind"], "passed":passed,
            "stopped":stopped, "focus_ok":focus_ok, "completion":completion,
        })
    rates = {}
    for kind in ("tool_call","direct_response","tool_result_response"):
        group = [r for r in results if r["kind"] == kind]
        rates[kind] = sum(r["passed"] for r in group) / len(group)
    rates["clean_stop"] = sum(r["stopped"] for r in results) / len(results)
    passed = (
        rates["tool_call"] >= float(config["internal_minimum_valid_tool_call_rate"])
        and rates["direct_response"] >= float(config["internal_minimum_direct_response_rate"])
        and rates["tool_result_response"] >= float(config["internal_minimum_tool_result_response_rate"])
        and rates["clean_stop"] >= float(config["internal_minimum_clean_stop_rate"])
    )
    return {"passed":passed, "rates":rates, "cases":results}

def upload(api, repo, local, remote, message):
    api.upload_file(
        repo_id=repo, repo_type="model", path_or_fileobj=str(local),
        path_in_repo=remote, commit_message=message
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is required")

    import torch
    api = HfApi(token=token)
    with tempfile.TemporaryDirectory(prefix="ember-v011-") as td:
        work = Path(td)
        config_path = fetch(CONFIG_URL, work/"config.json")
        data_path = fetch(DATA_URL, work/"data.py")
        config = json.loads(config_path.read_text())
        data = load_module(data_path)
        train_rows = data.build_examples("train", int(config["train_examples"]))
        val_rows = data.build_examples("validation", int(config["validation_examples"]))

        archive = fetch(PACKAGE_URL, work/"ember.zip")
        if sha(archive) != PACKAGE_SHA256:
            raise RuntimeError("package checksum mismatch")
        with zipfile.ZipFile(archive) as z:
            z.extractall(work/"src")
        import sys
        root = work/"src"/"ember"
        sys.path.insert(0, str(root))
        from src.checkpoint import load_checkpoint, save_checkpoint
        from src.model import EmberGPT, ModelConfig
        from src.quantize_int4 import export_int4_checkpoint
        from src.tokenizer import tokenizer_from_state_dict

        source_path = Path(hf_hub_download(
            repo_id=SOURCE_REPO, repo_type="model", filename=SOURCE_CHECKPOINT,
            token=token, local_dir=work/"source"
        ))
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
            "status":"PASS", "version":"0.0.11",
            "weighted_probe_loss":float(probe.item()),
            "train_examples":len(train), "validation_examples":len(val),
            "min_semantic_positions":min(i["semantic_positions"] for i in train+val),
            "max_semantic_positions":max(i["semantic_positions"] for i in train+val),
            "source_checkpoint_sha256":SOURCE_SHA256,
        }
        print(json.dumps(preflight, indent=2, sort_keys=True))
        if args.preflight_only:
            print("EMBER_V011_TRAINER_PREFLIGHT=PASS")
            return

        if not torch.cuda.is_available():
            raise RuntimeError("paid v0.0.11 training requires CUDA")
        device = "cuda"
        dtype = torch.float16
        amp_ctx = lambda: torch.autocast(device_type="cuda", dtype=dtype)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        random.seed(int(config["seed"]))
        torch.manual_seed(int(config["seed"]))
        torch.cuda.manual_seed_all(int(config["seed"]))

        owner = api.whoami()["name"]
        repo = f"{owner}/{config['output_model_name']}"
        api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
        remote_files = set(api.list_repo_files(repo, repo_type="model"))
        if "run-state.json" in remote_files:
            state_path = hf_hub_download(
                repo_id=repo, repo_type="model", filename="run-state.json",
                token=token, local_dir=work/"existing-state"
            )
            existing = json.loads(Path(state_path).read_text())
            if existing.get("status") in {"running","training_complete_pending_eval","evaluation_complete","complete"}:
                raise RuntimeError(f"refusing duplicate v0.0.11 run; state={existing.get('status')}")

        run_id = f"{config['run_name']}-{utc_now().strftime('%Y%m%dT%H%M%SZ')}"
        state = {
            "schema_version":1, "version":"0.0.11", "status":"running",
            "run_id":run_id, "started_at":utc_now().isoformat(),
            "source_repo":SOURCE_REPO, "source_checkpoint_sha256":SOURCE_SHA256,
        }
        state_path = write_json(work/"run-state.json", state)
        upload(api, repo, state_path, "run-state.json", "Start Ember v0.0.11 corrective SFT")

        model.to(device)
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config["learning_rate"]),
            betas=(0.9,0.95), weight_decay=0.01
        )
        gen = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
        best_val = float("inf")
        best_step = -1
        ckpt_dir = work/"checkpoints"/run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_path = ckpt_dir/"best.pt"
        latest_path = ckpt_dir/"latest.pt"
        history = []
        max_steps = int(config["max_steps"])

        for step in range(max_steps):
            lr = cosine_lr(
                step, float(config["learning_rate"]), int(config["warmup_steps"]),
                max_steps, float(config["min_lr_ratio"])
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            accum = int(config["gradient_accumulation_steps"])
            train_loss = 0.0
            for _ in range(accum):
                ids = torch.randint(len(train), (int(config["batch_size"]),), generator=gen).tolist()
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
                val_loss = evaluate(
                    model, val, int(config["batch_size"]), device, amp_ctx, torch, limit=120
                )
                record = {"step":step, "train_loss":train_loss, "val_weighted_loss":val_loss, "lr":lr}
                history.append(record)
                print(json.dumps(record), flush=True)
                if val_loss < best_val:
                    best_val, best_step = val_loss, step
                    save_checkpoint(
                        best_path, model=model, optimizer=optimizer, tokenizer=tokenizer,
                        model_config=model.cfg, train_config=config, step=step,
                        best_val_loss=best_val, run_id=run_id
                    )

            if (step + 1) % int(config["save_interval"]) == 0 or step + 1 == max_steps:
                save_checkpoint(
                    latest_path, model=model, optimizer=optimizer, tokenizer=tokenizer,
                    model_config=model.cfg, train_config=config, step=step,
                    best_val_loss=best_val, run_id=run_id
                )
            if (step + 1) % int(config["hub_checkpoint_interval"]) == 0:
                upload(api, repo, latest_path, "resume/latest.pt", f"Save v0.0.11 resume step {step}")

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
            raise RuntimeError("INT4 export failed")

        summary = {
            "schema_version":1, "version":"0.0.11", "run_id":run_id,
            "source_model":"ember-v0.0.9-t4",
            "best_step":best_step, "best_weighted_validation_loss":best_val,
            "history":history, "internal_smoke":smoke,
            "eos_aware_generation":True,
            "semantic_token_weight":config["semantic_token_weight"],
            "status":"training_complete_pending_eval" if smoke["passed"] else "failed_internal_smoke",
        }
        summary_path = write_json(work/"training-summary.json", summary)
        frozen_config = work/"ember_agent_semantic_sft_v0.0.11.json"
        frozen_config.write_text(config_path.read_text(), encoding="utf-8")
        upload(api, repo, best_path, f"checkpoints/{run_id}/best.pt", "Upload v0.0.11 best checkpoint")
        upload(api, repo, Path(int4_path), f"checkpoints/{run_id}/best.int4.pt", "Upload v0.0.11 INT4 checkpoint")
        upload(api, repo, latest_path, f"checkpoints/{run_id}/latest.pt", "Upload v0.0.11 latest checkpoint")
        upload(api, repo, frozen_config, "config/ember_agent_semantic_sft_v0.0.11.json", "Record v0.0.11 config")
        upload(api, repo, summary_path, "training-summary.json", "Record v0.0.11 training summary")

        state.update({
            "status":summary["status"], "updated_at":utc_now().isoformat(),
            "best_step":best_step, "best_weighted_validation_loss":best_val,
            "internal_smoke_pass":bool(smoke["passed"]),
        })
        state_path = write_json(work/"run-state.json", state)
        upload(api, repo, state_path, "run-state.json", "Update Ember v0.0.11 run state")

        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"MODEL_REPO={repo}")
        print(f"EMBER_V011_SFT={'PASS' if smoke['passed'] else 'FAIL'}")
        if not smoke["passed"]:
            raise RuntimeError("v0.0.11 internal semantic smoke failed; do not run promotion evaluation")

if __name__ == "__main__":
    main()
