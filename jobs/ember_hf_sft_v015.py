# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["huggingface-hub>=1.4", "sentencepiece>=0.2", "torch>=2.4"]
# ///
"""Train Ember v0.0.15 to make literal prompt copies win at generation start.

This phase starts only from accepted v0.0.9. It trains direct TARGET -> TARGET+EOS
examples, evaluates held-out first-token rank during training, and refuses
promotion unless the CPU-diagnostic gates are met. Paid GPU execution is separate
from --preflight-only.
"""
from __future__ import annotations

import argparse
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
ASSET_PIN = "14995a94a4d1594463e266c4c5fed0ecec329da9"
CONFIG_URL = f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{ASSET_PIN}/config/ember_copy_warmup_v0.0.15.json"
DATA_URL = f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{ASSET_PIN}/jobs/ember_sft_data_v015.py"
SOURCE_REPO = "Jmiller18899/ember-v0.0.9-t4"
SOURCE_CHECKPOINT = "checkpoints/ember-agent-v0.0.9-tool-sft-20260828T142047Z/best.pt"
SOURCE_SHA256 = "8299d52e8a852b9bd3e8403e086b48fd42d2babfd51c74eb94af29bd87ef2d13"
EXPECTED_HF_OWNER = "Jmiller18899"
HF_WRITE_CHECK_PATH = "preflight/v0.0.15-hf-write-check.json"
EOT = "<|endoftext|>"
DIAGNOSTICS = (
    ("Q7M4", "R8N5"),
    ("V9K2-4R7P", "W8L3-5S6Q"),
    ("58310429", "69421530"),
    ("openai/gpt-6-astra", "openai/gpt-5.6-sol"),
    ("https://example.test/a7Q9", "https://example.test/b8R2"),
    ("/tmp/ember/Q7M4/result.json", "/tmp/ember/R8N5/output.json"),
    ("Northfield Zephyr", "Westhaven Orion"),
    ("53*19+7", "61*17+9"),
    ("acct_Q7m4-5831", "acct_R8n5-6942"),
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, path: Path, expected: str | None = None) -> Path:
    urllib.request.urlretrieve(url, path)
    if expected and sha(path) != expected:
        raise RuntimeError(f"checksum mismatch for {url}")
    return path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("ember_sft_data_v015", path)
    if not spec or not spec.loader:
        raise RuntimeError("unable to load v0.0.15 data module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_hf_output_access(api, cfg: dict, work: Path):
    identity = api.whoami()
    owner = str(identity.get("name", "")).strip()
    if not owner:
        raise RuntimeError("HF_TOKEN identity check returned no account name")
    if owner.casefold() != EXPECTED_HF_OWNER.casefold():
        raise RuntimeError(f"HF_TOKEN belongs to {owner!r}; expected {EXPECTED_HF_OWNER!r}")
    repo = f"{owner}/{cfg['output_model_name']}"
    marker = work / "hf-write-preflight.json"
    marker.write_text(json.dumps({
        "status": "PASS",
        "version": "0.0.15",
        "owner": owner,
        "output_repo": repo,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")
    try:
        api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
        api.upload_file(
            repo_id=repo,
            repo_type="model",
            path_or_fileobj=str(marker),
            path_in_repo=HF_WRITE_CHECK_PATH,
            commit_message="Verify Ember v0.0.15 Hugging Face write access",
        )
    except Exception as exc:
        raise RuntimeError(
            f"HF_TOKEN cannot create or write the required output repo {repo}. "
            f"Use a Hugging Face token with write permission for the {EXPECTED_HF_OWNER} namespace."
        ) from exc
    return owner, repo


def prompt_for(value: str) -> str:
    return (
        "<|system|>\nYou are Ember. Copy TARGET from the current user message exactly. "
        "Do not explain, normalize, calculate, or call a tool. Stop at endoftext.\n"
        f"<|user|>\nIgnore old=K2P8 and fallback=77291. TARGET={value}. "
        "Reply with TARGET exactly once.\n<|assistant|>\n"
    )


def completion_for(value: str) -> str:
    return f"{value}\n{EOT}\n"


def cosine_lr(step: int, base: float, warmup: int, total: int, floor: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def encode_row(tokenizer, row: dict, cfg: dict, torch):
    prompt_ids = list(tokenizer.encode(row["prompt"]))
    full_ids = list(tokenizer.encode(row["prompt"] + row["completion"]))
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError(f"completion boundary changed: {row['id']}")
    if len(full_ids) > int(cfg["block_size"]) + 1:
        raise RuntimeError(f"sequence too long: {row['id']}")
    eot_id = int(tokenizer.encode(EOT)[-1])
    eot_positions = [i for i in range(len(prompt_ids), len(full_ids)) if int(full_ids[i]) == eot_id]
    if not eot_positions:
        raise RuntimeError(f"missing EOS: {row['id']}")
    eos_token_pos = eot_positions[0]
    block = int(cfg["block_size"])
    x = torch.full((block,), eot_id, dtype=torch.long)
    y = torch.full((block,), -100, dtype=torch.long)
    w = torch.zeros((block,), dtype=torch.float32)
    seq_x = torch.tensor(full_ids[:-1], dtype=torch.long)
    seq_y = torch.tensor(full_ids[1:], dtype=torch.long)
    x[:len(seq_x)] = seq_x
    first = len(prompt_ids) - 1
    y[first:len(seq_y)] = seq_y[first:]
    w[first:len(seq_y)] = 1.0
    eos_target = eos_token_pos - 1
    for pos in range(first, eos_target):
        w[pos] = float(cfg["copy_token_weight"])
    w[first] = float(cfg["first_token_weight"])
    w[eos_target] = float(cfg["eos_token_weight"])
    return {"x": x, "y": y, "w": w, "row": row}


def batch(rows, indices, device, torch):
    return tuple(torch.stack([rows[i][key] for i in indices]).to(device) for key in ("x", "y", "w"))


def weighted_loss(model, x, y, weights, torch):
    import torch.nn.functional as F
    logits, _ = model(x, None)
    raw = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction="none").reshape_as(y)
    active = y.ne(-100)
    denom = (weights * active).sum().clamp_min(1.0)
    return (raw * weights * active).sum() / denom


def evaluate_loss(model, rows, cfg, device, amp_ctx, torch):
    model.eval(); values = []
    limit = min(len(rows), 128)
    with torch.inference_mode():
        for start in range(0, limit, int(cfg["batch_size"])):
            ids = list(range(start, min(limit, start + int(cfg["batch_size"]))))
            x, y, w = batch(rows, ids, device, torch)
            with amp_ctx():
                loss = weighted_loss(model, x, y, w, torch)
            values.append(float(loss.item()))
    model.train()
    return sum(values) / len(values)


def greedy(model, tokenizer, prompt: str, device, torch, max_new=80):
    eot_id = int(tokenizer.encode(EOT)[-1])
    ids = list(tokenizer.encode(prompt))
    seq = torch.tensor([ids], dtype=torch.long, device=device)
    out = []
    with torch.inference_mode():
        for _ in range(max_new):
            logits, _ = model(seq[:, -int(model.cfg.block_size):], None)
            nxt = int(torch.argmax(logits[0, -1]).item())
            out.append(nxt)
            seq = torch.cat([seq, torch.tensor([[nxt]], device=device)], dim=1)
            if nxt == eot_id:
                return tokenizer.decode(out), True
    return tokenizer.decode(out), False


def first_token_stats(model, tokenizer, prompt: str, expected: str, device, torch):
    p = list(tokenizer.encode(prompt))
    full = list(tokenizer.encode(prompt + completion_for(expected)))
    if full[:len(p)] != p:
        raise RuntimeError("diagnostic completion boundary changed")
    expected_id = int(full[len(p)])
    x = torch.tensor([p[-int(model.cfg.block_size):]], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits, _ = model(x, None)
        row = logits[0, -1]
        score = float(row[expected_id].item())
        rank = 1 + int((row > row[expected_id]).sum().item())
    return {"rank": rank, "top5": rank <= 5, "top20": rank <= 20, "logit": score}


def completion_nll(model, tokenizer, prompt: str, value: str, device, torch) -> float:
    p = list(tokenizer.encode(prompt)); full = list(tokenizer.encode(prompt + completion_for(value)))
    if full[:len(p)] != p:
        return float("inf")
    x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    y = full[1:]; first = len(p) - 1
    with torch.inference_mode():
        logits, _ = model(x, None)
        lp = torch.log_softmax(logits[0], dim=-1)
        vals = [-float(lp[i, int(y[i])].item()) for i in range(first, len(y))]
    return sum(vals) / len(vals)


def diagnostic(model, tokenizer, cfg, device, torch):
    model.eval(); rows = []
    for value, corrupt in DIAGNOSTICS:
        prompt = prompt_for(value)
        text, stopped = greedy(model, tokenizer, prompt, device, torch)
        visible = text.split(EOT, 1)[0].strip()
        first = first_token_stats(model, tokenizer, prompt, value, device, torch)
        expected_nll = completion_nll(model, tokenizer, prompt, value, device, torch)
        corrupt_nll = completion_nll(model, tokenizer, prompt, corrupt, device, torch)
        rows.append({"value": value, "generated": visible, "exact": visible == value, "clean_stop": stopped,
                     "first_token": first, "expected_nll": expected_nll, "corrupt_nll": corrupt_nll,
                     "expected_wins": expected_nll < corrupt_nll})
    n = len(rows)
    metrics = {
        "first_token_top20_rate": sum(r["first_token"]["top20"] for r in rows) / n,
        "first_token_top5_rate": sum(r["first_token"]["top5"] for r in rows) / n,
        "teacher_forced_expected_win_rate": sum(r["expected_wins"] for r in rows) / n,
        "clean_stop_rate": sum(r["clean_stop"] for r in rows) / n,
        "exact_copy_rate": sum(r["exact"] for r in rows) / n,
        "mean_first_token_rank": sum(r["first_token"]["rank"] for r in rows) / n,
    }
    gates = {
        "top20": metrics["first_token_top20_rate"] >= float(cfg["minimum_first_token_top20_rate"]),
        "top5": metrics["first_token_top5_rate"] >= float(cfg["minimum_first_token_top5_rate"]),
        "teacher_forced": metrics["teacher_forced_expected_win_rate"] >= float(cfg["minimum_teacher_forced_expected_win_rate"]),
        "clean_stop": metrics["clean_stop_rate"] >= float(cfg["minimum_clean_stop_rate"]),
        "exact_copy": metrics["exact_copy_rate"] >= float(cfg["minimum_exact_copy_rate"]),
    }
    model.train()
    return {"passed": all(gates.values()), "metrics": metrics, "gates": gates, "cases": rows}


def score(diag: dict, val_loss: float):
    m = diag["metrics"]
    return (m["first_token_top20_rate"], m["first_token_top5_rate"], m["exact_copy_rate"],
            m["teacher_forced_expected_win_rate"], -m["mean_first_token_rank"], -val_loss)


def upload(api, repo, local: Path, remote: str, message: str):
    api.upload_file(repo_id=repo, repo_type="model", path_or_fileobj=str(local), path_in_repo=remote, commit_message=message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    import torch
    api = HfApi(token=token)
    with tempfile.TemporaryDirectory(prefix="ember-v015-") as td:
        work = Path(td)
        cfg = json.loads(fetch(CONFIG_URL, work / "config.json").read_text())
        data_path = fetch(DATA_URL, work / "data.py")
        if cfg.get("version") != "0.0.15" or cfg.get("phase") != "first-token-literal-copy-warmup":
            raise RuntimeError("unexpected v0.0.15 config")
        if cfg.get("source_model_name") != "ember-v0.0.9-t4":
            raise RuntimeError("v0.0.15 must start from v0.0.9")
        owner, repo = verify_hf_output_access(api, cfg, work)
        data = load_module(data_path)
        train_rows = data.build_examples("train", int(cfg["train_examples"]))
        val_rows = data.build_examples("validation", int(cfg["validation_examples"]))
        data.assert_clean(train_rows, val_rows)
        package = fetch(PACKAGE_URL, work / "ember.zip", PACKAGE_SHA256)
        with zipfile.ZipFile(package) as z: z.extractall(work / "src")
        sys.path.insert(0, str(work / "src" / "ember"))
        from src.checkpoint import load_checkpoint, save_checkpoint
        from src.model import EmberGPT, ModelConfig
        from src.tokenizer import tokenizer_from_state_dict
        source_path = Path(hf_hub_download(repo_id=SOURCE_REPO, repo_type="model", filename=SOURCE_CHECKPOINT, token=token, local_dir=work / "source"))
        if sha(source_path) != SOURCE_SHA256: raise RuntimeError("source checkpoint checksum mismatch")
        source = load_checkpoint(source_path, device="cpu")
        tokenizer = tokenizer_from_state_dict(source["tokenizer"])
        model = EmberGPT(ModelConfig(**source["model_config"])); model.load_state_dict(source["model_state"])
        train = [encode_row(tokenizer, r, cfg, torch) for r in train_rows]
        val = [encode_row(tokenizer, r, cfg, torch) for r in val_rows]
        baseline = diagnostic(model, tokenizer, cfg, "cpu", torch)
        preflight = {"status": "PASS", "source": SOURCE_REPO, "train": len(train), "validation": len(val), "baseline": baseline["metrics"],
                     "hf": {"owner": owner, "output_repo": repo, "write_check": "PASS", "write_check_path": HF_WRITE_CHECK_PATH},
                     "diagnostic_gates": {k: cfg[k] for k in cfg if k.startswith("minimum_")}}
        print(json.dumps(preflight, indent=2), flush=True)
        if args.preflight_only:
            print("EMBER_HF_V015_PREFLIGHT=PASS", flush=True); return
        if not torch.cuda.is_available(): raise RuntimeError("v0.0.15 training requires explicitly approved T4 GPU")
        device = "cuda"; model.to(device); model.train()
        amp = lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        torch.backends.cuda.matmul.allow_tf32 = True
        random.seed(int(cfg["seed"])); torch.manual_seed(int(cfg["seed"])); torch.cuda.manual_seed_all(int(cfg["seed"]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), betas=(0.9, 0.95), weight_decay=0.01)
        generator = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))
        run_id = f"{cfg['run_name']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        ckpt = work / "checkpoints"; ckpt.mkdir(); best_path = ckpt / "best.pt"; latest_path = ckpt / "latest.pt"
        best_score = None; best_step = -1; history = []
        max_steps = int(cfg["max_steps"]); accum = int(cfg["gradient_accumulation_steps"])
        for step in range(max_steps):
            lr = cosine_lr(step, float(cfg["learning_rate"]), int(cfg["warmup_steps"]), max_steps, float(cfg["min_lr_ratio"]))
            for group in optimizer.param_groups: group["lr"] = lr
            optimizer.zero_grad(set_to_none=True); train_loss = 0.0
            for _ in range(accum):
                ids = torch.randint(len(train), (int(cfg["batch_size"]),), generator=generator).tolist()
                x, y, w = batch(train, ids, device, torch)
                with amp(): loss = weighted_loss(model, x, y, w, torch) / accum
                if not bool(torch.isfinite(loss).item()): raise RuntimeError(f"non-finite loss at {step}")
                loss.backward(); train_loss += float(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"])); optimizer.step()
            do_eval = step == 0 or (step + 1) % int(cfg["eval_interval"]) == 0 or step + 1 == max_steps
            if do_eval:
                val_loss = evaluate_loss(model, val, cfg, device, amp, torch)
                diag = diagnostic(model, tokenizer, cfg, device, torch)
                current = score(diag, val_loss)
                record = {"step": step, "train_loss": train_loss, "val_loss": val_loss, "lr": lr, "diagnostic": diag["metrics"], "passed": diag["passed"]}
                history.append(record); print(json.dumps(record), flush=True)
                if best_score is None or current > best_score:
                    best_score, best_step = current, step
                    save_checkpoint(best_path, model=model, optimizer=optimizer, tokenizer=tokenizer, model_config=model.cfg, train_config=cfg,
                                    step=step, best_val_loss=val_loss, run_id=run_id)
            if (step + 1) % int(cfg["save_interval"]) == 0 or step + 1 == max_steps:
                save_checkpoint(latest_path, model=model, optimizer=optimizer, tokenizer=tokenizer, model_config=model.cfg, train_config=cfg,
                                step=step, best_val_loss=min((r["val_loss"] for r in history), default=float("inf")), run_id=run_id)
        best = load_checkpoint(best_path, device=device); model.load_state_dict(best["model_state"]); final = diagnostic(model, tokenizer, cfg, device, torch)
        report = {"version": "0.0.15", "run_id": run_id, "source": SOURCE_REPO, "best_step": best_step, "baseline": baseline,
                  "final": final, "history": history, "promotion": "PASS" if final["passed"] else "FAIL"}
        report_path = work / "v0.0.15-report.json"; report_path.write_text(json.dumps(report, indent=2) + "\n")
        upload(api, repo, best_path, f"checkpoints/{run_id}/best.pt", "Ember v0.0.15 best first-token checkpoint")
        upload(api, repo, latest_path, f"checkpoints/{run_id}/latest.pt", "Ember v0.0.15 latest checkpoint")
        upload(api, repo, report_path, "evaluation/v0.0.15-report.json", "Ember v0.0.15 diagnostic report")
        state = work / "run-state.json"; state.write_text(json.dumps({"status": "evaluation_complete", "promotion": report["promotion"], "run_id": run_id, "best_step": best_step}, indent=2) + "\n")
        upload(api, repo, state, "run-state.json", "Complete Ember v0.0.15 warmup")
        print(f"EMBER_HF_V015_PROMOTION={report['promotion']}", flush=True)
        if not final["passed"]: raise RuntimeError("v0.0.15 finished but failed diagnostic promotion gates")

if __name__ == "__main__":
    main()
