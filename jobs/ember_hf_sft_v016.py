# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["huggingface-hub>=1.4", "sentencepiece>=0.2", "torch>=2.4"]
# ///
"""Consolidate Ember v0.0.15 first-token gains into exact sequence copying.

v0.0.15 made the expected first token competitive but exact held-out copying
plateaued. v0.0.16 resumes from the saved v0.0.15 best checkpoint, lowers the
relative first-token weight, increases continuation-copy weight, and selects the
best checkpoint by exact copy and continuation-token accuracy before the legacy
first-token metrics. CPU preflight remains separate from paid T4 execution.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

BASE_PIN = "67a94d20530901a2524da8010408d5f2c728a165"
BASE_URL = f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{BASE_PIN}/jobs/ember_hf_sft_v015.py"
CONFIG_PIN = "42ed91ac6e6b8ff22873fec6706c4f243f57f2fa"
CONFIG_URL = f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{CONFIG_PIN}/config/ember_sequence_copy_v0.0.16.json"
DATA_PIN = "14995a94a4d1594463e266c4c5fed0ecec329da9"
DATA_URL = f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{DATA_PIN}/jobs/ember_sft_data_v015.py"
SOURCE_REPO = "Jmiller18899/ember-v0.0.15-t4"
EXPECTED_HF_OWNER = "Jmiller18899"
HF_WRITE_CHECK_PATH = "preflight/v0.0.16-hf-write-check.json"


def load_url_module(url: str, path: Path, name: str):
    urllib.request.urlretrieve(url, path)
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"unable to load module {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_hf_output_access(api: HfApi, cfg: dict, work: Path):
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
        "version": "0.0.16",
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
            commit_message="Verify Ember v0.0.16 Hugging Face write access",
        )
    except Exception as exc:
        raise RuntimeError(
            f"HF_TOKEN cannot create or write the required output repo {repo}. "
            f"Use a Hugging Face token with write permission for the {EXPECTED_HF_OWNER} namespace."
        ) from exc
    return owner, repo


def resolve_v015_source(api: HfApi, token: str, work: Path, base):
    state_path = Path(hf_hub_download(
        repo_id=SOURCE_REPO,
        repo_type="model",
        filename="run-state.json",
        token=token,
        local_dir=work / "source-state",
    ))
    state = json.loads(state_path.read_text())
    if state.get("status") != "evaluation_complete":
        raise RuntimeError(f"v0.0.15 source state is not evaluation_complete: {state}")
    run_id = str(state.get("run_id", "")).strip()
    if not run_id:
        raise RuntimeError("v0.0.15 run-state.json has no run_id")
    remote_path = f"checkpoints/{run_id}/best.pt"
    source_path = Path(hf_hub_download(
        repo_id=SOURCE_REPO,
        repo_type="model",
        filename=remote_path,
        token=token,
        local_dir=work / "source-model",
    ))
    return source_path, {
        "repo": SOURCE_REPO,
        "checkpoint": remote_path,
        "sha256": base.sha(source_path),
        "run_id": run_id,
        "v015_promotion": state.get("promotion"),
        "v015_best_step": state.get("best_step"),
    }


def continuation_top1_counts(model, tokenizer, prompt: str, expected: str, device, torch, base):
    prompt_ids = list(tokenizer.encode(prompt))
    full = list(tokenizer.encode(prompt + base.completion_for(expected)))
    if full[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError("continuation diagnostic boundary changed")
    eot_id = int(tokenizer.encode(base.EOT)[-1])
    eot_positions = [i for i in range(len(prompt_ids), len(full)) if int(full[i]) == eot_id]
    if not eot_positions:
        raise RuntimeError("continuation diagnostic has no EOS")
    first_target = len(prompt_ids) - 1
    eos_target = eot_positions[0] - 1
    positions = list(range(first_target + 1, eos_target))
    if not positions:
        return 0, 0
    x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    y = full[1:]
    with torch.inference_mode():
        logits, _ = model(x, None)
        pred = torch.argmax(logits[0], dim=-1)
    correct = sum(int(pred[pos].item()) == int(y[pos]) for pos in positions)
    return correct, len(positions)


def diagnostic(model, tokenizer, cfg, device, torch, base):
    result = base.diagnostic(model, tokenizer, cfg, device, torch)
    correct = 0
    total = 0
    for value, _ in base.DIAGNOSTICS:
        c, n = continuation_top1_counts(model, tokenizer, base.prompt_for(value), value, device, torch, base)
        correct += c
        total += n
    rate = correct / total if total else 0.0
    result["metrics"]["continuation_top1_rate"] = rate
    result["gates"]["continuation_top1"] = rate >= float(cfg["minimum_continuation_top1_rate"])
    result["passed"] = all(result["gates"].values())
    return result


def score(diag: dict, val_loss: float):
    m = diag["metrics"]
    return (
        m["exact_copy_rate"],
        m["continuation_top1_rate"],
        m["clean_stop_rate"],
        m["teacher_forced_expected_win_rate"],
        m["first_token_top5_rate"],
        m["first_token_top20_rate"],
        -m["mean_first_token_rank"],
        -val_loss,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required")

    import torch

    api = HfApi(token=token)
    with tempfile.TemporaryDirectory(prefix="ember-v016-") as td:
        work = Path(td)
        base = load_url_module(BASE_URL, work / "v015_base.py", "ember_v015_base")
        cfg_path = base.fetch(CONFIG_URL, work / "config.json")
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("version") != "0.0.16" or cfg.get("phase") != "sequence-copy-consolidation":
            raise RuntimeError("unexpected v0.0.16 config")
        if cfg.get("source_model_name") != "ember-v0.0.15-t4":
            raise RuntimeError("v0.0.16 must start from v0.0.15")
        if float(cfg["first_token_weight"]) >= float(cfg["copy_token_weight"]):
            raise RuntimeError("v0.0.16 must emphasize continuation copy tokens over the first token")

        data_path = base.fetch(DATA_URL, work / "data.py")
        data = base.load_module(data_path)
        train_rows = data.build_examples("train", int(cfg["train_examples"]))
        val_rows = data.build_examples("validation", int(cfg["validation_examples"]))
        data.assert_clean(train_rows, val_rows)
        owner, repo = verify_hf_output_access(api, cfg, work)

        package = base.fetch(base.PACKAGE_URL, work / "ember.zip", base.PACKAGE_SHA256)
        with zipfile.ZipFile(package) as z:
            z.extractall(work / "src")
        sys.path.insert(0, str(work / "src" / "ember"))
        from src.checkpoint import load_checkpoint, save_checkpoint
        from src.model import EmberGPT, ModelConfig
        from src.tokenizer import tokenizer_from_state_dict

        source_path, source_info = resolve_v015_source(api, token, work, base)
        source = load_checkpoint(source_path, device="cpu")
        source_cfg = source.get("train_config") or {}
        if str(source_cfg.get("version")) != "0.0.15":
            raise RuntimeError(f"expected a v0.0.15 checkpoint, got train_config={source_cfg}")
        if str(source.get("run_id", "")) != source_info["run_id"]:
            raise RuntimeError("v0.0.15 checkpoint run_id does not match run-state.json")

        tokenizer = tokenizer_from_state_dict(source["tokenizer"])
        model = EmberGPT(ModelConfig(**source["model_config"]))
        model.load_state_dict(source["model_state"])
        train = [base.encode_row(tokenizer, row, cfg, torch) for row in train_rows]
        val = [base.encode_row(tokenizer, row, cfg, torch) for row in val_rows]
        baseline = diagnostic(model, tokenizer, cfg, "cpu", torch, base)

        preflight = {
            "status": "PASS",
            "version": "0.0.16",
            "source": source_info,
            "train": len(train),
            "validation": len(val),
            "baseline": baseline["metrics"],
            "hf": {
                "owner": owner,
                "output_repo": repo,
                "write_check": "PASS",
                "write_check_path": HF_WRITE_CHECK_PATH,
            },
            "diagnostic_gates": {k: cfg[k] for k in cfg if k.startswith("minimum_")},
        }
        print(json.dumps(preflight, indent=2), flush=True)
        print(f"EMBER_V016_SOURCE_CHECKPOINT={source_info['checkpoint']}", flush=True)
        print(f"EMBER_V016_SOURCE_SHA256={source_info['sha256']}", flush=True)
        if args.preflight_only:
            print("EMBER_HF_V016_PREFLIGHT=PASS", flush=True)
            return

        if not torch.cuda.is_available():
            raise RuntimeError("v0.0.16 training requires explicitly approved T4 GPU")
        device = "cuda"
        model.to(device)
        model.train()
        amp = lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        torch.backends.cuda.matmul.allow_tf32 = True
        random.seed(int(cfg["seed"]))
        torch.manual_seed(int(cfg["seed"]))
        torch.cuda.manual_seed_all(int(cfg["seed"]))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["learning_rate"]),
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )
        generator = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))
        run_id = f"{cfg['run_name']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        ckpt = work / "checkpoints"
        ckpt.mkdir()
        best_path = ckpt / "best.pt"
        latest_path = ckpt / "latest.pt"
        best_score = None
        best_step = -1
        history = []
        max_steps = int(cfg["max_steps"])
        accum = int(cfg["gradient_accumulation_steps"])

        for step in range(max_steps):
            lr = base.cosine_lr(
                step,
                float(cfg["learning_rate"]),
                int(cfg["warmup_steps"]),
                max_steps,
                float(cfg["min_lr_ratio"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            for _ in range(accum):
                ids = torch.randint(len(train), (int(cfg["batch_size"]),), generator=generator).tolist()
                x, y, w = base.batch(train, ids, device, torch)
                with amp():
                    loss = base.weighted_loss(model, x, y, w, torch) / accum
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError(f"non-finite loss at {step}")
                loss.backward()
                train_loss += float(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            optimizer.step()

            do_eval = step == 0 or (step + 1) % int(cfg["eval_interval"]) == 0 or step + 1 == max_steps
            if do_eval:
                val_loss = base.evaluate_loss(model, val, cfg, device, amp, torch)
                diag = diagnostic(model, tokenizer, cfg, device, torch, base)
                current = score(diag, val_loss)
                record = {
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": lr,
                    "diagnostic": diag["metrics"],
                    "passed": diag["passed"],
                }
                history.append(record)
                print(json.dumps(record), flush=True)
                if best_score is None or current > best_score:
                    best_score = current
                    best_step = step
                    save_checkpoint(
                        best_path,
                        model=model,
                        optimizer=optimizer,
                        tokenizer=tokenizer,
                        model_config=model.cfg,
                        train_config=cfg,
                        step=step,
                        best_val_loss=val_loss,
                        run_id=run_id,
                    )
            if (step + 1) % int(cfg["save_interval"]) == 0 or step + 1 == max_steps:
                save_checkpoint(
                    latest_path,
                    model=model,
                    optimizer=optimizer,
                    tokenizer=tokenizer,
                    model_config=model.cfg,
                    train_config=cfg,
                    step=step,
                    best_val_loss=min((r["val_loss"] for r in history), default=float("inf")),
                    run_id=run_id,
                )

        best = load_checkpoint(best_path, device=device)
        model.load_state_dict(best["model_state"])
        final = diagnostic(model, tokenizer, cfg, device, torch, base)
        report = {
            "version": "0.0.16",
            "phase": cfg["phase"],
            "run_id": run_id,
            "source": source_info,
            "best_step": best_step,
            "baseline": baseline,
            "final": final,
            "history": history,
            "promotion": "PASS" if final["passed"] else "FAIL",
        }
        report_path = work / "v0.0.16-report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        base.upload(api, repo, best_path, f"checkpoints/{run_id}/best.pt", "Ember v0.0.16 best sequence-copy checkpoint")
        base.upload(api, repo, latest_path, f"checkpoints/{run_id}/latest.pt", "Ember v0.0.16 latest checkpoint")
        base.upload(api, repo, report_path, "evaluation/v0.0.16-report.json", "Ember v0.0.16 diagnostic report")
        state = work / "run-state.json"
        state.write_text(json.dumps({
            "status": "evaluation_complete",
            "promotion": report["promotion"],
            "run_id": run_id,
            "best_step": best_step,
            "source_checkpoint": source_info["checkpoint"],
            "source_sha256": source_info["sha256"],
        }, indent=2) + "\n")
        base.upload(api, repo, state, "run-state.json", "Complete Ember v0.0.16 sequence-copy consolidation")
        print(f"EMBER_HF_V016_PROMOTION={report['promotion']}", flush=True)
        print(json.dumps(final["metrics"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
