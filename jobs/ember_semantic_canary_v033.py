"""One CPU copy-replay canary from v0.0.31, with unchanged learning gates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from jobs import ember_semantic_train_v032 as base

DEFAULT_CONFIG = ROOT / "config/ember_semantic_v0.0.33.json"


def file_hashes(config: dict) -> dict:
    for path, expected in config["pinned_files"].items():
        if base.data_check.digest(ROOT / path) != expected:
            raise ValueError(f"pinned source changed: {path}")
    paths = [*config["pinned_files"], "jobs/ember_semantic_canary_v033.py",
             "config/ember_semantic_v0.0.33.json"]
    return {p: base.data_check.digest(ROOT / p) for p in paths}


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if (path.resolve() != DEFAULT_CONFIG.resolve() or config["version"] != "0.0.33"
            or config["training_authorized"] is not True
            or config["gpu_training_authorized"] is not False
            or config["production_authorized"] is not False):
        raise ValueError("unsupported CPU experiment configuration")
    previous = json.loads(base.DEFAULT_CONFIG.read_text())
    if (config["canary"]["gate"] != previous["canary"]["gate"]
            or config["dataset"] != previous["dataset"]):
        raise ValueError("the retry must retain the existing gates and dataset")
    file_hashes(config)
    return config


def exact_counts(result: dict, group: str) -> str:
    rows = result[group]
    return f"{sum(bool(r['exact']) for r in rows)}/{len(rows)}"


def summary_markdown(report: dict) -> str:
    lines = [f"# Ember v0.0.33 CPU canary: {report['status']}", "",
             f"Run: `{report['run_id']}`. CPU only; no GPU job is submitted.", ""]
    if "learning_gate" not in report:
        lines.append("The measured learning report is incomplete; see the saved report and job logs.")
        return "\n".join(lines) + "\n"
    before, after = report["before"], report["after"]
    original, candidate = report["baseline_copy"], report["candidate_copy"]
    rows = [
        ("Training loss", f"{before['train_loss']:.5f}", f"{after['train_loss']:.5f}"),
        ("Development loss", f"{before['development_loss']:.5f}", f"{after['development_loss']:.5f}"),
    ]
    for label, key in (("Training exact answers", "train_probe"), ("Development exact answers", "development_probe")):
        rows.append((label, f"{before[key]['passed']}/{before[key]['total']}",
                     f"{after[key]['passed']}/{after[key]['total']}"))
    for label, key in (("Legacy exact copies", "legacy_cases"), ("Expanded exact copies", "expanded_cases")):
        rows.append((label, exact_counts(original, key), exact_counts(candidate, key)))
    lines.extend(["| Measurement | Source | Selected checkpoint |", "| --- | --- | --- |"])
    lines.extend(f"| {label} | {left} | {right} |" for label, left, right in rows)
    failed = [key for key, passed in report["learning_gate"]["checks"].items() if not passed]
    lines.extend(["", "Failed learning checks: " + (", ".join(f"`{key}`" for key in failed) or "none") + ".", "",
                  "| Step | Development loss | Legacy copies | Expanded copies | Copy preserved |",
                  "| --- | --- | --- | --- | --- |"])
    for entry in report["checkpoint_reports"]:
        result = entry["copy"]
        lines.append(f"| {entry['step']} | {entry['development_loss']:.5f} | "
                     f"{exact_counts(result, 'legacy_cases')} | {exact_counts(result, 'expanded_cases')} | "
                     f"{'PASS' if entry['copy_protection']['passed'] else 'FAIL'} |")
    lines.extend(["", f"Selected step: {report['best_step']}, using development loss only.",
                  "These are development probes, not a held-out semantic promotion or production approval."])
    return "\n".join(lines) + "\n"


def train(model, train_set, replay_set, dev_set, tokenizer, baseline_copy, source,
          config, identity, run_id, output, api, repo, torch):
    """Keep the v0.0.32 optimizer, adding recorded copy checks without selecting on them."""
    import trackio
    settings = config["canary"]
    model.to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.95), weight_decay=0.01)
    rng = random.Random(config["seed"])
    best_loss, best_step = float("inf"), 0
    history, checkpoints, seen_replay = [], [], set()
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    for step in range(1, settings["max_steps"] + 1):
        if time.monotonic() - start > settings["training_time_limit_seconds"]:
            raise TimeoutError("bounded CPU training and checkpoint-check time limit reached")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = max(0.0, (step - settings["warmup_steps"]) / (settings["max_steps"] - settings["warmup_steps"]))
        scale = min(1.0, step / settings["warmup_steps"]) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = config["learning_rate"] * scale
        losses = []
        for name, dataset, batch_size, weight in (
            ("semantic", train_set, settings["semantic_batch_size"], 1 - config["copy_loss_fraction"]),
            ("copy", replay_set, settings["copy_batch_size"], config["copy_loss_fraction"]),
        ):
            indices = [rng.randrange(len(dataset)) for _ in range(batch_size)]
            if name == "copy":
                seen_replay.update(indices)
            x, y = base.stack_batch(dataset, indices, "cpu", torch)
            _, loss = model(x, y)
            if loss is None or not bool(torch.isfinite(loss)):
                raise ValueError("non-finite training loss")
            (weight * loss).backward()
            losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        metric = {"step": step, "semantic_loss": losses[0], "copy_loss": losses[1],
                  "learning_rate": optimizer.param_groups[0]["lr"], "gradient_norm": float(norm),
                  "copy_replay_examples_seen": len(seen_replay)}
        history.append(metric)
        if step % settings["eval_interval"] == 0 or step == settings["max_steps"]:
            loss = base.evaluate_loss(model, dev_set, "cpu", torch)
            metric["development_loss"] = loss
            path = output / "checkpoints" / f"step-{step:04d}.pt"
            base.save_checkpoint(path, model, optimizer, source, config, identity, run_id, step, rng, torch)
            if loss < best_loss:
                best_loss, best_step = loss, step
                shutil.copyfile(path, output / "best.pt")
            copy_result = base.copy_diagnostic(model, tokenizer, torch)
            protection = base.copy_protection(baseline_copy, copy_result)
            checkpoints.append({"step": step, "checkpoint_path": str(path.relative_to(output)),
                                "checkpoint_sha256": base.data_check.digest(path), "development_loss": loss,
                                "copy": copy_result, "copy_protection": protection})
            metric.update(copy_preserved=int(protection["passed"]),
                          legacy_exact_copy_rate=copy_result["metrics"]["exact_copy_rate"],
                          expanded_exact_copy_rate=copy_result["metrics"]["expanded_exact_copy_rate"])
            base.write_json(output / "history.json", history)
            base.write_json(output / "checkpoint-reports.json", checkpoints)
            print(json.dumps({"event": "checkpoint", **metric, "best_step": best_step,
                              "failed_copy_checks": [k for k, passed in protection["checks"].items() if not passed]}), flush=True)
            api.upload_folder(repo_id=repo, folder_path=str(output), path_in_repo=f"runs/{run_id}",
                              allow_patterns=["best.pt", "checkpoints/*.pt", "checkpoint-reports.json", "history.json", "identity.json", "config.json"],
                              commit_message=f"Save CPU checkpoint and copy evidence at step {step}")
        trackio.log(metric, step=step)
    del optimizer
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state"])
    return {"best_step": best_step, "history": history, "checkpoint_reports": checkpoints,
            "replay_seen_indices": sorted(seen_replay)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("v033-results"))
    args = parser.parse_args()
    config = load_config(args.config)
    import torch
    from huggingface_hub import HfApi
    if not os.environ.get("HF_TOKEN", "").strip():
        raise ValueError("HF_TOKEN is required for private checkpoint persistence")
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TRACKIO_DIR"] = str(output / "trackio")
    import trackio
    run_id = "ember-v033-canary-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    identity = {"files": file_hashes(config), "dataset": config["dataset"], "version": config["version"]}
    report = {"schema_version": 1, "version": "0.0.33", "run_id": run_id, "mode": "canary",
              "identity": identity, "status": "ERROR", "production_authorized": False, "gpu_training_authorized": False,
              "code_commit": os.environ.get("GITHUB_SHA"), "runtime": {"torch": torch.__version__, "device": "cpu"}}
    api = HfApi(token=os.environ["HF_TOKEN"])
    repo = config["canary"]["output_repo"]
    api.create_repo(repo, private=True, exist_ok=True)
    if not api.repo_info(repo).private:
        raise ValueError("output model repository must be private")
    base.write_json(output / "identity.json", identity)
    base.write_json(output / "config.json", config)
    api.upload_file(repo_id=repo, path_or_fileobj=str(output / "identity.json"), path_in_repo=f"runs/{run_id}/identity.json",
                    commit_message="Verify private CPU experiment persistence before training")
    trackio.init(project="ember-semantic-v033", name=run_id,
                 config={"mode": "canary", "learning_rate": config["learning_rate"], "copy_loss_fraction": config["copy_loss_fraction"]})
    try:
        with tempfile.TemporaryDirectory(prefix="ember-v033-") as td:
            model, tokenizer, source, splits, source_cfg = base.load_inputs(config, Path(td), torch)
            report["source"] = source_cfg
            train_rows = base.select_pairs(splits["train"], config["canary"]["pairs_per_family"], config["seed"])
            dev_rows = base.select_pairs(splits["validation"], 1, config["seed"] + 1)
            replay = base.replay_rows(3600, config["canary"]["copy_replay_per_kind"])
            report["rows"] = {"training": len(train_rows), "development_loss": len(dev_rows), "copy_replay": len(replay),
                              "train_probe": len(train_rows), "development_probe": len(dev_rows)}
            report["selected_ids"] = {"train_probe": [r["id"] for r in train_rows], "development_probe": [r["id"] for r in dev_rows],
                                      "copy_replay": [r["id"] for r in replay]}
            eos = base.semantic_gate.token_contract(tokenizer)["eos_id"]
            def encode(rows):
                return [base.data_check.encode_row(tokenizer, r, config["block_size"], config["generation_budget"], eos, torch)[0] for r in rows]
            train_set, dev_set, replay_set = encode(train_rows), encode(dev_rows), encode(replay)
            def measure():
                return {"train_loss": base.evaluate_loss(model, train_set, "cpu", torch),
                        "development_loss": base.evaluate_loss(model, dev_set, "cpu", torch),
                        "train_probe": base.generate_probe(model, tokenizer, train_rows, torch, config["generation_budget"]),
                        "development_probe": base.generate_probe(model, tokenizer, dev_rows, torch, config["generation_budget"])}
            baseline_copy = base.copy_diagnostic(model, tokenizer, torch)
            report["baseline_copy"] = baseline_copy
            if not baseline_copy["passed"]:
                raise ValueError("source no longer reproduces its existing copy gate")
            before = measure()
            report["before"] = before
            base.write_json(output / "report.json", report)
            print(json.dumps({"event": "baseline", "train_loss": before["train_loss"], "development_loss": before["development_loss"],
                              "train_exact": before["train_probe"]["passed"], "development_exact": before["development_probe"]["passed"]}), flush=True)
            trained = train(model, train_set, replay_set, dev_set, tokenizer, baseline_copy, source,
                            config, identity, run_id, output, api, repo, torch)
            report.update(best_step=trained["best_step"], optimizer_steps=len(trained["history"]),
                          checkpoint_reports=trained["checkpoint_reports"],
                          replay_seen_ids=[replay[i]["id"] for i in trained["replay_seen_indices"]])
            report["model_state_changed"] = any(not torch.equal(v.cpu(), source["model_state"][k]) for k, v in model.state_dict().items())
            if not report["model_state_changed"]:
                raise ValueError("training did not change model weights")
            after = measure()
            candidate_copy = base.copy_diagnostic(model, tokenizer, torch)
            selected = next(c for c in trained["checkpoint_reports"] if c["step"] == trained["best_step"])
            if (selected["checkpoint_sha256"] != base.data_check.digest(output / "best.pt")
                    or selected["copy"] != candidate_copy):
                raise ValueError("selected checkpoint copy evidence did not reproduce")
            protection = base.copy_protection(baseline_copy, candidate_copy)
            learning = base.learning_gate(before, after, protection, config["canary"]["gate"])
            report.update(after=after, candidate_copy=candidate_copy, copy_protection=protection, learning_gate=learning,
                          status="PASS" if learning["passed"] else "FAIL",
                          meaning="CPU replay learning canary; no held-out semantic promotion, GPU submission, or production approval.")
    except Exception as error:
        report["status"] = "ERROR"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        try:
            trackio.finish()
        except Exception as error:
            report["telemetry_error"] = {"type": type(error).__name__, "message": str(error)}
        if (output / "best.pt").exists():
            report["best_checkpoint_sha256"] = base.data_check.digest(output / "best.pt")
        base.write_json(output / "report.json", report)
        (output / "summary.md").write_text(summary_markdown(report))
        commit = api.upload_folder(repo_id=repo, folder_path=str(output), path_in_repo=f"runs/{run_id}",
                                   ignore_patterns=["publication.json"], commit_message=f"Save CPU canary result: {report['status']}")
        publication = {"repo_id": repo, "revision": commit.oid, "run_id": run_id,
                       "report_path": f"runs/{run_id}/report.json", "status": report["status"], "url": str(commit.commit_url)}
        base.write_json(output / "publication.json", publication)
        print(json.dumps({"event": "complete", **publication,
                          "failed_learning_checks": [k for k, passed in report.get("learning_gate", {}).get("checks", {}).items() if not passed]}), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
