"""Bounded semantic learning experiment for Ember's native PyTorch model.

CPU canary first; a paid run requires its immutable passing report. Development
loss selects checkpoints. The frozen semantic gate is used only after training.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from jobs import ember_sft_data_semantic_v1 as semantic_data
from jobs import ember_semantic_data_preflight as data_check
from jobs import ember_semantic_gate as semantic_gate
from jobs import ember_sft_data_v026 as copy_data

DEFAULT_CONFIG = ROOT / "config/ember_semantic_v0.0.32.json"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def file_hashes(config: dict) -> dict:
    for path, expected in config["pinned_files"].items():
        if data_check.digest(ROOT / path) != expected:
            raise ValueError(f"pinned source changed: {path}")
    paths = [*config["pinned_files"], "jobs/ember_semantic_train_v032.py", "config/ember_semantic_v0.0.32.json"]
    return {p: data_check.digest(ROOT / p) for p in paths}


def select_pairs(rows: list[dict], pairs_per_family: int, seed: int) -> list[dict]:
    families = defaultdict(lambda: defaultdict(list))
    for row in rows:
        families[row["family"]][row["pair_id"]].append(row)
    chosen = []
    rng = random.Random(seed)
    for family in sorted(families):
        pairs = sorted(families[family])
        rng.shuffle(pairs)
        for key in pairs[:pairs_per_family]:
            pair = families[family][key]
            if len(pair) != 2 or {r["pair_side"] for r in pair} != {0, 1}:
                raise ValueError("incomplete counterfactual pair")
            chosen.extend(sorted(pair, key=lambda r: r["id"]))
    if len(chosen) != len(semantic_data.FAMILIES) * pairs_per_family * 2:
        raise ValueError("not enough balanced semantic pairs")
    return chosen


def replay_rows(total: int, per_kind: int | None = None) -> list[dict]:
    train = copy_data.build_examples("train", total)
    validation = copy_data.build_examples("validation", 450)
    copy_data.assert_clean(train, validation)
    if per_kind is not None:
        selected = []
        for kind in copy_data.KINDS:
            selected.extend([r for r in train if r["kind"] == kind][:per_kind])
        train = selected
    # Historical generator stays byte-identical; the training adapter ends at EOS.
    return [{**r, "completion": r["completion"].rstrip()} for r in train]


def fixture_case(row: dict) -> dict:
    case = {"id": row["id"], "kind": row["kind"]}
    if row["kind"] == "tool_call":
        call = semantic_gate.strict_json(row["completion"].removeprefix(semantic_data.TOOL).split(semantic_data.EOT)[0].strip())
        case.update(expected_tool=call["name"], argument_variants=[call["arguments"]],
                    arguments_schema={"type": "object", "properties": {k: {"type": "string"} for k in call["arguments"]},
                                      "required": list(call["arguments"]), "additionalProperties": False})
    else:
        case["accepted_responses"] = [row["completion"].split(semantic_data.EOT)[0].strip()]
    return case


def generate_probe(model, tokenizer, rows: list[dict], torch, budget: int) -> dict:
    model.to("cpu").eval()
    output, counts = [], dict.fromkeys(semantic_data.KINDS, 0)
    for row in rows:
        generated = semantic_gate.generate_completion(model, tokenizer, torch, row["prompt"], budget)
        score = semantic_gate.score_case(fixture_case(row), generated["completion"])
        counts[row["kind"]] += int(score["passed"])
        output.append({"id": row["id"], "family": row["family"], "prompt": row["prompt"],
                       "expected": row["completion"], **generated, "score": score})
    return {"total": len(rows), "passed": sum(counts.values()), "by_kind": counts, "cases": output}


def copy_diagnostic(model, tokenizer, torch) -> dict:
    # Reuse the actual historical diagnostic and thresholds, without executing
    # its remote training launcher or changing the old source files.
    from jobs import ember_hf_sft_v015 as base
    from jobs import ember_hf_sft_v016 as continuation
    from jobs import ember_hf_sft_v030 as prior
    scope = {"diagnostic": continuation.diagnostic,
             "continuation_top1_counts": continuation.continuation_top1_counts}
    exec(compile(prior.HELPERS, "pinned_v030_copy_diagnostics", "exec"), scope)
    cfg = json.loads((ROOT / "config/ember_multi_position_v0.0.31.json").read_text())
    model.to("cpu").eval()
    result = scope["v030_diagnostic"](model, tokenizer, copy_data, cfg, "cpu", torch, base)
    model.eval()
    return result


def copy_protection(baseline: dict, candidate: dict) -> dict:
    checks = {"existing_copy_gate": candidate["passed"] is True}
    for group in ("legacy_cases", "expanded_cases"):
        good = {r["value"] for r in baseline[group] if r["exact"] and r["clean_stop"]}
        retained = {r["value"] for r in candidate[group] if r["exact"] and r["clean_stop"]}
        checks[group + "_retained"] = good <= retained
    for key in ("exact_copy_rate", "continuation_top1_rate", "clean_stop_rate",
                "expanded_exact_copy_rate", "expanded_continuation_top1_rate", "expanded_clean_stop_rate"):
        checks[key + "_non_regression"] = candidate["metrics"][key] + 1e-9 >= baseline["metrics"][key]
    return {"passed": all(checks.values()), "checks": checks}


def learning_gate(before: dict, after: dict, protection: dict, thresholds: dict) -> dict:
    checks = {
        "train_loss_improved": after["train_loss"] <= before["train_loss"] * (1 - thresholds["minimum_train_loss_reduction"]),
        "development_loss_improved": after["development_loss"] < before["development_loss"],
        "train_exact_gain": after["train_probe"]["passed"] - before["train_probe"]["passed"] >= thresholds["minimum_train_exact_gain"],
        "development_exact_gain": after["development_probe"]["passed"] - before["development_probe"]["passed"] >= thresholds["minimum_development_exact_gain"],
        "copy_preserved": protection["passed"] is True,
    }
    for kind in ("tool_call", "tool_result_response"):
        checks[kind + "_train_gain"] = after["train_probe"]["by_kind"][kind] > before["train_probe"]["by_kind"][kind]
    for kind in semantic_data.KINDS:
        checks[kind + "_development_non_regression"] = after["development_probe"]["by_kind"][kind] >= before["development_probe"]["by_kind"][kind]
    return {"passed": all(checks.values()), "checks": checks}


def validate_canary(report: dict, identity: dict, thresholds: dict) -> None:
    if report.get("mode") != "canary" or report.get("status") != "PASS" or report.get("learning_gate", {}).get("passed") is not True:
        raise ValueError("GPU training requires a passing CPU learning canary")
    if report.get("identity") != identity or report.get("best_step", 0) <= 0:
        raise ValueError("canary does not match this source/data/trainer/config")
    if not report.get("learning_gate", {}).get("checks") or not all(report["learning_gate"]["checks"].values()):
        raise ValueError("canary has missing or failed checks")
    protection = copy_protection(report["baseline_copy"], report["candidate_copy"])
    computed = learning_gate(report["before"], report["after"], protection, thresholds)
    if not computed["passed"] or computed != report["learning_gate"] or report.get("model_state_changed") is not True:
        raise ValueError("canary PASS does not agree with its measured evidence")


def stack_batch(encoded: list, indices: list[int], device: str, torch):
    x = torch.stack([encoded[i][0] for i in indices])
    y = torch.stack([encoded[i][1] for i in indices])
    # Remove only ignored right padding. Causal attention cannot depend on it.
    end = int(y.ne(-100).nonzero()[:, 1].max()) + 1
    return x[:, :end].to(device), y[:, :end].to(device)


def evaluate_loss(model, encoded: list, device: str, torch, batch_size: int = 8) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            x, y = stack_batch(encoded, list(range(start, min(start + batch_size, len(encoded)))), device, torch)
            _, loss = model(x, y)
            tokens = int(y.ne(-100).sum())
            if loss is None or not bool(torch.isfinite(loss)):
                raise ValueError("non-finite evaluation loss")
            total += float(loss) * tokens
            count += tokens
    if count == 0:
        raise ValueError("no supervised evaluation tokens")
    return total / count


def cpu_state(value, torch):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_state(v, torch) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_state(v, torch) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(v, torch) for v in value)
    return value


def save_checkpoint(path, model, optimizer, source, config, identity, run_id, step, rng, torch):
    torch.save({"format": "ember-checkpoint-v1", "run_id": run_id, "step": step,
                "model_config": source["model_config"], "tokenizer": source["tokenizer"],
                "model_state": cpu_state(model.state_dict(), torch),
                "optimizer_state": cpu_state(optimizer.state_dict(), torch),
                "rng_state": rng.getstate(), "torch_rng_state": torch.get_rng_state(),
                "train_config": config, "source_identity": identity, "production_authorized": False}, path)


def train(model, train_set, replay_set, dev_set, source, config, mode, identity, run_id, output, api, repo, torch):
    import trackio
    settings = config[mode]
    device = "cpu" if mode == "canary" else "cuda"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.95), weight_decay=0.01)
    rng = random.Random(config["seed"])
    best_loss, best_step = float("inf"), 0
    history = []
    start = time.monotonic()
    for step in range(1, settings["max_steps"] + 1):
        if time.monotonic() - start > settings["training_time_limit_seconds"]:
            raise TimeoutError("bounded training time limit reached")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = max(0.0, (step - settings["warmup_steps"]) / (settings["max_steps"] - settings["warmup_steps"]))
        scale = min(1.0, step / settings["warmup_steps"]) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = config["learning_rate"] * scale
        losses = []
        for dataset, batch_size, weight in ((train_set, settings["semantic_batch_size"], 1 - config["copy_loss_fraction"]),
                                             (replay_set, settings["copy_batch_size"], config["copy_loss_fraction"])):
            x, y = stack_batch(dataset, [rng.randrange(len(dataset)) for _ in range(batch_size)], device, torch)
            _, loss = model(x, y)
            if loss is None or not bool(torch.isfinite(loss)):
                raise ValueError("non-finite training loss")
            (weight * loss).backward()
            losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        metric = {"step": step, "semantic_loss": losses[0], "copy_loss": losses[1],
                  "learning_rate": optimizer.param_groups[0]["lr"], "gradient_norm": float(norm)}
        trackio.log(metric, step=step)
        history.append(metric)
        if step % settings["eval_interval"] == 0 or step == settings["max_steps"]:
            loss = evaluate_loss(model, dev_set, device, torch)
            metric["development_loss"] = loss
            if loss < best_loss:
                best_loss, best_step = loss, step
                save_checkpoint(output / "best.pt", model, optimizer, source, config, identity, run_id, step, rng, torch)
            save_checkpoint(output / "latest.pt", model, optimizer, source, config, identity, run_id, step, rng, torch)
            write_json(output / "history.json", history)
            print(json.dumps({"event": "validation", **metric, "best_step": best_step}), flush=True)
            api.upload_folder(repo_id=repo, folder_path=str(output), path_in_repo=f"runs/{run_id}",
                              allow_patterns=["best.pt", "latest.pt", "history.json", "identity.json", "config.json"],
                              commit_message=f"Save {mode} progress at optimizer step {step}")
    del optimizer
    model.to("cpu")
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state"])
    del best
    return best_step, history


def load_inputs(config: dict, work: Path, torch):
    from huggingface_hub import hf_hub_download
    token = os.environ["HF_TOKEN"]
    source_cfg = json.loads((ROOT / "config/ember_semantic_data_v1.json").read_text())["reference_model"]
    path = Path(hf_hub_download(source_cfg["repo_id"], filename=source_cfg["checkpoint_path"],
                               revision=source_cfg["revision"], token=token, local_dir=work / "source-model"))
    if data_check.digest(path) != source_cfg["checkpoint_sha256"]:
        raise ValueError("reference checkpoint hash mismatch")
    source = torch.load(path, map_location="cpu", weights_only=False)
    if source["step"] != 479 or source["train_config"]["version"] != "0.0.31":
        raise ValueError("unexpected source identity")
    splits = {}
    for split, sha in config["dataset"]["files"].items():
        file = Path(hf_hub_download(config["dataset"]["repo_id"], repo_type="dataset", revision=config["dataset"]["revision"],
                                   filename=config["dataset"]["prefix"] + "/" + split + ".jsonl", token=token,
                                   local_dir=work / "dataset"))
        if data_check.digest(file) != sha:
            raise ValueError("repaired dataset hash mismatch")
        splits[split] = [json.loads(line) for line in file.read_text().splitlines()]
    semantic_data.validate_dataset(splits)
    data_check.check_separation(splits, json.loads((ROOT / "config/ember_semantic_v1.json").read_text()))
    archive = ROOT / "ember-v0.0.7-hf-ready.zip"
    if data_check.digest(archive) != data_check.PACKAGE_SHA256:
        raise ValueError("model implementation archive changed")
    with zipfile.ZipFile(archive) as package:
        package.extractall(work / "package")
    sys.path.insert(0, str(work / "package/ember"))
    from src.model import EmberGPT, ModelConfig
    from src.tokenizer import tokenizer_from_state_dict
    tokenizer = tokenizer_from_state_dict(source["tokenizer"])
    model = EmberGPT(ModelConfig(**source["model_config"]))
    model.load_state_dict(source["model_state"])
    return model, tokenizer, source, splits, source_cfg


def evaluate_int4(model, tokenizer, output, torch, baseline_copy):
    from src.quantize_int4 import export_int4_checkpoint, dequantize_tensor
    export_int4_checkpoint(str(output / "best.pt"), str(output / "best.int4.pt"))
    quantized = torch.load(output / "best.int4.pt", map_location="cpu", weights_only=False)
    restored = {k: dequantize_tensor(v) for k, v in quantized["quantized_state"].items()}
    restored.update(quantized.get("passthrough_state", {}))
    model.load_state_dict(restored)
    spec = json.loads((ROOT / "config/ember_semantic_v1.json").read_text())
    result = semantic_gate.evaluate_model(model, tokenizer, torch, spec, "int4")
    copy = copy_diagnostic(model, tokenizer, torch)
    return {"bytes": (output / "best.int4.pt").stat().st_size(), "sha256": data_check.digest(output / "best.int4.pt"),
            "semantic": result, "copy": copy, "copy_protection": copy_protection(baseline_copy, copy)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("canary", "train"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("v032-results"))
    parser.add_argument("--canary-revision")
    parser.add_argument("--canary-report")
    args = parser.parse_args()
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    config = json.loads(args.config.read_text())
    if args.config.resolve() != DEFAULT_CONFIG.resolve() or config["version"] != "0.0.32" or not config["training_authorized"] or config["production_authorized"]:
        raise ValueError("unsupported experiment configuration")
    if args.mode == "train" and not torch.cuda.is_available():
        raise ValueError("the bounded training mode requires its allocated GPU")
    if not os.environ.get("HF_TOKEN", "").strip():
        raise ValueError("HF_TOKEN is required for private checkpoint persistence")
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TRACKIO_DIR"] = str(output / "trackio")
    import trackio
    run_id = "ember-v032-" + args.mode + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    api = HfApi(token=os.environ["HF_TOKEN"])
    repo = config[args.mode]["output_repo"]
    identity = {"files": file_hashes(config), "dataset": config["dataset"], "version": config["version"]}
    report = {"schema_version": 1, "version": "0.0.32", "run_id": run_id, "mode": args.mode,
              "identity": identity, "status": "ERROR", "production_authorized": False,
              "code_commit": os.environ.get("GITHUB_SHA"), "runtime": {"torch": torch.__version__, "device": "cpu" if args.mode == "canary" else "cuda"}}
    if args.mode == "train":
        if not args.canary_revision or not re.fullmatch(r"[a-f0-9]{40}", args.canary_revision) or not args.canary_report:
            raise ValueError("GPU training requires an immutable CPU canary report")
        path = hf_hub_download(config["canary"]["output_repo"], filename=args.canary_report,
                               revision=args.canary_revision, token=os.environ["HF_TOKEN"])
        validate_canary(json.loads(Path(path).read_text()), identity, config["canary"]["gate"])
        report["canary_evidence"] = {"revision": args.canary_revision, "report": args.canary_report}
    api.create_repo(repo, private=True, exist_ok=True)
    if not api.repo_info(repo).private:
        raise ValueError("output model repository must be private")
    write_json(output / "identity.json", identity)
    write_json(output / "config.json", config)
    api.upload_file(repo_id=repo, path_or_fileobj=str(output / "identity.json"), path_in_repo=f"runs/{run_id}/identity.json",
                    commit_message="Verify private experiment persistence before training")
    trackio.init(project="ember-semantic-v032", name=run_id, config={"mode": args.mode, "learning_rate": config["learning_rate"]})
    try:
        with tempfile.TemporaryDirectory(prefix="ember-v032-") as td:
            model, tokenizer, source, splits, source_cfg = load_inputs(config, Path(td), torch)
            report["source"] = source_cfg
            canary_train = select_pairs(splits["train"], config["canary"]["pairs_per_family"], config["seed"])
            dev_probe = select_pairs(splits["validation"], 1, config["seed"] + 1)
            train_rows = canary_train if args.mode == "canary" else splits["train"]
            dev_rows = dev_probe if args.mode == "canary" else splits["validation"]
            replay = replay_rows(3600, 4 if args.mode == "canary" else None)
            report["rows"] = {"training": len(train_rows), "development_loss": len(dev_rows), "copy_replay": len(replay),
                              "train_probe": len(canary_train), "development_probe": len(dev_probe)}
            report["selected_ids"] = {"train_probe": [r["id"] for r in canary_train], "development_probe": [r["id"] for r in dev_probe],
                                      "copy_replay": [r["id"] for r in replay]}
            eos = semantic_gate.token_contract(tokenizer)["eos_id"]
            def encode(rows):
                return [data_check.encode_row(tokenizer, r, config["block_size"], config["generation_budget"], eos, torch)[0] for r in rows]
            train_set, dev_set, replay_set, probe_set = encode(train_rows), encode(dev_rows), encode(replay), encode(canary_train)
            def measure():
                model.to("cpu").eval()
                return {"train_loss": evaluate_loss(model, probe_set, "cpu", torch),
                        "development_loss": evaluate_loss(model, dev_set, "cpu", torch),
                        "train_probe": generate_probe(model, tokenizer, canary_train, torch, config["generation_budget"]),
                        "development_probe": generate_probe(model, tokenizer, dev_probe, torch, config["generation_budget"])}
            baseline_copy = copy_diagnostic(model, tokenizer, torch)
            if not baseline_copy["passed"]:
                raise ValueError("source no longer reproduces its existing copy gate")
            before = measure()
            report.update(before=before, baseline_copy=baseline_copy)
            write_json(output / "report.json", report)
            print(json.dumps({"event": "baseline", "train_loss": before["train_loss"], "development_loss": before["development_loss"],
                              "train_exact": before["train_probe"]["passed"], "development_exact": before["development_probe"]["passed"]}), flush=True)
            best_step, history = train(model, train_set, replay_set, dev_set, source, config, args.mode, identity, run_id, output, api, repo, torch)
            report["best_step"], report["optimizer_steps"] = best_step, len(history)
            report["model_state_changed"] = any(not torch.equal(v.cpu(), source["model_state"][k]) for k, v in model.state_dict().items())
            if not report["model_state_changed"]:
                raise ValueError("training did not change model weights")
            after = measure()
            candidate_copy = copy_diagnostic(model, tokenizer, torch)
            protection = copy_protection(baseline_copy, candidate_copy)
            learning = learning_gate(before, after, protection, config["canary"]["gate"])
            report.update(after=after, candidate_copy=candidate_copy, copy_protection=protection, learning_gate=learning)
            if args.mode == "canary":
                report["status"] = "PASS" if learning["passed"] else "FAIL"
                report["meaning"] = "A small learning/readiness canary, not a held-out semantic promotion or production approval."
            else:
                spec = json.loads((ROOT / "config/ember_semantic_v1.json").read_text())
                full = semantic_gate.evaluate_model(model.to("cpu"), tokenizer, torch, spec, "full")
                report["semantic_full"] = full
                report["int4"] = evaluate_int4(model, tokenizer, output, torch, baseline_copy)
                passed = (full["summary"]["semantic_gate_pass"] and report["int4"]["semantic"]["summary"]["semantic_gate_pass"]
                          and protection["passed"] and report["int4"]["copy_protection"]["passed"])
                report["status"] = "PASS" if passed else "FAIL"
                report["meaning"] = "Experimental candidate evaluated with the unchanged semantic and copy gates; no deployment."
    except Exception as error:
        report["status"] = "ERROR"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        trackio.finish()
        write_json(output / "report.json", report)
        if (output / "best.pt").exists():
            report["best_checkpoint_sha256"] = data_check.digest(output / "best.pt")
            write_json(output / "report.json", report)
        commit = api.upload_folder(repo_id=repo, folder_path=str(output), path_in_repo=f"runs/{run_id}",
                                    ignore_patterns=["publication.json"], commit_message=f"Save {args.mode} result: {report['status']}")
        publication = {"repo_id": repo, "revision": commit.oid, "run_id": run_id,
                       "report_path": f"runs/{run_id}/report.json", "status": report["status"], "url": str(commit.commit_url)}
        write_json(output / "publication.json", publication)
        print(json.dumps({"event": "complete", **publication}), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
