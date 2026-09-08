"""Build and validate semantic repair data on CPU; never update model weights."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile
import unicodedata
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from jobs import ember_sft_data_semantic_v1 as data

PACKAGE_SHA256 = "27e8f7c80317652a22b3d58a0bd474724491a685dfe9e20c0b997b7c5907a289"
DEFAULT_CONFIG = ROOT / "config/ember_semantic_data_v1.json"
ROLE_PATTERN = re.compile(r"<\|(system|user|assistant|tool_result)\|>")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def parse_prompt(prompt: str) -> list[dict]:
    pieces = ROLE_PATTERN.split(prompt)
    return [{"role": pieces[i], "content": pieces[i + 1].strip()}
            for i in range(1, len(pieces), 2) if pieces[i + 1].strip()]


def input_signature(messages: list[dict]) -> tuple:
    result = []
    for message in messages:
        role, content = message["role"], message["content"].strip()
        if role == "system":
            continue
        if role == "tool_result" or (role == "assistant" and content.startswith(data.TOOL)):
            payload = content.removeprefix(data.TOOL).strip()
            content = data.compact(json.loads(payload))
        result.append((role, normalized(content)))
    return tuple(result)


def argument_key(tool: str, arguments: dict) -> tuple:
    # Expressions differing only in spaces and case/space variants of strings
    # count as the same reserved benchmark argument, irrespective of key order.
    return tool, tuple(sorted((key, re.sub(r"\s+", "", normalized(value)) if key == "expression"
                              else normalized(value)) for key, value in arguments.items()))


def check_separation(splits: dict, spec: dict) -> dict:
    cases = [parse_prompt(case["prompt"]) for case in spec["cases"]]
    reserved_inputs = {input_signature(messages) for messages in cases}
    reserved_users = {normalized(m["content"]) for messages in cases for m in messages if m["role"] == "user"}
    reserved_arguments = set()
    for case, messages in zip(spec["cases"], cases):
        for arguments in case.get("argument_variants", []):
            reserved_arguments.add(argument_key(case["expected_tool"], arguments))
        for message in messages:
            if message["role"] == "assistant" and message["content"].startswith(data.TOOL):
                call = json.loads(message["content"].removeprefix(data.TOOL).strip())
                reserved_arguments.add(argument_key(call["name"], call["arguments"]))
    signatures = set()
    for split, rows in splits.items():
        for row in rows:
            signature = input_signature(row["messages"])
            user = normalized(row["messages"][1]["content"])
            if signature in reserved_inputs or user in reserved_users:
                raise ValueError(f"frozen benchmark input reused: {row['id']}")
            if signature in signatures:
                raise ValueError(f"duplicate normalized input within/across splits: {row['id']}")
            signatures.add(signature)
            calls = ([row["completion"]] if row["kind"] == "tool_call" else
                     [m["content"] for m in row["messages"] if m["role"] == "assistant"])
            for content in calls:
                call = json.loads(content.removeprefix(data.TOOL).split(data.EOT)[0].strip())
                if argument_key(call["name"], call["arguments"]) in reserved_arguments:
                    raise ValueError(f"reserved benchmark tool arguments reused: {row['id']}")
    return {"status": "PASS", "normalized_unique_inputs": len(signatures),
            "reserved_benchmark_cases": len(cases), "reserved_argument_sets": len(reserved_arguments),
            "benchmark_input_matches": 0, "benchmark_argument_matches": 0,
            "scope": "Known frozen fixture inputs and exact argument sets; shared concepts and answer labels are allowed."}


def verify_frozen(config: dict) -> dict:
    checked = {}
    for relative, key in (("config/ember_semantic_v1.json", "frozen_evaluation_spec_sha256"),
                          ("jobs/ember_semantic_gate.py", "frozen_evaluator_sha256")):
        actual = digest(ROOT / relative)
        if actual != config[key]:
            raise ValueError(f"frozen evaluation changed: {relative}")
        checked[relative] = actual
    return checked


def encode_row(tokenizer, row: dict, block_size: int, generation_budget: int, eos_id: int, torch):
    prompt_ids = tokenizer.encode(row["prompt"])
    ids = tokenizer.encode(row["prompt"] + row["completion"])
    if not prompt_ids or ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(f"unstable completion boundary: {row['id']}")
    if not len(prompt_ids) < len(ids) <= block_size + 1:
        raise ValueError(f"context overflow: {row['id']} has {len(ids)} tokens, limit {block_size + 1}")
    if len(prompt_ids) + generation_budget > block_size:
        raise ValueError(f"generation context overflow: {row['id']} needs {len(prompt_ids)} + {generation_budget}, limit {block_size}")
    if ids[-1] != eos_id or ids.count(eos_id) != 1 or len(ids) - len(prompt_ids) > generation_budget:
        raise ValueError(f"completion lacks one final EOS or exceeds generation budget: {row['id']}")
    x = torch.full((block_size,), eos_id, dtype=torch.long)
    y = torch.full((block_size,), -100, dtype=torch.long)
    x[:len(ids) - 1] = torch.tensor(ids[:-1], dtype=torch.long)
    start, end = len(prompt_ids) - 1, len(ids) - 1
    y[start:end] = torch.tensor(ids[len(prompt_ids):], dtype=torch.long)
    # Check the supervised sequence directly, including first completion and EOS.
    if y[y != -100].tolist() != ids[len(prompt_ids):] or int(y[end - 1]) != eos_id:
        raise ValueError("completion-only target alignment failed")
    if bool((y[:start] != -100).any()) or bool((y[end:] != -100).any()):
        raise ValueError("prompt or padding accidentally supervised")
    return (x, y), {"prompt": len(prompt_ids), "completion": len(ids) - len(prompt_ids), "total": len(ids)}


def lengths(values: list[int]) -> dict:
    return {"min": min(values), "median": statistics.median(values), "max": max(values), "sum": sum(values)}


def cpu_check(splits: dict, config: dict) -> dict:
    import torch
    from huggingface_hub import hf_hub_download
    from jobs.ember_semantic_gate import token_contract

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise ValueError("HF_TOKEN is required for the private reference checkpoint")
    source = config["reference_model"]
    if not re.fullmatch(r"[a-f0-9]{40}", source["revision"]):
        raise ValueError("reference revision must be an immutable commit")
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    archive = ROOT / "ember-v0.0.7-hf-ready.zip"
    if digest(archive) != PACKAGE_SHA256:
        raise ValueError("source package checksum mismatch")
    with tempfile.TemporaryDirectory(prefix="ember-data-cpu-") as td:
        work = Path(td)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "source")
        sys.path.insert(0, str(work / "source/ember"))
        from src.model import EmberGPT, ModelConfig
        from src.tokenizer import tokenizer_from_state_dict

        checkpoint_path = Path(hf_hub_download(repo_id=source["repo_id"], revision=source["revision"],
                                               filename=source["checkpoint_path"], token=token,
                                               local_dir=work / "checkpoint"))
        if digest(checkpoint_path) != source["checkpoint_sha256"]:
            raise ValueError("checkpoint checksum mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["step"] != source["step"] or checkpoint["train_config"]["version"] != source["version"]:
            raise ValueError("reference checkpoint identity mismatch")
        if checkpoint["model_config"]["block_size"] != config["block_size"]:
            raise ValueError("context size differs from reference checkpoint")
        tokenizer = tokenizer_from_state_dict(checkpoint["tokenizer"])
        contract = token_contract(tokenizer)
        encoded, token_stats = {}, {}
        for split, rows in splits.items():
            encoded[split], sizes = [], []
            for row in rows:
                item, size = encode_row(tokenizer, row, config["block_size"], config["generation_budget"], contract["eos_id"], torch)
                encoded[split].append(item)
                sizes.append(size)
            token_stats[split] = {key: lengths([size[key] for size in sizes]) for key in sizes[0]}

        model = EmberGPT(ModelConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        sample_losses = {}
        backward_count, finite_gradients = 0, 0
        for split, rows in splits.items():
            chosen = {}
            for i, row in enumerate(rows):
                chosen.setdefault(row["family"], i)
            family_losses = {}
            for family, index in sorted(chosen.items()):
                x, y = encoded[split][index]
                model.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(split == "train"):
                    _, loss = model(x.unsqueeze(0), y.unsqueeze(0))
                if loss is None or not bool(torch.isfinite(loss)):
                    raise ValueError(f"non-finite CPU loss for {family}")
                family_losses[family] = float(loss.detach())
                if split == "train":
                    loss.backward()
                    gradients = [p.grad for p in model.parameters() if p.grad is not None]
                    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                        raise ValueError(f"non-finite or missing CPU gradients for {family}")
                    finite_gradients += len(gradients)
                    backward_count += 1
            sample_losses[split] = family_losses
        if any(not torch.equal(value, checkpoint["model_state"][name]) for name, value in model.state_dict().items()):
            raise ValueError("CPU preflight changed model state")
        model.zero_grad(set_to_none=True)
        return {"status": "PASS", "device": "cpu", "reference_model": source,
                "tokenizer_sha256": hashlib.sha256(data.compact(checkpoint["tokenizer"]).encode()).hexdigest(),
                "vocab_size": tokenizer.vocab_size, "special_tokens": contract,
                "encoded_rows": sum(len(rows) for rows in splits.values()), "tokens": token_stats,
                "truncated_rows": 0, "prompt_loss_tokens": 0, "padding_loss_tokens": 0,
                "rows_with_supervised_final_eos": sum(len(rows) for rows in splits.values()),
                "backward_samples": backward_count, "finite_gradient_tensors_checked": finite_gradients,
                "sample_losses_one_per_family": sample_losses,
                "optimizer_steps": 0, "model_state_unchanged": True,
                "limitation": "48 forward samples / 24 backward samples check numerical compatibility, not model answer quality."}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def publish(output: Path, manifest: dict, repo_id: str) -> dict:
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(repo_id, repo_type="dataset")
    if not info.private:
        raise ValueError("refusing to publish to a public dataset repository")
    prefix = data.VERSION + "/" + manifest["dataset_content_sha256"]
    # Each run gets immutable data paths plus its own report; no shared latest pointer.
    run = os.environ.get("GITHUB_RUN_ID", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    destinations = {"train.jsonl": prefix + "/train.jsonl", "validation.jsonl": prefix + "/validation.jsonl",
                    "manifest.json": prefix + f"/runs/{run}/manifest.json",
                    "report.json": prefix + f"/runs/{run}/report.json", "README.md": prefix + "/README.md"}
    commit = api.create_commit(repo_id, repo_type="dataset", parent_commit=info.sha,
                               commit_message=f"Add CPU-validated {data.VERSION} data ({run})",
                               operations=[CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(output / filename))
                                           for filename, path in destinations.items()])
    return {"repo_id": repo_id, "repo_type": "dataset", "private": True, "revision": commit.oid,
            "paths": destinations, "commit_url": str(commit.commit_url)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("semantic-data-results"))
    parser.add_argument("--data-only", action="store_true", help="Skip tokenizer/model checks; cannot publish")
    parser.add_argument("--publish", action="store_true", help="Publish validated synthetic data to its private dataset repo")
    args = parser.parse_args()
    if args.data_only and args.publish:
        parser.error("publication requires the real checkpoint CPU preflight")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "FAIL", "dataset_version": data.VERSION, "training_run_started": False,
              "deployment_performed": False, "optimizer_steps": 0}
    try:
        config = json.loads(args.config.read_text())
        if config["dataset_version"] != data.VERSION or config["training_authorized"] or config["deployment_authorized"]:
            raise ValueError("configuration must select this dataset and prohibit training/deployment")
        if config["completion_only_loss"] is not True:
            raise ValueError("only completion-only loss is supported")
        frozen = verify_frozen(config)
        splits = data.build_dataset(config["seed"])
        for split, key in (("train", "train_examples"), ("validation", "validation_examples")):
            if len(splits[split]) != config[key]:
                raise ValueError("configured example count differs from generated data")
        report["data"] = data.validate_dataset(splits)
        spec = json.loads((ROOT / "config/ember_semantic_v1.json").read_text())
        report["separation"] = check_separation(splits, spec)
        repeated = data.build_dataset(config["seed"])
        for split, rows in splits.items():
            payload = data.jsonl_bytes(rows)
            if payload != data.jsonl_bytes(repeated[split]):
                raise ValueError("dataset build is not deterministic")
            (output / f"{split}.jsonl").write_bytes(payload)
        report["cpu"] = {"status": "NOT_RUN"} if args.data_only else cpu_check(splits, config)
        report["status"] = "DATA_ONLY_PASS" if args.data_only else "PASS"
        report["meaning"] = "Dataset validation only; this does not change the candidate's semantic FAIL."
        hashes = {name: digest(output / name) for name in ("train.jsonl", "validation.jsonl")}
        content_hash = hashlib.sha256(data.compact(hashes).encode()).hexdigest()
        manifest = {"schema_version": 1, "dataset_version": data.VERSION, "seed": config["seed"],
                    "dataset_content_sha256": content_hash, "files": hashes, "frozen_evaluation": frozen,
                    "source_code_sha": os.environ.get("GITHUB_SHA"),
                    "source_files": {name: digest(ROOT / name) for name in
                                     ("jobs/ember_sft_data_semantic_v1.py", "jobs/ember_semantic_data_preflight.py",
                                      "config/ember_semantic_data_v1.json")},
                    "source_package_sha256": PACKAGE_SHA256, "reference_model": config["reference_model"],
                    "tokenizer_sha256": report["cpu"].get("tokenizer_sha256"),
                    "block_size": config["block_size"], "generation_budget": config["generation_budget"],
                    "completion_only_loss": True, "ignore_index": -100,
                    "validation_status": report["status"], "training_run_started": False,
                    "dataset_card": "README.md"}
        write_json(output / "manifest.json", manifest)
        (output / "README.md").write_text(
            "# Ember semantic repair data v1\n\n"
            "Original synthetic examples authored for Ember. No external text was scraped and no model generated the labels.\n\n"
            "2,880 training and 576 validation rows cover 24 balanced families: exact tool arguments, concrete direct responses, "
            "and answers grounded in supplied tool results. Each pair has different correct answers and remains in one split. "
            "Negative tests are checked by the validator and are not training labels.\n\n"
            "Validation is a development split: templates and some concept answers are shared. It is not an unseen-concept benchmark. "
            "The separate frozen 36-case semantic gate is excluded by normalized input and exact tool-argument checks. "
            "Those checks do not claim exhaustive semantic decontamination.\n\n"
            "Use completion-only targets, supervise the actual final EOS, and reject context overflow instead of truncating. "
            "See manifest.json and the per-run report for hashes and actual checkpoint CPU validation. "
            "A data PASS does not imply that the candidate model passes its semantic gate. No optimizer step or deployment is performed.\n",
            encoding="utf-8")
        write_json(output / "report.json", report)
        if args.publish:
            publication = publish(output, manifest, config["output_dataset_repo"])
            write_json(output / "publication.json", publication)
        print(json.dumps({"status": report["status"], "rows": {k: len(v) for k, v in splits.items()},
                          "dataset_content_sha256": content_hash, "optimizer_steps": 0}, sort_keys=True))
        return 0
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        write_json(output / "report.json", report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
