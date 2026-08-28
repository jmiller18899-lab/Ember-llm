# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "huggingface-hub>=1.4",
#   "sentencepiece>=0.2",
#   "torch>=2.4",
# ]
# ///
"""Evaluate an Ember checkpoint with deterministic ClawAgent contract cases.

The job is CPU-safe. It records a reproducible baseline for v0.0.7 and compares
v0.0.8 against both absolute tool-use gates and that baseline before promotion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import urllib.request
import zipfile


PACKAGE_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/main/ember-v0.0.7-hf-ready.zip"
PACKAGE_SHA256 = "27e8f7c80317652a22b3d58a0bd474724491a685dfe9e20c0b997b7c5907a289"
EVAL_SPEC_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/main/config/ember_v0.0.8_eval.json"
EVAL_SPEC_SHA256 = "e006aa0f7c50797e0466a87fa3f1e35a1f00a63baf1f5113cf6e574844079bd4"
MODEL_NAME_PATTERN = re.compile(r"^ember-v\d+\.\d+\.\d+-t4$")
SPECIAL_TOKENS = [
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|tool_result|>",
    "<|endoftext|>",
]


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


def extract_json_object(text: str) -> dict | None:
    """Return the first balanced JSON object in text, or None."""
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
                candidate = text[start : index + 1]
                try:
                    value = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, dict) else None
    return None


def visible_text(text: str) -> str:
    for token in SPECIAL_TOKENS:
        text = text.replace(token, " ")
    return " ".join(text.split())


def tool_name(payload: dict) -> str:
    direct = payload.get("name") or payload.get("tool")
    if isinstance(direct, str):
        return direct
    function = payload.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return ""


def tool_arguments(payload: dict):
    if "arguments" in payload:
        return payload["arguments"]
    function = payload.get("function")
    if isinstance(function, dict):
        return function.get("arguments")
    return None


def score_case(case: dict, completion: str) -> dict:
    kind = case["kind"]
    readable = visible_text(completion)
    if kind == "tool_call":
        marker_present = "<|tool|>" in completion
        after_marker = completion.split("<|tool|>", 1)[1] if marker_present else completion
        payload = extract_json_object(after_marker)
        expected = case["expected_tool"]
        name_matches = bool(payload) and tool_name(payload) == expected
        arguments = tool_arguments(payload or {})
        arguments_valid = isinstance(arguments, (dict, str)) and bool(arguments)
        passed = marker_present and name_matches and arguments_valid
        return {
            "passed": passed,
            "marker_present": marker_present,
            "json_valid": payload is not None,
            "tool_name_matches": name_matches,
            "arguments_present": arguments_valid,
        }
    if kind in {"direct_response", "tool_result_response"}:
        no_extra_tool_call = "<|tool|>" not in completion
        nonempty = len(readable) >= 3
        return {
            "passed": no_extra_tool_call and nonempty,
            "nonempty": nonempty,
            "no_extra_tool_call": no_extra_tool_call,
        }
    raise ValueError(f"unsupported evaluation case kind: {kind}")


def aggregate_scores(results: list[dict]) -> dict:
    mapping = {
        "tool_call": "valid_tool_call_rate",
        "direct_response": "direct_response_rate",
        "tool_result_response": "tool_result_response_rate",
    }
    metrics = {}
    for kind, metric_name in mapping.items():
        selected = [row for row in results if row["kind"] == kind]
        if not selected:
            raise RuntimeError(f"evaluation spec has no {kind} cases")
        metrics[metric_name] = sum(bool(row["score"]["passed"]) for row in selected) / len(selected)
    metrics["all_generations_nonempty"] = all(len(visible_text(row["completion"])) >= 3 for row in results)
    return metrics


def latest_checkpoint_path(files: list[str], suffix: str) -> str:
    candidates = sorted(
        path for path in files
        if path.startswith("checkpoints/") and path.endswith(suffix)
    )
    if not candidates:
        raise RuntimeError(f"model repository has no checkpoint ending in {suffix}")
    return candidates[-1]


def generate_completion(model, tokenizer, torch, prompt: str, generation: dict) -> str:
    prompt_ids = tokenizer.encode(prompt)
    max_new_tokens = int(generation["max_new_tokens"])
    available = int(model.cfg.block_size) - len(prompt_ids)
    if available < 1:
        raise RuntimeError("evaluation prompt exceeds the checkpoint context window")
    max_new_tokens = min(max_new_tokens, available)
    x = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.inference_mode():
        y = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=float(generation["temperature"]),
            top_k=int(generation["top_k"]),
        )
    generated_ids = y[0].tolist()[len(prompt_ids) :]
    return tokenizer.decode(generated_ids)


def fixed_validation_loss(model, tokenizer, torch, val_text: str, settings: dict) -> float:
    """Evaluate evenly spaced, deterministic slices of the verified val set."""
    token_ids = tokenizer.encode(val_text)
    block_size = int(model.cfg.block_size)
    batch_size = int(settings["batch_size"])
    batches = int(settings["batches"])
    example_count = batch_size * batches
    max_start = len(token_ids) - block_size - 1
    if max_start < 0:
        raise RuntimeError("verified validation corpus is too short for the model context")
    if example_count < 1:
        raise RuntimeError("validation-loss evaluation requires at least one example")
    starts = [
        round(index * max_start / max(1, example_count - 1))
        for index in range(example_count)
    ]
    losses = []
    with torch.inference_mode():
        for offset in range(0, example_count, batch_size):
            selected = starts[offset : offset + batch_size]
            x = torch.tensor(
                [token_ids[start : start + block_size] for start in selected],
                dtype=torch.long,
            )
            y = torch.tensor(
                [token_ids[start + 1 : start + block_size + 1] for start in selected],
                dtype=torch.long,
            )
            _, loss = model(x, y)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("checkpoint produced non-finite fixed validation loss")
            losses.append(float(loss.item()))
    return sum(losses) / len(losses)


def special_token_contract(tokenizer) -> dict:
    """Validate dedicated marker IDs while allowing SentencePiece's shared dummy prefix."""
    token_ids = {token_text: tokenizer.encode(token_text) for token_text in SPECIAL_TOKENS}
    has_shared_dummy_prefix = (
        all(len(ids) == 2 for ids in token_ids.values())
        and len({ids[0] for ids in token_ids.values()}) == 1
    )
    shared_prefix_id = next(iter(token_ids.values()))[0] if has_shared_dummy_prefix else None
    signatures = {
        token_text: ids[1:] if has_shared_dummy_prefix else ids
        for token_text, ids in token_ids.items()
    }
    atomic = all(len(signature) == 1 for signature in signatures.values())
    unique = len({tuple(signature) for signature in signatures.values()}) == len(SPECIAL_TOKENS)
    return {
        "atomic": atomic,
        "unique": unique,
        "shared_prefix_id": shared_prefix_id,
        "ids": token_ids,
        "signatures": signatures,
    }


def compare_for_promotion(candidate: dict, baseline: dict, thresholds: dict) -> dict:
    baseline_loss = float(baseline["metrics"]["fixed_validation_loss"])
    candidate_loss = float(candidate["metrics"]["fixed_validation_loss"])
    relative_improvement = (baseline_loss - candidate_loss) / baseline_loss
    candidate_metrics = candidate["metrics"]
    baseline_metrics = baseline["metrics"]
    checks = {
        "validation_loss_improved": relative_improvement >= float(
            thresholds["minimum_relative_validation_loss_improvement"]
        ),
        "absolute_tool_call_gate": candidate_metrics["valid_tool_call_rate"] >= float(
            thresholds["minimum_valid_tool_call_rate"]
        ),
        "absolute_direct_response_gate": candidate_metrics["direct_response_rate"] >= float(
            thresholds["minimum_direct_response_rate"]
        ),
        "absolute_tool_result_gate": candidate_metrics["tool_result_response_rate"] >= float(
            thresholds["minimum_tool_result_response_rate"]
        ),
        "tool_call_non_regression": candidate_metrics["valid_tool_call_rate"] >= baseline_metrics[
            "valid_tool_call_rate"
        ],
        "direct_response_non_regression": candidate_metrics["direct_response_rate"] >= baseline_metrics[
            "direct_response_rate"
        ],
        "tool_result_non_regression": candidate_metrics["tool_result_response_rate"] >= baseline_metrics[
            "tool_result_response_rate"
        ],
        "technical_gate": bool(candidate["technical_pass"]),
    }
    return {
        "baseline_fixed_validation_loss": baseline_loss,
        "candidate_fixed_validation_loss": candidate_loss,
        "relative_validation_loss_improvement": relative_improvement,
        "checks": checks,
        "promotion_eligible": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a private Ember model repository")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if not MODEL_NAME_PATTERN.fullmatch(args.model_name):
        raise ValueError("model name must look like ember-v0.0.8-t4")

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN was not injected as a Job secret")

    import torch
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    owner = api.whoami()["name"]
    model_repo = f"{owner}/{args.model_name}"
    corpus_repo = f"{owner}/ember-corpus-v0.0.7"

    with tempfile.TemporaryDirectory(prefix="ember-eval-") as temporary:
        work = Path(temporary)
        archive = download_verified(PACKAGE_URL, work / "ember.zip", PACKAGE_SHA256)
        spec_path = download_verified(EVAL_SPEC_URL, work / "eval.json", EVAL_SPEC_SHA256)
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        root = work / "src" / "ember"
        sys.path.insert(0, str(root))

        from src.checkpoint import load_checkpoint
        from src.model import EmberGPT, ModelConfig
        from src.quantize_int4 import dequantize_tensor
        from src.tokenizer import tokenizer_from_state_dict

        files = api.list_repo_files(model_repo, repo_type="model")
        best_repo_path = latest_checkpoint_path(files, "/best.pt")
        int4_repo_path = latest_checkpoint_path(files, "/best.int4.pt")
        best_path = Path(hf_hub_download(
            repo_id=model_repo,
            repo_type="model",
            filename=best_repo_path,
            token=token,
            local_dir=work / "model",
        ))
        int4_path = Path(hf_hub_download(
            repo_id=model_repo,
            repo_type="model",
            filename=int4_repo_path,
            token=token,
            local_dir=work / "model",
        ))
        val_path = Path(hf_hub_download(
            repo_id=corpus_repo,
            repo_type="dataset",
            filename="data/val.txt",
            token=token,
            local_dir=work / "corpus",
        ))

        checkpoint = load_checkpoint(best_path, device="cpu")
        tokenizer = tokenizer_from_state_dict(checkpoint["tokenizer"])
        model = EmberGPT(ModelConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        torch.manual_seed(int(spec["generation"]["seed"]))
        validation_loss = fixed_validation_loss(
            model,
            tokenizer,
            torch,
            val_path.read_text(encoding="utf-8"),
            spec["validation_loss"],
        )

        token_contract = special_token_contract(tokenizer)
        special_tokens_atomic = bool(token_contract["atomic"])
        special_tokens_unique = bool(token_contract["unique"])

        case_results = []
        for case in spec["cases"]:
            completion = generate_completion(model, tokenizer, torch, case["prompt"], spec["generation"])
            case_results.append({
                "id": case["id"],
                "kind": case["kind"],
                "completion": completion,
                "score": score_case(case, completion),
            })
        metrics = aggregate_scores(case_results)
        metrics["fixed_validation_loss"] = validation_loss

        del model
        gc.collect()
        int4 = torch.load(int4_path, map_location="cpu", weights_only=False)
        int4_state = {name: dequantize_tensor(value) for name, value in int4["quantized_state"].items()}
        int4_state.update(int4.get("passthrough_state", {}))
        int4_model = EmberGPT(ModelConfig(**int4["model_config"]))
        int4_model.load_state_dict(int4_state)
        int4_tokenizer = tokenizer_from_state_dict(int4["tokenizer"])
        int4_completion = generate_completion(
            int4_model,
            int4_tokenizer,
            torch,
            spec["cases"][0]["prompt"],
            {**spec["generation"], "max_new_tokens": 24},
        )
        int4_smoke_pass = len(visible_text(int4_completion)) >= 3

        technical_pass = (
            special_tokens_atomic
            and special_tokens_unique
            and metrics["all_generations_nonempty"]
            and int4_smoke_pass
        )
        result = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "label": args.label,
            "model_repo": model_repo,
            "role": "baseline" if args.model_name == spec["baseline_model_name"] else "candidate",
            "checkpoint": {
                "best_path": best_repo_path,
                "best_bytes": best_path.stat().st_size,
                "int4_path": int4_repo_path,
                "int4_bytes": int4_path.stat().st_size,
                "step": int(checkpoint["step"]),
                "best_validation_loss": float(checkpoint["best_val_loss"]),
                "run_id": checkpoint["run_id"],
            },
            "token_contract": token_contract,
            "metrics": metrics,
            "int4_smoke": {"passed": int4_smoke_pass, "completion": int4_completion},
            "technical_pass": technical_pass,
            "cases": case_results,
            "evaluation_spec_sha256": EVAL_SPEC_SHA256,
            "package_sha256": PACKAGE_SHA256,
        }

        if result["role"] == "candidate":
            baseline_repo = f"{owner}/{spec['baseline_model_name']}"
            try:
                baseline_path = hf_hub_download(
                    repo_id=baseline_repo,
                    repo_type="model",
                    filename="evaluations/latest.json",
                    token=token,
                    local_dir=work / "baseline",
                )
                baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
                result["promotion"] = compare_for_promotion(result, baseline, spec["promotion"])
            except Exception as exc:
                result["promotion"] = {
                    "promotion_eligible": False,
                    "blocked_reason": f"baseline evaluation unavailable: {type(exc).__name__}: {exc}",
                }
        else:
            result["promotion"] = {"promotion_eligible": None, "reason": "baseline recorded"}

        output = work / "evaluation.json"
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        api.upload_file(
            repo_id=model_repo,
            repo_type="model",
            path_or_fileobj=str(output),
            path_in_repo=f"evaluations/{args.label}-{timestamp}.json",
            commit_message=f"Record Ember evaluation {args.label}",
        )
        api.upload_file(
            repo_id=model_repo,
            repo_type="model",
            path_or_fileobj=str(output),
            path_in_repo="evaluations/latest.json",
            commit_message=f"Update latest Ember evaluation to {args.label}",
        )
        if result["role"] == "candidate":
            remote_files = set(api.list_repo_files(model_repo, repo_type="model"))
            state = {}
            if "run-state.json" in remote_files:
                state_download = hf_hub_download(
                    repo_id=model_repo,
                    repo_type="model",
                    filename="run-state.json",
                    token=token,
                    local_dir=work / "state",
                )
                state = json.loads(Path(state_download).read_text(encoding="utf-8"))
            state.update({
                "status": "evaluation_complete",
                "evaluation_status": "PASS" if technical_pass else "FAIL",
                "promotion_eligible": bool(result["promotion"].get("promotion_eligible")),
                "evaluation_path": "evaluations/latest.json",
                "evaluation_created_at": result["created_at"],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            state_path = work / "run-state.json"
            state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            api.upload_file(
                repo_id=model_repo,
                repo_type="model",
                path_or_fileobj=str(state_path),
                path_in_repo="run-state.json",
                commit_message="Complete Ember v0.0.8 evaluation state",
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        print(f"EMBER_EVALUATION={'PASS' if technical_pass else 'FAIL'}")
        print(f"MODEL_REPO={model_repo}")

        if not technical_pass:
            raise RuntimeError("Ember technical evaluation gate failed")


if __name__ == "__main__":
    main()
