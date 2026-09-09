"""Conservative, deterministic CPU semantic gate; never trains or deploys Ember.

Keep the legacy promotion evidence immutable. New reports live only under
evaluations/semantic-v1/. Response oracles are bounded fixture answer sets,
not a general language-quality judge. See docs/ember-semantic-gate.md.
"""
from __future__ import annotations

import argparse
from collections import Counter
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


CONTRACT = "ember-semantic-v1"
KINDS = ("tool_call", "direct_response", "tool_result_response")
EOT = "<|endoftext|>"
TOOL = "<|tool|>"
TOKENS = ("<|system|>", "<|user|>", "<|assistant|>", TOOL, "<|tool_result|>", EOT)
PACKAGE_SHA256 = "27e8f7c80317652a22b3d58a0bd474724491a685dfe9e20c0b997b7c5907a289"
PACKAGE_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/24f4a4cecc3be8b195c1216eca6fceba24968358/ember-v0.0.7-hf-ready.zip"
DEFAULT_SPEC = Path(__file__).resolve().parents[1] / "config/ember_semantic_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(text: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)


def normalize_answer(text: str) -> str:
    # Preserve punctuation, digits, signs, negation, units, and every word.
    return " ".join(text.split()).casefold()


def validate_spec(spec: dict) -> None:
    from jsonschema import Draft202012Validator

    if spec.get("contract") != CONTRACT or spec.get("schema_version") != 1:
        raise ValueError("unsupported semantic evaluation contract")
    cases = spec.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("evaluation cases must be a nonempty list")
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate evaluation case IDs")
    if set(case["kind"] for case in cases) != set(KINDS):
        raise ValueError("all three evaluation groups are required")
    for case in cases:
        if not case["prompt"].endswith("<|assistant|>\n"):
            raise ValueError(f"invalid prompt boundary: {case['id']}")
        if case["kind"] == "tool_call":
            schema = case["arguments_schema"]
            Draft202012Validator.check_schema(schema)
            if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
                raise ValueError("tool schemas must be closed objects")
            variants = case["argument_variants"]
            if not variants:
                raise ValueError("tool case lacks argument oracles")
            for value in variants:
                Draft202012Validator(schema).validate(value)
        else:
            variants = case["accepted_responses"]
            if not variants or any(not isinstance(v, str) or not v.strip() for v in variants):
                raise ValueError("response case lacks complete-answer oracles")
            if any(re.search(r"<\|[^>]*\|>", v) for v in variants):
                raise ValueError("answer oracles cannot contain role markers")


def score_case(case: dict, completion: str) -> dict:
    """Score the entire supplied completion, without prefix/substring credit.

    First-turn semantics are also recorded diagnostically. They cannot override
    a missing EOS, extra role, repeated marker, or trailing text in the gate.
    """
    from jsonschema import Draft202012Validator

    if not isinstance(completion, str):
        raise ValueError("completion must be a string")
    body, separator, tail = completion.partition(EOT)
    body = body.strip()
    markers = re.findall(r"<\|[^>]*\|>", body)
    expected_markers = [TOOL] if case["kind"] == "tool_call" else []
    checks = {
        "endoftext_present": bool(separator),
        "single_endoftext": completion.count(EOT) == 1,
        "no_trailing_output": bool(separator) and not tail.strip(),
        "no_extra_roles": markers == expected_markers,
        "nonempty": bool(body),
    }
    semantic = {}
    if case["kind"] == "tool_call":
        payload = None
        error = None
        try:
            if not body.startswith(TOOL):
                raise ValueError("tool envelope must start at the beginning of the answer")
            payload = strict_json(body[len(TOOL):].strip())
        except (ValueError, RecursionError) as exc:
            error = str(exc)
        envelope_ok = isinstance(payload, dict) and set(payload) == {"name", "arguments"}
        arguments = payload.get("arguments") if envelope_ok else None
        schema_ok = Draft202012Validator(case["arguments_schema"]).is_valid(arguments)
        # JSON serialization preserves scalar types (True is not the number 1).
        canonical = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False)
        exact = schema_ok and any(
            canonical(arguments) == canonical(expected) for expected in case["argument_variants"]
        )
        semantic = {
            "single_canonical_json_envelope": envelope_ok,
            "tool_name_correct": envelope_ok and payload["name"] == case["expected_tool"],
            "argument_schema_valid": schema_ok,
            "argument_values_exact": exact,
        }
        detail = {"json_error": error}
    elif case["kind"] in KINDS[1:]:
        semantic = {"whole_answer_matches_oracle": normalize_answer(body) in {
            normalize_answer(value) for value in case["accepted_responses"]
        }}
        detail = {}
    else:
        raise ValueError(f"unsupported case kind: {case['kind']}")
    return {
        "passed": all(checks.values()) and all(semantic.values()),
        "semantic_content_pass": all(semantic.values()),
        "clean_output_pass": all(checks.values()),
        "checks": {**checks, **semantic},
        "failure_reasons": [key for key, ok in {**checks, **semantic}.items() if not ok],
        **detail,
    }


def summarize(cases: list[dict], rows: list[dict]) -> dict:
    expected = {case["id"]: case["kind"] for case in cases}
    actual = [row["id"] for row in rows]
    if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise ValueError("incomplete, duplicate, or unexpected evaluation results")
    if any(row["kind"] != expected[row["id"]] for row in rows):
        raise ValueError("evaluation result kind differs from the spec")
    groups = {}
    for kind in KINDS:
        selected = [row for row in rows if row["kind"] == kind]
        if not selected:
            raise ValueError(f"missing evaluation group: {kind}")
        count = len(selected)
        passed = sum(row["score"]["passed"] is True for row in selected)
        groups[kind] = {
            "passed": passed, "total": count, "pass_rate": passed / count,
            "semantic_content_rate": sum(row["score"]["semantic_content_pass"] is True for row in selected) / count,
            "clean_output_rate": sum(row["score"]["clean_output_pass"] is True for row in selected) / count,
        }
    reasons = Counter(reason for row in rows for reason in row["score"]["failure_reasons"])
    return {
        "groups": groups,
        "passed": sum(row["score"]["passed"] is True for row in rows),
        "total": len(rows),
        "failure_counts": dict(reasons),
        "semantic_gate_pass": all(row["score"]["passed"] is True for row in rows),
    }


def token_contract(tokenizer) -> dict:
    encoded = {token: tokenizer.encode(token) for token in TOKENS}
    dummy_prefix = all(len(ids) == 2 for ids in encoded.values()) and len({v[0] for v in encoded.values()}) == 1
    signatures = {token: ids[1:] if dummy_prefix else ids for token, ids in encoded.items()}
    if any(len(ids) != 1 for ids in signatures.values()) or len({tuple(v) for v in signatures.values()}) != len(TOKENS):
        raise ValueError("special tokens must be atomic and distinct")
    eos_id = signatures[EOT][0]
    if tokenizer.decode([eos_id]).strip() != EOT:
        raise ValueError("EOS token does not round-trip")
    return {"signatures": signatures, "eos_id": eos_id}


def generate_completion(model, tokenizer, torch, prompt: str, max_new_tokens: int) -> dict:
    """Greedy CPU decoding stops only when the model emits its real EOS token."""
    contract = token_contract(tokenizer)
    ids = tokenizer.encode(prompt)
    if not ids or len(ids) + max_new_tokens > int(model.cfg.block_size):
        raise ValueError("prompt plus full output budget exceeds the context window")
    generated = []
    reason = "max_new_tokens"
    with torch.inference_mode():
        x = torch.tensor([ids], dtype=torch.long, device="cpu")
        for _ in range(max_new_tokens):
            logits, _ = model(x)
            next_logits = logits[0, -1, :]
            if not bool(torch.isfinite(next_logits).all().item()):
                raise ValueError("non-finite generation logits")
            next_id = int(torch.argmax(next_logits).item())
            generated.append(next_id)
            if next_id == contract["eos_id"]:
                reason = "eos"
                break
            x = torch.cat((x, torch.tensor([[next_id]], dtype=torch.long, device="cpu")), dim=1)
    return {"completion": tokenizer.decode(generated), "generated_ids": generated, "stop_reason": reason}


def evaluate_model(model, tokenizer, torch, spec: dict, variant: str) -> dict:
    model.eval()
    rows = []
    for case in spec["cases"]:
        generation = generate_completion(model, tokenizer, torch, case["prompt"], spec["generation"]["max_new_tokens"])
        score = score_case(case, generation["completion"])
        rows.append({"id": case["id"], "kind": case["kind"], "cohort": case["cohort"], **generation, "score": score})
        print(f"CASE {variant} {case['id']} {'PASS' if score['passed'] else 'FAIL'}", flush=True)
    return {"summary": summarize(spec["cases"], rows), "cases": rows}


def verify_source(spec: dict, legacy: dict, checkpoint: dict) -> None:
    source = spec["source"]
    if checkpoint.get("format") != "ember-checkpoint-v1":
        raise ValueError("unexpected checkpoint format")
    if checkpoint.get("step") != source["step"] or checkpoint.get("run_id") != source["run_id"]:
        raise ValueError("checkpoint identity mismatch")
    if checkpoint.get("train_config", {}).get("version") != source["version"]:
        raise ValueError("checkpoint training version mismatch")
    if legacy.get("model_repo") != source["repo_id"]:
        raise ValueError("legacy evaluation belongs to a different repository")
    if legacy.get("checkpoint", {}).get("best_path") != source["checkpoint_path"]:
        raise ValueError("legacy evaluation refers to a different checkpoint path")
    if legacy.get("checkpoint", {}).get("step") != source["step"]:
        raise ValueError("legacy evaluation step differs from the candidate")
    if legacy.get("promotion", {}).get("promotion_eligible") is not True:
        raise ValueError("legacy promotion is not an explicit PASS")


def persist_report(report: dict, output: Path, api=None) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if api is None:
        return "local_artifact"
    from huggingface_hub import CommitOperationAdd
    timestamp = report["created_at"].replace(":", "").replace("-", "")
    paths = [f"evaluations/semantic-v1/{timestamp}-{report['spec_sha256'][:12]}.json", "evaluations/semantic-v1/latest.json"]
    api.create_commit(
        repo_id=report["source"]["repo_id"], repo_type="model",
        operations=[CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(output)) for path in paths],
        commit_message="Record Ember semantic-v1 CPU gate (legacy promotion unchanged)",
    )
    return paths[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=Path("semantic-results"))
    parser.add_argument("--revision", help="Optional immutable Hugging Face model commit")
    parser.add_argument("--publish", action="store_true", help="Save under evaluations/semantic-v1/ only")
    args = parser.parse_args()
    spec = strict_json(args.spec.read_text(encoding="utf-8"))
    validate_spec(spec)
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required for the private candidate")
    import torch
    from huggingface_hub import HfApi, hf_hub_download

    torch.set_num_threads(2)
    torch.manual_seed(spec["generation"]["seed"])
    torch.use_deterministic_algorithms(True)
    api = HfApi(token=token)
    source = spec["source"]
    revision = args.revision or api.model_info(source["repo_id"]).sha
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("evaluation requires an immutable model revision")
    with tempfile.TemporaryDirectory(prefix="ember-semantic-") as td:
        work = Path(td)
        archive = work / "ember.zip"
        urllib.request.urlretrieve(PACKAGE_URL, archive)
        if sha256_file(archive) != PACKAGE_SHA256:
            raise ValueError("Ember source package checksum mismatch")
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        sys.path.insert(0, str(work / "src" / "ember"))
        from src.model import EmberGPT, ModelConfig
        from src.quantize_int4 import dequantize_tensor, quantize_tensor
        from src.tokenizer import tokenizer_from_state_dict

        def download(filename):
            return Path(hf_hub_download(repo_id=source["repo_id"], filename=filename, revision=revision,
                                        token=token, local_dir=work / "candidate"))

        legacy_path = download("evaluations/latest.json")
        legacy = strict_json(legacy_path.read_text(encoding="utf-8"))
        best_path = download(source["checkpoint_path"])
        int4_path = download(source["int4_path"])
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        verify_source(spec, legacy, checkpoint)
        int4 = torch.load(int4_path, map_location="cpu", weights_only=False)
        if int4.get("format") != "ember-int4-v1" or any(int4.get(key) != checkpoint.get(key) for key in ("model_config", "tokenizer", "train_config")):
            raise ValueError("INT4 metadata does not match the source checkpoint")
        # Export format has no source hash/step. Verify every exported tensor
        # against deterministic quantization of the pinned full checkpoint.
        qstate = int4["quantized_state"]
        passthrough = int4.get("passthrough_state", {})
        if set(qstate) & set(passthrough) or set(qstate) | set(passthrough) != set(checkpoint["model_state"]):
            raise ValueError("INT4 tensor keys do not match the source checkpoint")
        for name, tensor in checkpoint["model_state"].items():
            if torch.is_floating_point(tensor) and tensor.numel() >= 32:
                expected = quantize_tensor(tensor)
                actual = qstate.get(name)
                if not isinstance(actual, dict) or set(expected) != set(actual):
                    raise ValueError(f"INT4 source mismatch: {name}")
                for key, value in expected.items():
                    equal = torch.equal(value, actual[key]) if torch.is_tensor(value) else value == actual[key]
                    if not equal:
                        raise ValueError(f"INT4 source mismatch: {name}.{key}")
            elif name not in passthrough or not torch.equal(tensor, passthrough[name]):
                raise ValueError(f"INT4 passthrough source mismatch: {name}")

        tokenizer = tokenizer_from_state_dict(checkpoint["tokenizer"])
        model = EmberGPT(ModelConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state"])
        full_result = evaluate_model(model, tokenizer, torch, spec, "full")
        del model, checkpoint
        gc.collect()
        restored = {name: dequantize_tensor(value) for name, value in qstate.items()}
        restored.update(passthrough)
        model = EmberGPT(ModelConfig(**int4["model_config"]))
        model.load_state_dict(restored)
        int4_result = evaluate_model(model, tokenizer_from_state_dict(int4["tokenizer"]), torch, spec, "int4")

        legacy_cases = [case for case in spec["cases"] if case["cohort"] == "legacy_recheck"]
        original_rows = {row["id"]: row for row in legacy["cases"]}
        raw_rows = [{"id": case["id"], "kind": case["kind"], "completion": original_rows[case["id"]]["completion"],
                     "score": score_case(case, original_rows[case["id"]]["completion"])} for case in legacy_cases]
        gate_pass = full_result["summary"]["semantic_gate_pass"] and int4_result["summary"]["semantic_gate_pass"]
        report = {
            "schema_version": 1, "contract": CONTRACT,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {**source, "revision": revision, "checkpoint_sha256": sha256_file(best_path),
                       "int4_sha256": sha256_file(int4_path), "int4_source_verified": True},
            "spec_sha256": sha256_file(args.spec), "evaluator_sha256": sha256_file(Path(__file__)),
            "package_sha256": PACKAGE_SHA256, "code_commit": os.environ.get("GITHUB_SHA"),
            "runtime": {"python": sys.version, "torch": torch.__version__, "device": "cpu", "threads": 2},
            "generation": {**spec["generation"], "strategy": "argmax", "stop": "model_emitted_eos_only"},
            "policy": {"required_pass_rate_per_group_per_variant": 1.0, "production_authorized": False},
            "legacy_evidence": {"sha256": sha256_file(legacy_path), "promotion_eligible": True,
                                "metrics": legacy["metrics"], "promotion": legacy["promotion"]},
            "legacy_raw_recheck": {"summary": summarize(legacy_cases, raw_rows), "cases": raw_rows},
            "variants": {"full": full_result, "int4": int4_result},
            "semantic_gate_pass": gate_pass, "status": "PASS" if gate_pass else "FAIL",
            "production_authorized": False,
            "limitations": spec["limitations"],
        }
        output = args.output_dir / "report.json"
        destination = persist_report(report, output, api if args.publish else None)
        print(json.dumps({"status": report["status"], "source_revision": revision, "report": str(output),
                          "persisted": destination, "variants": {k: v["summary"] for k, v in report["variants"].items()}}, indent=2), flush=True)
        print(f"EMBER_SEMANTIC_V1={report['status']}", flush=True)
        return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
