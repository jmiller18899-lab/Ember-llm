# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0", "transformers==5.17.0", "peft==0.20.0", "accelerate==1.15.0", "huggingface-hub==1.31.0"]
# ///
"""Read-only model evaluation: preserve raw output and diagnose token cutoffs.

No training, promotion, deployment, or automatic GPU retries. Only new diagnostic
JSON is uploaded; existing model weights and benchmark files are never overwritten.
Use --upload-report PATH to retry evidence upload without loading a model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

REPO = "Jmiller18899/ember-qwen3.5-4b-sft-v1"
RUN_ID = "direct-answer-audit-20260922-v1"
SOURCE_COMMIT = "1a4988596b46f07ebb321ce85f18804e12a767de"
LIMITS = (96, 192)
# These are consumed diagnostic/regression cases, NOT untouched test data.
REVIEW_IDS = {
    "v2-confirmation-clarification-0014", "v2-confirmation-clarification-0062",
    "fresh-natural-v2-03", "fresh-natural-v2-07", "v3-confirm-08",
    "v3-confirm-09", "v3-confirm-11", "v3-confirm-12", "v3-confirm-15",
    "natural-v1-29", "v3-confirm-14",
}


def generation_record(tokenizer: Any, token_ids: list[int], limit: int,
                      eos_ids: list[int]) -> dict[str, Any]:
    if limit <= 0 or len(token_ids) > limit:
        raise ValueError("Invalid generation length or token limit")
    ids = [int(i) for i in token_ids]
    eos = [int(i) for i in eos_ids]
    ended = bool(ids and ids[-1] in eos)
    hit_limit = len(ids) == limit
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    return {
        "raw_token_ids": ids,
        "raw_decoded": tokenizer.decode(ids, skip_special_tokens=False),
        "decoded_text": decoded,
        "output": decoded.strip(),
        "normalization": "skip_special_tokens=True, then str.strip(); no answer extraction",
        "generated_tokens": len(ids), "max_new_tokens": limit,
        "eos_token_ids": eos, "ended_with_eos": ended,
        "hit_token_limit": hit_limit,
        # EOS exactly at the cap is a normal stop, not evidence of truncation.
        "stop_reason": "eos" if ended else "length" if hit_limit else "unknown",
    }


def pair_saved(base: list[dict], final: list[dict]) -> list[tuple[dict, dict]]:
    def index(rows):
        result = {}
        for r in rows:
            if not isinstance(r.get("id"), str) or r["id"] in result:
                raise ValueError("Missing or duplicate case ID")
            if not isinstance(r.get("output"), str):
                raise ValueError("Missing output")
            if r.get("scoring") not in {"exact", "rubric"}:
                raise ValueError("Unknown scoring method")
            expected_score = r["output"] == r["answer"] if r["scoring"] == "exact" else None
            if r.get("exact_match") is not expected_score:
                raise ValueError("Saved score disagrees with saved output")
            result[r["id"]] = r
        return result
    before, after = index(base), index(final)
    if set(before) != set(after):
        raise ValueError("Benchmark case sets differ")
    pairs = [(r, after[r["id"]]) for r in base]
    for b, f in pairs:
        for key in ("prompt", "family", "suite", "scoring", "answer", "rubric"):
            if b.get(key) != f.get(key):
                raise ValueError(f"Case {b['id']} changed {key}")
    return pairs


def exact_counts(rows: list[dict]) -> dict[str, int]:
    exact = [r for r in rows if r["scoring"] == "exact"]
    return {"pass": sum(r["output"] == r["answer"] for r in exact), "total": len(exact)}


def read_saved(api, revision: str):
    from huggingface_hub import hf_hub_download
    def load(path):
        file = Path(hf_hub_download(REPO, path, revision=revision, token=api.token))
        return json.loads(file.read_text(encoding="utf-8"))
    base, final = load("evaluation/base_4b.json"), load("evaluation/final.json")
    manifest = load("manifest.json")
    pairs = pair_saved(base, final)
    if manifest.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("Unexpected training source; do not silently compare another run")
    if manifest.get("model") != "Qwen/Qwen3.5-4B":
        raise ValueError("Unexpected base model")
    if not REVIEW_IDS <= {r["id"] for r in final}:
        raise ValueError("Required diagnostic cases missing")
    summary = {
        "repo": REPO, "revision": revision, "source_commit": SOURCE_COMMIT,
        "base_exact": exact_counts(base), "final_exact": exact_counts(final),
        "total_cases": len(final), "rubric_cases": sum(r["scoring"] == "rubric" for r in final),
        "exact_regressions": [b["id"] for b, f in pairs if b["exact_match"] is True and f["exact_match"] is False],
        "exact_improvements": [b["id"] for b, f in pairs if b["exact_match"] is False and f["exact_match"] is True],
        "remaining_exact_failures": [f["id"] for _, f in pairs if f["exact_match"] is False],
        "legacy_stop_evidence_complete": all("raw_token_ids" in r and "stop_reason" in r for r in base + final),
        "training_started": False, "production_ready": False,
    }
    return base, final, manifest, summary


def publish(api, report_path: Path) -> None:
    # This path loads neither torch nor model weights and can recover upload alone.
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("repo") != REPO or report.get("run_id") != RUN_ID:
        raise ValueError("Unexpected report destination or run ID")
    payload = report_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    path = f"diagnostics/{RUN_ID}/report-{digest[:16]}.json"
    api.upload_file(repo_id=REPO, path_in_repo=path, path_or_fileobj=payload,
                    commit_message="Save evaluation-only direct-answer diagnostic evidence")
    from huggingface_hub import hf_hub_download
    saved = Path(hf_hub_download(REPO, path, token=api.token)).read_bytes()
    if hashlib.sha256(saved).hexdigest() != digest:
        raise RuntimeError("Uploaded diagnostic readback did not match")
    print("EVIDENCE_VERIFIED " + json.dumps({"repo": REPO, "path": path, "sha256": digest}), flush=True)


def run_evaluation(api, revision: str, base: list[dict], final: list[dict], manifest: dict, audit: dict):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for the bounded BF16 replay; no CPU fallback")
    set_seed(431)
    torch.set_num_threads(2)
    base_id, base_revision = manifest["model"], manifest["revision"]
    tokenizer = AutoTokenizer.from_pretrained(base_id, revision=base_revision)
    tokenizer.pad_token = tokenizer.eos_token
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        base_id, revision=base_revision, dtype=torch.bfloat16, device_map={"": 0},
        output_loading_info=True, key_mapping={r"^model.language_model\.": "model."})
    if loading["missing_keys"] or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise RuntimeError("Base weight loading mismatch")
    selected_ids = REVIEW_IDS | set(audit["remaining_exact_failures"])
    selected = [r for r in final if r["id"] in selected_ids]
    reference = {"base": {r["id"]: r for r in base}, "adapter": {r["id"]: r for r in final}}
    report = {"repo": REPO, "run_id": RUN_ID, "adapter_revision": revision,
        "base_model": base_id, "base_revision": base_revision, "legacy_audit": audit,
        "source_commit": SOURCE_COMMIT, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "system": manifest["system"], "seed": 431, "dtype": "bfloat16",
        "do_sample": False, "enable_thinking": False, "limits": list(LIMITS),
        "training_started": False, "training_steps": 0, "production_ready": False,
        "case_status": "consumed regression diagnostics; not an independent generalization test",
        "loading_info": loading, "generation_configs": {}, "records": []}
    path = Path("ember-direct-answer-report.json")
    def save():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        temporary.replace(path)
    save()
    for variant in ("base", "adapter"):
        if variant == "adapter":
            model = PeftModel.from_pretrained(model, REPO, revision=revision, is_trainable=False)
        model.requires_grad_(False)
        model.eval()
        report["generation_configs"][variant] = model.generation_config.to_dict()
        eos = model.generation_config.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
        for row in selected:
            ids = tokenizer.apply_chat_template(
                [{"role": "system", "content": manifest["system"]}, {"role": "user", "content": row["prompt"]}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False,
                return_tensors="pt", return_dict=False).to(model.device)
            for limit in LIMITS:
                torch.cuda.synchronize()
                start = time.monotonic()
                with torch.inference_mode():
                    generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                        max_new_tokens=limit, do_sample=False, use_cache=True,
                        pad_token_id=tokenizer.pad_token_id)
                torch.cuda.synchronize()
                seconds = time.monotonic() - start
                record = generation_record(tokenizer, generated[0, ids.shape[-1]:].tolist(), limit, eos_ids)
                record.update(id=row["id"], family=row["family"], prompt=row["prompt"],
                    scoring=row["scoring"], answer=row.get("answer"), rubric=row.get("rubric"),
                    variant=variant, input_token_ids=ids[0].tolist(), seconds=seconds,
                    legacy_output=reference[variant][row["id"]]["output"])
                record["reproduces_legacy"] = record["output"] == record["legacy_output"]
                record["exact_match"] = record["output"] == row["answer"] if row["scoring"] == "exact" else None
                report["records"].append(record)
                save()
                print("RESULT " + json.dumps(record, ensure_ascii=False), flush=True)
    comparisons = []
    for variant in ("base", "adapter"):
        for row in selected:
            small, large = [r for r in report["records"] if r["variant"] == variant and r["id"] == row["id"]]
            comparisons.append({"variant": variant, "id": row["id"],
                "old_output_reproduced_at_96": small["reproduces_legacy"],
                "stop_at_96": small["stop_reason"], "stop_at_192": large["stop_reason"],
                "output_changed_with_larger_limit": small["output"] != large["output"],
                "exact_96": small["exact_match"], "exact_192": large["exact_match"]})
    report["comparisons"] = comparisons
    report["summary"] = {"cases": len(selected), "generations": len(report["records"]),
        "legacy_replay_mismatches_at_96": sum(not r["old_output_reproduced_at_96"] for r in comparisons),
        "length_stops_at_96": sum(r["stop_at_96"] == "length" for r in comparisons),
        "changed_with_larger_limit": sum(r["output_changed_with_larger_limit"] for r in comparisons),
        "unknown_stops": sum(r["stop_reason"] == "unknown" for r in report["records"])}
    report["evaluation_completed"] = True
    save()
    print("FINAL_SUMMARY " + json.dumps(report["summary"]), flush=True)
    # No Trackio import: evidence publication cannot inherit its failed Xet session.
    publish(api, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--upload-report", type=Path)
    args = parser.parse_args()
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required; credentials must not be printed")
    api = HfApi(token=token)
    if api.whoami()["name"].lower() != "jmiller18899":
        raise RuntimeError("Unexpected Hugging Face account")
    if args.upload_report:
        publish(api, args.upload_report)
        return
    revision = os.environ.get("EMBER_AUDIT_REVISION")
    if not revision:
        if not args.audit_only:
            raise RuntimeError("GPU evaluation requires the CPU-verified pinned revision")
        revision = api.model_info(REPO).sha
    base, final, manifest, audit = read_saved(api, revision)
    Path("ember-audit-summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("SAVED_AUDIT " + json.dumps(audit), flush=True)
    if not args.audit_only:
        run_evaluation(api, revision, base, final, manifest, audit)

if __name__ == "__main__":
    main()
