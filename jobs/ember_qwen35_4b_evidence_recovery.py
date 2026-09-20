"""Recover final evidence for the completed 4B LoRA without rerunning GPU training."""
import json
import os
from pathlib import Path

REPO = "Jmiller18899/ember-qwen3.5-4b-sft-v1"
FAILED_JOB_ID = "6aafdb0451992417dfccceb6"


def _family_counts(rows):
    keys = sorted({(row["suite"], row["family"]) for row in rows})
    return {
        f"{suite}/{family}": {
            "exact_pass": sum(row.get("exact_match") is True for row in rows
                              if (row["suite"], row["family"]) == (suite, family)),
            "exact_total": sum(row.get("scoring") == "exact" for row in rows
                               if (row["suite"], row["family"]) == (suite, family)),
            "manual_total": sum(row.get("scoring") == "rubric" for row in rows
                                if (row["suite"], row["family"]) == (suite, family)),
        }
        for suite, family in keys
    }


def build_summary(base, final):
    assert len(base) == len(final)
    base_families = _family_counts(base)
    final_families = _family_counts(final)
    base_exact = {
        "pass": sum(row.get("exact_match") is True for row in base),
        "total": sum(row.get("scoring") == "exact" for row in base),
    }
    final_exact = {
        "pass": sum(row.get("exact_match") is True for row in final),
        "total": sum(row.get("scoring") == "exact" for row in final),
    }
    assert base_exact["total"] == final_exact["total"]
    return {
        "base": base_families,
        "final": final_families,
        "base_exact": base_exact,
        "final_exact": final_exact,
        "exact_delta": final_exact["pass"] - base_exact["pass"],
        "strict_family_regressions": [
            family for family in base_families
            if final_families[family]["exact_pass"] < base_families[family]["exact_pass"]
        ],
        "human_review_pending": True,
        "production_ready": False,
        "comparison": "Pinned Qwen3.5-4B base versus fresh conservative rank-8 LoRA.",
        "training_examples": 928,
        "steps_completed": 116,
        "eval_case_count": len(final),
    }


def main():
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN")
    assert token, "HF_TOKEN is required"
    api = HfApi(token=token)
    info = api.repo_info(REPO)
    assert info.private, "Output repo must remain private"
    files = set(api.list_repo_files(REPO))
    required = {
        "adapter_model.safetensors",
        "adapter_config.json",
        "evaluation/base_4b.json",
        "evaluation/final.json",
        "manifest.json",
        "dataset.json",
        "metrics.jsonl",
    }
    missing = sorted(required - files)
    assert not missing, f"Training outputs missing: {missing}"

    def load_json(path):
        local = hf_hub_download(REPO, path, token=token)
        return json.loads(Path(local).read_text())

    base = load_json("evaluation/base_4b.json")
    final = load_json("evaluation/final.json")
    manifest = load_json("manifest.json")
    assert len(base) == 142 and len(final) == 142
    assert manifest["max_steps"] == 116
    assert sum(manifest["counts"]["train"].values()) == 928

    summary = build_summary(base, final)
    assert summary["base_exact"] == {"pass": 60, "total": 68}
    assert summary["final_exact"] == {"pass": 65, "total": 68}
    receipt = {
        "status": "evidence_recovered",
        "original_job_id": FAILED_JOB_ID,
        "training_rerun": False,
        "adapter_preserved": True,
        "failure_stage": "post-training bulk evidence upload",
        "failure_reason": "Hugging Face Xet Trackio bucket returned HTTP 404",
        "source_commit": os.environ.get("GITHUB_SHA"),
    }
    summary_bytes = json.dumps(summary, indent=2).encode()
    receipt_bytes = json.dumps(receipt, indent=2).encode()
    api.create_commit(
        repo_id=REPO,
        operations=[
            CommitOperationAdd(path_in_repo="comparison.json", path_or_fileobj=summary_bytes),
            CommitOperationAdd(path_in_repo="run-evidence/comparison.json", path_or_fileobj=summary_bytes),
            CommitOperationAdd(path_in_repo="run-evidence/recovery.json", path_or_fileobj=receipt_bytes),
        ],
        commit_message="Recover completed 4B training evidence without retraining",
    )
    print(json.dumps({"receipt": receipt, "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
