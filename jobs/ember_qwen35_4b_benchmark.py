# /// script
# dependencies = ["torch==2.11.0", "transformers==5.17.0", "accelerate==1.15.0", "huggingface-hub==1.31.0"]
# ///
"""Reproducible Ember baseline benchmark for Qwen3.5-4B; no training."""
import json
import os
import time
from pathlib import Path

import torch
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, set_seed

BASE = "Qwen/Qwen3.5-4B"
EVIDENCE_REPO = "Jmiller18899/ember-qwen3.5-2b-sft-v3"
OUTPUT_PREFIX = "benchmarks/qwen35-4b-base"
SYSTEM = "You are Ember. Answer the current request directly and concisely. Preserve supplied facts and follow the requested format."


def counts(items):
    keys = sorted({(row["suite"], row["family"]) for row in items})
    return {
        f"{suite}/{family}": {
            "exact_pass": sum(
                row["exact_match"] is True
                for row in items
                if (row["suite"], row["family"]) == (suite, family)
            ),
            "exact_total": sum(
                row["scoring"] == "exact"
                for row in items
                if (row["suite"], row["family"]) == (suite, family)
            ),
            "manual_total": sum(
                row["scoring"] == "rubric"
                for row in items
                if (row["suite"], row["family"]) == (suite, family)
            ),
        }
        for suite, family in keys
    }


def main():
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    assert api.whoami()["name"].lower() == "jmiller18899"
    assert api.repo_info(EVIDENCE_REPO).private
    set_seed(431)
    torch.set_num_threads(2)

    base_revision = api.model_info(BASE).sha
    dataset = json.loads(Path(hf_hub_download(EVIDENCE_REPO, "dataset.json", token=token)).read_text())
    natural = json.loads(Path(hf_hub_download(EVIDENCE_REPO, "natural-evaluation-cases.json", token=token)).read_text())
    ember_v3 = json.loads(Path(hf_hub_download(EVIDENCE_REPO, "evaluation/final.json", token=token)).read_text())
    rows = [{**row, "suite": "consumed_v2_confirmation_regression"} for row in dataset["confirmation"]] + natural
    assert len(rows) == 142

    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=base_revision)
    tokenizer.pad_token = tokenizer.eos_token
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        BASE,
        revision=base_revision,
        dtype=torch.bfloat16,
        device_map={"": 0},
        output_loading_info=True,
        key_mapping={r"^model.language_model\.": "model."},
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading.get("mismatched_keys"), loading.get("mismatched_keys")
    assert not loading.get("error_msgs"), loading.get("error_msgs")
    model.eval()

    results = []
    for index, row in enumerate(rows, 1):
        ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=False,
        ).to(model.device)
        started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=96,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        output = tokenizer.decode(generated[0, ids.shape[-1]:], skip_special_tokens=True).strip()
        results.append(
            {
                **row,
                "output": output,
                "exact_match": (output == row["answer"]) if row["scoring"] == "exact" else None,
                "requires_human_review": row["scoring"] == "rubric",
                "seconds": time.monotonic() - started,
            }
        )
        if index % 10 == 0 or index == len(rows):
            exact = [item for item in results if item["exact_match"] is not None]
            print(json.dumps({
                "completed": index,
                "total": len(rows),
                "exact_pass_so_far": sum(item["exact_match"] is True for item in exact),
                "exact_total_so_far": len(exact),
            }), flush=True)

    v3_counts = counts(ember_v3)
    base4_counts = counts(results)
    v3_exact = sum(value["exact_pass"] for value in v3_counts.values())
    base4_exact = sum(value["exact_pass"] for value in base4_counts.values())
    exact_total = sum(value["exact_total"] for value in base4_counts.values())
    summary = {
        "candidate": BASE,
        "candidate_revision": base_revision,
        "system": SYSTEM,
        "seed": 431,
        "generation": {"do_sample": False, "enable_thinking": False, "max_new_tokens": 96},
        "eval_case_count": len(results),
        "exact_total": exact_total,
        "ember_v3_exact_pass": v3_exact,
        "qwen35_4b_base_exact_pass": base4_exact,
        "exact_delta_vs_ember_v3": base4_exact - v3_exact,
        "ember_v3": v3_counts,
        "qwen35_4b_base": base4_counts,
        "families_better_than_v3": [
            key for key in base4_counts
            if base4_counts[key]["exact_pass"] > v3_counts[key]["exact_pass"]
        ],
        "families_worse_than_v3": [
            key for key in base4_counts
            if base4_counts[key]["exact_pass"] < v3_counts[key]["exact_pass"]
        ],
        "human_review_pending": True,
        "training_started": False,
    }
    print("FINAL_SUMMARY", json.dumps(summary), flush=True)

    operations = [
        CommitOperationAdd(
            path_in_repo=f"{OUTPUT_PREFIX}/evaluation.json",
            path_or_fileobj=json.dumps(results, indent=2).encode(),
        ),
        CommitOperationAdd(
            path_in_repo=f"{OUTPUT_PREFIX}/summary.json",
            path_or_fileobj=json.dumps(summary, indent=2).encode(),
        ),
        CommitOperationAdd(
            path_in_repo=f"{OUTPUT_PREFIX}/loading-info.json",
            path_or_fileobj=json.dumps(loading, default=str, indent=2).encode(),
        ),
    ]
    commit = api.create_commit(
        repo_id=EVIDENCE_REPO,
        repo_type="model",
        operations=operations,
        commit_message="Preserve Qwen3.5-4B Ember baseline benchmark",
    )
    print(json.dumps({"evidence_commit": commit.oid, "evidence_url": commit.commit_url}), flush=True)


if __name__ == "__main__":
    main()
