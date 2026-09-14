# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0", "transformers==5.17.0", "peft==0.20.0", "trl==1.13.0", "accelerate==1.15.0", "huggingface-hub==1.31.0", "trackio==0.37.1", "datasets==5.0.1"]
# ///
"""One fixed-budget Qwen adapter experiment; no deployment or automatic retries."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time

MODEL = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
REPO = "Jmiller18899/ember-qwen3.5-2b-sft-v1"
SYSTEM = "You are Ember. Answer the current request directly and concisely. Preserve supplied facts and follow the requested format."


def examples(split, n):
    # Disjoint entities and prompts. Synthetic tests measure this narrow scope only.
    names = {"train": ["Ari", "Bea", "Cleo", "Dane", "Elin"],
             "dev": ["Faye", "Gus", "Hana"],
             "confirmation": ["Imani", "Jules", "Keiko", "Lars"]}[split]
    objects = {"train": ["roster", "diagram", "checklist", "schedule", "draft"],
               "dev": ["outline", "catalog"],
               "confirmation": ["itinerary", "inventory", "brochure"]}[split]
    rows = []
    def add(family, prompt, answer):
        rows.append(dict(id=f"{split}-{family}-{len(rows):04d}", family=family, prompt=prompt, answer=answer))
    for i in range(n):
        name, obj = names[i % len(names)], objects[(i // len(names)) % len(objects)]
        tag = f"{split.upper()}-{i:04d}"
        amount = {"train": 100, "dev": 600, "confirmation": 900}[split] + i
        greeting = f"Hello, {name}! Your reference is {tag}."
        add("greeting", f"Greet the visitor using exactly this sentence: {greeting}", greeting)
        texts = [f"{name} checked {amount} {obj} entries.", f"Warning: {tag} is delayed but still running.",
                 f"Error {tag}: the operation could not finish.", f"Success: {tag} finished without issues."]
        text = texts[i % 4]
        add("copy", f"Copy the following text exactly, without interpreting it: {text}", text)
        add("extraction", f"Return only the reference code from this note: {name} reviewed {amount} entries; reference {tag}.", tag)
        status = [f"Task {tag} finished normally, with all checks passing.",
                  f"Task {tag} is behind schedule but remains operational.",
                  f"Task {tag} failed and cannot proceed.",
                  f"Task {tag} completed successfully without any reported problems.",
                  f"Task {tag} still works, but remaining capacity is critically low.",
                  f"Task {tag} was blocked by invalid input and stopped."][i % 6]
        add("classification", f"Classify this status as success, warning, or error. Output only the label: {status}",
            ["success", "warning", "error"][i % 3])
        prompts = [f"Thank {name} for reviewing the {obj} for {tag}.",
                   f"Write a brief thank-you to {name}, who checked the {obj} for {tag}.",
                   f"Draft a message welcoming {name} to project {tag}.",
                   f"{name} sent this message: 'Thank you for checking the {obj} for {tag}.' Write a short reply."]
        answers = [f"Thank you, {name}, for reviewing the {obj} for {tag}.",
                   f"Thanks, {name}, for checking the {obj} for {tag}.",
                   f"Welcome to project {tag}, {name}!",
                   f"You're welcome, {name}! I'm glad I could help."]
        add("drafting", prompts[i % 4], answers[i % 4])
        add("writing", f"Turn this fact into a short bug title. Use the exact form 'SUBJECT loads slowly (REFERENCE).' Fact: the {obj} loads slowly; reference {tag}.",
            f"{obj.capitalize()} loads slowly ({tag}).")
    return rows


def data():
    sets = {"train": examples("train", 80), "dev": examples("dev", 8), "confirmation": examples("confirmation", 10)}
    prompts = [r["prompt"] for rows in sets.values() for r in rows]
    assert len(prompts) == len(set(prompts))
    assert all(r["answer"] and r["prompt"] for rows in sets.values() for r in rows)
    return sets


def encode(tokenizer, row):
    prompt = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM},
        {"role": "user", "content": row["prompt"]}], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    answer = tokenizer.encode(row["answer"], add_special_tokens=False) + [tokenizer.convert_tokens_to_ids("<|im_end|>")]
    assert len(prompt) + len(answer) <= 256, "No silent truncation allowed"
    return {"input_ids": prompt + answer, "labels": [-100] * len(prompt) + answer}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preflight", action="store_true")
    args = p.parse_args()
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig, TrainerCallback, set_seed
    from trl import SFTConfig, SFTTrainer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    set_seed(431)
    torch.set_num_threads(2)
    sets = data()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    encoded = {k: Dataset.from_list([encode(tokenizer, r) for r in v]) for k, v in sets.items()}
    collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id)
    b = collator([encoded["train"][0], encoded["train"][1]])
    assert (b["labels"] == -100).any() and (b["labels"] != -100).any()
    out = Path("qwen-sft-results")
    out.mkdir(exist_ok=True)
    (out / "dataset.json").write_text(json.dumps(sets, indent=2))
    manifest = {"model": MODEL, "revision": REVISION, "system": SYSTEM,
        "source_commit": os.environ.get("GITHUB_SHA"), "seed": 431,
        "data_sha256": hashlib.sha256((out / "dataset.json").read_bytes()).hexdigest(),
        "counts": {k: dict(Counter(r["family"] for r in v)) for k, v in sets.items()},
        "max_steps": 120, "learning_rate": 0.00002, "production_ready": False,
        "evaluation_scope": "synthetic narrow task checks; drafting requires human review; not general intelligence"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.preflight:
        config = Qwen3_5TextConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
            layer_types=["linear_attention"]*3+["full_attention"],
            rope_parameters={"rope_type":"default", "rope_theta":10000., "partial_rotary_factor":1., "mrope_section":[2,3,3]})
        model = Qwen3_5ForCausalLM(config)
    else:
        assert torch.cuda.is_available(), "GPU required"
        api = HfApi()
        assert api.repo_info(REPO).private, "Output must remain private"
        api.upload_file(repo_id=REPO, path_in_repo="manifest.json", path_or_fileobj=str(out / "manifest.json"))
        api.upload_file(repo_id=REPO, path_in_repo="dataset.json", path_or_fileobj=str(out / "dataset.json"))
        model, info = Qwen3_5ForCausalLM.from_pretrained(MODEL, revision=REVISION,
            dtype=torch.bfloat16, device_map={"":0}, output_loading_info=True,
            key_mapping={r"^model.language_model\.": "model."})
        assert not info["missing_keys"], f"Missing base weights: {info['missing_keys']}"
        assert not info.get("mismatched_keys"), "Mismatched base weights"
        assert not info.get("error_msgs"), "Weight loading failed"
        (out / "loading-info.json").write_text(json.dumps(info, default=str, indent=2))
    model.config.use_cache = False
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    config = SFTConfig(output_dir=str(out), max_steps=2 if args.preflight else 120,
        per_device_train_batch_size=1, gradient_accumulation_steps=1 if args.preflight else 8,
        per_device_eval_batch_size=1, learning_rate=2e-5, warmup_steps=0 if args.preflight else 6,
        bf16=not args.preflight, fp16=False, gradient_checkpointing=not args.preflight,
        gradient_checkpointing_kwargs={"use_reentrant":False}, max_length=256,
        dataset_kwargs={"skip_prepare_dataset":True}, packing=False,
        logging_steps=1 if args.preflight else 5, save_steps=30, save_total_limit=2,
        eval_strategy="no" if args.preflight else "steps", eval_steps=30,
        push_to_hub=not args.preflight, hub_model_id=REPO, hub_private_repo=True,
        hub_strategy="every_save", report_to=[], seed=431, data_seed=431,
        use_cpu=args.preflight)
    trainer = SFTTrainer(model=model, args=config, train_dataset=encoded["train"],
        eval_dataset=encoded["dev"], processing_class=tokenizer, data_collator=collator, peft_config=lora)
    assert all("lora_" in n for n, p in trainer.model.named_parameters() if p.requires_grad)
    if args.preflight:
        result = trainer.train()
        assert 0 < result.training_loss < 100
        print("CPU_PREFLIGHT_PASS: real tokenizer, all data, completion masking, hybrid Qwen forward/backward and LoRA", flush=True)
        return
    import trackio
    trackio.init(project="ember-qwen-sft", name="qwen35-2b-v1", space_id=None, config=manifest)
    class Monitor(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                trackio.log({**logs, "optimizer_step": state.global_step})
                with (out / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps({"step":state.global_step, **logs}) + "\n")
    trainer.add_callback(Monitor())
    def evaluate(label):
        results = []
        trainer.model.eval()
        for row in sets["confirmation"]:
            ids = tokenizer.apply_chat_template([{"role":"system", "content":SYSTEM},
                {"role":"user", "content":row["prompt"]}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=False).to(trainer.model.device)
            start = time.monotonic()
            with torch.inference_mode():
                generated = trainer.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    max_new_tokens=64, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id)
            text = tokenizer.decode(generated[0, ids.shape[-1]:], skip_special_tokens=True).strip()
            results.append({**row,"output":text,"exact_match":text == row["answer"],
                "requires_human_review":row["family"] == "drafting", "seconds":time.monotonic()-start})
        (out / f"{label}.json").write_text(json.dumps(results, indent=2))
        api.upload_file(repo_id=REPO, path_in_repo=f"evaluation/{label}.json", path_or_fileobj=str(out / f"{label}.json"))
        return results
    baseline = evaluate("baseline")
    trainer.train()
    trainer.save_model()
    trainer.push_to_hub(commit_message="Save fixed-budget Ember Qwen SFT adapter; experimental only")
    final = evaluate("final")
    # Strict families may veto regression. Flexible drafts never use exact match as a quality gate.
    counts = lambda rows: {f: sum(r["exact_match"] for r in rows if r["family"] == f)
                          for f in sorted({r["family"] for r in rows})}
    before, after = counts(baseline), counts(final)
    summary = {"baseline_exact":before, "final_exact":after,
        "strict_family_regressions":[f for f in before if f != "drafting" and after[f] < before[f]],
        "drafting_needs_human_review":True, "production_ready":False,
        "comparison":"same pinned BF16 base, tokenizer, prompts and greedy generation; GGUF results are separate"}
    (out / "comparison.json").write_text(json.dumps(summary, indent=2))
    trackio.finish()
    api.upload_folder(repo_id=REPO, folder_path=str(out), path_in_repo="run-evidence",
        allow_patterns=["*.json", "*.jsonl"])
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
