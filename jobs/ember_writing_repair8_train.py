# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","accelerate==1.15.0","huggingface-hub==1.31.0","jinja2==3.1.6","trackio"]
# ///
"""One WR8 meaning-aware experiment: Repair2 parent, v3 contrastive loss, no promotion.

Reuse the checksum-pinned WR2 benchmark engine, not its candidate weights.
v3 is a training/selection checker only. Frozen 744-case scores stay on v2.
CPU --preflight performs no Hub writes. --launch uses the existing secret.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from collections import Counter

BRANCH = "cursor/ember-meaning-wr8-4cca"
OUT = "Jmiller18899/ember-qwen3.5-4b-writing-repair8-20261009"
SOURCE = "Jmiller18899/ember-qwen3.5-4b-repair2"
SOURCE_REV = "daf938bba5d4e6b650ec9d34a2d3ac56706cf549"
ENGINE_COMMIT = "43220b453807782091a9209384091e82fed086f8"
ENGINE_SHA = "e20ccd3ac99de436c4fbd5e586e8bd303201e39965500b137bfcd2fa739d0eeb"
DATA_SHA = "bf9e56c50957bcb359e7c8d20d49c61bc3eb4689eeb078f475a5220a984fd316"
V3_SHA = "77f1608c1dbaf4cb7ee27a9a57386babf8e00bed1d689f22c42a3379536e6195"
FLAVOR, TIMEOUT = "l4x1", "90m"
LR, STEPS, ACCUM, MAX_LEN, SEED = 7.5e-7, 128, 4, 384, 431
AUTO_PROMOTION = False
AUTO_RETRY = False
HISTORY = {
    "jobs/ember_writing_repair3_data.py": ("68302b05c47825c33509ed9c5047f57c22f40842",
        "db60589c1a150d4f1feb15762e2bdc7914ee5dd6d7ed9b5fb8038019b4cbd1a1"),
    "jobs/ember_writing_repair4_data.py": ("5f2caf0360cd8e878500180abebc08aa4b1f4cf3",
        "ea73fe0e4154778f1f1de8cf99e4b7ed876b42163780e5c67db33bf8bf9eaf66"),
    "jobs/ember_writing_repair5_data.py": ("4e406a094c0494e684eea9850c70242c441bc3ef",
        "a41ad5e8194e1280d310247450d2ec5d8d167927727103ffa8181dbc25c516fa"),
    "jobs/ember_writing_repair6_data.py": ("fb7b0cfe692cc409cde62ea5caf1cb5b6f391773",
        "a6cbaaf11625d616606bc9fd902c09c3cd9156a3d2576de89f2b5a12fc5fcf01"),
    "jobs/ember_writing_repair7_data.py": ("7d9310a9b02c9833a536ca424d8491d9f7ae73a3",
        "67d328bc54e666db6a4f05cbce368f1510a6a2f39c9387d1e04efe1219e3a061"),
}


def verify(raw, sha):
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("Pinned source checksum mismatch")
    return raw


def validate_launch(env, output_exists):
    ref = env.get("GITHUB_REF")
    local = env.get("WR8_ALLOW_LOCAL_LAUNCH") == "1"
    if output_exists:
        raise ValueError("Output already exists; duplicate launch/retry refused")
    if not re.fullmatch("[0-9a-f]{40}", env.get("WR8_CODE_COMMIT", "")) or not env.get("HF_TOKEN"):
        raise ValueError("Pinned code and existing HF secret are required")
    if local:
        if env.get("WR8_LOCAL_BRANCH") != BRANCH:
            raise ValueError("Local launch must use the isolated WR8 branch")
        return True
    if ref != "refs/heads/" + BRANCH:
        raise ValueError("Launch must use the isolated WR8 branch")
    if env.get("GITHUB_RUN_ATTEMPT") != "1":
        raise ValueError("Automatic or manual workflow reruns cannot relaunch training")
    return True


def validate_storage(files, receipt):
    allowed = {".gitattributes", "storage-preflight.json"}
    if set(files) - allowed or "storage-preflight.json" not in files:
        raise ValueError("Output has a launch or candidate; refusing duplicate training")
    expected = {"status": "PASS", "output_repo": OUT, "source_model": SOURCE,
                "source_revision": SOURCE_REV, "training_started": False}
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("Private storage reservation mismatch")


def encode_ids(prefix, answer, end, max_len, kind):
    if not prefix or not answer or not isinstance(end, int):
        raise ValueError("Missing prompt, answer or end token")
    target = list(answer) + [end]
    if len(prefix) + len(target) > max_len:
        raise ValueError("No truncation allowed")
    if kind not in (0, 1):
        raise ValueError("kind must be 0 (SFT) or 1 (unlikelihood)")
    return {"input_ids": list(prefix) + target, "labels": [-100] * len(prefix) + target, "kind": kind}


def choose_selected(baseline, snapshots):
    eligible = [s for s in snapshots
                if s["step"] > 0 and s["copies"] <= baseline["copies"]
                and s["force_fails"] <= baseline["force_fails"]]
    if not eligible:
        return 0
    return max(eligible, key=lambda s: (s["v3_pass"] - s["copies"], -s["step"]))["step"]


def read_source(path, commit, sha):
    import urllib.request
    local = Path(__file__).resolve().parent / Path(path).name
    if local.is_file():
        return verify(local.read_bytes(), sha)
    url = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/" + commit + "/" + path
    return verify(urllib.request.urlopen(url, timeout=60).read(), sha)


def load_module(name, raw, folder):
    path = folder / (name + ".py")
    path.write_bytes(raw)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def load_experiment_modules(work, engine, data_raw, v3_raw):
    """Stage the v3 checker before the data module imports it.

    The GPU job uploads only this trainer. Sibling files are fetched into a temp
    directory, so `import ember_meaning_preservation_v3` cannot see jobs/.
    """
    frozen = load_module("wr8_frozen_engine", engine, work)
    checker = load_module("ember_meaning_preservation_v3", v3_raw, work)
    inserted = str(work) not in sys.path
    if inserted:
        sys.path.insert(0, str(work))
    try:
        data = load_module("wr8_data", data_raw, work)
    finally:
        if inserted and sys.path and sys.path[0] == str(work):
            sys.path.pop(0)
    return frozen, data, checker


def training_never_started(files):
    """True when a reservation exists but no optimizer evidence was written."""
    names = set(files)
    if "launch.json" not in names:
        return False
    if any(name.endswith(".safetensors") for name in names):
        return False
    if any(name.startswith(("checkpoint", "checkpoints/", "candidate/")) for name in names):
        return False
    started = {
        "evidence/run-spec.json",
        "evidence/training-complete.json",
        "evidence/baseline-744.json",
        "evidence/training-metrics.json",
    }
    return names.isdisjoint(started)


def prepare():
    revision = os.environ.get("WR8_CODE_COMMIT", "")
    if not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("WR8_CODE_COMMIT must be a pinned Git commit")
    work = Path(tempfile.mkdtemp(prefix="ember-wr8-"))
    engine = read_source("jobs/ember_writing_repair2_train.py", ENGINE_COMMIT, ENGINE_SHA)
    data_raw = read_source("jobs/ember_writing_repair8_data.py", revision, DATA_SHA)
    v3_raw = read_source("jobs/ember_meaning_preservation_v3.py", revision, V3_SHA)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    T, W, V3 = load_experiment_modules(work, engine, data_raw, v3_raw)
    os.environ["WR2_CODE_COMMIT"] = ENGINE_COMMIT
    old_rows, old_dev, G, E, M, suites = T.load_inputs(work / "history")
    if (T.SOURCE_MODEL, T.SOURCE_REV) != (SOURCE, SOURCE_REV):
        raise ValueError("Frozen parent mismatch")
    import ember_writing_repair1_data as D
    rows = W.build_train_rows(D.retention("train"))
    encoded_rows = W.encode_pairs(rows)
    dev = W.writing("dev")
    history = [r for items in suites.values() for r, _, _ in items] + old_rows + old_dev + D.writing("train")
    for rel, (commit, sha) in HISTORY.items():
        mod = load_module("wr8_hist_" + Path(rel).stem, read_source(rel, commit, sha), work)
        history += mod.writing("train") + mod.writing("dev")
    benchmark_sources = [W.shortening_source(r) for items in suites.values() for r, _, _ in items
                         if W.shortening_source(r)]
    if len(benchmark_sources) != 30:
        raise ValueError("Frozen shortening coverage changed")
    audit = W.audit(rows, dev, history, benchmark_sources)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(T.BASE, revision=T.BASE_REV)
    tok.pad_token = tok.eos_token
    route = G.build_route(E, M, "v3_baseline")
    encoded = []
    for r in encoded_rows:
        kind, value = route(r["prompt"])
        if r["family"] in ("shortening", "recipient") and kind != "model":
            raise ValueError("Writing unexpectedly routed to deterministic tool")
        system = value if kind == "model" else M.O.SYSTEM
        prefix = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": r["prompt"]}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        encoded.append(encode_ids(prefix, tok.encode(r["answer"], add_special_tokens=False),
                                  tok.convert_tokens_to_ids("<|im_end|>"), MAX_LEN, r["kind"]))
    if sum(x["kind"] == 1 for x in encoded) != 96:
        raise ValueError("Expected 96 unlikelihood sequences")
    for r in dev:
        if route(r["prompt"])[0] != "model":
            raise ValueError("Fresh development must test the model")
    raw = W.sft_bytes(rows)
    spec = {"experiment": "Writing Repair 8", "data_version": W.VERSION, "code_commit": revision,
            "data_code_sha256": DATA_SHA, "v3_sha256": V3_SHA, "frozen_engine_commit": ENGINE_COMMIT,
            "frozen_engine_sha256": ENGINE_SHA, "source_model": SOURCE, "source_revision": SOURCE_REV,
            "base": T.BASE, "base_revision": T.BASE_REV, "policy_sha256": T.POLICY_SHA,
            "grader_commit": T.GRADER_COMMIT, "benchmark_sha256": T.SUITE_SHA,
            "training_rows": len(rows), "encoded_sequences": len(encoded),
            "fresh_development_rows": len(dev), "diagnostic_baseline_before_optimizer": True,
            "families": dict(Counter(r["family"] for r in rows)),
            "unlikelihood_sequences": sum(x["kind"] == 1 for x in encoded),
            "training_sha256": hashlib.sha256(raw).hexdigest(), "epochs": 1, "learning_rate": LR,
            "optimizer_steps": STEPS, "microbatch": 1, "gradient_accumulation": ACCUM, "warmup_steps": 8,
            "max_length": MAX_LEN, "max_training_tokens": max(len(r["input_ids"]) for r in encoded),
            "seed": SEED, "save_every_steps": 32, "hardware": FLAVOR, "timeout": TIMEOUT, "output_repo": OUT,
            "automatic_retry": AUTO_RETRY, "automatic_promotion": AUTO_PROMOTION, "production_ready": False,
            "answer_only_loss": True, "contrastive_unlikelihood": True,
            "final_holdouts_evaluated": False, "semantic_review_required": True,
            "candidate_selection": "v3 development shortening; copies and force-change floors",
            "runtime_gate": False, "data_audit": audit,
            "acceptance": {"minimum_benchmark_pass": 715, "maximum_benchmark_verbatim_copies": 5,
                           "protected_family_regressions_allowed": 0, "shortening_regressions_allowed": 0,
                           "new_diagnostic_copies_allowed": 0, "v3_false_accept_on_rejected": 0}}
    (work / "train.jsonl").write_bytes(raw)
    (work / "dev.json").write_text(json.dumps(dev, indent=2))
    print("WR8_PREFLIGHT_PASS", json.dumps(spec), flush=True)
    return work, T, W, V3, G, E, M, suites, rows, encoded_rows, encoded, dev, tok, route, spec


def launch(prepared):
    work, T, W, V3, G, E, M, suites, rows, encoded_rows, encoded, dev, tok, route, spec = prepared
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    validate_launch(os.environ, False)
    api = HfApi(token=os.environ["HF_TOKEN"])
    if api.whoami().get("name") != "Jmiller18899":
        raise ValueError("Wrong Hugging Face account")
    if api.repo_exists(OUT):
        if not api.model_info(OUT).private:
            raise ValueError("Candidate output must stay private")
        files = api.list_repo_files(OUT)
        if training_never_started(files):
            prior = json.loads(Path(hf_hub_download(OUT, "evidence/launch-submission.json")).read_text())
            prior_job = api.inspect_job(job_id=prior["job_id"])
            if getattr(prior_job.status, "stage", None) not in ("ERROR", "CANCELED", "CANCELLED"):
                raise ValueError("Previous WR8 job is still active or already finished training")
        else:
            receipt = json.loads(Path(hf_hub_download(OUT, "storage-preflight.json")).read_text())
            validate_storage(files, receipt)
    else:
        api.create_repo(OUT, repo_type="model", private=True, exist_ok=False)
    raw_trainer = Path(__file__).read_bytes()
    manifest = dict(spec, trainer_sha256=hashlib.sha256(raw_trainer).hexdigest())
    api.create_commit(repo_id=OUT, operations=[
        CommitOperationAdd(path_in_repo="launch.json", path_or_fileobj=json.dumps(manifest, indent=2).encode()),
        CommitOperationAdd(path_in_repo="train/sft_train_only.jsonl", path_or_fileobj=(work / "train.jsonl").read_bytes()),
        CommitOperationAdd(path_in_repo="development/fresh-dev.json", path_or_fileobj=(work / "dev.json").read_bytes()),
        CommitOperationAdd(path_in_repo="source/ember_writing_repair8_train.py", path_or_fileobj=raw_trainer),
        CommitOperationAdd(path_in_repo="source/ember_writing_repair8_data.py", path_or_fileobj=Path(W.__file__).read_bytes()),
        CommitOperationAdd(path_in_repo="source/ember_meaning_preservation_v3.py",
                           path_or_fileobj=Path(V3.__file__).read_bytes()),
    ], commit_message="Reserve audited WR8 experiment; do not promote")
    job = api.run_uv_job(script=str(Path(__file__).resolve()), python="3.11", flavor=FLAVOR, timeout=TIMEOUT,
                         env={"WR8_CODE_COMMIT": spec["code_commit"], "TOKENIZERS_PARALLELISM": "false"},
                         secrets={"HF_TOKEN": os.environ["HF_TOKEN"]}, labels={"name": "ember-writing-repair8-20261009"})
    receipt = {"job_id": job.id, "url": job.url, "output_repo": OUT, "gpu_timeout": TIMEOUT, "automatic_retry": False}
    print("WR8_GPU_JOB_SUBMITTED", json.dumps(receipt), flush=True)
    api.upload_file(repo_id=OUT, path_in_repo="evidence/launch-submission.json",
                    path_or_fileobj=json.dumps(receipt, indent=2).encode(), commit_message="Record WR8 job submission")


def train(prepared):
    work, T, W, V3, G, E, M, suites, rows, encoded_rows, encoded, dev, tok, route, spec = prepared
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from peft import PeftModel
    from transformers import Qwen3_5ForCausalLM, Trainer, TrainingArguments, TrainerCallback, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    api = HfApi(token=os.environ["HF_TOKEN"])
    if api.whoami().get("name") != "Jmiller18899":
        raise ValueError("Wrong Hugging Face account")
    reservation = json.loads(Path(hf_hub_download(OUT, "launch.json")).read_text())
    for key, value in spec.items():
        if reservation.get(key) != value:
            raise ValueError("Launch reservation mismatch: " + key)
    verify(Path(__file__).read_bytes(), reservation["trainer_sha256"])
    if not api.model_info(OUT).private or any(f.endswith(".safetensors") for f in api.list_repo_files(OUT)):
        raise ValueError("Require fresh private output; refusing overwrite/retry")

    def upload(path, data):
        return api.upload_file(repo_id=OUT, path_in_repo=path,
                               path_or_fileobj=json.dumps(data, ensure_ascii=False, indent=2).encode(),
                               commit_message="WR8 experimental evidence; no promotion")

    upload("evidence/run-spec.json", spec)
    set_seed(SEED)
    base, loading = Qwen3_5ForCausalLM.from_pretrained(T.BASE, revision=T.BASE_REV, dtype=torch.bfloat16,
        device_map={"": 0}, output_loading_info=True, key_mapping={r"^model.language_model\.": "model."})
    if loading["missing_keys"] or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise RuntimeError("Base model weights did not load exactly")
    model = PeftModel.from_pretrained(base, SOURCE, revision=SOURCE_REV, is_trainable=True)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n for n, p in trainable):
        raise ValueError("Only the existing LoRA adapter may change")

    def digest():
        h = hashlib.sha256()
        for n, p in trainable:
            h.update(n.encode())
            h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def generate(system, prompt):
        model.eval()
        ids = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=False).to(model.device)
        with torch.inference_mode():
            output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=96,
                                    do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id)
        return tok.decode(output[0, ids.shape[-1]:], skip_special_tokens=True).strip()

    def dev_outputs():
        return [dict(r, output=generate(route(r["prompt"])[1], r["prompt"]), semantic_grade=None,
                     manual_review_pending=True) for r in dev]

    def meaning_snapshot(step, outputs):
        short = [r for r in outputs if r["family"] == "shortening"]
        stats = W.meaning_stats([(r["id"], r["source"], r["output"]) for r in short])
        lengths = W.length_stats([(r["source"], r["output"]) for r in short])
        force_fails = sum(any(str(reason).startswith("force_changed") for reason in row["reasons"])
                          for row in stats["rows"])
        return {"step": step, "copies": lengths["verbatim_copy"], "not_shorter": lengths["not_shorter"],
                "v3_pass": stats["counts"].get("pass", 0), "v3_fail": stats["counts"].get("fail", 0),
                "v3_review": stats["counts"].get("review", 0), "force_fails": force_fails,
                "grader_version": V3.GRADER_VERSION, "length": lengths, "meaning": stats}

    def checkpoint(folder, path):
        model.save_pretrained(folder)
        tok.save_pretrained(folder)
        return api.upload_folder(repo_id=OUT, folder_path=str(folder), path_in_repo=path,
                                 commit_message="Persist isolated WR8 checkpoint; no promotion")

    before_digest = digest()
    before, before_records = T.evaluate(suites, route, generate, G)
    upload("evidence/baseline-744.json", {"scores": before, "records": before_records})
    if {s: x["pass"] for s, x in before.items()} != T.CORRECTED or sum(x["review"] for x in before.values()) != 4:
        raise RuntimeError("Frozen 714/744 baseline failed to reproduce; no training performed")
    baseline_dev = dev_outputs()
    upload("evidence/dev-before.json", baseline_dev)
    baseline_meaning = meaning_snapshot(0, baseline_dev)
    upload("evidence/meaning-before.json", baseline_meaning)
    before_lengths = {"benchmark": W.length_stats([(W.shortening_source(r["row"]), r["output"])
                       for r in before_records if W.shortening_source(r["row"])]),
                      "diagnostic": W.length_stats([(r["source"], r["output"])
                       for r in baseline_dev if r["family"] == "shortening"])}
    upload("evidence/shortening-before.json", before_lengths)
    print("WR8_DIAGNOSTIC_BASELINE_SAVED", json.dumps({"cases": len(baseline_dev), "meaning": baseline_meaning}), flush=True)
    checkpoint(work / "checkpoint-0", "checkpoint-0")
    snapshots = [baseline_meaning]
    import trackio
    trackio.init(project="ember-writing", name="writing-repair8-20261009", space_id=None, embed=False,
                 auto_log_gpu=False, auto_log_cpu=False,
                 config={"model": SOURCE, "learning_rate": LR, "epochs": 1, "training_rows": len(encoded)})
    metrics = []

    class Save(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            values = {k: v for k, v in (logs or {}).items() if isinstance(v, (int, float))}
            if values:
                trackio.log(values, step=state.global_step)
                metrics.append(dict(step=state.global_step, **values))
                print("WR8_TRAINING_METRICS", json.dumps(metrics[-1]), flush=True)

        def on_save(self, args, state, control, **kwargs):
            folder = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            api.upload_folder(repo_id=OUT, folder_path=str(folder), path_in_repo=f"checkpoints/step-{state.global_step}",
                              commit_message=f"Persist WR8 step {state.global_step}")
            model.eval()
            snap = meaning_snapshot(state.global_step, dev_outputs())
            snapshots.append(snap)
            upload("evidence/progress.json", {"step": state.global_step, "total_steps": STEPS,
                                              "automatic_promotion": False, "meaning": snap})
            upload("evidence/training-metrics.json", metrics)
            upload("evidence/meaning-snapshots.json", snapshots)
            print("WR8_CHECKPOINT_SAVED", json.dumps(snap), flush=True)
            model.train()

    def collate(examples):
        kinds = torch.tensor([x["kind"] for x in examples], dtype=torch.long)
        padded = T.pad_batch([{k: v for k, v in x.items() if k != "kind"}], tok.pad_token_id)
        batch = {k: torch.tensor(v, dtype=torch.long) for k, v in padded.items()}
        batch["kind"] = kinds
        return batch

    class ContrastiveTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            kinds = inputs.pop("kind")
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            losses = []
            for i, kind in enumerate(kinds.tolist()):
                li, la = shift_logits[i], shift_labels[i]
                mask = la != -100
                if not bool(mask.any()):
                    losses.append(li.sum() * 0)
                    continue
                if kind == 1:
                    logp = torch.nn.functional.log_softmax(li, dim=-1)
                    token_p = logp.gather(-1, la.clamp(min=0).unsqueeze(-1)).squeeze(-1).exp().clamp(max=1 - 1e-5)
                    nll = -torch.log1p(-token_p)
                    losses.append(nll[mask].mean())
                else:
                    losses.append(torch.nn.functional.cross_entropy(li, la, ignore_index=-100))
            loss = torch.stack(losses).mean()
            return (loss, outputs) if return_outputs else loss

    model.train()
    model.config.use_cache = False
    args = TrainingArguments(output_dir=str(work / "training"), num_train_epochs=1,
        per_device_train_batch_size=1, gradient_accumulation_steps=ACCUM, learning_rate=LR, warmup_steps=8,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps", save_steps=32, save_total_limit=5, logging_steps=8, report_to=[],
        seed=SEED, data_seed=SEED, remove_unused_columns=False, dataloader_num_workers=0)
    trainer = ContrastiveTrainer(model=model, args=args, train_dataset=encoded, data_collator=collate, callbacks=[Save()])
    print("WR8_TRAINING_START", json.dumps({"encoded": 512, "steps": STEPS, "baseline_reproduced": True}), flush=True)
    try:
        result = trainer.train()
    finally:
        trackio.finish()
        upload("evidence/training-metrics.json", metrics)
    after_digest = digest()
    T.verify_training(trainer.state.global_step, float(result.training_loss), before_digest, after_digest)
    saved = checkpoint(work / "candidate", "candidate")
    selected_step = choose_selected(baseline_meaning, snapshots)
    upload("evidence/selection.json", {"baseline": baseline_meaning, "snapshots": snapshots,
                                       "selected_step": selected_step, "runtime_gate": False})
    completion = dict(spec, training_completed=True, steps_completed=trainer.state.global_step,
                      training_loss=float(result.training_loss), candidate_commit=saved.oid,
                      adapter_subfolder="candidate", selected_step=selected_step,
                      starting_digest=before_digest, candidate_digest=after_digest, evaluation_complete=False)
    upload("evidence/training-complete.json", completion)
    print("WR8_TRAINING_COMPLETE", json.dumps({"selected_step": selected_step, "final_step": trainer.state.global_step}), flush=True)
    if selected_step != trainer.state.global_step:
        reload_from = work / "checkpoint-0" if selected_step == 0 else work / "training" / f"checkpoint-{selected_step}"
        reload_base, loading = Qwen3_5ForCausalLM.from_pretrained(T.BASE, revision=T.BASE_REV, dtype=torch.bfloat16,
            device_map={"": 0}, output_loading_info=True, key_mapping={r"^model.language_model\.": "model."})
        if loading["missing_keys"] or loading.get("mismatched_keys") or loading.get("error_msgs"):
            raise RuntimeError("Base model weights did not reload exactly")
        model = PeftModel.from_pretrained(reload_base, str(reload_from)).eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    candidate_dev = dev_outputs()
    upload("evidence/dev-after.json", candidate_dev)
    after, after_records = T.evaluate(suites, route, generate, G)
    upload("evidence/candidate-744.json", {"scores": after, "records": after_records})
    after_meaning = meaning_snapshot(trainer.state.global_step, candidate_dev)
    upload("evidence/meaning-after.json", after_meaning)
    after_lengths = {"benchmark": W.length_stats([(W.shortening_source(r["row"]), r["output"])
                      for r in after_records if W.shortening_source(r["row"])]),
                     "diagnostic": W.length_stats([(r["source"], r["output"])
                      for r in candidate_dev if r["family"] == "shortening"])}
    upload("evidence/shortening-after.json", after_lengths)
    report = dict(completion, evaluation_complete=True, before=before, after=after,
                  comparison=T.compare_records(before_records, after_records),
                  shortening_before=before_lengths, shortening_after=after_lengths,
                  meaning_before=baseline_meaning, meaning_after=after_meaning,
                  selected_step=selected_step, manual_review_completed=False, runtime_gate=False)
    upload("evidence/final-report.json", report)
    card = "---\nbase_model: Qwen/Qwen3.5-4B\nlibrary_name: peft\n---\n# Ember Writing Repair 8\n\nExperimental candidate only. Not promoted or deployed.\nMeaning-preservation-v3 is a training/selection checker, not a runtime gate.\nSee evidence/final-report.json. Final holdouts remain unused.\n"
    api.upload_file(repo_id=OUT, path_in_repo="README.md", path_or_fileobj=card.encode(), commit_message="Document WR8 experiment")
    print("WR8_FINAL_REPORT", json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    prepared = prepare()
    if args.preflight:
        return
    if args.launch:
        launch(prepared)
    else:
        train(prepared)


if __name__ == "__main__":
    main()
