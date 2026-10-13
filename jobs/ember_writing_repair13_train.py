# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","accelerate==1.15.0","huggingface-hub==1.31.0","jinja2==3.1.6","trackio"]
# ///
"""Controlled WR13 data revision from Repair2; 52 steps, no promotion.

Use the approved clear shortening examples with the unchanged WR12 loss and
schedule. Measure Repair2 and pinned WR11 before optimizer steps. Persist all
checkpoint responses and retain passing cases against both controls.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from collections import Counter

BRANCH = "codex/ember-writing-repair13-20261013"
OUT = "Jmiller18899/ember-qwen3.5-4b-writing-repair13-20261013-startupfix1"
SOURCE = "Jmiller18899/ember-qwen3.5-4b-repair2"
SOURCE_REV = "daf938bba5d4e6b650ec9d34a2d3ac56706cf549"
ENGINE_COMMIT = "43220b453807782091a9209384091e82fed086f8"
ENGINE_SHA = "e20ccd3ac99de436c4fbd5e586e8bd303201e39965500b137bfcd2fa739d0eeb"
GRADER_COMMIT = "25924014c0e5d5a580a296b2841a1e6f6cbe3bb4"
GRADER_SHA = "e134bf3919ea2871e7a10d3e90de877996f9c631414203762ab09c012bc42a9b"
DATA_SHA = "d4119d8c3d7750a5f8d3bf76ea484df57b51b1556bf2820b279bf1ba56a52456"
SHARED_DATA_SHA = "6b647782292b38d8fa49474692f0b44b2d940e5fdc7a17e52a6e8e65cabe1fe2"
EXPECTED_TRAIN_SHA = "9e5860c1b994c529855a8dbe39a36f8c22f0ea5e4df39793fa057a466088647e"
COMPARATOR = "Jmiller18899/ember-qwen3.5-4b-writing-repair11-20261012"
COMPARATOR_REV = "6efea448e6511cb36dbafcbb1950e884783f9bbf"
COMPARATOR_SUBFOLDER = "candidate"
V3_SHA = "77f1608c1dbaf4cb7ee27a9a57386babf8e00bed1d689f22c42a3379536e6195"
FLAVOR, TIMEOUT = "l4x1", "90m"
LR, STEPS, ACCUM, MAX_LEN, SEED = 7.5e-7, 52, 8, 384, 431
SAVE_EVERY, WARMUP = 13, 4
SEMANTIC_WEIGHT = 0.5
PARENT_COMMIT = "18c817b25bb367bc2a92d2d0fd3a4658f9c1f2fa"
PARENT_DATA_SHA = "7ced8bfc4b8a3c4ef6d6516e2c01e53b4fe20a9eaab34e6fe2755338803e2c54"
ANCHORS = re.compile(r"\b(?:please remember|do not forget|no one may|only if|only after|"
                     r"should|must|may|might|can|will|until|before|after|please)\b", re.I)
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
    "jobs/ember_writing_repair8_data.py": ("9e12e104de220e99b51ee28e5efaa74a95113391",
        "bf9e56c50957bcb359e7c8d20d49c61bc3eb4689eeb078f475a5220a984fd316"),
    "jobs/ember_writing_repair9_data.py": ("d1475425fcdabdf2c490831dedb5d0cfe113a98b",
        "2cdec699a002bff5c020878b0b599072a3346391e6a805e69b728e3eaea426ff"),
}


def verify(raw, sha):
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("Pinned source checksum mismatch")
    return raw


def validate_training_payload(rows):
    import ember_writing_repair13_data as data
    raw = data.sft_bytes(rows)
    verify(raw, EXPECTED_TRAIN_SHA)
    if len(rows) != 416 or sum(r.get("loss") == "copy_margin" for r in rows) != 96:
        raise ValueError("Approved WR13 training composition changed")
    return raw


def benchmark_index(records):
    if len(records) != 744:
        raise ValueError("Benchmark must contain all 744 cases")
    indexed = {}
    for row in records:
        key = (row["suite"], row["id"])
        if key in indexed or not isinstance(row.get("ok"), bool):
            raise ValueError("Invalid or duplicate benchmark case")
        indexed[key] = row
    return indexed


def validate_benchmark_controls(baseline_records, comparator_records):
    baseline, comparator = benchmark_index(baseline_records), benchmark_index(comparator_records)
    if set(baseline) != set(comparator):
        raise ValueError("Control benchmark coverage differs")
    if sum(r["ok"] for r in baseline.values()) != 714 or sum(r["ok"] for r in comparator.values()) != 715:
        raise ValueError("Repair2 or WR11 failed to reproduce its recorded score")
    if any(row["ok"] and not comparator[key]["ok"] for key, row in baseline.items()):
        raise ValueError("WR11 no-regression control failed to reproduce")
    return True


def qualify_benchmark(candidate_records, baseline_records, comparator_records, selected_step, benchmark_copies):
    validate_benchmark_controls(baseline_records, comparator_records)
    before, control, after = (benchmark_index(rows) for rows in (baseline_records, comparator_records, candidate_records))
    if set(after) != set(before):
        raise ValueError("Candidate benchmark coverage differs")
    lost_baseline = [key for key in before if before[key]["ok"] and not after[key]["ok"]]
    lost_wr11 = [key for key in control if control[key]["ok"] and not after[key]["ok"]]
    passes = sum(row["ok"] for row in after.values())
    return dict(qualifies=selected_step > 0 and passes >= 715 and benchmark_copies <= 5
                and not lost_baseline and not lost_wr11, passes=passes, selected_step=selected_step,
                benchmark_copies=benchmark_copies, lost_baseline_passes=lost_baseline, lost_wr11_passes=lost_wr11)


def verify_finished(steps, loss, before_digest, after_digest):
    if steps != STEPS:
        raise ValueError("Training did not finish the scheduled WR13 steps")
    if not math.isfinite(loss):
        raise ValueError("Non-finite training loss")
    if before_digest == after_digest:
        raise ValueError("Adapter weights did not change")
    return True


def validate_launch(env, output_exists):
    ref = env.get("GITHUB_REF")
    local = env.get("WR13_ALLOW_LOCAL_LAUNCH") == "1"
    if output_exists:
        raise ValueError("Output already exists; duplicate launch/retry refused")
    if not re.fullmatch("[0-9a-f]{40}", env.get("WR13_CODE_COMMIT", "")) or not env.get("HF_TOKEN"):
        raise ValueError("Pinned code and existing HF secret are required")
    if local:
        if env.get("WR13_LOCAL_BRANCH") != BRANCH:
            raise ValueError("Local launch must use the isolated WR13 branch")
        return True
    if ref != "refs/heads/" + BRANCH:
        raise ValueError("Launch must use the isolated WR13 branch")
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


def encode_ids(prefix, answer, end, max_len):
    if not prefix or not answer or not isinstance(end, int):
        raise ValueError("Missing prompt, answer or end token")
    target = list(answer) + [end]
    if len(prefix) + len(target) > max_len:
        raise ValueError("No truncation allowed")
    ids = list(prefix) + target
    return {"input_ids": ids, "labels": [-100] * len(prefix) + target, "attention_mask": [1] * len(ids)}


def anchor_mask(answer, offsets):
    """Select every token overlapping a semantic phrase, using real tokenizer offsets."""
    spans = [match.span() for match in ANCHORS.finditer(answer)]
    if not spans:
        raise ValueError("Shortening target has no semantic anchors")
    if not offsets or any(not (0 <= a < b <= len(answer)) for a,b in offsets):
        raise ValueError("Invalid answer token offsets")
    for start, stop in spans:
        for char in range(start, stop):
            if not answer[char].isspace() and not any(a <= char < b for a,b in offsets):
                raise ValueError("Semantic phrase missing tokenizer coverage")
    return [int(any(a < stop and b > start for start,stop in spans)) for a,b in offsets]


def encode_anchor_mask(prefix_length, target_mask):
    if prefix_length < 1 or not target_mask or any(x not in (0,1) for x in target_mask):
        raise ValueError("Invalid semantic mask")
    return [0] * prefix_length + list(target_mask) + [0]


def semantic_nll(logits, labels, mask):
    """Causally shifted CE on selected answer tokens; no prompt or EOS weight."""
    import torch
    if labels.shape != mask.shape or logits.shape[:2] != labels.shape:
        raise ValueError("Semantic loss shape mismatch")
    if torch.any((mask != 0) & (mask != 1)) or torch.any((labels == -100) & (mask != 0)):
        raise ValueError("Semantic loss cannot include prompt tokens")
    selected = mask[:, 1:].to(logits.dtype)
    if selected.sum().item() <= 0:
        raise ValueError("No selected semantic answer tokens")
    per_token = torch.nn.functional.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                   labels[:, 1:].reshape(-1), ignore_index=-100, reduction="none").reshape_as(selected)
    return (per_token * selected).sum() / selected.sum()


def choose_selected(baseline, snapshots, comparator=None):
    """Select on development evidence; baseline wins ties and lost passes are ineligible."""
    def indexed(snapshot):
        rows = snapshot.get("meaning", {}).get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Per-case meaning evidence is required")
        result = {}
        for row in rows:
            ident, status = row.get("id"), row.get("status")
            if not isinstance(ident, str) or ident in result or status not in ("pass", "fail", "review"):
                raise ValueError("Invalid or duplicate development case")
            result[ident] = row
        if sum(r["status"] == "pass" for r in rows) != snapshot["v3_pass"]:
            raise ValueError("Per-case and aggregate pass counts differ")
        return result
    before = indexed(baseline)
    controls = [baseline] + ([comparator] if comparator is not None else [])
    indexed_controls = [indexed(control) for control in controls]
    if any(set(rows) != set(before) for rows in indexed_controls):
        raise ValueError("Comparison model development coverage differs")
    protected = {key for rows in indexed_controls for key, row in rows.items() if row["status"] == "pass"}
    drops = lambda rows: sum("reminder_dropped" in row.get("reasons", []) for row in rows.values())
    critical = lambda row: {x for x in row.get("reasons", [])
                            if x == "reminder_dropped" or x.startswith("force_changed")}
    if not isinstance(baseline.get("copy_ids"), list) or len(set(baseline["copy_ids"])) != baseline["copies"]:
        raise ValueError("Per-case copy evidence is required")
    eligible = [baseline]
    for snap in snapshots:
        after = indexed(snap)
        if set(after) != set(before):
            raise ValueError("Development case coverage differs")
        if snap["step"] <= 0:
            continue
        if not isinstance(snap.get("copy_ids"), list) or len(set(snap["copy_ids"])) != snap["copies"]:
            raise ValueError("Per-case copy evidence is required")
        target_gain = any(critical(before[key]) and after[key]["status"] == "pass" for key in before)
        new_critical = any(critical(after[key]) - critical(control[key])
                           for control in indexed_controls for key in before)
        if (all(snap["copies"] <= control["copies"] and snap["force_fails"] <= control["force_fails"]
                and set(snap["copy_ids"]) <= set(control["copy_ids"]) for control in controls)
                and all(drops(after) <= drops(control) for control in indexed_controls)
                and target_gain and not new_critical
                and all(after[key]["status"] == "pass" for key in protected)):
            eligible.append(snap)
    return max(eligible, key=lambda snap: (snap["v3_pass"], -snap["copies"], -snap["step"]))["step"]


def checkpoint_response_evidence(step, outputs, snapshot, checkpoint_path, checkpoint_revision):
    """Retain exact response text with its frozen grade and checkpoint identity."""
    if snapshot.get("step") != step or not outputs:
        raise ValueError("Checkpoint step or response coverage differs")
    expected_path = "checkpoint-0" if step == 0 else f"checkpoints/step-{step}"
    if checkpoint_path != expected_path or not re.fullmatch(r"[0-9a-f]{40}", checkpoint_revision):
        raise ValueError("Pinned checkpoint identity is required")
    grades = snapshot.get("meaning", {}).get("rows", [])
    indexed = {r["id"]: r for r in grades}
    response_ids = [r["id"] for r in outputs]
    short_ids = {r["id"] for r in outputs if r["family"] == "shortening"}
    if len(indexed) != len(grades) or len(set(response_ids)) != len(response_ids) or set(indexed) != short_ids:
        raise ValueError("Checkpoint response and grade identities differ")
    if sum(r["status"] == "pass" for r in grades) != snapshot["v3_pass"]:
        raise ValueError("Checkpoint pass count differs from its case grades")
    responses = copy.deepcopy(outputs)
    for row in responses:
        if not isinstance(row.get("output"), str):
            raise ValueError("Raw checkpoint response text is required")
        row["semantic_grade"] = copy.deepcopy(indexed.get(row["id"]))
        row["manual_review_pending"] = True
    return dict(schema_version=1, step=step, checkpoint_repo=OUT,
                checkpoint_path=checkpoint_path, checkpoint_revision=checkpoint_revision,
                responses=responses, snapshot=copy.deepcopy(snapshot), automatic_promotion=False)


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


def load_experiment_modules(work, engine, data_raw, v3_raw, grader_raw):
    """Stage pinned local imports before executing WR13 modules.

    The GPU job uploads only this trainer. v3 imports the frozen v2 grader, and
    the data module imports v3, so both files have to be loaded first.
    """
    frozen = load_module("wr13_frozen_engine", engine, work)
    load_module("ember_drafting_repair_candidates_eval", grader_raw, work)
    checker = load_module("ember_meaning_preservation_v3", v3_raw, work)
    load_module("ember_writing_repair12_data", read_source("jobs/ember_writing_repair12_data.py",
                os.environ.get("WR13_CODE_COMMIT", PARENT_COMMIT), SHARED_DATA_SHA), work)
    data = load_module("ember_writing_repair13_data", data_raw, work)
    return frozen, data, checker


def training_never_started(files):
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


def failed_before_optimizer(files):
    names = set(files)
    if "launch.json" not in names:
        return False
    if any(name.startswith(("checkpoints/", "candidate/")) for name in names):
        return False
    finished = {
        "evidence/training-complete.json",
        "evidence/progress.json",
        "evidence/selection.json",
        "evidence/candidate-744.json",
        "evidence/dev-after.json",
    }
    return names.isdisjoint(finished)


def collate_rows(examples):
    """Keep one copy-margin pair intact. Microbatch size is 1."""
    if len(examples) != 1:
        raise ValueError("WR13 microbatch is 1 so each keep-pass pair stays together")
    example = examples[0]
    required = ("input_ids", "labels", "attention_mask", "loss_kind", "semantic_mask",
                "copy_input_ids", "copy_labels", "copy_attention_mask",
                "overedit_input_ids", "overedit_labels", "overedit_attention_mask")
    if any(key not in example for key in required):
        raise ValueError("Keep-pass batch is missing a field")
    if example["loss_kind"] not in (0, 1):
        raise ValueError("loss_kind must be 0 (SFT) or 1 (keep-pass copy margin)")
    if example["loss_kind"] == 1 and not any(label != -100 for label in example["copy_labels"]):
        raise ValueError("Copy-margin row has no source tokens")
    if example["loss_kind"] == 1 and not any(label != -100 for label in example["overedit_labels"]):
        raise ValueError("Keep-pass row has no over-edit tokens")
    mask, labels = example["semantic_mask"], example["labels"]
    if len(mask) != len(labels) or any(x not in (0,1) for x in mask):
        raise ValueError("Invalid semantic mask alignment")
    if any(label == -100 and selected for label,selected in zip(labels,mask)) or mask[-1] != 0:
        raise ValueError("Semantic loss must exclude prompt and end tokens")
    if bool(sum(mask)) != bool(example["loss_kind"]):
        raise ValueError("Only shortening rows receive semantic anchor loss")
    return {key: [example[key]] for key in required}


def prepare():
    revision = os.environ.get("WR13_CODE_COMMIT", "")
    if not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("WR13_CODE_COMMIT must be a pinned Git commit")
    work = Path(tempfile.mkdtemp(prefix="ember-wr13-"))
    engine = read_source("jobs/ember_writing_repair2_train.py", ENGINE_COMMIT, ENGINE_SHA)
    data_raw = read_source("jobs/ember_writing_repair13_data.py", revision, DATA_SHA)
    v3_raw = read_source("jobs/ember_meaning_preservation_v3.py", revision, V3_SHA)
    grader_raw = read_source("jobs/ember_drafting_repair_candidates_eval.py", GRADER_COMMIT, GRADER_SHA)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    T, W, V3 = load_experiment_modules(work, engine, data_raw, v3_raw, grader_raw)
    os.environ["WR2_CODE_COMMIT"] = ENGINE_COMMIT
    old_rows, old_dev, G, E, M, suites = T.load_inputs(work / "history")
    if (T.SOURCE_MODEL, T.SOURCE_REV) != (SOURCE, SOURCE_REV):
        raise ValueError("Frozen parent mismatch")
    import ember_writing_repair1_data as D
    rows = W.build_train_rows(D.retention("train"))
    validate_training_payload(rows)
    encoded_rows = W.encode_examples(rows)
    if len(rows) != STEPS * ACCUM:
        raise ValueError("Optimizer step count does not match one epoch")
    fresh_dev = W.writing("dev")
    history = [r for items in suites.values() for r, _, _ in items] + old_rows + old_dev + D.writing("train")
    for rel, (commit, sha) in HISTORY.items():
        mod = load_module("wr13_hist_" + Path(rel).stem, read_source(rel, commit, sha), work)
        history += mod.writing("train") + mod.writing("dev")
    benchmark_sources = [W.shortening_source(r) for items in suites.values() for r, _, _ in items
                         if W.shortening_source(r)]
    if len(benchmark_sources) != 30:
        raise ValueError("Frozen shortening coverage changed")
    old_data = load_module("wr13_wr10_data", read_source("jobs/ember_writing_repair10_data.py",
                          "d03630b73ab6ba1951b640b440548e1ef6a51055",
                          "0da8b389c2b3905ab3601cd4cc3351357ecd490714a80985e24397ad686cf221"), work)
    parent_data = load_module("wr13_parent_data", read_source("jobs/ember_writing_repair11_data.py",
                             PARENT_COMMIT, PARENT_DATA_SHA), work)
    retained = lambda rs: [r for r in rs if r["family"] != "shortening"]
    if retained(rows) != retained(W.W.build_train_rows(D.retention("train"))):
        raise ValueError("The approved 320 retained rows changed")
    observed_dev = parent_data.writing("dev") + W.W.writing("dev")
    history += old_data.writing("train") + old_data.writing("dev")
    history += parent_data.writing("train") + W.W.writing("train") + observed_dev
    audit = W.audit(rows, fresh_dev, history, benchmark_sources)
    audit["observed_development_rows"] = len(observed_dev)
    dev = observed_dev + fresh_dev
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(T.BASE, revision=T.BASE_REV)
    tok.pad_token = tok.eos_token
    route = G.build_route(E, M, "v3_baseline")
    end = tok.convert_tokens_to_ids("<|im_end|>")
    encoded = []
    for r in encoded_rows:
        kind, value = route(r["prompt"])
        if r["family"] in ("shortening", "recipient") and kind != "model":
            raise ValueError("Writing unexpectedly routed to deterministic tool")
        system = value if kind == "model" else M.O.SYSTEM
        prefix = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": r["prompt"]}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        item = encode_ids(prefix, tok.encode(r["answer"], add_special_tokens=False), end, MAX_LEN)
        item["loss_kind"] = r["loss_kind"]
        if r["loss_kind"] == 1:
            answer_encoding = tok(r["answer"], add_special_tokens=False, return_offsets_mapping=True)
            if answer_encoding["input_ids"] != tok.encode(r["answer"], add_special_tokens=False):
                raise ValueError("Token offsets do not match answer IDs")
            item["semantic_mask"] = encode_anchor_mask(len(prefix), anchor_mask(r["answer"], answer_encoding["offset_mapping"]))
        else:
            item["semantic_mask"] = [0] * len(item["labels"])
        if r["loss_kind"] == 1:
            copied = encode_ids(prefix, tok.encode(r["rejected"], add_special_tokens=False), end, MAX_LEN)
            over = encode_ids(prefix, tok.encode(r["overedit"], add_special_tokens=False), end, MAX_LEN)
            item["copy_input_ids"] = copied["input_ids"]
            item["copy_labels"] = copied["labels"]
            item["copy_attention_mask"] = copied["attention_mask"]
            item["overedit_input_ids"] = over["input_ids"]
            item["overedit_labels"] = over["labels"]
            item["overedit_attention_mask"] = over["attention_mask"]
        else:
            item["copy_input_ids"] = [tok.pad_token_id or 0]
            item["copy_labels"] = [-100]
            item["copy_attention_mask"] = [0]
            item["overedit_input_ids"] = [tok.pad_token_id or 0]
            item["overedit_labels"] = [-100]
            item["overedit_attention_mask"] = [0]
        encoded.append(item)
    if sum(x["loss_kind"] == 1 for x in encoded) != 96:
        raise ValueError("Expected 96 keep-pass copy-margin examples")
    for r in dev:
        if route(r["prompt"])[0] != "model":
            raise ValueError("Fresh development must test the model")
    raw = W.sft_bytes(rows)
    spec = {"experiment": "Writing Repair 13", "data_version": W.VERSION, "code_commit": revision,
            "data_code_sha256": DATA_SHA, "v3_sha256": V3_SHA, "frozen_engine_commit": ENGINE_COMMIT,
            "frozen_engine_sha256": ENGINE_SHA, "source_model": SOURCE, "source_revision": SOURCE_REV,
            "base": T.BASE, "base_revision": T.BASE_REV, "policy_sha256": T.POLICY_SHA,
            "grader_commit": T.GRADER_COMMIT, "benchmark_sha256": T.SUITE_SHA,
            "training_rows": len(rows), "encoded_sequences": len(encoded),
            "fresh_development_rows": len(fresh_dev), "observed_development_rows": len(observed_dev),
            "development_rows": len(dev), "diagnostic_baseline_before_optimizer": True,
            "families": dict(Counter(r["family"] for r in rows)),
            "copy_margin_pairs": sum(x["loss_kind"] == 1 for x in encoded),
            "keep_pass_pairs": sum(x["loss_kind"] == 1 for x in encoded),
            "training_sha256": hashlib.sha256(raw).hexdigest(), "epochs": 1, "learning_rate": LR,
            "optimizer_steps": STEPS, "microbatch": 1, "gradient_accumulation": ACCUM, "warmup_steps": WARMUP,
            "max_length": MAX_LEN, "max_training_tokens": max(len(r["input_ids"]) for r in encoded),
            "seed": SEED, "save_every_steps": SAVE_EVERY, "hardware": FLAVOR, "timeout": TIMEOUT, "output_repo": OUT,
            "automatic_retry": AUTO_RETRY, "automatic_promotion": AUTO_PROMOTION, "production_ready": False,
            "answer_only_loss": True, "copy_margin": True, "keep_pass_margin": True, "contrastive_unlikelihood": False,
            "semantic_anchor_loss_weight": SEMANTIC_WEIGHT,
            "semantic_anchor_tokens": sum(sum(r["semantic_mask"]) for r in encoded),
            "parent_result": "WR12 retained Repair2 at 714/744; WR11 remains the measured 715/744 comparison",
            "comparator_model": COMPARATOR, "comparator_revision": COMPARATOR_REV,
            "comparator_subfolder": COMPARATOR_SUBFOLDER,
            "benchmark_controls_before_optimizer": True, "expected_training_sha256": EXPECTED_TRAIN_SHA,
            "final_holdouts_evaluated": False, "semantic_review_required": True,
            "candidate_selection": "baseline-inclusive; target fixed; per-case pass/copy/critical retention against Repair2 and WR11",
            "runtime_gate": False, "data_audit": audit,
            "acceptance": {"minimum_benchmark_pass": 715, "maximum_benchmark_verbatim_copies": 5,
                           "protected_family_regressions_allowed": 0, "shortening_regressions_allowed": 0,
                           "new_diagnostic_copies_allowed": 0, "v3_false_accept_on_rejected": 0}}
    (work / "train.jsonl").write_bytes(raw)
    (work / "dev.json").write_text(json.dumps(dev, indent=2))
    print("WR13_PREFLIGHT_PASS", json.dumps(spec), flush=True)
    return work, T, W, V3, G, E, M, suites, rows, encoded_rows, encoded, dev, tok, route, spec


def launch(prepared):
    work, T, W, V3, G, E, M, suites, rows, encoded_rows, encoded, dev, tok, route, spec = prepared
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    validate_launch(os.environ, False)
    api = HfApi(token=os.environ["HF_TOKEN"])
    if api.whoami().get("name") != "Jmiller18899":
        raise ValueError("Wrong Hugging Face account")
    if not api.repo_exists(OUT) or not api.model_info(OUT).private:
        raise ValueError("Private storage preflight must pass before launching")
    receipt = json.loads(Path(hf_hub_download(OUT, "storage-preflight.json")).read_text())
    validate_storage(api.list_repo_files(OUT), receipt)
    raw_trainer = Path(__file__).read_bytes()
    manifest = dict(spec, trainer_sha256=hashlib.sha256(raw_trainer).hexdigest())
    api.create_commit(repo_id=OUT, operations=[
        CommitOperationAdd(path_in_repo="launch.json", path_or_fileobj=json.dumps(manifest, indent=2).encode()),
        CommitOperationAdd(path_in_repo="train/sft_train_only.jsonl", path_or_fileobj=(work / "train.jsonl").read_bytes()),
        CommitOperationAdd(path_in_repo="development/fresh-dev.json", path_or_fileobj=(work / "dev.json").read_bytes()),
        CommitOperationAdd(path_in_repo="source/ember_writing_repair13_train.py", path_or_fileobj=raw_trainer),
        CommitOperationAdd(path_in_repo="source/ember_writing_repair13_data.py", path_or_fileobj=Path(W.__file__).read_bytes()),
        CommitOperationAdd(path_in_repo="source/ember_writing_repair12_data.py", path_or_fileobj=Path(W.W.__file__).read_bytes()),
        CommitOperationAdd(path_in_repo="source/ember_meaning_preservation_v3.py",
                           path_or_fileobj=Path(V3.__file__).read_bytes()),
    ], commit_message="Reserve audited WR13 experiment; do not promote")
    job = api.run_uv_job(script=str(Path(__file__).resolve()), python="3.11", flavor=FLAVOR, timeout=TIMEOUT,
                         env={"WR13_CODE_COMMIT": spec["code_commit"], "TOKENIZERS_PARALLELISM": "false"},
                         secrets={"HF_TOKEN": os.environ["HF_TOKEN"]}, labels={"name": "ember-writing-repair13-20261013"})
    receipt = {"job_id": job.id, "url": job.url, "output_repo": OUT, "gpu_timeout": TIMEOUT, "automatic_retry": False}
    print("WR13_GPU_JOB_SUBMITTED", json.dumps(receipt), flush=True)
    api.upload_file(repo_id=OUT, path_in_repo="evidence/launch-submission.json",
                    path_or_fileobj=json.dumps(receipt, indent=2).encode(), commit_message="Record WR13 job submission")


def storage_preflight(prepared):
    """Verify the saved training credential can read both controls and save private results."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=os.environ["HF_TOKEN"])
    if api.whoami().get("name") != "Jmiller18899":
        raise ValueError("Wrong Hugging Face account")
    for repo, revision, path in ((SOURCE, SOURCE_REV, "adapter_config.json"),
                                (COMPARATOR, COMPARATOR_REV, COMPARATOR_SUBFOLDER + "/adapter_config.json")):
        hf_hub_download(repo, path, revision=revision)
    if api.repo_exists(OUT):
        if not api.model_info(OUT).private:
            raise ValueError("WR13 output must be private")
        files = api.list_repo_files(OUT)
        if set(files) - {".gitattributes", "storage-preflight.json"}:
            raise ValueError("WR13 output already contains a launch or results")
    else:
        api.create_repo(OUT, repo_type="model", private=True, exist_ok=False)
    receipt = dict(status="PASS", output_repo=OUT, source_model=SOURCE, source_revision=SOURCE_REV,
                   training_started=False, training_sha256=EXPECTED_TRAIN_SHA,
                   comparator_model=COMPARATOR, comparator_revision=COMPARATOR_REV)
    api.upload_file(repo_id=OUT, path_in_repo="storage-preflight.json",
                    path_or_fileobj=json.dumps(receipt, indent=2).encode(), commit_message="Verify WR13 private checkpoint storage")
    saved = json.loads(Path(hf_hub_download(OUT, "storage-preflight.json")).read_text())
    if saved != receipt or not api.model_info(OUT).private:
        raise ValueError("Private storage receipt failed to round-trip")
    print("WR13_STORAGE_PREFLIGHT_PASS", json.dumps(receipt), flush=True)


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
                               commit_message="WR13 experimental evidence; no promotion")

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

    def lora_parameters():
        found = [(n, p) for n, p in model.named_parameters() if "lora_" in n]
        if not found:
            raise ValueError("LoRA weights missing")
        return found

    def digest():
        h = hashlib.sha256()
        for n, p in lora_parameters():
            h.update(n.encode())
            h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def generate(system, prompt, active_model=None):
        m = model if active_model is None else active_model
        m.eval()
        ids = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=False).to(m.device)
        with torch.inference_mode():
            output = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=96,
                                    do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id)
        return tok.decode(output[0, ids.shape[-1]:], skip_special_tokens=True).strip()

    def dev_outputs(active_model=None):
        return [dict(r, output=generate(route(r["prompt"])[1], r["prompt"], active_model), semantic_grade=None,
                     manual_review_pending=True) for r in dev]

    def meaning_snapshot(step, outputs):
        short = [r for r in outputs if r["family"] == "shortening"]
        stats = W.meaning_stats([(r["id"], r["source"], r["output"]) for r in short])
        lengths = W.length_stats([(r["source"], r["output"]) for r in short])
        force_fails = sum(any(str(reason).startswith("force_changed") for reason in row["reasons"])
                          for row in stats["rows"])
        return {"step": step, "copies": lengths["verbatim_copy"], "not_shorter": lengths["not_shorter"],
                "copy_ids": [r["id"] for r in short if W.norm(r["source"]) == W.norm(r["output"])],
                "v3_pass": stats["counts"].get("pass", 0), "v3_fail": stats["counts"].get("fail", 0),
                "v3_review": stats["counts"].get("review", 0), "force_fails": force_fails,
                "grader_version": V3.GRADER_VERSION, "length": lengths, "meaning": stats}

    def checkpoint(folder, path):
        model.save_pretrained(folder)
        tok.save_pretrained(folder)
        return api.upload_folder(repo_id=OUT, folder_path=str(folder), path_in_repo=path,
                                 commit_message="Persist isolated WR13 checkpoint; no promotion")

    def collate(examples):
        packed = collate_rows(examples)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in packed.items()}

    def sequence_nll(logits, labels):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                                 shift_labels.view(-1), ignore_index=-100)

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
    print("WR13_DIAGNOSTIC_BASELINE_SAVED", json.dumps({"cases": len(baseline_dev), "meaning": baseline_meaning}), flush=True)
    baseline_saved = checkpoint(work / "checkpoint-0", "checkpoint-0")
    upload("evidence/checkpoint-responses/step-0.json", checkpoint_response_evidence(
        0, baseline_dev, baseline_meaning, "checkpoint-0", baseline_saved.oid))
    control_base, control_loading = Qwen3_5ForCausalLM.from_pretrained(T.BASE, revision=T.BASE_REV,
        dtype=torch.bfloat16, device_map={"": 0}, output_loading_info=True,
        key_mapping={r"^model.language_model\.": "model."})
    if control_loading["missing_keys"] or control_loading.get("mismatched_keys") or control_loading.get("error_msgs"):
        raise RuntimeError("WR11 base weights did not load exactly")
    control_model = PeftModel.from_pretrained(control_base, COMPARATOR, revision=COMPARATOR_REV,
        subfolder=COMPARATOR_SUBFOLDER, is_trainable=False).eval()
    control_generate = lambda system, prompt: generate(system, prompt, control_model)
    comparator_scores, comparator_records = T.evaluate(suites, route, control_generate, G)
    upload("evidence/wr11-control-744.json", {"scores": comparator_scores, "records": comparator_records,
                                             "model": COMPARATOR, "revision": COMPARATOR_REV})
    validate_benchmark_controls(before_records, comparator_records)
    comparator_dev = dev_outputs(control_model)
    comparator_meaning = meaning_snapshot(0, comparator_dev)
    upload("evidence/wr11-control-dev.json", comparator_dev)
    upload("evidence/wr11-control-meaning.json", comparator_meaning)
    del control_generate, control_model, control_base
    gc.collect()
    torch.cuda.empty_cache()
    if digest() != before_digest:
        raise RuntimeError("Comparing WR11 changed the starting Repair2 adapter")
    upload("evidence/controls-verified.json", {"repair2_passes": 714, "wr11_passes": 715,
        "benchmark_cases_each": 744, "development_cases_each": len(dev), "training_started": False})
    print("WR13_CONTROLS_VERIFIED", json.dumps({"repair2": 714, "wr11": 715}), flush=True)
    snapshots = [baseline_meaning]
    import trackio
    trackio.init(project="ember-writing", name="writing-repair13-20261013", space_id=None, embed=False,
                 auto_log_gpu=False, auto_log_cpu=False,
                 config={"model": SOURCE, "learning_rate": LR, "epochs": 1, "training_rows": len(encoded)})
    metrics = []

    class Save(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            values = {k: v for k, v in (logs or {}).items() if isinstance(v, (int, float))}
            if values:
                trackio.log(values, step=state.global_step)
                metrics.append(dict(step=state.global_step, **values))
                print("WR13_TRAINING_METRICS", json.dumps(metrics[-1]), flush=True)

        def on_save(self, args, state, control, **kwargs):
            folder = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            saved = api.upload_folder(repo_id=OUT, folder_path=str(folder), path_in_repo=f"checkpoints/step-{state.global_step}",
                              commit_message=f"Persist WR13 step {state.global_step}")
            model.eval()
            outputs = dev_outputs()
            snap = meaning_snapshot(state.global_step, outputs)
            upload(f"evidence/checkpoint-responses/step-{state.global_step}.json", checkpoint_response_evidence(
                state.global_step, outputs, snap, f"checkpoints/step-{state.global_step}", saved.oid))
            snapshots.append(snap)
            upload("evidence/progress.json", {"step": state.global_step, "total_steps": STEPS,
                                              "automatic_promotion": False, "meaning": snap})
            upload("evidence/training-metrics.json", metrics)
            upload("evidence/meaning-snapshots.json", snapshots)
            print("WR13_CHECKPOINT_SAVED", json.dumps(snap), flush=True)
            model.train()

    class MarginTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss_kind = inputs.pop("loss_kind")
            semantic_mask = inputs.pop("semantic_mask")
            copy_ids = inputs.pop("copy_input_ids")
            copy_labels = inputs.pop("copy_labels")
            copy_mask = inputs.pop("copy_attention_mask")
            over_ids = inputs.pop("overedit_input_ids")
            over_labels = inputs.pop("overedit_labels")
            over_mask = inputs.pop("overedit_attention_mask")
            labels = inputs.get("labels")
            outputs = model(**inputs)
            nll = sequence_nll(outputs.logits, labels)
            if int(loss_kind.view(-1)[0].item()) == 1:
                copy_out = model(input_ids=copy_ids, attention_mask=copy_mask)
                over_out = model(input_ids=over_ids, attention_mask=over_mask)
                loss = (nll
                        + W.copy_margin(nll, sequence_nll(copy_out.logits, copy_labels))
                        + W.copy_margin(nll, sequence_nll(over_out.logits, over_labels))
                        + SEMANTIC_WEIGHT * semantic_nll(outputs.logits, labels, semantic_mask))
            else:
                loss = nll
            return (loss, outputs) if return_outputs else loss

    model.train()
    model.config.use_cache = False
    args = TrainingArguments(output_dir=str(work / "training"), num_train_epochs=1,
        per_device_train_batch_size=1, gradient_accumulation_steps=ACCUM, learning_rate=LR, warmup_steps=WARMUP,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps", save_steps=SAVE_EVERY, save_total_limit=5, logging_steps=4, report_to=[],
        seed=SEED, data_seed=SEED, remove_unused_columns=False, dataloader_num_workers=0)
    trainer = MarginTrainer(model=model, args=args, train_dataset=encoded, data_collator=collate, callbacks=[Save()])
    print("WR13_TRAINING_START", json.dumps({"examples": 416, "steps": STEPS, "baseline_reproduced": True}), flush=True)
    try:
        result = trainer.train()
    finally:
        trackio.finish()
        upload("evidence/training-metrics.json", metrics)
    final_digest = digest()
    verify_finished(trainer.state.global_step, float(result.training_loss), before_digest, final_digest)
    selected_step = choose_selected(baseline_meaning, snapshots, comparator_meaning)
    upload("evidence/selection.json", {"baseline": baseline_meaning, "comparator": comparator_meaning, "snapshots": snapshots,
                                       "selected_step": selected_step, "runtime_gate": False})
    if selected_step != trainer.state.global_step:
        reload_from = work / "checkpoint-0" if selected_step == 0 else work / "training" / f"checkpoint-{selected_step}"
        reload_base, loading = Qwen3_5ForCausalLM.from_pretrained(T.BASE, revision=T.BASE_REV, dtype=torch.bfloat16,
            device_map={"": 0}, output_loading_info=True, key_mapping={r"^model.language_model\.": "model."})
        if loading["missing_keys"] or loading.get("mismatched_keys") or loading.get("error_msgs"):
            raise RuntimeError("Base model weights did not reload exactly")
        model = PeftModel.from_pretrained(reload_base, str(reload_from)).eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    candidate_digest = digest()
    saved = checkpoint(work / "candidate", "candidate")
    completion = dict(spec, training_completed=True, steps_completed=trainer.state.global_step,
                      training_loss=float(result.training_loss), candidate_commit=saved.oid,
                      adapter_subfolder="candidate", selected_step=selected_step,
                      starting_digest=before_digest, final_digest=final_digest,
                      candidate_digest=candidate_digest, evaluation_complete=False)
    upload("evidence/training-complete.json", completion)
    print("WR13_TRAINING_COMPLETE", json.dumps({"selected_step": selected_step, "final_step": trainer.state.global_step}), flush=True)
    candidate_dev = dev_outputs()
    upload("evidence/dev-after.json", candidate_dev)
    after, after_records = T.evaluate(suites, route, generate, G)
    upload("evidence/candidate-744.json", {"scores": after, "records": after_records})
    after_meaning = meaning_snapshot(selected_step, candidate_dev)
    upload("evidence/meaning-after.json", after_meaning)
    after_lengths = {"benchmark": W.length_stats([(W.shortening_source(r["row"]), r["output"])
                      for r in after_records if W.shortening_source(r["row"])]),
                     "diagnostic": W.length_stats([(r["source"], r["output"])
                      for r in candidate_dev if r["family"] == "shortening"])}
    upload("evidence/shortening-after.json", after_lengths)
    qualification = qualify_benchmark(after_records, before_records, comparator_records,
                                      selected_step, after_lengths["benchmark"]["verbatim_copy"])
    development_reproduced = selected_step > 0 and choose_selected(baseline_meaning,
        [baseline_meaning, after_meaning], comparator_meaning) == selected_step
    report = dict(completion, evaluation_complete=True, before=before, after=after,
                  wr11_control=comparator_scores, wr11_meaning=comparator_meaning,
                  benchmark_qualification=qualification, development_qualification_reproduced=development_reproduced,
                  qualifying_improvement=qualification["qualifies"] and development_reproduced,
                  comparison=T.compare_records(before_records, after_records),
                  shortening_before=before_lengths, shortening_after=after_lengths,
                  meaning_before=baseline_meaning, meaning_after=after_meaning,
                  selected_step=selected_step, manual_review_completed=False, runtime_gate=False)
    upload("evidence/final-report.json", report)
    card = "---\nbase_model: Qwen/Qwen3.5-4B\nlibrary_name: peft\n---\n# Ember Writing Repair 13\n\nExperimental candidate only. Not promoted or deployed.\nKeep-pass plus copy-margin training from Repair2. Meaning-preservation-v3 is not a runtime gate.\nThe candidate folder is the selected checkpoint. See evidence/final-report.json.\n"
    api.upload_file(repo_id=OUT, path_in_repo="README.md", path_or_fileobj=card.encode(), commit_message="Document WR13 experiment")
    print("WR13_FINAL_REPORT", json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--launch", action="store_true")
    group.add_argument("--storage-preflight", action="store_true")
    args = parser.parse_args()
    prepared = prepare()
    if args.preflight:
        return
    if args.storage_preflight:
        storage_preflight(prepared)
    elif args.launch:
        launch(prepared)
    else:
        train(prepared)


if __name__ == "__main__":
    main()
