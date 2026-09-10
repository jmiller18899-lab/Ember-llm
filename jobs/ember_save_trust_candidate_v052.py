"""Reproduce and save the bounded Ember v0.0.52 trust-region candidate.

This is a guarded candidate save, not an unconditional production promotion.
It reproduces the published 8e-7 alternating trust-region endpoint from the
pinned v0.0.31 step-479 checkpoint. Publication is refused unless the endpoint
matches the previously observed deterministic result exactly: 16 accepted
cycles, 0 rejected cycles, 0/8 exact placement copies, 22/35 placement tokens,
and 7/8 valid/correct-tool free-running envelopes.

The checkpoint and a verified INT4 companion are uploaded to a new private
Hugging Face candidate repository. The existing promoted Ember repository is
never modified by this job.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

from huggingface_hub import HfApi

from jobs import ember_alternating_trust_region as trust

trace = trust.trace
tiny = trust.tiny
ladder = trust.ladder
protected = trust.protected
data = trust.data
objectives = trust.objectives
base = trust.base

CANDIDATE_VERSION = "0.0.52"
CANDIDATE_MODEL_NAME = "ember-v0.0.52-t4"
EXPECTED_OWNER = "Jmiller18899"
COPY_LR = 8e-7
EXPECTED_ACCEPTED_CYCLES = 16
EXPECTED_REJECTED_CYCLES = 0
EXPECTED_EXACT = 0
EXPECTED_TOKEN_TOP1 = 22
EXPECTED_TOKENS = 35
EXPECTED_JSON = 7
EXPECTED_TOOL = 7


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload(api: HfApi, repo: str, local: Path, remote: str, message: str) -> None:
    api.upload_file(
        repo_id=repo,
        repo_type="model",
        path_or_fileobj=str(local),
        path_in_repo=remote,
        commit_message=message,
    )


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required")

    import torch

    torch.set_num_threads(2)
    cfg = trust.trace.load_config()
    torch.manual_seed(int(cfg["seed"]))
    torch.use_deterministic_algorithms(True)

    api = HfApi(token=token)
    identity = api.whoami()
    owner = str(identity.get("name", "")).strip()
    if owner.casefold() != EXPECTED_OWNER.casefold():
        raise RuntimeError(f"HF_TOKEN belongs to {owner!r}; expected {EXPECTED_OWNER!r}")
    repo = f"{owner}/{CANDIDATE_MODEL_NAME}"

    with tempfile.TemporaryDirectory(prefix="ember-v052-candidate-") as td:
        work = Path(td)
        student, tokenizer, source, _splits, source_ref = base.load_inputs(
            json.loads(base.DEFAULT_CONFIG.read_text()), work / "inputs", torch
        )
        if source.get("step") != 479 or source.get("train_config", {}).get("version") != "0.0.31":
            raise RuntimeError("candidate source is not pinned v0.0.31 step 479")
        if float(student.cfg.dropout) != 0:
            raise RuntimeError("zero dropout required for deterministic reproduction")
        student.to("cpu").eval()
        pristine = copy.deepcopy(student.state_dict())
        source_hash = trace.state_digest(student)
        teacher = copy.deepcopy(student).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

        values = data.target_values(cfg)
        template, template_report = data.v048d.discover_template(
            student, tokenizer, torch, cfg, values["template"]
        )
        development = data.v048d.build_cases(
            {k: values["development"][k] for k in sorted(data.PLACEMENT_SUBTYPES)},
            "v051_place_dev",
        )
        full_baseline, chosen_rows, selected, selected_baseline, baseline_wrong = ladder.choose_cases(
            student, tokenizer, torch, cfg, template, development
        )
        baseline_structure = tiny.structure_probe(student, tokenizer, torch, cfg, selected)
        if not trust.structure_pass(baseline_structure):
            raise RuntimeError("source no longer reproduces 7/8 structural floor")

        tool_id, _ = objectives.token_contract(tokenizer)
        placement_examples = [
            objectives.supervised_example(tokenizer, case, "placement", template, tool_id)
            for case in selected
        ]
        structural_examples, structural_records = protected.prepare_structural_replay(
            teacher, tokenizer, torch, cfg, selected
        )

        rung = trust.run_rung(
            student,
            tokenizer,
            torch,
            cfg,
            template,
            selected,
            baseline_wrong,
            placement_examples,
            structural_examples,
            pristine,
            source_hash,
            COPY_LR,
        )
        final_place = rung["final_selected"]
        final_structure = rung["final_structure"]
        observed = {
            "cycles_accepted": int(rung["cycles_accepted"]),
            "cycles_rejected": int(rung["cycles_rejected"]),
            "exact_top1": int(final_place["exact_top1"]),
            "token_top1": int(final_place["token_top1"]),
            "tokens": int(final_place["tokens"]),
            "json_valid": int(final_structure["envelope_json_valid"]),
            "tool_correct": int(final_structure["tool_name_correct"]),
        }
        expected = {
            "cycles_accepted": EXPECTED_ACCEPTED_CYCLES,
            "cycles_rejected": EXPECTED_REJECTED_CYCLES,
            "exact_top1": EXPECTED_EXACT,
            "token_top1": EXPECTED_TOKEN_TOP1,
            "tokens": EXPECTED_TOKENS,
            "json_valid": EXPECTED_JSON,
            "tool_correct": EXPECTED_TOOL,
        }
        if observed != expected:
            raise RuntimeError(f"v0.0.52 reproduction mismatch: observed={observed}, expected={expected}")

        candidate_hash = trace.state_digest(student)
        if candidate_hash == source_hash:
            raise RuntimeError("candidate model state did not change")

        run_id = f"ember-v052-trust-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        checkpoint_path = work / "best.pt"
        candidate = copy.deepcopy(source)
        candidate["model_state"] = {
            name: tensor.detach().cpu().clone()
            for name, tensor in student.state_dict().items()
        }
        candidate["step"] = EXPECTED_ACCEPTED_CYCLES
        candidate["run_id"] = run_id
        candidate["train_config"] = {
            **dict(source.get("train_config") or {}),
            "version": CANDIDATE_VERSION,
            "phase": "alternating-trust-region-candidate",
            "source_version": "0.0.31",
            "source_step": 479,
            "copy_learning_rate": COPY_LR,
            "repair_learning_rate": trust.REPAIR_LR,
            "accepted_cycles": EXPECTED_ACCEPTED_CYCLES,
            "promotion_authorized": False,
            "production_authorized": False,
        }
        candidate["candidate_metadata"] = {
            "source_state_sha256": source_hash,
            "candidate_state_sha256": candidate_hash,
            "reproduction": observed,
            "strict_copy_gate": {
                "required_exact": trust.ACCEPT_EXACT,
                "observed_exact": observed["exact_top1"],
                "passed": observed["exact_top1"] >= trust.ACCEPT_EXACT,
            },
            "strict_structure_gate": {
                "required_valid": trust.ACCEPT_STRUCTURE,
                "observed_json_valid": observed["json_valid"],
                "observed_tool_correct": observed["tool_correct"],
                "passed": observed["json_valid"] >= trust.ACCEPT_STRUCTURE
                and observed["tool_correct"] >= trust.ACCEPT_STRUCTURE,
            },
        }
        torch.save(candidate, checkpoint_path)

        # Confirm the saved checkpoint carries the exact reproduced model state.
        reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if reloaded.get("train_config", {}).get("version") != CANDIDATE_VERSION:
            raise RuntimeError("saved candidate version mismatch")
        verify_model = copy.deepcopy(student)
        verify_model.load_state_dict(reloaded["model_state"])
        verify_model.eval()
        if trace.state_digest(verify_model) != candidate_hash:
            raise RuntimeError("saved checkpoint state digest mismatch")
        del verify_model

        # Export INT4 using the same package already loaded by the pinned source loader.
        from src.quantize_int4 import export_int4_checkpoint, dequantize_tensor
        from src.model import EmberGPT, ModelConfig
        from src.tokenizer import tokenizer_from_state_dict

        int4_path = work / "best.int4.pt"
        export_int4_checkpoint(str(checkpoint_path), str(int4_path))
        int4 = torch.load(int4_path, map_location="cpu", weights_only=False)
        int4_state = {
            name: dequantize_tensor(value)
            for name, value in int4["quantized_state"].items()
        }
        int4_state.update(int4.get("passthrough_state", {}))
        int4_model = EmberGPT(ModelConfig(**int4["model_config"]))
        int4_model.load_state_dict(int4_state)
        int4_model.eval()
        int4_tokenizer = tokenizer_from_state_dict(int4["tokenizer"])
        prompt_ids = int4_tokenizer.encode("<|system|>You are Ember.<|user|>Ready?<|assistant|>")
        x = torch.tensor([prompt_ids], dtype=torch.long)
        with torch.inference_mode():
            logits, _ = int4_model(x, None)
        if not bool(torch.isfinite(logits).all().item()):
            raise RuntimeError("INT4 reconstruction produced non-finite logits")

        report = {
            "status": "CANDIDATE_SAVED",
            "version": CANDIDATE_VERSION,
            "model_name": CANDIDATE_MODEL_NAME,
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": source_ref,
            "source_state_sha256": source_hash,
            "candidate_state_sha256": candidate_hash,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "int4_sha256": sha256_file(int4_path),
            "int4_bytes": int4_path.stat().st_size,
            "reproduction": observed,
            "template": template_report,
            "selected_case_ids": [case["id"] for case in selected],
            "selection": chosen_rows,
            "selected_baseline": selected_baseline,
            "full_baseline": full_baseline,
            "structural_replay": structural_records,
            "strict_promotion_gate": {
                "copy": observed["exact_top1"] >= trust.ACCEPT_EXACT,
                "structure": observed["json_valid"] >= trust.ACCEPT_STRUCTURE
                and observed["tool_correct"] >= trust.ACCEPT_STRUCTURE,
                "eligible": observed["exact_top1"] >= trust.ACCEPT_EXACT
                and observed["json_valid"] >= trust.ACCEPT_STRUCTURE
                and observed["tool_correct"] >= trust.ACCEPT_STRUCTURE,
            },
            "production_pointer_changed": False,
        }
        report_path = work / "candidate-report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        state = {
            "status": "candidate_saved",
            "version": CANDIDATE_VERSION,
            "run_id": run_id,
            "best_step": EXPECTED_ACCEPTED_CYCLES,
            "candidate_checkpoint": f"checkpoints/{run_id}/best.pt",
            "candidate_int4": f"checkpoints/{run_id}/best.int4.pt",
            "strict_promotion_eligible": bool(report["strict_promotion_gate"]["eligible"]),
            "promotion": "PENDING_FORMAL_EVAL",
            "production_authorized": False,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        state_path = work / "run-state.json"
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
        remote_dir = f"checkpoints/{run_id}"
        upload(api, repo, checkpoint_path, f"{remote_dir}/best.pt", "Save Ember v0.0.52 trust-region candidate")
        upload(api, repo, int4_path, f"{remote_dir}/best.int4.pt", "Save Ember v0.0.52 trust-region candidate INT4")
        upload(api, repo, report_path, "candidate/candidate-report.json", "Record Ember v0.0.52 candidate evidence")
        upload(api, repo, state_path, "run-state.json", "Record Ember v0.0.52 candidate state")

        print("EMBER_V052_CANDIDATE_SAVE=PASS", flush=True)
        print(f"MODEL_REPO={repo}", flush=True)
        print(f"RUN_ID={run_id}", flush=True)
        print(f"CHECKPOINT_PATH={remote_dir}/best.pt", flush=True)
        print(f"INT4_PATH={remote_dir}/best.int4.pt", flush=True)
        print(f"CHECKPOINT_SHA256={report['checkpoint_sha256']}", flush=True)
        print(f"INT4_SHA256={report['int4_sha256']}", flush=True)
        print(f"STRICT_PROMOTION_ELIGIBLE={str(report['strict_promotion_gate']['eligible']).lower()}", flush=True)
        print(json.dumps(observed, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
