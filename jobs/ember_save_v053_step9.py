"""Reproduce and save the Ember v0.0.53 step-9 routing-anchor operating point.

Starts from the saved v0.0.52 candidate and applies exactly nine rank-margin
first-token anchor updates at LR 6.4e-7. Publication is refused unless the
previously observed operating point is reproduced: zero accidental direct tool
entries, 4/4 explicit tool entries, 4/4 direct generation routing, 4/4 tool
generation routing, 22/35 placement tokens, and 8/8 JSON/correct-tool structure.

Saves a private v0.0.53 candidate plus INT4 companion. It does not promote or
change any production pointer.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

from huggingface_hub import HfApi
import torch

from jobs import ember_hf_eval as ev
from jobs import ember_v052_first_token_logits as ft
from jobs import ember_v053_first_token_anchor as base
from jobs import ember_v053_first_token_margin_anchor as margin
from jobs import ember_alternating_trust_region as trust

VERSION = "0.0.53"
MODEL_NAME = "ember-v0.0.53-t4"
SOURCE_NAME = "ember-v0.0.52-t4"
LR = 6.4e-7
STEPS = 9
EXPECTED_COPY_TOKENS = 22
EXPECTED_TOKENS = 35
EXPECTED_STRUCTURE = 8


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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

    torch.set_num_threads(2)
    torch.manual_seed(1337)
    torch.use_deterministic_algorithms(True)

    api = HfApi(token=token)
    owner = api.whoami()["name"]
    source_repo = f"{owner}/{SOURCE_NAME}"
    candidate_repo = f"{owner}/{MODEL_NAME}"

    with tempfile.TemporaryDirectory(prefix="ember-v053-step9-save-") as td:
        work = Path(td)
        archive = ev.download_verified(ev.PACKAGE_URL, work / "ember.zip", ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        import sys
        sys.path.insert(0, str(work / "src" / "ember"))

        spec = json.loads(Path("config/ember_v0.0.8_eval.json").read_text())
        routing_cases = [c for c in spec["cases"] if c["kind"] in {"direct_response", "tool_call"}]

        source_path = ft.candidate_checkpoint_path(api, source_repo)
        model, tokenizer, source_ckpt = ft.load_model(source_repo, source_path, work, work / "src" / "ember")
        if source_ckpt.get("train_config", {}).get("version") != "0.0.52":
            raise RuntimeError("source is not saved v0.0.52 candidate")
        model.eval()
        source_hash = trust.trace.state_digest(model)
        pristine = copy.deepcopy(model.state_dict())

        cfg, template, template_report, selected, selected_ids, source_ref, *_ = base.build_placement_fixture(work)
        baseline_route = base.routing_probe(model, tokenizer, routing_cases)
        baseline_place = base.placement_probe(model, tokenizer, selected, template)
        baseline_structure = base.structure_probe(model, tokenizer, cfg, selected)
        if baseline_route["direct_argmax_tool_count"] != 2 or baseline_route["tool_argmax_tool_count"] != 4:
            raise RuntimeError("v0.0.52 routing baseline drifted")
        if int(baseline_place["token_top1"]) != EXPECTED_COPY_TOKENS or int(baseline_place["tokens"]) != EXPECTED_TOKENS:
            raise RuntimeError("v0.0.52 placement baseline drifted")
        if int(baseline_structure["envelope_json_valid"]) < 7 or int(baseline_structure["tool_name_correct"]) < 7:
            raise RuntimeError("v0.0.52 structure baseline drifted")

        tool_id, special_ids = margin.routing_contract(tokenizer)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
        updates = []
        for step in range(1, STEPS + 1):
            update = margin.anchor_update(model, tokenizer, optimizer, routing_cases, tool_id, special_ids)
            route = base.routing_probe(model, tokenizer, routing_cases)
            updates.append({
                "step": step,
                "update": update,
                "direct_argmax_tool_count": int(route["direct_argmax_tool_count"]),
                "tool_argmax_tool_count": int(route["tool_argmax_tool_count"]),
                "routing": margin.compact_route(route),
            })
            print(json.dumps({
                "event": "v053_step9_probe",
                "step": step,
                "direct_argmax_tool_count": route["direct_argmax_tool_count"],
                "tool_argmax_tool_count": route["tool_argmax_tool_count"],
                "routing": margin.compact_route(route),
            }), flush=True)

        route = base.routing_probe(model, tokenizer, routing_cases)
        place = base.placement_probe(model, tokenizer, selected, template)
        structure = base.structure_probe(model, tokenizer, cfg, selected)
        generation = base.generation_probe(model, tokenizer, routing_cases, spec["generation"])
        gates = base.gate_record(route, generation, place, structure)

        observed = {
            "direct_argmax_tool_count": int(route["direct_argmax_tool_count"]),
            "tool_argmax_tool_count": int(route["tool_argmax_tool_count"]),
            "direct_generation_pass": int(generation["direct_pass"]),
            "tool_generation_pass": int(generation["tool_pass"]),
            "copy_tokens": int(place["token_top1"]),
            "tokens": int(place["tokens"]),
            "json_valid": int(structure["envelope_json_valid"]),
            "tool_correct": int(structure["tool_name_correct"]),
        }
        expected = {
            "direct_argmax_tool_count": 0,
            "tool_argmax_tool_count": 4,
            "direct_generation_pass": 4,
            "tool_generation_pass": 4,
            "copy_tokens": EXPECTED_COPY_TOKENS,
            "tokens": EXPECTED_TOKENS,
            "json_valid": EXPECTED_STRUCTURE,
            "tool_correct": EXPECTED_STRUCTURE,
        }
        if observed != expected or not all(gates.values()):
            raise RuntimeError(f"v0.0.53 step-9 reproduction mismatch: observed={observed}, expected={expected}, gates={gates}")

        candidate_hash = trust.trace.state_digest(model)
        if candidate_hash == source_hash:
            raise RuntimeError("step-9 candidate state did not change")

        run_id = f"ember-v053-step9-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        best_path = work / "best.pt"
        candidate = copy.deepcopy(source_ckpt)
        candidate["model_state"] = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        candidate["step"] = STEPS
        candidate["run_id"] = run_id
        candidate["train_config"] = {
            **dict(source_ckpt.get("train_config") or {}),
            "version": VERSION,
            "phase": "first-token-rank-margin-anchor-candidate",
            "source_version": "0.0.52",
            "source_checkpoint": source_path,
            "anchor_learning_rate": LR,
            "anchor_steps": STEPS,
            "direct_margin": margin.DIRECT_MARGIN,
            "tool_margin": margin.TOOL_MARGIN,
            "production_authorized": False,
            "promotion_authorized": False,
        }
        candidate["candidate_metadata"] = {
            "source_state_sha256": source_hash,
            "candidate_state_sha256": candidate_hash,
            "reproduction": observed,
            "gates": gates,
            "selected_case_ids": selected_ids,
        }
        torch.save(candidate, best_path)

        reloaded = torch.load(best_path, map_location="cpu", weights_only=False)
        if reloaded.get("train_config", {}).get("version") != VERSION:
            raise RuntimeError("saved v0.0.53 version metadata mismatch")
        verify = copy.deepcopy(model)
        verify.load_state_dict(reloaded["model_state"])
        verify.eval()
        if trust.trace.state_digest(verify) != candidate_hash:
            raise RuntimeError("saved v0.0.53 state digest mismatch")
        del verify

        from src.quantize_int4 import export_int4_checkpoint, dequantize_tensor
        from src.model import EmberGPT, ModelConfig
        from src.tokenizer import tokenizer_from_state_dict

        int4_path = work / "best.int4.pt"
        export_int4_checkpoint(str(best_path), str(int4_path))
        int4 = torch.load(int4_path, map_location="cpu", weights_only=False)
        int4_state = {name: dequantize_tensor(value) for name, value in int4["quantized_state"].items()}
        int4_state.update(int4.get("passthrough_state", {}))
        int4_model = EmberGPT(ModelConfig(**int4["model_config"]))
        int4_model.load_state_dict(int4_state)
        int4_model.eval()
        int4_tok = tokenizer_from_state_dict(int4["tokenizer"])
        smoke_ids = int4_tok.encode("<|system|>You are Ember.<|user|>Ready?<|assistant|>\n")
        with torch.inference_mode():
            logits, _ = int4_model(torch.tensor([smoke_ids], dtype=torch.long), None)
        if not bool(torch.isfinite(logits).all().item()):
            raise RuntimeError("INT4 reconstruction produced non-finite logits")

        report = {
            "status": "CANDIDATE_SAVED",
            "version": VERSION,
            "model_name": MODEL_NAME,
            "model_repo": candidate_repo,
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_repo": source_repo,
            "source_checkpoint": source_path,
            "source_state_sha256": source_hash,
            "candidate_state_sha256": candidate_hash,
            "checkpoint_sha256": sha256_file(best_path),
            "int4_sha256": sha256_file(int4_path),
            "int4_bytes": int4_path.stat().st_size,
            "learning_rate": LR,
            "steps": STEPS,
            "baseline": {
                "routing": margin.compact_route(baseline_route),
                "placement": baseline_place,
                "structure": baseline_structure,
            },
            "updates": updates,
            "final": {
                "routing": margin.compact_route(route),
                "observed": observed,
                "placement": place,
                "structure": structure,
                "generation": generation,
                "gates": gates,
            },
            "template": template_report,
            "selected_case_ids": selected_ids,
            "production_pointer_changed": False,
            "promotion_status": "NOT_RUN",
        }
        report_path = work / "candidate-report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

        state = {
            "status": "candidate_saved",
            "version": VERSION,
            "run_id": run_id,
            "best_step": STEPS,
            "candidate_checkpoint": f"checkpoints/{run_id}/best.pt",
            "candidate_int4": f"checkpoints/{run_id}/best.int4.pt",
            "routing_gate_pass": True,
            "copy_preserved": True,
            "structure_preserved": True,
            "promotion": "NOT_RUN",
            "production_authorized": False,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        state_path = work / "run-state.json"
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

        api.create_repo(repo_id=candidate_repo, repo_type="model", private=True, exist_ok=True)
        remote_dir = f"checkpoints/{run_id}"
        upload(api, candidate_repo, best_path, f"{remote_dir}/best.pt", "Save Ember v0.0.53 step-9 candidate")
        upload(api, candidate_repo, int4_path, f"{remote_dir}/best.int4.pt", "Save Ember v0.0.53 step-9 candidate INT4")
        upload(api, candidate_repo, report_path, "candidate/candidate-report.json", "Record Ember v0.0.53 step-9 evidence")
        upload(api, candidate_repo, state_path, "run-state.json", "Record Ember v0.0.53 candidate state")

        print("EMBER_V053_STEP9_SAVE=PASS", flush=True)
        print(f"MODEL_REPO={candidate_repo}", flush=True)
        print(f"RUN_ID={run_id}", flush=True)
        print(f"CHECKPOINT_PATH={remote_dir}/best.pt", flush=True)
        print(f"INT4_PATH={remote_dir}/best.int4.pt", flush=True)
        print(f"CHECKPOINT_SHA256={report['checkpoint_sha256']}", flush=True)
        print(f"INT4_SHA256={report['int4_sha256']}", flush=True)
        print(json.dumps(observed, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
