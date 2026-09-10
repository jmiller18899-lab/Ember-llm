"""Bounded single-token four-way tool-family anchor for Ember v0.0.54.

The free-running first-divergence trace localized the earliest wrong tool-family
decision to one token immediately after the stable common tool prefix. Rather
than teacher-forcing an entire argument key/name/envelope, this diagnostic only
optimizes that ONE family-discriminating token:

    common prefix -> {weather token | calculator token | web_search token | get_time token}

The loss is contrastive only among those four family tokens. It does not reward
any value token, punctuation token, later tool-name token, or direct-response
text. Only transformer blocks 3-5 are trainable. Every rung starts from the exact
saved v0.0.53 step-9 checkpoint and rolls back/stops if copy drops below 22/35
or held-out true-tool entry drops below 8/8.

CPU-only. No checkpoint save, export, promotion, or integration.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

from huggingface_hub import HfApi
import torch
import torch.nn.functional as F

from jobs import ember_hf_eval as ev
from jobs import ember_v052_first_token_logits as ft
from jobs import ember_v053_heldout_routing as held
from jobs import ember_v053_first_token_anchor as anchor
from jobs import ember_v054_routing_repair as v54
from jobs import ember_v054_free_run_divergence as div
from jobs import ember_v054_block345_family as b345
from jobs import ember_alternating_trust_region as trust

OUT = Path("v054-family-token-anchor")
LRS = (8e-7, 1.6e-6, 3.2e-6)
MAX_STEPS = 12
FAMILY_MARGIN = 0.50
GRAD_CLIP = 0.25
COPY_FLOOR = 22
STRUCTURE_FLOOR = 7


def continuation_ids(tokenizer, text: str, shared_prefix_id: int | None) -> list[int]:
    ids = list(tokenizer.encode(text))
    if shared_prefix_id is not None and ids and int(ids[0]) == int(shared_prefix_id):
        ids = ids[1:]
    return [int(x) for x in ids]


def common_family_contract(tokenizer, shared_prefix_id: int | None) -> dict:
    sequences = {
        name: continuation_ids(tokenizer, text, shared_prefix_id)
        for name, text in div.CANONICAL.items()
    }
    minimum = min(len(x) for x in sequences.values())
    common_len = 0
    for index in range(minimum):
        values = {seq[index] for seq in sequences.values()}
        if len(values) != 1:
            break
        common_len += 1
    if common_len < 1:
        raise RuntimeError("tool-family canonical continuations have no common token prefix")
    family_ids = {name: seq[common_len] for name, seq in sequences.items()}
    if len(set(family_ids.values())) != len(family_ids):
        raise RuntimeError(f"family decision tokens are not unique: {family_ids}")
    common_ids = next(iter(sequences.values()))[:common_len]
    return {
        "common_ids": common_ids,
        "common_len": common_len,
        "family_ids": family_ids,
        "family_tokens": {
            name: ft.display_token(tokenizer, tid, shared_prefix_id)
            for name, tid in family_ids.items()
        },
        "canonical_sequences": sequences,
    }


def logits_at_family_decision(model, tokenizer, prompt: str, contract: dict):
    context = list(tokenizer.encode(prompt)) + list(contract["common_ids"])
    x = torch.tensor([context], dtype=torch.long)
    logits, _ = model(x, None)
    return logits[0, -1]


def family_probe(model, tokenizer, cases: list[dict], contract: dict) -> dict:
    family_ids = contract["family_ids"]
    rows = []
    with torch.inference_mode():
        for case in cases:
            expected = case["expected_tool"]
            logits = logits_at_family_decision(model, tokenizer, case["prompt"], contract).float().cpu()
            probs = torch.softmax(logits, dim=-1)
            scored = {name: float(logits[tid]) for name, tid in family_ids.items()}
            ranked = sorted(scored, key=scored.get, reverse=True)
            target_id = family_ids[expected]
            target_logit = float(logits[target_id])
            best_wrong = max(float(logits[tid]) for name, tid in family_ids.items() if name != expected)
            rows.append({
                "id": case["id"],
                "expected_tool": expected,
                "predicted_tool_family": ranked[0],
                "correct": ranked[0] == expected,
                "expected_rank": ranked.index(expected) + 1,
                "target_token": contract["family_tokens"][expected],
                "target_probability": float(probs[target_id]),
                "target_minus_best_wrong_logit": target_logit - best_wrong,
                "family_ranking": ranked,
                "family_logits": scored,
            })
    return {
        "correct": sum(int(r["correct"]) for r in rows),
        "total": len(rows),
        "rows": rows,
    }


def greedy_common_prefix_probe(model, tokenizer, cases: list[dict], contract: dict) -> dict:
    rows = []
    for case in cases:
        context = [int(x) for x in tokenizer.encode(case["prompt"])]
        matched = 0
        for expected_id in contract["common_ids"]:
            x = torch.tensor([context], dtype=torch.long)
            with torch.inference_mode():
                logits, _ = model(x, None)
            chosen = int(torch.argmax(logits[0, -1]).item())
            if chosen != int(expected_id):
                break
            context.append(chosen)
            matched += 1
        rows.append({"id": case["id"], "matched": matched, "required": contract["common_len"]})
    return {
        "fully_matched": sum(int(r["matched"] == r["required"]) for r in rows),
        "total": len(rows),
        "rows": rows,
    }


def family_update(model, tokenizer, optimizer, cases: list[dict], contract: dict):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    family_ids = contract["family_ids"]
    losses = []
    margins = []
    for case in cases:
        expected = case["expected_tool"]
        logits = logits_at_family_decision(model, tokenizer, case["prompt"], contract)
        target = logits[family_ids[expected]]
        wrong = torch.stack([
            logits[tid]
            for name, tid in family_ids.items()
            if name != expected
        ]).max()
        loss = F.relu(wrong + FAMILY_MARGIN - target)
        (loss / len(cases)).backward()
        losses.append(float(loss.detach()))
        margins.append(float((target - wrong).detach()))
    params = [p for p in model.parameters() if p.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    return {
        "hinge_mean": sum(losses) / len(losses),
        "mean_target_margin": sum(margins) / len(margins),
        "gradient_norm": float(grad_norm),
    }


def summary_markdown(report: dict) -> str:
    lines = [
        "# Ember v0.0.54 single-token tool-family anchor",
        "",
        "Source: exact saved v0.0.53 step-9 checkpoint.",
        f"Trainable groups: {', '.join(sorted(b345.TRAIN_GROUPS))}.",
        f"Family margin: {FAMILY_MARGIN:.2f}; LR ladder: {', '.join(f'{x:.1e}' for x in LRS)}.",
        f"Family decision position: token {report['family_contract']['common_len'] + 1} of the canonical tool prefix.",
        f"Family tokens: {report['family_contract']['family_tokens']}",
        "",
        "| LR | Step | Dev family | Held-out family | Held-out D/T gen | Tool names | Copy | Structure | Fixed D/T | Phase pass |",
        "| ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- |",
    ]
    for rung in report.get("rungs", []):
        for probe in rung.get("candidate_probes", []):
            h = probe["heldout_eval"]["metrics"]
            s = probe["structure"]
            f = probe["fixed_generation"]
            lines.append(
                f"| {rung['lr']:.1e} | {probe['step']} | {probe['dev_family']['correct']}/{probe['dev_family']['total']} | "
                f"{probe['heldout_family']['correct']}/{probe['heldout_family']['total']} | "
                f"{h['direct_generation_pass']}/12 / {h['tool_generation_pass']}/8 | {h['tool_name_correct']}/8 | "
                f"{probe['placement']['token_top1']}/35 | {s['envelope_json_valid']}/8 / {s['tool_name_correct']}/8 | "
                f"{f['direct_pass']}/4 / {f['tool_pass']}/4 | {probe['phase_accepted']} |"
            )
    lines += [
        "",
        f"Family operating point found: {report.get('family_operating_point_found', False)}",
        f"Selected LR/step: {report.get('selected_lr')} / {report.get('selected_step')}",
        f"Interpretation: {report.get('interpretation', '')}",
        "",
        "No checkpoint was saved, exported, promoted, or integrated.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN required")
    torch.set_num_threads(2)
    torch.manual_seed(20260910)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    report = {
        "schema_version": 1,
        "diagnostic": "ember-v054-single-family-token-anchor-v1",
        "status": "ERROR",
        "source": held.MODEL_NAME,
        "lrs": list(LRS),
        "max_steps": MAX_STEPS,
        "family_margin": FAMILY_MARGIN,
        "trainable_groups": sorted(b345.TRAIN_GROUPS),
        "cpu_only": True,
        "checkpoint_save_authorized": False,
        "promotion_authorized": False,
        "rungs": [],
    }

    with tempfile.TemporaryDirectory(prefix="ember-v054-family-token-") as td:
        work = Path(td)
        archive = ev.download_verified(ev.PACKAGE_URL, work / "ember.zip", ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        import sys
        sys.path.insert(0, str(work / "src" / "ember"))

        api = HfApi(token=token)
        owner = api.whoami()["name"]
        repo = f"{owner}/{held.MODEL_NAME}"
        model, tokenizer, checkpoint = held.load_full(repo, work, token)
        if checkpoint.get("train_config", {}).get("version") != "0.0.53":
            raise RuntimeError("expected saved v0.0.53 checkpoint")
        model.eval()
        pristine = copy.deepcopy(model.state_dict())
        source_hash = trust.trace.state_digest(model)
        b345.configure_trainable(model)

        special_contract = ev.special_token_contract(tokenizer)
        family_contract = common_family_contract(tokenizer, special_contract.get("shared_prefix_id"))
        report["family_contract"] = family_contract

        held_tools = [c for c in held.CASES if c["kind"] == "tool_call"]
        dev_tools = [c for c in v54.TRAIN if c["kind"] == "tool_call"]
        common_held = greedy_common_prefix_probe(model, tokenizer, held_tools, family_contract)
        common_dev = greedy_common_prefix_probe(model, tokenizer, dev_tools, family_contract)
        if common_held["fully_matched"] != len(held_tools):
            raise RuntimeError(f"held-out tool cases do not share stable greedy prefix: {common_held}")
        report["greedy_common_prefix_baseline"] = {"heldout": common_held, "development": common_dev}

        cfg, template, template_report, selected, selected_ids, *_ = anchor.build_placement_fixture(work)
        fixed_spec = json.loads(Path("config/ember_v0.0.8_eval.json").read_text())
        fixed_cases = [c for c in fixed_spec["cases"] if c["kind"] in {"direct_response", "tool_call"}]
        baseline_place = anchor.placement_probe(model, tokenizer, selected, template)
        baseline_structure = anchor.structure_probe(model, tokenizer, cfg, selected)
        baseline_held_family = family_probe(model, tokenizer, held_tools, family_contract)
        baseline_dev_family = family_probe(model, tokenizer, dev_tools, family_contract)
        baseline_route = v54.routing_counts(model, tokenizer, held.CASES)
        report["baseline"] = {
            "placement": baseline_place,
            "structure": baseline_structure,
            "heldout_family": baseline_held_family,
            "development_family": baseline_dev_family,
            "heldout_routing": baseline_route,
        }
        if int(baseline_place["token_top1"]) != COPY_FLOOR:
            raise RuntimeError("v0.0.53 copy baseline drifted")
        if int(baseline_structure["envelope_json_valid"]) < 8 or int(baseline_structure["tool_name_correct"]) < 8:
            raise RuntimeError("v0.0.53 structure baseline drifted")

        found = False
        best = None
        best_score = (-1, -1)
        for lr in LRS:
            model.load_state_dict(pristine)
            b345.configure_trainable(model)
            model.eval()
            if trust.trace.state_digest(model) != source_hash:
                raise RuntimeError("rung reset failed")
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=float(lr),
                weight_decay=0.0,
            )
            rung = {"lr": float(lr), "steps": [], "candidate_probes": [], "rollback": None, "phase_accepted": False}

            for step in range(1, MAX_STEPS + 1):
                pre_model = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
                pre_opt = copy.deepcopy(optimizer.state_dict())
                update = family_update(model, tokenizer, optimizer, dev_tools, family_contract)
                held_family = family_probe(model, tokenizer, held_tools, family_contract)
                dev_family = family_probe(model, tokenizer, dev_tools, family_contract)
                held_route = v54.routing_counts(model, tokenizer, held.CASES)
                place = anchor.placement_probe(model, tokenizer, selected, template)
                rec = {
                    "step": step,
                    "update": update,
                    "heldout_family": held_family["correct"],
                    "development_family": dev_family["correct"],
                    "heldout_tool_entry": held_route["tool_ok"],
                    "heldout_direct_entry": held_route["direct_ok"],
                    "copy_tokens": int(place["token_top1"]),
                }
                rung["steps"].append(rec)
                print(json.dumps({"event": "family_token_probe", "lr": lr, **rec}), flush=True)

                if int(place["token_top1"]) < COPY_FLOOR or int(held_route["tool_ok"]) < 8:
                    params = dict(model.named_parameters())
                    with torch.no_grad():
                        for name, value in pre_model.items():
                            params[name].copy_(value)
                    optimizer.load_state_dict(pre_opt)
                    model.eval()
                    rung["rollback"] = {
                        "step": step,
                        "reason": "copy" if int(place["token_top1"]) < COPY_FLOOR else "tool_entry",
                    }
                    print(json.dumps({"event": "family_token_rollback", "lr": lr, **rung["rollback"]}), flush=True)
                    break

                score = (held_family["correct"], dev_family["correct"])
                if score > best_score:
                    best_score = score
                    best = {"lr": float(lr), "step": step, "state": copy.deepcopy(model.state_dict())}

                if held_family["correct"] == held_family["total"] and dev_family["correct"] == dev_family["total"]:
                    held_eval = held.evaluate(model, tokenizer, f"family-token-{lr:.1e}-s{step}")
                    structure = anchor.structure_probe(model, tokenizer, cfg, selected)
                    fixed_generation = anchor.generation_probe(model, tokenizer, fixed_cases, fixed_spec["generation"])
                    hm = held_eval["metrics"]
                    gates = {
                        "heldout_family": held_family["correct"] == 8,
                        "development_family": dev_family["correct"] == 16,
                        "heldout_tool_entry": hm["tool_first_token_pass"] == 8,
                        "heldout_tool_generation": hm["tool_generation_pass"] == 8,
                        "heldout_tool_names": hm["tool_name_correct"] == 8,
                        "direct_nonregression": hm["direct_first_token_pass"] >= 3 and hm["direct_generation_pass"] >= 3,
                        "copy": int(place["token_top1"]) >= COPY_FLOOR,
                        "structure": int(structure["envelope_json_valid"]) >= STRUCTURE_FLOOR and int(structure["tool_name_correct"]) >= STRUCTURE_FLOOR,
                        "fixed": int(fixed_generation["direct_pass"]) == 4 and int(fixed_generation["tool_pass"]) == 4,
                    }
                    accepted = all(gates.values())
                    probe = {
                        "step": step,
                        "heldout_family": held_family,
                        "dev_family": dev_family,
                        "heldout_eval": held_eval,
                        "placement": place,
                        "structure": structure,
                        "fixed_generation": fixed_generation,
                        "gates": gates,
                        "phase_accepted": accepted,
                    }
                    rung["candidate_probes"].append(probe)
                    print(json.dumps({
                        "event": "family_token_full_gate",
                        "lr": lr,
                        "step": step,
                        "heldout": hm,
                        "copy": place["token_top1"],
                        "structure_json": structure["envelope_json_valid"],
                        "structure_tool": structure["tool_name_correct"],
                        "gates": gates,
                        "accepted": accepted,
                    }), flush=True)
                    if accepted:
                        rung["phase_accepted"] = True
                        report["family_operating_point_found"] = True
                        report["selected_lr"] = float(lr)
                        report["selected_step"] = step
                        report["selected_probe"] = probe
                        found = True
                        break

            report["rungs"].append(rung)
            del optimizer
            if found:
                break

        if found:
            report["interpretation"] = (
                "A one-token four-way family anchor cleared held-out/development family selection and free-running tool generation while preserving copy/structure. "
                "The next phase should reproduce this point and add a separate direct-routing anchor, not broaden the tool-envelope loss."
            )
        else:
            report["family_operating_point_found"] = False
            report["selected_lr"] = None
            report["selected_step"] = None
            if best is not None:
                model.load_state_dict(best["state"])
                b345.configure_trainable(model)
                model.eval()
                best_held_family = family_probe(model, tokenizer, held_tools, family_contract)
                best_dev_family = family_probe(model, tokenizer, dev_tools, family_contract)
                best_eval = held.evaluate(model, tokenizer, "family-token-best")
                best_place = anchor.placement_probe(model, tokenizer, selected, template)
                best_structure = anchor.structure_probe(model, tokenizer, cfg, selected)
                report["best_observed"] = {
                    "lr": best["lr"],
                    "step": best["step"],
                    "heldout_family": best_held_family,
                    "development_family": best_dev_family,
                    "heldout_eval": best_eval,
                    "placement": best_place,
                    "structure": best_structure,
                }
            report["interpretation"] = (
                "No tested single-token family anchor cleared the free-running tool gate inside the protected copy/structure trust region. "
                "Use the best margins and rollback point to choose the next local intervention; do not save or promote."
            )

        report["status"] = "PASS"
        report["elapsed_seconds"] = time.monotonic() - started
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (OUT / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
        print(json.dumps({
            "event": "family_token_complete",
            "family_operating_point_found": report["family_operating_point_found"],
            "selected_lr": report["selected_lr"],
            "selected_step": report["selected_step"],
            "baseline_heldout_family": baseline_held_family["correct"],
            "baseline_dev_family": baseline_dev_family["correct"],
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
