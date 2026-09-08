"""CPU-only residual envelope-entry/logit diagnostic for Ember v0.0.45.

The v0.0.44 prompt stack and pinned v0.0.31 step-479 checkpoint are frozen.
This runner does not search prompts and cannot train. It re-identifies the six
stable envelope failures inside their five structural subtypes, measures the
first-token competition between <|tool|> and the greedy alternative, and then
forces exactly one <|tool|> token diagnostically to ask whether the downstream
JSON/tool machinery is already intact.

The forced-marker result is intervention evidence only. It is never counted as
baseline success and never authorizes placement learning, training, GPU work,
promotion, deployment, or production integration.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, median
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import ember_envelope_preflight_v044 as prior

base = prior.base
control = prior.control
DEFAULT_CONFIG = ROOT / "config/ember_residual_logit_v0.0.45.json"
V044_CONFIG = json.loads(prior.DEFAULT_CONFIG.read_text(encoding="utf-8"))
V044_SYSTEM_SELECTION = {subtype: "baseline" for subtype in prior.TARGET_SUBTYPES}
TOOL = base.semantic_gate.TOOL
EOT = base.semantic_gate.EOT

TARGET_SUBTYPES = {
    "short_code/len4",
    "short_code/len5",
    "long_code/4x4",
    "long_code/3x5",
    "path/plain_leaf",
}
EXPECTED_FAILURE_IDS = {
    "system_target_short_code_02",
    "system_target_short_code_03",
    "system_target_long_code_02",
    "system_target_long_code_05",
    "system_target_long_code_08",
    "system_target_path_05",
}


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if path.resolve() != DEFAULT_CONFIG.resolve() or cfg.get("version") != "0.0.45":
        raise ValueError("unsupported v0.0.45 diagnostic configuration")
    if any(cfg.get(key) is not False for key in (
        "training_authorized",
        "gpu_training_authorized",
        "production_authorized",
        "placement_learning_authorized",
    )):
        raise ValueError("v0.0.45 is diagnostic-only; all authorization flags must be false")
    if set(cfg.get("target_subtypes", [])) != TARGET_SUBTYPES:
        raise ValueError("v0.0.45 target subtypes changed")
    if set(cfg.get("expected_failure_ids", [])) != EXPECTED_FAILURE_IDS:
        raise ValueError("v0.0.45 expected residual failures changed")
    if int(cfg.get("expected_target_cases", 0)) != 24 or int(cfg.get("expected_failures", 0)) != 6:
        raise ValueError("v0.0.45 requires 24 matched residual-subtype cases and six failures")
    if int(cfg.get("generation_budget", 0)) != int(V044_CONFIG["generation_budget"]):
        raise ValueError("v0.0.45 generation budget must stay identical to v0.0.44")
    if int(cfg.get("top_k_tokens", 0)) < 2:
        raise ValueError("v0.0.45 requires at least two top-token diagnostics")
    rules = cfg.get("interpretation", {})
    if int(rules.get("near_boundary_tool_rank_max", 0)) < 1:
        raise ValueError("invalid near-boundary rank")
    if float(rules.get("near_boundary_logit_deficit_max", -1)) < 0:
        raise ValueError("invalid near-boundary deficit")
    if int(rules.get("competitive_tool_rank_max", 0)) < int(rules["near_boundary_tool_rank_max"]):
        raise ValueError("competitive rank must not be stricter than near-boundary rank")
    if float(rules.get("competitive_logit_deficit_max", -1)) < float(rules["near_boundary_logit_deficit_max"]):
        raise ValueError("competitive deficit must not be stricter than near-boundary deficit")
    rescue = float(rules.get("downstream_rescue_rate_minimum", -1))
    if not 0.0 <= rescue <= 1.0:
        raise ValueError("invalid rescue-rate heuristic")
    return cfg


def frozen_v044_cases() -> list[dict]:
    cases = prior.build_final_cases(V044_CONFIG, V044_SYSTEM_SELECTION)
    if len(cases) != 90:
        raise ValueError("v0.0.44 frozen battery no longer has 90 cases")
    return cases


def prompt_digest(cases: list[dict]) -> str:
    payload = "\n\0\n".join(case["id"] + "\n" + case["prompt"] for case in cases)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def target_cases() -> list[dict]:
    all_cases = frozen_v044_cases()
    selected = [case for case in all_cases if case["subtype"] in TARGET_SUBTYPES]
    if len(selected) != 24:
        raise ValueError(f"expected 24 residual-subtype cases, got {len(selected)}")
    counts = defaultdict(int)
    for case in selected:
        counts[case["subtype"]] += 1
        if case["system_variant"] != "v037_schema_system":
            raise ValueError(f"v0.0.45 system line is not frozen for {case['id']}")
    expected_counts = {
        "short_code/len4": 5,
        "short_code/len5": 5,
        "long_code/4x4": 5,
        "long_code/3x5": 5,
        "path/plain_leaf": 4,
    }
    if dict(counts) != expected_counts:
        raise ValueError(f"residual subtype composition changed: {dict(counts)}")
    return selected


def _decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)])
    except Exception as exc:  # diagnostic display must not crash the run
        return f"<decode-error:{type(exc).__name__}>"


def first_token_probe(model, tokenizer, torch, prompt: str, top_k: int) -> dict:
    contract = base.semantic_gate.token_contract(tokenizer)
    tool_id = int(contract["signatures"][TOOL][0])
    eos_id = int(contract["eos_id"])
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("empty diagnostic prompt")
    with torch.inference_mode():
        x = torch.tensor([prompt_ids], dtype=torch.long, device="cpu")
        logits, _ = model(x)
        next_logits = logits[0, -1, :].float()
    if not bool(torch.isfinite(next_logits).all().item()):
        raise ValueError("non-finite first-token logits")
    tool_logit = float(next_logits[tool_id].item())
    masked = next_logits.clone()
    masked[tool_id] = -float("inf")
    best_other_id = int(torch.argmax(masked).item())
    best_other_logit = float(masked[best_other_id].item())
    margin = tool_logit - best_other_logit
    rank = int((next_logits > next_logits[tool_id]).sum().item()) + 1
    log_probs = torch.log_softmax(next_logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = float((-(probs * log_probs)).sum().item())
    k = min(int(top_k), int(next_logits.numel()))
    top_values, top_ids = torch.topk(next_logits, k=k)
    top = [
        {
            "rank": index + 1,
            "token_id": int(token_id),
            "text": _decode_token(tokenizer, int(token_id)),
            "logit": float(value),
            "probability": float(probs[int(token_id)].item()),
        }
        for index, (value, token_id) in enumerate(zip(top_values.tolist(), top_ids.tolist()))
    ]
    return {
        "prompt_tokens": len(prompt_ids),
        "tool_token_id": tool_id,
        "tool_token_text": _decode_token(tokenizer, tool_id),
        "tool_rank": rank,
        "tool_logit": tool_logit,
        "tool_probability": float(probs[tool_id].item()),
        "tool_margin_vs_best_other": margin,
        "tool_logit_deficit": max(0.0, -margin),
        "best_other_token_id": best_other_id,
        "best_other_text": _decode_token(tokenizer, best_other_id),
        "best_other_logit": best_other_logit,
        "eos_rank": int((next_logits > next_logits[eos_id]).sum().item()) + 1,
        "eos_probability": float(probs[eos_id].item()),
        "entropy_nats": entropy,
        "top_tokens": top,
    }


def forced_tool_completion(model, tokenizer, torch, prompt: str, max_new_tokens: int) -> dict:
    """Force only the first <|tool|> token, then return to ordinary greedy decoding."""
    contract = base.semantic_gate.token_contract(tokenizer)
    tool_id = int(contract["signatures"][TOOL][0])
    eos_id = int(contract["eos_id"])
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids or len(prompt_ids) + max_new_tokens > int(model.cfg.block_size):
        raise ValueError("forced diagnostic exceeds the context window")
    generated = [tool_id]
    reason = "max_new_tokens"
    with torch.inference_mode():
        x = torch.tensor([prompt_ids + [tool_id]], dtype=torch.long, device="cpu")
        for _ in range(max_new_tokens - 1):
            logits, _ = model(x)
            next_logits = logits[0, -1, :]
            if not bool(torch.isfinite(next_logits).all().item()):
                raise ValueError("non-finite forced-continuation logits")
            next_id = int(torch.argmax(next_logits).item())
            generated.append(next_id)
            if next_id == eos_id:
                reason = "eos"
                break
            x = torch.cat((x, torch.tensor([[next_id]], dtype=torch.long, device="cpu")), dim=1)
    completion = tokenizer.decode(generated)
    if not completion.lstrip().startswith(TOOL):
        raise ValueError("forced special token did not decode as the tool marker")
    return {"completion": completion, "generated_ids": generated, "stop_reason": reason}


def _summary_values(rows: list[dict]) -> dict:
    if not rows:
        return {"cases": 0}
    margins = [float(row["first_token"]["tool_margin_vs_best_other"]) for row in rows]
    ranks = [int(row["first_token"]["tool_rank"]) for row in rows]
    probs = [float(row["first_token"]["tool_probability"]) for row in rows]
    target_tokens = [int(row["target_token_count"]) for row in rows]
    return {
        "cases": len(rows),
        "tool_rank_median": float(median(ranks)),
        "tool_rank_mean": float(mean(ranks)),
        "tool_margin_median": float(median(margins)),
        "tool_margin_mean": float(mean(margins)),
        "tool_probability_median": float(median(probs)),
        "target_token_count_median": float(median(target_tokens)),
        "forced_valid_tool": sum(
            bool(row["forced_score"]["envelope_json_valid"] and row["forced_score"]["tool_name_correct"])
            for row in rows
        ),
    }


def interpret(rows: list[dict], cfg: dict) -> dict:
    failures = [row for row in rows if not row["baseline_score"]["envelope_json_valid"]]
    passes = [row for row in rows if row["baseline_score"]["envelope_json_valid"]]
    if len(failures) != int(cfg["expected_failures"]):
        raise ValueError("cannot interpret an unexpected failure count")
    rules = cfg["interpretation"]
    rescued = [
        row for row in failures
        if row["forced_score"]["envelope_json_valid"] and row["forced_score"]["tool_name_correct"]
    ]
    near = [
        row for row in failures
        if row["first_token"]["tool_rank"] <= int(rules["near_boundary_tool_rank_max"])
        and row["first_token"]["tool_logit_deficit"] <= float(rules["near_boundary_logit_deficit_max"])
    ]
    competitive = [
        row for row in failures
        if row["first_token"]["tool_rank"] <= int(rules["competitive_tool_rank_max"])
        and row["first_token"]["tool_logit_deficit"] <= float(rules["competitive_logit_deficit_max"])
    ]
    rescue_rate = len(rescued) / len(failures)
    near_fraction = len(near) / len(failures)
    competitive_fraction = len(competitive) / len(failures)
    if rescue_rate >= float(rules["downstream_rescue_rate_minimum"]):
        if near_fraction >= 0.5:
            regime = "downstream_intact_near_envelope_entry_boundary"
        elif competitive_fraction >= 0.5:
            regime = "downstream_intact_tool_token_competitive_but_suppressed"
        else:
            regime = "downstream_intact_different_first_token_regime"
    else:
        regime = "deeper_decoding_difference_after_envelope_entry"
    return {
        "heuristic_only": True,
        "regime": regime,
        "failures": _summary_values(failures),
        "matched_passes": _summary_values(passes),
        "forced_rescue_count": len(rescued),
        "forced_rescue_rate": rescue_rate,
        "near_boundary_count": len(near),
        "near_boundary_fraction": near_fraction,
        "competitive_count": len(competitive),
        "competitive_fraction": competitive_fraction,
        "rescued_ids": [row["id"] for row in rescued],
        "near_boundary_ids": [row["id"] for row in near],
        "competitive_ids": [row["id"] for row in competitive],
        "meaning": "Descriptive CPU evidence only; no learning phase is authorized by this classification.",
    }


def per_subtype(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["subtype"]].append(row)
    out = {}
    for subtype, items in sorted(grouped.items()):
        failures = [row for row in items if not row["baseline_score"]["envelope_json_valid"]]
        passes = [row for row in items if row["baseline_score"]["envelope_json_valid"]]
        out[subtype] = {
            "all": _summary_values(items),
            "failures": _summary_values(failures),
            "passes": _summary_values(passes),
            "failure_ids": [row["id"] for row in failures],
        }
    return out


def summary_markdown(report: dict) -> str:
    interpretation = report["interpretation"]
    lines = [
        "# Ember v0.0.45 residual envelope logit diagnostic",
        "",
        "CPU-only diagnostic. Prompts and model weights are frozen. No optimizer, training, GPU, promotion, deployment, or integration occurred.",
        "",
        f"- Frozen residual cases: **{report['target_case_count']}**",
        f"- Reproduced residual failures: **{len(report['failure_ids'])}**",
        f"- Forced-marker rescue: **{interpretation['forced_rescue_count']}/{len(report['failure_ids'])} ({interpretation['forced_rescue_rate']:.1%})**",
        f"- Near-boundary heuristic: **{interpretation['near_boundary_count']}/{len(report['failure_ids'])}**",
        f"- Competitive-tool heuristic: **{interpretation['competitive_count']}/{len(report['failure_ids'])}**",
        f"- Diagnostic regime: **{interpretation['regime']}**",
        "",
        "| Group | Median tool rank | Median tool margin | Median tool probability | Forced valid tool |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, key in (("Failures", "failures"), ("Matched passes", "matched_passes")):
        values = interpretation[key]
        lines.append(
            f"| {label} | {values['tool_rank_median']:.1f} | {values['tool_margin_median']:+.4f} | "
            f"{values['tool_probability_median']:.6f} | {values['forced_valid_tool']}/{values['cases']} |"
        )
    lines += [
        "",
        "The one-token forced-marker intervention is diagnostic only. It is never credited as baseline success and does not teach or authorize slot placement.",
        "",
        "`placement_learning_authorized=false` remains in force regardless of the diagnostic regime.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("v045-results"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not os.environ.get("HF_TOKEN", "").strip():
        raise ValueError("HF_TOKEN is required to read the pinned private checkpoint")

    import torch

    torch.set_num_threads(2)
    torch.manual_seed(int(cfg["seed"]))
    torch.use_deterministic_algorithms(True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "version": "0.0.45",
        "phase": cfg["phase"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ERROR",
        "code_commit": os.environ.get("GITHUB_SHA"),
        "training_authorized": False,
        "gpu_training_authorized": False,
        "production_authorized": False,
        "placement_learning_authorized": False,
        "source_v044_run": 34288515663,
        "source_v044_score": "84/90",
    }
    try:
        source_cfg = json.loads(base.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="ember-v045-") as td:
            model, tokenizer, source, _splits, source_ref = base.load_inputs(source_cfg, Path(td), torch)
            model.to("cpu").eval()
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != cfg["source_version"]:
                raise ValueError("v0.0.45 must measure the pinned v0.0.31 step-479 checkpoint")

            reference_rows = []
            for case in control.reference_cases(V044_CONFIG):
                generation = base.semantic_gate.generate_completion(
                    model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"])
                )
                score = control.score_reference(case, generation["completion"])
                reference_rows.append({"id": case["id"], **generation, "score": score})
            reference_gate = control.reference_summary(reference_rows, V044_CONFIG)
            if not reference_gate["passed"]:
                raise ValueError("historical four-case control no longer reproduces 4/4")

            frozen_all = frozen_v044_cases()
            cases = target_cases()
            rows = []
            for case in cases:
                baseline = base.semantic_gate.generate_completion(
                    model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"])
                )
                baseline_score = control.score_case(case, baseline["completion"])
                first = first_token_probe(model, tokenizer, torch, case["prompt"], int(cfg["top_k_tokens"]))
                forced = forced_tool_completion(
                    model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"])
                )
                forced_score = control.score_case(case, forced["completion"])
                target_ids = tokenizer.encode(case["target"])
                row = {
                    "id": case["id"],
                    "kind": case["kind"],
                    "subtype": case["subtype"],
                    "target": case["target"],
                    "target_token_count": len(target_ids),
                    "target_token_ids": target_ids,
                    "prompt": case["prompt"],
                    "baseline": baseline,
                    "baseline_score": baseline_score,
                    "first_token": first,
                    "forced_tool": forced,
                    "forced_score": forced_score,
                }
                rows.append(row)
                print(json.dumps({
                    "event": "diagnostic_case",
                    "id": case["id"],
                    "subtype": case["subtype"],
                    "baseline_valid": baseline_score["envelope_json_valid"],
                    "tool_rank": first["tool_rank"],
                    "tool_margin": first["tool_margin_vs_best_other"],
                    "forced_valid_tool": bool(forced_score["envelope_json_valid"] and forced_score["tool_name_correct"]),
                }), flush=True)

            failure_ids = {row["id"] for row in rows if not row["baseline_score"]["envelope_json_valid"]}
            if failure_ids != EXPECTED_FAILURE_IDS:
                raise ValueError(f"frozen residual failures changed: {sorted(failure_ids)}")
            interpretation = interpret(rows, cfg)
            report.update(
                status="COMPLETE",
                source={
                    "repo_id": source_ref["repo_id"],
                    "checkpoint_path": source_ref["checkpoint_path"],
                    "revision": source_ref["revision"],
                    "checkpoint_sha256": source_ref["checkpoint_sha256"],
                    "step": source["step"],
                    "version": source["train_config"]["version"],
                },
                reference_gate=reference_gate,
                frozen_90_prompt_sha256=prompt_digest(frozen_all),
                target_case_count=len(rows),
                failure_ids=sorted(failure_ids),
                matched_pass_count=len(rows) - len(failure_ids),
                cases=rows,
                per_subtype=per_subtype(rows),
                interpretation=interpretation,
                meaning="Residual envelope-entry diagnostic only. No model or prompt mutation and no learning authorization.",
            )
    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        (output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if report.get("status") == "COMPLETE":
            (output / "summary.md").write_text(summary_markdown(report), encoding="utf-8")

    print(json.dumps({
        "event": "complete",
        "status": report["status"],
        "failure_ids": report["failure_ids"],
        "interpretation": report["interpretation"],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
