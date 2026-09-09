"""CPU-only structural-prefix diagnostic for Ember v0.0.46.

The pinned v0.0.31 step-479 checkpoint and the measured v0.0.44 prompt stack are
frozen. v0.0.45 showed that four residual failures remain invalid even after a
single <|tool|> marker is forced, while two long-code failures are pure entry-
suppression cases. This runner teacher-forces only the non-value canonical tool
JSON prefix on the four non-rescued failure subtypes and all matched passes.

The teacher-forced prefix stops immediately after the opening quote of the query
value. No target-value token is forced, compared, supervised, or scored. There is
no optimizer, backward pass, GPU path, checkpoint write, promotion, deployment,
or production integration.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import mean, median
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import ember_residual_logit_v045 as prior

base = prior.base
control = prior.control
DEFAULT_CONFIG = ROOT / "config/ember_structural_prefix_v0.0.46.json"
V044_CONFIG = prior.V044_CONFIG

NONRESCUED_FAILURE_IDS = {
    "system_target_short_code_02",
    "system_target_short_code_03",
    "system_target_long_code_08",
    "system_target_path_05",
}
ENTRY_SUPPRESSION_IDS = {
    "system_target_long_code_02",
    "system_target_long_code_05",
}
STRUCTURAL_SUBTYPES = {
    "short_code/len4",
    "short_code/len5",
    "long_code/3x5",
    "path/plain_leaf",
}
EXPECTED_STRUCTURAL_COUNTS = {
    "short_code/len4": 6,
    "short_code/len5": 4,
    "long_code/3x5": 3,
    "path/plain_leaf": 4,
}


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if path.resolve() != DEFAULT_CONFIG.resolve() or cfg.get("version") != "0.0.46":
        raise ValueError("unsupported v0.0.46 diagnostic configuration")
    if any(cfg.get(key) is not False for key in (
        "training_authorized",
        "gpu_training_authorized",
        "production_authorized",
        "placement_learning_authorized",
    )):
        raise ValueError("v0.0.46 is diagnostic-only; all authorization flags must be false")
    if set(cfg.get("nonrescued_failure_ids", [])) != NONRESCUED_FAILURE_IDS:
        raise ValueError("v0.0.46 non-rescued failure cohort changed")
    if set(cfg.get("entry_suppression_ids", [])) != ENTRY_SUPPRESSION_IDS:
        raise ValueError("v0.0.46 entry-suppression cohort changed")
    if set(cfg.get("structural_subtypes", [])) != STRUCTURAL_SUBTYPES:
        raise ValueError("v0.0.46 structural subtypes changed")
    if int(cfg.get("expected_residual_cases", 0)) != 24:
        raise ValueError("v0.0.46 requires the 24-case v0.0.45 residual cohort")
    if int(cfg.get("expected_structural_cases", 0)) != 17:
        raise ValueError("v0.0.46 requires 17 structural-prefix cases")
    if int(cfg.get("expected_nonrescued_failures", 0)) != 4:
        raise ValueError("v0.0.46 requires four non-rescued failures")
    if int(cfg.get("expected_structural_passes", 0)) != 13:
        raise ValueError("v0.0.46 requires 13 matched structural passes")
    if int(cfg.get("expected_entry_suppression_cases", 0)) != 2:
        raise ValueError("v0.0.46 requires two entry-suppression cases")
    if int(cfg.get("generation_budget", 0)) != int(V044_CONFIG["generation_budget"]):
        raise ValueError("generation budget changed from v0.0.44/v0.0.45")
    segments = cfg.get("canonical_prefix_segments", [])
    if not segments or any(set(item) != {"name", "text"} for item in segments):
        raise ValueError("invalid structural-prefix segment contract")
    names = [item["name"] for item in segments]
    if len(names) != len(set(names)):
        raise ValueError("structural-prefix stage names must be unique")
    prefix = "".join(item["text"] for item in segments)
    expected = '<|tool|>{"name":"web_search","arguments":{"query":"'
    if prefix != expected:
        raise ValueError("canonical structural prefix changed")
    if not prefix.endswith('"'):
        raise ValueError("structural prefix must stop at the opening value quote")
    if float(cfg.get("matched_pass_complete_rate_minimum", -1)) < 0.0 or float(cfg["matched_pass_complete_rate_minimum"]) > 1.0:
        raise ValueError("invalid matched-pass stability threshold")
    return cfg


def residual_cases() -> list[dict]:
    rows = prior.target_cases()
    if len(rows) != 24:
        raise ValueError("v0.0.45 residual cohort changed")
    return rows


def structural_cases() -> list[dict]:
    rows = [
        case for case in residual_cases()
        if case["subtype"] in STRUCTURAL_SUBTYPES and case["id"] not in ENTRY_SUPPRESSION_IDS
    ]
    if len(rows) != 17:
        raise ValueError(f"expected 17 structural cases, got {len(rows)}")
    counts = Counter(case["subtype"] for case in rows)
    if dict(counts) != EXPECTED_STRUCTURAL_COUNTS:
        raise ValueError(f"structural cohort composition changed: {dict(counts)}")
    if not NONRESCUED_FAILURE_IDS <= {case["id"] for case in rows}:
        raise ValueError("structural cohort lost a non-rescued failure")
    return rows


def entry_suppression_cases() -> list[dict]:
    by_id = {case["id"]: case for case in residual_cases()}
    if not ENTRY_SUPPRESSION_IDS <= set(by_id):
        raise ValueError("entry-suppression case missing from frozen cohort")
    return [by_id[case_id] for case_id in sorted(ENTRY_SUPPRESSION_IDS)]


def canonical_prefix(cfg: dict) -> str:
    return "".join(item["text"] for item in cfg["canonical_prefix_segments"])


def token_plan(tokenizer, prompt: str, segments: list[dict]) -> tuple[list[int], list[dict]]:
    """Map semantic structural stages to exact vocabulary tokens without values."""
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("empty prompt")
    running_text = prompt
    running_ids = list(prompt_ids)
    plan: list[dict] = []
    for segment in segments:
        running_text += segment["text"]
        encoded = tokenizer.encode(running_text)
        if encoded[: len(running_ids)] != running_ids:
            raise ValueError(f"tokenization changed before structural stage {segment['name']}")
        new_ids = encoded[len(running_ids) :]
        if not new_ids:
            raise ValueError(f"structural stage produced no tokens: {segment['name']}")
        for offset, token_id in enumerate(new_ids):
            plan.append({
                "stage": segment["name"],
                "stage_token_index": offset,
                "expected_token_id": int(token_id),
                "expected_token_text": prior._decode_token(tokenizer, int(token_id)),
            })
        running_ids = list(encoded)
    if tokenizer.decode(running_ids[len(prompt_ids) :]).strip() == "":
        raise ValueError("structural prefix tokenization unexpectedly decoded empty")
    return list(prompt_ids), plan


def structural_prefix_probe(model, tokenizer, torch, prompt: str, segments: list[dict]) -> dict:
    """Teacher-force non-value structure only; never receives the target value."""
    prompt_ids, plan = token_plan(tokenizer, prompt, segments)
    expected_prefix_ids: list[int] = []
    steps = []
    with torch.inference_mode():
        for index, item in enumerate(plan):
            x = torch.tensor([prompt_ids + expected_prefix_ids], dtype=torch.long, device="cpu")
            if x.shape[1] >= int(model.cfg.block_size):
                raise ValueError("structural-prefix probe exceeds context window")
            logits, _ = model(x)
            next_logits = logits[0, -1, :].float()
            if not bool(torch.isfinite(next_logits).all().item()):
                raise ValueError("non-finite structural-prefix logits")
            expected_id = int(item["expected_token_id"])
            expected_logit = float(next_logits[expected_id].item())
            masked = next_logits.clone()
            masked[expected_id] = -float("inf")
            best_other_id = int(torch.argmax(masked).item())
            best_other_logit = float(masked[best_other_id].item())
            greedy_id = int(torch.argmax(next_logits).item())
            probs = torch.softmax(next_logits, dim=-1)
            rank = int((next_logits > next_logits[expected_id]).sum().item()) + 1
            steps.append({
                "index": index,
                **item,
                "expected_rank": rank,
                "expected_logit": expected_logit,
                "expected_probability": float(probs[expected_id].item()),
                "expected_margin_vs_best_other": expected_logit - best_other_logit,
                "greedy_token_id": greedy_id,
                "greedy_token_text": prior._decode_token(tokenizer, greedy_id),
                "greedy_match": greedy_id == expected_id,
                "best_other_token_id": best_other_id,
                "best_other_token_text": prior._decode_token(tokenizer, best_other_id),
            })
            expected_prefix_ids.append(expected_id)
    mismatches = [step for step in steps if not step["greedy_match"]]
    return {
        "prefix_token_count": len(steps),
        "all_structural_tokens_greedy": not mismatches,
        "structural_top1_rate": sum(step["greedy_match"] for step in steps) / len(steps),
        "first_non_greedy": mismatches[0] if mismatches else None,
        "steps": steps,
    }


def _group_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"cases": 0}
    all_steps = [step for row in rows for step in row["prefix_probe"]["steps"]]
    complete = sum(bool(row["prefix_probe"]["all_structural_tokens_greedy"]) for row in rows)
    first_stages = Counter(
        row["prefix_probe"]["first_non_greedy"]["stage"]
        for row in rows if row["prefix_probe"]["first_non_greedy"] is not None
    )
    return {
        "cases": len(rows),
        "complete_prefix_top1": complete,
        "complete_prefix_top1_rate": complete / len(rows),
        "token_top1_rate": sum(bool(step["greedy_match"]) for step in all_steps) / len(all_steps),
        "median_expected_rank": float(median(step["expected_rank"] for step in all_steps)),
        "mean_expected_rank": float(mean(step["expected_rank"] for step in all_steps)),
        "median_expected_margin": float(median(step["expected_margin_vs_best_other"] for step in all_steps)),
        "first_divergence_stage_counts": dict(first_stages),
    }


def stage_summary(rows: list[dict], failure_ids: set[str]) -> dict:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"failures": [], "passes": []})
    for row in rows:
        group = "failures" if row["id"] in failure_ids else "passes"
        for step in row["prefix_probe"]["steps"]:
            grouped[step["stage"]][group].append(step)
    out = {}
    for stage, groups in grouped.items():
        out[stage] = {}
        for group, steps in groups.items():
            out[stage][group] = {
                "tokens": len(steps),
                "top1_rate": sum(bool(step["greedy_match"]) for step in steps) / len(steps) if steps else None,
                "median_rank": float(median(step["expected_rank"] for step in steps)) if steps else None,
                "median_margin": float(median(step["expected_margin_vs_best_other"] for step in steps)) if steps else None,
            }
    return out


def interpret(structural_rows: list[dict], cfg: dict) -> dict:
    failures = [row for row in structural_rows if row["id"] in NONRESCUED_FAILURE_IDS]
    passes = [row for row in structural_rows if row["id"] not in NONRESCUED_FAILURE_IDS]
    if len(failures) != 4 or len(passes) != 13:
        raise ValueError("unexpected structural failure/pass split")
    failure_summary = _group_summary(failures)
    pass_summary = _group_summary(passes)
    pass_stable = pass_summary["complete_prefix_top1_rate"] >= float(cfg["matched_pass_complete_rate_minimum"])
    failure_complete = int(failure_summary["complete_prefix_top1"])
    if not pass_stable:
        regime = "canonical_prefix_not_stable_enough_for_binary_localization"
    elif failure_complete == len(failures):
        regime = "nonrescued_failures_diverge_after_structural_prefix"
    elif failure_complete == 0:
        regime = "nonrescued_failures_diverge_within_structural_prefix"
    else:
        regime = "mixed_structural_prefix_and_later_divergence"
    return {
        "heuristic_only": True,
        "regime": regime,
        "matched_prefix_stable": pass_stable,
        "failures": failure_summary,
        "matched_passes": pass_summary,
        "stage_summary": stage_summary(structural_rows, NONRESCUED_FAILURE_IDS),
        "failure_first_divergence": {
            row["id"]: (
                None if row["prefix_probe"]["first_non_greedy"] is None
                else {
                    "index": row["prefix_probe"]["first_non_greedy"]["index"],
                    "stage": row["prefix_probe"]["first_non_greedy"]["stage"],
                    "expected_rank": row["prefix_probe"]["first_non_greedy"]["expected_rank"],
                    "expected_margin_vs_best_other": row["prefix_probe"]["first_non_greedy"]["expected_margin_vs_best_other"],
                    "greedy_token_text": row["prefix_probe"]["first_non_greedy"]["greedy_token_text"],
                }
            )
            for row in failures
        },
        "meaning": "Descriptive structural-prefix evidence only; no learning phase is authorized.",
    }


def summary_markdown(report: dict) -> str:
    i = report["interpretation"]
    lines = [
        "# Ember v0.0.46 structural-prefix diagnostic",
        "",
        "CPU-only. Model weights and prompts are frozen; target argument values are never teacher-forced.",
        "",
        f"- Structural cases: **{report['structural_case_count']}**",
        f"- Non-rescued failures: **{len(report['nonrescued_failure_ids'])}**",
        f"- Matched passes: **{report['matched_pass_count']}**",
        f"- Entry-suppression cases tracked separately: **{len(report['entry_suppression'])}**",
        f"- Diagnostic regime: **{i['regime']}**",
        "",
        "| Group | Complete prefix top-1 | Token top-1 | Median rank | Median margin |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, key in (("Four failures", "failures"), ("13 matched passes", "matched_passes")):
        g = i[key]
        lines.append(
            f"| {label} | {g['complete_prefix_top1']}/{g['cases']} ({g['complete_prefix_top1_rate']:.1%}) "
            f"| {g['token_top1_rate']:.1%} | {g['median_expected_rank']:.1f} | {g['median_expected_margin']:+.4f} |"
        )
    lines += ["", "First structural divergence for each non-rescued failure:", ""]
    for case_id, divergence in sorted(i["failure_first_divergence"].items()):
        if divergence is None:
            lines.append(f"- `{case_id}`: none before the value boundary")
        else:
            lines.append(
                f"- `{case_id}`: `{divergence['stage']}` (rank {divergence['expected_rank']}, "
                f"margin {divergence['expected_margin_vs_best_other']:+.4f})"
            )
    lines += [
        "",
        "The prefix ends at the opening query-value quote. No target-value token is inserted or scored.",
        "`placement_learning_authorized=false` remains in force.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("v046-results"))
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
        "version": "0.0.46",
        "phase": cfg["phase"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ERROR",
        "code_commit": os.environ.get("GITHUB_SHA"),
        "training_authorized": False,
        "gpu_training_authorized": False,
        "production_authorized": False,
        "placement_learning_authorized": False,
        "source_v045_run": 34289689685,
        "source_v045_regime": "deeper_decoding_difference_after_envelope_entry",
    }
    try:
        source_cfg = json.loads(base.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="ember-v046-") as td:
            model, tokenizer, source, _splits, source_ref = base.load_inputs(source_cfg, Path(td), torch)
            model.to("cpu").eval()
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != cfg["source_version"]:
                raise ValueError("v0.0.46 must measure the pinned v0.0.31 step-479 checkpoint")

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

            residual = residual_cases()
            baseline_rows = []
            for case in residual:
                generation = base.semantic_gate.generate_completion(
                    model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"])
                )
                score = control.score_case(case, generation["completion"])
                baseline_rows.append({"id": case["id"], "subtype": case["subtype"], "score": score})
            failure_ids = {row["id"] for row in baseline_rows if not row["score"]["envelope_json_valid"]}
            if failure_ids != prior.EXPECTED_FAILURE_IDS:
                raise ValueError(f"v0.0.45 six-case failure set changed: {sorted(failure_ids)}")

            structural_rows = []
            for case in structural_cases():
                probe = structural_prefix_probe(
                    model, tokenizer, torch, case["prompt"], cfg["canonical_prefix_segments"]
                )
                structural_rows.append({
                    "id": case["id"],
                    "kind": case["kind"],
                    "subtype": case["subtype"],
                    "is_nonrescued_failure": case["id"] in NONRESCUED_FAILURE_IDS,
                    "prefix_probe": probe,
                })
                first = probe["first_non_greedy"]
                print(json.dumps({
                    "event": "structural_case",
                    "id": case["id"],
                    "subtype": case["subtype"],
                    "failure": case["id"] in NONRESCUED_FAILURE_IDS,
                    "prefix_top1": probe["all_structural_tokens_greedy"],
                    "first_divergence_stage": None if first is None else first["stage"],
                    "first_divergence_rank": None if first is None else first["expected_rank"],
                }), flush=True)

            entry_rows = []
            for case in entry_suppression_cases():
                first = prior.first_token_probe(model, tokenizer, torch, case["prompt"], 8)
                forced = prior.forced_tool_completion(
                    model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"])
                )
                forced_score = control.score_case(case, forced["completion"])
                if not (forced_score["envelope_json_valid"] and forced_score["tool_name_correct"]):
                    raise ValueError(f"v0.0.45 entry rescue no longer reproduces for {case['id']}")
                entry_rows.append({
                    "id": case["id"],
                    "subtype": case["subtype"],
                    "first_token": first,
                    "forced_score": forced_score,
                })

            interpretation = interpret(structural_rows, cfg)
            structural_failure_ids = {
                row["id"] for row in structural_rows if row["is_nonrescued_failure"]
            }
            if structural_failure_ids != NONRESCUED_FAILURE_IDS:
                raise ValueError("non-rescued structural failure set changed")
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
                frozen_90_prompt_sha256=prior.prompt_digest(prior.frozen_v044_cases()),
                canonical_prefix=canonical_prefix(cfg),
                canonical_prefix_contains_target_value=False,
                residual_failure_ids=sorted(failure_ids),
                structural_case_count=len(structural_rows),
                nonrescued_failure_ids=sorted(structural_failure_ids),
                matched_pass_count=len(structural_rows) - len(structural_failure_ids),
                structural_cases=structural_rows,
                entry_suppression=entry_rows,
                interpretation=interpretation,
                meaning="Structural-prefix localization only. Target values remain unforced and no learning is authorized.",
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
        "nonrescued_failure_ids": report["nonrescued_failure_ids"],
        "interpretation": report["interpretation"],
        "entry_suppression": [
            {
                "id": row["id"],
                "tool_rank": row["first_token"]["tool_rank"],
                "tool_margin": row["first_token"]["tool_margin_vs_best_other"],
                "best_other_text": row["first_token"]["best_other_text"],
            }
            for row in report["entry_suppression"]
        ],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
