"""CPU-only serialization-neutral envelope-template diagnostic for Ember v0.0.47.

v0.0.46 proved that re-tokenizing one compact JSON spelling is not a valid
structural control: successful envelopes themselves follow a different token
path. This runner derives non-value prefix templates from the *actual generated
IDs* of successful v0.0.44/v0.0.45 residual envelopes, then cross-validates those
templates leave-one-out on matched passing cases before using them to localize
the four non-rescued failures.

No target argument value is ever included in a template. Templates stop at the
last generated token whose decoded text ends no later than the opening quote of
the query value. No optimizer, backward pass, GPU path, checkpoint write,
promotion, deployment, or production integration exists here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from statistics import mean, median
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import ember_structural_prefix_v046 as prior

v045 = prior.prior
base = v045.base
control = v045.control
DEFAULT_CONFIG = ROOT / "config/ember_envelope_template_v0.0.47.json"
TOOL = v045.TOOL
NONRESCUED_FAILURE_IDS = set(prior.NONRESCUED_FAILURE_IDS)
ENTRY_SUPPRESSION_IDS = set(prior.ENTRY_SUPPRESSION_IDS)
EXPECTED_FAILURE_IDS = set(v045.EXPECTED_FAILURE_IDS)


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if path.resolve() != DEFAULT_CONFIG.resolve() or cfg.get("version") != "0.0.47":
        raise ValueError("unsupported v0.0.47 diagnostic configuration")
    if any(cfg.get(key) is not False for key in (
        "training_authorized", "gpu_training_authorized", "production_authorized",
        "placement_learning_authorized",
    )):
        raise ValueError("v0.0.47 is diagnostic-only; all authorization flags must be false")
    expected = {
        "expected_reference_cases": 4,
        "expected_residual_cases": 24,
        "expected_residual_failures": 6,
        "expected_template_sources": 18,
        "expected_structural_cases": 17,
        "expected_nonrescued_failures": 4,
        "expected_structural_passes": 13,
        "expected_entry_suppression_cases": 2,
    }
    for key, value in expected.items():
        if int(cfg.get(key, -1)) != value:
            raise ValueError(f"v0.0.47 changed {key}")
    threshold = float(cfg.get("matched_pass_leave_one_out_complete_rate_minimum", -1))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("invalid leave-one-out stability threshold")
    if int(cfg.get("generation_budget", 0)) != int(v045.V044_CONFIG["generation_budget"]):
        raise ValueError("generation budget changed from the frozen envelope battery")
    return cfg


def residual_cases() -> list[dict]:
    rows = v045.target_cases()
    if len(rows) != 24:
        raise ValueError("v0.0.45 residual cohort changed")
    return rows


def _query_value_start(completion: str) -> int:
    match = re.search(r'"query"\s*:\s*"', completion)
    if match is None:
        raise ValueError("valid web_search envelope lacks a textual query boundary")
    return match.end()


def _stage_milestones(prefix_text: str) -> list[tuple[int, str]]:
    milestones: list[tuple[int, str]] = []
    marker = prefix_text.find(TOOL)
    if marker < 0:
        raise ValueError("template prefix lacks tool marker")
    milestones.append((marker + len(TOOL), "tool_marker"))
    patterns = [
        (r'"name"', "name_key"),
        (r'"web_search"', "tool_name"),
        (r'"arguments"', "arguments_key"),
        (r'"query"', "query_key"),
    ]
    cursor = marker + len(TOOL)
    for pattern, name in patterns:
        match = re.search(pattern, prefix_text[cursor:])
        if match is None:
            raise ValueError(f"template prefix lacks {name}")
        end = cursor + match.end()
        milestones.append((end, name))
        cursor = end
    milestones.append((len(prefix_text), "value_boundary"))
    return milestones


def _stage_for_end(char_end: int, milestones: list[tuple[int, str]]) -> str:
    for end, name in milestones:
        if char_end <= end:
            return name
    return milestones[-1][1]


def extract_template(row: dict, tokenizer) -> dict:
    """Extract only the successful generated structural prefix, never its value."""
    score = row["baseline_score"]
    if not (score["envelope_json_valid"] and score["tool_name_correct"]):
        raise ValueError("template sources must be successful correct-tool envelopes")
    generated_ids = list(row["baseline"]["generated_ids"])
    completion = row["baseline"]["completion"]
    value_start = _query_value_start(completion)
    keep = 0
    decoded_prefix = ""
    char_ends: list[int] = []
    for index in range(1, len(generated_ids) + 1):
        decoded = tokenizer.decode(generated_ids[:index])
        if not completion.startswith(decoded):
            raise ValueError("generated token prefix does not round-trip as a completion prefix")
        if len(decoded) > value_start:
            break
        keep = index
        decoded_prefix = decoded
        char_ends.append(len(decoded))
    if keep < 1 or not decoded_prefix.startswith(TOOL):
        raise ValueError("no safe non-value envelope prefix could be extracted")
    if len(decoded_prefix) > value_start:
        raise ValueError("template crossed the query-value boundary")
    prefix_ids = generated_ids[:keep]
    milestones = _stage_milestones(decoded_prefix)
    stages = [_stage_for_end(end, milestones) for end in char_ends]
    digest = hashlib.sha256(",".join(str(v) for v in prefix_ids).encode("ascii")).hexdigest()[:16]
    return {
        "template_id": digest,
        "prefix_ids": prefix_ids,
        "prefix_text": decoded_prefix,
        "prefix_token_count": len(prefix_ids),
        "value_char_start": value_start,
        "prefix_char_end": len(decoded_prefix),
        "stages": stages,
        "source_ids": [row["id"]],
    }


def build_template_library(rows: list[dict], tokenizer) -> list[dict]:
    grouped: dict[tuple[int, ...], dict] = {}
    for row in rows:
        if not (row["baseline_score"]["envelope_json_valid"] and row["baseline_score"]["tool_name_correct"]):
            continue
        template = extract_template(row, tokenizer)
        key = tuple(template["prefix_ids"])
        if key not in grouped:
            grouped[key] = template
        else:
            grouped[key]["source_ids"].extend(template["source_ids"])
    templates = sorted(grouped.values(), key=lambda item: (-len(item["source_ids"]), item["template_id"]))
    for template in templates:
        template["source_ids"] = sorted(set(template["source_ids"]))
    return templates


def score_template(model, tokenizer, torch, prompt: str, template: dict) -> dict:
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("empty prompt")
    forced: list[int] = []
    steps = []
    with torch.inference_mode():
        for index, expected_id in enumerate(template["prefix_ids"]):
            x = torch.tensor([prompt_ids + forced], dtype=torch.long, device="cpu")
            if x.shape[1] >= int(model.cfg.block_size):
                raise ValueError("template probe exceeds context window")
            logits, _ = model(x)
            next_logits = logits[0, -1, :].float()
            if not bool(torch.isfinite(next_logits).all().item()):
                raise ValueError("non-finite template logits")
            expected_id = int(expected_id)
            expected_logit = float(next_logits[expected_id].item())
            masked = next_logits.clone()
            masked[expected_id] = -float("inf")
            best_other_id = int(torch.argmax(masked).item())
            best_other_logit = float(masked[best_other_id].item())
            greedy_id = int(torch.argmax(next_logits).item())
            log_probs = torch.log_softmax(next_logits, dim=-1)
            steps.append({
                "index": index,
                "stage": template["stages"][index],
                "expected_token_id": expected_id,
                "expected_token_text": v045._decode_token(tokenizer, expected_id),
                "expected_rank": int((next_logits > next_logits[expected_id]).sum().item()) + 1,
                "expected_margin_vs_best_other": expected_logit - best_other_logit,
                "expected_log_probability": float(log_probs[expected_id].item()),
                "greedy_token_id": greedy_id,
                "greedy_token_text": v045._decode_token(tokenizer, greedy_id),
                "greedy_match": greedy_id == expected_id,
            })
            forced.append(expected_id)
    mismatches = [step for step in steps if not step["greedy_match"]]
    return {
        "template_id": template["template_id"],
        "source_ids": template["source_ids"],
        "prefix_token_count": len(steps),
        "complete_top1": not mismatches,
        "token_top1_rate": sum(step["greedy_match"] for step in steps) / len(steps),
        "mean_log_probability": mean(step["expected_log_probability"] for step in steps),
        "mean_margin": mean(step["expected_margin_vs_best_other"] for step in steps),
        "first_non_greedy": mismatches[0] if mismatches else None,
        "steps": steps,
    }


def _score_key(result: dict) -> tuple:
    return (
        int(result["complete_top1"]),
        float(result["token_top1_rate"]),
        float(result["mean_log_probability"]),
        float(result["mean_margin"]),
    )


def best_template_score(model, tokenizer, torch, prompt: str, templates: list[dict], case_id: str) -> dict:
    allowed = [
        template for template in templates
        if any(source_id != case_id for source_id in template["source_ids"])
    ]
    if not allowed:
        raise ValueError(f"no leave-one-out template available for {case_id}")
    scored = [score_template(model, tokenizer, torch, prompt, template) for template in allowed]
    best = max(scored, key=_score_key)
    return {"best": best, "candidate_count": len(scored)}


def _group_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"cases": 0}
    best = [row["template_probe"]["best"] for row in rows]
    first = Counter(
        result["first_non_greedy"]["stage"]
        for result in best if result["first_non_greedy"] is not None
    )
    return {
        "cases": len(rows),
        "complete_top1": sum(result["complete_top1"] for result in best),
        "complete_top1_rate": sum(result["complete_top1"] for result in best) / len(best),
        "median_token_top1_rate": float(median(result["token_top1_rate"] for result in best)),
        "median_mean_log_probability": float(median(result["mean_log_probability"] for result in best)),
        "first_divergence_stage_counts": dict(first),
    }


def interpret(rows: list[dict], cfg: dict) -> dict:
    failures = [row for row in rows if row["id"] in NONRESCUED_FAILURE_IDS]
    passes = [row for row in rows if row["id"] not in NONRESCUED_FAILURE_IDS]
    if len(failures) != 4 or len(passes) != 13:
        raise ValueError("unexpected v0.0.47 structural split")
    failure_summary = _group_summary(failures)
    pass_summary = _group_summary(passes)
    stable = pass_summary["complete_top1_rate"] >= float(cfg["matched_pass_leave_one_out_complete_rate_minimum"])
    failure_complete = int(failure_summary["complete_top1"])
    if not stable:
        regime = "successful_envelope_templates_not_leave_one_out_stable"
    elif failure_complete == 4:
        regime = "nonrescued_failures_diverge_after_observed_envelope_prefix"
    elif failure_complete == 0:
        regime = "nonrescued_failures_diverge_within_observed_envelope_prefix"
    else:
        regime = "mixed_observed_prefix_and_later_divergence"
    return {
        "heuristic_only": True,
        "regime": regime,
        "matched_templates_stable": stable,
        "failures": failure_summary,
        "matched_passes": pass_summary,
        "failure_first_divergence": {
            row["id"]: (
                None if row["template_probe"]["best"]["first_non_greedy"] is None
                else {
                    "stage": row["template_probe"]["best"]["first_non_greedy"]["stage"],
                    "index": row["template_probe"]["best"]["first_non_greedy"]["index"],
                    "expected_rank": row["template_probe"]["best"]["first_non_greedy"]["expected_rank"],
                    "expected_margin_vs_best_other": row["template_probe"]["best"]["first_non_greedy"]["expected_margin_vs_best_other"],
                    "greedy_token_text": row["template_probe"]["best"]["first_non_greedy"]["greedy_token_text"],
                    "template_id": row["template_probe"]["best"]["template_id"],
                }
            )
            for row in failures
        },
        "meaning": "Serialization-neutral descriptive evidence only; no learning phase is authorized.",
    }


def summary_markdown(report: dict) -> str:
    i = report["interpretation"]
    lines = [
        "# Ember v0.0.47 serialization-neutral envelope-template diagnostic",
        "",
        "CPU-only. Templates come from actual successful generated token IDs and stop before any query value token.",
        "",
        f"- Successful template sources: **{report['template_source_count']}**",
        f"- Unique observed templates: **{len(report['templates'])}**",
        f"- Structural comparison cases: **{report['structural_case_count']}**",
        f"- Matched-pass leave-one-out complete rate: **{i['matched_passes']['complete_top1']}/{i['matched_passes']['cases']} ({i['matched_passes']['complete_top1_rate']:.1%})**",
        f"- Diagnostic regime: **{i['regime']}**",
        "",
        "| Group | Complete observed prefix top-1 | Median token top-1 | Median mean log-prob |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in (("Four failures", "failures"), ("13 matched passes", "matched_passes")):
        g = i[key]
        lines.append(
            f"| {label} | {g['complete_top1']}/{g['cases']} ({g['complete_top1_rate']:.1%}) | "
            f"{g['median_token_top1_rate']:.1%} | {g['median_mean_log_probability']:+.4f} |"
        )
    lines += [
        "",
        "Leave-one-out means a passing case cannot validate a template supported only by itself.",
        "Target argument values are never part of any forced template.",
        "No optimizer, GPU, training, promotion, deployment, or production integration ran.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("v047-results"))
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
        "version": "0.0.47",
        "phase": cfg["phase"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ERROR",
        "code_commit": os.environ.get("GITHUB_SHA"),
        "training_authorized": False,
        "gpu_training_authorized": False,
        "production_authorized": False,
        "placement_learning_authorized": False,
        "source_v046_run": 34303409879,
    }
    try:
        source_cfg = json.loads(base.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="ember-v047-") as td:
            model, tokenizer, source, _splits, source_ref = base.load_inputs(source_cfg, Path(td), torch)
            model.to("cpu").eval()
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != cfg["source_version"]:
                raise ValueError("v0.0.47 must measure the pinned v0.0.31 step-479 checkpoint")

            reference_rows = []
            for case in control.reference_cases(v045.V044_CONFIG):
                generation = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"]))
                score = control.score_reference(case, generation["completion"])
                reference_rows.append({"id": case["id"], **generation, "score": score})
            reference_gate = control.reference_summary(reference_rows, v045.V044_CONFIG)
            if not reference_gate["passed"]:
                raise ValueError("historical four-case control no longer reproduces 4/4")

            cases = residual_cases()
            baseline_rows = []
            for case in cases:
                baseline = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"]))
                score = control.score_case(case, baseline["completion"])
                baseline_rows.append({
                    "id": case["id"], "kind": case["kind"], "subtype": case["subtype"],
                    "prompt": case["prompt"], "baseline": baseline, "baseline_score": score,
                })
            failure_ids = {row["id"] for row in baseline_rows if not row["baseline_score"]["envelope_json_valid"]}
            if failure_ids != EXPECTED_FAILURE_IDS:
                raise ValueError(f"frozen residual failures changed: {sorted(failure_ids)}")
            source_rows = [row for row in baseline_rows if row["baseline_score"]["envelope_json_valid"] and row["baseline_score"]["tool_name_correct"]]
            if len(source_rows) != int(cfg["expected_template_sources"]):
                raise ValueError("unexpected successful template-source count")
            templates = build_template_library(source_rows, tokenizer)
            if not templates:
                raise ValueError("no successful envelope templates were extracted")

            structural_ids = {case["id"] for case in prior.structural_cases()}
            by_id = {row["id"]: row for row in baseline_rows}
            structural_rows = []
            for case_id in sorted(structural_ids):
                row = dict(by_id[case_id])
                row["template_probe"] = best_template_score(model, tokenizer, torch, row["prompt"], templates, case_id)
                structural_rows.append(row)
                best = row["template_probe"]["best"]
                print(json.dumps({
                    "event": "template_case", "id": case_id,
                    "failure": case_id in NONRESCUED_FAILURE_IDS,
                    "best_template": best["template_id"],
                    "complete_top1": best["complete_top1"],
                    "token_top1_rate": best["token_top1_rate"],
                    "first_divergence_stage": None if best["first_non_greedy"] is None else best["first_non_greedy"]["stage"],
                }), flush=True)

            interpretation = interpret(structural_rows, cfg)
            entry = []
            for case_id in sorted(ENTRY_SUPPRESSION_IDS):
                row = by_id[case_id]
                probe = v045.first_token_probe(model, tokenizer, torch, row["prompt"], 8)
                entry.append({
                    "id": case_id,
                    "tool_rank": probe["tool_rank"],
                    "tool_margin": probe["tool_margin_vs_best_other"],
                    "best_other_text": probe["best_other_text"],
                })

            report.update(
                status="COMPLETE",
                source={
                    "repo_id": source_ref["repo_id"], "checkpoint_path": source_ref["checkpoint_path"],
                    "revision": source_ref["revision"], "checkpoint_sha256": source_ref["checkpoint_sha256"],
                    "step": source["step"], "version": source["train_config"]["version"],
                },
                reference_gate=reference_gate,
                failure_ids=sorted(failure_ids),
                template_source_count=len(source_rows),
                templates=templates,
                structural_case_count=len(structural_rows),
                structural_cases=structural_rows,
                entry_suppression=entry,
                interpretation=interpretation,
                meaning="Observed-token envelope-template diagnostic only; no model or prompt mutation and no learning authorization.",
            )
    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        if report.get("status") == "COMPLETE":
            (output / "summary.md").write_text(summary_markdown(report), encoding="utf-8")

    print(json.dumps({
        "event": "complete", "status": report["status"],
        "template_count": len(report["templates"]),
        "interpretation": report["interpretation"],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
