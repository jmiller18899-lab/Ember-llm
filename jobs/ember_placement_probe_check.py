"""CPU-only diagnostic: is the v0.0.50 placement probe able to move at all?

v0.0.50 reported `placement_token_top1_gain` of exactly 0.0 at all four
checkpoints while entry moved 0 -> 1 -> 5 -> 6 on the same student, and while
placement loss oscillated with no trend. The plumbing is correct -- the probe is
called on the same updated student, and the gain compares latest to before -- so
before spending another optimizer run this asks whether the *metric* can move.

Three things are measured on the untouched v0.0.31 step-479 source:

1. the headline probe, using v0.0.50's exact configuration, development values
   and discovered template, so the number is comparable to the run;
2. a decomposition by token position, because a teacher-forced probe that forces
   every previous value token measures two very different things at once -- the
   hard first token after the value-free prefix, and the easy continuation
   inside a value the model already copies well; and
3. the same probe on v0.0.48's development values, to separate "the metric is
   blind" from "v0.0.50 happened to draw an easier or harder set".

No optimizer, model write, GPU submission, promotion, deployment, or production
integration is possible in this runner. It reads a checkpoint and reports.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import median
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ember_v049_replay.replay_value_rows reads a config key its own config does not
# define; the compat shim monkeypatches the working version at import. v0.0.50's
# value bookkeeping goes through that function, so the shim has to be imported
# before ember_v050_data is used -- exactly as ember_v050_canary_compat does.
from jobs import ember_v049_replay_compat  # noqa: F401
from jobs import ember_v048_data as v048d
from jobs import ember_v048_objectives as objectives
from jobs import ember_v050_data as v050d

base = v048d.base
V050_CONFIG = ROOT / "config/ember_teacher_kl_v0.0.50.json"

# v0.0.48's published before-probe, for scale.
V048_REPORTED_BEFORE = {"exact_top1": 1, "cases": 24, "token_top1": 90, "tokens": 171}


def load_config(path: Path = V050_CONFIG) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("version") != "0.0.50":
        raise ValueError("the probe check must reuse the v0.0.50 configuration verbatim")
    if any(cfg.get(k) is not False for k in (
        "gpu_training_authorized", "production_authorized", "promotion_authorized"
    )):
        raise ValueError("probe check refuses a configuration that authorizes anything")
    return cfg


def positional_breakdown(probe: dict) -> dict:
    """Split the teacher-forced probe by token index within the value.

    Index 0 is the first token after the value-free envelope prefix -- the
    boundary the phase is actually trying to teach. Indexes 1+ are continuation
    inside a value, which the copy battery already scores above 90%.
    """
    by_index_total: dict[int, int] = defaultdict(int)
    by_index_top1: dict[int, int] = defaultdict(int)
    ranks_first: list[int] = []
    ranks_rest: list[int] = []
    for row in probe["rows"]:
        for index, token in enumerate(row["tokens"]):
            by_index_total[index] += 1
            if token["top1"]:
                by_index_top1[index] += 1
            (ranks_first if index == 0 else ranks_rest).append(int(token["rank"]))
    positions = sorted(by_index_total)
    return {
        "by_index": [
            {
                "index": index,
                "tokens": by_index_total[index],
                "top1": by_index_top1[index],
                "rate": by_index_top1[index] / by_index_total[index],
            }
            for index in positions
        ],
        "first_token": _rank_stats(ranks_first),
        "later_tokens": _rank_stats(ranks_rest),
    }


def _rank_stats(ranks: list[int]) -> dict:
    if not ranks:
        return {"tokens": 0}
    top1 = sum(1 for rank in ranks if rank == 1)
    ordered = sorted(ranks)
    return {
        "tokens": len(ranks),
        "top1": top1,
        "top1_rate": top1 / len(ranks),
        "median_rank": median(ordered),
        "p90_rank": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max_rank": ordered[-1],
    }


def headroom(probe: dict, breakdown: dict, cfg: dict) -> dict:
    """Can the configured gate be reached, and how far away is it?"""
    tokens = probe["tokens"]
    first = breakdown["first_token"]
    wrong_first = first["tokens"] - first.get("top1", 0)
    ceiling_if_first_fixed = (probe["token_top1"] + wrong_first) / tokens if tokens else None

    # Cases whose only error is the first token are one flip from exact.
    one_flip = 0
    wrong_positions: list[int] = []
    for row in probe["rows"]:
        wrong = [i for i, token in enumerate(row["tokens"]) if not token["top1"]]
        wrong_positions.append(len(wrong))
        if wrong == [0]:
            one_flip += 1

    gate = cfg["selection_gate"]
    required_gain = float(gate["minimum_placement_token_top1_gain"])
    available = (ceiling_if_first_fixed - probe["token_top1_rate"]) if ceiling_if_first_fixed else 0.0
    return {
        "token_top1_rate": probe["token_top1_rate"],
        "ceiling_if_first_token_fixed": ceiling_if_first_fixed,
        "available_token_gain": available,
        "required_token_gain": required_gain,
        "token_gate_reachable": available >= required_gain,
        "tokens_needed_for_token_gate": int(round(required_gain * tokens)),
        "wrong_first_tokens": wrong_first,
        "cases_one_flip_from_exact": one_flip,
        "cases_already_exact": probe["exact_top1"],
        "exact_gate_reachable": one_flip > 0 or probe["exact_top1"] < probe["cases"],
        "wrong_position_counts": sorted(wrong_positions),
    }


def target_lengths(probe: dict) -> dict:
    lengths = sorted(row["target_token_count"] for row in probe["rows"])
    return {
        "minimum": lengths[0],
        "median": median(lengths),
        "maximum": lengths[-1],
        "degenerate_single_token_cases": sum(1 for n in lengths if n <= 1),
    }


def verdict(headroom_report: dict, breakdown: dict) -> dict:
    """State plainly whether the metric is a usable signal for placement."""
    first = breakdown["first_token"]
    later = breakdown["later_tokens"]
    findings = []
    saturated = headroom_report["token_top1_rate"] >= 0.95
    dominated = (
        later.get("top1_rate", 0.0) - first.get("top1_rate", 0.0) >= 0.5
        and later.get("tokens", 0) > first.get("tokens", 0)
    )
    if saturated:
        findings.append(
            "token_top1_rate starts at or above 95%, so the metric has almost no room to rise"
        )
    if dominated:
        findings.append(
            "token_top1_rate is dominated by already-correct continuation positions; "
            "the boundary token the phase is teaching is a minority of the tokens counted"
        )
    if not headroom_report["token_gate_reachable"]:
        findings.append(
            "the configured minimum_placement_token_top1_gain exceeds every gain "
            "available even if every first token were fixed"
        )
    if headroom_report["cases_one_flip_from_exact"] == 0:
        findings.append(
            "no development case is a single token from exact, so placement_exact_gain >= 1 "
            "requires fixing several positions in one case rather than one"
        )
    return {
        "metric_saturated": saturated,
        "metric_dominated_by_easy_positions": dominated,
        "token_gate_reachable": headroom_report["token_gate_reachable"],
        "probe_is_a_usable_placement_signal": not (saturated or dominated)
        and headroom_report["token_gate_reachable"],
        "findings": findings,
    }


def summary_markdown(report: dict) -> str:
    main = report["v050_development_set"]
    probe, breakdown = main["probe"], main["breakdown"]
    head, verdict_report = main["headroom"], main["verdict"]
    lines = [
        "# Ember placement probe check", "",
        "Read-only CPU diagnostic on the untouched v0.0.31 step-479 source.",
        "No optimizer, training, GPU submission, promotion, or deployment occurred.", "",
        "## Headline, on v0.0.50's own development set", "",
        "| Measurement | Value |", "| --- | ---: |",
        f"| Cases | {probe['cases']} |",
        f"| Exact top-1 | {probe['exact_top1']}/{probe['cases']} |",
        f"| Token top-1 | {probe['token_top1']}/{probe['tokens']} ({probe['token_top1_rate']:.1%}) |",
        f"| v0.0.48 reported before | {V048_REPORTED_BEFORE['token_top1']}/{V048_REPORTED_BEFORE['tokens']} "
        f"({V048_REPORTED_BEFORE['token_top1'] / V048_REPORTED_BEFORE['tokens']:.1%}) |",
        "",
        "## By token position", "",
        "Index 0 is the first token after the value-free prefix -- the boundary being taught.", "",
        "| Index | Tokens | Top-1 | Rate |", "| ---: | ---: | ---: | ---: |",
    ]
    for entry in breakdown["by_index"]:
        lines.append(
            f"| {entry['index']} | {entry['tokens']} | {entry['top1']} | {entry['rate']:.1%} |"
        )
    first, later = breakdown["first_token"], breakdown["later_tokens"]
    lines += [
        "",
        f"- First token: {first.get('top1', 0)}/{first['tokens']} top-1, median rank {first.get('median_rank')}, max rank {first.get('max_rank')}",
        f"- Later tokens: {later.get('top1', 0)}/{later['tokens']} top-1, median rank {later.get('median_rank')}, max rank {later.get('max_rank')}",
        "",
        "## Can the configured gate move?", "",
        f"- Current token top-1 rate: {head['token_top1_rate']:.1%}",
        f"- Ceiling if every first token were fixed: {head['ceiling_if_first_token_fixed']:.1%}",
        f"- Gain available: {head['available_token_gain']:.1%}; gain required: {head['required_token_gain']:.1%}",
        f"- Token gate reachable: {'yes' if head['token_gate_reachable'] else 'NO'}",
        f"- Tokens that must flip to clear the token gate: {head['tokens_needed_for_token_gate']}",
        f"- Cases one token from exact: {head['cases_one_flip_from_exact']}/{probe['cases']}",
        "",
        "## Verdict", "",
        f"- Metric saturated: {'yes' if verdict_report['metric_saturated'] else 'no'}",
        f"- Metric dominated by already-correct positions: {'yes' if verdict_report['metric_dominated_by_easy_positions'] else 'no'}",
        f"- **Usable placement signal: {'yes' if verdict_report['probe_is_a_usable_placement_signal'] else 'NO'}**",
        "",
    ]
    lines += [f"- {finding}" for finding in verdict_report["findings"]] or ["- no defect found"]
    control = report.get("v048_development_set")
    if control and "probe" in control:
        c = control["probe"]
        lines += [
            "", "## Control: the same probe on v0.0.48's development values", "",
            f"- Exact top-1: {c['exact_top1']}/{c['cases']}",
            f"- Token top-1: {c['token_top1']}/{c['tokens']} ({c['token_top1_rate']:.1%})",
            "",
            "A large gap between the two development sets would mean v0.0.50 drew a different "
            "difficulty, not that the metric is blind.", "",
        ]
    return "\n".join(lines)


def _measure(model, tokenizer, torch, cases, template, cfg) -> dict:
    probe = objectives.placement_probe(model, tokenizer, torch, cases, template)
    breakdown = positional_breakdown(probe)
    head = headroom(probe, breakdown, cfg)
    slim = {k: v for k, v in probe.items() if k != "rows"}
    slim["target_lengths"] = target_lengths(probe)
    return {
        "probe": slim,
        "rows": probe["rows"],
        "breakdown": breakdown,
        "headroom": head,
        "verdict": verdict(head, breakdown),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=V050_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("probe-check-results"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not os.environ.get("HF_TOKEN", "").strip():
        raise ValueError("HF_TOKEN is required for the pinned private checkpoint")

    import torch
    torch.set_num_threads(2)
    torch.manual_seed(int(cfg["seed"]))
    torch.use_deterministic_algorithms(True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "diagnostic": "placement-probe-check",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ERROR",
        "code_commit": os.environ.get("GITHUB_SHA"),
        "reads_only": True,
        "gpu_training_authorized": False,
        "production_authorized": False,
        "promotion_authorized": False,
        "question": (
            "v0.0.50 reported placement_token_top1_gain of exactly 0.0 at every checkpoint. "
            "Can this metric move at all on the untouched source?"
        ),
    }
    try:
        source_cfg = json.loads(base.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="ember-probe-check-") as td:
            model, tokenizer, source, _splits, source_ref = base.load_inputs(source_cfg, Path(td), torch)
            model.to("cpu").eval()
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != cfg["source_version"]:
                raise ValueError("probe check source is not the untouched v0.0.31 step 479")

            values = v050d.target_values(cfg)
            template, template_report = v048d.discover_template(model, tokenizer, torch, cfg, values["template"])
            place_dev = v048d.build_cases(
                {k: values["development"][k] for k in sorted(v050d.PLACEMENT_SUBTYPES)}, "v050_place_dev")
            print(json.dumps({"event": "reconstructed", "cases": len(place_dev),
                              "prefix_tokens": len(template["prefix_ids"])}), flush=True)
            main_result = _measure(model, tokenizer, torch, place_dev, template, cfg)
            print(json.dumps({"event": "v050_probe", **main_result["probe"]}), flush=True)

            # Control: v0.0.48's own development values, same template and probe.
            v048_cfg = v048d.load_config(v048d.DEFAULT_CONFIG)
            v048_values = v048d.synthetic_values(v048_cfg)
            v048_dev = v048d.build_cases(
                {k: v048_values["development"][k] for k in sorted(v048d.PLACEMENT_SUBTYPES)}, "v048_place_dev")
            control_result = _measure(model, tokenizer, torch, v048_dev, template, cfg)
            print(json.dumps({"event": "v048_control_probe", **control_result["probe"]}), flush=True)

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
                template=template_report,
                v050_development_set=main_result,
                v048_development_set=control_result,
                v048_reported_before=V048_REPORTED_BEFORE,
                meaning="Diagnostic only. Nothing is trained, promoted, or authorized by this runner.",
            )
    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        (output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        if report.get("status") == "COMPLETE":
            (output / "summary.md").write_text(summary_markdown(report), encoding="utf-8")

    print(json.dumps({
        "event": "complete",
        "status": report["status"],
        "verdict": report["v050_development_set"]["verdict"],
        "headroom": report["v050_development_set"]["headroom"],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
