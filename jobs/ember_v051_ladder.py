"""Ember v0.0.51: a CPU dose-response ladder for protected placement learning.

v0.0.48 moved placement token top-1 by 33 points and destroyed the model.
v0.0.50 preserved the model perfectly -- 85/90 familiar, teacher KL 0.0103 of an
0.08 budget -- and moved placement by exactly 0.0 at every checkpoint. The probe
check then ruled out both benign explanations: the metric is healthy (51.8% token
top-1, 12.4% of headroom available against 3% required, five token flips would
register a gain) and v0.0.50 did not draw an easier development set.

So the open question is not whether placement can be measured. It is whether any
update size exists where placement moves while frozen-teacher KL preservation
still holds. This runner measures that directly: the same protected recipe at
several placement update sizes, each restarting from the untouched v0.0.31
step-479 source, reporting placement gain against teacher KL and parameter drift
at every rung.

Two deliberate choices make the ladder interpretable:

- entry is dropped entirely (weight 0). It reached +6 under v0.0.50's most
  conservative recipe and its gradient only competes with placement here;
- the learning rate is the *only* thing that varies between rungs, and the first
  rung is v0.0.50's own rate, so it should reproduce the known 0.0 result and
  prove the ladder measures the same quantity.

This is a diagnostic, not a promotion canary. A ladder where nothing moves is a
complete result, not a failure, so it exits zero either way and says plainly
whether an operating point was found. No GPU submission, promotion, deployment,
or production integration is possible here.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ember_v049_replay reads a config key its own config does not define; the compat
# shim patches the working version at import, and v0.0.50's value bookkeeping
# goes through it. Import order matters here, so it is explicit.
from jobs import ember_v049_replay_compat  # noqa: F401
from jobs import ember_v048_data as v048d
from jobs import ember_v048_objectives as objectives
from jobs import ember_v048_regression as regression
from jobs import ember_v050_data as v050d
from jobs import ember_v050_distill as distill

base = v048d.base
DEFAULT_CONFIG = ROOT / "config/ember_placement_ladder_v0.0.51.json"
V050_CONFIG = ROOT / "config/ember_teacher_kl_v0.0.50.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("version") != "0.0.51":
        raise ValueError("unsupported v0.0.51 configuration")
    if cfg.get("cpu_learning_authorized") is not True:
        raise ValueError("CPU ladder authorization missing")
    if any(cfg.get(k) is not False for k in (
        "gpu_training_authorized", "production_authorized", "promotion_authorized"
    )):
        raise ValueError("v0.0.51 cannot authorize GPU, production, or promotion")

    if float(cfg.get("entry_loss_weight", -1)) != 0.0:
        raise ValueError("v0.0.51 drops the entry objective; its weight must be zero")
    weights = {k: float(cfg[k + "_loss_weight"]) for k in ("entry", "placement", "tool_kl", "copy_kl")}
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("objective weights must sum to one")
    preservation = weights["tool_kl"] + weights["copy_kl"]
    if preservation < 0.5 or preservation <= weights["placement"]:
        raise ValueError("frozen-teacher preservation must dominate the placement term")

    rungs = [float(x) for x in cfg.get("ladder_learning_rates", [])]
    if len(rungs) < 2:
        raise ValueError("a ladder needs at least two rungs")
    if rungs != sorted(rungs) or len(set(rungs)) != len(rungs):
        raise ValueError("ladder learning rates must be strictly increasing")
    if rungs[0] != float(cfg["control_learning_rate"]):
        raise ValueError("the first rung must be the v0.0.50 control rate")
    if max(rungs) > float(cfg["maximum_ladder_learning_rate"]):
        raise ValueError("ladder exceeds its own learning-rate bound")

    v050 = json.loads(V050_CONFIG.read_text(encoding="utf-8"))
    if float(cfg["control_learning_rate"]) != float(v050["learning_rate"]):
        raise ValueError("the control rung must match v0.0.50's measured learning rate")
    if int(cfg["steps_per_rung"]) != int(v050["max_optimizer_steps"]):
        raise ValueError("step count must match v0.0.50 so the control rung is comparable")
    for key in ("template_values_per_subtype", "train_values_per_subtype",
                "development_values_per_subtype", "tool_distill_values_per_variant",
                "copy_distill_values_per_variant", "generation_budget"):
        if cfg[key] != v050[key]:
            raise ValueError(f"{key} must match v0.0.50 so the development set is identical")
    point, gate = cfg["operating_point"], v050["selection_gate"]
    for key in ("minimum_placement_exact_gain", "minimum_placement_token_top1_gain",
                "maximum_tool_teacher_kl", "maximum_copy_teacher_kl"):
        if point[key] != gate[key]:
            raise ValueError(f"{key} must not be relaxed from v0.0.50")
    return cfg


def relative_drift(model, reference: dict, torch) -> dict:
    worst = 0.0
    total = 0.0
    count = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            ref = reference[name]
            norm = float(ref.norm().item())
            distance = float((param.detach() - ref).norm().item())
            value = distance / norm if norm > 0 else distance
            worst = max(worst, value)
            total += value
            count += 1
    return {"max_relative_drift": worst, "mean_relative_drift": total / count if count else 0.0}


def _global_grad_norm(model, torch) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().norm().item()) ** 2
    return total ** 0.5


def gradient_balance(student, teacher, tokenizer, torch, cfg, place_examples,
                     tool_rows, copy_rows, rng) -> dict:
    """How large is the placement gradient beside the preservation gradient?

    Measured before any optimizer step, with separate backward passes and no
    update. This is the instrument that says which rung should have mattered.
    """
    place_idx = [rng.randrange(len(place_examples)) for _ in range(int(cfg["placement_batch_size"]))]
    tool_idx = [rng.randrange(len(tool_rows)) for _ in range(int(cfg["tool_distill_batch_size"]))]
    copy_idx = [rng.randrange(len(copy_rows)) for _ in range(int(cfg["copy_distill_batch_size"]))]
    temperature = float(cfg["distill_temperature"])

    student.zero_grad(set_to_none=True)
    place_loss = objectives.batch_loss(student, torch, place_examples, place_idx)
    (float(cfg["placement_loss_weight"]) * place_loss).backward()
    weighted_place = _global_grad_norm(student, torch)

    student.zero_grad(set_to_none=True)
    tool_kl = distill.batch_teacher_kl(student, teacher, tokenizer, torch, tool_rows, tool_idx, temperature)
    copy_kl = distill.batch_teacher_kl(student, teacher, tokenizer, torch, copy_rows, copy_idx, temperature)
    (float(cfg["tool_kl_loss_weight"]) * tool_kl + float(cfg["copy_kl_loss_weight"]) * copy_kl).backward()
    weighted_preservation = _global_grad_norm(student, torch)
    student.zero_grad(set_to_none=True)

    return {
        "weighted_placement_grad_norm": weighted_place,
        "weighted_preservation_grad_norm": weighted_preservation,
        "placement_share": (
            weighted_place / (weighted_place + weighted_preservation)
            if (weighted_place + weighted_preservation) > 0 else None
        ),
        "placement_loss": float(place_loss.detach()),
        "tool_kl": float(tool_kl.detach()),
        "copy_kl": float(copy_kl.detach()),
    }


def run_rung(student, teacher, tokenizer, torch, cfg, learning_rate, place_examples,
             tool_rows, copy_rows, place_dev, template, before_place, reference_state) -> dict:
    """One rung: the shared protected recipe at a single placement update size."""
    rng = random.Random(int(cfg["seed"]))
    balance = gradient_balance(student, teacher, tokenizer, torch, cfg,
                               place_examples, tool_rows, copy_rows, random.Random(int(cfg["seed"])))
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate,
                                  weight_decay=float(cfg["weight_decay"]))
    temperature = float(cfg["distill_temperature"])
    total_steps = int(cfg["steps_per_rung"])
    interval = int(cfg["checkpoint_interval"])
    history, trajectory = [], []
    student.train()
    for step in range(1, total_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        place_idx = [rng.randrange(len(place_examples)) for _ in range(int(cfg["placement_batch_size"]))]
        tool_idx = [rng.randrange(len(tool_rows)) for _ in range(int(cfg["tool_distill_batch_size"]))]
        copy_idx = [rng.randrange(len(copy_rows)) for _ in range(int(cfg["copy_distill_batch_size"]))]
        place_loss = objectives.batch_loss(student, torch, place_examples, place_idx)
        tool_kl = distill.batch_teacher_kl(student, teacher, tokenizer, torch, tool_rows, tool_idx, temperature)
        copy_kl = distill.batch_teacher_kl(student, teacher, tokenizer, torch, copy_rows, copy_idx, temperature)
        total = (
            float(cfg["placement_loss_weight"]) * place_loss
            + float(cfg["tool_kl_loss_weight"]) * tool_kl
            + float(cfg["copy_kl_loss_weight"]) * copy_kl
        )
        if not bool(torch.isfinite(total).item()):
            raise ValueError("non-finite v0.0.51 ladder loss")
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(student.parameters(), float(cfg["gradient_clip"]),
                                              error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % interval == 0:
            event = {
                "step": step,
                "placement_loss": float(place_loss.detach()),
                "tool_kl_loss": float(tool_kl.detach()),
                "copy_kl_loss": float(copy_kl.detach()),
                "combined_loss": float(total.detach()),
                "gradient_norm": float(norm),
            }
            history.append(event)
            print(json.dumps({"event": "ladder_step", "learning_rate": learning_rate, **event}), flush=True)
        if step % interval == 0:
            student.eval()
            probe = objectives.placement_probe(student, tokenizer, torch, place_dev, template)
            trajectory.append({
                "step": step,
                "placement_exact_gain": probe["exact_top1"] - before_place["exact_top1"],
                "placement_token_top1_gain": probe["token_top1_rate"] - before_place["token_top1_rate"],
                "drift": relative_drift(student, reference_state, torch),
            })
            print(json.dumps({"event": "ladder_checkpoint", "learning_rate": learning_rate,
                              "step": step, **trajectory[-1]}), flush=True)
            student.train()
    del optimizer
    student.eval()

    after_place = objectives.placement_probe(student, tokenizer, torch, place_dev, template)
    tool_probe = distill.distill_probe(student, teacher, tokenizer, torch, tool_rows,
                                       int(cfg["tool_distill_batch_size"]), temperature)
    copy_probe = distill.distill_probe(student, teacher, tokenizer, torch, copy_rows,
                                       int(cfg["copy_distill_batch_size"]), temperature)
    point = cfg["operating_point"]
    exact_gain = after_place["exact_top1"] - before_place["exact_top1"]
    token_gain = after_place["token_top1_rate"] - before_place["token_top1_rate"]
    preserved = (
        tool_probe["teacher_kl"] <= float(point["maximum_tool_teacher_kl"])
        and copy_probe["teacher_kl"] <= float(point["maximum_copy_teacher_kl"])
    )
    learned = (
        exact_gain >= int(point["minimum_placement_exact_gain"])
        and token_gain >= float(point["minimum_placement_token_top1_gain"])
    )
    return {
        "learning_rate": learning_rate,
        "gradient_balance": balance,
        "history": history,
        "trajectory": trajectory,
        "placement": {k: v for k, v in after_place.items() if k != "rows"},
        "placement_exact_gain": exact_gain,
        "placement_token_top1_gain": token_gain,
        "tool_teacher_kl": tool_probe["teacher_kl"],
        "copy_teacher_kl": copy_probe["teacher_kl"],
        "tool_distill_token_top1": tool_probe["token_top1_rate"],
        "copy_distill_token_top1": copy_probe["token_top1_rate"],
        "drift": relative_drift(student, reference_state, torch),
        "preservation_held": preserved,
        "placement_moved": learned,
        "candidate": preserved and learned,
    }


def choose_rung(rungs: list[dict]) -> dict:
    """Prefer a rung that both learned and preserved; else the most informative one.

    With nothing qualifying, the rung reported is the largest update that still
    preserved, because that is the one whose familiar-90 evidence bounds how much
    update the protected recipe can absorb.
    """
    qualifying = [r for r in rungs if r["candidate"]]
    if qualifying:
        best = max(qualifying, key=lambda r: (r["placement_exact_gain"], r["placement_token_top1_gain"]))
        return {"index": rungs.index(best), "reason": "learned and preserved"}
    preserving = [r for r in rungs if r["preservation_held"]]
    if preserving:
        best = max(preserving, key=lambda r: r["learning_rate"])
        return {"index": rungs.index(best), "reason": "largest update that still preserved"}
    return {"index": 0, "reason": "no rung preserved; reporting the smallest update"}


def summary_markdown(report: dict) -> str:
    lines = [
        f"# Ember v0.0.51 placement dose-response ladder", "",
        "CPU diagnostic. No GPU submission, promotion, deployment, or production integration occurred.",
        "The familiar 90 cases and four historical controls are evaluation-only.", "",
        f"Baseline placement on the untouched source: **{report['before']['placement']['exact_top1']}"
        f"/{report['before']['placement']['cases']} exact**, "
        f"{report['before']['placement']['token_top1']}/{report['before']['placement']['tokens']} token top-1 "
        f"({report['before']['placement']['token_top1_rate']:.1%}).", "",
        "## The frontier", "",
        "| Rung | Learning rate | Exact gain | Token gain | Tool KL | Copy KL | Max drift | Placement grad share |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, rung in enumerate(report["rungs"]):
        share = rung["gradient_balance"]["placement_share"]
        lines.append(
            f"| {index} | {rung['learning_rate']:.1e} | {rung['placement_exact_gain']:+d} "
            f"| {rung['placement_token_top1_gain']:+.1%} | {rung['tool_teacher_kl']:.4f} "
            f"| {rung['copy_teacher_kl']:.6f} | {rung['drift']['max_relative_drift']:.2e} "
            f"| {'' if share is None else f'{share:.1%}'} |"
        )
    budget = report["operating_point"]["maximum_tool_teacher_kl"]
    lines += [
        "",
        f"Teacher KL budget: {budget}. Required gains: "
        f"{report['operating_point']['minimum_placement_exact_gain']} exact case and "
        f"{report['operating_point']['minimum_placement_token_top1_gain']:.0%} of tokens.",
        "",
        f"**Operating point found: {'YES' if report['operating_point_found'] else 'no'}** "
        f"({report['selected']['reason']}, rung {report['selected']['index']}).",
        "",
        "## Control rung", "",
        f"Rung 0 is v0.0.50's own learning rate. It reported an exact gain of "
        f"{report['rungs'][0]['placement_exact_gain']:+d} and a token gain of "
        f"{report['rungs'][0]['placement_token_top1_gain']:+.1%}, against v0.0.50's measured 0 and 0.0%.",
        "",
        "## Evaluation-only regression evidence on the selected rung", "",
    ]
    familiar = report.get("selected_evaluation", {}).get("familiar_90")
    if familiar:
        lines += [
            f"- Familiar canonical JSON: {familiar['envelope_json_valid']}/90",
            f"- Familiar correct tool: {familiar['correct_tool']}/90",
            f"- Familiar regression gate: {'PASS' if report['selected_evaluation']['familiar_gate']['passed'] else 'FAIL'}",
            f"- Reference controls: {report['selected_evaluation']['reference']['passed_cases']}/4",
            f"- Historical copy protection: {'PASS' if report['selected_evaluation']['copy_guard']['passed'] else 'FAIL'}",
        ]
    lines += [
        "",
        "A ladder where nothing moves is a complete result, not a failure. Nothing here is a "
        "promotion evaluation and nothing authorizes GPU training.", "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("v051-results"))
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
        "version": "0.0.51",
        "phase": cfg["phase"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ERROR",
        "code_commit": os.environ.get("GITHUB_SHA"),
        "cpu_learning_authorized": True,
        "gpu_training_authorized": False,
        "production_authorized": False,
        "promotion_authorized": False,
        "operating_point": cfg["operating_point"],
        "question": (
            "Does any placement update size move the objective while frozen-teacher KL "
            "preservation still holds?"
        ),
    }
    try:
        source_cfg = json.loads(base.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="ember-v051-") as td:
            student, tokenizer, source, _splits, source_ref = base.load_inputs(source_cfg, Path(td), torch)
            student.to("cpu").eval()
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != cfg["source_version"]:
                raise ValueError("v0.0.51 source is not the untouched v0.0.31 step 479")
            teacher = copy.deepcopy(student).to("cpu").eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            pristine = {name: tensor.detach().clone() for name, tensor in teacher.named_parameters()}
            pristine_state = copy.deepcopy(teacher.state_dict())

            values = v050d.target_values(cfg)
            template, template_report = v048d.discover_template(student, tokenizer, torch, cfg, values["template"])
            place_train = v048d.build_cases(
                {k: values["train"][k] for k in sorted(v050d.PLACEMENT_SUBTYPES)}, "v051_place_train")
            place_dev = v048d.build_cases(
                {k: values["development"][k] for k in sorted(v050d.PLACEMENT_SUBTYPES)}, "v051_place_dev")
            tool_rows, tool_report = v050d.prepare_tool_rows(teacher, tokenizer, torch, cfg)
            copy_rows, copy_report = v050d.prepare_copy_rows(teacher, tokenizer, torch, cfg)

            tool_id, _eos = objectives.token_contract(tokenizer)
            place_examples = [
                objectives.supervised_example(tokenizer, case, "placement", template, tool_id)
                for case in place_train
            ]

            before_place = objectives.placement_probe(student, tokenizer, torch, place_dev, template)
            before_familiar = regression.familiar_90(student, tokenizer, torch, cfg)
            before_reference = regression.references(student, tokenizer, torch, cfg)
            before_copy = base.copy_diagnostic(student, tokenizer, torch)
            if before_familiar["envelope_json_valid"] != 84 or before_familiar["correct_tool"] != 84:
                raise ValueError("source no longer reproduces the familiar 84/90 baseline")
            if not before_reference["passed"]:
                raise ValueError("source no longer reproduces 4/4 reference controls")
            print(json.dumps({"event": "baseline", "placement_exact": before_place["exact_top1"],
                              "placement_token_top1_rate": before_place["token_top1_rate"],
                              "familiar_correct_tool": before_familiar["correct_tool"]}), flush=True)

            rungs, best_state, best_index = [], None, None
            for index, learning_rate in enumerate(cfg["ladder_learning_rates"]):
                student.load_state_dict(copy.deepcopy(pristine_state))
                student.to("cpu").eval()
                rung = run_rung(student, teacher, tokenizer, torch, cfg, float(learning_rate),
                                place_examples, tool_rows, copy_rows, place_dev, template,
                                before_place, pristine)
                rungs.append(rung)
                print(json.dumps({"event": "rung_complete", "index": index, **{
                    k: rung[k] for k in ("learning_rate", "placement_exact_gain",
                                         "placement_token_top1_gain", "tool_teacher_kl",
                                         "copy_teacher_kl", "preservation_held", "candidate")}}), flush=True)

            selected = choose_rung(rungs)
            best_index = selected["index"]
            # Re-run the selected rung so its state is the one evaluated. The loop is
            # deterministic, so this reproduces exactly the rung reported above.
            student.load_state_dict(copy.deepcopy(pristine_state))
            student.to("cpu").eval()
            run_rung(student, teacher, tokenizer, torch, cfg,
                     float(cfg["ladder_learning_rates"][best_index]), place_examples,
                     tool_rows, copy_rows, place_dev, template, before_place, pristine)

            after_familiar = regression.familiar_90(student, tokenizer, torch, cfg)
            after_reference = regression.references(student, tokenizer, torch, cfg)
            after_copy = base.copy_diagnostic(student, tokenizer, torch)
            copy_guard = base.copy_protection(before_copy, after_copy)
            familiar_gate = regression.floor_checks(after_familiar, {**cfg, "gate": {
                "minimum_familiar_envelope_json": cfg["operating_point"]["minimum_familiar_envelope_json"],
                "minimum_familiar_correct_tool": cfg["operating_point"]["minimum_familiar_correct_tool"],
            }})
            chosen = rungs[best_index]
            operating_point_found = bool(
                chosen["candidate"] and familiar_gate["passed"]
                and after_reference["passed"] and copy_guard["passed"]
            )

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
                distillation={"tool": tool_report, "copy": copy_report},
                before={
                    "placement": {k: v for k, v in before_place.items() if k != "rows"},
                    "familiar_90": before_familiar,
                    "reference": before_reference,
                },
                rungs=rungs,
                selected=selected,
                selected_evaluation={
                    "learning_rate": float(cfg["ladder_learning_rates"][best_index]),
                    "familiar_90": after_familiar,
                    "familiar_gate": familiar_gate,
                    "reference": after_reference,
                    "copy_guard": copy_guard,
                },
                operating_point_found=operating_point_found,
                meaning="Dose-response diagnostic; no promotion or GPU authorization.",
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
        "operating_point_found": report["operating_point_found"],
        "selected": report["selected"],
        "rungs": [
            {k: r[k] for k in ("learning_rate", "placement_exact_gain",
                               "placement_token_top1_gain", "tool_teacher_kl", "preservation_held")}
            for r in report["rungs"]
        ],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
