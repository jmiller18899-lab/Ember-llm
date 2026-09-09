"""Fresh disjoint target and distillation data for Ember v0.0.50."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from jobs import ember_v049_replay as v049
from jobs import ember_v048_data as v048d

base = v049.base
copy_data = v049.copy_data
v044 = v049.v044
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config/ember_teacher_kl_v0.0.50.json"
ENTRY_SUBTYPES = set(v049.ENTRY_SUBTYPES)
PLACEMENT_SUBTYPES = set(v049.PLACEMENT_SUBTYPES)
TARGET_SUBTYPES = ENTRY_SUBTYPES | PLACEMENT_SUBTYPES


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if path.resolve() != DEFAULT_CONFIG.resolve() or cfg.get("version") != "0.0.50":
        raise ValueError("unsupported v0.0.50 configuration")
    if cfg.get("cpu_learning_authorized") is not True:
        raise ValueError("v0.0.50 CPU learning authorization missing")
    if any(cfg.get(k) is not False for k in ("gpu_training_authorized", "production_authorized", "promotion_authorized")):
        raise ValueError("v0.0.50 cannot authorize GPU, production, or promotion")
    if set(cfg.get("entry_subtypes", [])) != ENTRY_SUBTYPES or set(cfg.get("placement_subtypes", [])) != PLACEMENT_SUBTYPES:
        raise ValueError("v0.0.50 target subtype sets changed")
    weights = [float(cfg[k]) for k in ("entry_loss_weight", "placement_loss_weight", "tool_kl_loss_weight", "copy_kl_loss_weight")]
    if any(w <= 0 for w in weights) or abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError("v0.0.50 weights must be positive and sum to one")
    if float(cfg["tool_kl_loss_weight"]) + float(cfg["copy_kl_loss_weight"]) < 0.75:
        raise ValueError("v0.0.50 requires at least 75% teacher-distribution preservation weight")
    if not 0 < float(cfg["learning_rate"]) <= 1e-7:
        raise ValueError("v0.0.50 learning rate exceeds minimal-delta bound")
    if not 1 <= int(cfg["max_optimizer_steps"]) <= 40:
        raise ValueError("v0.0.50 max steps exceed minimal-delta bound")
    if int(cfg["checkpoint_interval"]) <= 0 or int(cfg["max_optimizer_steps"]) % int(cfg["checkpoint_interval"]) != 0:
        raise ValueError("invalid v0.0.50 checkpoint interval")
    if cfg.get("historical_kind_floor") != v048d.v043.EXPECTED_KIND_FLOOR:
        raise ValueError("historical kind floors changed")
    if cfg.get("historical_subtype_floor") != v048d.v043.EXPECTED_SUBTYPE_FLOOR:
        raise ValueError("historical subtype floors changed")
    return cfg


def historical_used_values() -> set[str]:
    used = set(v049._used_values())
    old_cfg = v049.load_config()
    old_targets = v049.target_values(old_cfg)
    used.update(v for phase in old_targets.values() for values in phase.values() for v in values)
    for family in ("tool", "copy"):
        used.update(row["target"] for row in v049.replay_value_rows(old_cfg, family))
    return used


def target_values(cfg: dict) -> dict[str, dict[str, list[str]]]:
    used = historical_used_values()
    counts = {
        "template": int(cfg["template_values_per_subtype"]),
        "train": int(cfg["train_values_per_subtype"]),
        "development": int(cfg["development_values_per_subtype"]),
    }
    out = {phase: {} for phase in counts}
    for phase, count in counts.items():
        for subtype in sorted(TARGET_SUBTYPES):
            kind, variant = v048d.SUBTYPE_VARIANT[subtype]
            values = []
            i = 0
            while len(values) < count:
                seed = copy_data._digest("v050-target", phase, subtype, i)
                value = copy_data._render(kind, variant, seed)
                i += 1
                if value in used or v044.subtype_for(kind, value) != subtype:
                    continue
                used.add(value)
                values.append(value)
            out[phase][subtype] = values
    flat = [v for phase in out.values() for values in phase.values() for v in values]
    if len(flat) != len(set(flat)) or set(flat) & set(copy_data.HELD_OUT_VALUES):
        raise ValueError("v0.0.50 target values overlap protected values")
    return out


def distill_value_rows(cfg: dict, family: str) -> list[dict]:
    if family not in {"tool", "copy"}:
        raise ValueError(family)
    used = historical_used_values()
    current = target_values(cfg)
    used.update(v for phase in current.values() for values in phase.values() for v in values)
    count = int(cfg[f"{family}_distill_values_per_variant"])
    rows = []
    for kind in copy_data.KINDS:
        for variant in range(int(copy_data.VARIANTS[kind])):
            made = 0
            i = 0
            while made < count:
                seed = copy_data._digest("v050-distill", family, kind, variant, i)
                value = copy_data._render(kind, variant, seed)
                i += 1
                if value in used:
                    continue
                used.add(value)
                rows.append({"kind": kind, "variant": variant, "target": value})
                made += 1
    values = {row["target"] for row in rows}
    if values & set(copy_data.HELD_OUT_VALUES):
        raise ValueError("v0.0.50 distillation values overlap familiar held-out battery")
    return rows


def tool_prompt(kind: str, value: str) -> dict:
    variant, user = v044.frozen_user(kind, value)
    prompt = v044._prompt(user, v044.baseline_system(kind))
    tool, field = v044.TOOL_BY_KIND[kind]
    return {"kind": kind, "target": value, "variant_name": variant, "expected_tool": tool, "argument_key": field, "prompt": prompt}


def prepare_tool_rows(teacher, tokenizer, torch, cfg: dict) -> tuple[list[dict], dict]:
    kept = []
    attempted = Counter()
    for index, row in enumerate(distill_value_rows(cfg, "tool")):
        case = {"id": f"v050_tool_distill_{row['kind']}_{index:03d}", **tool_prompt(row["kind"], row["target"])}
        attempted[row["kind"]] += 1
        generation = base.semantic_gate.generate_completion(teacher, tokenizer, torch, case["prompt"], int(cfg["generation_budget"]))
        score = v048d.control.score_case(case, generation["completion"])
        if score["envelope_json_valid"] and score["tool_name_correct"] and generation["generated_ids"]:
            kept.append({**case, "source_ids": [int(x) for x in generation["generated_ids"]]})
    by_kind = Counter(row["kind"] for row in kept)
    if len(kept) < int(cfg["minimum_tool_distill_examples"]):
        raise ValueError(f"too few tool distillation examples: {len(kept)}")
    for kind in copy_data.KINDS:
        if by_kind[kind] < int(cfg["minimum_tool_distill_per_kind"]):
            raise ValueError(f"too few tool distillation examples for {kind}: {by_kind[kind]}")
    return kept, {"attempted": dict(attempted), "kept": len(kept), "kept_by_kind": dict(by_kind)}


def prepare_copy_rows(teacher, tokenizer, torch, cfg: dict) -> tuple[list[dict], dict]:
    kept = []
    for index, row in enumerate(distill_value_rows(cfg, "copy")):
        value = row["target"]
        prompt = copy_data.prompt(value, copy_data._corrupt(value, 3), copy_data._corrupt(value, 11))
        generation = base.semantic_gate.generate_completion(teacher, tokenizer, torch, prompt, int(cfg["generation_budget"]))
        ids = [int(x) for x in generation["generated_ids"]]
        if ids and not generation["completion"].lstrip().startswith(v048d.v045.TOOL):
            kept.append({"id": f"v050_copy_distill_{row['kind']}_{index:03d}", "kind": row["kind"], "prompt": prompt, "source_ids": ids})
    if len(kept) < int(cfg["minimum_copy_distill_examples"]):
        raise ValueError(f"too few copy distillation examples: {len(kept)}")
    return kept, {"kept": len(kept), "kept_by_kind": dict(Counter(row["kind"] for row in kept))}
