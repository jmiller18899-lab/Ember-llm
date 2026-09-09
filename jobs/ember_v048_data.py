"""Synthetic data and value-free template helpers for Ember v0.0.48."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import ember_envelope_template_v047_compat as observed
from jobs import ember_envelope_preflight_v044 as v044

v047 = observed.v047
v045 = v047.v045
base = v045.base
control = v045.control
copy_data = v044.copy_data
v043 = v044.prior
DEFAULT_CONFIG = ROOT / "config/ember_two_objective_v0.0.48.json"
ENTRY_SUBTYPES = {"short_code/len4", "short_code/len5", "long_code/4x4", "long_code/3x5"}
PLACEMENT_SUBTYPES = {"short_code/len4", "short_code/len5", "long_code/3x5", "path/plain_leaf"}
ALL_SYNTHETIC_SUBTYPES = ENTRY_SUBTYPES | PLACEMENT_SUBTYPES
SUBTYPE_VARIANT = dict(v043.SUBTYPE_VARIANT)


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if path.resolve() != DEFAULT_CONFIG.resolve() or cfg.get("version") != "0.0.48":
        raise ValueError("unsupported v0.0.48 configuration")
    if cfg.get("cpu_learning_authorized") is not True:
        raise ValueError("CPU canary authorization missing")
    if any(cfg.get(k) is not False for k in ("gpu_training_authorized", "production_authorized", "promotion_authorized")):
        raise ValueError("v0.0.48 cannot authorize GPU, production, or promotion")
    if set(cfg.get("entry_subtypes", [])) != ENTRY_SUBTYPES or set(cfg.get("placement_subtypes", [])) != PLACEMENT_SUBTYPES:
        raise ValueError("v0.0.48 objective subtype sets changed")
    if int(cfg.get("optimizer_steps", 0)) not in range(1, 301):
        raise ValueError("invalid CPU canary step count")
    if not 0 < float(cfg.get("learning_rate", 0)) <= 5e-6:
        raise ValueError("learning rate exceeds canary bound")
    if abs(float(cfg["entry_loss_weight"]) + float(cfg["placement_loss_weight"]) - 1.0) > 1e-9:
        raise ValueError("objective weights must sum to one")
    if cfg.get("historical_kind_floor") != v043.EXPECTED_KIND_FLOOR or cfg.get("historical_subtype_floor") != v043.EXPECTED_SUBTYPE_FLOOR:
        raise ValueError("historical regression floors changed")
    return cfg


def historical_used_values() -> set[str]:
    used = set(copy_data.HELD_OUT_VALUES)
    cfg43 = v043.load_config(v043.DEFAULT_CONFIG)
    for folds in v043.calibration_values(cfg43).values():
        for values in folds.values():
            used.update(values)
    cfg44 = v044.load_config(v044.DEFAULT_CONFIG)
    for folds in v044.calibration_values(cfg44).values():
        for values in folds.values():
            used.update(values)
    return used


def synthetic_values(cfg: dict) -> dict[str, dict[str, list[str]]]:
    used = historical_used_values()
    counts = {
        "template": int(cfg["template_values_per_subtype"]),
        "train": int(cfg["train_values_per_subtype"]),
        "development": int(cfg["development_values_per_subtype"]),
    }
    out = {phase: {} for phase in counts}
    for phase, count in counts.items():
        for subtype in sorted(ALL_SYNTHETIC_SUBTYPES):
            kind, variant = SUBTYPE_VARIANT[subtype]
            values = []
            i = 0
            while len(values) < count:
                seed = copy_data._digest("v048", phase, subtype, i)
                value = copy_data._render(kind, variant, seed)
                i += 1
                if value in used or v044.subtype_for(kind, value) != subtype:
                    continue
                used.add(value)
                values.append(value)
            out[phase][subtype] = values
    flat = [v for groups in out.values() for values in groups.values() for v in values]
    if len(flat) != len(set(flat)) or set(flat) & set(copy_data.HELD_OUT_VALUES):
        raise ValueError("v0.0.48 synthetic values are not disjoint")
    return out


def prompt_for(subtype: str, value: str) -> dict:
    kind, _variant = SUBTYPE_VARIANT[subtype]
    _uid, user = v044.frozen_user(kind, value)
    prompt = v044._prompt(user, v044.baseline_system(kind))
    tool, field = v044.TOOL_BY_KIND[kind]
    if (tool, field) != ("web_search", "query"):
        raise ValueError("unexpected canary tool mapping")
    return {"kind": kind, "subtype": subtype, "target": value, "expected_tool": tool, "argument_key": field, "prompt": prompt}


def build_cases(values: dict[str, list[str]], prefix: str) -> list[dict]:
    rows = []
    for subtype, items in sorted(values.items()):
        for i, value in enumerate(items):
            rows.append({"id": f"{prefix}_{subtype.replace('/', '_')}_{i:02d}", **prompt_for(subtype, value)})
    return rows


def discover_template(model, tokenizer, torch, cfg: dict, values: dict[str, list[str]]) -> tuple[dict, dict]:
    rows = []
    for case in build_cases(values, "template"):
        generation = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"]))
        score = control.score_case(case, generation["completion"])
        rows.append({**case, "baseline": generation, "baseline_score": score})
    successes = [r for r in rows if r["baseline_score"]["envelope_json_valid"] and r["baseline_score"]["tool_name_correct"]]
    if len(successes) < int(cfg["template_minimum_successes"]):
        raise ValueError(f"too few successful synthetic template sources: {len(successes)}")
    templates = v047.build_template_library(successes, tokenizer)
    boundary = [t for t in templates if t.get("reaches_value_boundary") is True]
    if not boundary:
        raise ValueError("no value-free boundary template found")
    chosen = max(boundary, key=lambda t: (len(t["source_ids"]), len(t["prefix_ids"])))
    dominance = len(chosen["source_ids"]) / len(successes)
    if dominance < float(cfg["template_dominance_minimum"]):
        raise ValueError(f"template dominance too low: {dominance:.1%}")
    return chosen, {
        "cases": len(rows), "successful_sources": len(successes), "unique_templates": len(templates),
        "boundary_templates": len(boundary), "chosen_template_id": chosen["template_id"],
        "chosen_source_count": len(chosen["source_ids"]), "dominance": dominance,
        "prefix_token_count": len(chosen["prefix_ids"]), "prefix_text": chosen["prefix_text"],
    }


def value_continuation_ids(tokenizer, template: dict, target: str) -> list[int]:
    expected = template["prefix_text"] + target
    candidates = [tokenizer.encode(expected), tokenizer.encode(target), tokenizer.encode('"' + target)]
    for ids in candidates:
        for start in range(len(ids)):
            suffix = [int(x) for x in ids[start:]]
            if suffix and tokenizer.decode(suffix) == target and tokenizer.decode(list(template["prefix_ids"]) + suffix) == expected:
                return suffix
    raise ValueError(f"cannot form exact continuation tokens for {target!r}")
