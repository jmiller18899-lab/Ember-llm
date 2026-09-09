"""Replay corpora and configuration for the Ember v0.0.49 protected-learning canary.

v0.0.48 proved entry and placement are both learnable and that a plain 50/50
two-objective update destroys everything else. The replay corpus built here is
the protection term: broad synthetic envelope and copy behaviour, captured from
the *source* model's own correct output, so the update has something to hold onto
across every kind that collapsed.

Every replay value is synthetic and disjoint from the familiar 90-case battery
and from every prior calibration and canary value. The familiar 90 and the four
historical controls stay evaluation-only.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import ember_v048_data as prior

base = prior.base
control = prior.control
copy_data = prior.copy_data
v043 = prior.v043
v044 = prior.v044
v045 = prior.v045
v047 = prior.v047

DEFAULT_CONFIG = ROOT / "config/ember_protected_learning_v0.0.49.json"
ENTRY_SUBTYPES = set(prior.ENTRY_SUBTYPES)
PLACEMENT_SUBTYPES = set(prior.PLACEMENT_SUBTYPES)
ALL_SYNTHETIC_SUBTYPES = ENTRY_SUBTYPES | PLACEMENT_SUBTYPES
SUBTYPE_VARIANT = dict(prior.SUBTYPE_VARIANT)
REPLAY_KINDS = tuple(copy_data.KINDS)
EOT = base.semantic_gate.EOT


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if path.resolve() != DEFAULT_CONFIG.resolve() or cfg.get("version") != "0.0.49":
        raise ValueError("unsupported v0.0.49 configuration")
    if cfg.get("cpu_learning_authorized") is not True:
        raise ValueError("CPU canary authorization missing")
    if any(cfg.get(k) is not False for k in (
        "gpu_training_authorized", "production_authorized", "promotion_authorized"
    )):
        raise ValueError("v0.0.49 cannot authorize GPU, production, or promotion")
    if set(cfg.get("entry_subtypes", [])) != ENTRY_SUBTYPES or set(cfg.get("placement_subtypes", [])) != PLACEMENT_SUBTYPES:
        raise ValueError("v0.0.49 keeps the v0.0.48 objective subtype sets")
    if int(cfg.get("optimizer_steps", 0)) not in range(1, 301):
        raise ValueError("invalid CPU canary step count")

    # v0.0.48 ran 120 steps at 1.2e-6 and destroyed the familiar battery. v0.0.49
    # is the smaller-update arm, so the bound is tightened rather than inherited.
    learning_rate = float(cfg.get("learning_rate", 0))
    if not 0 < learning_rate <= 5e-7:
        raise ValueError("v0.0.49 requires a substantially smaller update than v0.0.48")
    if learning_rate >= float(prior.load_config(prior.DEFAULT_CONFIG)["learning_rate"]):
        raise ValueError("v0.0.49 learning rate must be below the v0.0.48 rate")

    drift = float(cfg.get("trust_region_relative_drift", 0))
    if not 0 < drift <= 1e-3:
        raise ValueError("v0.0.49 requires an explicit trust region")

    weights = {k: float(cfg[k + "_loss_weight"]) for k in ("entry", "placement", "replay")}
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("objective weights must sum to one")
    if weights["replay"] < 0.5:
        raise ValueError("protection must dominate: replay weight must be at least 0.5")
    if weights["replay"] <= weights["entry"] + weights["placement"]:
        raise ValueError("replay weight must exceed the combined learning weight")

    if int(cfg.get("replay_minimum_kinds_covered", 0)) < 8:
        raise ValueError("replay must cover nearly every kind that collapsed in v0.0.48")
    for key in ("replay_minimum_envelope_rows", "replay_minimum_copy_rows"):
        if int(cfg.get(key, 0)) < 1:
            raise ValueError(f"invalid {key}")
    steps = [int(s) for s in cfg.get("trajectory_probe_steps", [])]
    if any(not 0 < s < int(cfg["optimizer_steps"]) for s in steps) or sorted(set(steps)) != steps:
        raise ValueError("trajectory probe steps must be increasing and inside the run")

    if cfg.get("historical_kind_floor") != v043.EXPECTED_KIND_FLOOR or cfg.get("historical_subtype_floor") != v043.EXPECTED_SUBTYPE_FLOOR:
        raise ValueError("historical regression floors changed")
    return cfg


def historical_used_values() -> set[str]:
    """Everything any earlier phase has already touched, plus the v0.0.48 values."""
    used = prior.historical_used_values()
    cfg48 = prior.load_config(prior.DEFAULT_CONFIG)
    for groups in prior.synthetic_values(cfg48).values():
        for values in groups.values():
            used.update(values)
    return used


def synthetic_values(cfg: dict) -> dict[str, dict[str, list[str]]]:
    """Objective values, by subtype, disjoint from the battery and from v0.0.48."""
    used = historical_used_values()
    counts = {
        "template": int(cfg["template_values_per_subtype"]),
        "train": int(cfg["train_values_per_subtype"]),
        "development": int(cfg["development_values_per_subtype"]),
    }
    out: dict[str, dict[str, list[str]]] = {phase: {} for phase in counts}
    for phase, count in counts.items():
        for subtype in sorted(ALL_SYNTHETIC_SUBTYPES):
            kind, variant = SUBTYPE_VARIANT[subtype]
            values: list[str] = []
            i = 0
            while len(values) < count:
                seed = copy_data._digest("v049", phase, subtype, i)
                value = copy_data._render(kind, variant, seed)
                i += 1
                if value in used or v044.subtype_for(kind, value) != subtype:
                    continue
                used.add(value)
                values.append(value)
            out[phase][subtype] = values
    _assert_disjoint(out)
    return out


def replay_values(cfg: dict, used: set[str] | None = None) -> dict[str, dict[str, list[str]]]:
    """Fresh values across every kind, spread over that kind's templates.

    v0.0.48's synthetic values only covered five code-shaped subtypes, but the
    collapse hit mixed, url, path, model_id, digits and entity too. Replay has to
    reach the behaviour it is protecting, so it is generated per kind here.
    """
    seen = historical_used_values() if used is None else used
    counts = {
        "envelope": int(cfg["replay_envelope_values_per_kind"]),
        "copy": int(cfg["replay_copy_values_per_kind"]),
    }
    out: dict[str, dict[str, list[str]]] = {phase: {} for phase in counts}
    for phase, count in counts.items():
        for kind in REPLAY_KINDS:
            values: list[str] = []
            i = 0
            while len(values) < count:
                variant = len(values) % int(copy_data.VARIANTS[kind])
                seed = copy_data._digest("v049-replay", phase, kind, i)
                value = copy_data._render(kind, variant, seed)
                i += 1
                if value in seen:
                    continue
                seen.add(value)
                values.append(value)
            out[phase][kind] = values
    _assert_disjoint(out)
    return out


def interference_values(cfg: dict, used: set[str]) -> dict[str, list[str]]:
    """A small per-kind cohort used only to trace interference during the run."""
    count = int(cfg["interference_probe_values_per_kind"])
    out: dict[str, list[str]] = {}
    for kind in REPLAY_KINDS:
        values: list[str] = []
        i = 0
        while len(values) < count:
            variant = len(values) % int(copy_data.VARIANTS[kind])
            seed = copy_data._digest("v049-interference", kind, i)
            value = copy_data._render(kind, variant, seed)
            i += 1
            if value in used:
                continue
            used.add(value)
            values.append(value)
        out[kind] = values
    return out


def _assert_disjoint(groups: dict[str, dict[str, list[str]]]) -> None:
    flat = [v for phase in groups.values() for values in phase.values() for v in values]
    if len(flat) != len(set(flat)):
        raise ValueError("v0.0.49 synthetic values repeat")
    if set(flat) & set(copy_data.HELD_OUT_VALUES):
        raise ValueError("v0.0.49 synthetic values touch the familiar battery")


def prompt_for(subtype: str, value: str) -> dict:
    return prior.prompt_for(subtype, value)


def build_cases(values: dict[str, list[str]], prefix: str) -> list[dict]:
    return prior.build_cases(values, prefix)


def envelope_prompt_for_kind(kind: str, value: str) -> dict:
    """The frozen v0.0.44 envelope prompt, for any kind rather than only subtypes."""
    user_id, user = v044.frozen_user(kind, value)
    tool, field = v044.TOOL_BY_KIND[kind]
    return {
        "kind": kind,
        "subtype": v044.subtype_for(kind, value),
        "target": value,
        "expected_tool": tool,
        "argument_key": field,
        "user_variant": user_id,
        "prompt": v044._prompt(user, v044.baseline_system(kind)),
    }


def copy_prompt_for_kind(kind: str, value: str) -> dict:
    """The v0.0.26 bare-value copy prompt, the task the copy guard measures."""
    return {
        "kind": kind,
        "target": value,
        "prompt": copy_data.prompt_for(value),
    }


def capture_envelope_replay(model, tokenizer, torch, cfg: dict, values: dict[str, list[str]]) -> tuple[list[dict], dict]:
    """Replay targets are the source model's own correct envelopes.

    Supervising a *correct* answer the source cannot already produce would be a
    third learning objective wearing protection's clothes. Filtering to what the
    source already gets right keeps this term purely preservative, which is what
    the familiar-90 and copy gates actually measure.
    """
    rows, attempted = [], 0
    for kind in REPLAY_KINDS:
        for index, value in enumerate(values[kind]):
            case = envelope_prompt_for_kind(kind, value)
            attempted += 1
            generation = base.semantic_gate.generate_completion(
                model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"])
            )
            score = control.score_case(case, generation["completion"])
            if not (score["envelope_json_valid"] and score["tool_name_correct"]):
                continue
            rows.append({
                "id": f"replay_envelope_{kind}_{index:02d}",
                **case,
                "completion": generation["completion"],
            })
    return rows, _replay_summary("envelope", rows, attempted)


def capture_copy_replay(model, tokenizer, torch, cfg: dict, values: dict[str, list[str]]) -> tuple[list[dict], dict]:
    """Replay targets are the source model's own exact, cleanly stopped copies."""
    rows, attempted = [], 0
    for kind in REPLAY_KINDS:
        for index, value in enumerate(values[kind]):
            case = copy_prompt_for_kind(kind, value)
            attempted += 1
            generation = base.semantic_gate.generate_completion(
                model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"])
            )
            completion = generation["completion"]
            if EOT not in completion or completion.split(EOT)[0].strip() != value:
                continue
            rows.append({
                "id": f"replay_copy_{kind}_{index:02d}",
                **case,
                "completion": completion,
            })
    return rows, _replay_summary("copy", rows, attempted)


def _replay_summary(name: str, rows: list[dict], attempted: int) -> dict:
    kinds = sorted({row["kind"] for row in rows})
    return {
        "corpus": name,
        "attempted": attempted,
        "captured": len(rows),
        "capture_rate": len(rows) / attempted if attempted else None,
        "kinds_covered": len(kinds),
        "kinds": kinds,
    }


def enforce_replay_floor(cfg: dict, envelope_summary: dict, copy_summary: dict) -> None:
    """Measured first, enforced second, so the numbers survive a failure."""
    if envelope_summary["captured"] < int(cfg["replay_minimum_envelope_rows"]):
        raise ValueError(
            f"envelope replay corpus too small: {envelope_summary['captured']}"
        )
    if copy_summary["captured"] < int(cfg["replay_minimum_copy_rows"]):
        raise ValueError(f"copy replay corpus too small: {copy_summary['captured']}")
    floor = int(cfg["replay_minimum_kinds_covered"])
    for summary in (envelope_summary, copy_summary):
        if summary["kinds_covered"] < floor:
            raise ValueError(
                f"{summary['corpus']} replay covers only {summary['kinds_covered']} kinds"
            )


def discover_template(model, tokenizer, torch, cfg: dict, values: dict[str, list[str]]):
    return prior.discover_template(model, tokenizer, torch, cfg, values)


def value_continuation_ids(tokenizer, template: dict, target: str) -> list[int]:
    return prior.value_continuation_ids(tokenizer, template, target)
