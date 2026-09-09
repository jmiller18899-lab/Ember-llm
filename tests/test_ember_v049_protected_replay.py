from __future__ import annotations

import json
from pathlib import Path

from jobs import ember_v049_replay as v049

ROOT = Path(__file__).resolve().parents[1]


def test_v049_restarts_from_source_and_is_cpu_only():
    cfg = v049.load_config()
    assert cfg["version"] == "0.0.49"
    assert cfg["source_version"] == "0.0.31"
    assert cfg["cpu_learning_authorized"] is True
    assert cfg["gpu_training_authorized"] is False
    assert cfg["production_authorized"] is False
    assert cfg["promotion_authorized"] is False


def test_v049_is_substantially_smaller_than_v048_update():
    cfg = v049.load_config()
    old = json.loads(v049.v048d.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert cfg["learning_rate"] <= old["learning_rate"] / 4
    assert cfg["optimizer_steps"] < old["optimizer_steps"]
    assert cfg["gradient_clip"] < old["gradient_clip"]
    assert cfg["entry_loss_weight"] < old["entry_loss_weight"]
    assert cfg["placement_loss_weight"] < old["placement_loss_weight"]


def test_v049_replay_dominates_loss_and_has_hard_retention_gates():
    cfg = v049.load_config()
    replay_weight = cfg["tool_replay_loss_weight"] + cfg["copy_replay_loss_weight"]
    target_weight = cfg["entry_loss_weight"] + cfg["placement_loss_weight"]
    assert replay_weight > target_weight
    assert cfg["gate"]["minimum_tool_replay_token_top1"] >= 0.95
    assert cfg["gate"]["minimum_copy_replay_token_top1"] >= 0.95


def test_target_and_replay_values_do_not_touch_familiar_90():
    cfg = v049.load_config()
    target = v049.target_values(cfg)
    target_values = {v for phase in target.values() for values in phase.values() for v in values}
    tool_values = {r["target"] for r in v049.replay_value_rows(cfg, "tool")}
    copy_values = {r["target"] for r in v049.replay_value_rows(cfg, "copy")}
    held = set(v049.copy_data.HELD_OUT_VALUES)
    assert not (target_values & held)
    assert not (tool_values & held)
    assert not (copy_values & held)
    variants = sum(v049.copy_data.VARIANTS.values())
    assert len(tool_values) == variants * cfg["tool_replay_values_per_variant"]
    assert len(copy_values) == variants * cfg["copy_replay_values_per_variant"]
    assert len(tool_values) > len(copy_values)


def test_replay_spans_every_kind_and_variant():
    cfg = v049.load_config()
    for family in ("tool", "copy"):
        rows = v049.replay_value_rows(cfg, family)
        pairs = {(r["kind"], r["variant"]) for r in rows}
        expected = {(kind, variant) for kind in v049.copy_data.KINDS for variant in range(v049.copy_data.VARIANTS[kind])}
        assert pairs == expected


def test_v049_source_uses_family_specific_replay_counts_directly():
    cfg = v049.load_config()
    variants = sum(v049.copy_data.VARIANTS.values())
    assert len(v049.replay_value_rows(cfg, "tool")) == variants * cfg["tool_replay_values_per_variant"]
    assert len(v049.replay_value_rows(cfg, "copy")) == variants * cfg["copy_replay_values_per_variant"]


def test_canary_has_no_gpu_or_checkpoint_write_path():
    text = "\n".join([
        (ROOT / "jobs/ember_v049_replay.py").read_text(encoding="utf-8"),
        (ROOT / "jobs/ember_v049_replay_compat.py").read_text(encoding="utf-8"),
        (ROOT / "jobs/ember_v049_canary.py").read_text(encoding="utf-8"),
    ])
    forbidden = [".cuda(", 'device="cuda"', 'to("cuda")', "save_checkpoint(", "push_to_hub("]
    assert not any(token in text for token in forbidden)


def test_familiar_battery_is_evaluation_only_in_canary():
    replay_source = (ROOT / "jobs/ember_v049_replay.py").read_text(encoding="utf-8")
    compat_source = (ROOT / "jobs/ember_v049_replay_compat.py").read_text(encoding="utf-8")
    canary_source = (ROOT / "jobs/ember_v049_canary.py").read_text(encoding="utf-8")
    assert "familiar_90" not in replay_source
    assert "familiar_90" not in compat_source
    assert canary_source.count("regression.familiar_90") == 2
    assert "copy_data.HELD_OUT_VALUES" in replay_source
