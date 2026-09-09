from __future__ import annotations

import inspect
from pathlib import Path

from jobs import ember_v051_data as data
from jobs import ember_v051_canary as canary

ROOT = Path(__file__).resolve().parents[1]


def test_v051_is_placement_focused_cpu_only_teacher_protected():
    cfg = data.load_config()
    assert cfg["version"] == "0.0.51"
    assert cfg["cpu_learning_authorized"] is True
    assert cfg["gpu_training_authorized"] is False
    assert cfg["promotion_authorized"] is False
    assert cfg["production_authorized"] is False
    assert cfg["placement_loss_weight"] > cfg["entry_loss_weight"]
    assert cfg["tool_kl_loss_weight"] + cfg["copy_kl_loss_weight"] >= 0.75
    assert cfg["learning_rate"] <= 1.5e-7
    assert cfg["max_optimizer_steps"] <= 60


def test_v051_targets_are_fresh_disjoint_from_familiar_and_history():
    cfg = data.load_config()
    values = data.target_values(cfg)
    flat = [v for phase in values.values() for rows in phase.values() for v in rows]
    assert len(flat) == len(set(flat))
    assert not (set(flat) & set(data.copy_data.HELD_OUT_VALUES))
    assert not (set(flat) & data.historical_used_values())


def test_v051_distillation_values_are_fresh_and_disjoint():
    cfg = data.load_config()
    targets = data.target_values(cfg)
    target_set = {v for phase in targets.values() for rows in phase.values() for v in rows}
    tool = {row["target"] for row in data.distill_value_rows(cfg, "tool")}
    copy = {row["target"] for row in data.distill_value_rows(cfg, "copy")}
    assert not (tool & target_set)
    assert not (copy & target_set)
    assert not (tool & set(data.copy_data.HELD_OUT_VALUES))
    assert not (copy & set(data.copy_data.HELD_OUT_VALUES))
    assert not (tool & data.historical_used_values())
    assert not (copy & data.historical_used_values())


def test_v051_selection_loop_remains_synthetic_only():
    source = inspect.getsource(canary.main)
    loop_start = source.index("for step in range")
    final_eval = source.index("after_copy =", loop_start)
    loop = source[loop_start:final_eval]
    assert "familiar_90" not in loop
    assert "references(" not in loop
    assert "copy_diagnostic" not in loop


def test_v051_has_no_gpu_or_checkpoint_promotion_path():
    paths = [ROOT / "jobs/ember_v051_data.py", ROOT / "jobs/ember_v051_canary.py"]
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    forbidden = [".cuda(", 'device="cuda"', 'to("cuda")', "save_checkpoint(", "push_to_hub"]
    assert not any(token in source for token in forbidden)
