from __future__ import annotations

import inspect
from pathlib import Path

from jobs import ember_v050_data as data
from jobs import ember_v050_distill as distill
from jobs import ember_v050_canary as canary

ROOT = Path(__file__).resolve().parents[1]


def test_v050_is_cpu_only_minimal_delta_and_teacher_heavy():
    cfg = data.load_config()
    assert cfg["version"] == "0.0.50"
    assert cfg["cpu_learning_authorized"] is True
    assert cfg["gpu_training_authorized"] is False
    assert cfg["production_authorized"] is False
    assert cfg["promotion_authorized"] is False
    assert cfg["learning_rate"] <= 1e-7
    assert cfg["max_optimizer_steps"] <= 40
    assert cfg["tool_kl_loss_weight"] + cfg["copy_kl_loss_weight"] >= 0.75


def test_v050_target_values_are_fresh_and_disjoint():
    cfg = data.load_config()
    values = data.target_values(cfg)
    flat = [v for phase in values.values() for rows in phase.values() for v in rows]
    assert len(flat) == len(set(flat))
    assert not (set(flat) & set(data.copy_data.HELD_OUT_VALUES))
    assert not (set(flat) & data.historical_used_values())


def test_v050_distill_values_are_not_familiar_or_prior_targets():
    cfg = data.load_config()
    targets = data.target_values(cfg)
    target_set = {v for phase in targets.values() for rows in phase.values() for v in rows}
    tool = {row["target"] for row in data.distill_value_rows(cfg, "tool")}
    copy = {row["target"] for row in data.distill_value_rows(cfg, "copy")}
    assert not (tool & set(data.copy_data.HELD_OUT_VALUES))
    assert not (copy & set(data.copy_data.HELD_OUT_VALUES))
    assert not (tool & target_set)
    assert not (copy & target_set)
    assert not (tool & data.historical_used_values())
    assert not (copy & data.historical_used_values())


def test_selection_gate_requires_both_learning_and_teacher_retention():
    cfg = data.load_config()
    before_entry = {"top1": 18}
    after_entry = {"top1": 19}
    before_place = {"exact_top1": 0, "token_top1_rate": 0.50}
    after_place = {"exact_top1": 1, "token_top1_rate": 0.54}
    tool = {"token_top1_rate": 0.995, "teacher_kl": 0.02}
    copy = {"token_top1_rate": 0.995, "teacher_kl": 0.02}
    assert distill.selection_checks(before_entry, after_entry, before_place, after_place, tool, copy, cfg)["passed"]
    tool["teacher_kl"] = 0.2
    assert not distill.selection_checks(before_entry, after_entry, before_place, after_place, tool, copy, cfg)["passed"]


def test_candidate_selection_does_not_use_familiar_90_or_reference_controls():
    source = inspect.getsource(canary.main)
    loop_start = source.index("for step in range")
    final_eval = source.index("after_copy =", loop_start)
    loop_text = source[loop_start:final_eval]
    assert "familiar_90" not in loop_text
    assert "references(" not in loop_text
    assert "copy_diagnostic" not in loop_text


def test_v050_has_no_cuda_or_checkpoint_promotion_path():
    paths = [
        ROOT / "jobs/ember_v050_data.py",
        ROOT / "jobs/ember_v050_distill.py",
        ROOT / "jobs/ember_v050_canary.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    forbidden = [".cuda(", 'device="cuda"', 'to("cuda")', "save_checkpoint(", "push_to_hub", "promotion_authorized\": true"]
    assert not any(token in source for token in forbidden)
