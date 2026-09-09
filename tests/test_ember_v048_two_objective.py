from __future__ import annotations

import inspect
from pathlib import Path

from jobs import ember_v048_data as data
from jobs import ember_v048_objectives as objectives
from jobs import ember_v048_regression as regression

ROOT = Path(__file__).resolve().parents[1]


def test_v048_config_is_cpu_only_and_two_objective():
    cfg = data.load_config()
    assert cfg["version"] == "0.0.48"
    assert cfg["cpu_learning_authorized"] is True
    assert cfg["gpu_training_authorized"] is False
    assert cfg["production_authorized"] is False
    assert cfg["promotion_authorized"] is False
    assert set(cfg["entry_subtypes"]) == data.ENTRY_SUBTYPES
    assert set(cfg["placement_subtypes"]) == data.PLACEMENT_SUBTYPES


def test_v048_synthetic_values_are_disjoint_from_heldout_and_each_other():
    values = data.synthetic_values(data.load_config())
    flat = [v for groups in values.values() for items in groups.values() for v in items]
    assert len(flat) == len(set(flat))
    assert not (set(flat) & set(data.copy_data.HELD_OUT_VALUES))


def test_v048_prompts_use_the_frozen_v044_stack():
    values = data.synthetic_values(data.load_config())
    for subtype in sorted(data.ALL_SYNTHETIC_SUBTYPES):
        value = values["development"][subtype][0]
        case = data.prompt_for(subtype, value)
        kind, _ = data.SUBTYPE_VARIANT[subtype]
        _uid, user = data.v044.frozen_user(kind, value)
        assert case["prompt"] == data.v044._prompt(user, data.v044.baseline_system(kind))


def test_entry_objective_supervises_only_the_tool_entry_position():
    source = inspect.getsource(objectives.supervised_example)
    assert 'objective == "entry"' in source
    assert "y[-1] = tool_id" in source
    assert 'objective != "placement"' in source


def test_placement_objective_uses_target_only_after_the_observed_prefix():
    source = inspect.getsource(objectives.supervised_example)
    assert "value_continuation_ids" in source
    assert 'template["prefix_ids"]' in source
    assert "start = len(prompt_ids) + len(prefix_ids) - 1" in source


def test_familiar_90_is_read_only_regression_code_not_objective_data():
    objective_source = (ROOT / "jobs/ember_v048_objectives.py").read_text(encoding="utf-8")
    regression_source = inspect.getsource(regression.familiar_90)
    assert "frozen_v044_cases" not in objective_source
    assert "frozen_v044_cases" in regression_source


def test_v048_has_no_cuda_execution_path():
    for path in (
        ROOT / "jobs/ember_v048_data.py",
        ROOT / "jobs/ember_v048_objectives.py",
        ROOT / "jobs/ember_v048_regression.py",
        ROOT / "jobs/ember_v048_canary.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert ".cuda(" not in source
        assert 'to("cuda")' not in source
        assert 'device="cuda"' not in source
