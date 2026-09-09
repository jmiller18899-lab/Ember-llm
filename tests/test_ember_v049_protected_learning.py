from __future__ import annotations

import copy
import json

import pytest

from jobs import ember_v048_data as v048
from jobs import ember_v048_gate as v048_gate
from jobs import ember_v049_data as data
from jobs import ember_v049_gate as gate_logic
from jobs import ember_v049_objectives as objectives

CFG = json.loads(data.DEFAULT_CONFIG.read_text(encoding="utf-8"))
V048_CFG = json.loads(v048.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_config_is_cpu_only_and_cannot_authorize_gpu_or_promotion():
    cfg = data.load_config(data.DEFAULT_CONFIG)
    assert cfg["version"] == "0.0.49"
    assert cfg["source_version"] == "0.0.31"
    assert cfg["cpu_learning_authorized"] is True
    assert cfg["gpu_training_authorized"] is False
    assert cfg["production_authorized"] is False
    assert cfg["promotion_authorized"] is False


def test_update_is_substantially_smaller_than_the_v048_update():
    assert CFG["learning_rate"] < V048_CFG["learning_rate"]
    assert CFG["learning_rate"] <= V048_CFG["learning_rate"] / 5
    assert CFG["optimizer_steps"] == V048_CFG["optimizer_steps"], (
        "step count is held equal so the learning rate and trust region are the change"
    )
    assert 0 < CFG["trust_region_relative_drift"] <= 1e-3


def test_a_learning_rate_at_or_above_v048_is_rejected():
    bad = copy.deepcopy(CFG)
    bad["learning_rate"] = V048_CFG["learning_rate"]
    with pytest.raises(ValueError):
        _validate(bad)


def test_protection_must_dominate_the_learning_terms():
    assert CFG["replay_loss_weight"] >= 0.5
    assert CFG["replay_loss_weight"] > CFG["entry_loss_weight"] + CFG["placement_loss_weight"]
    weak = copy.deepcopy(CFG)
    weak.update(entry_loss_weight=0.25, placement_loss_weight=0.25, replay_loss_weight=0.5)
    with pytest.raises(ValueError, match="must exceed"):
        _validate(weak)


def test_gate_thresholds_are_not_relaxed_from_v048():
    # Lowering the learning bar while lowering the update would make a PASS
    # meaningless, so every v0.0.48 threshold is carried over unchanged.
    assert CFG["gate"] == V048_CFG["gate"]
    assert CFG["historical_kind_floor"] == V048_CFG["historical_kind_floor"]
    assert CFG["historical_subtype_floor"] == V048_CFG["historical_subtype_floor"]


def _validate(cfg: dict):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cfg.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        # load_config pins the path, so exercise the same checks directly.
        original = data.DEFAULT_CONFIG
        try:
            data.DEFAULT_CONFIG = path
            return data.load_config(path)
        finally:
            data.DEFAULT_CONFIG = original


def test_replay_covers_every_kind_that_collapsed_in_v048():
    plan = data.replay_values(CFG)
    assert set(plan) == {"envelope", "copy"}
    for corpus in plan.values():
        assert set(corpus) == set(data.copy_data.KINDS)
        assert len(corpus) == 9
        for values in corpus.values():
            assert len(values) == CFG["replay_envelope_values_per_kind"]
    # v0.0.48's synthetic values only reached five code-shaped subtypes.
    assert set(plan["envelope"]) - v048.ALL_SYNTHETIC_SUBTYPES


def test_no_training_value_touches_the_familiar_battery_or_earlier_phases():
    used = data.historical_used_values()
    objective = data.synthetic_values(CFG)
    plan = data.replay_values(CFG, set(used))
    everything = [
        value
        for groups in list(objective.values()) + list(plan.values())
        for values in groups.values()
        for value in values
    ]
    assert everything
    assert len(everything) == len(set(everything))
    for value in everything:
        assert value not in data.copy_data.HELD_OUT_VALUES
        assert value not in used


def test_replay_supervises_only_the_source_completion():
    tokenizer = _CharTokenizer()
    row = {"prompt": "abc", "completion": "xyz"}
    x, y = objectives.replay_example(tokenizer, row)
    prompt_ids = tokenizer.encode("abc")
    completion_ids = tokenizer.encode("xyz")
    assert x == prompt_ids + completion_ids[:-1]
    assert y[: len(prompt_ids) - 1] == [-100] * (len(prompt_ids) - 1)
    # position len(prompt)-1 predicts the first completion token
    assert y[len(prompt_ids) - 1] == completion_ids[0]
    assert y[len(prompt_ids) - 2 + len(completion_ids)] == completion_ids[-1]
    assert sum(1 for label in y if label != -100) == len(completion_ids)


def test_overlong_replay_rows_are_dropped_and_counted():
    tokenizer = _CharTokenizer()
    rows = [
        {"prompt": "ab", "completion": "cd"},
        {"prompt": "a" * 40, "completion": "b" * 40},
    ]
    examples, summary = objectives.build_replay_examples(tokenizer, rows, max_len=16)
    assert summary == {"rows": 2, "examples": 1, "dropped_too_long": 1}
    assert len(examples) == 1


def test_trust_region_bounds_drift_even_after_a_huge_step():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(8, 8, bias=True)
    trust = objectives.TrustRegion(model, torch, 1e-3)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.ones_like(param))
    unbounded = trust.measure(model)
    assert unbounded["max_relative_drift"] > 1e-3
    assert unbounded["within_trust_region"] is False
    projection = trust.project(model)
    assert projection["clipped_tensors"] >= 1
    bounded = trust.measure(model)
    # float32 rescaling lands fractionally over the target norm; the containment
    # check tolerates exactly that much and no more.
    assert bounded["max_relative_drift"] <= 1e-3 * (1 + objectives.DRIFT_TOLERANCE)
    assert bounded["max_relative_drift"] < unbounded["max_relative_drift"] / 100
    assert bounded["within_trust_region"] is True


def test_drift_tolerance_does_not_excuse_a_real_violation():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(8, 8, bias=False)
    trust = objectives.TrustRegion(model, torch, 1e-3)
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(1.01)
    measured = trust.measure(model)
    assert measured["max_relative_drift"] > 1e-3 * (1 + objectives.DRIFT_TOLERANCE)
    assert measured["within_trust_region"] is False


def test_trust_region_leaves_a_small_step_alone():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(8, 8, bias=False)
    trust = objectives.TrustRegion(model, torch, 1e-2)
    before = [p.detach().clone() for p in model.parameters()]
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(1 + 1e-4)
    projection = trust.project(model)
    assert projection["clipped_tensors"] == 0
    for param, original in zip(model.parameters(), before):
        assert torch.allclose(param.detach(), original * (1 + 1e-4))


def _decision_inputs(drift_ok: bool):
    before_entry = {"top1": 19, "cases": 24, "mean_loss": 4.0}
    after_entry = {"top1": 24, "cases": 24, "mean_loss": 1.0}
    before_place = {"exact_top1": 1, "cases": 24, "mean_loss": 3.0, "token_top1_rate": 0.52}
    after_place = {"exact_top1": 9, "cases": 24, "mean_loss": 1.0, "token_top1_rate": 0.86}
    drift = {
        "max_relative_drift": 1e-4 if drift_ok else 1e-2,
        "mean_relative_drift": 1e-5,
        "limit": 2e-4,
        "within_trust_region": drift_ok,
    }
    return before_entry, after_entry, before_place, after_place, drift


def test_gate_fails_when_the_trust_region_is_violated():
    be, ae, bp, ap, drift = _decision_inputs(drift_ok=False)
    decision = gate_logic.decide(
        be, ae, bp, ap, {"passed": True}, {"passed": True}, {"passed": True},
        True, drift, CFG,
    )
    assert decision["checks"]["trust_region"] is False
    assert decision["passed"] is False


def test_gate_passes_only_when_learning_and_every_protection_check_hold():
    be, ae, bp, ap, drift = _decision_inputs(drift_ok=True)
    decision = gate_logic.decide(
        be, ae, bp, ap, {"passed": True}, {"passed": True}, {"passed": True},
        True, drift, CFG,
    )
    assert decision["passed"] is True
    # every v0.0.48 check survives, plus the new one
    assert set(v048_gate.decide(
        be, ae, bp, ap, {"passed": True}, {"passed": True}, {"passed": True}, True, CFG,
    )["checks"]) < set(decision["checks"])
    for name in ("familiar", "reference", "copy"):
        broken = gate_logic.decide(
            be, ae, bp, ap,
            {"passed": name != "reference"},
            {"passed": name != "familiar"},
            {"passed": name != "copy"},
            True, drift, CFG,
        )
        assert broken["passed"] is False


def test_interference_summary_traces_the_run_from_start_to_finish():
    before = {"rate": 1.0, "correct_envelope_and_tool": 18, "by_kind": {"url": 2}, "cases": 18}
    after = {"rate": 0.5, "correct_envelope_and_tool": 9, "by_kind": {"url": 1}, "cases": 18}
    trajectory = [{
        "step": 40,
        "drift": {"max_relative_drift": 1e-4},
        "interference": {"rate": 0.8, "correct_envelope_and_tool": 14, "by_kind": {"url": 2}},
        "entry": {"top1": 22},
        "placement": {"exact_top1": 4},
    }]
    summary = gate_logic.interference_summary(before, after, trajectory)
    assert [point["step"] for point in summary["points"]] == [0, 40, "final"]
    assert summary["retained"] is False
    assert summary["points"][1]["entry_top1"] == 22


def test_runner_cannot_authorize_gpu_promotion_or_deployment():
    text = (data.ROOT / "jobs/ember_v049_canary.py").read_text(encoding="utf-8")
    assert '"gpu_training_authorized": False' in text
    assert '"promotion_authorized": False' in text
    assert "cuda" not in text.lower()
    for module in ("ember_v049_canary.py", "ember_v049_objectives.py", "ember_v049_data.py"):
        source = (data.ROOT / "jobs" / module).read_text(encoding="utf-8")
        assert "hf_hub_upload" not in source
        assert "upload_file" not in source
        assert "create_repo" not in source


class _CharTokenizer:
    """One id per character, so token boundaries in the tests are exact."""

    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, ids) -> str:
        return "".join(chr(int(i)) for i in ids)


def _toy_run(monkeypatch, torch, probe_steps, drift_limit):
    """Exercise the real three-term loop on a toy model.

    v0.0.48 burned two CPU runs on harness faults that only appeared once the
    optimizer started. This runs the actual loop end to end in under a second.
    """
    from jobs import ember_v048_data as v048_data

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(97, 16)
            self.lin = torch.nn.Linear(16, 97)
            self.config = type("C", (), {"block_size": 64})()

        def forward(self, x):
            return self.lin(self.emb(x)), None

    tokenizer = _CharTokenizer()
    monkeypatch.setattr(objectives, "token_contract", lambda _t: (5, 6))
    monkeypatch.setattr(
        v048_data, "value_continuation_ids",
        lambda _t, _tem, target: [ord(c) % 97 for c in target],
    )
    monkeypatch.setattr(_CharTokenizer, "encode", lambda _s, t: [ord(c) % 97 for c in t])

    cases = [
        {"id": f"c{i}", "subtype": "short_code/len4", "prompt": f"pp{i}", "target": f"vv{i}"}
        for i in range(6)
    ]
    replay = [{"prompt": f"rp{i}", "completion": f"rc{i}"} for i in range(6)]
    cfg = {
        "seed": 1, "optimizer_steps": 6,
        "entry_batch_size": 2, "placement_batch_size": 2, "replay_batch_size": 2,
        "learning_rate": 2e-7, "weight_decay": 0.0, "gradient_clip": 1.0,
        "entry_loss_weight": 0.15, "placement_loss_weight": 0.15, "replay_loss_weight": 0.70,
        "trust_region_relative_drift": drift_limit,
        "trajectory_probe_steps": probe_steps,
    }
    fired = []

    def probe(_model, step):
        fired.append(step)
        return {
            "interference": {"rate": 1.0, "correct_envelope_and_tool": 18, "by_kind": {}},
            "entry": {"top1": 20},
            "placement": {"exact_top1": 3},
        }

    result = objectives.run_canary(
        Toy(), tokenizer, torch, cfg, cases, cases, replay,
        {"prefix_ids": [1, 2, 3], "prefix_text": "abc"}, probe=probe,
    )
    return result, fired


def test_three_term_loop_runs_and_reports_all_three_losses(monkeypatch):
    torch = pytest.importorskip("torch")
    result, fired = _toy_run(monkeypatch, torch, [2, 4], 2e-4)
    assert fired == [2, 4]
    assert [snapshot["step"] for snapshot in result["trajectory"]] == [2, 4]
    assert result["objective_rows"] == {"entry": 6, "placement": 6, "replay": 6}
    for event in result["history"]:
        for key in ("entry_loss", "placement_loss", "replay_loss", "combined_loss"):
            assert event[key] > 0
    assert result["final_drift"]["within_trust_region"] is True


def test_report_says_plainly_when_the_trust_region_never_engaged(monkeypatch):
    torch = pytest.importorskip("torch")
    loose, _fired = _toy_run(monkeypatch, torch, [2], 1.0)
    assert loose["final_drift"]["engaged"] is False
    assert loose["final_drift"]["steps_clipped"] == 0
    assert loose["final_drift"]["headroom"] < 0.01

    tight, _fired = _toy_run(monkeypatch, torch, [2], 1e-9)
    assert tight["final_drift"]["engaged"] is True
    assert tight["final_drift"]["steps_clipped"] == 6
    assert tight["final_drift"]["within_trust_region"] is True


def test_an_empty_replay_corpus_is_refused(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(objectives, "token_contract", lambda _t: (5, 6))
    with pytest.raises(ValueError, match="non-empty replay corpus"):
        objectives.run_canary(
            object(), _CharTokenizer(), torch,
            {"trust_region_relative_drift": 1e-4}, [], [], [],
            {"prefix_ids": [], "prefix_text": ""},
        )
