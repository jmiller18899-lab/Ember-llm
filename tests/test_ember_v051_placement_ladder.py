from __future__ import annotations

import copy as copymod
import json

import pytest

from jobs import ember_v048_objectives as objectives
from jobs import ember_v050_distill as distill
from jobs import ember_v051_ladder as ladder

CFG = json.loads(ladder.DEFAULT_CONFIG.read_text(encoding="utf-8"))
V050 = json.loads(ladder.V050_CONFIG.read_text(encoding="utf-8"))


def _validate(cfg: dict, tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return ladder.load_config(path)


def test_config_is_cpu_only_and_authorizes_nothing():
    cfg = ladder.load_config()
    assert cfg["version"] == "0.0.51"
    assert cfg["source_version"] == "0.0.31"
    assert cfg["cpu_learning_authorized"] is True
    for key in ("gpu_training_authorized", "production_authorized", "promotion_authorized"):
        assert cfg[key] is False


def test_entry_is_dropped_and_preservation_still_dominates():
    assert CFG["entry_loss_weight"] == 0.0
    preservation = CFG["tool_kl_loss_weight"] + CFG["copy_kl_loss_weight"]
    assert preservation >= 0.5
    assert preservation > CFG["placement_loss_weight"]
    assert CFG["placement_loss_weight"] == V050["entry_loss_weight"] + V050["placement_loss_weight"], (
        "placement inherits exactly the weight entry gave up"
    )


def test_entry_cannot_be_quietly_reintroduced(tmp_path):
    bad = dict(CFG, entry_loss_weight=0.05, placement_loss_weight=0.15)
    with pytest.raises(ValueError, match="weight must be zero"):
        _validate(bad, tmp_path)


def test_the_first_rung_is_the_v050_control(tmp_path):
    assert CFG["ladder_learning_rates"][0] == V050["learning_rate"]
    assert CFG["steps_per_rung"] == V050["max_optimizer_steps"]
    moved = dict(CFG, control_learning_rate=2e-07,
                 ladder_learning_rates=[2e-07, 4e-07, 1.6e-06])
    with pytest.raises(ValueError, match="match v0.0.50's measured learning rate"):
        _validate(moved, tmp_path)


def test_rungs_must_be_strictly_increasing_and_bounded(tmp_path):
    with pytest.raises(ValueError, match="strictly increasing"):
        _validate(dict(CFG, ladder_learning_rates=[1e-07, 1.6e-06, 4e-07]), tmp_path)
    with pytest.raises(ValueError, match="at least two rungs"):
        _validate(dict(CFG, ladder_learning_rates=[1e-07]), tmp_path)
    with pytest.raises(ValueError, match="learning-rate bound"):
        _validate(dict(CFG, ladder_learning_rates=[1e-07, 1.0]), tmp_path)


def test_the_development_set_is_identical_to_v050(tmp_path):
    for key in ("template_values_per_subtype", "train_values_per_subtype",
                "development_values_per_subtype", "tool_distill_values_per_variant",
                "copy_distill_values_per_variant", "generation_budget"):
        assert CFG[key] == V050[key]
    with pytest.raises(ValueError, match="development set is identical"):
        _validate(dict(CFG, development_values_per_subtype=8), tmp_path)


def test_the_operating_point_is_not_relaxed_from_v050(tmp_path):
    for key in ("minimum_placement_exact_gain", "minimum_placement_token_top1_gain",
                "maximum_tool_teacher_kl", "maximum_copy_teacher_kl"):
        assert CFG["operating_point"][key] == V050["selection_gate"][key]
    relaxed = copymod.deepcopy(CFG)
    relaxed["operating_point"]["minimum_placement_token_top1_gain"] = 0.005
    with pytest.raises(ValueError, match="must not be relaxed"):
        _validate(relaxed, tmp_path)


def _rung(lr, exact, token, tool_kl, preserved, candidate):
    return {
        "learning_rate": lr, "placement_exact_gain": exact,
        "placement_token_top1_gain": token, "tool_teacher_kl": tool_kl,
        "preservation_held": preserved, "candidate": candidate,
    }


def test_choose_rung_prefers_one_that_learned_and_preserved():
    rungs = [
        _rung(1e-7, 0, 0.0, 0.01, True, False),
        _rung(4e-7, 1, 0.04, 0.02, True, True),
        _rung(1.6e-6, 3, 0.09, 0.05, True, True),
        _rung(6.4e-6, 5, 0.20, 0.90, False, False),
    ]
    chosen = ladder.choose_rung(rungs)
    assert chosen["index"] == 2
    assert chosen["reason"] == "learned and preserved"


def test_choose_rung_falls_back_to_the_largest_update_that_preserved():
    rungs = [
        _rung(1e-7, 0, 0.0, 0.01, True, False),
        _rung(4e-7, 0, 0.0, 0.03, True, False),
        _rung(1.6e-6, 0, 0.01, 0.99, False, False),
    ]
    chosen = ladder.choose_rung(rungs)
    assert chosen["index"] == 1
    assert chosen["reason"] == "largest update that still preserved"


def test_choose_rung_when_nothing_preserved():
    rungs = [_rung(1e-7, 0, 0.0, 0.9, False, False), _rung(4e-7, 1, 0.05, 0.9, False, False)]
    chosen = ladder.choose_rung(rungs)
    assert chosen["index"] == 0
    assert "no rung preserved" in chosen["reason"]


def test_relative_drift_is_zero_at_the_source_and_grows_with_change():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(8, 8, bias=False)
    reference = {name: p.detach().clone() for name, p in model.named_parameters()}
    assert ladder.relative_drift(model, reference, torch)["max_relative_drift"] == 0.0
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(1.01)
    drift = ladder.relative_drift(model, reference, torch)
    assert drift["max_relative_drift"] == pytest.approx(0.01, rel=1e-3)


class _Tok:
    def encode(self, text):
        return [ord(c) % 97 for c in text]

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)


def _toy(torch):
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(97, 16)
            self.lin = torch.nn.Linear(16, 97)
            self.config = type("C", (), {"block_size": 64})()

        def forward(self, x):
            return self.lin(self.emb(x)), None

    return Toy()


def _stub_everything(monkeypatch, torch, probe_sequence):
    """Stub only the model-dependent helpers; the ladder's own logic stays real."""
    calls = {"probe": 0}

    def fake_placement_probe(model, tokenizer, _torch, cases, template):
        index = min(calls["probe"], len(probe_sequence) - 1)
        calls["probe"] += 1
        exact, rate = probe_sequence[index]
        return {"cases": 24, "exact_top1": exact, "exact_top1_rate": exact / 24,
                "token_top1": int(rate * 170), "tokens": 170, "token_top1_rate": rate,
                "mean_loss": 3.0, "rows": []}

    def fake_kl(student, teacher, tokenizer, _torch, rows, indices, temperature):
        x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        logits, _ = student(x)
        return logits.pow(2).mean() * 1e-3

    def fake_distill_probe(student, teacher, tokenizer, _torch, rows, batch_size=8, temperature=1.0):
        return {"token_top1_rate": 0.997, "teacher_kl": 0.004}

    monkeypatch.setattr(objectives, "placement_probe", fake_placement_probe)
    monkeypatch.setattr(distill, "batch_teacher_kl", fake_kl)
    monkeypatch.setattr(distill, "distill_probe", fake_distill_probe)
    return calls


def _ladder_cfg():
    return {
        "seed": 5, "steps_per_rung": 4, "checkpoint_interval": 2,
        "placement_batch_size": 2, "tool_distill_batch_size": 2, "copy_distill_batch_size": 2,
        "weight_decay": 0.0, "gradient_clip": 0.25, "distill_temperature": 1.0,
        "placement_loss_weight": 0.20, "tool_kl_loss_weight": 0.55, "copy_kl_loss_weight": 0.25,
        "operating_point": CFG["operating_point"],
    }


def test_run_rung_executes_the_real_three_term_loop(monkeypatch):
    torch = pytest.importorskip("torch")
    _stub_everything(monkeypatch, torch, [(1, 0.52), (2, 0.56), (4, 0.60)])
    model = _toy(torch)
    teacher = copymod.deepcopy(model)
    pristine = {n: p.detach().clone() for n, p in teacher.named_parameters()}
    examples = [([1, 2, 3], [-100, -100, 4]) for _ in range(4)]
    rows = [{"kind": "url", "target": "x"} for _ in range(4)]
    before = {"exact_top1": 1, "token_top1_rate": 0.52, "cases": 24}

    rung = ladder.run_rung(model, teacher, _Tok(), torch, _ladder_cfg(), 4e-7,
                           examples, rows, rows, [], {"prefix_ids": [], "prefix_text": ""},
                           before, pristine)

    assert [event["step"] for event in rung["history"]] == [1, 2, 4]
    assert [point["step"] for point in rung["trajectory"]] == [2, 4]
    for event in rung["history"]:
        assert event["placement_loss"] > 0 and event["tool_kl_loss"] > 0
    assert rung["drift"]["max_relative_drift"] > 0, "the rung must actually move the model"
    balance = rung["gradient_balance"]
    assert balance["weighted_placement_grad_norm"] > 0
    assert balance["weighted_preservation_grad_norm"] > 0
    assert 0.0 <= balance["placement_share"] <= 1.0


def test_a_rung_is_a_candidate_only_when_it_both_learned_and_preserved(monkeypatch):
    torch = pytest.importorskip("torch")
    _stub_everything(monkeypatch, torch, [(1, 0.52), (1, 0.52), (1, 0.52)])
    model = _toy(torch)
    teacher = copymod.deepcopy(model)
    pristine = {n: p.detach().clone() for n, p in teacher.named_parameters()}
    rung = ladder.run_rung(
        model, teacher, _Tok(), torch, _ladder_cfg(), 4e-7,
        [([1, 2, 3], [-100, -100, 4])] * 4, [{"kind": "url", "target": "x"}] * 4,
        [{"kind": "url", "target": "x"}] * 4, [], {"prefix_ids": [], "prefix_text": ""},
        {"exact_top1": 1, "token_top1_rate": 0.52, "cases": 24}, pristine)
    assert rung["placement_exact_gain"] == 0
    assert rung["placement_token_top1_gain"] == 0.0
    assert rung["preservation_held"] is True
    assert rung["placement_moved"] is False
    assert rung["candidate"] is False


def test_rerunning_a_rung_from_the_same_source_reproduces_it(monkeypatch):
    """The selected rung is re-run for evaluation, so that must be deterministic."""
    torch = pytest.importorskip("torch")
    _stub_everything(monkeypatch, torch, [(1, 0.52), (2, 0.56), (3, 0.58)])
    model = _toy(torch)
    teacher = copymod.deepcopy(model)
    pristine_state = copymod.deepcopy(teacher.state_dict())
    pristine = {n: p.detach().clone() for n, p in teacher.named_parameters()}
    args = ([([1, 2, 3], [-100, -100, 4])] * 4, [{"kind": "url", "target": "x"}] * 4,
            [{"kind": "url", "target": "x"}] * 4, [], {"prefix_ids": [], "prefix_text": ""},
            {"exact_top1": 1, "token_top1_rate": 0.52, "cases": 24}, pristine)

    first = ladder.run_rung(model, teacher, _Tok(), torch, _ladder_cfg(), 1.6e-6, *args)
    model.load_state_dict(copymod.deepcopy(pristine_state))
    second = ladder.run_rung(model, teacher, _Tok(), torch, _ladder_cfg(), 1.6e-6, *args)

    assert first["drift"] == second["drift"]
    assert [e["combined_loss"] for e in first["history"]] == [e["combined_loss"] for e in second["history"]]


def test_runner_cannot_authorize_gpu_promotion_or_deployment():
    text = (ladder.ROOT / "jobs/ember_v051_ladder.py").read_text(encoding="utf-8")
    assert '"gpu_training_authorized": False' in text
    assert '"promotion_authorized": False' in text
    assert "cuda" not in text.lower()
    for forbidden in ("upload_file", "create_repo", "hf_hub_upload"):
        assert forbidden not in text
