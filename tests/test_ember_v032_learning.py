from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from jobs import ember_semantic_train_v032 as learning
from jobs import ember_submit_v032 as submit
from jobs import ember_sft_data_semantic_v1 as data

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads(learning.DEFAULT_CONFIG.read_text())


def copy_result():
    keys = ("exact_copy_rate", "continuation_top1_rate", "clean_stop_rate", "expanded_exact_copy_rate",
            "expanded_continuation_top1_rate", "expanded_clean_stop_rate")
    return {"passed": True, "metrics": dict.fromkeys(keys, 1.0),
            "legacy_cases": [{"value": "A", "exact": True, "clean_stop": True}],
            "expanded_cases": [{"value": "B", "exact": True, "clean_stop": True}]}


def evidence():
    before = {"train_loss": 4.0, "development_loss": 4.0,
              "train_probe": {"passed": 0, "by_kind": dict.fromkeys(data.KINDS, 0)},
              "development_probe": {"passed": 0, "by_kind": dict.fromkeys(data.KINDS, 0)}}
    after = {"train_loss": 3.0, "development_loss": 3.5,
             "train_probe": {"passed": 3, "by_kind": dict.fromkeys(data.KINDS, 1)},
             "development_probe": {"passed": 1, "by_kind": {"tool_call": 1, "direct_response": 0, "tool_result_response": 0}}}
    copy_baseline = copy_result()
    protection = learning.copy_protection(copy_baseline, copy_baseline)
    gate = learning.learning_gate(before, after, protection, CFG["canary"]["gate"])
    return {"mode": "canary", "status": "PASS", "identity": {"fixed": "identity"}, "best_step": 120,
            "model_state_changed": True, "before": before, "after": after,
            "baseline_copy": copy_baseline, "candidate_copy": copy_baseline, "learning_gate": gate}


def test_pins_and_entire_pairs_stay_in_their_original_splits():
    learning.file_hashes(CFG)
    splits = data.build_dataset()
    train = learning.select_pairs(splits["train"], 2, CFG["seed"])
    dev = learning.select_pairs(splits["validation"], 1, CFG["seed"] + 1)
    assert len(train) == 96 and len(dev) == 48
    assert {r["pair_id"] for r in train}.isdisjoint({r["pair_id"] for r in dev})
    assert train == learning.select_pairs(splits["train"], 2, CFG["seed"])
    assert all(r["split"] == "train" for r in train)


def test_copy_replay_excludes_diagnostics_and_ends_at_eos():
    rows = learning.replay_rows(3600, 4)
    assert len(rows) == 36
    assert {r["value"] for r in rows}.isdisjoint(learning.copy_data.HELD_OUT_VALUES)
    assert all(r["completion"].endswith(data.EOT) for r in rows)
    for _, value, _ in learning.copy_data.LEGACY_DIAGNOSTICS:
        from jobs import ember_hf_sft_v015 as base
        assert base.prompt_for(value) == learning.copy_data.legacy_prompt_for(value)


def test_correct_prefix_and_wrong_argument_still_fail_development_scoring():
    record = data.make_record("tool_call/weather_literal", 0, 0, 20260908)
    case = learning.fixture_case(record)
    assert learning.semantic_gate.score_case(case, record["completion"])["passed"]
    assert not learning.semantic_gate.score_case(case, record["completion"].replace("Asheville", "Eugene"))["passed"]
    assert not learning.semantic_gate.score_case(case, record["completion"] + "noise")["passed"]


def test_copy_count_cannot_hide_loss_of_a_previously_correct_case():
    original = copy_result()
    changed = copy_result()
    changed["expanded_cases"][0]["value"] = "C"
    assert not learning.copy_protection(original, changed)["passed"]


@pytest.mark.parametrize("metric", ["continuation_top1_rate", "expanded_clean_stop_rate", "expanded_exact_copy_rate"])
def test_copy_regression_blocks_readiness(metric):
    original, candidate = copy_result(), copy_result()
    candidate["metrics"][metric] = 0.9
    assert not learning.copy_protection(original, candidate)["passed"]


def test_matching_measured_canary_allows_next_stage():
    report = evidence()
    learning.validate_canary(report, report["identity"], CFG["canary"]["gate"])


@pytest.mark.parametrize("mutation", ["status", "mode", "identity", "best_step", "learning", "weights", "forged"])
def test_gpu_authorization_fails_closed_on_invalid_canary(mutation):
    report = evidence()
    if mutation in {"status", "mode", "identity"}:
        report[mutation] = "wrong"
    elif mutation == "best_step":
        report["best_step"] = 0
    elif mutation == "learning":
        report["after"]["development_loss"] = 4.5
    elif mutation == "weights":
        report["model_state_changed"] = False
    else:
        report["learning_gate"] = {"passed": True, "checks": {"made_up": True}}
    with pytest.raises(ValueError):
        learning.validate_canary(report, {"fixed": "identity"}, CFG["canary"]["gate"])


def test_loss_reduction_without_better_answers_is_not_learning_pass():
    report = evidence()
    report["after"]["train_probe"] = copy.deepcopy(report["before"]["train_probe"])
    gate = learning.learning_gate(report["before"], report["after"], {"passed": True}, CFG["canary"]["gate"])
    assert not gate["passed"]


def test_launch_script_pins_source_and_canary_and_never_contains_token():
    publication = {"revision": "b" * 40, "report_path": "runs/canary/report.json"}
    script = submit.job_script("a" * 40, publication, {"jobs/test.py": "c" * 64})
    compile(script, "job", "exec")
    assert "--canary-revision" in script and "--mode" in script and '"train"' in script
    assert "hf_" not in script and "main/" not in script
    with pytest.raises(ValueError):
        submit.job_script("main", publication, {})


def test_cpu_training_really_updates_and_saves_weights_with_completion_masks(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    logs = []
    monkeypatch.setitem(sys.modules, "trackio", SimpleNamespace(log=lambda values, step: logs.append(step)))
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(torch.zeros(5))
        def forward(self, x, y):
            logits = self.scores.expand(*x.shape, 5)
            return logits, torch.nn.functional.cross_entropy(logits.reshape(-1, 5), y.reshape(-1))
    model = Tiny()
    initial = model.scores.detach().clone()
    dataset = [(torch.tensor([1, 2, 3, 0, 0]), torch.tensor([-100, 3, 4, -100, -100]))]
    x, y = learning.stack_batch(dataset, [0], "cpu", torch)
    assert x.shape == (1, 3) and y.tolist() == [[-100, 3, 4]]
    cfg = {"seed": 1, "learning_rate": 0.02, "copy_loss_fraction": 0.25,
           "canary": {"max_steps": 2, "warmup_steps": 1, "semantic_batch_size": 1, "copy_batch_size": 1,
                      "eval_interval": 1, "training_time_limit_seconds": 20}}
    uploads = []
    api = SimpleNamespace(upload_folder=lambda **kwargs: uploads.append(kwargs))
    step, history = learning.train(model, dataset, dataset, dataset, {"model_config": {}, "tokenizer": {}},
                                   cfg, "canary", {}, "test", tmp_path, api, "test", torch)
    assert step > 0 and len(history) == 2 and logs == [1, 2]
    assert not torch.equal(initial, model.scores)
    saved = torch.load(tmp_path / "best.pt", weights_only=False)
    assert saved["optimizer_state"]["state"] and saved["production_authorized"] is False
    assert uploads and history[-1]["development_loss"] < history[0]["development_loss"]
