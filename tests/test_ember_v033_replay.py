from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from jobs import ember_semantic_canary_v033 as retry

CFG = json.loads(retry.DEFAULT_CONFIG.read_text())


def copy_result(passed=True):
    keys = ("exact_copy_rate", "continuation_top1_rate", "clean_stop_rate", "expanded_exact_copy_rate",
            "expanded_continuation_top1_rate", "expanded_clean_stop_rate")
    return {"passed": passed, "metrics": dict.fromkeys(keys, 1.0 if passed else 0.0),
            "legacy_cases": [{"value": "A", "exact": passed, "clean_stop": True}],
            "expanded_cases": [{"value": "B", "exact": passed, "clean_stop": True}]}


def test_retry_preserves_source_dataset_gates_and_probe_selection():
    cfg = retry.load_config(retry.DEFAULT_CONFIG)
    old = json.loads(retry.base.DEFAULT_CONFIG.read_text())
    files = retry.file_hashes(cfg)
    assert "jobs/ember_semantic_train_v032.py" in files
    assert all(files[path] == sha for path, sha in old["pinned_files"].items())
    for key in ("seed", "learning_rate", "block_size", "generation_budget", "dataset"):
        assert cfg[key] == old[key]
    assert cfg["canary"]["gate"] == old["canary"]["gate"]
    assert cfg["canary"]["pairs_per_family"] == old["canary"]["pairs_per_family"]
    assert cfg["canary"]["max_steps"] == 120
    assert cfg["gpu_training_authorized"] is False and "train" not in cfg


@pytest.mark.parametrize("field,value", [("gpu_training_authorized", True), ("production_authorized", True),
                                         ("training_authorized", False), ("version", "0.0.32")])
def test_unsupported_authorization_or_version_is_rejected(tmp_path, monkeypatch, field, value):
    path = tmp_path / "config.json"
    cfg = copy.deepcopy(CFG)
    cfg[field] = value
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(retry, "DEFAULT_CONFIG", path)
    with pytest.raises(ValueError, match="unsupported CPU"):
        retry.load_config(path)


def test_weakened_learning_gate_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    cfg = copy.deepcopy(CFG)
    cfg["canary"]["gate"]["minimum_train_exact_gain"] = 0
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(retry, "DEFAULT_CONFIG", path)
    with pytest.raises(ValueError, match="retain the existing gates"):
        retry.load_config(path)


def test_broader_copy_replay_is_balanced_includes_old_replay_and_excludes_evaluation():
    rows = retry.base.replay_rows(3600, CFG["canary"]["copy_replay_per_kind"])
    old = retry.base.replay_rows(3600, 4)
    assert len(rows) == 360
    assert Counter(r["kind"] for r in rows) == dict.fromkeys(retry.base.copy_data.KINDS, 40)
    assert {r["id"] for r in old} <= {r["id"] for r in rows}
    assert {r["value"] for r in rows}.isdisjoint(retry.base.copy_data.HELD_OUT_VALUES)
    assert all(r["completion"].endswith(retry.base.semantic_data.EOT) for r in rows)
    assert CFG["copy_loss_fraction"] == 0.75 and CFG["canary"]["copy_batch_size"] == 6


def test_real_optimizer_weights_copy_more_and_checks_every_checkpoint_without_selecting_on_copy(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    logs = []
    monkeypatch.setitem(sys.modules, "trackio", SimpleNamespace(log=lambda values, step: logs.append(step)))
    training_modes = []
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(torch.zeros(5))
        def forward(self, x, y):
            training_modes.append(self.training)
            logits = self.scores.expand(*x.shape, 5)
            return logits, torch.nn.functional.cross_entropy(logits.reshape(-1, 5), y.reshape(-1))
    model = Tiny()
    source = {"model_config": {}, "tokenizer": {}}
    semantic = [(torch.tensor([1, 2, 0]), torch.tensor([-100, 1, -100]))]
    replay = [(torch.tensor([1, 2, 0]), torch.tensor([-100, 3, -100]))]
    cfg = copy.deepcopy(CFG)
    cfg["learning_rate"] = 0.02
    cfg["canary"].update(max_steps=3, warmup_steps=1, semantic_batch_size=1, copy_batch_size=1,
                         eval_interval=1, training_time_limit_seconds=20)
    losses = iter([0.5, 0.4, 0.6])
    monkeypatch.setattr(retry.base, "evaluate_loss", lambda *args: next(losses))
    checks = []
    def diagnose(current, tokenizer, torch):
        current.eval()
        checks.append(current.scores.detach().clone())
        return copy_result(passed=len(checks) != 2)
    monkeypatch.setattr(retry.base, "copy_diagnostic", diagnose)
    uploads = []
    api = SimpleNamespace(upload_folder=lambda **kwargs: uploads.append(kwargs))
    result = retry.train(model, semantic, replay, semantic, None, copy_result(), source,
                         cfg, {}, "test", tmp_path, api, "test", torch)
    assert result["best_step"] == 2  # Lowest dev loss, despite the recorded copy failure.
    assert len(checks) == len(uploads) == len(result["checkpoint_reports"]) == 3
    assert logs == [1, 2, 3] and all(training_modes)
    assert model.scores[3] > model.scores[1] > 0  # The 75% copy target receives the stronger update.
    assert torch.equal(model.scores, checks[1])
    assert result["replay_seen_indices"] == [0]
    for entry in result["checkpoint_reports"]:
        path = tmp_path / entry["checkpoint_path"]
        assert retry.base.data_check.digest(path) == entry["checkpoint_sha256"]
        saved = torch.load(path, weights_only=False)
        assert saved["step"] == entry["step"] and saved["optimizer_state"]["state"]
        assert saved["rng_state"] and saved["torch_rng_state"].numel() > 0
    assert (tmp_path / "best.pt").read_bytes() == (tmp_path / "checkpoints/step-0002.pt").read_bytes()
    assert not result["checkpoint_reports"][1]["copy_protection"]["passed"]
    assert json.loads((tmp_path / "checkpoint-reports.json").read_text()) == result["checkpoint_reports"]


def test_summary_explains_copy_failure_and_does_not_claim_promotion():
    probe = {"passed": 0, "total": 2}
    before = {"train_loss": 4.0, "development_loss": 4.0, "train_probe": probe, "development_probe": probe}
    report = {"status": "FAIL", "run_id": "test", "before": before, "after": before,
              "baseline_copy": copy_result(), "candidate_copy": copy_result(False), "best_step": 40,
              "learning_gate": {"checks": {"copy_preserved": False}},
              "checkpoint_reports": [{"step": 40, "development_loss": 4.0, "copy": copy_result(False),
                                       "copy_protection": {"passed": False}}]}
    summary = retry.summary_markdown(report)
    assert "`copy_preserved`" in summary and "| 40 |" in summary and "| 1/1 | 0/1 |" in summary
    assert "no GPU job is submitted" in summary and "not a held-out semantic promotion" in summary
    assert "incomplete" in retry.summary_markdown({"status": "ERROR", "run_id": "test"})
