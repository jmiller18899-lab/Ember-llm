"""Forward workflow fixtures are synthetic and never reported as live forecasts."""
from dataclasses import replace
import json

import pytest
import torch

from jobs import ember_ufc_forward as forward
from jobs.ember_bsq import BSQForecaster
from jobs.ember_bsq_train import load_config
from jobs.ember_ufc_data import Bout, DAY, FEATURE_NAMES, SOURCE_FILES


@pytest.fixture
def setup(tmp_path, monkeypatch):
    clock = [forward.timestamp("2030-01-05T00:00:00Z")]
    monkeypatch.setattr(forward, "now_seconds", lambda: clock[0])
    torch.set_num_threads(1)
    models = tmp_path / "models"
    models.mkdir()
    cfg = load_config(forward.ROOT / "config/ember_ufc_forecast.json")
    for seed in forward.SEEDS:
        folder = models / f"seed-{seed}"
        folder.mkdir()
        for mode in forward.MODES:
            args = dict(feature_count=len(FEATURE_NAMES), bits=4, hidden_size=4, mode=mode)
            torch.manual_seed(seed)
            model = BSQForecaster(**args)
            torch.save({"schema_version": cfg["schema_version"], "config": dict(cfg, seed=seed),
                        "model_args": args, "model_state_dict": model.state_dict(), "symmetric_binary": True,
                        "normalization": {"feature_mean": torch.zeros(20), "feature_scale": torch.ones(20)}},
                       folder / f"{mode}.pt")
    torch.save({"weights": torch.zeros(20), "scale": torch.ones(20)}, models / "logistic.pt")
    report = {"data_kind": "historical_ufc_retrospective", "selected_by_validation_log_loss": "continuous",
              "dataset_sha256": "0" * 64, "scores": {m: {"validation": {"log_loss": .6 + i / 20}}
                                                        for i, m in enumerate(forward.MODELS)}}
    (models / "benchmark.json").write_text(json.dumps(report))
    registry = tmp_path / "registry.json"
    forward.register(models, registry)
    source = tmp_path / "source"
    source.mkdir()
    for name in SOURCE_FILES:
        (source / name).write_text("synthetic source fixture\n")
    stats = {"sig_l": 10., "sig_a": 20., "td_l": 1., "td_a": 2., "sub": 0.,
             "control": 30., "control_known_seconds": 300.}
    base = forward.timestamp("2030-01-01T00:00:00Z")
    history = [Bout(f"prior-{i}", f"past-card-{i}", base + i * DAY,
                    ("Alpha", "Zulu"), ("id-a", "id-z"), "W/L", "Decision", 300., (dict(stats), dict(stats)))
               for i in range(2)]
    monkeypatch.setattr(forward, "load_bouts", lambda _: (history, {"fixture": True}))
    schedule = tmp_path / "schedule.json"
    card = {"source_url": "https://example.com/synthetic-card", "captured_at": forward.iso(clock[0]),
            "bouts": [{"fight_id": "new-fight", "event_id": "new-card", "event_date": "2030-01-10",
                       "fighters": [{"id": "id-z", "name": "Zulu"}, {"id": "id-a", "name": "Alpha"}]}]}
    schedule.write_text(json.dumps(card))
    return dict(clock=clock, models=models, registry=registry, source=source, schedule=schedule,
                card=card, history=history, output=tmp_path / "forecast.json", root=tmp_path)


def predict(s):
    return forward.forecast(s["registry"], s["models"], s["source"], s["schedule"], s["output"])


def test_freezes_models_and_records_unlabelled_probabilities(setup):
    result = predict(setup)
    row = result["predictions"][0]
    assert result["registry"]["primary_model"] == "continuous"
    assert row["fighters"][0]["name"] == "Alpha"
    assert row["prior_bouts"] == [2, 2]
    assert set(row["probabilities"]) == set(forward.MODELS)
    assert "target" not in row and "winner_id" not in row
    assert all(0 <= p <= 1 for p in row["probabilities"].values())
    with pytest.raises(FileExistsError):
        predict(setup)
    with pytest.raises(FileExistsError):
        forward.register(setup["models"], setup["registry"])


@pytest.mark.parametrize("change", ["checkpoint", "report", "code"])
def test_changed_frozen_assets_are_rejected(setup, monkeypatch, change):
    if change == "code":
        monkeypatch.setattr(forward, "CODE_FILES", ())
    else:
        path = setup["models"] / ("benchmark.json" if change == "report" else "seed-17/continuous.pt")
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        predict(setup)
    assert not setup["output"].exists()


@pytest.mark.parametrize("change", ["late", "too_early", "future_capture", "stale_capture", "result", "duplicate", "identity"])
def test_invalid_or_unverifiable_schedules_fail_or_record_exclusion(setup, change):
    card = setup["card"]
    row = card["bouts"][0]
    if change == "late":
        row["event_date"] = "2030-01-05"
    elif change == "too_early":
        row["event_date"] = "2030-02-01"
    elif change == "future_capture":
        card["captured_at"] = "2030-01-06T00:00:00Z"
    elif change == "stale_capture":
        card["captured_at"] = "2030-01-01T00:00:00Z"
    elif change == "result":
        row["winner_id"] = "id-a"
    elif change == "duplicate":
        card["bouts"].append(dict(row))
    else:
        row["fighters"][0]["name"] = "Wrong name"
    setup["schedule"].write_text(json.dumps(card))
    if change == "identity":
        result = predict(setup)
        assert not result["predictions"]
        assert result["excluded"][0]["reason"] == "unknown_or_mismatched_identity"
    else:
        with pytest.raises(ValueError):
            predict(setup)


def test_unavailable_history_never_changes_features(setup):
    expected = predict(setup)["predictions"][0]
    setup["output"] = setup["root"] / "second.json"
    setup["history"].append(replace(setup["history"][0], fight_id="future-result",
                                    event_date=setup["clock"][0], outcome="L/W", method="KO/TKO"))
    actual = predict(setup)["predictions"][0]
    assert actual["features"] == expected["features"]
    assert actual["probabilities"] == expected["probabilities"]


def result_file(s, **changes):
    row = {"fight_id": "new-fight", "event_id": "new-card", "event_date": "2030-01-10", "fighter_ids": ["id-a", "id-z"],
           "winner_id": "id-a", "status": "decisive", "observed_at": "2030-01-12T01:00:00Z",
           "source_url": "https://example.com/synthetic-result"}
    row.update(changes)
    path = s["root"] / "results.json"
    path.write_text(json.dumps([row]))
    return path


def test_scoring_uses_exact_frozen_probabilities_and_never_changes_forecast(setup):
    frozen = predict(setup)
    before = setup["output"].read_bytes()
    setup["clock"][0] = forward.timestamp("2030-01-13T00:00:00Z")
    scored = forward.evaluate(setup["output"], result_file(setup), setup["root"] / "score.json")
    p = frozen["predictions"][0]["probabilities"]["continuous"]
    assert scored["scores"]["continuous"]["brier"] == pytest.approx((p - 1) ** 2)
    assert scored["scored"] == 1 and not scored["pending"]
    assert not scored["production_promotion"]
    assert setup["output"].read_bytes() == before


@pytest.mark.parametrize("change", ["early", "wrong_winner", "different_matchup", "tamper", "duplicate", "wrong_card_draw", "rescheduled"])
def test_invalid_scoring_inputs_fail(setup, change):
    predict(setup)
    setup["clock"][0] = forward.timestamp("2030-01-13T00:00:00Z")
    changes = {"early": {"observed_at": "2030-01-10T12:00:00Z"}, "wrong_winner": {"winner_id": "other"},
               "different_matchup": {"fighter_ids": ["id-a", "other"]},
               "wrong_card_draw": {"status": "draw", "event_id": "other-card"},
               "rescheduled": {"event_date": "2030-01-11"}}.get(change, {})
    outcomes = result_file(setup, **changes)
    if change == "tamper":
        document = json.loads(setup["output"].read_text())
        document["predictions"][0]["probabilities"]["continuous"] = .99
        setup["output"].write_text(json.dumps(document))
    elif change == "duplicate":
        outcomes.write_text(json.dumps(json.loads(outcomes.read_text()) * 2))
    with pytest.raises(ValueError):
        forward.evaluate(setup["output"], outcomes, setup["root"] / "score.json")


def test_draw_excluded_and_unreported_outcomes_remain_pending(setup):
    predict(setup)
    setup["clock"][0] = forward.timestamp("2030-01-13T00:00:00Z")
    outcomes = result_file(setup, status="draw", winner_id=None)
    result = forward.evaluate(setup["output"], outcomes, setup["root"] / "draw.json")
    assert result["scored"] == 0 and result["scores"]["continuous"] is None
    assert result["excluded"] == [{"fight_id": "new-fight", "reason": "draw"}]
    outcomes.write_text("[]")
    pending = forward.evaluate(setup["output"], outcomes, setup["root"] / "pending.json")
    assert pending["pending"] == ["new-fight"]
