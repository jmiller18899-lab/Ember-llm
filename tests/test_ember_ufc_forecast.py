from __future__ import annotations

from dataclasses import replace
import csv
import json
from pathlib import Path

import pytest

from jobs.ember_ufc_data import Bout, DAY, FEATURE_NAMES, build_examples, load_bouts
from jobs.ember_bsq_data import chronological_split, load_examples, write_smoke_data, timestamp
from jobs.ember_bsq_train import load_config, predict_checkpoint, run
from jobs.ember_ufc_benchmark import BOUNDARIES

ROOT = Path(__file__).resolve().parents[1]
BASE = timestamp("2010-01-01T00:00:00Z")


def bout(number, *, date=None, outcome="W/L"):
    stats = {"sig_l": 10., "sig_a": 20., "td_l": 1., "td_a": 2., "sub": 0.,
             "control": 30., "control_known_seconds": 300.}
    return Bout(f"fight-{number}", f"event-{number}", BASE + number * 10 * DAY if date is None else date,
                ("Zulu", "Alpha"), ("id-z", "id-a"), outcome, "Decision - Unanimous", 300.,
                (dict(stats), dict(stats)))


def test_current_outcome_and_fight_statistics_never_enter_its_features():
    history = [bout(0), bout(1, outcome="L/W"), bout(2)]
    before, _ = build_examples(history)
    huge = {k: v * 100 for k, v in history[-1].stats[0].items()}
    changed = history[:-1] + [replace(history[-1], outcome="L/W", method="KO/TKO", stats=(huge, huge))]
    after, _ = build_examples(changed)
    assert len(before) == len(after) == 1
    assert before[0]["features"] == after[0]["features"]
    assert before[0]["target"] == 1 - after[0]["target"]
    assert before[0]["fighter_a"] == after[0]["fighter_a"] == "Alpha"


def test_same_day_and_future_fights_cannot_change_prior_features():
    history = [bout(0), bout(1)]
    one, _ = build_examples(history + [bout(2)])
    two, _ = build_examples(history + [bout(2), bout(3, date=bout(2).event_date), bout(4)])
    assert one[0]["features"] == two[0]["features"] == two[1]["features"]
    assert one[0]["prior_bouts"] == two[1]["prior_bouts"] == [2, 2]
    assert timestamp(two[0]["latest_input_fight_date"]) + 2 * DAY < timestamp(two[0]["as_of"])


def test_result_order_does_not_control_fighter_a_or_label():
    history = [bout(0), bout(1)]
    target = bout(2)
    reverse = replace(target, fighters=target.fighters[::-1], fighter_ids=target.fighter_ids[::-1],
                      stats=target.stats[::-1], outcome="L/W")
    a, _ = build_examples(history + [target])
    b, _ = build_examples(history + [reverse])
    assert a == b


def test_draws_and_no_contests_are_excluded_as_targets():
    rows, audit = build_examples([bout(0), bout(1), bout(2, outcome="D/D"), bout(3, outcome="NC/NC")])
    assert rows == []
    assert audit["draw_or_no_contest_target"] == 2


def test_fixed_date_splits_are_independent_of_the_dataset_end(tmp_path):
    cfg = load_config(ROOT / "config/ember_ufc_forecast.json")
    path = tmp_path / "data.jsonl"
    write_smoke_data(path, cfg)
    rows = load_examples(path, cfg)
    rewritten = []
    for i, row in enumerate(rows):
        year = 2021 + i // 80
        as_of = timestamp(f"{year}-01-01T00:00:00Z") + (i % 80) * 4 * DAY
        rewritten.append(replace(row, as_of=as_of, label_available_at=as_of + 3 * DAY))
    # Put the final block in the held-out era.
    rewritten[160:] = [replace(r, as_of=r.as_of + 366 * DAY,
                               label_available_at=r.label_available_at + 366 * DAY) for r in rewritten[160:]]
    parts, _ = chronological_split(rewritten, BOUNDARIES)
    assert len(parts[0]) == len(parts[1]) == len(parts[2]) == 80
    assert max(r.label_available_at for r in parts[0]) < BOUNDARIES[0]
    assert max(r.label_available_at for r in parts[1]) < BOUNDARIES[1]


def test_symmetric_training_checkpoint_reverses_matchup_probability(tmp_path):
    cfg = load_config(ROOT / "config/ember_ufc_forecast.json")
    cfg["max_steps"] = 2
    assert cfg["feature_names"] == FEATURE_NAMES
    path = tmp_path / "data.jsonl"
    write_smoke_data(path, cfg)
    output = tmp_path / "models"
    run(cfg, path, output, symmetric_binary=True)
    x = [[.1 * (i + 1) for i in range(len(FEATURE_NAMES))]]
    opposite = [[-value for value in x[0]]]
    for mode in ("continuous", "bsq", "hybrid"):
        p = predict_checkpoint(output / f"{mode}.pt", x, FEATURE_NAMES)["probability"]
        q = predict_checkpoint(output / f"{mode}.pt", opposite, FEATURE_NAMES)["probability"]
        assert p + q == pytest.approx(1., abs=1e-6)


def test_csv_event_aliases_deduplicate_and_conflicting_results_are_dropped(tmp_path):
    def write(name, rows):
        with (tmp_path / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write("ufc_event_details.csv", [
        {"EVENT": name, "URL": "event-url", "DATE": "January 01, 2010"} for name in ("UFC Alias", "UFC Original")])
    write("ufc_fighter_tott.csv", [{"FIGHTER": "Alpha", "URL": "id-a"}, {"FIGHTER": "Zulu", "URL": "id-z"}])
    result = {"EVENT": "UFC Alias", "BOUT": "Alpha vs. Zulu", "OUTCOME": "W/L", "URL": "fight-url",
              "TIME FORMAT": "3 Rnd (5-5-5)", "ROUND": "1", "TIME": "3:00", "METHOD": "KO/TKO"}
    results = [result, dict(result, EVENT="UFC Original")]
    write("ufc_fight_results.csv", results)
    stats = [{"EVENT": event, "BOUT": "Alpha vs. Zulu", "FIGHTER": name, "ROUND": "Round 1",
              "SIG.STR.": "10 of 20", "TD": "1 of 2", "SUB.ATT": "0", "CTRL": "0:30"}
             for event in ("UFC Alias", "UFC Original") for name in ("Alpha", "Zulu")]
    write("ufc_fight_stats.csv", stats)
    bouts, audit = load_bouts(tmp_path)
    assert len(bouts) == 1
    assert bouts[0].stats[0]["sig_l"] == 10
    assert audit["duplicate_result_rows"] == 1
    results[-1]["OUTCOME"] = "L/W"
    write("ufc_fight_results.csv", results)
    bouts, audit = load_bouts(tmp_path)
    assert not bouts
    assert audit["conflicting_result_ids"] == 1
