"""Freeze UFC models, record pre-event forecasts, and score later outcomes.

Local hashes detect changed inputs; they are not independent proof of time.
Archive the registry and forecasts externally before events begin.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from jobs.ember_ufc_data import DAY, FEATURE_NAMES, SOURCE_FILES, FighterState, iso, load_bouts, update_states

MODES = ("continuous", "bsq", "hybrid")
SEEDS = (17, 29, 43)
MODELS = (*MODES, "logistic", "elo")
ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = ("jobs/ember_ufc_forward.py", "jobs/ember_ufc_data.py", "jobs/ember_bsq.py", "jobs/ember_bsq_train.py")


def now_seconds():
    return datetime.now(timezone.utc).timestamp()


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return parsed.timestamp()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(record):
    payload = {k: v for k, v in record.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_new(path, record):
    record = dict(record, sha256=digest(record))
    encoded = json.dumps(record, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    return record


def load_record(path, schema):
    record = json.loads(path.read_text())
    if record.get("schema") != schema or record.get("sha256") != digest(record):
        raise ValueError("record schema or checksum mismatch")
    return record


def register(benchmark_dir, output):
    """Record the existing validation-selected models without fitting new weights."""
    import torch
    report = json.loads((benchmark_dir / "benchmark.json").read_text())
    primary = report["selected_by_validation_log_loss"]
    if (report["data_kind"] != "historical_ufc_retrospective" or primary not in MODELS
            or primary != min(report["scores"], key=lambda m: report["scores"][m]["validation"]["log_loss"])):
        raise ValueError("use the original validation-selected UFC benchmark")
    assets = {"benchmark.json": file_hash(benchmark_dir / "benchmark.json")}
    for seed in SEEDS:
        for mode in MODES:
            name = f"seed-{seed}/{mode}.pt"
            checkpoint = torch.load(benchmark_dir / name, map_location="cpu", weights_only=True)
            cfg = checkpoint["config"]
            if (cfg["feature_names"] != FEATURE_NAMES or cfg["task"] != "binary"
                    or cfg["history_length"] != 1 or cfg["seed"] != seed
                    or checkpoint["model_args"]["mode"] != mode or not checkpoint.get("symmetric_binary")):
                raise ValueError("incompatible UFC checkpoint")
            assets[name] = file_hash(benchmark_dir / name)
    assets["logistic.pt"] = file_hash(benchmark_dir / "logistic.pt")
    return save_new(output, {
        "schema": "ember-ufc-forward-registry-v1", "registered_at": iso(now_seconds()),
        "primary_model": primary, "models": list(MODELS), "seeds": list(SEEDS),
        "feature_names": FEATURE_NAMES, "training_dataset_sha256": report["dataset_sha256"],
        "assets": assets, "implementation": {p: file_hash(ROOT / p) for p in CODE_FILES},
        "protocol": {"min_prior_bouts": 2, "min_lead_days": 1, "max_lead_days": 7,
                     "primary_metric": "log_loss", "secondary_metrics": ["brier", "accuracy"],
                     "target": "fighter_a_wins_conditional_on_decisive_result",
                     "history_availability_delay_days": 2, "refitting": False},
        "external_pre_event_timestamp_required": True,
    })


def check_assets(registry, benchmark_dir):
    for name, expected in registry["assets"].items():
        path = (benchmark_dir / name).resolve()
        if not path.is_relative_to(benchmark_dir.resolve()) or file_hash(path) != expected:
            raise ValueError("registered model assets changed")
    if registry["implementation"] != {p: file_hash(ROOT / p) for p in CODE_FILES}:
        raise ValueError("registered inference implementation changed")


def matchup_features(states, ids, as_of):
    a, b = [states[key] for key in ids]
    if min(len(a.history), len(b.history)) < 2:
        return None
    values = [[x - y for x, y in zip(a.vector(as_of), b.vector(as_of))]]
    return {"features": values, "prior_bouts": [len(a.history), len(b.history)],
            "latest_input_fight_date": iso(max(a.history[-1]["event_date"], b.history[-1]["event_date"])),
            "elo_probability": 1 / (1 + 10 ** ((b.elo - a.elo) / 400))}


def probabilities(benchmark_dir, features, elo):
    import torch
    from jobs.ember_bsq_train import predict_checkpoint
    result = {mode: sum(predict_checkpoint(benchmark_dir / f"seed-{seed}/{mode}.pt", features,
                                          FEATURE_NAMES)["probability"] for seed in SEEDS) / len(SEEDS)
              for mode in MODES}
    checkpoint = torch.load(benchmark_dir / "logistic.pt", map_location="cpu", weights_only=True)
    x = torch.tensor(features[0], dtype=torch.float32)
    result["logistic"] = float(((x / checkpoint["scale"]) @ checkpoint["weights"]).sigmoid())
    result["elo"] = elo
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in result.values()):
        raise ValueError("invalid model probability")
    return result


def forecast(registry_path, benchmark_dir, source_dir, schedule_path, output):
    """Use only available history; reject result-bearing or late schedules."""
    issued = now_seconds()
    registry = load_record(registry_path, "ember-ufc-forward-registry-v1")
    if timestamp(registry["registered_at"]) > issued:
        raise ValueError("registry is in the future")
    check_assets(registry, benchmark_dir)
    source_hashes = {p: file_hash(source_dir / p) for p in SOURCE_FILES}
    schedule_bytes = schedule_path.read_bytes()
    schedule = json.loads(schedule_bytes)
    if set(schedule) != {"source_url", "captured_at", "bouts"} or not schedule["source_url"].startswith("https://"):
        raise ValueError("schedule requires source URL, capture time, and bouts only")
    if not issued - DAY <= timestamp(schedule["captured_at"]) <= issued or not schedule["bouts"]:
        raise ValueError("use a nonempty schedule verified within the last day")
    bouts, audit = load_bouts(source_dir)
    states, names = defaultdict(FighterState), defaultdict(set)
    known_fights = {b.fight_id for b in bouts}
    for bout in bouts:
        for identity, name in zip(bout.fighter_ids, bout.fighters):
            names[identity].add(name)
        if bout.event_date + 2 * DAY < issued:
            update_states(states, bout)
    predicted, excluded, seen, card_fighters, card_dates = [], [], set(), set(), {}
    required = {"fight_id", "event_id", "event_date", "fighters"}
    for row in schedule["bouts"]:
        if set(row) != required:
            raise ValueError("schedule bout contains unsupported fields (including any results)")
        if not all(isinstance(row[k], str) and row[k].strip() for k in required - {"fighters"}):
            raise ValueError("nonempty fight/event identifiers and dates are required")
        event_date = datetime.combine(date.fromisoformat(row["event_date"]), datetime.min.time(), timezone.utc).timestamp()
        if not DAY <= event_date - issued <= 7 * DAY:
            raise ValueError("forecast must be issued 1–7 days before the UTC event date")
        if row["event_id"] in card_dates and card_dates[row["event_id"]] != event_date:
            raise ValueError("conflicting dates for the same card")
        card_dates[row["event_id"]] = event_date
        fighters = row["fighters"]
        if (not isinstance(fighters, list) or len(fighters) != 2
                or any(set(f) != {"id", "name"} or not all(isinstance(v, str) and v.strip() for v in f.values()) for f in fighters)):
            raise ValueError("provide two fighter IDs and names")
        fighters = sorted(fighters, key=lambda f: f["id"])
        ids = [f["id"] for f in fighters]
        if len(set(ids)) != 2 or any((row["event_id"], key) in card_fighters for key in ids):
            raise ValueError("duplicate fighter on card")
        if row["fight_id"] in seen or row["fight_id"] in known_fights:
            raise ValueError("duplicate or already-resulted fight")
        seen.add(row["fight_id"])
        card_fighters.update((row["event_id"], key) for key in ids)
        base = dict(row, fighters=fighters)
        if any(f["name"] not in names[f["id"]] for f in fighters):
            excluded.append(dict(base, reason="unknown_or_mismatched_identity"))
            continue
        values = matchup_features(states, ids, issued)
        if values is None:
            excluded.append(dict(base, reason="insufficient_prior_history"))
            continue
        predicted.append(dict(base, **values, probabilities=probabilities(benchmark_dir, values["features"], values["elo_probability"])))
    if source_hashes != {p: file_hash(source_dir / p) for p in SOURCE_FILES}:
        raise ValueError("source files changed during inference")
    check_assets(registry, benchmark_dir)
    if any(datetime.combine(date.fromisoformat(r["event_date"]), datetime.min.time(), timezone.utc).timestamp() - now_seconds() < DAY
           for r in schedule["bouts"]):
        raise ValueError("issuance window closed during inference")
    return save_new(output, {"schema": "ember-ufc-forward-forecast-v1", "issued_at": iso(issued),
                            "registry": registry, "source_sha256": source_hashes,
                            "schedule_sha256": hashlib.sha256(schedule_bytes).hexdigest(), "schedule": schedule,
                            "source_audit": audit, "feature_names": FEATURE_NAMES,
                            "predictions": predicted, "excluded": excluded,
                            "status": "awaiting_external_timestamp_and_future_results"})


def evaluate(forecast_path, outcomes_path, output):
    """Score the frozen probabilities; never retrain or revise predictions."""
    frozen = load_record(forecast_path, "ember-ufc-forward-forecast-v1")
    registry = frozen["registry"]
    if registry.get("sha256") != digest(registry):
        raise ValueError("registry checksum mismatch")
    raw = outcomes_path.read_bytes()
    outcomes = json.loads(raw)
    predictions = {r["fight_id"]: r for r in frozen["predictions"]}
    seen, used, excluded = set(), [], []
    for outcome in outcomes:
        fight_id = outcome["fight_id"]
        if fight_id in seen or fight_id not in predictions:
            raise ValueError("duplicate or unforecast outcome")
        seen.add(fight_id)
        row = predictions[fight_id]
        event_date = timestamp(row["event_date"] + "T00:00:00+00:00")
        if not event_date + 2 * DAY <= timestamp(outcome["observed_at"]) <= now_seconds():
            raise ValueError("outcome not yet available under the two-day convention")
        if not outcome["source_url"].startswith("https://"):
            raise ValueError("outcome source URL required")
        ids = [f["id"] for f in row["fighters"]]
        if outcome["event_id"] != row["event_id"]:
            raise ValueError("result card does not match forecast")
        if outcome["status"] != "matchup_changed" and (
                sorted(outcome["fighter_ids"]) != ids or outcome["event_date"] != row["event_date"]):
            raise ValueError("result must match the exact forecast matchup and date")
        if outcome["status"] in {"draw", "no_contest", "cancelled", "matchup_changed"}:
            excluded.append({"fight_id": fight_id, "reason": outcome["status"]})
            continue
        if outcome["status"] != "decisive" or outcome["winner_id"] not in ids:
            raise ValueError("result must match the exact forecast matchup")
        used.append((row, int(outcome["winner_id"] == ids[0])))
    scores = {}
    for model in MODELS:
        loss, brier, correct = 0., 0., 0
        for row, target in used:
            p = row["probabilities"][model]
            if not isinstance(p, (float, int)) or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("invalid saved probability")
            clipped = min(1 - 1e-6, max(1e-6, p))
            loss -= target * math.log(clipped) + (1 - target) * math.log1p(-clipped)
            brier += (p - target) ** 2
            correct += int((p >= .5) == bool(target))
        scores[model] = ({"log_loss": loss / len(used), "brier": brier / len(used), "accuracy": correct / len(used)} if used else None)
    return save_new(output, {"schema": "ember-ufc-forward-score-v1", "evaluated_at": iso(now_seconds()),
                            "forecast_sha256": frozen["sha256"], "outcomes_sha256": hashlib.sha256(raw).hexdigest(),
                            "primary_model": registry["primary_model"], "scored": len(used), "scores": scores,
                            "pending": sorted(set(predictions) - seen), "excluded": excluded,
                            "interpretation": "descriptive scores; external pre-event timestamp must be verified separately",
                            "production_promotion": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    register_parser = commands.add_parser("register")
    register_parser.add_argument("--benchmark-dir", type=Path, required=True)
    register_parser.add_argument("--output", type=Path, required=True)
    predict_parser = commands.add_parser("predict")
    for name in ("registry", "benchmark-dir", "source-dir", "schedule", "output"):
        predict_parser.add_argument("--" + name, type=Path, required=True)
    score_parser = commands.add_parser("score")
    for name in ("forecast", "outcomes", "output"):
        score_parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "register":
        result = register(args.benchmark_dir, args.output)
    elif args.command == "predict":
        result = forecast(args.registry, args.benchmark_dir, args.source_dir, args.schedule, args.output)
    else:
        result = evaluate(args.forecast, args.outcomes, args.output)
    print(json.dumps({"output": str(args.output), "sha256": result["sha256"], "schema": result["schema"]}))


if __name__ == "__main__":
    main()
