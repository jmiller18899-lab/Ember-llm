"""Offline examples and chronological splits for the BSQ experiment."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path


def timestamp(value: str) -> float:
    if not isinstance(value, str):
        raise ValueError("timestamps must be ISO-8601 strings with a timezone")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return parsed.timestamp()


@dataclass(frozen=True)
class Example:
    sample_id: str
    event_id: str
    as_of: float
    label_available_at: float
    features: list[list[float]]
    target: float


def load_examples(path: Path, cfg: dict) -> list[Example]:
    """Validate declared availability; collectors must also audit source contents."""
    result, ids = [], set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                for key in ("sample_id", "event_id", "source"):
                    if not isinstance(row[key], str) or not row[key].strip():
                        raise ValueError(f"{key} must be a nonempty string")
                if row["sample_id"] in ids:
                    raise ValueError("duplicate sample_id")
                as_of = timestamp(row["as_of"])
                label_time = timestamp(row["label_available_at"])
                if label_time < as_of + cfg["horizon_seconds"]:
                    raise ValueError("label must be available at or after the forecast horizon")
                observed = [timestamp(t) for t in row["observed_at"]]
                if len(observed) != cfg["history_length"] or observed != sorted(set(observed)):
                    raise ValueError("observed_at must contain the configured history in strict time order")
                if observed[-1] > as_of:
                    raise ValueError("future observations cannot be input features")
                if row["feature_names"] != cfg["feature_names"]:
                    raise ValueError("feature_names and their order must match the config")
                features = row["features"]
                if not isinstance(features, list) or len(features) != len(observed):
                    raise ValueError("features must match observed_at")
                for observation in features:
                    if not isinstance(observation, list) or len(observation) != len(cfg["feature_names"]):
                        raise ValueError("wrong feature width")
                    if any(type(v) not in (int, float) or not math.isfinite(v) for v in observation):
                        raise ValueError("features must be finite numbers; preprocess missing values explicitly")
                target = row["target"]
                if type(target) not in (int, float) or not math.isfinite(target):
                    raise ValueError("target must be a finite number")
                if cfg["task"] == "binary" and target not in (0, 1):
                    raise ValueError("binary targets must be 0 or 1")
                result.append(Example(row["sample_id"], row["event_id"], as_of,
                                      label_time, features, float(target)))
                ids.add(row["sample_id"])
                if len(result) > 50000:
                    raise ValueError("this CPU experiment supports at most 50000 examples")
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if len(result) < 15:
        raise ValueError("need at least 15 examples for three chronological partitions")
    return sorted(result, key=lambda row: (row.as_of, row.sample_id))


def chronological_split(rows: list[Example]):
    """60/20/20 by unique prediction time; purge labels unavailable at the next split."""
    times = sorted({row.as_of for row in rows})
    if len(times) < 5:
        raise ValueError("need at least five distinct prediction times")
    validation_start, test_start = times[int(len(times) * .6)], times[int(len(times) * .8)]
    train = [r for r in rows if r.as_of < validation_start and r.label_available_at < validation_start]
    validation = [r for r in rows if validation_start <= r.as_of < test_start
                  and r.label_available_at < test_start]
    test = [r for r in rows if r.as_of >= test_start]
    partitions = [train, validation, test]
    if min(map(len, partitions)) < 2:
        raise ValueError("too few examples remain after purging labels at split boundaries")
    event_sets = [{r.event_id for r in part} for part in partitions]
    if any(event_sets[i] & event_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("the same event crosses partitions; group or deduplicate related examples")
    return partitions, {"train": len(train), "validation": len(validation), "test": len(test),
                        "purged": len(rows) - sum(map(len, partitions)),
                        "validation_start": validation_start, "test_start": test_start}


def write_smoke_data(path: Path, cfg: dict):
    """Known synthetic relationship for plumbing checks, never a real-world dataset."""
    import random

    rng = random.Random(cfg["seed"])
    base = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()

    def iso(seconds):
        return datetime.fromtimestamp(seconds, timezone.utc).isoformat()

    with path.open("x", encoding="utf-8") as stream:
        for i in range(240):
            as_of = base + i * (cfg["horizon_seconds"] + 60)
            features = [[rng.gauss(0, 1) for _ in cfg["feature_names"]]
                        for _ in range(cfg["history_length"])]
            score = features[-1][0] + .4 * features[-2][0] + rng.gauss(0, .1)
            target = int(score > 0) if cfg["task"] == "binary" else score
            row = {"sample_id": f"synthetic-{i}", "event_id": f"synthetic-{i}",
                   "source": "synthetic plumbing fixture; not real-world evidence",
                   "as_of": iso(as_of), "label_available_at": iso(as_of + cfg["horizon_seconds"]),
                   "observed_at": [iso(as_of - (cfg["history_length"] - 1 - j) * 60)
                                   for j in range(cfg["history_length"])],
                   "feature_names": cfg["feature_names"], "features": features, "target": target}
            stream.write(json.dumps(row, allow_nan=False) + "\n")
