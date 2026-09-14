"""Reconstruct UFC matchup features using only prior fight results/statistics.

Reads Greco1899/scrape_ufc_stats CSV exports without executing its scraper code.
Outcome corrections and source publication times are not archived: this is a
retrospective benchmark with explicit conservative date assumptions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
from pathlib import Path
import csv
import re

DAY = 86400
SOURCE_FILES = ("ufc_event_details.csv", "ufc_fight_results.csv", "ufc_fight_stats.csv", "ufc_fighter_tott.csv")
FEATURE_NAMES = ["elo_diff", "log_prior_bouts_diff", "win_rate_diff", "recent_win_rate_diff",
                 "finish_win_rate_diff", "ko_loss_rate_diff", "submission_loss_rate_diff",
                 "sig_landed_per_min_diff", "sig_absorbed_per_min_diff", "sig_accuracy_diff",
                 "sig_defense_diff", "takedowns_per_15min_diff", "takedown_accuracy_diff",
                 "takedown_defense_diff", "control_share_diff", "control_missing_share_diff",
                 "submission_attempts_per_15min_diff", "duration_minutes_diff", "layoff_log_days_diff",
                 "prior_opponent_elo_diff"]


def clean(value):
    return " ".join(value.split())


def records(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [{k: clean(v) for k, v in row.items()} for row in csv.DictReader(stream)]


def date_seconds(value):
    return datetime.strptime(value, "%B %d, %Y").replace(tzinfo=timezone.utc).timestamp()


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def clock(value):
    minute, second = map(int, value.split(":"))
    if minute < 0 or not 0 <= second < 60:
        raise ValueError("invalid round clock")
    return minute * 60 + second


def pair(value):
    landed, attempted = map(int, value.split(" of "))
    if not 0 <= landed <= attempted:
        raise ValueError("invalid landed/attempted counts")
    return landed, attempted


@dataclass
class Bout:
    fight_id: str
    event_id: str
    event_date: float
    fighters: tuple[str, str]
    fighter_ids: tuple[str, str]
    outcome: str
    method: str
    duration: float
    stats: tuple[dict, dict]


def load_bouts(source_dir: Path):
    audit = Counter()
    events = {r["EVENT"]: r for r in records(source_dir / SOURCE_FILES[0])}
    identities = defaultdict(set)
    for r in records(source_dir / SOURCE_FILES[3]):
        identities[r["FIGHTER"]].add(r["URL"])
    stats = defaultdict(list)
    for r in records(source_dir / SOURCE_FILES[2]):
        stats[(r["EVENT"], r["BOUT"])].append(r)
    results = defaultdict(list)
    for r in records(source_dir / SOURCE_FILES[1]):
        results[r["URL"]].append(r)
    audit["source_result_rows"] = sum(map(len, results.values()))
    audit["duplicate_result_rows"] = audit["source_result_rows"] - len(results)
    bouts = []
    for fight_id, variants in results.items():
        signatures = {tuple(sorted((k, v) for k, v in r.items() if k != "EVENT")) for r in variants}
        if len(signatures) != 1:
            audit["conflicting_result_ids"] += 1
            continue
        candidates = [r for r in variants if r["EVENT"] in events]
        if not candidates:
            audit["missing_event"] += 1
            continue
        dates = {events[r["EVENT"]]["DATE"] for r in candidates}
        if len(dates) != 1:
            audit["conflicting_event_dates"] += 1
            continue
        row = candidates[0]
        event = events[row["EVENT"]]
        event_date = date_seconds(event["DATE"])
        # Modern five-minute rounds avoid ambiguous historic overtime formats.
        if event_date < date_seconds("January 01, 2000") or not re.fullmatch(r"[235] Rnd \(5(?:-5){1,4}\)", row["TIME FORMAT"]):
            audit["excluded_era_or_round_format"] += 1
            continue
        fighters = tuple(row["BOUT"].split(" vs. "))
        if len(fighters) != 2 or any(len(identities[name]) != 1 for name in fighters):
            audit["ambiguous_fighter_identity"] += 1
            continue
        round_rows = {}
        conflict = False
        # Renamed-event aliases may contain the same round rows. Count each once.
        for variant in variants:
            for s in stats[(variant["EVENT"], variant["BOUT"])]:
                key = (s["FIGHTER"], s["ROUND"])
                comparable = {k: v for k, v in s.items() if k != "EVENT"}
                if key in round_rows and round_rows[key] != comparable:
                    conflict = True
                round_rows[key] = comparable
        if conflict:
            audit["conflicting_round_rows"] += 1
            continue
        try:
            rounds = int(row["ROUND"])
            final_time = clock(row["TIME"])
            scheduled = int(row["TIME FORMAT"][0])
            if not 1 <= rounds <= scheduled or not 0 <= final_time <= 300:
                raise ValueError("invalid fight duration")
            duration = 300 * (rounds - 1) + final_time
            if duration <= 0 or len(round_rows) != 2 * rounds:
                raise ValueError("missing or extra round rows")
            totals = []
            for fighter in fighters:
                total = defaultdict(float)
                for number in range(1, rounds + 1):
                    s = round_rows[(fighter, f"Round {number}")]
                    sig_l, sig_a = pair(s["SIG.STR."])
                    td_l, td_a = pair(s["TD"])
                    for key, value in (("sig_l", sig_l), ("sig_a", sig_a), ("td_l", td_l), ("td_a", td_a),
                                       ("sub", float(s["SUB.ATT"]))):
                        if not math.isfinite(value) or value < 0:
                            raise ValueError("invalid numeric statistic")
                        total[key] += value
                    round_time = final_time if number == rounds else 300
                    if s["CTRL"] not in {"--", "---", ""}:
                        control = clock(s["CTRL"])
                        if control > round_time:
                            raise ValueError("control time exceeds round duration")
                        total["control"] += control
                        total["control_known_seconds"] += round_time
                totals.append(dict(total))
            if row["OUTCOME"] not in {"W/L", "L/W", "D/D", "NC/NC"}:
                raise ValueError("unknown outcome")
        except (ValueError, KeyError):
            audit["invalid_or_missing_fight_statistics"] += 1
            continue
        bouts.append(Bout(fight_id, event["URL"], event_date, fighters,
                          tuple(next(iter(identities[f])) for f in fighters), row["OUTCOME"],
                          row["METHOD"], duration, tuple(totals)))
    audit["usable_historical_bouts"] = len(bouts)
    return sorted(bouts, key=lambda b: (b.event_date, b.fight_id)), dict(audit)


@dataclass
class FighterState:
    elo: float = 1500.
    history: list[dict] = field(default_factory=list)

    def vector(self, as_of):
        h = self.history
        recent = h[-5:]
        n = len(h)
        duration = sum(r["duration"] for r in recent)
        total = lambda key: sum(r.get(key, 0.) for r in recent)
        # Symmetric fixed pseudo-counts avoid undefined rates in sparse history.
        accuracy = lambda made, attempts: (total(made) + 1) / (total(attempts) + 2)
        return [self.elo / 400, math.log1p(n), (sum(r["score"] for r in h) + 1) / (n + 2),
                (total("score") + 1) / (len(recent) + 2), total("finish_win") / len(recent),
                total("ko_loss") / len(recent), total("sub_loss") / len(recent),
                60 * total("sig_l") / duration, 60 * total("opp_sig_l") / duration,
                accuracy("sig_l", "sig_a"), 1 - accuracy("opp_sig_l", "opp_sig_a"),
                900 * total("td_l") / duration, accuracy("td_l", "td_a"),
                1 - accuracy("opp_td_l", "opp_td_a"),
                total("control") / max(total("control_known_seconds"), 1),
                1 - total("control_known_seconds") / duration, 900 * total("sub") / duration,
                duration / (60 * len(recent)), math.log1p((as_of - h[-1]["event_date"]) / DAY),
                total("opponent_elo") / (400 * len(recent))]


def update_states(states, bout):
    a, b = [states[identity] for identity in bout.fighter_ids]
    old = [a.elo, b.elo]
    score = {"W/L": 1., "L/W": 0., "D/D": .5, "NC/NC": .5}[bout.outcome]
    expected = 1 / (1 + 10 ** ((b.elo - a.elo) / 400))
    change = 32 * (score - expected) if bout.outcome != "NC/NC" else 0.
    a.elo += change
    b.elo -= change
    for i, state in enumerate((a, b)):
        own_score = score if i == 0 else 1 - score
        record = dict(bout.stats[i])
        record.update({"opp_" + k: v for k, v in bout.stats[1 - i].items()})
        record.update({"score": own_score, "finish_win": float(own_score == 1 and bout.method in {"KO/TKO", "Submission"}),
                       "ko_loss": float(own_score == 0 and bout.method == "KO/TKO"),
                       "sub_loss": float(own_score == 0 and bout.method == "Submission"),
                       "duration": bout.duration, "event_date": bout.event_date, "opponent_elo": old[1 - i]})
        state.history.append(record)


def build_examples(bouts, min_prior_bouts=2):
    states = defaultdict(FighterState)
    pending = []
    examples, excluded = [], Counter()
    for serial, bout in enumerate(sorted(bouts, key=lambda b: (b.event_date, b.fight_id))):
        as_of = bout.event_date - DAY
        # Conservative date-only convention: data usable two days after a card.
        while pending and pending[0][0] < as_of:
            _, _, prior = heapq.heappop(pending)
            update_states(states, prior)
        order = sorted(range(2), key=lambda i: bout.fighter_ids[i])
        first, second = order
        a, b = (states[bout.fighter_ids[i]] for i in order)
        if bout.outcome not in {"W/L", "L/W"}:
            excluded["draw_or_no_contest_target"] += 1
        elif min(len(a.history), len(b.history)) < min_prior_bouts:
            excluded["insufficient_prior_history"] += 1
        else:
            av, bv = a.vector(as_of), b.vector(as_of)
            target = int(bout.outcome.split("/")[first] == "W")
            examples.append({"sample_id": bout.fight_id, "event_id": bout.event_id,
                             "source": "UFCStats retrospective CSV export; two-day availability assumption",
                             "as_of": iso(as_of), "label_available_at": iso(bout.event_date + 2 * DAY),
                             "observed_at": [iso(as_of)], "feature_names": FEATURE_NAMES,
                             "features": [[x - y for x, y in zip(av, bv)]], "target": target,
                             "fighter_a": bout.fighters[first], "fighter_b": bout.fighters[second],
                             "event_date": iso(bout.event_date), "elo_probability": 1 / (1 + 10 ** ((b.elo - a.elo) / 400)),
                             "prior_bouts": [len(a.history), len(b.history)],
                             "latest_input_fight_date": iso(max(a.history[-1]["event_date"], b.history[-1]["event_date"]))})
        heapq.heappush(pending, (bout.event_date + 2 * DAY, serial, bout))
    return examples, dict(excluded)


def build(source_dir: Path, output: Path, source_revision: str):
    bouts, audit = load_bouts(source_dir)
    examples, excluded = build_examples(bouts)
    if not examples:
        raise ValueError("no eligible UFC examples")
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / "ufc_examples.jsonl"
    dataset.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in examples), encoding="utf-8")
    manifest = {"source_repository": "https://github.com/Greco1899/scrape_ufc_stats",
                "source_revision": source_revision,
                "source_sha256": {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest() for name in SOURCE_FILES},
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "audit": audit, "excluded_targets": excluded, "examples": len(examples),
                "target": "fighter A wins, conditional on a decisive result",
                "first_event": examples[0]["event_date"], "last_event": examples[-1]["event_date"],
                "feature_names": FEATURE_NAMES, "min_prior_usable_ufc_bouts_per_fighter": 2,
                "ordering": "fighter URL order, independent of result/corner",
                "time_convention": "prediction one day before event date; labels/statistics usable two days after event date",
                "limitations": ["retrospective source; corrections and true publication times are not archived",
                                "prior history covers usable UFC bouts only; no regional fight history",
                                "excludes debut/sparse-history targets, draws, and no contests",
                                "no historical odds, injuries, rankings, or weigh-in information"]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_dir, args.output_dir, args.source_revision), indent=2))


if __name__ == "__main__":
    main()
