# UFC forward evaluation

Owner: Ember forecasting operator. This workflow uses the existing small CPU
models; it does not retrain EmberGPT, launch a paid job, or connect a production
tool. Its purpose is to record predictions before events and score them later.

## Fixed protocol

- Keep the original validation-selected continuous ensemble as the primary
  model. Compare BSQ, hybrid, logistic regression, and Elo on the same bouts.
  Average neural seeds 17, 29, and 43; do not select seeds using future results.
- Primary score: log loss. Also report Brier score and winner accuracy. Small
  differences in winner accuracy alone do not establish useful probability quality.
- Issue one initial batch per card between seven days and one day before its
  **UTC calendar event date**. These deliberately conservative date cutoffs
  do not attempt to infer the actual local start time.
- Build features at the actual issuance time. Historical bouts become usable
  strictly more than two days after their UTC event date. This differs from the
  retrospective benchmark's fixed nominal one-day-before prediction time;
  report forward results separately.
- Both fighters need two earlier usable UFC bouts. Use stable fighter-profile
  URLs and verify their names. Record sparse/unknown matchups as exclusions.
- Include every eligible bout on the verified card. Do not pick favorable bouts
  after examining probabilities, drop losses, or replace an archived forecast.
- Exclude draws, no contests, cancellations, changed opponents, and rescheduled
  bouts with explicit reasons. Keep unreported results pending.
- Do not refit weights, tune thresholds, or promote a model from this workflow.
  Review probability quality and coverage over multiple new cards before making
  an integration decision. The score output is descriptive, not a promotion gate.

## 1. Freeze the existing benchmark

Use the saved `ufc-benchmark` directory from the first historical run. It contains
`benchmark.json`, `logistic.pt`, and the three seed directories with their models.
The code records the validation selection, checkpoint hashes, training dataset
hash, inference-source hashes, and registration time. It verifies the existing
checkpoint feature ordering, seed, representation, and symmetry configuration.

```bash
python -m jobs.ember_ufc_forward register --benchmark-dir /path/to/ufc-benchmark --output registry.json
```

The checked-in `config/ember_ufc_forward_registry.json` records the original
saved models. Use it with those exact assets and the matching inference code;
changed model files or inference code require a new explicitly identified
registration. Do not replace the existing registry.

## 2. Verify and archive a future card

Obtain the complete scheduled card from UFC/UFCStats, confirm fighter identities
against the historical source, and record when it was checked. The schedule must
be checked within 24 hours of generating predictions. Review late replacements
and withdrawals. A source URL and user-entered timestamp are provenance claims;
the program does not independently verify the page.

Save a JSON object shaped like this (placeholders, not a real card):

```json
{
  "source_url": "https://www.ufc.com/event/VERIFIED-EVENT",
  "captured_at": "YYYY-MM-DDTHH:MM:SS+00:00",
  "bouts": [{
    "fight_id": "STABLE-FIGHT-ID",
    "event_id": "STABLE-EVENT-ID",
    "event_date": "YYYY-MM-DD",
    "fighters": [
      {"id": "http://ufcstats.com/fighter-details/FIRST-ID", "name": "Exact fighter name"},
      {"id": "http://ufcstats.com/fighter-details/SECOND-ID", "name": "Exact fighter name"}
    ]
  }]
}
```

The schedule has no winner, target, outcome, odds, or fight-statistics fields.
The program rejects unexpected fields, duplicate bouts/fighters, conflicting
card dates, stale schedules, and fights already present in the result source.

Refresh and validate the four UFCStats CSV inputs using the existing adapter.
Archive the source files and source revision. Do not silently use present-day
career averages or assume a stale export contains all prior fights. The source
may contain later corrections; original publication timing remains unverified.

```bash
python -m jobs.ember_ufc_forward predict --registry registry.json --benchmark-dir /path/to/ufc-benchmark --source-dir /path/to/ufc-source --schedule verified-card.json --output first-card-forecast.json
```

The output contains the input schedule, source hashes, feature vectors, all model
probabilities, exclusions, and registry. Existing output files are never
overwritten. It rechecks model/source hashes and the time window after inference.

**Archive the registry, complete source snapshot, and forecast in an independently
timestamped location before the event.** A GitHub commit made before the event
can provide an external timestamp for the committed files. Keep a receipt/link.
Local hashes catch changed content but are not signatures and do not prove when
it was created. The local clock can be changed. The scorer cannot verify an
external timestamp or detect selection between multiple separately saved batches;
the first archived complete-card forecast is the one that counts. Do not call a
historical replay, a synthetic fixture, or an unanchored file a verified forward test.

## 3. Score results later

After the event-date-plus-two-days cutoff, collect verified results separately:

```json
[{
  "fight_id": "STABLE-FIGHT-ID",
  "event_id": "STABLE-EVENT-ID",
  "event_date": "YYYY-MM-DD",
  "fighter_ids": ["FIRST-ID", "SECOND-ID"],
  "winner_id": "FIRST-ID",
  "status": "decisive",
  "observed_at": "YYYY-MM-DDTHH:MM:SS+00:00",
  "source_url": "https://www.ufcstats.com/event-details/VERIFIED-ID"
}]
```

Use the exact IDs saved in the forecast. Other statuses are `draw`, `no_contest`,
`cancelled`, and `matchup_changed` (including a rescheduled bout). A changed
matchup is excluded, never scored as though it were the original prediction.
The source URL documents the operator's verification; the scorer does not fetch it.

```bash
python -m jobs.ember_ufc_forward score --forecast first-card-forecast.json --outcomes verified-results.json --output first-card-scores.json
```

Scoring needs no model inference. It checks the forecast checksum, result timing,
fight/card/date/participant matching, and duplicates. Missing outcomes remain
pending. Corrections get a new result/report file with a documented reason;
the original forecast stays unchanged. Aggregate every eligible card rather than
reporting only favorable cards. No bankroll or betting-return claim is produced.

## Definition of done

- Registry matches the original saved models and inference implementation.
- Tests cover inference, temporal separation, asset changes, exclusions, and scoring.
- Actual card and input-source snapshots are independently verified and archived.
- Predictions have an external pre-event timestamp.
- Later results are scored with pending/excluded bouts visible.

The first two are preparation. The remaining steps require a verified future
card and its eventual results; software tests alone do not complete them.
