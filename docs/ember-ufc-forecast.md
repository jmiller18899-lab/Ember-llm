# Ember UFC winner forecasting: first retrospective experiment

The first real task is pre-fight winner probability, conditional on the bout
having a decisive result. This is a separately trained forecasting component;
EmberGPT's language-model weights and the live ClawAgent tool registry are not
changed. It does not yet predict finish method, round, or betting returns.

## Data

Source: [Greco1899's UFCStats CSV exports](https://github.com/Greco1899/scrape_ufc_stats),
revision `44a4022696135ddcfd1536b100ff3e909209dc3d`.
The source contains 8,899 result rows through September 5, 2026. The adapter
removes 25 repeated fight IDs, resolves compatible event-name aliases, and
excludes ambiguous fighter identities and unsupported historical round formats.
It uses fighter-profile URLs for identity only, not present-day career averages.

There are 8,509 usable historical bouts and 4,816 eligible prediction examples.
Both fighters must have at least two earlier usable UFC bouts. Draws and no
contests are excluded as prediction targets. Earlier draws/no contests may
contribute historical statistics; their result score is neutral (0.5), and no
contests do not update Elo. Thus smoothed result-rate features are not literal
official win/loss records.

The 20 features are differences between the fighters' prior Elo, experience,
smoothed result rates, recent finish history, striking rates and defense,
takedowns, control, submission attempts, fight duration, layoff, and prior
opponent strength. Performance statistics use the latest five usable bouts;
experience, cumulative result rate, and Elo use the earlier usable history.
Missing control-time coverage is represented explicitly.

Fighter A is assigned by fighter URL order, independent of the winner or source
row order. Neural predictions are constrained so that reversing every matchup
difference reverses the probability: P(A over B) = 1 - P(B over A).

## Historical availability and evaluation

UFC event dates do not provide original publication timestamps. The experiment
uses a declared conservative convention: make the forecast one day before the
event date, and treat a past card's outcomes/statistics as usable two days after
its event date. Inputs only use bouts whose assumed availability is strictly
earlier than that forecast. No result from the same card enters its features.
The 72-hour config horizon reflects this date convention, not a 72-hour fight.

This is retrospective reconstruction. The source may contain later corrections,
and original publication times have not been independently archived or verified.
The loader can check the declared chronology; it cannot establish that the source
was identical at each historical cutoff. It also lacks regional fight histories,
historical injuries, odds, rankings, and weigh-in information.

The split was fixed before training:

| Partition | Prediction dates | Bouts |
| --- | --- | ---: |
| Train | Before January 1, 2022 | 3,270 |
| Validation | January 1, 2022 through December 31, 2023 | 645 |
| Test | January 1, 2024 onward, through the source cutoff | 901 |

Whole cards stay in one partition. Labels unavailable before the next
partition are purged (none crossed the cutoffs in this snapshot). Feature
normalization uses only training rows. Historical fighter states may incorporate
earlier validation/test bouts when predicting a later bout, as an operational
model could; model weights are not updated using those labels.

Each neural representation trains for at most 300 CPU steps with fixed seeds
17, 29, and 43. Checkpoints are selected using validation log loss. The three
seeds are averaged into one probability per bout. The continuous, BSQ, and
hybrid models use the same data, step budget, and selection rule. The hybrid
has additional input parameters. Elo and a regularized logistic regression
serve as simpler baselines. Test results select neither hyperparameters nor
the preferred representation.

## First results

| Model | Correct / 901 | Accuracy | Log loss | Brier score |
| --- | ---: | ---: | ---: | ---: |
| Continuous features | 565 | 62.7% | 0.64399 | 0.22632 |
| BSQ | 573 | 63.6% | 0.64679 | 0.22743 |
| Hybrid | 568 | 63.0% | 0.64465 | 0.22654 |
| Logistic regression | 550 | 61.0% | 0.64826 | 0.22839 |
| Elo | 494 | 54.8% | 0.68243 | 0.24472 |
| Constant training-label frequency | 445 | 49.4% | 0.69342 | 0.25014 |

Lower log loss and Brier score are better. The continuous model was selected
by validation log loss (0.65778, versus BSQ 0.66447), and had slightly better
test probability scores. BSQ picked eight more winners at the 0.5 threshold.
This does not establish a general advantage for BSQ.

The BSQ-minus-continuous mean test log-loss difference is +0.00280. A paired
bootstrap resampling 114 whole cards 1,000 times gives an approximate 95%
interval of [-0.00224, +0.00762]. It crosses zero. This interval describes
variation across these cards; it does not capture all model-selection,
source-quality, or future distribution uncertainty.

Twenty focused tests pass, including current-outcome/statistics exclusion,
same-day isolation, result-independent fighter ordering, duplicate handling,
fixed date splits, and checkpoint probability reversal. This is historical
forecast evidence, not a live evaluation or a deployment gate.

## Reproduce

From the Ember repository root, with the CPU dependencies from the BSQ runbook:

```bash
git clone https://github.com/Greco1899/scrape_ufc_stats.git /tmp/ufc-source
git -C /tmp/ufc-source checkout 44a4022696135ddcfd1536b100ff3e909209dc3d
python -m jobs.ember_ufc_data --source-dir /tmp/ufc-source --source-revision 44a4022696135ddcfd1536b100ff3e909209dc3d --output-dir /tmp/ufc-data
python -m pytest -q tests/test_ember_bsq_forecast.py tests/test_ember_ufc_forecast.py
python -m jobs.ember_ufc_benchmark --data /tmp/ufc-data/ufc_examples.jsonl --output-dir /tmp/ufc-benchmark
```

Choose new output paths for reruns. The data builder records source-file hashes
and the dataset hash, `8198c839f58e7b198e6442114e71dfcb00fa06a75278ed2aae23a41322917d1f`.
The benchmark saves each checkpoint, per-seed reports, validation/test scores,
reliability bins, and held-out bout probabilities. No scraper execution or paid
training job is needed for this initial dataset.

The next useful check is a pre-registered forward evaluation on newly scheduled
bouts, recording probabilities before results exist. Validate source updates,
calibration, and inference inputs before connecting a forecast tool to Ember.

The [forward-test workflow](ember-ufc-forward-test.md) now freezes the existing
models, records unlabelled future matchups, and scores separately collected results.
Its local receipts need an external pre-event timestamp to qualify as forward evidence.
