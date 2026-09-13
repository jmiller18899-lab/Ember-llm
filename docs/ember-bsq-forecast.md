# Experimental BSQ forecasting component

This adds a separately trainable numeric forecasting component to Ember's
repository. It is an offline CPU experiment, not a modification to EmberGPT's
text tokenizer, INT4 export, tool router, or protected checkpoints. It does not
yet register a forecast tool in ClawAgent or train Ember to call that tool.

The first domain-specific run is documented in [UFC winner forecasting](ember-ufc-forecast.md).
That adapter uses a single matchup snapshot, fixed evaluation dates, and an
optional binary symmetry constraint for difference features. The generic
runner's defaults remain the original chronological split and unconstrained output.

The reason for this boundary is practical: previous Ember experiments recorded
a tradeoff between new learning and retained tool behavior. This component lets
us evaluate forecasting before choosing a connection to the language model.

## Architecture and comparison

`jobs/ember_bsq.py` learns an encoder for numeric observations. The BSQ layer
normalizes the latent vector, assigns each coordinate to +/-1/sqrt(bits), and
uses a straight-through gradient during training. A GRU processes the history
and predicts either one binary event probability or one continuous value.

The trainer compares three modes, each with a fresh model and the same seed,
training rows, step budget, and validation selection rule:

- `continuous`: learned continuous features;
- `bsq`: quantized features;
- `hybrid`: quantized features plus the original standardized measurements.

The hybrid mode has additional GRU input parameters; the report includes model
sizes. This is an exploratory comparison, not an exact parameter-matched study.
There is also a constant baseline fitted only on the training labels.

The objective combines forecast error, reconstruction error, commitment, and
an approximate bit-entropy term. Binary tasks use log loss; regression tasks
use MSE with target normalization learned on the training partition. Test
regression scores and predictions are returned in original target units.
The implementation is inspired by [the original BSQ paper](https://arxiv.org/abs/2406.07548).
[Kronos](https://arxiv.org/abs/2508.02739) provides a forecasting precedent;
this small experiment is not a reproduction of Kronos.

## Run a plumbing check

Use Python 3.11 and the same CPU PyTorch version used by existing Ember jobs:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
python -m pip install pytest==8.3.5 numpy==2.2.6
python -m pytest -q tests/test_ember_bsq_forecast.py
python -m jobs.ember_bsq_train --smoke --output-dir /tmp/ember-bsq-smoke
```

Choose a new output directory for each run. The command writes `report.json`
and `continuous.pt`, `bsq.pt`, and `hybrid.pt`. Synthetic data tests plumbing
and learnability only. Even a good score is not evidence of real-world skill.
No job submission, network collection, GPU launch, or promotion occurs.

## Real data and scraping

A scraper is optional. Start with historical JSONL files prepared from APIs,
datasets, exports, or logs. A collector must pair observations available when
the prediction would be made with the actual outcome measured afterward.
Fetching current pages alone does not create a valid historical training set.

Copy `config/ember_bsq_forecast.json` and set the target, binary/regression task,
forecast horizon, ordered feature names, and history length. Every JSONL row
must have this shape (illustrative values, not a training dataset):

```json
{
  "sample_id": "task-123-at-1000",
  "event_id": "task-123",
  "source": "runtime-log-export-2026-09",
  "as_of": "2026-09-01T10:00:00Z",
  "label_available_at": "2026-09-01T11:00:00Z",
  "observed_at": ["2026-09-01T09:57:00Z", "2026-09-01T09:58:00Z", "2026-09-01T09:59:00Z", "2026-09-01T10:00:00Z"],
  "feature_names": ["tool_latency_ms", "retry_count"],
  "features": [[120, 0], [130, 0], [800, 1], [1200, 2]],
  "target": 1
}
```

The configuration for that example would describe a precisely defined task
failure event within 3600 seconds and use those two feature names. Define what
counts as a failure before collecting labels. `observed_at` must describe when
the inputs were available, including publication delays and later revisions.
`label_available_at` must be at or after the entire forecast horizon.
Related snapshots of the same task, fight, or outcome share an `event_id`.
Choose a consistent sampling cadence, or include elapsed time as an explicit
feature. Preprocess missing values explicitly and add missingness indicators
where useful; this experiment rejects missing and non-finite numeric values.

```bash
python -m jobs.ember_bsq_train --config /path/to/forecast.json --data /path/to/examples.jsonl --output-dir /tmp/ember-bsq-real-001
```

The loader checks declared timestamps and feature order. It cannot prove that
a value inside a feature was not computed using future information; audit the
collector and transformations separately. The 60/20/20 chronological split
keeps tied prediction times together, purges training/validation examples whose
labels are unavailable before the next partition, and rejects related events
that cross partitions. Normalization uses only retained training rows.

Checkpoint selection uses validation forecast loss only. The test partition
is evaluated afterward. Reported probabilities are uncalibrated estimates;
log loss, Brier score, and reliability bins help inspect them. This initial
regression head produces point estimates, not uncertainty intervals.

## Use a trained experimental component

```python
from pathlib import Path
from jobs.ember_bsq_train import predict_checkpoint

result = predict_checkpoint(
    Path('/tmp/ember-bsq-real-001/hybrid.pt'),
    [[120, 0], [130, 0], [800, 1], [1200, 2]],
    ['tool_latency_ms', 'retry_count'],
)
```

Choose the first real target and dataset, run the comparison across fresh time
periods and several seeds, and check whether BSQ adds value. Adopt a mode using
validation evidence, reserving new test periods for final confirmation. A later
integration can expose a `forecast` tool and teach Ember to call it and report
its result. That requires tool examples and Ember's existing semantic and
regression gates. Updating the core text model with a BSQ loss would be a
separate architecture experiment. Web collection belongs in the data pipeline;
it is not a capability the neural model itself needs to implement.
