# Ember training balance — verified recommendation

**Test punctuation weight 0.05 next**, down from 0.20. Keep placement 0.20,
tool-KL 0.55, copy-KL 0.25 and LR 1e-7. This is a candidate for a full test,
not an established training solution. No full trajectory at 0.05 was launched.

## Evidence used

The completed CPU diagnostic (run 34399461056, 2 minutes 20 seconds) replayed
all 40 updates of the prior 0.20 run and tested five isolated single-update
weights before updates 1, 13, 23 and 40. At each state, every trial inherited
the same model, AdamW history and current training examples.

| Punctuation weight | Copying loss improvement retained versus weight 0, range across four states | Punctuation loss improved at all four states? | Prespecified local screen |
| ---: | ---: | --- | --- |
| 0 | 100% | No | Reference |
| 0.025 | 98.6–99.5% | No | FAIL |
| **0.05** | **96.8–98.0%** | **Yes** | **PASS; smallest eligible weight** |
| 0.10 | 92.3–94.0% | Yes | PASS |
| 0.20 | 79.2–83.9% | Yes | FAIL |

These percentages measure the actual decrease in current-batch placement loss,
not copying accuracy or exact-match gains. The screen, specified before the run,
required at least 90% of weight-zero placement improvement at every sampled
state, no punctuation loss increase beyond 2e-6, and both batch KL losses at or
below 0.08. Both 0.05 and 0.10 passed; 0.05 is the smallest eligible candidate.

At the first update, weight 0.025 increased punctuation loss by 0.00041914.
Weight 0.05 decreased it by 0.00210172 while retaining 98.0% of the placement
improvement. Weight 0.20 retained only 79.2%. At the later sampled states,
0.05 retained 96.8%, 97.4% and 98.0% respectively and decreased punctuation
loss each time.

This establishes a **local measured tradeoff**: at these states and batches,
the larger punctuation weight sacrificed more placement improvement. It does
not establish gradient opposition, the cause of all prior failures, or the
result of training for 40 steps at a different weight.

## Numerical issue and evidence limits

The initial run's float32 full-vector reductions produced inaccurate direction
metrics: a zero-weight update compared with itself had cosine about 0.984,
which should be 1. All vector norms, cosines and first-order predictions from
that run are therefore excluded from this recommendation. The recommendation
uses actual forward-evaluated before/after losses, which do not depend on those
reduction metrics.

A correction uses chunked float64 accumulation for diagnostic dot products and
norms while preserving float32 optimizer math. Its first repeat (34399992177)
stopped at update 1: the gradient norm differed from the reference by
0.00002861, exceeding the original 0.00002000 absolute replay tolerance.
A scale-aware norm check was added, retaining the original loss checks and
selection rule. The next repeat (34400271461) stopped at update 2 on a norm
difference of 0.00003815. These incomplete repeats are not evidence for a
completed corrected direction analysis. No further tolerance changes or retries
were made. The corrected direction analysis remains incomplete.

The first completed run exactly matched all 40 saved loss and gradient-norm
records, with maximum replay error zero. It reproduced every refreshed training
generation and verified that temporary trials did not change the live model or
optimizer. Source restoration and frozen teacher hashes passed. Its artifact
hash and training snapshot hash were independently verified, and the local
recommendation rule was recomputed from the saved losses.

All 67 tests passed on the completed diagnostic's runner. After the measurement
and replay-check fixes, all 69 code tests passed locally and on the runner;
the later diagnostic failures were runtime replay checks, not unit-test failures.

## Next full test

The proposed configuration records **0.20 / 0.55 / 0.25 / 0.05** for placement,
tool-KL, copy-KL and punctuation respectively. It is not automatically loaded
by training. A future 40-update CPU test should start from the pristine source,
use that weight from update 1 with its own optimizer history, and retain every
original learning, individual-case preservation, copy, reference and KL gate.
Separate fresh evaluation examples must remain outside training.

All sampled states here inherited the previous 0.20 optimizer history. Thus,
0.05 has **not** demonstrated preservation of all 84 cases or better exact
copying after a full run. No held-out evaluations, model export, promotion,
GPU training or ClawAgent integration occurred during this diagnostic.

## Files and provenance

- [Completed loss diagnostic](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34399461056)
- [Full initial artifact](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34399461056/artifacts/10122948528)
- [Verified scalar-loss evidence](../reports/ember-training-balance-verified.json)
- [Proposed balance](../config/ember_training_balance_recommendation.json)
- [Protocol and correction record](ember-training-balance-plan.md)

Evaluated initial commit: `9d1ab300a90852f2f237308af862bb8a6cf7f5a4`.
Initial artifact ZIP SHA256:
`eaa4754a3c5ceef8b766cdd08ed89d85fff40ed18b4ac4e93208fe3d174b9df4`.
