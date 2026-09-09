# Ember rung-0 trajectory diagnostic

The first-update diagnostic preserved the source-passing familiar cases, but its
data and objectives differed from the ladder. The ladder's source is now available
at `ea3fe6a9f1ccaac59a91e5c6b3e8ebee70700e1c`. This experiment asks when its known
short-code losses first occur relative to its placement-learning threshold.

## Fixed experiment

- Start from pinned, SHA256-verified v0.0.31 step 479, CPU float32, two threads.
- Reproduce rung 0's original data, deterministic batch sequence, AdamW optimizer,
  LR 1e-7, 40 steps, gradient clip 0.25, weight decay 0.01, and objective weights
  0.20 placement / 0.55 tool KL / 0.25 copy KL. Entry weight remains zero.
- Preserve the previous source-level replay fix; it produces the same values as
  the ladder's compatibility shim. No hyperparameter or data change is tested.
- Before optimization, fix all batches and require the original 84/90 familiar,
  4/4 reference and 1/24 exact, 88/170 token placement baselines.
- After every update, observe all ten familiar short codes and all 24 development
  placement cases. Record generated responses, source-passing case IDs, and
  margins at the first changed token for lost cases.
- Run full familiar/copy/reference/KL guards at steps 1 and 40 and at the first
  step clearing both original placement-learning thresholds, at most three states.
  Evaluation never changes the schedule, gradients, run duration, or parameters.
- Compare the endpoint to published rung 0: 82/90 familiar; 2/24 exact placement;
  95/170 placement tokens; tool/copy KL within 1e-5 of published values; only
  expanded-case retention failing the historical copy guard.
- Verify model hashes around evaluation, restore RNG state after each observation,
  and verify pristine restore and unchanged teacher at completion. Limit script
  runtime to 30 minutes and workflow to 40 minutes including setup.
- Save report/data/summary artifacts, including coverage by kind and subtype.
  No checkpoint export, promotion, GPU job, or ClawAgent integration.

## Interpretation limits

The first short-code failure is measured at every step. Other familiar/copy
failures are observed only at the specified full-evaluation states. A learning
state with all short codes retained requires the full guards to establish an
operating point. Any such state not fully evaluated is explicitly listed as
unverified; it cannot support a claim that all possible windows were ruled out.

This is a diagnostic replication on previously used development data. A passing
state would still need a separate fresh evaluation before promotion.

## Corrections to the earlier gradient interpretation

KL to an identical frozen teacher and a squared L2 penalty to identical source
weights both have zero mathematical gradient at initialization. Float32 residuals
are not a nonzero preservation mechanism. A parameter projection can restrict an
actual optimizer proposal; it does not imply that a uniform trust-region scaling
changes its direction. The first-update margin projection does restrict direction.

A ratio of gradient norms is not a percentage of the AdamW update. Matching a pair
of norms also does not prove identical model or optimizer state. This diagnostic
uses direct state hashes and verifies the original batch and endpoint numerically.

The final short-code losses do not establish that damage occurred at step 1 or
prove inadequate distillation coverage caused it. The familiar long-code score
was 7/10, below short-code's 8/10; short codes were the kind that regressed.
The original ladder evaluated full regression guards only on its selected rung,
so the report did not rule out every unselected rung or intermediate state.
