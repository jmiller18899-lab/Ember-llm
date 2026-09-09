# Ember v0.0.51 placement dose-response ladder — result

CPU run `34315691408`, job `102351312730`, commit `ee6c1e239aa82c79575f05631638025e14714680`.
Artifact `ember-v051-ladder-34315691408`, ID `10090502469`. Source: untouched
v0.0.31 step 479 for every rung. No GPU, promotion, deployment, or integration.

## The frontier

| Rung | Learning rate | Exact gain | Token gain | Tool teacher KL | Preserved | Candidate |
| ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| 0 | 1e-7 | **+1** | **+4.12%** | 0.000753 | yes | yes |
| 1 | 4e-7 | +3 | +15.88% | 0.010252 | yes | yes |
| 2 | 1.6e-6 | +6 | +29.41% | 0.035103 | yes | yes |
| 3 | 6.4e-6 | +6 | +32.35% | 0.030052 | yes | yes |

Required to clear the bar: 1 exact case and 3% of tokens. Teacher KL budget 0.08.

**Placement moves at every rung, and every rung stays inside the KL budget.**
The gain is monotone in learning rate and saturates between rung 2 and rung 3 at
+6 exact of a possible 23, while KL does not even reach half its budget.

Rung 3's trajectory shows the movement is not a late artefact: +6 exact and
+30.6% tokens by step 20, +32.4% by step 40, with maximum relative parameter
drift growing 0.00765 → 0.00984.

## Rung 0 contradicts v0.0.50, and that is the finding

Rung 0 runs at **v0.0.50's own learning rate** and gained +1 exact and +4.12% of
tokens. v0.0.50 measured exactly 0 and 0.0% at that rate, four checkpoints
running.

Two things differ between them, and neither is the learning rate:

- v0.0.50 carried an **entry objective at weight 0.10** competing for the same
  bounded update. v0.0.51 drops it entirely.
- placement's weight **doubled**, 0.10 → 0.20, inheriting what entry gave up.

So v0.0.50's zero was not a learning-rate floor. Placement was starved of its
share of the gradient by an objective that had already finished learning. The
gate v0.0.50 failed was clearable at v0.0.50's own update size.

### A limitation of this control, stated plainly

Rung 0 is a control for the *learning rate*, not for the whole recipe, and the
design document overstated it as "a true control". It cannot be a strict
replication of v0.0.50 while also dropping entry — those are incompatible by
construction. The seed also differs (2026090951 against 2026090950), so the
minibatch draws are not identical. The weight change is by far the most likely
explanation for a 0 → +1 exact and 0.0% → 4.12% swing, but this run does not
isolate it from sampling on its own.

## Why `operating_point_found` is nevertheless false

The selector chose rung 3 — the largest gain among preserving rungs — and the
final evaluation-only battery on that rung did not clear its gates.

That is a defect in the selector, not in the ladder. It maximises placement gain
among rungs that preserve *by teacher KL*, when the phase's actual goal is the
**smallest** update that clears the learning bar. It therefore selected the rung
with maximum parameter drift (0.00984 max relative, roughly fifty times the
v0.0.49 trust region) rather than rung 0, which cleared the same bar at a
hundredth of the KL cost.

The run also demonstrates something worth keeping: **teacher KL inside budget did
not guarantee the familiar battery held.** KL on the distillation corpora is a
useful cheap proxy, but it is not a substitute for the 90-case evaluation, and a
rung can satisfy one while failing the other.

## Two reporting gaps in this runner

Both are omissions in `jobs/ember_v051_ladder.py`, not in the experiment:

1. the final evaluation-only numbers — familiar 90, references, copy protection —
   are written to `report.json` and `summary.md` but never printed to stdout, so
   the reason `operating_point_found` is false cannot be read from the logs;
2. the step-0 gradient balance is recorded per rung in the report but is likewise
   absent from the `rung_complete` event, so the instrument built to explain the
   rung spacing cannot be read from the logs either.

## What follows

The cheap, decisive next step is to evaluate the **smallest** sufficient rung
rather than the largest. Rung 0 clears the learning bar at a tool KL of 0.000753
— a hundredth of the budget and below even v0.0.50's 0.0103 — so it is the
strongest candidate to hold the familiar 90. Rung 1 is the fallback if rung 0
does not.

That requires fixing the selector to prefer the smallest clearing rung, printing
both missing blocks to stdout, and re-running. It does not require another
ladder: the frontier is now known.

Error-focused placement learning is no longer the interesting question either.
Placement moves readily once it is given gradient share; the open question is how
little update is needed to keep the familiar battery intact while it does.
