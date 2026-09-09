# Ember v0.0.51 placement dose-response ladder

## Why a ladder and not error-focused placement learning

The obvious next move after v0.0.50 was to reshape the placement loss toward the
positions that are wrong. The evidence says the problem is gradient *magnitude*,
not shape, so shaping it first would have taught us nothing.

| | v0.0.48 | v0.0.50 |
| --- | ---: | ---: |
| Placement loss weight | 0.50 | 0.10 |
| Learning rate | 1.2e-6 | 1e-7 |
| Steps | 120 | 40 |
| Competing preservation weight | none | 0.80 |
| Placement token gain | +33 pts | **0.0** |

Roughly two orders of magnitude less placement-specific movement. Exactly zero is
what that predicts. Reshaping a gradient that small still yields nothing, and the
result would not distinguish "error-focusing does not work" from "nothing was
going to move at this size."

There is also a specific hazard in shaping first. The natural target is the first
token after the value-free prefix, which the probe check measured at 3/24 (12.5%)
against 85/146 (58.2%) for later tokens. But `exact` requires every position, and
later positions are frequently wrong too -- several development cases miss four to
nine tokens. Over-weighting the boundary could buy first tokens and lose
continuation, netting zero exact gain and looking identical to failure. Designing
that weighting needs positional data from a run where placement actually moved,
and no such run exists under protection.

## What the ladder measures

Four rungs, each restarting from the untouched **v0.0.31 step 479**, running the
same protected recipe at a different placement update size:

| Rung | Learning rate | Note |
| ---: | ---: | --- |
| 0 | 1e-7 | v0.0.50's own rate: the control |
| 1 | 4e-7 | |
| 2 | 1.6e-6 | above v0.0.48's 1.2e-6 |
| 3 | 6.4e-6 | |

The learning rate is the **only** thing that varies. Step count stays at
v0.0.50's 40, and the configuration refuses to load unless the development-set
counts, the generation budget and the step count all match v0.0.50 exactly, so
rung 0 is a true control: it should reproduce the measured 0 exact / 0.0% token
gain. If it does not, the ladder is not measuring what v0.0.50 measured and
nothing above it can be trusted.

**Entry is dropped entirely.** It reached +6 under v0.0.50's most conservative
recipe, so it needs no further budget, and its gradient only competes with
placement here. Its 0.10 weight passes to placement, which the config asserts:
placement 0.20, tool KL 0.55, copy KL 0.25. Preservation must still dominate.

## Instrumentation

Each rung reports, beside its placement gain:

- **teacher KL** for the tool and copy corpora, against the same 0.08 budget
  v0.0.50 used and left roughly eight times unspent;
- **relative parameter drift** from the source, maximum and mean;
- **gradient balance at step 0** -- the weighted placement gradient norm beside
  the weighted preservation gradient norm, measured with separate backward passes
  and no optimizer step. This is the instrument that says which rung *should*
  have mattered, rather than leaving the rung spacing a guess;
- a checkpointed trajectory at step 20 and 40.

The familiar 90 cases, the four historical controls and the copy diagnostic run
**once** on the selected rung, and stay evaluation-only throughout. The selected
rung is re-run from the pristine source before that evaluation; the loop is
deterministic and a test pins that re-running a rung reproduces it exactly.

## Selection and outcome

A rung is a *candidate* when it both moved placement (at least one exact case and
3% of tokens, the unrelaxed v0.0.50 thresholds) and preserved (both teacher KLs
inside budget). The reported rung is the best candidate; failing that, the largest
update that still preserved, because that rung's familiar-90 evidence bounds how
much update this recipe can absorb.

This is a diagnostic, not a promotion canary, so it **exits zero either way** and
states plainly whether an operating point was found. A ladder where nothing moves
is a complete result. That is a deliberate departure from v0.0.48 and v0.0.50,
whose non-zero exit on a scientific FAIL reads as a broken job in the Actions UI.

## How to read it

- **Placement moves at some rung with KL inside budget** → the operating point
  exists and the ladder locates it. Error-focusing then becomes a targeted
  efficiency lever with data behind it: the positional breakdown from that rung
  says whether the boundary token is the binding constraint.
- **Placement moves only where KL blows past budget** → placement and the
  preserved behaviour are entangled at this scale, and no further tuning of a
  full-parameter update will separate them. The next mechanism is a
  parameter-subset update -- a single block, or adapter-style -- not more search.
- **Nothing moves at any rung, KL intact throughout** → the ladder did not reach
  far enough, and the step-0 gradient balance says by how much, since it measures
  how small the placement gradient is beside the preservation gradient.

The bar for detection is low: the probe check measured 12.4% of token headroom
against 3% required, five token flips out of 170 to register a gain, and three
development cases sitting one token from exact.

No GPU training, promotion, deployment, or production integration is possible in
this runner.
