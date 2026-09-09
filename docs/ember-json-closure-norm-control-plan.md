# JSON-closing norm-matched control

## The question

The projection arm (run `34387531160`) lost learning against the unprotected
baseline: exact placement 2/24 to 1/24 and placement tokens 95/170 to 93/170.
It also shrank every update. Retained proposal norm averaged 99.5421% and ranged
99.1917% to 99.6508%, because removing the closure components from a proposal can
only shorten it.

Compounded over 40 steps, a 0.46% mean shrinkage is `0.9954^40 = 0.83`, roughly
17% less cumulative update magnitude. The learning deficit is +5/170 tokens
against the baseline's +7/170, about 29% less gain. Those are the same order, so
the projection arm cannot presently distinguish

- removing the closure directions suppressed learning, from
- taking 17% less total step suppressed learning.

## The control

One arm, identical to the projection run in every other respect: same pinned
v0.0.31 step-479 source, same SHA-pinned ladder config, seed, batch schedule,
learning rate, objective weights, same fresh anchor and holdout cohorts, same
probe steps.

At each step the optimizer proposal is scaled by the fraction the projection
would have retained on that same proposal, and applied in its original direction:

    scale = ||project(raw)|| / ||raw||
    theta <- theta + scale * raw

The scale is recomputed here rather than replayed, because the projection run's
per-step fractions were not published — only their mean, min and max. The
realized fractions are recorded so the two magnitude profiles can be compared
against that published range.

The arms therefore differ in exactly one thing: whether the closure directions
are removed, and the endpoint gate checks exactly that. Each step records the
fraction of the applied update lying inside the basis: the projection retains
none of it (its own bound is 1e-3, and it measured 9.2e-5), while the control
must retain it, so the gate requires at least half of the `sqrt(1 - scale^2)`
that the raw proposal carried. Magnitude is gated separately, within 10% of the
intended fraction.

Both tolerances sit above a noise floor worth recording. At a learning rate of
1e-7 the per-element update is near float32 resolution, so assigning it to the
parameters rounds away on the order of a percent of the update — in both arms
alike. The first version of this control gated on the cosine between the applied
and proposed updates and aborted at step 1 with cosine 0.9921. That guard was
unusable in principle, not merely too tight: removing 8 directions out of
millions leaves a cosine of about 0.995, so rounding noise and the intervention
are indistinguishable in that quantity. The basis fraction separates them by
three orders of magnitude.

## Reading the outcome

- Control lands near the projection arm (about 1/24 exact, about 93/170 tokens):
  the learning loss is the shrinkage, not the direction removal. Closure
  protection is then cheap in learning terms, and the projection should be
  retried with the step size scaled back up.
- Control lands near the unprotected baseline (2/24, 95/170): the shrinkage is
  incidental and removing the closure directions is what costs learning. The
  approach then has a real learning cost to weigh against its retention gain.

Retention is worth reading either way. If the control also holds 8/8 short codes
at steps 13 and 23, the retention gain attributed to closure protection was also
just the smaller step.

## Scope

CPU only, one 40-step trajectory, bounded at 1800 s. No GPU training, checkpoint
export, promotion, deployment or integration. Pristine restore and frozen-teacher
hashes verified at exit.
