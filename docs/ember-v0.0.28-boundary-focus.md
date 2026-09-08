# Ember v0.0.28 — spending the gradient where sequences can still flip

## What v0.0.27 proved, and what it cost

The worst-position hinge moved its target hard: `sequence_margin_health` went
from −0.8871 to −0.6846 while exact copy did not move a single case. That is a
better problem than any run before v0.0.26 had. The signal is present; it is
being spent in the wrong place.

### How far the failing sequences actually are

`health` is the mean over 90 cases of `clamp(min_gap, −2.0, +1.2)`. The 20 exact
cases have positive `min_gap`, so their clamped contribution is at most 24. That
bounds the rest:

| | health | Σ clamped | failing-70 mean worst margin |
| --- | ---: | ---: | :---: |
| baseline (v0.0.26) | −0.8871 | −79.84 | between −1.483 and −1.141 |
| final (v0.0.27) | −0.6846 | −61.61 | between −1.223 and −0.880 |

The whole run bought +18.23 of clamped margin — **+0.260 per failing case** if
every point of it landed there. The average failing sequence is still about a
full logit short of the decision boundary. Waiting for the average to reach zero
is four more identical runs, optimistically, and nothing says the trend stays
linear.

### The count does not need the average

It needs the sequences *nearest* zero to cross. And v0.0.27's mining was pointed
away from them. Hardness was `relu(sequence_margin - worst_gap)`, which grows
without bound as a row gets worse:

| row's worst margin | v0.0.27 mining priority | can it flip? |
| ---: | ---: | :--- |
| −3.00 | 4.20 | no |
| −0.05 | 1.25 | yes, easily |
| +0.50 | 0.70 | already correct |

Half of every batch went to the top of that column. **The cheapest sequences to
convert were the ones the mining step was least likely to show the model.**

### A gap in what the objective even looks at

`encode_row` lays `copy_token_weight` across `[first, eos_target)`, then
overwrites `w[first]` with `first_token_weight` (0.4) and `w[eos_target]` with
`eos_token_weight` (2.5). Every mask in the codebase — the loss's, the
continuation metric's, `sequence_margin_health`'s — selects on the copy weight.
So all of them span `first+1 .. eos_target−1` and **omit the first token and the
EOS decision**, both of which greedy exact copy tests.

A sequence whose weakest decision is its first token has been invisible to the
entire v0.0.27 apparatus. How much that currently costs is unknown: independence
arithmetic argues it is small (continuation-only independence at 0.8205 over
7.67 tokens predicts 0.2194 against a measured 0.2222, leaving little room for a
separate first-token factor), but expanded first-token *top-1* has never been
reported at all — only top-5, top-20 and a mean rank that a handful of outliers
can dominate. The fix costs nothing, so it is not worth arguing about.

## What v0.0.28 changes

One idea, applied in four places. Learning rate, step count, batch size and
curriculum are untouched — v0.0.27 showed the signal is misallocated, not
absent, so adding steps or raising the learning rate would answer a question
nobody asked.

| | v0.0.27 | v0.0.28 |
| --- | --- | --- |
| mining rank | `relu(margin − worst_gap)`, unbounded | in-band rows by hinge; out-of-band rows take a **constant** low priority |
| sequence hinge | uniform | ×1.0 in band, ×0.30 outside — slowed, not abandoned |
| `sequence_margin` | 1.20 | **0.60** — stop widening margins that are already safe |
| objective span | copy positions only | **every supervised decision**: first token, continuation, EOS |
| selection key | continuation-only health | **full-span health** |
| progress verdict | exact copy and continuation only | **also a large move in health** |

With `sequence_margin` 0.60 and `boundary_band_low` −0.75, the ordering inverts
to `−0.05` (0.65) > `−3.00` (0.405, the constant) > `+0.50` (0.10). Near-boundary
rows lead, hopeless rows still appear, already-correct rows sit last.

`margin_health_floor` and `margin_health_ceiling` are now separate config keys
rather than reusing `sequence_margin`, so lowering the hinge margin does not
silently rescale the health statistic. −0.6846 stays comparable.

## The preflight is the deliverable

A clamped mean cannot say how many sequences are near the boundary. A mean of
−0.88 is equally consistent with 70 cases packed at −0.88 and with a spread
holding a dozen cases just short of zero — and those two situations call for
opposite decisions.

So the free CPU preflight now prints, before any GPU guard:

- `EMBER_V028_SEQUENCES_WITHIN_REACH=<rate>` — the fraction of held-out cases
  whose worst decision sits in `[−0.75, 0)`;
- percentiles and a histogram of per-case worst margins;
- the specific cases inside that band, nearest-first;
- how many cases have their weakest decision at the first token or at EOS;
- `EMBER_V028_ADVICE=NO_REACHABLE_SEQUENCES_DO_NOT_LAUNCH_T4` when the band is
  empty.

**Read that number before spending anything.** If the band is near-empty,
boundary focus cannot work, and the honest next lever is capacity or
representation rather than more objective shaping. Learning that costs a CPU
job; learning it the other way costs a T4.

## Reading the result

- **Exact copy rises while health holds** — boundary focus works; keep going and
  watch `sequences_within_reach` refill from below.
- **`sequences_within_reach` grows but exact copy does not** — sequences are
  queueing at the boundary without crossing. That argues for a larger margin
  target on in-band rows specifically, not for more steps.
- **Health rises again and the band stays empty** — the distribution is
  translating without spreading. Objective shaping has done what it can; the next
  spend belongs on capacity or the digits-style representation work.

`progress` reports ADVANCED / FLAT / REGRESSED separately from `promotion`, and
now counts a large health move on its own — v0.0.27 earned that and did not get
it.

## Run order

1. `audit` — free, no token.
2. `python -m pytest -q tests`.
3. `preflight` — **the decision point.** Read `sequences_within_reach` and the
   histogram.
4. `train` — only if the band is not empty. Re-disarm `.github/ember-v028.trigger`
   to `bootstrap` afterwards.

v0.0.20 stays authoritative until a candidate reports promotion PASS.
