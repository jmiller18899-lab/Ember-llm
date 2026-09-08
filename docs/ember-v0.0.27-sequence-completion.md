# Ember v0.0.27 — converting continuation into exact copies

> **Result: the mechanism works, the allocation does not.**
> `sequence_margin_health` −0.8871 → −0.6846 (+0.2025), expanded continuation
> 0.8151 → 0.8205, expanded mean first-token rank 23.74 → 16.18, expanded exact
> copy unchanged at 22.22%. Promotion FAIL, progress FLAT — and that FLAT was my
> own reporting bug: `v027_progress` looked only at exact copy and continuation,
> so a +0.2025 move in the statistic the run was built to move counted for
> nothing. Fixed in v0.0.28. Analysis in
> [`reports/ember-v0.0.27-result.json`](../reports/ember-v0.0.27-result.json);
> next phase in
> [`docs/ember-v0.0.28-boundary-focus.md`](ember-v0.0.28-boundary-focus.md).

## What v0.0.26 left behind

v0.0.26 fixed the format gap, restored a learning rate that moves the model, and
gained six points of per-token copy accuracy. It did not gain much exact copy.
The reason is arithmetic, and it is the whole basis of this phase.

Mean continuation length on the legacy battery is 69 tokens over 9 cases, or
7.67 tokens. If per-token errors were independent, exact copy would be
`accuracy ** 7.67`:

| checkpoint | per-token | predicted exact copy | measured |
| --- | ---: | ---: | ---: |
| v0.0.20 baseline | 0.7534 | 0.1141 (10.3/90) | 0.1889 (17/90) |
| v0.0.26 best (step 219) | 0.8151 | 0.2086 (18.8/90) | 0.2222 (20/90) |
| v0.0.26 later checkpoint | 0.8300 | 0.2397 (21.6/90) | 0.2111 (19/90) |

At the baseline, measured exact copy ran well *above* the independent
prediction — errors were clustered, which is what a systematic structural
failure looks like. At v0.0.26's best they sit right on it. **The errors are now
scattered**, so there is no longer one broken thing to find, and exact copy is
being crushed multiplicatively by length. Reaching the legacy 0.5556 gate needs
per-token accuracy of 0.9262, up from 0.8151.

Two consequences drive the design.

### The selector probably picked the worse checkpoint

Look at the last two rows of that table. Step 219 measured one case *above* its
prediction; the later checkpoint measured 2.6 cases *below* its own. The
standard deviation of a 90-case rate near 0.22 is about 3.9 cases, so both are
ordinary fluctuations — and the higher-continuation checkpoint is the better
estimate of the underlying model. Leading the selection key with a 90-case count
recreates the v0.0.25 resolution problem one order of magnitude down.

There is no higher-resolution *discrete* signal to switch to. Teacher-forced
completion at every position and greedy exact copy are the same event: if the
argmax is right at every position given the correct prefix, greedy decoding
follows exactly that path, and conversely. Clean stop is already 1.0, so EOS
does not separate them either.

So selection moves to a continuous statistic: `sequence_margin_health`, the mean
across held-out cases of each sequence's **worst-position margin** — the gap
between the correct token's logit and its best competitor, minimised over the
sequence, clamped to `[-2.0, sequence_margin]`. It measures the same thing exact
copy does (how close each sequence is to being right everywhere) without
quantising to 1/90, and the clamp stops one catastrophic case from dominating.

### The loss has never applied sequence-level pressure

The margin term inherited from v0.0.20, and carried unchanged into v0.0.26,
averages its hinge over every copy token in the batch. A sequence with one bad
position out of eight therefore receives one eighth of the gradient that its
single failing position needs — while under independent errors, that position is
the *only* thing standing between the row and an exact copy.

v0.0.27 adds a term on each row's weakest copy token:

```python
sequence_hinge[target_mask] = relu(sequence_margin - gap)
worst = sequence_hinge.max(dim=1).values     # per row
sequence_term = (worst * rows).sum() / rows.sum()
```

`tests/test_ember_v027_sequence_completion.py` pins the property this rests on:
a row with one position 2.2 below the margin and three comfortable ones scores
2.2, while a row with four positions each 0.55 below — the same mean hinge —
scores 0.55. The concentrated failure is the one that cannot produce an exact
copy, and it now carries four times the loss.

Hard mining also returns, scored per sequence rather than per token. It was
removed for v0.0.26 because the pool it drew from could not contain the answer;
with format parity established that objection is gone, and worst-position
hardness is aimed at exactly the rows that fail to complete.

## What changes, and what deliberately does not

| | v0.0.26 | v0.0.27 |
| --- | --- | --- |
| curriculum | format-parity, 90-case battery | **unchanged** |
| learning rate | 1.2e-6 | **unchanged** |
| steps / batch / weights | 600 / 8×2 / 8.0-0.4-2.5 | **unchanged** |
| source checkpoint | v0.0.20 | v0.0.26 best |
| loss | CE + mean-token margin hinge | CE + mean-token hinge + **worst-position hinge** (margin 1.20, weight 2.0) |
| batch composition | uniform | **half uniform, half worst-sequence mined** |
| selection key | 90-case exact copy first | **`sequence_margin_health` first**, exact copy as tiebreak |
| reporting | promotion only | promotion **and** progress (ADVANCED / FLAT / REGRESSED), plus exact copy by kind and by length |

Everything v0.0.26 established is held fixed on purpose. This run is a clean
test of one hypothesis: that sequence-level pressure converts per-token accuracy
into whole-string copies.

## Protection

`best.pt` is seeded with the v0.0.26 weights, and the seed carries a `+1e9`
validation-loss tiebreaker, so a checkpoint that merely ties cannot replace them.
A checkpoint must also clear four non-regression guards. Slack is allocated by
resolution: none on the high-resolution metrics (expanded continuation is ~700
tokens), one case on expanded exact copy (σ ≈ 3.9 cases), none on the legacy nine.

`protected_expanded_continuation_top1_rate` is set at 0.8100 rather than the
measured 0.8151 so that float differences between the CPU baseline measurement
and the GPU evaluation cannot disqualify the baseline against itself.

## Reading the result

- **Exact copy jumps while continuation holds** — sequence pressure works. The
  remaining question becomes length, which `final_by_length` now answers.
- **Continuation rises again but exact copy tracks the independence curve** —
  worst-position pressure is not enough; the next lever is exposure (scheduled
  sampling) or the digits-style representation work.
- **Continuation falls** — the sequence term at weight 2.0 is over-driving.
  Lower `sequence_margin_loss_weight` before touching anything else.

Either way, `progress` reports ADVANCED / FLAT / REGRESSED separately from
`promotion`, so a genuine advance is not filed as a bare FAIL the way v0.0.26 was.

## Run order

1. `audit` — free, no token. Confirms the carried curriculum still reports
   `FORMAT_PARITY`, and that the v0.0.15 curriculum still reports `FORMAT_GAP`.
2. `python -m pytest -q tests` — includes the numerical guards on the new loss
   and the mining selector.
3. `preflight` (CPU) — records the v0.0.26 baseline including the new
   `sequence_margin_health`, exercises the mining path, and asserts parity.
4. `train` — one approved T4 session. Re-disarm `.github/ember-v027.trigger` to
   `bootstrap` afterwards.

v0.0.20 stays authoritative until a candidate reports promotion PASS.
