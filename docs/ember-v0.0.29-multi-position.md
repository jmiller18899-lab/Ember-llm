# Ember v0.0.29 — retiring more than one bad decision per row

## Where v0.0.28 left the model

| | v0.0.27 | v0.0.28 |
| --- | ---: | ---: |
| expanded exact copy | 20/90 | **28/90** |
| expanded continuation | 599/730 | **632/730** |
| expanded first-token top-1 | 78/90 | **85/90** |
| legacy exact copy | 4/9 | **5/9** |
| legacy continuation | 60/69 | **62/69** |
| full-span margin health | −0.8377 | **−0.5865** |
| cases in `[−0.75, 0)` | 16 | 21 |

Boundary focus did what v0.0.27's uniform pressure could not: it turned margin
movement into whole strings. Extending the objective to the first token paid
too — seven more cases get their opening token right, and cases whose *weakest*
decision is the first token fell from 13 to 9.

## Three things the numbers say

### The pipeline is flowing, not draining

The band held 16 cases at the start and 21 at the end **while 8 crossed out of
it**. That means 13 cases were pulled up into it from below. The band is
refilling faster than it empties, and those 21 are worth up to +23.3 points of
exact copy on their own — enough to clear the 0.40 expanded gate without any
help from the rest of the distribution.

This inference took two numbers and a subtraction, which is two numbers too
fragile. v0.0.29 reports the actual per-case transition matrix
(below-band → within-reach → correct), so the next run states it instead.

### The loss retires one bad decision per row; rows have more than one

```
730 × (1 − 0.8658)  =  98.0   wrong continuation decisions
                    ÷  62     failing rows
                    =  1.58   per failing row
        + 5 first-token failures  →  ≈ 1.66 wrong decisions per failing row
```

v0.0.28's sequence term reduces each row to the **maximum** of its hinges, so
one pass can retire at most one of those. That is the binding constraint now,
and it is orthogonal to everything v0.0.28 changed.

v0.0.29 sums the **k worst** hinges per row, with `k = 2` to match the measured
count. Summing rather than averaging matters: at `k = 1` the statistic is
*exactly* v0.0.28's, so the second push is added without diluting the first. A
test asserts that equivalence by running both trainers' helpers on the same
tensors.

### Four gates can never be met, and one of them just started biting

`5/9` is `0.5555555555555556` in double precision. The config stores
`0.5555555556`. The scaffold compares with a raw `>=`, so:

```
5/9 >= 0.5555555556   →   False        (short by 4.4e-11)
```

v0.0.28 landed on exactly 5/9 and will therefore be recorded as *failing* the
legacy exact-copy gate it actually met. The same holds for `7/9` against
`minimum_first_token_top20_rate` and `8/9` against the top-5 and teacher-forced
gates — masked so far only because those rates have been 1.0.

**Worth checking the `gates` block of your v0.0.28 report.** v0.0.29 routes every
threshold comparison through `v029_meets`, which applies the same tolerance the
protection checks have always used.

## What changes

| | v0.0.28 | v0.0.29 |
| --- | --- | --- |
| sequence statistic | `max` of row hinges | **sum of the k=2 worst** |
| gate comparison | raw `>=` | **tolerance-aware** |
| band reporting | two counts | **per-case transition matrix** |
| gate reporting | rate vs threshold | **+ discrete distance to each bar** |
| source checkpoint | v0.0.27 | **v0.0.28** |
| everything else | — | **unchanged** |

Learning rate, step count, batch size, token weights, `sequence_margin`, the
band bounds and the curriculum are all held where they have been since v0.0.26.
A test compares the v0.0.29 config against v0.0.28's key by key and fails if any
of them drifted.

## Reading gate distance

A rate hides the denominator, and the denominators here differ by an order of
magnitude:

| gate | measured | needs | short by |
| --- | ---: | ---: | ---: |
| expanded exact copy ≥ 0.40 | 28/90 | 36/90 | 8 cases |
| expanded continuation ≥ 0.90 | 632/730 | 657/730 | 25 tokens |
| legacy continuation ≥ 0.90 | 62/69 | **63/69 = 0.9130** | 1 token |
| legacy exact copy ≥ 5/9 | 5/9 | — | met |

The legacy continuation gate reads as 0.14 points away and is really **one
token** — and clearing it overshoots to 0.9130, because 0.90 is not expressible
on a 69-token denominator. Both the preflight and the final report now print
this table.

## A judgement call for you, not for me

The expanded gates (`0.40` exact copy, `0.90` continuation) are numbers I picked
in v0.0.26 as "ambitious but reachable". They are not derived from anything.
v0.0.28 now beats the v0.0.20 authoritative checkpoint on every metric and meets
the historical legacy bar of 5/9. **Whether that is enough to make v0.0.28
authoritative is your call**, and it does not depend on v0.0.29 — I have not
changed any promotion gate, because moving a bar after seeing the result is how
you stop being able to trust it.

## Reading the v0.0.29 result

- **Exact copy rises and the band shrinks** — the queue is draining faster than
  it fills. Watch for the transition matrix showing few `below_band →
  within_reach` moves; that is the signal to stop shaping and look at capacity.
- **Exact copy rises and the band holds or grows** — keep going at fixed config;
  a longer run at these settings becomes the cheapest next win.
- **Band grows, exact copy flat** — sequences are queueing without crossing, and
  `k` is not the limit. Raise in-band pressure rather than `k`.

## Run order

1. `audit` — free.
2. `python -m pytest -q tests`.
3. `preflight` — reads the v0.0.28 baseline, prints `EMBER_V029_GATE_DISTANCE`,
   `sequences_within_reach` and the worst-margin histogram.
4. `train` — one approved T4. Re-disarm `.github/ember-v029.trigger` afterwards.
