# Placement probe check — result

Read-only CPU diagnostic, GitHub Actions run `34313755514`, job `102345554202`,
commit `46b0ce42c70ee3b7a32e781563530d9532ce0eaa`. Artifact
`ember-placement-probe-check-34313755514`, ID `10089324257`.
Source: untouched v0.0.31 step 479. Nothing was trained, written, or authorized.

## The question

v0.0.50 reported `placement_token_top1_gain` of exactly 0.0 at all four
checkpoints while entry moved 0 → 1 → 5 → 6 on the same student. The plumbing was
verified correct by reading, so the remaining question was whether the metric
could move at all.

## Answer: the probe is fine. The hypothesis was wrong.

| Measurement | v0.0.50's dev set | v0.0.48's dev set (control) |
| --- | ---: | ---: |
| Cases | 24 | 24 |
| Exact top-1 | 1/24 | 1/24 |
| Token top-1 | 88/170 (51.8%) | 90/171 (52.6%) |
| Mean loss | 3.267 | 3.086 |
| Target length (min/median/max) | 3 / 5.5 / 15 | 3 / 6.0 / 15 |
| Degenerate single-token targets | 0 | 0 |

The control reproduces v0.0.48's **published** before-probe exactly — 90/171 token
top-1 and 1/24 exact — which confirms the diagnostic rebuilt the real pipeline
rather than an approximation of it.

Verdict fields, all measured rather than argued:

- `metric_saturated`: **false** — 51.8% is nowhere near a ceiling;
- `metric_dominated_by_easy_positions`: **false**;
- `token_gate_reachable`: **true**;
- `probe_is_a_usable_placement_signal`: **true**;
- `findings`: **empty**.

## How much movement the gate actually needed

- Current token top-1 rate: 51.8%.
- Ceiling if every first token were fixed: 64.1%.
- Gain available: **12.4%**; gain required by `minimum_placement_token_top1_gain`: **3%**.
- Tokens that had to flip to clear the token gate: **5 of 170**.
- Cases already exact: 1. Cases **one token** from exact: **3**.
- Wrong-position counts per case: `[0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 9]`.

So `placement_exact_gain >= 1` was three single-token flips away, and the token
gate was five tokens away out of a hundred and seventy.

## What was predicted, and what the numbers say

The hypothesis going in was that the teacher-forced probe was dominated by easy
continuation positions and already near a 6/7 ceiling, making the gate
unreachable by construction. That is **not** what the source shows.

It was directionally right about one thing: the first token after the value-free
prefix is much harder than the rest — 3/24 correct (12.5%) against 85/146 (58.2%)
for later tokens. But that 45.7-point gap sits just under the domination
threshold, and more importantly the later positions are *also* frequently wrong,
with several cases missing 4 to 9 tokens. The metric measures real, distributed
error with ample room to rise.

A second prediction was also wrong: v0.0.48's post-training 147/171 (86.0%) was
read as possibly "the ceiling of the easy positions." The measured ceiling from
fixing first tokens alone is 64.1%, so v0.0.48 improved later positions
substantially too.

## Consequence

Both benign explanations are eliminated. The probe is not blind, and v0.0.50 did
not draw an easier development set — 51.8% against v0.0.48's 52.6%, with
identical exact counts.

Therefore v0.0.50's four consecutive zeros are a real measurement: **the
placement objective received effectively no useful gradient at that update
size.** The difference between v0.0.48's +33 points and v0.0.50's +0 is the
update, not the data and not the metric.

This supports giving placement a materially larger update budget under the
frozen-teacher KL preservation that v0.0.50 showed holds the model at 85/90 with
roughly eight times its KL budget unspent. It does not on its own prove placement
is learnable under protection; it establishes that the experiment can detect the
answer, and that only five token flips are needed to register one.

Entry needs no further budget: it reached +6 under v0.0.50's conservative recipe.
