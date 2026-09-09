# Ember v0.0.49 protected-learning CPU canary

v0.0.48 answered the question it was built to answer and failed on a different
one. Both target behaviours are learnable:

| Objective | Before | After |
| --- | ---: | ---: |
| Entry top-1 | 19/24 | **24/24** |
| Placement exact top-1 | 1/24 | **9/24** |
| Placement token top-1 | 52.6% | **86.0%** |

And a plain 50/50 two-objective update at `1.2e-6` for 120 steps destroyed
almost everything else: the familiar battery fell from 84/90 to **33/90** correct
tool, `get_time` lost its reference envelope (4/4 → 3/4), copy protection failed,
and short_code, long_code and mixed each went to **0/10**.

v0.0.49 restarts from the untouched **v0.0.31 step 479** source and changes the
update, not the objectives. The v0.0.48 candidate is discarded and is not the
base for anything.

## The three changes

**1. A replay term that dominates.** Loss weights move from 0.5/0.5 to
`entry 0.15`, `placement 0.15`, `replay 0.70`. The config refuses to load unless
replay is at least 0.5 *and* strictly greater than the two learning terms
combined, so protection cannot quietly become the minority term in a later edit.

**2. Replay that reaches what actually collapsed.** v0.0.48's synthetic values
covered five code-shaped subtypes. The collapse hit `mixed`, `url`, `path`,
`model_id`, `digits` and `entity` as well, so v0.0.49 generates fresh values
across **all nine kinds**, spread over each kind's templates, and builds two
corpora: the frozen v0.0.44 envelope prompt, and the v0.0.26 bare-value copy
prompt that the copy guard actually measures.

Replay targets are **the source model's own correct output** — envelopes it
already emits validly with the right tool, copies it already reproduces exactly
with a clean stop. Supervising a correct answer the source cannot yet produce
would be a third learning objective wearing protection's clothes, and it would
make a PASS unreadable. Filtering to what the source already gets right keeps the
term purely preservative, which is what the familiar-90 and copy gates measure.

**3. A trust region, not just a smaller learning rate.** The learning rate drops
6× to `2e-7`, and the config rejects anything at or above the v0.0.48 rate. But
AdamW takes a roughly `lr`-sized step per parameter *regardless of gradient
magnitude*, so a lower rate only slows drift — it does not bound it. After every
optimizer step each parameter tensor is projected back into an L2 ball of radius
`2e-4 × ‖source tensor‖`. The achieved drift is measured and reported, and the
gate fails if the region was ever exceeded.

Step count stays at **120**, deliberately, so the learning rate and the trust
region are the change rather than a shorter run.

One caveat is stated up front rather than discovered later: at `2e-7` the run may
never reach the cap, in which case the trust region protected nothing and the
learning rate and replay term did all the work. The report says which happened —
`steps_clipped`, `engaged`, and the fraction of the cap the drift actually
reached — so the mechanism cannot take credit it did not earn. The cap is
deliberately not tuned to bind: the per-step drift of an 80M model under AdamW is
not known ahead of this run, and a cap guessed tight enough to bind could stop
the objectives moving at all and produce an uninformative null.

## What is not changed

Every v0.0.48 gate threshold, unchanged: entry top-1 gain ≥ 1, placement exact
gain ≥ 1, placement token gain ≥ 5 points, entry loss reduction ≥ 3%, placement
loss reduction ≥ 5%, familiar envelope and tool ≥ 84/90, all kind and subtype
floors, 4/4 references, and the existing copy guard. Lowering the learning bar
while lowering the update would make a PASS meaningless, so a test asserts the
whole `gate` block is byte-identical to v0.0.48's.

The familiar 90 cases and the four historical controls remain **evaluation-only**.
A test walks every objective and replay value and asserts none of them touches
the held-out battery or any value used by v0.0.43, v0.0.44 or v0.0.48.

## The interference trajectory

v0.0.48 could say only that the model was fine before and ruined after. v0.0.49
probes an 18-case synthetic envelope cohort (two per kind) plus both development
objectives at steps 40 and 80, alongside the measured drift. The report renders
that as a trajectory, so the outcome is a curve rather than two endpoints: at
what drift the objectives start moving, and at what drift protection starts
breaking.

That is the number the next phase needs. If entry and placement move while the
probe holds, there is an operating point and the trajectory says roughly where.
If the probe falls at the same drift where the objectives move, the two are
entangled at this scale and no learning rate will separate them — which would
point at a different mechanism (parameter-subset updates, or a preservation term
on the distribution rather than on argmax) rather than at more tuning.

## How to read the outcome

A **PASS** means both objectives moved by the v0.0.48 margins while the familiar
battery, the four controls, the copy guard and the trust region all held. That is
learning evidence only. It is not a fresh promotion evaluation, it does not
authorize GPU training, and — as v0.0.48's own design note requires — if the
familiar battery reaches 86/90 a fully disjoint secondary 90-case confirmation
battery is still needed before any GPU decision.

A **FAIL on the learning checks with protection intact** is the more likely
result at this update size, and it is informative rather than wasted: combined
with the trajectory it bounds how much update the objectives need, against how
much the protected behaviour tolerates.

A **FAIL on protection** at `2e-7` with 70% replay and a `2e-4` trust region
would be the strong result: it would mean interference is not a step-size problem
at all, and that the entry and placement behaviours share representation with the
envelope and copy behaviours closely enough that no bounded full-parameter update
separates them.

No GPU training, promotion, deployment, or production integration is possible in
this runner.
