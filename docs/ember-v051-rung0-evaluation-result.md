# Ember v0.0.51 corrected ladder — rung 0 evaluation

CPU run `34348466440`, job `102455609482`, commit `ea3fe6a9f1ccaac59a91e5c6b3e8ebee70700e1c`.
Artifact `ember-v051-ladder-34348466440`, ID `10103409558`.

## The answer: rung 0 does not hold the familiar battery

The corrected selector chose rung 0 — "smallest update that cleared the bar" —
and its evaluation-only result is:

| Measurement | Source | Rung 0 (1e-7) |
| --- | ---: | ---: |
| Familiar canonical JSON | 84/90 | **82/90** |
| Familiar correct tool | 84/90 | **82/90** |
| Reference controls | 4/4 | **4/4** |
| Copy protection | — | **FAIL** |

Failed familiar checks: `json_floor`, `tool_floor`, `kind_short_code`,
`subtype_short_code_len4`. Failed copy check: `expanded_cases_retained`.

Per kind, the loss is entirely in one place:

| Kind | Source | Rung 0 |
| --- | ---: | ---: |
| short_code | 8/10 | **6/10** |
| long_code | 7/10 | 7/10 |
| digits | 10/10 | 10/10 |
| model_id | 10/10 | 10/10 |
| url | 10/10 | 10/10 |
| path | 9/10 | 9/10 |
| entity | 10/10 | 10/10 |
| expression | 10/10 | 10/10 |
| mixed | 10/10 | 10/10 |

`operating_point_found` is **false**. Every rung on the ladder learns, and the
smallest one that clears the learning bar still costs two familiar cases and the
copy retention check. There is no operating point in this ladder.

## Why: the preservation term has no gradient when it is needed

The step-0 gradient balance, now printed, is decisive and identical across all
four rungs:

- weighted placement gradient norm: **27.70**
- weighted preservation gradient norm: **0.000129**
- placement share of the update: **99.99953%**

The placement gradient is roughly **214,000 times** the preservation gradient at
the first step. That is not a weighting mistake. It is structural: KL(teacher ‖
student) is exactly zero while the student still equals the teacher, so its
gradient is zero too. The loss weights say preservation carries 0.80, but at the
moment the first step is taken it carries nothing.

**Frozen-teacher KL is purely reactive.** It can pull the model back after it has
drifted; it cannot prevent the drift that does the damage. Everything the earlier
phases read as "preservation dominates" was true of the loss weights and false of
the gradients.

This also explains the apparent contradiction in the numbers. Rung 0's tool
teacher KL is 0.000753 — a hundredth of budget — while the familiar battery still
falls two cases. The KL is measured on the distillation corpora, and those corpora
do not contain what broke.

## A second flaw, in the replay corpus itself

Distillation rows are kept only where the *teacher already succeeds*. That rule
was introduced deliberately in v0.0.49, to stop the preservation term smuggling in
a third learning objective, and v0.0.50 and v0.0.51 inherited it. It has a side
effect nobody checked:

**coverage is proportional to existing competence.** The kinds most at risk are
the kinds the model is worst at, and those contribute the fewest preservation
rows. short_code is the weakest kind on the familiar battery at 8/10, and
short_code is exactly where all of rung 0's loss landed.

## What the run confirmed

- **Replication.** rungs reproduced the first ladder to within float noise:
  +1/+4.12%, +3/+15.88%, +6/+29.41%, +6/+32.35%. The frontier is stable.
- **Pristine restore verified.** All four rungs measured identical step-0
  gradients, `[27.702926783, 0.000129273]`, proving each rung really restarted
  from the source and the ladder is not confounded.
- **Reference controls held at 4/4** on rung 0, unlike v0.0.48's 3/4.
- Rung 0's maximum relative drift is 0.000396, about twice the v0.0.49 trust
  region, for a two-case familiar loss.

## Consequence

Making the update smaller is exhausted as a strategy. Rung 0 is already the
smallest update that learns anything, its drift is 4e-4, and it still breaks
short_code. The lever that remains is not size but *reach*: constrain which
parameters may move, rather than how far all of them move.

Two candidates follow directly from the mechanism above, and they are
complementary rather than alternatives:

1. **A preservation term with gradient at step 0.** An L2 anchor to the source
   weights, or the v0.0.49 trust-region projection, both act from the first step
   instead of waiting for divergence. The trust region already exists and is
   tested.
2. **A preservation corpus that covers the weak kinds.** Sampling rows the teacher
   gets *wrong* is not usable as a distillation target, but the familiar battery's
   own weak kinds can be covered by fresh synthetic values of the same subtypes,
   weighted up rather than filtered out by success rate.

No GPU training, promotion, deployment, or production integration occurred.
