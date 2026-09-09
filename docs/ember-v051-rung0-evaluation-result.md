# Ember v0.0.51 corrected ladder — rung 0 evaluation

CPU run `34348466440`, job `102455609482`, commit
`ea3fe6a9f1ccaac59a91e5c6b3e8ebee70700e1c`.
Artifact `ember-v051-ladder-34348466440`, ID `10103409558`.

The selected rung 0 (LR 1e-7, 40 steps) gained one exact placement case and
7/170 placement tokens (+4.12 percentage points), but failed preservation:

| Measurement | Source | Rung 0 |
| --- | ---: | ---: |
| Familiar canonical JSON | 84/90 | 82/90 |
| Familiar correct tool | 84/90 | 82/90 |
| Familiar short codes | 8/10 | 6/10 |
| Reference controls | 4/4 | 4/4 |
| Copy protection | baseline | FAIL |

The failed familiar checks were `json_floor`, `tool_floor`, `kind_short_code`,
and `subtype_short_code_len4`. The failed copy check was `expanded_cases_retained`.
Other kind scores were unchanged: long_code 7/10, path 9/10, and all other kinds
10/10. Tool KL was 0.00075252; copy KL was 0.000250235.

`operating_point_found` was false for the selected rung. Full familiar/copy
regression tests were applied only to that rung at its endpoint. Therefore the
run does not establish failure of every unselected rung or every intermediate
step, nor does it identify the first update that lost a familiar case.

## Corrected interpretation of the gradient evidence

Step-0 weighted placement gradient norm was 27.70292678 and weighted preservation
gradient norm was 0.000129273. Their norm ratio says that the preservation gradient
was tiny at initialization. It does not identify a percentage contribution to
AdamW's actual parameter update or establish when behavioral damage happened.

KL to an identical teacher has zero gradient in exact arithmetic. A squared L2
anchor to identical source weights also has zero gradient there. Small float32
residuals do not change either fact. A uniform trust region caps distance, while
a projection onto allowed directions can change the direction of a proposal.

Identical gradient norms across rungs are consistent with pristine restoration,
but do not independently prove it; direct state hashes and optimizer-state
checks are stronger evidence. Low teacher KL on distillation rows does not
imply retention on a different familiar-case set.

The teacher-success filter may affect coverage, but the observed regressions
alone do not establish a coverage defect caused the loss. Short codes were the
kind that regressed, while long codes had the lowest source score (7/10).

The separate first-update diagnostic preserved all 84 source-passing familiar
cases in both arms, with no placement gain. Its data and loss mixture differed
from this ladder. The next exact replication is specified in
[the rung-0 trajectory plan](ember-rung0-trajectory-plan.md).

No GPU training, promotion, deployment, or production integration occurred.
