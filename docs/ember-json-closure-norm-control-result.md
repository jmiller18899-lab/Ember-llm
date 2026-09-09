# JSON-closing norm-matched control — verified result

CPU run `34392756958`, job `102604883875`, commit `cc42210e85512a30aa9a5070449430c361886214`.
Artifact `ember-json-closure-control-34392756958`, ID `10120745963`.
Status `COMPLETE` in 699.42 s. 42 tests passed. Read from the completed job event log.

## The answer: the shrinkage explains nothing

The control took the projection's magnitude schedule with its direction removal
dropped, and landed on the unprotected baseline, not on the projection arm.

| Measurement at step 40 | Unprotected baseline | Projection | Norm-matched control |
| --- | ---: | ---: | ---: |
| Exact placement | 2/24 | 1/24 | **2/24** |
| Placement token top-1 | 95/170 | 93/170 | **95/170** |
| Learning gate | PASS | FAIL | **PASS** |
| Familiar short codes retained | 6/8 | 6/8 | 6/8 |
| Copy protection | FAIL | FAIL | FAIL |
| Reference controls | 4/4 | 4/4 | 4/4 |

Roughly 17% less cumulative update magnitude cost no exact case and no token.
The hypothesis that the projection arm's learning loss was incidental shrinkage
is refuted. Removing the eight closure directions is what cost the exact case and
the two tokens.

## The retention gain was also the directions, not the smaller step

| Observed step | Short codes: baseline | projection | control |
| --- | ---: | ---: | ---: |
| 1  | 8/8 | 8/8 | 8/8 |
| 12 | 8/8 | 8/8 | 8/8 |
| 13 | 7/8 | 8/8 | **7/8** |
| 23 | 6/8 | 8/8 | **6/8** |
| 40 | 6/8 | 6/8 | 6/8 |

The control loses `system_target_short_code_01` at step 13 and
`system_target_short_code_09` at step 23, the same cases at the same steps as the
unprotected trajectory. Only the projection held 8/8 through step 23. Closure
protection therefore does delay the structural regression, and the delay is not
an artifact of taking shorter steps.

Both effects are real and they point in opposite directions. The projection buys
retention through step 23 and costs the endpoint's learning. That is a genuine
trade-off to be priced, not an experimental artifact to be removed.

## The protection missed the examples it was built from

Fresh anchors degrade identically in both arms: 8/8 at steps 1, 12 and 13, 7/8 at
23, 6/8 at 40. The projection was built from those eight examples' closing
decisions and still did not keep them structurally valid under free generation,
while it did protect the familiar short codes at 13 and 23. This is consistent
with the prefix-coverage gap the projection run identified: the anchors fail
through changed argument text, which the source-prefix protection never covers.
Fresh holdout stayed 6/6 in both arms.

## Arm integrity

`all_40_nonzero_norm_matched_updates` passed: every step's magnitude landed
within 10% of the fraction the projection would have retained, and every step
kept at least half the basis component that the proposal carried, where the
projection retains none of it. Endpoint `protected_closing_tokens_still_top1` was
true here as well, which is expected: the control never moved along those
directions enough to flip decisions with source margins of 4.27 to 8.46.

`endpoint_passed` is false in this arm too, on the same acceptance criteria the
projection failed: familiar floors, source-case retention and copy protection.
Only the learning gate differs between the arms.

## A note on precision

The first attempt at this control aborted at step 1 on a cosine bound. At this
learning rate the per-element update is near float32 resolution, so assignment
rounds away about a percent of every update in every arm of every run in this
line of work. It cancels between arms and does not threaten these comparisons,
but it is the noise floor against which sub-percent effects should be read.

## Scope

CPU only. No GPU training, checkpoint export, promotion, deployment or
integration. Pristine restore and frozen-teacher hashes verified at exit.

## Next

The trade-off is now priced on one seed at one dose. What is not measured is
whether it is tunable: a ladder over the projection strength, from no removal to
full removal, would show whether any dose keeps the step-23 retention without
losing the step-40 exact case. This is a proposal; no further run was launched.
