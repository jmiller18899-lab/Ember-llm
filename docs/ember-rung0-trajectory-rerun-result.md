# Rung-0 trajectory re-run: full guards at 1, 12, 13, 23, 40

CPU run `34374547081`, job `102543901469`, commit `aaae2dac9e48bfcbfb9dc676878c62d2cc487076`.
Artifact `ember-rung0-trajectory-34374547081`, ID `10114445146`.
Status `COMPLETE` in 1618.10 s under a 3000 s bound. 28 tests passed.

Evidence below is read from the completed job event log. The artifact was not
downloaded into this workspace; the storage host is unreachable from here, as it
was for the first-update diagnostic.

## The trajectory is unchanged

`endpoint_reproduction` is true on every field: placement 2/24 and 95/170,
familiar 82/90 JSON and tool, tool KL 0.00075252, copy KL 0.00025024, and
`expanded_cases_retained` as the only failing copy check. `first_short_code_loss_step`
is 13 and `first_placement_learning_step` is 40, as before. The three added
batteries therefore cost wall time and nothing else.

## Full battery by state

| Step | Familiar correct tool | Source cases retained | Copy protection | Reference | Tool KL budget | Copy KL budget | Learning gate |
| ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 1  | 84/90 | 84/84 | PASS | 4/4 | PASS | PASS | FAIL |
| 12 | 84/90 | 84/84 | PASS | 4/4 | PASS | PASS | FAIL |
| 13 | 83/90 | 83/84 | PASS | 4/4 | PASS | PASS | FAIL |
| 23 | 82/90 | 82/84 | PASS | 4/4 | PASS | PASS | FAIL |
| 40 | 82/90 | 82/84 | **FAIL** | 4/4 | PASS | PASS | PASS |

## What this settles

**Step 12 is fully preserved.** Every preservation check passes on the complete
battery: familiar 84/90 unchanged from source, all 84 source-passing cases
retained, copy protection intact, references 4/4, both KL budgets. The earlier
caveat — that step 12 could not be called preserved because only short codes had
been observed there — is resolved in favour of preservation. The single failing
check at step 12 is the learning gate, which is not a preservation check.

**The step-13 short-code loss is the first preservation failure of any kind.**
Nothing else fails at 13; copy protection still passes. The original reading was
right, and now it rests on measurement rather than on the absence of it.

**The familiar losses are exactly the short-code losses.** 84 at steps 1 and 12,
83 at 13 after `system_target_short_code_01`, 82 at 23 after
`system_target_short_code_09`. No case of any other kind was lost at any measured
state, and familiar retention does not move again between 23 and 40.

**The copy-protection failure happened after step 23.** It passes at 1, 12, 13 and
23 and fails only at 40, so its onset lies in the window 24-40 and remains
unmeasured. This is now the only unlocated preservation failure in the run.

## A detail the per-step trace makes visible

Placement token top-1 climbs 88 to 95 across the run, passing 94/170 at step 38 —
a 3.53 point gain, already over the 3 point threshold. Exact placement stays at
1/24 until step 40. The learning gate is therefore pinned to step 40 by its
one-additional-exact-case requirement, not by token accuracy.

## Scope

CPU only. No GPU training, checkpoint export, promotion, deployment or
integration. Pristine restore and frozen-teacher hashes verified at exit, and
`observation()` verified that no probe altered the model.

## Not run

Narrowing the copy-protection onset inside 24-40 would take one more trace with
batteries at, say, 30 and 35. This is a proposal; no further run was launched.
