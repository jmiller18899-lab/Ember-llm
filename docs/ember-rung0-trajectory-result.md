# Ember rung-0 CPU trajectory — verified result

Completed on 2026-09-09 in **20 minutes 21 seconds**. All **25 tests passed**.
The original 40-step trajectory was reproduced; **no operating point was found**.

The first short-code structural regression occurred at **step 13**, a second at
**step 23**, and the original placement-learning gate first passed at **step 40**.
No observed step satisfied both that learning gate and short-code retention.

| Step | Source-passing short codes retained | Exact placement | Placement tokens | Learning gate |
| --- | ---: | ---: | ---: | --- |
| Source | 8/8 | 1/24 | 88/170 | baseline |
| 1 | 8/8 | 1/24 | 88/170 | FAIL |
| 7 | 8/8 | 1/24 | 89/170 | FAIL |
| 12 | 8/8 | 1/24 | 90/170 | FAIL |
| 13 | 7/8 | 1/24 | 90/170 | FAIL |
| 23 | 6/8 | 1/24 | 91/170 | FAIL |
| 38 | 6/8 | 1/24 | 94/170 | FAIL |
| 40 | 6/8 | 2/24 | 95/170 | PASS |

Placement is the existing conditional, teacher-forced value-token probe. It is
not end-to-end task accuracy. The learning gate requires at least one additional
exact case and a gain of at least three percentage points in token top-1 accuracy.
The endpoint gained 7/170 tokens, or 4.12 percentage points.

## Preservation and replication

Full familiar/copy/reference/KL checks ran at steps **1 and 40**. Step 40 was also
the first learning pass, so the event-triggered full check did not add another state.

| Full check | Step 1 | Step 40 |
| --- | ---: | ---: |
| Familiar canonical JSON | 84/90 | 82/90 |
| Familiar correct tool | 84/90 | 82/90 |
| Individual source-passing familiar cases retained | 84/84 | 82/84 |
| Copy protection | PASS | FAIL |
| Reference controls | 4/4 | 4/4 |
| Tool teacher KL | 0.0000006674 | 0.0007525200 |
| Copy teacher KL | 0.0000001494 | 0.0002502354 |

Both KL measurements stayed below their original 0.08 budgets. At step 40, only
`expanded_cases_retained` failed the copy checks. The expanded case `12+33+18`
was lost and `acct_ulq7-2291` was gained, which explains why aggregate copy rates
could hold while individual-case retention failed.

All seven endpoint replication checks passed. Recorded training metrics at steps
1, 20 and 40 exactly match the original artifact's stored floating-point values.
Source checkpoint identity and preservation-corpus counts match the original.
State hashes verified that observations did not change the model, the original
source was restored, and the frozen teacher remained unchanged.

## What broke

- **Step 13 — `system_target_short_code_01` (target `K8J3`).** At response token
  position 12 (zero-based), the original JSON-closing token `"},"` lost to a
  newline. Its source-versus-newline margin changed from +0.140453 to -0.010977.
  The response then ended with its argument string still open.
- **Step 23 — `system_target_short_code_09` (target `CZ9X`).** The first divergence
  was at response position 7: `the` changed to `C`. That source-versus-candidate
  margin changed from +0.864825 to -0.011695. The resulting response also ended
  with an unclosed argument string. The first divergence does not by itself
  identify the later closing-token failure's cause.

Both source responses already had wrong argument values; their protected property
was valid JSON with the correct tool name. These losses therefore establish a
structural regression, not the loss of two previously correct argument copies.

## Conclusion and next experiment

The exact original recipe preserves the full tested behavior at step 1. Its first
observed short-code regression occurs after accumulated updates, at step 13. This
rejects the claim that its two measured familiar failures happened on the first
update. It does not establish when other, unobserved errors first occurred.

Stopping earlier within these 40 steps cannot satisfy the unchanged learning gate:
that gate passes only at step 40, when preservation fails. This conclusion applies
to this one fixed trajectory; it does not rule out other data, update directions,
learning rates, or model designs. Full non-short-code/copy checks were not run at
intermediate states, so step 12 must not be called fully preserved.

A focused next experiment would test protection of JSON closing tokens on **fresh
short-code examples**, while leaving value copying free to improve. Compare it
with the same original recipe under the same learning and preservation gates.
Do not train on the two frozen failing cases or treat this reused development set
as a fresh promotion evaluation. This is a proposal; no additional run was launched.

The preservation pool contains 7/130 short-code tool rows (five four-character,
two five-character). That is measured coverage, not proof that coverage caused
the regression.

No GPU training, checkpoint export, promotion, deployment, or ClawAgent integration
occurred. The source checkpoint was not modified.

## Evidence

- [Completed CPU run](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34369464055)
- [Full report, original data and summary artifact](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34369464055/artifacts/10112186697)
- [Machine-readable verified findings](../reports/ember-rung0-trajectory-verified.json)

Evaluated commit: `06d8d02322f51404aaed2f27b8481fee6a954b2d`.
Artifact ZIP SHA256: `faeb2aafa046bb5938669c34e37eed0888f3457d16cfacde693732b87805c5b3`.
The downloaded ZIP digest matches GitHub's artifact digest; its report and data
were inspected, and the embedded data-file digest was independently checked.
