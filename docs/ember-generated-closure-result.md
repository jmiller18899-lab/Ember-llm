# Ember generated-prefix closure — verified CPU result

The 40-update run completed in **7 minutes 59 seconds** on 2026-09-09.
All **63 implementation tests passed** locally and on the runner.
**Every preservation check passed, but the copying-learning gate failed.**

## Endpoint comparison

| Measurement | Source | Original rung 0 | Fixed-prefix projection | Generated-prefix punctuation loss |
| --- | ---: | ---: | ---: | ---: |
| Familiar valid JSON / correct tool | 84/90 | 82/90 | 82/90 | 84/90 |
| Source-passing familiar cases retained | 84/84 | 82/84 | 82/84 | 84/84 |
| Exact placement | 1/24 | 2/24 | 1/24 | 1/24 |
| Placement token top-1 | 88/170 | 95/170 | 93/170 | 89/170 |
| Copy preservation | Baseline | FAIL | FAIL | PASS |
| Reference controls | 4/4 | 4/4 | 4/4 | 4/4 |

The endpoint gained one correctly predicted placement token, or 0.59 percentage
points, and no exact placement case. The unchanged learning requirements demand
at least three percentage points and one additional exact case. Both were missed.
Placement is a conditional teacher-forced value-token probe, not end-to-end copying
accuracy. Fresh training and holdout examples had **zero exact arguments** despite
retaining valid JSON and the correct tool name.

All 84 source-passing familiar cases were individually retained, with no newly
passing familiar cases. Both legacy and expanded copy-case retention passed,
along with every aggregate copy-rate check. Tool KL was 0.001346764 and copy KL
0.000149199, both below their original 0.08 budgets. All 40 updates were finite
and nonzero.

## Observed states

| Step | Familiar short-code retention | Fresh training structure | Fresh holdout structure | Exact placement | Placement tokens |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8/8 | 8/8 | 6/6 | 1/24 | 88/170 |
| 12 | 8/8 | 8/8 | 6/6 | 1/24 | 88/170 |
| 13 | 8/8 | 8/8 | 6/6 | 1/24 | 88/170 |
| 23 | 8/8 | 8/8 | 6/6 | 1/24 | 90/170 |
| 40 | 8/8 | 8/8 | 6/6 | 1/24 | 89/170 |

Only these five candidate states were probed. Full familiar/copy/reference/KL
checks were evaluated at the source and step 40. This does not establish the
behavior of every intervening state or rule out every possible operating point.

## What was actually trained

Original rung-0 placement and teacher-KL objectives, weights and batches were
retained. A fixed 0.20 closing-punctuation loss was added before the shared
0.25 gradient clip. No projection was applied. Eight fresh training prompts were
regenerated before updates 1, 6, 11, 16, 21, 26, 31 and 36. Six separate fresh
holdout prompts never supplied gradients.

All 64 refresh examples were usable; none were rejected. Each supervised only
the punctuation token decoding to `"},"`. Generated argument text was conditioning
context only, with its output labels masked. This does not mean shared model
parameters or argument generation are unaffected by punctuation training.

**The intended broken-ending repair was not exercised in this run:** none of the
64 collected argument bodies were unclosed. Only one differed from its source
prefix, at the step-36 refresh. Target `Q2JK` generated the argument
`the current Kubernet`, followed by valid closing punctuation. That wrong argument
was not relabeled as a correct task answer.

The result demonstrates preservation for this fixed recipe, alongside insufficient
copying gains. It does not demonstrate repair of broken JSON after changed
arguments. The additional objective may have altered the copying update, but this
run did not measure separate objective gradients or include a matched-update
control, so it does not establish the cause of the reduced copying gain.

## Conclusion and next decision

Keep the existing source checkpoint. Do not promote this experiment. A useful
next diagnostic would measure the punctuation and placement gradient interaction
on training-only examples before selecting a different loss balance. Any direct
repair test also needs fresh broken generated prefixes so that the proposed
mechanism is actually exercised. No further run was launched.

No GPU, model export, deployment or ClawAgent integration occurred. Source restore
and unchanged teacher state hashes passed.

## Verified evidence

Source identity, state hash, original configuration, original placement values,
template, distillation data and batch schedule matched the previous artifact.
The artifact ZIP digest and embedded data digest passed independent verification;
refresh timing, training-only membership and pure punctuation labels also passed.

- [Completed run](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34395767464)
- [Full report, data and summary](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34395767464/artifacts/10121745514)
- [Machine-readable verification](../reports/ember-generated-closure-verified.json)
- [Prespecified experiment](ember-generated-closure-plan.md)

Evaluated commit: `829084a03446fccc62d82ddc9b19b9b6380fbe70`.
ZIP SHA256: `8184897340e23be052ea83df027ba2c93e8094989f4fd59da52c462a55a86952`.
