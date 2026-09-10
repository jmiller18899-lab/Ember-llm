# Full 40-step run at punctuation weight 0.05 — verified result

**The run completed, but the endpoint failed both learning and preservation.**
Execution took 6 minutes 7 seconds. Local validation passed 70 tests; the
workflow's selected experiment test suite passed 64 tests.

## Endpoint comparison

| Measurement | Source | Punctuation 0.20 | Punctuation 0.05 |
| --- | ---: | ---: | ---: |
| Familiar source-passing cases retained | 84/84 | 84/84 | 83/84 |
| Familiar valid JSON / correct tool | 84/90 | 84/90 | 83/90 |
| Exact placement | 1/24 | 1/24 | 1/24 |
| Placement token top-1 | 88/170 | 89/170 | 90/170 |
| Original copy preservation | Baseline | PASS | PASS |
| Reference controls | 4/4 | 4/4 | 4/4 |
| Original six-example holdout structure | 6/6 | 6/6 | 6/6 |
| Additional fresh holdout structure | 6/6 | Not evaluated | 5/6 |

The 0.05 endpoint gained two placement tokens (1.18 percentage points) and no
exact case. It missed both learning requirements: at least three percentage
points and one additional exact case. Placement is the existing conditional
teacher-forced value-token probe, not end-to-end task accuracy.

The familiar lost case was `system_target_short_code_09` (target `CZ9X`). It
produced `{"arguments":{"query":"C8XX` and stopped without closing the JSON.
The additional fresh lost case was `closure005_holdout_1_02` (target `XA4KC`).
It generated `55B3` without entering the tool envelope at all. These are distinct
observed failure forms; this experiment does not establish a common cause.

All eight punctuation training prompts and the original six-example holdout
retained structure, but neither cohort had any exact argument matches at the
endpoint. The new holdout also had zero exact arguments.

## Observed states

| Step | Familiar short-code retention | Exact placement | Placement tokens | Additional fresh holdout structure |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8/8 | 1/24 | 88/170 | 6/6 |
| 12 | 8/8 | 1/24 | 89/170 | 5/6 |
| 13 | 8/8 | 1/24 | 89/170 | 5/6 |
| 23 | 8/8 | 1/24 | 90/170 | 5/6 |
| 40 | 7/8 | 1/24 | 90/170 | 5/6 |

Only the listed steps were probed. The first observed new-holdout loss was at
step 12; its exact onset was not measured. Likewise, the first observed familiar
loss was at step 40. No sampled state met the learning gate. This is not an
exhaustive search of all intervening states.

## Training and other checks

The run started from the pristine v0.0.31 step-479 source and empty AdamW state,
using punctuation 0.05 from the first update. It used the original placement,
tool-KL and copy-KL weights (0.20/0.55/0.25), learning rate, training data, template,
batch schedule and eight punctuation training prompts. The extra holdout was
selected using source structure only and supplied no gradients.

All 40 updates were finite and nonzero. All 64 refreshed training generations
were usable; three refresh records contained changed argument text, but none
contained an unclosed argument. Thus direct repair of broken generated endings
again was not exercised by the training refreshes.

Copy preservation passed every legacy, expanded-case and aggregate check.
Reference controls passed 4/4. Tool KL was 0.000693925 and copy KL 0.000226261,
both below the original 0.08 budgets. Familiar floors failed only JSON/tool,
short-code kind and four-character short-code subtype checks.

## Conclusion

The earlier single-update recommendation did not hold over a complete 0.05
trajectory. A local training-loss advantage was insufficient to produce exact
copying gains while retaining all existing behavior. Weight 0.05 is now recorded
as **tested and failed**, not a recommended validated setting.

Keep the original source checkpoint. Do not promote either punctuation run.
The results do not justify another weight change by themselves; future work
should address the training-example coverage gap and test structure and exact
values together. No further experiment was launched.

Source identity/state, original recipe/data/schedule, cohort separation, report
checks and artifact hashes were independently verified. Source restoration and
unchanged teacher hashes passed. No GPU, checkpoint export, promotion,
deployment or ClawAgent integration occurred.

## Evidence

- [Completed run](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34431553570)
- [Full report and data artifact](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34431553570/artifacts/10134807731)
- [Machine-readable verification](../reports/ember-closure005-verified.json)
- [Prespecified protocol](ember-closure005-plan.md)

Evaluated commit: `d8aa88575c1ff5bf9c588c32f3fc20738daa2f7b`.
ZIP SHA256: `fb25716843f45bd3237c14a5cd24511f34915595e2161e8a129cd4a922723391`.
