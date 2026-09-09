# Ember JSON-closing protection — verified result

Completed on 2026-09-09 in **12 minutes 16 seconds**. All **37 tests passed**.
The projection mechanism passed its checks on all 40 updates, but the
**preselected step-40 scientific endpoint failed**.

## Comparison at step 40

| Measurement | Original unprotected recipe | JSON-closing projection |
| --- | ---: | ---: |
| Familiar valid JSON / correct tool | 82/90 | 82/90 |
| Source-passing familiar cases retained | 82/84 | 82/84 |
| Exact placement | 2/24 | 1/24 |
| Placement token top-1 | 95/170 | 93/170 |
| Copy protection | FAIL | FAIL |
| Reference controls | 4/4 | 4/4 |

The common source started at 84/90 familiar tool structure, 1/24 exact placement
and 88/170 placement tokens. The protected endpoint gained 5/170 tokens
(2.94 percentage points) and no exact placement case. It missed both original
learning requirements: at least three percentage points and one additional exact
case. Placement is the existing conditional teacher-forced value-token probe,
not end-to-end task accuracy.

## Observed intermediate states

| Observed step | Familiar short-code cases retained | Fresh anchors valid | Fresh holdout valid | Exact placement | Placement tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 8/8 | 8/8 | 6/6 | 1/24 | 88/170 |
| 12 | 8/8 | 8/8 | 6/6 | 1/24 | 90/170 |
| 13 | 8/8 | 8/8 | 6/6 | 1/24 | 90/170 |
| 23 | 8/8 | 7/8 | 6/6 | 1/24 | 91/170 |
| 40 | 6/8 | 6/8 | 6/6 | 1/24 | 93/170 |

At the observed steps 13 and 23, the protected run retained familiar short-code
cases that the original run had lost. At step 40, both lost the same two:
`system_target_short_code_01` and `system_target_short_code_09`.
Only the listed steps were probed. The first failure time between observed states
was not measured, and this run cannot rule out every intermediate operating point.
Full familiar/copy/reference/KL checks were run on the source and at step 40.

## Why the local closing-token protection was insufficient

All eight protected closing tokens remained the top choice **at their original
source prefixes**, including at step 40. Two of those same examples nevertheless
produced invalid JSON when allowed to generate freely:

- `json_anchors_0_04`, target `RPCU`: the source argument was `the newest stable
  Rust release`; the candidate instead generated `RSDU` and stopped before
  closing the string.
- `json_anchors_0_06`, target `PB7R`: the candidate similarly generated `P7R5R`
  and stopped with its argument string open.

In both cases, the first token difference was at response position 7 (zero-based),
while the protected closing token was at position 13 in the source response.
The candidate did not reach the protected prefix. This demonstrates a coverage
limitation of protecting source-prefix decisions: it does not ensure closure
after changed argument text. It does not prove this is the only failure mechanism.

The six separate fresh holdout examples retained valid structure, but had zero
exact argument matches at the endpoint. Their structural retention must not be
reported as task success. The eight anchor examples also had zero exact arguments.

## Mechanism and other guards

- Eight independent directions were constructed from fresh structurally valid
  responses, with six four-character and two five-character targets.
- The eight source closure margins ranged from 4.26997 to 8.46118. These were
  strong source decisions, not a sample selected for low closure margins.
- All 40 actual updates were nonzero and passed the float32 projection check.
  Maximum actual residual fraction was 0.00009183, below the fixed 0.001 limit.
- Retained proposal norm averaged 99.5421%, ranging from 99.1917% to 99.6508%.
  The experiment did not establish a pure direction effect versus norm reduction;
  it did not include a norm-matched control.
- At the endpoint, tool KL was 0.000669122 and copy KL 0.000239162, both within
  their original 0.08 budgets. Reference controls held at 4/4.
- Copy protection failed only `expanded_cases_retained`: `12+33+18` was lost and
  `acct_ulq7-2291` gained. Aggregate copy-rate checks held, but individual-case
  retention did not.

The original data, batch schedule and immutable source checkpoint were compared
with the prior trajectory artifact and matched. Pristine restore and unchanged
teacher hashes passed. ZIP and embedded data hashes were independently verified.

## Next focused test

A next experiment should test closing the JSON after **fresh candidate-generated
argument prefixes**, because those are the contexts missed here. Any structural
supervision should label only the required closing suffix; wrong argument values
must not be relabeled as correct task answers. Keep the independent copying
objective, fresh evaluation examples and all existing preservation requirements.
This is a proposal, not a claim of a fix; no further experiment was launched.

No GPU training, checkpoint export, promotion, deployment or ClawAgent integration
occurred. The source checkpoint was not modified.

## Evidence

- [Completed CPU run](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34387531160)
- [Full report, data and summary artifact](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34387531160/artifacts/10118782801)
- [Verified machine-readable findings](../reports/ember-json-closure-verified.json)

Evaluated commit: `69a73fcd5c0811c1a6ed0a79a54e7ab1e0269c48`.
Artifact ZIP SHA256: `761c84e17e69de7855be8c8a4816e747adaea7cc05576fd785bfd6ca9884b23f`.
