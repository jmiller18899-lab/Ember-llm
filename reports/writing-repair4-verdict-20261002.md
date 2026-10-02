# Ember Writing Repair 4: verdict

HF job `6abd34a9404719ba37613b46` (`l4x1`, 128 steps, final train loss 0.326).
Candidate: `Jmiller18899/ember-qwen3.5-4b-writing-repair4-20260930` at
`48d389e9221e72e5f5386746c33d044282ea068b`, adapter subfolder `candidate`.
**Verdict: rejected.** `Jmiller18899/ember-qwen3.5-4b-repair2` at
`daf938bba5d4e6b650ec9d34a2d3ac56706cf549` remains the reference.
The 96-case final writing and 96-case model-only holdouts were not loaded.

Evidence is the saved output of review run
[36746433780](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/36746433780),
committed with checksums under `reports/evidence/writing-repair4-20260930/`.
Per-row grades are in `reports/writing-repair4-diagnostic-review-20261002.json`;
`python jobs/ember_writing_repair4_review.py` recomputes every count below.

## Frozen 744-case benchmark

| | Pass | Fail | Review |
|---|---|---|---|
| Repair2 (reproduced in the same job) | 714 | 26 | 4 |
| WR4 | **713** | 27 | 4 |

- Regressions (3): `promo_v2/draft-short-00`, `promo_v2/draft-short-05`,
  `fresh_writing/fresh-shorten-03`. In all three WR4 returned the source
  sentence unchanged.
- Improvements (2): `suite_v3/drafting-01` ("her notebook" became "your
  notebook" for Priya) and `probes/probe-context-00` (a full answer instead of
  "Ines's item; Friday.").

## 64-case fresh diagnostic, manual semantic review

23 of 64 answers changed. 10 of those changes only add "Hi " before the
recipient's name, copying the fixed greeting used in every WR4 message
reference. The greeting is style and was not graded.

| Family | Repair2 pass / fail / review | WR4 pass / fail / review |
|---|---|---|
| shortening (32) | 17 / 15 / 0 | 17 / 15 / 0 |
| recipient (32) | 28 / 1 / 3 | 27 / 1 / 4 |
| **Total** | **45** | **44** |

Changed rows: 11 pass→pass, 9 fail→fail, 1 fail→pass, 1 pass→fail,
1 pass→review.

## Targeted WR3 weak spots

| Weak spot | Repair2 | WR4 | Reading |
|---|---|---|---|
| Compact requests (8 shortening structures) | 17/32 | 17/32 | No gain. "Please" was kept more often (dropped 8→4), but "check that" still became "ensure" in 2 rows, 2 rows moved the obligation onto the reader, and 3 more rows were returned unchanged. |
| Recipient possessives become "your" | 16/16 | 16/16 | Already correct before training. |
| Third-party owner named from ownership context | 4/8 | 3/8 | No gain. her 2/2 → 2/2, his 1/2 → 0/2, their 1/4 → 1/4. Leaving "the headset" is not wrong but drops the owner, so it is graded review. |
| Certainty and modal wording | 23/24 | 23/24 | No change. "should" became "will" in the same row before and after. "might/not certain" and "advice, not a limit" are kept in 8/8. Separately, WR4 turned one "must stay" into "Please keep" (counted under compact requests). |
| Negative-claim scope | 16/16 | 16/16 | Already correct; "I have not sent it" and "it has not been sent" are copied exactly. |
| "only after" conditions | 0/4 | 0/4 | Still dropped in every row. WR4 changed "can" to "may" in two, which is better permission wording, but "only" is still lost. |

## Reading

- **No clear diagnostic gain.** The fresh diagnostic falls 45 → 44 and the
  benchmark 714 → 713. Most of the movement is the greeting style.
- **The shortening regression is source copying, not padding.** Diagnostic
  rows returned verbatim went 1 → 4 of 32; benchmark shortening rows went
  5 → 8 of 30. The only non-verbatim outputs that were not shorter swap one
  word ("by" → "at") and appear in both runs. Mean output length rose from
  0.789 to 0.826 of the source on the diagnostic and 0.797 to 0.819 on the
  benchmark.
- **The curriculum likely caused it.** Every WR4 shortening reference was
  already strictly shorter than its source, but only slightly: training
  templates average 0.887 of source length and diagnostic references 0.870.
  Repair2 was already cutting more than the references (0.789). Training
  moved the model toward light trims, and sometimes no trim at all.
- **Training introduced "ensure".** Three of 16 WR4 shortening templates use
  "Please ensure", two of them replacing other verbs. On the diagnostic,
  "Please check that" became "Please ensure", changing a request to verify
  into a request to cause.
- The two benchmark improvements are a recipient possessive and a fuller
  context answer. They do not offset losing shortening on the skill WR4
  targeted.

## Scope

Grades are a self-review against the WR4 references and protected fields, not
independent human judgments. Only saved outputs were used: no model was
loaded, no new inference ran, and no paid job was launched. The benchmark
counts come from the job's own grader (`meaning-preservation-v2`, commit
`25924014c0e5d5a580a296b2841a1e6f6cbe3bb4`).
