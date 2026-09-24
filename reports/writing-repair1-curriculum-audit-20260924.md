# Writing Repair 1: curriculum and test-set audit

Date: 2026-09-24. Version: `writing-repair1-curriculum-v1.1`.
Source commit: `faa1571c583654a18a32210ddfd5d32444acfe40`.
Branch: `experiment/ember-writing-repair1-curriculum-20260924`.

**Data construction and audit: PASS. Automatic evaluation of the new writing sets using the frozen grader alone: NOT READY. No training or model inference was performed.**

## Delivered data

| Split | Shortening | Perspective | New retention practice | Total |
|---|---:|---:|---:|---:|
| Train | 160 | 96 | 256 | 512 |
| Development | 32 | 32 | 0 | 64 |
| Final test | 48 | 48 | 0 | 96 |

The 96 training perspective examples contain 48 recipient-owned and 48 third-party-owned cases. Development is 16/16; final test is 24/24. Shortening covers requests, requirements, recommendations, permission, optionality, no obligation, prohibition, possibility, probability, conditions, quantities, timing, and explicit subjects/actions.

Retention is 64 arithmetic, 48 extraction, 32 clock arithmetic, 48 grounding, 32 clarification/supplied-text transformations, and 32 direct-answer examples. These are newly authored or algorithmically generated rehearsal examples, not literal replay from an earlier training file. Grounding has 24 sufficient-evidence and 24 insufficient-evidence cases. Clarification has 16 missing-text and 16 supplied-text cases.

The training export contains only `id`, `prompt`, and correct `answer`. `train_metadata.jsonl` describes the same 512 rows; it is NOT an additional training split. Semantic annotations, rejected alternatives in unit tests, and final-test answers are not included in the training export.

## Verification evidence

Final CPU audit: Hugging Face job `Jmiller18899/6ab598146b030d633f68f729`, COMPLETED at `2026-09-24T21:37:38.890Z`, Python 3.11.2. It rebuilt all four JSONL files from the saved GitHub commit and matched the local checksums exactly.

- 95 scoped tests passed: 31 curriculum/release tests and 64 existing grading tests.
- All 672 authored reference records passed the bounded semantic-ledger or algorithmic-answer checks.
- No duplicate normalized prompts within or across the new splits; no cross-split source-text reuse.
- No exact prompt/source overlap detected against the 744 pinned baseline entries, 120 original benchmark rows, and 444 Repair2 training rows. This is 1,308 reference entries, not necessarily 1,308 unique examples.
- After masking people, possessions, dates, and scenario payloads, no identical prompt skeleton was shared across train/development/final-test splits.
- A three-word-gram Jaccard near-duplicate scan found no pair at or above 0.85. Maximum similarities: train/dev 0.0769, train/test 0.1081, dev/test 0.0851, all new records versus pinned history 0.2000. This heuristic is not a proof against all semantic contamination.
- Tokenization included the pinned Policy v3 system prompt, user prompt, reference answer, and end token. Train length 60-249 tokens; development 92-121; final test 98-130. No row exceeded 256. Longest reference answer including end token: 38. The audit script's conservative 512-token recommendation is not required by these measured lengths.
- Every new record routes to the model under the existing policy: train 512, development 64, final test 96. No answer was produced by running a model.

The pinned baseline suite manifest remained `dcd42ca2c869f83f3bf46569baeca9199b6d0b0c38eab98ec8beefac0ac6fe32`. The frozen grader remained SHA256 `e134bf3919ea2871e7a10d3e90de877996f9c631414203762ab09c012bc42a9b` at commit `25924014c0e5d5a580a296b2841a1e6f6cbe3bb4`.

The initial tokenizer-only audit failed because Jinja2 was missing, after 87 tests passed. Adding Jinja2 fixed the audit environment. A later final release removed accidental associations: state queries always answering 'closed', constant grounded values per topic, possessions always paired with the same time, and the lack of a threshold equality example. Regression tests exposed those shortcuts before the release changes; all now pass. No shortening reference was changed by this release.

## Important grading finding

Passing the data audit does not mean the frozen grader understands these new phrasings. Applying `meaning-preservation-v2` to the authored shortening references produced:

| Split | Pass | Fail | Review | Total |
|---|---:|---:|---:|---:|
| Train references | 70 | 50 | 40 | 160 |
| Development references | 0 | 24 | 8 | 32 |
| Final-test references | 0 | 40 | 8 | 48 |

Thus 170 of 240 shortening references were not automatically accepted. These are grades on authored reference answers, NOT Ember predictions. The conservative lexical checker does not recognize many valid rephrasings of obligation, permission, or uncertainty. The pattern author judged the targets faithful, and the semantic-ledger checks passed; no independent human adjudication has been performed.

Do not distort the data to make this checker pass. Do not alter the frozen 744-case baseline grader. The new writing sets require a separately declared semantic-review protocol, with the frozen checker reported only as a diagnostic. Keep automatic grades and blind manual/independent judgments in separate fields. Review or unresolved disagreement is not a pass.

## Frozen dataset checksums

| File | Rows | Bytes | SHA256 |
|---|---:|---:|---|
| train.jsonl | 512 | 121792 | `7cfc823e5674cfb409c4c7bad952431eea90fff87cfbe0ebbab95dc95008881e` |
| train_metadata.jsonl | 512 | 376983 | `690e6f0253e8a6bb1bc975141f6259c23f1f535cea8d3cc86b0a3b7df6b31381` |
| dev.jsonl | 64 | 60353 | `99e83610ce8b4d8ddaefe3786d6556dce16644ec81394d1d85e8ff19ac7e6baf` |
| test.jsonl | 96 | 93623 | `8cce1f5e7ee8797b805dfd25260be6b9027ac3d4b5af49e95e46b69c331e0f89` |

Rebuild the exact files from the repository root:

```sh
python jobs/ember_writing_repair1_release.py --output data/writing_repair1
python -m unittest discover -s tests -p 'test_writing_repair1*.py' -v
```

Use `ember_writing_repair1_release.py`, not the original catalog generator, for finalized exports. The complete generated JSONLs are supplied in the conversation ZIP; GitHub stores the deterministic generators, tests, manifest, and this report.

## Evaluation protocol and limits

Train only from train.jsonl. Use development cases for tuning. Select one candidate before evaluating the fixed final test; do not use final-test outputs to select prompts, learning rates, or checkpoints. Authoring/auditing the test references here is not a run of Ember on that test. Pretraining exposure cannot be established.

For new writing, blind the identities of baseline/candidate answers, judge against the source's actors, ownership, facts, force, certainty, polarity, quantities, timing, and conditions, then check concision and sendable-message form. Do not require one exact reference string. No meaning change can be traded for a shorter answer. Retain reason codes and unresolved reviews. The protocol has been written but has not yet been independently calibrated or applied to model outputs.

For existing capabilities, keep the same 744 cases, Policy v3, decoding, and frozen corrected grader. In addition, compare baseline and candidate directly on the original exact-answer tasks without tool interception, reporting those model-only results separately from tool-assisted scores. No new model-only score is claimed here.

This is synthetic, template-based data. The author reviewed transformation patterns and ran per-row integrity checks; this is not an independent semantic review of every row or evidence of real-world generalization. Tests cover the retrieved curriculum and grader files, not the full repository suite. The local workspace was a scoped checkout, and a full repository test run was not performed. No model weights, runtime rules, benchmark files, or deployment were changed; no training was launched.
