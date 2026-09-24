# Ember Writing Repair 1 — audited data release v1.2

Data preparation only. No training, model inference, promotion, or deployment was performed.

## Contents

| Set | Examples | Composition |
| --- | ---: | --- |
| Training | 512 | 160 shortening + 96 recipient perspective + 256 existing-skill practice |
| Development | 64 | 32 shortening + 32 recipient perspective |
| Final writing holdout | 96 | 48 shortening + 48 recipient perspective |
| Model-only holdout | 96 | 32 arithmetic + 24 extraction + 16 grounding + 16 clarification + 8 direct response |

Existing-skill training: 64 arithmetic, 64 extraction, 48 grounding, 48 clarification, 32 direct-response examples. These are newly generated checked references, not benchmark replay and not evidence that Ember already passes them. Recipient training is balanced between the recipient's property and another person's property (48 each). Shortening has 80 paired contrasts within training.

These are synthetic examples from authored sentence structures, not 512 independent natural conversations. Splits use different source/prompt structures and entity pools. Exact normalized prompts and content-derived masked structures are checked across splits. There is no claim of complete semantic decontamination or real-user generalization.

## Reproduce the exact files

The JSONL files are included in the accompanying conversation ZIP. This GitHub branch stores reproducible builders, tests, this usage record and the machine-readable audit; it does not contain the generated JSONL files themselves.

From this repository checkout:

```sh
python jobs/ember_writing_repair1_release.py --output writing-repair1-package
python -m unittest discover -s tests -p 'test_ember_writing_repair1*.py' -v
```

The canonical release entrypoint verifies the base construction snapshot, applies documented reference corrections, and produces the five exact file hashes in `reports/ember-writing-repair1-data-audit-20260924.json`.

Use ONLY this file for training, relative to the generated package:

```text
data/ember_writing_repair1/train/sft_train_only.jsonl
```

It has 512 rows with only `id`, `prompt`, `answer`. `train/train.jsonl` is an annotated representation of the SAME 512 examples. Do not concatenate both or use a directory wildcard. Evaluation files and unit tests must never enter training. The export helper rejects evaluation and mixed-split rows.

Use answer-only loss with prompt masking and an end-of-message token. Keep the established Policy v3 and tokenizer pins. When a tool would intercept a training arithmetic prompt, use the base Ember system prompt for model practice, never the tool answer as the system message. The pinned tokenizer measured a maximum training sequence of 227 tokens; all rows fit 256 without truncation.

## Audit and evaluation limits

The final HF CPU audit `Jmiller18899/6ab59def6b030d633f68f805` completed on 2026-09-24. All 89 scoped tests passed on Python 3.11.2 (25 data tests plus 64 existing grading tests). All 768 references passed the declared bounded data-contract checks. Local and remote dataset hashes matched. No exact overlaps were found against 1,230 unique normalized prompts from the pinned baseline, original benchmark and Repair2 training curriculum. The complete repository suite was not run.

**The new writing holdout is not ready for fully automatic meaning grading.** The unchanged conservative `meaning-preservation-v2` grader flags structural paraphrases in the author-intended reference answers:

| Shortening references | Pass | Fail | Review |
| --- | ---: | ---: | ---: |
| Training (160) | 128 | 0 | 32 |
| Development (32) | 0 | 20 | 12 |
| Final writing holdout (48) | 0 | 28 | 20 |

These are grader/reference compatibility counts, not Ember scores. Use the supplied source text, reference and per-case rubric for blinded manual meaning review, including flagged reference adjudication. Keep automatic and manual judgments separate. Do not count review as a pass, weaken the frozen baseline grader, or alter correct references just to satisfy literal checks. The builder's `validate_reference` is a data audit, not a general semantic-equivalence or model-output grader.

No Ember outputs have been generated for the new holdouts. Freeze the audit's hashes before training and do not use final holdout results to select checkpoints. The model-only holdout must bypass tools. Use the development set for tuning.

Author self-review was performed; no independent human review is claimed. Historical grader and runtime are unchanged. The old corrected main-suite baseline remains 181/192.

## Pins

- Audited builder/test source: `e77bbb848f1d0610863361adabaa9fe5ec0c47e1`
- Established baseline record: `e6c9d37120802349190d9d66eaf35e7301a64c2e`
- Fixed grader: `25924014c0e5d5a580a296b2841a1e6f6cbe3bb4`
- Base/tokenizer: `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Existing adapter: `Jmiller18899/ember-qwen3.5-4b-repair2@daf938bba5d4e6b650ec9d34a2d3ac56706cf549`
