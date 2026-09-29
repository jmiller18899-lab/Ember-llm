# Ember Writing Repair 3 — September 29, 2026

## Confirmed submission

Hugging Face accepted exactly one job: [Jmiller18899/6abb3d416b030d633f6a2b58](https://huggingface.co/jobs/Jmiller18899/6abb3d416b030d633f6a2b58), created `2026-09-29T04:23:29.406Z`. Direct job inspection confirmed `l4x1`, timeout **5,400 seconds**, and command `uv run --python 3.11 /data/ember_writing_repair3_train.py`. The initial verified state was **SCHEDULING — Waiting for requested hardware to become available**. This is submission confirmation, not evidence that training or evaluation has completed.

The launch workflow [36521168518](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/36521168518) completed successfully on immutable code commit `68302b05c47825c33509ed9c5047f57c22f40842`. Its logs record `WR3_GPU_JOB_SUBMITTED` with the same job ID. [Validation artifact 11012921666](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/36521168518/artifacts/11012921666) preserves this launch's preflight and both complete test reports.

One user-approved L4 job, maximum 90 minutes, **no automatic GPU retry**. No promotion, deployment, or change to the reference model. Do not start a second job merely because the list view temporarily omits this job; use the confirmed ID for inspection.

## Training configuration

- Parent: `Jmiller18899/ember-qwen3.5-4b-repair2@daf938bba5d4e6b650ec9d34a2d3ac56706cf549`, not WR1 or WR2 writing candidates.
- Base: `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- New private output: `Jmiller18899/ember-qwen3.5-4b-writing-repair3-20260929`.
- 512 training examples: 160 substantive shortening, 96 recipient/owner perspective, 256 retention (64 arithmetic, 64 extraction, 48 grounding, 48 clarification, 32 direct).
- 96 separate fresh development examples: 48 shortening and 48 recipient perspective. Development responses require manual semantic review; they receive no automatic semantic pass rate.
- One epoch, 128 optimizer updates, learning rate `7.5e-7`, microbatch 1, accumulation 4, warmup 8, seed 431, BF16, answer-only loss. Save checkpoints every 32 steps and save the final candidate before post-training evaluation.
- Frozen 744-case benchmark before and after. The original 714-pass baseline plus four reviews must reproduce before training. Reviews are not passes. Both final 96-case holdouts remain unused. No checkpoint selection based on holdouts.

## Verified CPU preflight

[Preparation run 36520788132](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/36520788132) completed successfully on code commit `88352ad64b5e9dc47d89f467f08cc249c7c73493`. [Preparation artifact 11012626937](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/36520788132/artifacts/11012626937) contains the logs, both complete test XML reports and comparison JSON. All checks were repeated successfully on the actual launch commit before GPU submission.

All 18 WR3 contracts passed locally and in Actions. Full repository tests under identical dependencies: baseline **1,329 passed, 2 failed**; WR3 **1,347 passed, 2 failed**. Both had 1,574 passing subtests and zero skipped tests. No new failing tests. The repository is **not entirely green**: both versions fail `test_gate_fires_on_every_temporal_case` and `test_explicit_answers` in `tests/test_ember_temporal_scoped_rule.py`. These existing tests and the frozen runtime/grader were left unchanged.

Real pinned tokenizer preflight verified maximum training length **245 tokens** against a 384-token ceiling, with no truncation. Exact normalized prompt/source checks against 1,576 observed historical rows found zero collisions for the new writing/development examples; train/development source structures do not overlap. This is bounded, exact-match validation, not proof against all semantic contamination or prior base-model exposure. References are synthetic and author-reviewed, not independently adjudicated.

Training export SHA-256: `cf65e949439f8f2299b2aa9a7d7e17c1916c01ca5a6c297ae009409cfa7f3cf8`.
Data generator SHA-256: `db60589c1a150d4f1feb15762e2bdc7914ee5dd6d7ed9b5fb8038019b4cbd1a1`.
Frozen benchmark engine: `43220b453807782091a9209384091e82fed086f8`, SHA-256 `e20ccd3ac99de436c4fbd5e586e8bd303201e39965500b137bfcd2fa739d0eeb`.
Frozen grader commit: `25924014c0e5d5a580a296b2841a1e6f6cbe3bb4`.

## Acceptance

Execution success alone does not establish improvement. Inspect saved per-case before/after evidence and fresh development outputs; report automatic benchmark changes separately from semantic review. Do not automatically promote a result or launch a retry.
