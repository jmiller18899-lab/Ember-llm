# Writing Repair 1 training scope — 2026-09-24

User approved starting training from the audited v1.2 curriculum. This is one experimental continuation of Repair2, not a new base model, promotion, or deployment.

## Fixed run

- Base/tokenizer: Qwen/Qwen3.5-4B at 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.
- Starting adapter: Jmiller18899/ember-qwen3.5-4b-repair2 at daf938bba5d4e6b650ec9d34a2d3ac56706cf549.
- Canonical data builder/usage record: 878a76ee7492504e648071626377a9524b4dec58, data v1.2.
- Training input: only data/ember_writing_repair1/train/sft_train_only.jsonl, 512 rows, SHA256 3352bc50c505dc5cecafab88909b176be3385b45f3346d08e222189019995165.
- Mix: 160 faithful-shortening, 96 recipient-perspective, 256 existing-skill practice examples.
- One epoch; microbatch 1; gradient accumulation 4; 128 optimizer updates; learning rate 1.5e-6; warmup 8; seed 431; max length 256; bfloat16; answer-only loss.
- Preserve existing LoRA architecture and Policy v3. Only LoRA parameters may update.
- Hardware: one L4. Hard job timeout: 3600 seconds. No automatic retries or duplicate launches.
- Output: a separate private repository, Jmiller18899/ember-qwen3.5-4b-writing-repair1-20260924. The final adapter goes in candidate/; the starting adapter and periodic checkpoints are retained separately.

## Evaluation and preservation

Reproduce the frozen corrected 744-case baseline before any optimizer step. Abort on baseline count mismatch. Save all baseline outputs. After training, save the candidate before evaluating it; run the same corrected 744 cases and record all per-case regressions and improvements, not just totals.

Collect before/after outputs for the 64 development writing cases without claiming automatic semantic scores. The new 96-case writing holdout and 96-case model-only holdout are never evaluated by this training runner and never enter the training export. The final checkpoint is fixed in advance, not selected using holdout performance. New writing references and predictions still need the declared semantic-review protocol; this run does not repair or weaken the frozen grader.

Save the initial checkpoint and upload checkpoints every 32 optimizer steps, plus the final candidate. Require 128 completed steps, finite loss, and a changed adapter digest before recording training completion. Runtime promotion remains false regardless of automatic scores.

## Verified before GPU submission

The direct Hugging Face OAuth connection could run jobs but received HTTP 403 when attempting to create the output repository. No GPU was launched on that credential. The existing GitHub Actions training-credential path (HF_2, falling back to HF_TOKEN) then successfully created the private destination and uploaded evidence/preflight.json. GitHub workflow run 36066623338 succeeded; remote test-file commit dbd2be99bbabde373be61be034124101cfd68228. No secret values were printed or copied out.

Twenty-four new trainer-contract tests were run before implementation and failed for the missing contracts. After implementation, the local scoped package passed 49 tests (25 data tests plus 24 trainer tests); pytest also reported 1551 subtests. Python 3.11 syntax parsing passed. The saved trainer matched the locally tested Git blob SHA 3273fb718477f751c7ba49c6214b8812b9a5ee25 and SHA256 2b256b4a5391471ea2f84a6880792ebe2ce3673f090cc7eae13299b294131b02.

GitHub preflight run 36067140136 additionally passed its pinned-byte and targeted data/grader/trainer-test steps as observed at writing time. Its full repository suite and tokenization steps were not yet complete when this scope receipt was written; this document is not a final test report or a training launch receipt.

Review was author self-review, not an independent code or semantic review. No existing model repository, data builder, fixed grader, or runtime-policy file was modified.
