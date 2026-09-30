# Ember Writing Repair 4

The September 29 saved-answer review rejected WR3: the frozen benchmark went
from 714/26/4 pass/fail/review to 712/28/4. Its development examples missed
compact requests and possessive-pronoun perspective; certainty and negative
claim scope remained weak. This experiment revises the writing curriculum.

The intended result is better meaning-preserving writing with prior abilities
retained. Use the same pinned Repair2 reference, frozen routing and frozen
744-case grader. The 96-case final writing and 96-case model-only holdouts
remain unused. This is an experimental adapter, not an automatic promotion.

## Data

- 128 compact shortening references: preserve request responsibility, facts,
  quantity, timing, modality, ownership and conditions. References must actually
  shorten the source; no fixed percentage cutoff is added to evaluation.
- 128 message references: 64 modal contrasts and 64 negative-scope contrasts.
  Named-owner and pronoun forms have paired targets. Both recipient ownership
  and third-party ownership are represented; her/his/their are covered.
- The exact 256 training-only retention examples remain: arithmetic 64,
  extraction 64, grounding 48, clarification 48, direct answers 32.
- 64 fresh diagnostic examples use separate source structures, names, objects
  and places. They are generated and saved before any optimizer update, then
  repeated after training. Their saved answers require semantic review.
- Check exact prompt/source overlap against the observed frozen benchmark,
  previous WR1/WR2 training/development and WR3 writing. These checks do not
  establish semantic decontamination from every possible source.

Data are authored synthetic references reviewed in this session, not independent
human judgments. Contracts test those references, not model-generated answers.
The benchmark grader is unchanged.

## Fixed experiment

Parent: `Jmiller18899/ember-qwen3.5-4b-repair2` at
`daf938bba5d4e6b650ec9d34a2d3ac56706cf549`.
Base: `Qwen/Qwen3.5-4B` at
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

Same one-epoch settings as WR3: 512 rows, 128 optimizer steps, learning rate
7.5e-7, microbatch 1, gradient accumulation 4, warmup 8, seed 431, maximum
encoded length 384 with no silent truncation. Existing LoRA parameters alone
may change. Save checkpoints every 32 steps in a new private output repository,
including checkpoint zero and the fixed final candidate before evaluation.

Use one L4 GPU (`l4x1`) with the existing 90-minute timeout. Reproduce the frozen
714-pass baseline and save the new 64-case diagnostic baseline before training.
Compare the same 744 cases and fresh diagnostic answers after training. Do not
automatically retry, select checkpoints by final holdout, or deploy new weights.

Trackio logs training metrics locally, and checkpoint/final callbacks persist
the same metrics in the private output model repository. The Jobs logs provide
live progress without creating another hosted dashboard.

## Submission

The current Hugging Face connector does not expose its Jobs operation. Reuse the
existing successful GitHub Actions launcher and its `EMBER_HF_TOKEN` secret;
never copy the credential into code or logs. The `.prepare` trigger runs CPU
contracts and real-tokenizer checks. The separate `.launch` trigger submits one
GPU job only after those checks pass. Keep its returned job ID and URL as the
submission evidence.
