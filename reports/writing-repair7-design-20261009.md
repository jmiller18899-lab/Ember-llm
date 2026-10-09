# WR7 controlled small-edit experiment — October 9, 2026

## Evidence and rationale
WR6 job 6ac64a21e7a0dae8a277b372 completed all 128 steps. Read-only workflow 37966076198 retrieved results at Hub revision f1dcced77b4ac6c32bd3f9cbbd620a4023153efb.
The frozen benchmark improved from 714/744 to 716/744, but shortening copies increased from 5 to 6; development copies increased from 4 to 5. Its preset copying gates failed. No promotion is warranted from aggregate score alone. Full semantic review is still pending.

This experiment follows the October 3 source-copying investigation's narrower recommendation, not WR6's broader hedge-removal curriculum. It is not Grok distillation.

## Controlled change
Start from Repair2 revision daf938bba5d4e6b650ec9d34a2d3ac56706cf549.
Reproduce the original WR5 export (SHA-256 887c0df22cb668afb33e52c65b217e42597e437658790566a3d44ccb177255bf), then replace exactly the 16 little-cut rows. Preserve all other 496 rows, row IDs, order after shuffle, optimizer and generation settings.
The new references cover optional complementizers, redundant quantifiers, shorter time expressions, consent wording, purpose phrases and reminders. Their character reductions are about 3–17%; references retain owners, counts, universal scope, modality, conditions and deadlines. An assistant reviewed all 16 instantiated references; this is not independent human validation.
The original 64 development cases remain in the same order; eight separate compact-edit cases are appended, never trained on. These explicitly protect everyone/every, reminders, obligations, recommendations, consent and negation. No reserved final holdout is loaded.

## Validation
Fail closed if the original export hash differs, anything other than the 16 intended rows changes, row order changes, new training prompts/sources overlap prior WR1–WR6 data or used evaluations, a benchmark four-token frame overlaps, a reference is not shorter, protected fields change, or tokenization exceeds 384 tokens. The existing answer-only masking and frozen 744-case evaluation remain in place.
Training reproduces the 714/744 Repair2 baseline before any optimizer update.

## Execution and acceptance
One l4x1 GPU job, 90-minute timeout, one epoch, 128 steps, learning rate 7.5e-7, accumulation 4, warmup 8, seed 431. Checkpoints persist every 32 steps, final adapter and evidence in a new private Hub repository. Trackio metrics are logged and persisted as JSON.
No automatic retry or promotion. A launch reservation prevents duplicate runs.
Acceptance retains the previous numerical gates (at least 715/744, at most 5 benchmark copies, no shortening/protected-family regressions, at least +3 net semantically acceptable development answers, no new development copies). Additionally inspect the eight compact cases individually for meaning preservation. Old and new development subsets must be reported separately, and the reserved holdout is only considered after all earlier gates pass.
