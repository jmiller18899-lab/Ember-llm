# Independent direct-answer learning trial

The previous language-quality diagnostic passed 0/24 task checks. This tests a
small, separate repair while preserving the tool router's representation.
Only the final transformer block (`blocks.5`) and output normalization (`ln_f`)
can learn. Token embeddings, the tied language-model output weights, and
blocks 0–4 remain frozen. The runtime's saved heads are not changed.

The CPU trial uses 72 short instruction/answer examples across writing,
rewriting, explaining, summarizing, classification, and planning/comparison.
Twelve separate development examples select the checkpoint by answer-token
negative log likelihood. The unchanged baseline participates in selection.
Training lasts at most 120 optimizer steps and 20 minutes, with batch size 2,
learning rate 0.0002, and snapshots at steps 40, 80, and 120.

Only answer tokens and the end-of-turn marker contribute to loss. Padding and
prompt tokens are masked. Tokenization must preserve the prompt boundary or
the trial stops before training. All frozen parameters are hashed before and
after learning; observed block_04 features must match exactly. The selected
adapter is reloaded before answer evaluation.

The original 24 quality cases are now development evidence. Twelve new direct
requests are consumed only if development loss improves by at least 10% and
the old quality score improves by at least three cases. New confirmation
requests never enter gradient updates or checkpoint selection. Failing the
learning gate leaves that confirmation unconsumed.

The workflow completing successfully means the learning diagnostic ran, not
that the model passed a quality gate. Check `development_progress_gate` and
the raw answers in the report. Deterministic lexical checks are limited and
cannot certify broad semantic quality. This is full-precision research only;
there is no INT4 export, production-pointer update, or automatic promotion.

The workflow preserves adapter tensors, optimizer/RNG state, training-example
order, exact source hashes, per-checkpoint losses, and before/after generations.
Failed tool-confirmation evidence remains separate and is not repaired by a
better direct-answer score.
