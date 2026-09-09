# Ember training-balance CPU diagnostic

One training-only diagnostic to guide the next punctuation loss weight. No new
training candidate, held-out evaluation, export, promotion or GPU run.

## Fixed protocol

Replay the 40 updates from generated-prefix run 34395767464, starting from its
immutable v0.0.31 step-479 checkpoint. The pinned snapshot contains only the
placement training values, discovered template, teacher-KL training rows, eight
punctuation training prompts, original schedule and training replay evidence.
It excludes development, holdout and familiar evaluation rows.

Before updates 1, 13, 23 and 40, measure independent gradients of placement,
tool-KL, copy-KL and punctuation objectives. Report raw and weighted norms and
cosines. The original weights are 0.20/0.55/0.25, with added punctuation 0.20.

At each sampled state, clone the current model and full AdamW history. Compare
one actual update at punctuation weights 0, 0.025, 0.05, 0.10 and 0.20. Each trial
starts from the identical model and optimizer state, uses the same current
training batches and punctuation prefixes, and keeps LR 1e-7, clip 0.25 and
weight decay 0.01. Only the trial's punctuation weight changes. Trial states are
discarded; the replay continues at the original 0.20 punctuation weight.

Report actual update norm, direction relative to the zero-punctuation control,
first-order changes predicted by each component gradient, and actual before/after
training losses. Also report predictions for the zero-punctuation update scaled
to each trial's norm. This separates size and direction locally; scaled-control
predictions are not measured losses from another optimizer step.

## Prespecified recommendation screen

A positive candidate weight is locally eligible only if **at every sampled state**:

- The zero-punctuation update decreases placement loss by more than 2e-6.
- The candidate retains at least 90% of that actual placement-loss decrease.
- Punctuation loss does not increase by more than 2e-6.
- Both measured training-batch teacher-KL losses remain at or below 0.08.

Suggest the smallest eligible positive weight for a future full test. If none
qualify, report no qualifying weight. Do not adjust the rule after measurement.
The recommendation is based only on training data and remains unvalidated for
retention or generalization.

All four sampled states inherit the 0.20 replay history. These local branches
are not full trajectories at alternative weights and cannot establish their
40-step outcome. Gradient signs alone also cannot explain AdamW's finite step.

## Integrity and bounds

Verify the checkpoint/state, snapshot digest and original batch schedule. Require
all refreshed generations to exactly reproduce saved training records, and all
40 replay loss/gradient-norm measurements to match within 2e-5 absolute error.
Hash the live model and optimizer around each diagnostic to prove they did not
change. Restore the source and verify the frozen teacher at the end.

CPU float32, two threads, deterministic, 30-minute script / 40-minute workflow
bound. Four sets of five temporary optimizer steps plus one 40-update replay.
No weights are saved. Local tests: 67 passed, including conflicting gradients,
cloned optimizer isolation, comparison with a real autograd AdamW step and the
recommendation rule's failure cases.
