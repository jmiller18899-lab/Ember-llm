# Full 40-update punctuation-weight 0.05 CPU test

Authorized by the user on 2026-09-10. Start from immutable v0.0.31 step 479
and empty AdamW history. Run exactly 40 updates at LR 1e-7, placement 0.20,
tool-KL 0.55, copy-KL 0.25 and punctuation 0.05, with original clipping,
weight decay, batch schedule and eight punctuation training prompts.

Reuse the generated-prefix training method: regenerate training arguments before
updates 1, 6, 11, 16, 21, 26, 31 and 36; label only pure closing punctuation.
Incorrect argument text remains masked conditioning context. The prior 0.20
run and balance diagnostic are historical comparisons, not optimizer history.

Probe steps 1, 12, 13, 23 and 40. Full familiar, copy, reference and teacher-KL
checks run at source and endpoint. Retain the existing six-example holdout as a
regression set. Add six new source-structurally-valid holdout examples (four
four-character and two five-character codes), using a separate fixed namespace
and excluding historical values and every previous cohort attempt. These new
examples supply no gradients. Stop preparation if their source-valid quota fails.

The endpoint must improve exact placement by at least one case and token top-1
by at least three percentage points, retain all 84 source-passing familiar cases
individually, pass every original copy/reference/KL/floor check, retain all source-
valid fresh cohorts, and execute 40 finite nonzero updates. Preserve exact
argument accuracy separately from structural validity. No gates are weakened.

CPU float32, deterministic, two threads; 30-minute script / 40-minute workflow
bound. No GPU, checkpoint export, promotion or ClawAgent integration. Restore
source weights and verify teacher state. Direction metrics from the earlier
balance diagnostic are not used here. Local validation: 70 tests passed.
