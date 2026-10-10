# WR9 completion review — October 10, 2026

## Verdict
Training and evaluation completed. WR9 does not pass the preset gates. Repair2 stays the reference. No promotion and no final-holdout evaluation.

## Evidence
- HF job: `6aca7d98fee2c9007018b02c`, stage COMPLETED.
- Started 2026-10-10 18:02:10Z, finished 18:25:32Z. Optimizer time 546s. Inside the 90-minute timeout.
- Private output: `Jmiller18899/ember-qwen3.5-4b-writing-repair9-20261010`.
- Code commit: `97544ff7d5adac2a4bfe4dbeab178810b7a0f02a`.
- 52 optimizer steps, training loss 4.528. Metrics file is non-empty.
- Checkpoints at steps 13, 26, 39 and 52. The `candidate/` adapter is the reloaded step-13 checkpoint.

## Frozen benchmark
Repair2 714/744 to WR9 step 13 at 713/744. One regression: `fresh_writing/fresh-shorten-03` (shortening family 3/6 to 2/6). No strict improvements. Exact, temporal, promo, held-out time, context holdout, suite v3 and probes stayed at the Repair2 pass counts.

Benchmark shortening copies stayed 5/30. Not-shorter stayed 7/30. Mean length ratio moved from 0.797 to 0.780. Diagnostic copies stayed 13/40, v3 passes stayed 5/40, and force-change fails stayed 12. Steps 26, 39 and 52 each had 11 force-change fails with the same 13 copies. The selection key ties on `v3_pass - copies` and then keeps the earlier step, so step 13 was graded.

## Decision
The copy cap held, which WR6–WR8 did not, and answers on the frozen 30 got shorter on average. The pass floor and the shortening-regression rule still fail. Do not promote.
