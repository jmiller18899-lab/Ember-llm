# WR10 completion review — October 10, 2026

## Verdict
Training and evaluation completed. WR10 does not pass the preset gates. Repair2 stays the reference. No promotion and no final-holdout evaluation.

## Evidence
- HF job: `6aca8cc1fee2c9007018bf2e`, stage COMPLETED.
- Started 2026-10-10 19:06:46Z. Final report printed 19:30:54Z. Inside the 90-minute timeout.
- Private output: `Jmiller18899/ember-qwen3.5-4b-writing-repair10-20261010`.
- Code commit: `c8f024595f180634a950c41a5f74dcdb0a62a730`.
- 52 optimizer steps, training loss 5.431. Metrics file is non-empty.
- Checkpoints at steps 13, 26, 39 and 52. The `candidate/` adapter is the reloaded step-39 checkpoint.

## Frozen benchmark
Repair2 714/744 to WR10 step 39 at 713/744. One regression: `fresh_writing/fresh-shorten-03` (shortening family 3/6 to 2/6). No strict improvements. Exact, temporal, promo, held-out time, context holdout, suite v3 and probes stayed at the Repair2 pass counts.

Benchmark shortening copies stayed 5/30. Not-shorter stayed 7/30. Mean length ratio moved from 0.797 to 0.780, the same length move WR9 made. Diagnostic copies fell 11/40 to 10/40. Diagnostic v3 passes stayed 6/40 and force-change fails stayed 11. The selection key kept step 39 because copies stayed at or below the Repair2 diagnostic floor.

## Decision
The copy cap held again, and the fresh diagnostic copied one fewer case. The pass floor and the shortening-regression rule still fail on the same reminder item WR9 lost. Do not promote.
