# WR9 copy-margin experiment — October 10, 2026

## Checked result
Hugging Face job `6ac96474fee2c9007017e696` completed. It trained WR8 from Repair2 for 128 steps (optimizer 581s, wall clock 22:02:33Z–22:26:15Z) and uploaded checkpoints at steps 32, 64, 96 and 128 plus a non-empty metrics file. Selection kept step 64. The frozen benchmark moved 714/744 to 713/744. Benchmark shortening copies moved 5 to 6, and not-shorter moved 7 to 8. Regressions were `suite_v3/drafting-19` and `fresh_writing/fresh-shorten-03`. Diagnostic copies fell from 10 to 8 and force-change fails from 17 to 15. Repair2 stays the reference. WR8 is not promoted.

The `candidate/` folder on that repo is the step-128 adapter. It was saved before the selected checkpoint was reloaded. The graded scores are step 64, stored at `checkpoints/step-64`.

## Why the method changes
WR8's unlikelihood target was a meaning-wrong rewrite (`should` to `must`, a dropped reminder, a dropped `exactly`). That target is not the failure the benchmark measured. The measured failure is returning the source unchanged. WR5 already found that teacher-forced likelihood preferred the verbatim source. WR9 therefore keeps a v3-passing short answer and puts the margin on that source:

`cross_entropy(short) + softplus(nll(short) - nll(source))`

Shared fact tokens appear in both sequence losses. The margin is carried by the extra source wording. v3 still only checks data and checkpoint selection. It is not a runtime gate.

WR8's diagnostic was best at step 64 and gave copies back by step 128. WR9 keeps the full 256-row retention and the 96 fresh shortening pairs, and stops at 52 optimizer steps (accumulation 8). Checkpoints are saved every 13 steps. The uploaded candidate is the selected checkpoint, after any reload.

## Acceptance
Same floors as WR8: at least 715/744, at most 5 benchmark copies, no shortening or protected-family regressions, no new diagnostic copies. Final holdouts stay unloaded. No automatic retry or promotion. One `l4x1` job, 90-minute timeout, parent Repair2 `daf938bba5d4e6b650ec9d34a2d3ac56706cf549`.
