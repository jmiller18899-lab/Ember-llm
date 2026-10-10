# WR10 keep-pass experiment — October 10, 2026

## Checked result
WR9 job `6aca7d98fee2c9007018b02c` completed from Repair2. Selected step 13 scored 713/744. Benchmark copies stayed 5/30 and not-shorter stayed 7/30. Mean length on the frozen 30 moved 0.797 to 0.780. The one regression was `fresh_writing/fresh-shorten-03`. No strict improvements. Diagnostic copies, v3 passes and force-change fails were unchanged at the selected step. Repair2 stays the reference. WR9 is not promoted.

`fresh-shorten-03` is a reminder item: "Please remember that the spare key must remain in locker 27 until Monday." The passing rewrite drops only `that`. WR9's source margin rewarded dropping more preamble, which is how a reminder disappears.

## Why the method changes
WR8 penalized meaning-wrong rewrites and the frozen benchmark copied more (5 to 6). WR9 put the margin on the verbatim source and held the copy cap, then lost a reminder pass. The measured pair of failures is now: return the source unchanged, or drop a remember/forget cue while trimming.

WR10 keeps WR9's source margin and adds the missing keep-pass term on the same 96 pairs:

`cross_entropy(short) + softplus(nll(short) - nll(source)) + softplus(nll(short) - nll(overedit))`

The over-edit drops a reminder or changes force/conditions. Shared fact tokens appear in all three sequence losses. v3 still only checks data and checkpoint selection. It is not a runtime gate.

The 256-row retention, 52 optimizer steps, accumulation 8, learning rate `7.5e-7`, and Repair2 parent stay fixed. Checkpoints are saved every 13 steps. The uploaded candidate is the selected checkpoint, after any reload.

## Acceptance
Same floors as WR9: at least 715/744, at most 5 benchmark copies, no shortening or protected-family regressions, no new diagnostic copies. Final holdouts stay unloaded. No automatic retry or promotion. One `l4x1` job, 90-minute timeout, parent Repair2 `daf938bba5d4e6b650ec9d34a2d3ac56706cf549`.
