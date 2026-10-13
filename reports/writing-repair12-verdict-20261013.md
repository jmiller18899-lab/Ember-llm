# WR12 completion review — October 13, 2026

WR12 completed training and evaluation but produced no checkpoint meeting the preset target-improvement rules. The selector correctly chose unchanged Repair2 at step 0. The exported candidate scores 714/744, below the 715 acceptance floor and below WR11 step 26’s 715/744. Keep Repair2 as the reference and WR11 as the existing experimental candidate; do not promote WR12 or treat its candidate folder as newly trained weights.

## Checkpoint evidence

The shortening development set has 80 cases: 40 observed WR11 cases and 40 fresh vocabulary variants. This differs from WR11’s 40-case development set, so totals must not be compared without separating the groups.

| Step | Passes / 80 | Copies / 80 | Reminder omissions | Force failures | New target passes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 13 | 20 | 9 | 21 | 0 |
| 13 | 13 | 20 | 9 | 20 | 0 |
| 26 | 13 | 20 | 9 | 20 | 0 |
| 39 | 13 | 20 | 9 | 20 | 0 |
| 52 | 13 | 20 | 9 | 22 | 0 |

All 13 baseline passing shortening cases remain passes at all checkpoints. No new development copies appear. No checkpoint turns a baseline reminder/force failure into a verified pass, so all trained checkpoints are ineligible. Step 13 changes one observed force failure to unresolved semantic review and adds a timing error on a different already-failing case. Steps 26/39/52 also introduce new force-error reasons; step 39 changes a fresh review case to a force failure. Lower aggregate error counts do not establish successful rewrites.

At selected step 0, observed WR11 shortening stays 7/40 passes, 10 copies, 5 reminder omissions and 11 force failures. Fresh WR12 shortening stays 6/40 passes, 10 copies, 4 reminder omissions and 10 force failures. Benchmark shortening copies stay 5/30 and not-shorter stays 7/30. Selected benchmark and development outputs are identical to their Repair2 baselines: zero recorded regressions and zero improvements.

## Verified identity and scope

Training job `6acda7a1b0ad3e9da5ea2dd2` completed all 52 steps. Created 2026-10-13T03:38:09.794Z and finished 2026-10-13T04:19:32.625Z (41m23s including hardware queue/build; not a measured GPU billing duration). Training loss 5.955198214604304; loss values are not directly comparable with WR11 because WR12 adds an auxiliary term. Trained checkpoint files differ from the parent, so training ran and changed weights.

Code: `9dfb10cb793cf479df038c7e49c0b406ba2dca6e`. Private output: `Jmiller18899/ember-qwen3.5-4b-writing-repair12-20261012`. Evidence pinned to Hub revision `cb0fc1cb45ef5b88df82a5e5b8e23ddd66e7308d`; transferred evidence SHA-256 `a2407a74457f51273e512331128c470e4c898bb6fd2fc7d68e7d99b254325dc4`. Candidate upload revision `f77cb06a41952c8e6df697df038b860a5c4e51b6`.

Candidate and checkpoint-0 files both have LFS SHA-256 `7695770af18d43e7ca02a9888f0bc2fa87b67c93920f11ddbaa37969ee753beb`, matching the previously verified Repair2 adapter file. The candidate LoRA parameter digest equals the starting digest `f3e11133143ec3f20ea36669a2ed7de0c4db168b6bde8cf351705cc4fea11a5f`; the final trained digest differs. Exporting the baseline here is deliberate rejection of trained checkpoints, not evidence of a failed optimizer or a trained improvement.

Recomputed the selector from all saved per-case snapshots, checked unique IDs and aligned outputs for 744 benchmark plus 112 development records, recomputed the 714 score, and confirmed unchanged selected outputs and adapter identity. Preparation passed 1,461 tests and the real-tokenizer preflight. No grader/router changes or final-holdout evaluation occurred.

Review examined saved grades and selected outputs. Raw intermediate-checkpoint responses were not persisted by this trainer; the fail-to-review paraphrases therefore have not received an independent text review. This limitation prevents claiming a semantic fix from the reduced early force count. No additional model inference, training retry, deployment or promotion was launched during completion review.

## Next decision

The experiment does not support repeating this semantic-token loss configuration as a successful repair. Before another training run, inspect the small set of intermediate fail-to-review paraphrases and improve the linguistic quality and variety of reminder/force training examples. Some reused templates contain singular/plural errors such as “each bookbinders”; this is a data-quality finding, not a proven cause of WR12’s result. Keep the current evaluation rules fixed when reporting this run.
