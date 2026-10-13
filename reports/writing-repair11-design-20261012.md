# Ember Writing Repair 11 — prompt-mix ablation

Status: prepared and validated. All 1,452 tests and real-tokenizer preflight passed. No training or promotion has started.

The October 12 diagnostic found Repair2 and selected WR10 step 39 both passed 4/24 prompts, dropped reminders in 5/16 reminder cases, and copied 14/24 sources verbatim. Their output text matched on 23/24 cases. Explicit instructions reduced reminder omissions but increased copying.

## One training variable

Start from Repair2 `Jmiller18899/ember-qwen3.5-4b-repair2` at `daf938bba5d4e6b650ec9d34a2d3ac56706cf549`.

Change only the shortening training request mix. Each of eight groups has six ordinary requests and six explicit preservation requests: 48 ordinary and 48 explicit shortening prompts in total. The other 320 prompts remain identical.

WR10's 416 training sources, targets, rejected copies, rejected over-edits, loss kinds and shuffled order are deliberately reused. This is an ablation, so reuse of this training split is intentional. It does not introduce today's 24 diagnostic prompts or their sources into training. Fresh development vocabulary supplies 56 examples (40 shortening, 16 recipient); their sources differ from WR10 development sources.

Keep WR10's answer-only loss, copy margin and over-edit margin, learning rate 7.5e-7, seed 431, one epoch, 52 optimizer steps, microbatch 1, accumulation 8, warmup 4 and length cap 384. Save steps 13/26/39/52. Frozen routing, v3 checking and existing benchmark acceptance rules remain the same.

## Checkpoint eligibility

Include step 0 (unchanged Repair2) in selection and prefer it on equal quality.

Every snapshot must have complete unique case IDs and consistent per-case/aggregate pass counts. Reject a candidate that loses any passing development shortening case, increases copy count, increases force failures, or increases reminder drops. Rank eligible snapshots by passing cases, then fewer copies, then earlier step.

This is a selection safeguard, separate from the training-data variable. Broader benchmark regressions still block promotion; development pass retention alone is insufficient.

## Evaluation and acceptance

Before training, reproduce the frozen 714/744 Repair2 baseline. At checkpoint saves, evaluate the fresh development set. After selecting one checkpoint, run the existing observed benchmark.

Existing acceptance remains: at least 715/744, at most 5/30 benchmark verbatim copies, zero protected-family regressions and zero shortening regressions. Report no gain when step 0 wins. No automatic retry, deployment or promotion.

Today's 24 paired prompts remain development-only. They can be rerun as a secondary comparison; a gain on those observed examples is insufficient evidence of general improvement. Reserved final holdouts remain excluded.

## Run boundary and persistence

Output: private `Jmiller18899/ember-qwen3.5-4b-writing-repair11-20261012`.

One L4 job, 90-minute timeout, estimated 20–30 minutes using the prior run as a rough guide. At the current $0.80/hour rate, the GPU cap is $1.20. Runtime may differ. The job saves intermediate adapters, the selected adapter, inputs and evaluation evidence privately before terminating.

The connected Hugging Face OAuth session currently grants read-repos and jobs but no write-repos scope. Training must use the repository's existing `EMBER_HF_TOKEN` secret, as previous runs did, and verify private writes before optimizer steps. Do not start a job that cannot persist checkpoints.

The new branch workflow first runs CPU contracts and real-tokenizer preflight. Training submission is enabled only by the exact commit message marker `Launch authorized WR11 prompt-mix training`, on this isolated branch, after preflight succeeds.

## Verification so far

Nine WR11 CPU tests pass. Seven new contracts failed against old behavior before implementation. Twelve existing WR10 tests passed at baseline.

The full local unittest attempt found 47 import errors because pytest/torch are absent. The first remote suite passed 1,381 tests but failed 71 because jsonschema was missing from its environment. With that dependency included, the documented pytest command passed all 1,452 tests. Real-tokenizer preflight also passed, with maximum training length 220 against the 384 cap. See remote-validation.json for the exact tested code hashes and the pinned run specification.

Exact isolation checks cover WR3–WR9 writing history, already-observed WR10 development sources, 30 observed benchmark shortening sources, and the diagnostic sources. WR10 training reuse is disclosed separately. These are exact/frame checks, not a semantic all-history overlap guarantee.
