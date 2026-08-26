# Ember v0.0.8 long-run runbook

Ember v0.0.8 is a cost-gated 3,000-step training milestone. Merging the
readiness changes does **not** launch a Hugging Face Job. Every evaluation and
training job must be selected manually from the **Ember Hugging Face Jobs**
workflow.

## Fixed experiment contract

- Source package: `ember-v0.0.7-hf-ready.zip`, verified by SHA-256 before use.
- Corpus: private `Jmiller18899/ember-corpus-v0.0.7`, which must still report
  `PASS`, 10M-20M tokens, and a 16,384-piece tokenizer.
- Architecture and tokenizer: unchanged from the completed v0.0.7 T4 gate.
- Initialization: fresh weights. Do not initialize from the 500-step v0.0.7
  checkpoint because that checkpoint used a different cosine schedule.
- Schedule: 3,000 optimizer steps, 150 warmup steps, batch size 8, gradient
  accumulation 4, and context length 512.
- Training exposure: exactly 49,152,000 token positions, approximately 3.28
  passes over the verified 15,005,553-token corpus.
- Durable recovery point: `resume/latest.pt` is uploaded every 500 steps with
  a checksum and matching manifest. A rerun may resume only a v0.0.8
  checkpoint produced by the same pinned configuration.

The prior 500-step T4 validation took about 12 minutes. The 3,000-step run is
expected to take roughly 75-90 minutes including evaluation, checkpointing,
INT4 export, and Hub uploads; the job timeout is three hours. Treat that as an
operational estimate, not a billing quote.

## Execution order

1. Merge the readiness PR only after **Ember CPU Validation** is green.
2. Manually dispatch `eval-v007`. This records the deterministic v0.0.7
   baseline in the private model repository. It is a CPU job and launches no
   GPU, but Hugging Face may still bill CPU Jobs usage.
3. Inspect the baseline artifact and confirm its technical checks completed.
4. Obtain explicit approval for one paid `t4-small` run. Do not select
   `train-v008` before that approval.
5. Manually dispatch `train-v008` exactly once. Do not start a parallel copy.
   The launcher has a run-state lock and refuses a recent live run or a model
   already marked training-complete.
6. If the job is interrupted, confirm there is no live job, then dispatch
   `train-v008` again. It will validate and resume the latest durable v0.0.8
   checkpoint rather than start over.
7. After training reaches `training_complete_pending_eval`, manually dispatch
   `eval-v008`. This is a CPU job.
8. Promote the candidate only when `evaluations/latest.json` reports
   `promotion_eligible: true`. ClawAgent integration is a separate, later
   change.

## Evaluation and promotion gates

Both versions run the same 12 held-out deterministic prompts: four tool-call
requests, four direct responses, and four responses to supplied tool results.
Generation uses seed 1337 and greedy decoding (`top_k: 1`). The candidate must:

- improve loss by at least 2% on the same 128 evenly spaced validation-corpus
  sequences used for both v0.0.7 and v0.0.8;
- produce valid marked JSON tool calls for at least 25% of tool cases;
- answer at least 75% of direct-response cases without an unnecessary tool
  call;
- answer at least 75% of tool-result cases without recursively calling a tool;
- avoid regression against each corresponding v0.0.7 structural rate; and
- load and generate non-empty output from the INT4 checkpoint.

These are minimum promotion gates, not a claim that the model is production
ready.

## Private artifacts

Training writes to `Jmiller18899/ember-v0.0.8-t4`:

- `run-state.json` - current lock and lifecycle status;
- `resume/latest.pt` and `resume/manifest.json` - interruption recovery;
- `resume/train.jsonl` - most recently persisted training log;
- `checkpoints/<run-id>/best.pt` - full checkpoint;
- `checkpoints/<run-id>/best.int4.pt` - portable INT4 checkpoint;
- `config/ember_agent_t4_long_v0.0.8.json` - exact run configuration;
- `evaluations/<timestamp>-candidate-v0.0.8.json` and
  `evaluations/latest.json` - evaluation and promotion decision; and
- `trackio/ember.db` - experiment metrics.

The baseline evaluation is stored under `evaluations/` in
`Jmiller18899/ember-v0.0.7-t4`.

## Recovery rules

- `running` state newer than four hours: assume a live job; do not relaunch.
- stale `running` or `error` state: verify Hugging Face Jobs has no active copy,
  then rerun the same `train-v008` mode to resume.
- checkpoint without a manifest, manifest without a checkpoint, checksum
  mismatch, or configuration mismatch: stop for manual review.
- `training_complete_pending_eval`, `evaluation_complete`, or `complete`:
  never relaunch training.
- failed promotion gate: keep v0.0.7 authoritative and review the evaluation;
  do not connect v0.0.8 to ClawAgent.
