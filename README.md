# Ember LLM

Ember is a compact GPT-style model built from scratch in PyTorch and designed
to grow into a native tool-using model for ClawAgent.

Current authoritative package: `ember-v0.0.7-hf-ready.zip`

Ember v0.0.7 keeps the memory-bounded 15M-token corpus pipeline and adds
finite-source corpus safety, a 500-step NVIDIA T4 validation configuration,
checkpoint/resume support, and portable INT4 checkpoint export.

## Completed gate: v0.0.7 T4 validation

The 500-step T4 validation gate is **complete**.

- Verified corpus: `Jmiller18899/ember-corpus-v0.0.7` — `corpus_stats.json`
  reports `PASS` with 15,005,553 Ember tokens (gate: 10M–20M) and a 16,384-piece
  SentencePiece BPE tokenizer.
- Post-fix T4 run: Hugging Face Job `Jmiller18899/6a8e6934984507d9db4e508b`
  (`t4-small`) completed all 500 steps with finite losses (best validation loss
  4.33) and exported INT4.
- Persisted artifacts in `Jmiller18899/ember-v0.0.7-t4`:
  `checkpoints/ember-agent-v0.0.7-t4-validation-20260826T042138Z/best.pt`
  (~332 MB) and `best.int4.pt` (~18 MB), plus training logs and Trackio metrics
  (`trackio/ember.db`).
- Trackio dashboard: `https://huggingface.co/spaces/Jmiller18899/ember-trackio`.
- A CPU inference smoke test loads `best.pt` and generates non-empty output,
  including Ember's agent tool-call special tokens.

## Reproducing the gate

1. Add the repository Actions secret `HF_TOKEN` (a token with write access to
   the `Jmiller18899` namespace).
2. Build/verify the corpus: run **Ember Hugging Face Jobs** in `corpus` mode
   (or confirm `Jmiller18899/ember-corpus-v0.0.7` reports `status: PASS`).
3. Run the local CPU validation: `python -m pytest -q` on the extracted package
   and `python -m py_compile jobs/*.py`.
4. Launch the paid T4 validation with **Ember Hugging Face Jobs** in `train`
   mode. `jobs/ember_hf_train.py` runs `scripts/preflight.py --require-cuda`,
   trains 500 steps, exports INT4, persists checkpoints, and publishes the
   Trackio dashboard. Launch paid T4 training only after the corpus and CPU/INT4
   preflight both pass.

The corpus workflow never launches GPU training.

## Next milestone

The v0.0.8 long-run milestone is prepared but not launched. It keeps the
validated v0.0.7 architecture and tokenizer, starts a coherent fresh 3,000-step
schedule, persists a resumable checkpoint every 500 steps, and gates promotion
with a deterministic 12-case tool-use evaluation plus INT4 inference.

The **Ember Hugging Face Jobs** workflow exposes three manual milestone modes:

- `eval-v007` records the CPU baseline;
- `train-v008` launches the cost-gated `t4-small` run only after explicit
  approval; and
- `eval-v008` evaluates the candidate on CPU and records the promotion result.

Merging these changes does not launch a job. Follow
[`docs/ember-v0.0.8-runbook.md`](docs/ember-v0.0.8-runbook.md) for the execution
order, recovery rules, private artifact locations, and promotion gates. Keep
v0.0.7 authoritative until the candidate reports `promotion_eligible: true`;
ClawAgent integration is deliberately deferred until then.
