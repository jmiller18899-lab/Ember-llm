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

## Current milestone result

The v0.0.8 3,000-step run and its CPU evaluation are complete. Training and
technical checks passed, fixed validation loss improved by 29%, and INT4 loaded
successfully. Promotion remains blocked because the held-out evaluation scored
0/4 correct tool names and 1/4 direct responses without an unnecessary tool
call. v0.0.7 therefore remains authoritative.

## Next milestone

v0.0.9 remains the last accepted experimental checkpoint. The later copy
canaries on Hugging Face failed their internal smoke gates:

- v0.0.11 (`Jmiller18899/ember-v0.0.11-t4`) learned EOS stopping and tool
  JSON shape but memorized a closed city/tech pool (`Ann Arbor`, `Boston`).
- v0.0.12 (`Jmiller18899/ember-v0.0.12-t4`, run
  `ember-agent-v0.0.12-copy-canary-20260902T220906Z`) initialized from that
  failed checkpoint, trained 40 T4 steps, and ended `failed_internal_smoke`
  with tool/direct/result rates of 0. The model kept valid `<|tool|>` JSON
  but substituted values (`Rapidton-7669` → `Rapids-3588`,
  `Maplewood-5253` → `Seattle`). Direct answers collapsed under 8x semantic
  weight.

Ember v0.0.13 is a short unique-copy canary initialized from the accepted
v0.0.9 checkpoint, not from v0.0.11 or v0.0.12. Each example has one short
copy target, semantic weight 3.0, and 160 T4 steps. The leftover
`.github/ember-hf.trigger` marker is reset to `bootstrap`, so a merge may
re-run the CPU persistence check. It does not launch GPU training.

The **Ember Hugging Face Jobs** workflow exposes three manual milestone modes:

- `eval-v007` records the CPU baseline;
- `train-v008` launches the cost-gated `t4-small` run only after explicit
  approval; and
- `eval-v008` evaluates the candidate on CPU and records the promotion result.
- `preflight-v009` validates the SFT data, tokenizer boundaries, checkpoint, and
  masked loss on CPU;
- `sft-v009` launches the explicitly approved tool-routing SFT;
- `eval-v009` evaluates that new candidate with the unchanged held-out gate;
- `preflight-v013` validates the short-copy canary on CPU from v0.0.9; and
- `sft-v013` launches the explicitly approved short-copy T4 canary.

Merging these changes does not launch GPU training. Follow
[`docs/ember-v0.0.8-runbook.md`](docs/ember-v0.0.8-runbook.md) for the execution
order, recovery rules, private artifact locations, and promotion gates. Keep
v0.0.7 authoritative until the candidate reports `promotion_eligible: true`;
ClawAgent integration is deliberately deferred until a candidate reports
`promotion_eligible: true`.
