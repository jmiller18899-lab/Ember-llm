# Ember LLM

Ember is a compact GPT-style model built from scratch in PyTorch and designed
to grow into a native tool-using model for ClawAgent.

Current authoritative package: `ember-v0.0.6-hf-ready.zip`

Ember v0.0.6 contains a memory-bounded 15M-token corpus pipeline with capped
near-dedup shingle work and allocator cleanup for standard GitHub-hosted CPU runners, a 500-step NVIDIA T4 validation configuration,
checkpoint/resume support, and portable INT4 checkpoint export.

## Current gate

1. Add the repository Actions secret `HF_TOKEN`.
2. Run **Ember v0.0.6 Corpus Pipeline** in `source-smoke` mode.
3. Run it in `full-corpus` mode.
4. Start paid T4 training only after the corpus and CPU/INT4 preflight both pass.

The corpus workflow never launches GPU training.
