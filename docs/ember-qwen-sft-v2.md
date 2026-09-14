# Qwen conversation repair v2

V1 improved template matching but failed natural generalization. The paired
30-request evaluation had 14/15 exact answers for each model. Assistant rubric
review scored the original base 26/30 and v1 24/30, with new identity confusion,
an unsupported "Shortened" claim, and an omitted fact in a rewrite. Both models
invented a parcel arrival and failed a basic subtraction. Full raw responses and
review judgments are in `reports/ember-qwen-sft-v1-natural-review.json`.

V2 restarts from the original pinned Qwen3.5-2B BF16 base, not the v1 adapter. It
uses the same short system instruction and completion-only rank-8 LoRA, but halves
the learning rate to 1e-5 and limits training to 72 optimizer steps (one epoch at
effective batch eight). The 576 deterministic synthetic training examples are
balanced across greetings, copy, extraction, classification, drafting, writing,
clarification, and arithmetic. Multiple task forms teach both message perspectives,
missing information, unsupported actions, and retention of all supplied facts.

Evaluation distinguishes three suites:
- 64 new parameterized confirmation examples, using disjoint names but shared
  training task patterns. This is narrow compositional evidence.
- 30 consumed natural-v1 cases, retained unchanged solely as regression checks.
- 16 newly handwritten natural requests with frozen rubrics and different wording.

All suites run on both the base and final adapter. Flexible prose and explanations
require review; exact-match counts only apply to explicit exact-format requests.
No aggregate exact-match score silently treats paraphrases as failures. Human
review remains pending even when every automated check passes. No deployment or
production promotion occurs automatically.

The CPU preflight validates every training example and all natural prompt lengths,
then runs two tiny hybrid-model LoRA updates using the same pinned dependencies.
Only after this passes may the single-launch guard submit the paid L4 job. The
two-hour timeout caps compute at $1.60 at the listed $0.80/hour rate. Checkpoints
are pushed every 18 steps to private `Jmiller18899/ember-qwen3.5-2b-sft-v2`, with
Trackio metrics and paired answers preserved separately from v1. No prior evidence
or checkpoint is overwritten.
