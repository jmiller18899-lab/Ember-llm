mode: bootstrap
purpose: keep the v0.0.31 next-promotion path CPU-only and disarmed by default
source_model_repo: Jmiller18899/ember-v0.0.31-t4
source_best_step: 479
copy_promotion: PASS
next_gate: held-out ClawAgent tool-routing/direct-response evaluation
required_before_eval: best.int4.pt exists and reconstructs successfully
next_action: export-int4 on CPU, then eval on CPU; no GPU training in this workflow
