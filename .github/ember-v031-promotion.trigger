mode: bootstrap
purpose: keep the v0.0.31 next-promotion path disarmed while the held-out CPU evaluation runs
source_model_repo: Jmiller18899/ember-v0.0.31-t4
source_best_step: 479
copy_promotion: PASS
int4_export_job: Jmiller18899/6a9f9151e686246ca69a9f48
int4_export: PASS
int4_path: checkpoints/ember-agent-v0.0.31-multi-position-consolidation-20260908T042320Z/best.int4.pt
evaluation_job: Jmiller18899/6a9f9190e686246ca69a9f4c
next_gate: held-out ClawAgent tool-routing/direct-response evaluation
next_action: observe this CPU evaluation only; do not submit another run
