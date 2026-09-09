mode: bootstrap
purpose: keep Ember v0.0.31 disarmed after a successful promoted T4 run; prevent duplicate paid GPU launches
source_checkpoint_repo: Jmiller18899/ember-v0.0.30-t4
source_best_step: 499
source_run_id: ember-agent-v0.0.30-multi-position-consolidation-20260908T033037Z
source_checkpoint: checkpoints/ember-agent-v0.0.30-multi-position-consolidation-20260908T033037Z/best.pt
preflight_job: Jmiller18899/6a9f8bc0259f8e97255eddc2
preflight_status: PASS
training_job: Jmiller18899/6a9f8cbee686246ca69a9ec5
training_commit: 8313adc2518bff49f499fe96d56c9b42949ff344
training_status: COMPLETED
promotion: PASS
progress: ADVANCED
run_id: ember-agent-v0.0.31-multi-position-consolidation-20260908T042320Z
best_step: 479
promoted_metrics: expanded exact-copy 43/90, expanded continuation 662/730, legacy exact-copy 6/9, legacy continuation 65/69, first-token top-1 89/90
margin_health: continuation-only -0.1983129014, full-sequence -0.2448485268
expanded_exact_copy_gate: PASS; measured 43/90; short_by 0 cases
expanded_continuation_gate: PASS; measured 662/730; short_by 0 tokens
legacy_exact_copy_gate: PASS; measured 6/9
legacy_continuation_gate: PASS; measured 65/69
clean_stop_rate: 1.0
band_flow: 9 within-reach cases crossed into correct; 0 correct cases fell out
format_parity: PASS
source_advance_transform: PASS
boundary_mining: PASS
output_repo_write_check: PASS
curriculum: unchanged since v0.0.26
sequence_worst_k: 2
hardware: t4-small
learning_rate: 1.2e-6
max_steps: 600
next_action: do not rerun v0.0.31; treat v0.0.31 best step 479 as the promoted checkpoint for downstream evaluation/deployment
