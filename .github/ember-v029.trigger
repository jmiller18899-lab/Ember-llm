mode: bootstrap
purpose: keep Ember v0.0.29 disarmed after completed training and a reporting-only failure; prevent duplicate paid GPU launches
source_checkpoint_repo: Jmiller18899/ember-v0.0.28-t4
preflight_job: Jmiller18899/6a9f774fe686246ca69a9d38
training_job: Jmiller18899/6a9f78a8259f8e97255edb0a
training_commit: 26d1333bcdf0177683d0747da0d86f5b68d0662b
source_checkpoint: checkpoints/ember-agent-v0.0.28-boundary-focus-repair-20260908T022648Z/best.pt
source_sha256: 2d09554553b1a47ea9d7f2e81858e2c3e317c76c6c5472c6fe1ffdf447896da4
v028_best_step: 599
training_status: model training completed; Hugging Face job marked ERROR only because final reporter raised KeyError gate_distance after best.pt/latest.pt upload
recovered_report: reports/ember-v0.0.29-recovered-result.json
v029_selected_best_step: 479
v029_progress: ADVANCED
v029_promotion: FAIL
recovered_final: expanded exact-copy 32/90, expanded continuation 642/730, legacy exact-copy 6/9, legacy continuation 65/69, first-token top-1 87/90
recovered_margin_health: continuation-only -0.3833099212, full-sequence -0.4361307849
legacy_exact_copy_gate: met; short_by 0 cases
legacy_continuation_gate: met at 65/69; short_by 0 tokens
expanded_exact_copy_gate: needs 36/90; short_by 4 cases
expanded_continuation_gate: needs 657/730; short_by 15 tokens
reporting_bug: v029_progress returns deltas/verdict but final logger reads progress.gate_distance and progress.band_flow; wire v029_gate_distance(final,cfg) and v029_band_flow(baseline,final,cfg) into the returned progress object before any future v0.0.29 rerun
format_parity: PASS
boundary_mining: PASS
sequence_worst_k: 2
hardware: t4-small
learning_rate: 1.2e-6
max_steps: 600
next_action: do not rerun v0.0.29 T4; validate/reuse the saved v0.0.29 best checkpoint and fix reporter in the next code evolution
