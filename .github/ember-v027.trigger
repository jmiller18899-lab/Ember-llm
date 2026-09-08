mode: bootstrap
purpose: disarm Ember v0.0.27 after successful CPU preflight; no paid GPU training
source_checkpoint_repo: Jmiller18899/ember-v0.0.26-t4
preflight_job: Jmiller18899/6a9f63a5e686246ca69a9afe
source_checkpoint: checkpoints/ember-agent-v0.0.26-format-parity-copy-repair-20260908T005821Z/best.pt
source_sha256: 2a97c4343d0073a74c46017069abb3dce2f729aa70fa39b782c0bc190cefe9e3
v026_best_step: 219
protected_baseline: v0.0.26 (expanded exact-copy 0.2222, expanded continuation 0.8151, legacy continuation 0.8696)
baseline_sequence_margin_health: -0.8871363491482204
baseline_sequences_all_positions_positive: 0.23333333333333334
format_parity: PASS
sequence_mining: PASS
output_repo_write_check: PASS
next_action: review CPU preflight before any explicit train-mode T4 launch
