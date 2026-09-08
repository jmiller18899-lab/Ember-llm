mode: bootstrap
purpose: disarm Ember v0.0.29 after successful one-time T4 submission; prevent duplicate paid GPU launches from later pushes
source_checkpoint_repo: Jmiller18899/ember-v0.0.28-t4
preflight_job: Jmiller18899/6a9f774fe686246ca69a9d38
training_job: Jmiller18899/6a9f78a8259f8e97255edb0a
training_commit: 26d1333bcdf0177683d0747da0d86f5b68d0662b
source_checkpoint: checkpoints/ember-agent-v0.0.28-boundary-focus-repair-20260908T022648Z/best.pt
source_sha256: 2d09554553b1a47ea9d7f2e81858e2c3e317c76c6c5472c6fe1ffdf447896da4
v028_best_step: 599
protected_baseline: v0.0.28 (expanded exact-copy 28/90, expanded continuation 632/730, legacy exact-copy 5/9, legacy continuation 62/69, first-token top-1 85/90)
sequences_within_reach: 21/90 in [-0.75, 0)
legacy_exact_copy_gate: met at 5/9; short_by 0 cases
legacy_continuation_gate: needs 63/69; short_by 1 token
expanded_exact_copy_gate: needs 36/90; short_by 8 cases
expanded_continuation_gate: needs 657/730; short_by 25 tokens
format_parity: PASS
boundary_mining: PASS
output_repo_write_check: PASS
sequence_worst_k: 2
hardware: t4-small
learning_rate: 1.2e-6
max_steps: 600
next_action: observe the submitted v0.0.29 training job only; do not launch another T4 session
