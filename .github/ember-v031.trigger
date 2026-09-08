mode: train
purpose: launch exactly one explicitly approved Ember v0.0.31 T4 session from the passed CPU preflight; disarm immediately after submission
source_checkpoint_repo: Jmiller18899/ember-v0.0.30-t4
source_best_step: 499
source_run_id: ember-agent-v0.0.30-multi-position-consolidation-20260908T033037Z
source_checkpoint: checkpoints/ember-agent-v0.0.30-multi-position-consolidation-20260908T033037Z/best.pt
preflight_job: Jmiller18899/6a9f8bc0259f8e97255eddc2
preflight_status: PASS
protected_baseline: v0.0.30 (expanded exact-copy 34/90, expanded continuation 651/730, legacy exact-copy 6/9, legacy continuation 65/69, first-token top-1 88/90)
remaining_promotion_distance: expanded exact copy short_by 2 cases; expanded continuation short_by 6 tokens
format_parity: PASS
source_advance_transform: PASS
boundary_mining: PASS
output_repo_write_check: PASS
curriculum: unchanged since v0.0.26
sequence_worst_k: 2
learning_rate: 1.2e-6
max_steps: 600
next_action: submit one T4 only, record HF job id, then return mode to bootstrap
