mode: preflight
purpose: retry Ember v0.0.31 CPU preflight after source-version resolver fix; no paid GPU training
source_checkpoint_repo: Jmiller18899/ember-v0.0.30-t4
source_best_step: 499
source_run_id: ember-agent-v0.0.30-multi-position-consolidation-20260908T033037Z
source_checkpoint: checkpoints/ember-agent-v0.0.30-multi-position-consolidation-20260908T033037Z/best.pt
previous_preflight_job: Jmiller18899/6a9f8b4d259f8e97255eddba (failed before training on resolver version string)
protected_baseline: v0.0.30 (expanded exact-copy 34/90, expanded continuation 651/730, legacy exact-copy 6/9, legacy continuation 65/69, first-token top-1 88/90)
remaining_promotion_distance: expanded exact copy short_by 2 cases; expanded continuation short_by 6 tokens
curriculum: unchanged since v0.0.26
sequence_worst_k: 2
learning_rate: 1.2e-6
max_steps: 600
next_action: read CPU preflight result and re-disarm before any T4 decision
