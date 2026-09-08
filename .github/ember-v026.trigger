mode: bootstrap
purpose: disarm Ember v0.0.26 branch trigger after successful CPU preflight; no paid GPU training
source_checkpoint_repo: Jmiller18899/ember-v0.0.20-t4
protected_baseline: v0.0.20 (legacy exact-copy 0.4444, continuation top-1 0.8406)
preflight_job: Jmiller18899/6a9f5babe686246ca69a9a17
format_parity: PASS (90/90 diagnostic templates supported)
expanded_baseline_exact_copy_rate: 0.18888888888888888
expanded_baseline_continuation_top1_rate: 0.7534246575342466
expanded_baseline_clean_stop_rate: 1.0
expanded_battery_cases: 90
legacy_battery_cases: 9
learning_rate_if_trained: 1.2e-6
next_action: review preflight result before any explicit train-mode launch
