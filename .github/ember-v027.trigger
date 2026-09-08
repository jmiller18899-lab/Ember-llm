mode: bootstrap
purpose: keep the Ember v0.0.27 trigger disarmed; no preflight and no paid GPU session is launched by merging this branch
source_checkpoint_repo: Jmiller18899/ember-v0.0.26-t4
previous_result: reports/ember-v0.0.26-result.json (promotion FAIL, progress ADVANCED)
runbook: docs/ember-v0.0.26-plateau-breakout.md
protected_baseline: v0.0.26 (expanded exact-copy 0.2222, expanded continuation 0.8151, legacy continuation 0.8696)
curriculum: unchanged from v0.0.26 (jobs/ember_sft_data_v026.py)
learning_rate: 1.2e-6, unchanged from v0.0.26
changed_this_phase: worst-position sequence hinge, sequence-level hard mining, continuous margin-health selection
next_action: run mode "audit" (free), then "preflight" (CPU) and record the baseline sequence_margin_health before considering "train"
