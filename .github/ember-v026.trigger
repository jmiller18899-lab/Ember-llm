mode: bootstrap
purpose: keep the Ember v0.0.26 trigger disarmed; no preflight and no paid GPU session is launched by merging this branch
source_checkpoint_repo: Jmiller18899/ember-v0.0.20-t4
protected_baseline: v0.0.20 (exact-copy 0.4444, continuation top-1 0.8406)
analysis: reports/ember-v0.0.25-plateau-analysis.json
runbook: docs/ember-v0.0.26-plateau-breakout.md
curriculum_audit: FORMAT_PARITY on jobs/ember_sft_data_v026.py, FORMAT_GAP on jobs/ember_sft_data_v015.py
legacy_battery_cases: 9
expanded_battery_cases: 90
learning_rate: 1.2e-6 (v0.0.25 ran at 1.8e-7)
next_action: run mode "audit" (free), then "preflight" (CPU) and record the 90-case baseline before considering "train"
