mode: bootstrap
purpose: keep the Ember v0.0.30 trigger disarmed; no preflight and no paid GPU session is launched by merging this branch
source_checkpoint_repo: Jmiller18899/ember-v0.0.29-t4
source_best_step: 479
previous_result: reports/ember-v0.0.29-reporter-defect.json (training ADVANCED, artifacts complete, reporter crashed after upload)
runbook: docs/ember-v0.0.30-consolidation.md
protected_baseline: v0.0.29 (expanded exact-copy 32/90, expanded continuation 642/730, legacy exact-copy 6/9, legacy continuation 65/69)
legacy_gates: both now PASS (exact copy 6/9 >= 5/9, continuation 65/69 = 0.9420 >= 0.90)
remaining_promotion_distance: expanded exact copy 4 cases (32/90 -> 36/90); expanded continuation 15 tokens (642/730 -> 657/730)
curriculum: unchanged since v0.0.26 (jobs/ember_sft_data_v026.py)
learning_rate: 1.2e-6, unchanged since v0.0.26
steps: 600, unchanged since v0.0.26 (v0.0.29 peaked at step 479, so a longer schedule is not indicated)
changed_this_phase: source checkpoint only, plus the reporter repair and a contract test that derives required keys from the transform text
next_action: run mode "audit" (free), then "preflight" (CPU), then decide
