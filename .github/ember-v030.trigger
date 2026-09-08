mode: bootstrap
purpose: disarm Ember v0.0.30 after successful CPU contract/preflight; no paid GPU training
source_checkpoint_repo: Jmiller18899/ember-v0.0.29-t4
source_best_step: 479
preflight_job: Jmiller18899/6a9f7f37259f8e97255edc0e
previous_result: reports/ember-v0.0.29-reporter-defect.json (training ADVANCED, artifacts complete, reporter crashed after upload)
runbook: docs/ember-v0.0.30-consolidation.md
protected_baseline: v0.0.29 (expanded exact-copy 32/90, expanded continuation 642/730, legacy exact-copy 6/9, legacy continuation 65/69)
legacy_gates: both PASS
remaining_promotion_distance: expanded exact copy short_by 4 cases (32/90 -> 36/90); expanded continuation short_by 15 tokens (642/730 -> 657/730)
preflight_status: PASS
reporter_contract: repaired; v0.0.30 preflight completed without the v0.0.29 gate_distance/band_flow reporting failure
format_parity: PASS
output_repo_write_check: PASS
curriculum: unchanged since v0.0.26 (jobs/ember_sft_data_v026.py)
sequence_worst_k: 2
learning_rate: 1.2e-6
max_steps: 600
changed_this_phase: source checkpoint only, plus reporter repair/producer-consumer contract protection
next_action: v0.0.30 T4 is evidence-supported; require explicit approval before one paid T4 launch
