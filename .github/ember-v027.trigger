mode: preflight
purpose: run Ember v0.0.27 audit and CPU preflight only; no paid GPU session
source_checkpoint_repo: Jmiller18899/ember-v0.0.26-t4
previous_result: reports/ember-v0.0.26-result.json (promotion FAIL, progress ADVANCED)
protected_baseline: v0.0.26 (expanded exact-copy 0.2222, expanded continuation 0.8151, legacy continuation 0.8696)
curriculum: unchanged from v0.0.26 (jobs/ember_sft_data_v026.py)
learning_rate: 1.2e-6, unchanged from v0.0.26
changed_this_phase: worst-position sequence hinge, sequence-level hard mining, continuous margin-health selection
next_action: verify CPU preflight sequence_margin_health, loss/mining guards, source checkpoint and writable v0.0.27 output repo before any train-mode launch
