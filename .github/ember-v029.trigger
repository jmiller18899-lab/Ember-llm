mode: bootstrap
purpose: keep the Ember v0.0.29 trigger disarmed; no preflight and no paid GPU session is launched by merging this branch
source_checkpoint_repo: Jmiller18899/ember-v0.0.28-t4
previous_result: reports/ember-v0.0.28-result.json (promotion FAIL, progress ADVANCED, exact copy 20/90 -> 28/90, legacy 4/9 -> 5/9)
runbook: docs/ember-v0.0.29-multi-position.md
protected_baseline: v0.0.28 (expanded exact-copy 28/90, expanded continuation 632/730, legacy exact-copy 5/9, first-token top-1 85/90)
curriculum: unchanged since v0.0.26 (jobs/ember_sft_data_v026.py)
learning_rate: 1.2e-6, unchanged since v0.0.26
steps: 600, unchanged since v0.0.26
changed_this_phase: sequence hinge sums the k=2 worst decisions per row; threshold gates compared with a tolerance; band-transition and gate-distance reporting
open_question: 21 held-out cases sit in [-0.75, 0); converting most of them clears the 0.40 expanded exact-copy gate on its own
next_action: run mode "audit" (free), then "preflight" (CPU) and read EMBER_V029_GATE_DISTANCE plus the band histogram, then decide
