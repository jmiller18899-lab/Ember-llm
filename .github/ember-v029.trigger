mode: bootstrap
purpose: disarm Ember v0.0.29 after CPU preflight submission; no paid GPU training
source_checkpoint_repo: Jmiller18899/ember-v0.0.28-t4
preflight_job: Jmiller18899/6a9f774fe686246ca69a9d38
previous_result: reports/ember-v0.0.28-result.json (promotion FAIL, progress ADVANCED, exact copy 20/90 -> 28/90, legacy 4/9 -> 5/9)
runbook: docs/ember-v0.0.29-multi-position.md
protected_baseline: v0.0.28 (expanded exact-copy 28/90, expanded continuation 632/730, legacy exact-copy 5/9, first-token top-1 85/90)
curriculum: unchanged since v0.0.26 (jobs/ember_sft_data_v026.py)
learning_rate: 1.2e-6, unchanged since v0.0.26
steps: 600, unchanged since v0.0.26
changed_this_phase: sequence hinge sums the k=2 worst decisions per row; threshold gates compared with a tolerance; band-transition and gate-distance reporting
next_action: observe CPU preflight only; do not launch T4 until gate-distance and band reports are reviewed
