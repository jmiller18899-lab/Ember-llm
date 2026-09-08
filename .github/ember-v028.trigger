mode: bootstrap
purpose: keep the Ember v0.0.28 trigger disarmed; no preflight and no paid GPU session is launched by merging this branch
source_checkpoint_repo: Jmiller18899/ember-v0.0.27-t4
previous_result: reports/ember-v0.0.27-result.json (promotion FAIL, progress FLAT, sequence_margin_health -0.8871 -> -0.6846)
runbook: docs/ember-v0.0.28-boundary-focus.md
protected_baseline: v0.0.27 (expanded exact-copy 0.2222, expanded continuation 0.8205, margin health -0.6846)
curriculum: unchanged since v0.0.26 (jobs/ember_sft_data_v026.py)
learning_rate: 1.2e-6, unchanged since v0.0.26
steps: 600, unchanged since v0.0.26
changed_this_phase: boundary-banded mining, out-of-band down-weighting, sequence_margin 1.20 -> 0.60, objective spans the first token and EOS
decision_gate: the free CPU preflight prints EMBER_V028_SEQUENCES_WITHIN_REACH and the worst-margin histogram; a near-empty band means do not launch the T4
next_action: run mode "audit" (free), then "preflight" (CPU), then decide
