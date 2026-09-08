mode: train
purpose: launch one explicitly approved Ember v0.0.26 T4 format-parity copy-repair session after successful CPU audit and preflight
source_checkpoint_repo: Jmiller18899/ember-v0.0.20-t4
protected_baseline: v0.0.20 (legacy exact-copy 0.4444, continuation top-1 0.8406)
preflight_job: Jmiller18899/6a9f5babe686246ca69a9a17
format_parity: PASS (90/90 diagnostic templates supported)
expanded_baseline_exact_copy_rate: 0.18888888888888888
expanded_baseline_continuation_top1_rate: 0.7534246575342466
expanded_baseline_clean_stop_rate: 1.0
expanded_battery_cases: 90
legacy_battery_cases: 9
learning_rate: 1.2e-6
max_steps: 600
hardware: t4-small
approval: explicit user approval in current session
next_action: submit exactly one T4 job, then immediately disarm to bootstrap after Hugging Face returns the job id
