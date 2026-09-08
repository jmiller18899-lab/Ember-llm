mode: bootstrap
purpose: keep the v0.0.31 next-promotion path disarmed after the held-out CPU evaluation
source_model_repo: Jmiller18899/ember-v0.0.31-t4
source_best_step: 479
copy_promotion: PASS
int4_export_job: Jmiller18899/6a9f9151e686246ca69a9f48
int4_export: PASS
int4_path: checkpoints/ember-agent-v0.0.31-multi-position-consolidation-20260908T042320Z/best.int4.pt
evaluation_job: Jmiller18899/6a9f9190e686246ca69a9f4c
evaluation_status: PASS
promotion_eligible: true
valid_tool_call_rate: 1.0
direct_response_rate: 1.0
tool_result_response_rate: 1.0
all_generations_nonempty: true
fixed_validation_loss: 3.5808298587799072
baseline_fixed_validation_loss: 4.457988262176514
relative_validation_loss_improvement: 0.19676103924247512
technical_gate: PASS
int4_smoke: PASS
semantic_quality_note: formal legacy evaluator passes, but raw completions remain noisy after the first scored segment; require a stricter semantic quality gate before production ClawAgent integration
next_action: do not rerun this promotion; design/run a stricter semantic tool-argument and response-quality gate before deployment
