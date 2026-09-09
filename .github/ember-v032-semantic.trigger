mode: bootstrap
purpose: keep the Ember v0.0.32 semantic gate disarmed; no job is submitted by merging this branch
candidate: Jmiller18899/ember-v0.0.31-t4 (promoted, best step 479, promotion_eligible true)
why: the v0.0.31 promotion record asks for a stricter semantic tool-argument and response-quality gate before ClawAgent integration
what_the_legacy_evaluator_checks: a tool marker, a parseable JSON object, a matching tool name, non-empty arguments; for responses, no tool marker and at least three visible characters
what_it_does_not_check: whether the arguments name anything from the prompt, whether required keys exist, whether generation terminated, and anything after the first scored element
rubric: config/ember_semantic_quality_v0.0.32.json
evaluator: jobs/ember_hf_semantic_eval_v032.py (read-only; never trains, uploads or promotes)
offline_proof: tests/test_ember_v032_semantic_gate.py -- eleven completions the legacy rubric accepts and the strict rubric rejects, each asserting its exact failing checks
next_action: run mode "rubric" (free, no token) to confirm the gate can fail, then "evaluate" on cpu-upgrade to score the promoted checkpoint
