mode: bootstrap
purpose: keep the Ember v0.0.33 trigger disarmed; no preflight and no paid GPU session is launched by merging this branch
source_checkpoint_repo: Jmiller18899/ember-v0.0.31-t4 (promoted, best step 479)
why: reports/ember-v0.0.32-rerun-and-training-proposal.json -- all four tool calls produced a correct envelope, tool name and argument key and failed only arguments_grounded, on a checkpoint that copies 43/90 held-out strings exactly
intended_variable: where the copy target sits in the completion
confounds: the implementation also changes the prompt wording, adds a tool/key vocabulary, adds scaffolding tokens to the loss at weight 1.0, and reshapes the supervised span; a result cannot attribute itself to placement alone without a control run
curriculum_values: unchanged since v0.0.26 (jobs/ember_sft_data_v026.py)
objective: unchanged since v0.0.29 (worst-k=2 boundary-focused sequence hinge)
learning_rate: 1.2e-6, unchanged since v0.0.26
steps: 600, unchanged since v0.0.26
changed_this_phase: completions become tool-call envelopes with the copied value in the argument slot; copy weight on the value, envelope_token_weight on the scaffolding
new_metric: envelope_slot_exact_rate -- the v0.0.32 arguments_grounded check over 90 held-out cases instead of 4
protection: the v0.0.31 bare-value result (43/90 exact, 662/730 continuation, 6/9 and 65/69 legacy) must not regress
untested_assumption: whether the tokenizer merges across the value's boundaries inside a JSON string; the preflight reports the encodable fraction and fails below 0.90
next_action: run mode "audit" (free; now also runs the CPU test suite), then "preflight" (CPU) and read EMBER_V033_TRAIN_ENCODING and EMBER_V033_BASELINE_SLOT_EXACT before considering "train"
