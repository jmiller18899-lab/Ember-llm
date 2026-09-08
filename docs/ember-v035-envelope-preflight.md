# Ember v0.0.35 known-tool agent-envelope preflight

v0.0.34 correctly stopped before training because its baseline envelope was not stable: 55/90 JSON-valid tool envelopes and 38/90 correct tool names. The failure clustered around tool names that were introduced by the experiment (`lookup`, `fetch_url`, and `read_file`) rather than demonstrated by the v0.0.8 agent evaluation.

v0.0.35 preserves that failed result and changes only the prompt/tool vocabulary. It still evaluates the pinned v0.0.31 step-479 checkpoint on the same 90 held-out values, with two distractor values in every request, and still has no optimizer or GPU path.

Every system line is copied exactly from `config/ember_v0.0.8_eval.json` for one of three demonstrated tools:

- `weather` for entity values, argument `location`.
- `calculator` for digit strings and expressions, argument `expression`.
- `web_search` for short codes, long codes, model IDs, URLs, paths, and mixed identifiers, argument `query`.

The gate is unchanged: at least 86/90 cases must produce a canonical JSON tool envelope and at least 86/90 must name the expected tool. `slot_exact` remains diagnostic, not a baseline pass condition.

`training_authorized=false`, `gpu_training_authorized=false`, and `production_authorized=false`.

## Measured result

GitHub Actions run `34280170317` evaluated commit `d8468a3b5f27dd18db32d0c12c4c95cd7abfede7` and returned **FAIL**, as required by the baseline gate.

- Focused tests: **141 passed**.
- Envelope JSON valid: **45/90 (50.0%)**.
- Correct tool name: **44/90 (48.9%)**.
- Slot evaluable: **44/90 (48.9%)**.
- Slot exact: **0/44 (0%)**.
- Right envelope + correct tool + wrong value: **44/90 (48.9%)**.
- Clean stop: **88/90 (97.8%)**.

The known-tool restriction materially clarified routing: 44 of the 45 JSON-valid envelopes (97.8%) named the expected tool. The remaining failure is primarily envelope production, not tool-name selection. Per kind, JSON-valid counts were short_code 7/10, long_code 7/10, digits 4/10, model_id 7/10, url 0/10, path 2/10, entity 8/10, expression 3/10, and mixed 7/10.

Inspection of the original v0.0.32 CPU artifact explains why this prompt was still the wrong baseline. Before v0.0.32 training, its strongest examples of the intended failure mode were single-target literal requests: for example, calculator literal produced 4/4 canonical envelopes with the correct tool name and 0/4 exact argument values in the training probe; weather literal also produced 4/4 correct envelopes/tool names and 0/4 exact values. v0.0.35 added two distractors to every request, an experimental requirement not present in the proposed agent-style baseline.

Per the gate contract, no optimizer, GPU job, promotion, deployment, or integration was run. The next preflight should remove the extra distractors, first verify the exact v0.0.8 reference tool prompts against the pinned checkpoint, then measure the 90 held-out values with one natural requested value per case. A future learning phase remains blocked unless the 90-case JSON-valid and correct-tool baselines both reach the configured threshold.
