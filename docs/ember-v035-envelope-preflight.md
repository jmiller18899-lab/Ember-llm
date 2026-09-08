# Ember v0.0.35 known-tool agent-envelope preflight

v0.0.34 correctly stopped before training because its baseline envelope was not stable: 55/90 JSON-valid tool envelopes and 38/90 correct tool names. The failure clustered around tool names that were introduced by the experiment (`lookup`, `fetch_url`, and `read_file`) rather than demonstrated by the v0.0.8 agent evaluation.

v0.0.35 preserves that failed result and changes only the prompt/tool vocabulary. It still evaluates the pinned v0.0.31 step-479 checkpoint on the same 90 held-out values, with two distractor values in every request, and still has no optimizer or GPU path.

Every system line is copied exactly from `config/ember_v0.0.8_eval.json` for one of three demonstrated tools:

- `weather` for entity values, argument `location`.
- `calculator` for digit strings and expressions, argument `expression`.
- `web_search` for short codes, long codes, model IDs, URLs, paths, and mixed identifiers, argument `query`.

The user request is natural and task-shaped. It names the requested value and explicitly says the other two values are not the requested location, expression, or query. The old rigid `TARGET=...` / `Reply with TARGET` envelope is not used.

The gate is unchanged: at least 86/90 cases must produce a canonical JSON tool envelope and at least 86/90 must name the expected tool. `slot_exact` remains diagnostic, not a baseline pass condition. A high envelope/tool baseline with low `slot_exact` would reproduce the v0.0.32 observation at 90-case scale: the model knows the envelope and routing but places the wrong value.

`training_authorized=false`, `gpu_training_authorized=false`, and `production_authorized=false`. A PASS authorizes no model update; it only makes the placement metric interpretable for a later, separately designed phase.
