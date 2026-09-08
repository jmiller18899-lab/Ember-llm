# Ember v0.0.39 targeted envelope matrix

v0.0.38 raised the held-out canonical-envelope baseline from 66/90 to 76/90 by changing only `short_code` and `long_code` user wording. Tool routing remained perfect whenever a canonical JSON envelope appeared, and every evaluable slot still contained the wrong value.

The remaining failures are concentrated in five `web_search(query)` value kinds:

- `short_code`: 8/10
- `long_code`: 6/10
- `url`: 6/10
- `path`: 9/10
- `mixed`: 7/10

The other four kinds (`digits`, `model_id`, `entity`, `expression`) are already 10/10 and are left unchanged in v0.0.39.

## Calibration design

v0.0.39 does not select a prompt on the 90-case held-out battery. Instead it creates four deterministic synthetic values for each weak kind, explicitly excluding every held-out target and corrupt value. Three user-message variants are measured for each synthetic value while keeping the schema-bearing v0.0.32 system line fixed:

1. `current_exact`: `Search the web for this exact <label>: "<value>".`
2. `tool_named`: `Use web_search to find current information about this exact <label>: "<value>".`
3. `lookup_exact`: `Look up this exact <label> on the web: "<value>".`

This produces 60 CPU-only calibration generations: 5 weak kinds × 4 synthetic values × 3 prompt variants. Selection uses only canonical-envelope and correct-tool counts, with a deterministic priority order for ties. `slot_exact` is not used to choose a prompt.

After one variant is frozen per weak kind, the runner measures the same 90 held-out values used by v0.0.36-v0.0.38 exactly once. The four already-perfect kinds keep their v0.0.37 schema-natural prompts unchanged.

## Gates

The exact four v0.0.8 tool prompts remain a control and must pass 4/4 for canonical envelope and correct tool name. The final 90-case gate remains unchanged at at least 86/90 JSON-valid envelopes and 86/90 correct tool names (95%). `slot_exact` remains diagnostic only.

No optimizer is present. `training_authorized=false`, `gpu_training_authorized=false`, and `production_authorized=false`. A failed baseline gate must stop the experiment. Even a passing baseline would only establish a valid starting point for a separate placement-learning design; it would not authorize training automatically.
