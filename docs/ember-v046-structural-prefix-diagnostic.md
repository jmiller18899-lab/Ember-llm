# Ember v0.0.46 structural-prefix diagnostic

v0.0.45 showed that the six stable 84/90 residual failures are not one mechanism. Two long-code failures (`system_target_long_code_02`, `system_target_long_code_05`) are entry-suppression cases: forcing only `<|tool|>` restores a canonical correct-tool envelope. Four failures remain invalid even when the marker is forced:

- `system_target_short_code_02`
- `system_target_short_code_03`
- `system_target_long_code_08`
- `system_target_path_05`

v0.0.46 keeps the pinned v0.0.31 step-479 checkpoint and the measured v0.0.44 prompt stack unchanged. It does not search prompts and does not train.

## Structural probe

The fixed non-value canonical prefix is:

`<|tool|>{"name":"web_search","arguments":{"query":"`

The runner teacher-forces that prefix token by token and records, at every position, the expected token's rank, probability, margin against the best alternative, and whether it is the greedy choice. Semantic stage labels cover the marker, JSON open, `name`, `web_search`, `arguments`, and `query` boundaries.

The probe stops immediately after the opening quote for the query value. No target-value token is forced, compared, supervised, or scored. `slot_exact` therefore remains an honest future learning target.

## Cohorts

The structural-prefix cohort contains 17 cases: the four non-rescued failures plus all 13 passing cases in the same structural subtypes after excluding the already-understood entry-suppression case `system_target_long_code_02`.

The two entry-suppression failures are tracked separately with their first-token competitors and their one-marker rescue rechecked, but they do not enter the structural-prefix classification.

The exact v0.0.8 four-case reference control must remain 4/4, and the 24-case v0.0.45 residual cohort must reproduce the same six failure IDs before the structural result is accepted.

## Interpretation

If the compact canonical prefix is stable on at least 80% of matched passes, the diagnostic can distinguish:

- failures that diverge within the structural prefix;
- failures whose whole structural prefix remains top-1 and therefore diverge later, at or after the value boundary;
- a mixed regime.

If matched passes themselves do not reliably prefer the compact prefix token-by-token, the report records that canonicalization is not stable enough for a binary localization rather than forcing a conclusion.

## Authorization boundary

`training_authorized=false`, `gpu_training_authorized=false`, `production_authorized=false`, and `placement_learning_authorized=false` remain hard requirements. There is no optimizer, backward pass, CUDA path, checkpoint write, promotion, deployment, or integration action in v0.0.46.
