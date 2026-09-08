# Ember v0.0.43 three-fold residual calibration

v0.0.42 established the strongest non-regressing envelope baseline so far: **84/90 (93.3%)**, with both kind-level and subtype-level protection gates passing. URL reached 10/10 and mixed reached 10/10. The global requirement remains **86/90 (95%)**, so placement learning is still blocked.

## Frozen v0.0.42 gains

v0.0.43 freezes the entire v0.0.42 safe stack before calibration. In particular:

- `url/one_mixed` keeps the v0.0.41 `quoted_text` gain;
- `url/two_segment` keeps the v0.0.42 `tool_query` gain;
- `mixed/upper` keeps the v0.0.41 `literal_query` gain;
- all URL and mixed cases are ineligible for replacement in v0.0.43.

The remaining six held-out envelope failures are localized to five structural subtypes:

- `short_code/len4`
- `short_code/len5`
- `long_code/4x4`
- `long_code/3x5`
- `path/plain_leaf`

Only two additional successes are needed to clear the global 95% baseline gate.

## Three-fold synthetic selection

Each residual subtype is calibrated on three deterministic, disjoint synthetic folds: `select`, `confirm`, and `stress`. Each fold contains six exact-subtype values excluded from all 90 held-out targets and corruptions.

The frozen v0.0.42 safe prompt is always the baseline candidate. Three new natural request framings are tested:

1. `lookup_exact`: `Use web_search to look up this exact <type>: "<value>".`
2. `search_literal`: `Search the web for exactly this <type>: "<value>".`
3. `web_lookup`: `Look up the exact <type> "<value>" with web_search.`

For short and long codes, `<type>` is `code`; for paths it is `file path`.

A challenger may replace the frozen baseline only if it:

- does not lose to the baseline on any of the three folds for either canonical envelope or correct tool name;
- strictly beats the baseline on at least two of the three folds;
- gains at least two cases combined across all three folds.

This rule is deliberately stricter than the one-fold v0.0.41 selector while avoiding an impossible requirement to beat a baseline on a fold that is already 6/6.

## Gates

After selection is frozen, the exact same 90-case held-out battery is measured once. All gates must pass:

- exact v0.0.8 reference control: 4/4 canonical envelopes and correct tool names;
- global envelope/tool baseline: at least 86/90 (95%);
- kind floors: `short_code >= 8`, `long_code >= 7`, `url >= 10`, `path >= 9`, `mixed >= 10`;
- subtype floors: `short_code/len4 >= 5`, `short_code/len5 >= 3`, `long_code/4x4 >= 5`, `long_code/3x5 >= 2`, `url/one_mixed >= 4`, `url/two_segment >= 3`, `path/plain_leaf >= 3`, `mixed/upper >= 3`.

`slot_exact` remains diagnostic only. `training_authorized=false`, `gpu_training_authorized=false`, and `production_authorized=false`.

A v0.0.43 PASS would establish a sufficiently strong envelope baseline for designing a separate value-placement learning phase. It would not itself train, promote, deploy, or integrate a checkpoint.
