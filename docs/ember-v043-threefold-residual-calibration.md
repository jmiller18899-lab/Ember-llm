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

## Measured result

GitHub Actions run `34286232002` evaluated commit `372fdef0d2a848f10613c66f499d0a6ac03a5219`. The workflow correctly returned **FAIL** because the global 95% envelope/tool gate was not met.

- Focused/inherited tests: **192 passed**.
- Exact v0.0.8 reference envelope: **4/4 (100%)**.
- Exact v0.0.8 reference tool name: **4/4 (100%)**.
- 90-case JSON-valid envelope: **84/90 (93.3%)**.
- 90-case correct tool name: **84/90 (93.3%)**.
- Correct tool conditional on a valid envelope: **84/84 (100%)**.
- Slot exact: **0/84 (0%)**.
- Right envelope + right tool + wrong value: **84/90 (93.3%)**.
- Clean stop: **90/90 (100%)**.
- Kind no-regression gate: **PASS**.
- Subtype no-regression gate: **PASS**.
- Artifact: `ember-v043-envelope-34286232002`, artifact ID `10079714722`.

The three-fold selector froze these choices:

| Subtype | Selected |
| --- | --- |
| `short_code/len4` | baseline |
| `short_code/len5` | baseline |
| `long_code/4x4` | baseline |
| `long_code/3x5` | baseline |
| `path/plain_leaf` | `web_lookup` |

Held-out kind results remained:

| Kind | Result |
| --- | ---: |
| `short_code` | 8/10 |
| `long_code` | 7/10 |
| `digits` | 10/10 |
| `model_id` | 10/10 |
| `url` | 10/10 |
| `path` | 9/10 |
| `entity` | 10/10 |
| `expression` | 10/10 |
| `mixed` | 10/10 |

`web_lookup` was a robust synthetic winner for `path/plain_leaf`, but on the held-out path cohort it changed which plain-leaf case failed rather than increasing the number of canonical envelopes. Path therefore remained 9/10 and the global score remained 84/90.

## Conclusion

v0.0.43 confirms that additional user-request wording changes are no longer moving the global baseline. The remaining six failures are stable: two short-code failures, three long-code failures, and one path failure. The next CPU-only calibration should keep the v0.0.42/v0.0.43 user prompts frozen and change a different variable: **system-line envelope activation for code-shaped lookups**, where five of the six residual failures remain.

A suitable v0.0.44 should test natural, tool-specific system-line variants on disjoint synthetic short/long-code folds without explicitly instructing the model to copy the value into the slot. The same 86/90 global gate and all existing no-regression floors should remain unchanged.

No optimizer, GPU job, training, promotion, deployment, or production integration was run.
