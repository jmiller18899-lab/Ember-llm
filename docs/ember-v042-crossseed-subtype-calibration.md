# Ember v0.0.42 cross-seed subtype calibration

v0.0.41 reached the strongest raw envelope score so far at **82/90**, but correctly failed its protection gates because `short_code/len5` regressed from the v0.0.40 floor of 3/4 to 2/4. At the same time, two subtype changes were genuine non-regressing gains: `url/one_mixed` improved to 4/4 and `mixed/upper` improved to 3/3.

v0.0.42 assembles the safest known stack before trying anything new:

- preserve v0.0.41 `url/one_mixed = quoted_text`;
- preserve v0.0.41 `mixed/upper = literal_query`;
- restore `short_code/len5` to the v0.0.40 exact-identifier prompt;
- keep every other subtype on its strongest non-regressing prompt.

That assembled floor corresponds to 83/90 if the deterministic held-out outcomes reproduce.

## Remaining calibration targets

Only six failure-bearing subtypes remain eligible for replacement:

- `short_code/len4`
- `short_code/len5`
- `long_code/4x4`
- `long_code/3x5`
- `url/two_segment`
- `path/plain_leaf`

Every subtype keeps the assembled safe prompt as a mandatory baseline candidate.

## Two-fold synthetic selection

For each target subtype, v0.0.42 generates two independent synthetic folds (`select` and `confirm`), six exact-subtype values per fold, all disjoint from the 90 held-out targets and corruptions.

Three challengers are compared with the baseline:

1. `typed_exact`: `Search the web for this exact <type>: "<value>".`
2. `tool_query`: `Use web_search with query "<value>".`
3. `query_exact`: `Use web_search. Set query exactly to "<value>".`

A challenger may replace the baseline only if it beats the baseline by at least one case on **both folds** for both canonical envelope and correct tool name, and by at least two cases combined. This is designed to reject one-seed synthetic wins like the v0.0.41 short-code len5 result.

## Protection gates

The final 90-case held-out battery is measured once after selection. All of these must pass:

- exact v0.0.8 control: 4/4 envelope and 4/4 tool;
- global envelope/tool baseline: at least 86/90 (95%);
- kind floors: `short_code >= 8`, `long_code >= 7`, `url >= 9`, `path >= 9`, `mixed >= 10`;
- subtype floors: `short_code/len4 >= 5`, `short_code/len5 >= 3`, `long_code/4x4 >= 5`, `long_code/3x5 >= 2`, `url/one_mixed >= 4`, `url/two_segment >= 2`, `path/plain_leaf >= 3`, `mixed/upper >= 3`.

`slot_exact` remains diagnostic only. No optimizer is present. `training_authorized=false`, `gpu_training_authorized=false`, and `production_authorized=false`.

Even a PASS only establishes a valid baseline for a separate placement-learning phase. It does not authorize training automatically.
