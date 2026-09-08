# Ember v0.0.41 structural-subtype envelope calibration

v0.0.40 established the strongest regression-safe envelope baseline so far: **80/90 (88.9%)** canonical JSON envelopes, with the correct tool name in all 80 valid envelopes and `slot_exact=0/80`. The historical kind-level no-regression gate passed.

The ten remaining envelope failures are not spread uniformly across the five weak web-search kinds. Inspection of the preserved v0.0.40 report localizes them to eight structural subtypes:

| Residual subtype | v0.0.40 held-out result | Failures |
| --- | ---: | ---: |
| `short_code/len4` | 5/6 | 1 |
| `short_code/len5` | 3/4 | 1 |
| `long_code/4x4` | 5/6 | 1 |
| `long_code/3x5` | 2/4 | 2 |
| `url/one_mixed` | 2/4 | 2 |
| `url/two_segment` | 2/3 | 1 |
| `path/plain_leaf` | 3/4 | 1 |
| `mixed/upper` | 2/3 | 1 |

All other structural subtypes are frozen on the v0.0.40 prompt stack and are not eligible for replacement.

## Calibration design

Each residual subtype carries the exact v0.0.40 safe prompt for its kind as a mandatory `baseline` candidate. Three challengers are evaluated:

1. `literal_query`: `Use web_search. Use the literal query string "<value>".`
2. `json_exact`: `Call web_search with JSON arguments and set query exactly to "<value>".`
3. `quoted_text`: `Use web_search to search for the exact quoted text "<value>".`

For each of the eight residual subtypes, eight deterministic synthetic values are generated from the same structural renderer while explicitly excluding all 90 held-out targets and corruptions. That produces **256 CPU-only calibration generations**: 8 subtypes × 8 synthetic values × 4 candidates.

The v0.0.40 baseline remains selected unless one challenger beats it by at least **2 of 8** cases on both canonical-envelope count and correct-tool count. The final held-out battery is never used for prompt selection.

## Protected gates

After subtype selection is frozen, the same 90-case held-out battery is measured once. Four gates must all pass:

1. Exact v0.0.8 reference control: 4/4 canonical envelope and correct tool name.
2. Global baseline: at least **86/90 (95%)** canonical envelopes and correct tool names.
3. v0.0.40 kind floors: `short_code>=8`, `long_code>=7`, `url>=7`, `path>=9`, `mixed>=9`.
4. v0.0.40 residual-subtype floors: `len4>=5`, `len5>=3`, `4x4>=5`, `3x5>=2`, `one_mixed>=2`, `two_segment>=2`, `plain_leaf>=3`, `mixed/upper>=2`.

`slot_exact` remains diagnostic only. No optimizer is present. `training_authorized=false`, `gpu_training_authorized=false`, and `production_authorized=false`. Even a passing report would establish only a valid starting envelope for a separate placement-learning phase; it would not authorize training automatically.
