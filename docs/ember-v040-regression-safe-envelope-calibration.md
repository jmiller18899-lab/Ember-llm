# Ember v0.0.40 regression-safe envelope calibration

v0.0.39 reached 77/90 canonical envelopes, but its prompt selector exposed a design flaw: it did not include the strongest previously measured prompt as a candidate for every weak kind. That allowed `path` to regress from 9/10 to 6/10 even while `long_code`, `url`, and `mixed` improved.

v0.0.40 makes prompt selection regression-safe.

## Historical floors

Each weak web-search kind carries forward the best previously measured held-out prompt and score as its mandatory baseline:

| Kind | Best prior prompt | Held-out floor |
| --- | --- | ---: |
| `short_code` | exact-identifier search | 8/10 |
| `long_code` | tool-named exact-identifier search | 7/10 |
| `url` | exact-URL search | 7/10 |
| `path` | v0.0.37/v0.0.38 natural search request | 9/10 |
| `mixed` | exact-identifier search | 9/10 |

The four already-perfect kinds (`digits`, `model_id`, `entity`, `expression`) remain on their v0.0.37 schema-natural prompts and are not calibrated.

## Synthetic calibration

For each weak kind, eight deterministic synthetic values are generated and explicitly excluded from all held-out targets and corruptions. Three candidates are measured for every synthetic value:

1. `baseline`: the best previously measured prompt for that specific kind.
2. `direct_query`: `Use web_search with query "<value>".`
3. `current_quoted`: `Use web_search to find current information about "<value>".`

This is 120 CPU-only calibration generations: 5 kinds × 8 synthetic values × 3 candidates.

The baseline remains selected unless one challenger beats it by at least **2 of 8 cases** on both canonical-envelope count and correct-tool count. The final 90 held-out prompts are never used for candidate selection.

## Gates

The frozen prompt selection is then measured once on the same 90-case battery used by v0.0.36-v0.0.39.

Three gates must all pass:

1. Exact v0.0.8 reference control: 4/4 canonical envelopes and 4/4 correct tool names.
2. Global baseline: at least 86/90 canonical envelopes and 86/90 correct tool names (95%).
3. Historical no-regression gate: `short_code >= 8`, `long_code >= 7`, `url >= 7`, `path >= 9`, and `mixed >= 9` on the 10-case held-out cohorts.

`slot_exact` remains diagnostic only. No optimizer is present. `training_authorized=false`, `gpu_training_authorized=false`, and `production_authorized=false`. A passing v0.0.40 report would establish only that the envelope baseline is suitable for designing a separate placement-learning phase; it would not itself authorize training.
