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

Three gates must all pass:

1. Exact v0.0.8 reference control: 4/4 canonical envelopes and 4/4 correct tool names.
2. Global baseline: at least 86/90 canonical envelopes and 86/90 correct tool names (95%).
3. Historical no-regression gate: `short_code >= 8`, `long_code >= 7`, `url >= 7`, `path >= 9`, and `mixed >= 9` on the 10-case held-out cohorts.

`slot_exact` remains diagnostic only. No optimizer is present. `training_authorized=false`, `gpu_training_authorized=false`, and `production_authorized=false`.

## Measured result

GitHub Actions run `34283295675` evaluated commit `7f6a5fc92202991b28ad1b6e0bcbf8db4f571c60` and correctly returned **FAIL** because the global 95% gate was not met.

- Focused/inherited tests: **177 passed**.
- Exact v0.0.8 reference envelope: **4/4 (100%)**.
- Exact v0.0.8 reference tool name: **4/4 (100%)**.
- 90-case JSON-valid envelope: **80/90 (88.9%)**.
- 90-case correct tool name: **80/90 (88.9%)**.
- Correct tool conditional on a valid envelope: **80/80 (100%)**.
- Slot exact: **0/80 (0%)**.
- Right envelope + right tool + wrong value: **80/90 (88.9%)**.
- Clean stop: **90/90 (100%)**.
- Historical no-regression gate: **PASS** for all five weak kinds.

Synthetic calibration counts and frozen choices were:

| Kind | Baseline | `direct_query` | `current_quoted` | Selected |
| --- | ---: | ---: | ---: | --- |
| `short_code` | 3/8 | 4/8 | 4/8 | baseline |
| `long_code` | 5/8 | 4/8 | 5/8 | baseline |
| `url` | 3/8 | 6/8 | 7/8 | `current_quoted` |
| `path` | 6/8 | 5/8 | 7/8 | baseline |
| `mixed` | 8/8 | 7/8 | 5/8 | baseline |

Only `url` cleared the required +2 replacement margin. The conservative margin correctly protected `path`, where `current_quoted` improved synthetic performance by only one case and therefore could not replace the known 9/10 held-out baseline.

Held-out per-kind results were:

| Kind | v0.0.39 | v0.0.40 | Historical floor | Result |
| --- | ---: | ---: | ---: | --- |
| `short_code` | 8/10 | 8/10 | 8/10 | floor kept |
| `long_code` | 7/10 | 7/10 | 7/10 | floor kept |
| `digits` | 10/10 | 10/10 | — | unchanged |
| `model_id` | 10/10 | 10/10 | — | unchanged |
| `url` | 7/10 | 7/10 | 7/10 | floor kept |
| `path` | 6/10 | 9/10 | 9/10 | restored |
| `entity` | 10/10 | 10/10 | — | unchanged |
| `expression` | 10/10 | 10/10 | — | unchanged |
| `mixed` | 9/10 | 9/10 | 9/10 | floor kept |

This produces the strongest regression-safe prompt stack measured so far: **80/90**, up from 77/90 in v0.0.39 and 76/90 in v0.0.38. The gain over v0.0.39 comes entirely from restoring the proven natural path prompt while preserving every other best-known cohort score.

## Conclusion

v0.0.40 solved the prompt-selection regression problem, but it did **not** solve the remaining envelope-generalization problem. The best safe stack is still six cases short of the required 86/90 threshold.

The residual failures are now stable and localized:

- `short_code`: 2 failures
- `long_code`: 3 failures
- `url`: 3 failures
- `path`: 1 failure
- `mixed`: 1 failure

The next CPU-only experiment should target those ten residual cases by structural subtype, not by broad kind-level wording. In particular, it should compare prompt behavior across the actual format variants within each kind while keeping the v0.0.40 best-safe prompt as the mandatory fallback.

No optimizer, GPU job, training, promotion, deployment, or production integration was run. This remains failing baseline evidence and must not authorize placement learning.
