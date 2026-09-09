# Ember v0.0.41 structural-subtype envelope calibration

v0.0.40 established the strongest regression-safe envelope baseline so far: **80/90 (88.9%)** canonical JSON envelopes, with the correct tool name in all 80 valid envelopes and `slot_exact=0/80`. The historical kind-level no-regression gate passed.

The ten remaining envelope failures were localized to eight structural subtypes:

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

All other structural subtypes were frozen on the v0.0.40 prompt stack and were not eligible for replacement.

## Calibration design

Each residual subtype carried the exact v0.0.40 safe prompt for its kind as a mandatory `baseline` candidate. Three challengers were evaluated:

1. `literal_query`: `Use web_search. Use the literal query string "<value>".`
2. `json_exact`: `Call web_search with JSON arguments and set query exactly to "<value>".`
3. `quoted_text`: `Use web_search to search for the exact quoted text "<value>".`

For each of the eight residual subtypes, eight deterministic synthetic values were generated from the same structural renderer while explicitly excluding all 90 held-out targets and corruptions. That produced **256 CPU-only calibration generations**: 8 subtypes × 8 synthetic values × 4 candidates.

The v0.0.40 baseline remained selected unless one challenger beat it by at least **2 of 8** cases on both canonical-envelope count and correct-tool count. The final held-out battery was never used for prompt selection.

## Protected gates

After subtype selection was frozen, the same 90-case held-out battery was measured once. Four gates had to pass:

1. Exact v0.0.8 reference control: 4/4 canonical envelope and correct tool name.
2. Global baseline: at least **86/90 (95%)** canonical envelopes and correct tool names.
3. v0.0.40 kind floors: `short_code>=8`, `long_code>=7`, `url>=7`, `path>=9`, `mixed>=9`.
4. v0.0.40 residual-subtype floors: `len4>=5`, `len5>=3`, `4x4>=5`, `3x5>=2`, `one_mixed>=2`, `two_segment>=2`, `plain_leaf>=3`, `mixed/upper>=2`.

`slot_exact` remained diagnostic only. No optimizer was present. `training_authorized=false`, `gpu_training_authorized=false`, and `production_authorized=false`.

## Setup-only first attempt

The first workflow attempt, run `34284060437`, stopped in the guard tests before any checkpoint inference. The synthetic URL renderer can occasionally emit a one-segment URL whose first character is a digit; the subtype sampler treated that as an error while requesting `url/one_mixed`. The sampler was corrected to skip rendered values that do not match the requested subtype. No held-out cases or model measurements from that setup-only attempt were consumed.

## Measured result

The corrected GitHub Actions run `34284235552` evaluated commit `0783e6d883647b7bcefdde42dfb9734f36729b53` and correctly returned **FAIL** because the global 95% gate and the short-code no-regression gates were not met.

- Focused/inherited tests: **186 passed**.
- Exact v0.0.8 reference envelope: **4/4 (100%)**.
- Exact v0.0.8 reference tool name: **4/4 (100%)**.
- 90-case JSON-valid envelope: **82/90 (91.1%)**.
- 90-case correct tool name: **82/90 (91.1%)**.
- Correct tool conditional on a valid envelope: **82/82 (100%)**.
- Slot exact: **0/82 (0%)**.
- Right envelope + right tool + wrong value: **82/90 (91.1%)**.
- Clean stop: **89/90 (98.9%)**.

Frozen subtype selections were:

| Residual subtype | Selected candidate |
| --- | --- |
| `short_code/len4` | `baseline` |
| `short_code/len5` | `json_exact` |
| `long_code/4x4` | `baseline` |
| `long_code/3x5` | `quoted_text` |
| `url/one_mixed` | `quoted_text` |
| `url/two_segment` | `baseline` |
| `path/plain_leaf` | `baseline` |
| `mixed/upper` | `literal_query` |

Held-out kind results were:

| Kind | v0.0.40 | v0.0.41 | Historical floor | Gate |
| --- | ---: | ---: | ---: | --- |
| `short_code` | 8/10 | **7/10** | 8/10 | **FAIL** |
| `long_code` | 7/10 | 7/10 | 7/10 | PASS |
| `url` | 7/10 | **9/10** | 7/10 | PASS |
| `path` | 9/10 | 9/10 | 9/10 | PASS |
| `mixed` | 9/10 | **10/10** | 9/10 | PASS |
| `digits` | 10/10 | 10/10 | — | unchanged |
| `model_id` | 10/10 | 10/10 | — | unchanged |
| `entity` | 10/10 | 10/10 | — | unchanged |
| `expression` | 10/10 | 10/10 | — | unchanged |

Protected residual-subtype results were:

| Residual subtype | v0.0.40 floor | v0.0.41 | Gate |
| --- | ---: | ---: | --- |
| `short_code/len4` | 5/6 | 5/6 | PASS |
| `short_code/len5` | 3/4 | **2/4** | **FAIL** |
| `long_code/4x4` | 5/6 | 5/6 | PASS |
| `long_code/3x5` | 2/4 | 2/4 | PASS |
| `url/one_mixed` | 2/4 | **4/4** | PASS |
| `url/two_segment` | 2/3 | 2/3 | PASS |
| `path/plain_leaf` | 3/4 | 3/4 | PASS |
| `mixed/upper` | 2/3 | **3/3** | PASS |

The structural matrix therefore found two strong, transferable improvements: `url/one_mixed` improved by **+2 cases** and `mixed/upper` improved by **+1 case**. However, `short_code/len5` selected `json_exact` on synthetic data and then regressed from 3/4 to 2/4 on held-out data. The subtype no-regression gate correctly caught that failure. Net global improvement was **+2 cases**, from 80/90 to 82/90.

## Safe composition implied by the evidence

The measured evidence supports a safer composition than the raw v0.0.41 selector output: keep the v0.0.41 winners only where the held-out no-regression gates passed, and revert `short_code/len5` to the v0.0.40 baseline prompt.

That composed stack would preserve:

- `url/one_mixed`: `quoted_text`, now 4/4.
- `mixed/upper`: `literal_query`, now 3/3.
- every other subtype on the v0.0.40 safe prompt, including `short_code/len5`.

Based on the observed held-out counts, that stack has an evidence-backed floor of **83/90**: v0.0.40's 80/90 plus the two recovered URL cases plus the recovered uppercase-mixed case, without accepting the short-code regression.

## Conclusion

v0.0.41 moved the raw measured baseline to **82/90 (91.1%)**, but it remains four cases short of the required 86/90 and failed the short-code kind/subtype no-regression gates. It does **not** authorize placement learning.

The next CPU-only phase should start from the safe composed **83/90** stack, freeze the now-solved `url/one_mixed` and `mixed/upper` subtypes, and target only the remaining residual structures: `short_code/len4`, `short_code/len5`, `long_code/4x4`, `long_code/3x5`, `url/two_segment`, and `path/plain_leaf`. To reduce another synthetic-to-held-out mismatch, candidate replacement should require agreement on two independent synthetic panels rather than one panel.

No optimizer, GPU job, training, promotion, deployment, or production integration was run. This remains failing baseline evidence and must not authorize a learning phase.
