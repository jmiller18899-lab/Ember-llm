# Ember v0.0.39 targeted envelope matrix

v0.0.38 raised the held-out canonical-envelope baseline from 66/90 to 76/90 by changing only `short_code` and `long_code` user wording. Tool routing remained perfect whenever a canonical JSON envelope appeared, and every evaluable slot still contained the wrong value.

The remaining failures were concentrated in five `web_search(query)` value kinds: `short_code` 8/10, `long_code` 6/10, `url` 6/10, `path` 9/10, and `mixed` 7/10. The other four kinds (`digits`, `model_id`, `entity`, `expression`) were already 10/10 and were left unchanged.

## Calibration design

v0.0.39 did not select a prompt on the 90-case held-out battery. It created four deterministic synthetic values for each weak kind, explicitly excluding every held-out target and corrupt value. Three user-message variants were measured for each synthetic value while keeping the schema-bearing v0.0.32 system line fixed:

1. `current_exact`: `Search the web for this exact <label>: "<value>".`
2. `tool_named`: `Use web_search to find current information about this exact <label>: "<value>".`
3. `lookup_exact`: `Look up this exact <label> on the web: "<value>".`

This produced 60 CPU-only calibration generations: 5 weak kinds × 4 synthetic values × 3 prompt variants. Selection used only canonical-envelope and correct-tool counts, with a deterministic priority order for ties. `slot_exact` was not used to choose a prompt.

After one variant was frozen per weak kind, the runner measured the same 90 held-out values used by v0.0.36-v0.0.38 exactly once. The four already-perfect kinds kept their v0.0.37 schema-natural prompts unchanged.

## Measured result

GitHub Actions run `34282669903` evaluated commit `2655aafb08843f7bf6010c886f9b8f62a2da7d5c` and correctly returned **FAIL** at the unchanged 95% CPU baseline gate.

- Focused/inherited tests: **168 passed**.
- Exact v0.0.8 reference envelope: **4/4 (100%)**.
- Exact v0.0.8 reference tool name: **4/4 (100%)**.
- 90-case JSON-valid envelope: **77/90 (85.6%)**.
- 90-case correct tool name: **77/90 (85.6%)**.
- Correct tool conditional on a valid envelope: **77/77 (100%)**.
- Slot exact: **0/77 (0%)**.
- Right envelope + right tool + wrong value: **77/90 (85.6%)**.
- Clean stop: **90/90 (100%)**.
- Targeted five-kind cohort: **37/50 (74.0%)**.
- Untouched four-kind cohort: **40/40 (100%)**.

Synthetic calibration selected:

| Kind | `current_exact` | `tool_named` | `lookup_exact` | Selected |
| --- | ---: | ---: | ---: | --- |
| short_code | 4/4 | 4/4 | 2/4 | `current_exact` |
| long_code | 2/4 | 3/4 | 2/4 | `tool_named` |
| url | 2/4 | 2/4 | 2/4 | `current_exact` |
| path | 1/4 | 0/4 | 0/4 | `current_exact` |
| mixed | 4/4 | 1/4 | 2/4 | `current_exact` |

Held-out per-kind results were:

| Kind | v0.0.38 | v0.0.39 | Change |
| --- | ---: | ---: | ---: |
| short_code | 8/10 | 8/10 | 0 |
| long_code | 6/10 | 7/10 | +1 |
| digits | 10/10 | 10/10 | 0 |
| model_id | 10/10 | 10/10 | 0 |
| url | 6/10 | 7/10 | +1 |
| path | 9/10 | 6/10 | -3 |
| entity | 10/10 | 10/10 | 0 |
| expression | 10/10 | 10/10 | 0 |
| mixed | 7/10 | 9/10 | +2 |

The matrix therefore found real gains for `long_code`, `url`, and `mixed`, but the net global improvement was only **+1 case**, from 76/90 to 77/90, because `path` regressed by three cases.

## Important design finding

The v0.0.39 matrix did not include the **existing v0.0.38 natural prompt** as a candidate for `url`, `path`, or `mixed`. The name `current_exact` was only truly the prior prompt for the two code kinds. This matters most for `path`: the synthetic selector chose `current_exact` at only 1/4 because the other two new variants were 0/4, even though the already-measured v0.0.38 natural path prompt was 9/10 on held-out data. The matrix therefore replaced a strong known baseline with a weak winner among three weak new alternatives.

That makes the next CPU-only calibration clear: preserve the best previously measured prompt per kind as a mandatory baseline candidate/fallback, then test new variants against it. At minimum, a follow-up should keep the v0.0.38 natural path wording unless a synthetic calibration candidate beats it convincingly, while retaining the v0.0.39 gains for `long_code`, `url`, and `mixed` for further validation.

## Gate conclusion

The required baseline remains at least 86/90 JSON-valid envelopes and 86/90 correct tool names. v0.0.39 reached 77/90, so it remains **9 cases short** and does not authorize placement learning.

No optimizer, GPU job, training, promotion, deployment, or production integration was run. This report is failing baseline evidence and must not authorize a learning phase.
