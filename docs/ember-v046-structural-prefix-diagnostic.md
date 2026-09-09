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

## Measured result

The tokenizer-stable corrected run was GitHub Actions run `34303409879`, job `102314978909`, on commit `4dcfaeae98871c86a6b841c5fc2004dfd4bef8e1`.

- Focused/inherited guards: **109 passed**.
- Structural diagnostic status: **COMPLETE**.
- Structural cohort: **17 cases** = four non-rescued failures + 13 matched passes.
- Entry-suppression cohort: the two v0.0.45 long-code cases remained tracked separately.
- Artifact: `ember-v046-structural-34303409879`, artifact ID `10085770340`, ZIP SHA256 `a29e618ee6a2a7e2e2d0dc082dade047eebc4cb11cadae123812e9556805deab`.
- No optimizer, GPU, training, promotion, deployment, or production integration ran.

### Compact-prefix stability

The compact canonical serialization was **not stable enough on matched passing cases** to support failure-vs-pass localization.

| Measurement | Four failures | 13 matched passes |
| --- | ---: | ---: |
| complete compact prefix top-1 | 0/4 (0%) | 0/13 (0%) |
| structural token top-1 rate | 64.6% | 68.6% |
| median expected token rank | 1 | 1 |
| median expected margin | +1.7564 | +2.1328 |

All 13 matched passing cases first diverged from the compact serialization at the `name_key` stage. Their median `name_key` expected-token rank was **281.5**, with median margin **-15.3560**. The four failures showed essentially the same pattern at that stage: median rank **271**, median margin **-14.1255**.

That means the probe was measuring a formatting/tokenization preference that valid envelopes themselves do not follow, not a failure-specific structural defect.

### Residual failure observations

Two of the four non-rescued failures still showed the already-known entry competition at the very first token:

- `system_target_short_code_03`: `<|tool|>` rank 3, margin -1.6754.
- `system_target_long_code_08`: `<|tool|>` rank 35, margin -1.1382.

The other two (`system_target_short_code_02`, `system_target_path_05`) selected `<|tool|>` greedily and then diverged at the same compact `name_key` representation that all matched passes also reject. Therefore v0.0.46 cannot validly localize their deeper failure from this fixed compact serialization.

The separately tracked v0.0.45 entry-suppression cases remained clear:

- `system_target_long_code_02`: `<|tool|>` rank 226, margin -3.7581.
- `system_target_long_code_05`: `<|tool|>` rank 72, margin -3.9198.

## Conclusion

v0.0.46 is a valid **null diagnostic**: a single compact JSON token prefix is not a stable representation of Ember's successful tool envelopes, so it must not be used as a training target or as evidence for a structural learning objective.

Do **not** launch the planned v0.0.47 learning canary from this result. The next experiment should first become serialization-neutral: derive the structural token path from actual successful envelopes (or score equivalent JSON structural states rather than one exact textual serialization), keep the target argument value unforced, and then compare the four non-rescued failures against matched passes. Only after that representation is demonstrated stable on the passing cohort should a learning canary be designed.