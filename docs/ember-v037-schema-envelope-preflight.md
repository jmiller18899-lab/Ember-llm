# Ember v0.0.37 schema-bearing envelope preflight

v0.0.36 established two facts without training: the pinned v0.0.31 step-479 checkpoint reproduced all four exact v0.0.8 tool envelopes with the correct tool name, but the 90-case single-target expansion reached only 64/90 canonical envelopes. All 64 valid envelopes used the correct tool and all 64 placed the wrong value.

v0.0.37 changes one prompt variable only. The 90 held-out values stay identical, single-target, balanced at ten cases per value kind, and mapped to the same demonstrated tools. The user requests stay natural. The system line changes from the short v0.0.8 wording to the schema-bearing agent wording used by the v0.0.32 semantic curriculum:

`You are Ember. Use <tool> with JSON arguments. Its only required argument is <field>, a string.`

This wording is checked directly against `jobs/ember_sft_data_semantic_v1.py` so the experiment cannot silently drift away from the v0.0.32 training envelope.

The exact four v0.0.8 tool prompts remain a control and must still pass 4/4 for canonical envelope and tool name. The full 90-case gate remains unchanged at at least 86/90 JSON-valid envelopes and 86/90 correct tool names. `slot_exact` remains diagnostic only.

The weak cohort is recorded separately: `short_code`, `long_code`, `url`, and `path`, the four kinds that were below 60% envelope validity in v0.0.36. This cohort contains 40 cases, but it does not replace or weaken the global 90-case gate.

No optimizer is present. `training_authorized=false`, `gpu_training_authorized=false`, and `production_authorized=false`.

## Measured result

GitHub Actions run `34281460629` evaluated commit `b9c18be6e028900aefd9f021df19f814ff3743b0` and correctly returned **FAIL** at the CPU baseline gate.

- Focused tests: **155 passed**.
- Exact v0.0.8 reference envelope: **4/4 (100%)**.
- Exact v0.0.8 reference tool name: **4/4 (100%)**.
- 90-case JSON-valid envelope: **66/90 (73.3%)**.
- 90-case correct tool name: **66/90 (73.3%)**.
- Correct tool conditional on a valid envelope: **66/66 (100%)**.
- Slot exact: **0/66 (0%)**.
- Right envelope + right tool + wrong value: **66/90 (73.3%)**.
- Clean stop: **90/90 (100%)**.
- Weak-kind JSON/tool envelope: **19/40 (47.5%)**.

Compared with v0.0.36, the schema-bearing system line improved the global envelope count by only two cases, from 64/90 to 66/90. It did not solve the weak cohort.

Per-kind JSON-valid/correct-tool counts were:

| Kind | v0.0.36 | v0.0.37 | Change |
| --- | ---: | ---: | ---: |
| short_code | 3/10 | 2/10 | -1 |
| long_code | 4/10 | 2/10 | -2 |
| url | 5/10 | 6/10 | +1 |
| path | 6/10 | 9/10 | +3 |
| digits | 9/10 | 10/10 | +1 |
| model_id | 9/10 | 10/10 | +1 |
| mixed | 8/10 | 7/10 | -1 |
| entity | 10/10 | 10/10 | 0 |
| expression | 10/10 | 10/10 | 0 |

The schema hint therefore helps paths and slightly helps URLs, digits, and model IDs, but makes the code-shaped values worse. Tool routing itself is not the problem: every canonical envelope named the expected tool. Placement is also consistently wrong on every evaluable case, preserving the intended `right envelope + right tool + wrong value` diagnostic.

## Conclusion

v0.0.37 does **not** clear the required 95% baseline. The result is 73.3%, well below the required 86/90. Per the preflight contract, the experiment stops here and no learning phase may consume this report as passing baseline evidence.

No optimizer, GPU job, training, promotion, deployment, or production integration was run.
