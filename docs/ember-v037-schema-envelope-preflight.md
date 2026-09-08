# Ember v0.0.37 schema-bearing envelope preflight

v0.0.36 established two facts without training: the pinned v0.0.31 step-479 checkpoint reproduced all four exact v0.0.8 tool envelopes with the correct tool name, but the 90-case single-target expansion reached only 64/90 canonical envelopes. All 64 valid envelopes used the correct tool and all 64 placed the wrong value.

v0.0.37 changes one prompt variable only. The 90 held-out values stay identical, single-target, balanced at ten cases per value kind, and mapped to the same demonstrated tools. The user requests stay natural. The system line changes from the short v0.0.8 wording to the schema-bearing agent wording used by the v0.0.32 semantic curriculum:

`You are Ember. Use <tool> with JSON arguments. Its only required argument is <field>, a string.`

This wording is checked directly against `jobs/ember_sft_data_semantic_v1.py` so the experiment cannot silently drift away from the v0.0.32 training envelope.

The exact four v0.0.8 tool prompts remain a control and must still pass 4/4 for canonical envelope and tool name. The full 90-case gate remains unchanged at at least 86/90 JSON-valid envelopes and 86/90 correct tool names. `slot_exact` remains diagnostic only.

The weak cohort is recorded separately: `short_code`, `long_code`, `url`, and `path`, the four kinds that were below 60% envelope validity in v0.0.36. This cohort contains 40 cases, but it does not replace or weaken the global 90-case gate.

No optimizer is present. `training_authorized=false`, `gpu_training_authorized=false`, and `production_authorized=false`. If the 95% baseline gate fails, the experiment stops and no GPU should be spent.
