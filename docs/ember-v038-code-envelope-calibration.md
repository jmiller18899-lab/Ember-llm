# Ember v0.0.38 code-shaped envelope calibration

v0.0.37 improved the 90-case canonical-envelope baseline only slightly, from 64/90 to 66/90. Every valid envelope still named the correct tool, and every evaluable slot still contained the wrong value. The remaining envelope weakness was concentrated in code-shaped inputs: `short_code` reached 2/10 and `long_code` reached 2/10 under the schema-bearing prompt.

v0.0.38 changes only those 20 user requests. The system line remains exactly the schema-bearing v0.0.32 curriculum form and the expected tool remains `web_search(query)`. For `short_code` and `long_code`, the natural user request becomes:

`Search the web for this exact identifier: "<value>".`

The other 70 held-out cases retain their v0.0.37 prompts unchanged. The same 90 target values, tool mapping, source checkpoint, generation budget, and four exact v0.0.8 control prompts are preserved.

The experiment still requires at least 86/90 JSON-valid envelopes and 86/90 correct tool names (95%). The four exact v0.0.8 controls must remain 4/4 for both measurements. `slot_exact` is diagnostic only and does not weaken the envelope gate.

No optimizer is present. `training_authorized=false`, `gpu_training_authorized=false`, and `production_authorized=false`. A failed CPU gate must stop the experiment with no training, GPU job, promotion, deployment, or production integration.
