# Ember v0.0.38 code-shaped envelope calibration

v0.0.37 improved the 90-case canonical-envelope baseline only slightly, from 64/90 to 66/90. Every valid envelope still named the correct tool, and every evaluable slot still contained the wrong value. The remaining envelope weakness was concentrated in code-shaped inputs: `short_code` reached 2/10 and `long_code` reached 2/10 under the schema-bearing prompt.

v0.0.38 changes only those 20 user requests. The system line remains exactly the schema-bearing v0.0.32 curriculum form and the expected tool remains `web_search(query)`. For `short_code` and `long_code`, the natural user request becomes:

`Search the web for this exact identifier: "<value>".`

The other 70 held-out cases retain their v0.0.37 prompts unchanged. The same 90 target values, tool mapping, source checkpoint, generation budget, and four exact v0.0.8 control prompts are preserved.

The experiment still requires at least 86/90 JSON-valid envelopes and 86/90 correct tool names (95%). The four exact v0.0.8 controls must remain 4/4 for both measurements. `slot_exact` is diagnostic only and does not weaken the envelope gate.

No optimizer is present. `training_authorized=false`, `gpu_training_authorized=false`, and `production_authorized=false`.

## Measured result

The first workflow attempt stopped in tests before model evaluation because `prompt_reference` accidentally pointed at the v0.0.37 Markdown result instead of the historical JSON eval config. That pointer was corrected to `config/ember_v0.0.8_eval.json`; no model result was produced or consumed from the failed setup attempt.

GitHub Actions run `34282045729` then evaluated commit `0c5f4e19b1f8d1b077f2267578343eb0dcd89940` and correctly returned **FAIL** at the CPU baseline gate.

- Focused/inherited tests: **160 passed**.
- Exact v0.0.8 reference envelope: **4/4 (100%)**.
- Exact v0.0.8 reference tool name: **4/4 (100%)**.
- 90-case JSON-valid envelope: **76/90 (84.4%)**.
- 90-case correct tool name: **76/90 (84.4%)**.
- Correct tool conditional on a valid envelope: **76/76 (100%)**.
- Slot exact: **0/76 (0%)**.
- Right envelope + right tool + wrong value: **76/90 (84.4%)**.
- Clean stop: **90/90 (100%)**.
- `short_code` + `long_code` envelope/tool: **14/20 (70.0%)**.
- Unchanged 70-case envelope/tool: **62/70 (88.6%)**.

Per-kind JSON-valid/correct-tool counts were:

| Kind | v0.0.37 | v0.0.38 | Change |
| --- | ---: | ---: | ---: |
| short_code | 2/10 | 8/10 | +6 |
| long_code | 2/10 | 6/10 | +4 |
| digits | 10/10 | 10/10 | 0 |
| model_id | 10/10 | 10/10 | 0 |
| url | 6/10 | 6/10 | 0 |
| path | 9/10 | 9/10 | 0 |
| entity | 10/10 | 10/10 | 0 |
| expression | 10/10 | 10/10 | 0 |
| mixed | 7/10 | 7/10 | 0 |

The code-specific wording therefore caused a clean +10-case gain in the exact cohort it changed, raising the global baseline from 66/90 to 76/90 while leaving the unchanged 70 cases at 62/70. Tool routing remains perfect whenever a canonical envelope appears, and value placement remains wrong on every evaluable case.

## Conclusion

v0.0.38 is a substantial calibration improvement but still does **not** clear the required 95% baseline. The result is 76/90 (84.4%), ten cases short of the required 86/90.

A useful constraint now follows from the measured counts: even if all six remaining code cases were fixed, the global score would only reach 82/90. A future CPU-only calibration must therefore also improve the residual non-code failures, currently concentrated in `url` (6/10), `mixed` (7/10), and `path` (9/10), while preserving the already-perfect `digits`, `model_id`, `entity`, and `expression` cohorts.

No optimizer, GPU job, training, promotion, deployment, or production integration was run. This report is failing baseline evidence and must not authorize a placement-learning phase.
