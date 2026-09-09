# Ember v0.0.36 single-target agent-envelope preflight

v0.0.34 and v0.0.35 are preserved as failed prompt-baseline experiments. Neither authorized training or GPU work.

The v0.0.35 result made routing much cleaner once JSON existed (44/45 JSON-valid envelopes named the correct tool), but envelope JSON validity itself was only 45/90. Inspection of the original v0.0.32 CPU report showed why: its clearest baseline examples of `right envelope + right tool + wrong argument value` were single-target literal requests. The preflight revisions had added `old` and `fallback` distractors that were not required by the proposed baseline and changed the behavior under measurement.

v0.0.36 removes that extra variable. It performs two CPU-only measurements against the pinned v0.0.31 step-479 checkpoint.

First, it runs the four exact tool-call prompts from `config/ember_v0.0.8_eval.json` without rewriting them: weather/Detroit, calculator/347×28, web search/Python release, and get time/Tokyo. All four must produce canonical JSON tool envelopes and all four must name the expected tool.

Second, it runs the same 90 historical held-out values used by the copy diagnostics, balanced at ten cases for each of nine value kinds. Each prompt contains only one requested value:

- entity → `weather(location)` using `What is the weather in <value> right now?`
- digits and expression → `calculator(expression)` using `Calculate <value>.`
- short_code, long_code, model_id, url, path, and mixed → `web_search(query)` using `Find current information about <value>.`

The system lines for those three tools are copied exactly from the v0.0.8 evaluation config. There is no `TARGET=...`, no literal-copy system role, and no old/fallback distractor.

The 90-case gate requires at least 86/90 canonical JSON envelopes and at least 86/90 correct tool names. `slot_exact` is diagnostic only.

`training_authorized=false`, `gpu_training_authorized=false`, and `production_authorized=false`.

## Measured result

GitHub Actions run `34280812628` evaluated commit `74c3772f20ea79b1ee1120846fc2e6c21cf17c81` and correctly returned **FAIL** at the CPU baseline gate.

### Exact v0.0.8 reference: PASS

All four exact historical tool prompts reproduced a canonical JSON tool envelope and the expected tool name:

- JSON-valid envelope: **4/4 (100%)**
- Correct tool name: **4/4 (100%)**
- Clean stop: **4/4 (100%)**

The argument values were nevertheless wrong in all four generated calls: the Detroit weather request used `Austin`; the 347×28 calculator request used `337`; the Python-release search used `recent WebAssembly announcements`; and the Tokyo time request used `America/Anchorage`. This directly reproduces the qualitative v0.0.32 finding on the exact v0.0.8 envelope: right envelope, right tool, wrong value.

### 90-case single-target baseline: FAIL

- JSON-valid envelope: **64/90 (71.1%)**
- Correct tool name: **64/90 (71.1%)**
- Slot evaluable: **64/90 (71.1%)**
- Slot exact: **0/64 (0%)**
- Right envelope + correct tool + wrong value: **64/90 (71.1%)**
- Clean stop: **89/90 (98.9%)**

Every JSON-valid envelope in the 90-case expansion named the expected tool: **64/64 (100%) conditional tool routing**. Every one of those 64 evaluable envelopes placed the wrong value in the expected argument slot. The remaining issue is therefore not tool-name routing; the expansion fails because 26 cases do not reliably enter the canonical JSON tool envelope at all.

Per-kind envelope/tool counts were:

| Value kind | JSON valid | Correct tool | Slot exact |
| --- | ---: | ---: | ---: |
| entity | 10/10 | 10/10 | 0/10 |
| expression | 10/10 | 10/10 | 0/10 |
| digits | 9/10 | 9/10 | 0/9 |
| model_id | 9/10 | 9/10 | 0/9 |
| mixed | 8/10 | 8/10 | 0/8 |
| path | 6/10 | 6/10 | 0/6 |
| url | 5/10 | 5/10 | 0/5 |
| long_code | 4/10 | 4/10 | 0/4 |
| short_code | 3/10 | 3/10 | 0/3 |

## Conclusion

The premise is now separated cleanly. The pinned checkpoint **does** retain the exact historical agent-envelope behavior: the four v0.0.8 tool prompts pass 4/4 for envelope and routing while missing the requested value. But that behavior is not yet stable across the proposed 90-case expansion: 71.1% is below the required 95% baseline threshold.

Therefore `slot_exact` is still not a valid global 90-case target metric for a learning phase, even though its value is 0/64 on the evaluable subset. Per the preflight contract, this experiment stops here. No optimizer, GPU job, promotion, deployment, or production integration was run, and a future phase must not consume this baseline as passing evidence.
