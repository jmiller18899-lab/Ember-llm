# Ember v0.0.34 agent-envelope baseline preflight

v0.0.34 is a new experiment design, not a continuation of the failed v0.0.33 optimizer run. The v0.0.33 evidence remains unchanged: its CPU canary improved semantic loss but failed `development_exact_gain` and `copy_preserved`, so no GPU job was authorized.

This phase answers a narrower question before any more learning is attempted: does the pinned v0.0.31 step-479 checkpoint already produce a stable tool-call envelope when the prompt looks like the agent prompts it originally learned?

The prompt style is anchored to `config/ember_v0.0.8_eval.json`: a short task-shaped system instruction followed by a natural user request. The rigid literal-copy wording (`TARGET=...`, `Reply with TARGET exactly once`) is not used. Each request still contains an old value and a fallback value so placement remains a real selection problem rather than a single-value copy task.

The full 90-case held-out battery from v0.0.26 is used, balanced at ten cases for each of nine value kinds. Tools are paired with kinds semantically:

| Value kind | Tool | Argument slot |
| --- | --- | --- |
| entity | `weather` | `location` |
| expression, digits | `calculator` | `expression` |
| model_id, short_code | `web_search` | `query` |
| long_code, mixed | `lookup` | `key` |
| url | `fetch_url` | `url` |
| path | `read_file` | `path` |

The preflight reports three levels separately:

1. `envelope_json_valid_rate`: the completion begins with `<|tool|>` and contains one canonical JSON object with `name` and `arguments`.
2. `envelope_tool_name_rate`: the JSON envelope names the expected tool.
3. `slot_exact_rate`: among envelopes with the correct tool name, the expected argument slot contains the held-out target exactly. This is the placement measurement; it is not a baseline gate.

It also records `right_envelope_tool_wrong_value_rate`, which is the specific v0.0.32-style failure pattern this preflight is meant to reproduce at larger scale.

The baseline gate requires both JSON validity and correct tool-name routing to be at least 95% (86 of 90 cases). A future learning phase must not consume this baseline unless those checks pass. If the gate fails, the prompt/envelope baseline is still wrong and the experiment stops without training.

Authorization is intentionally closed: `training_authorized=false`, `gpu_training_authorized=false`, and `production_authorized=false`. The runner contains no optimizer and no model write path. A PASS means only that `slot_exact` is meaningful to interpret on this prompt family; it does not authorize a training phase, promotion, deployment, or production integration.

## Measured result

GitHub Actions run `34279609411` evaluated commit `d00654b9206d8013a24998099ac8c532ebae83d2` and correctly returned **FAIL**.

- Envelope JSON valid: **55/90 (61.1%)**
- Correct tool name: **38/90 (42.2%)**
- Slot evaluable: **38/90 (42.2%)**
- Slot exact: **0/38 (0%)**
- Right envelope + correct tool + wrong value: **38/90 (42.2%)**
- Clean stop: **89/90 (98.9%)**

The failure is diagnostic, not an infrastructure failure: all 134 focused tests passed and all 90 cases ran. The new/unproven tool families caused most of the envelope collapse: `url` and `path` produced 0/10 valid JSON, `long_code` produced 10/10 valid JSON but 0/10 correct `lookup` routing, and `mixed` produced only 4/10 valid JSON with 0/10 correct `lookup` routing. By contrast, `digits` and `entity` each produced 10/10 valid JSON and 10/10 correct tool names, while `expression` produced 7/10 for both. Every one of the 38 correct-tool envelopes still placed the wrong value in the argument, reproducing the placement failure only on the subset where the envelope was stable.

Per the preflight contract, this result stops the experiment before training. No optimizer, GPU job, promotion, or integration was run. The next prompt revision should stay inside the tool vocabulary already demonstrated by the v0.0.8 agent envelope instead of introducing `lookup`, `fetch_url`, or `read_file` as baseline requirements.
