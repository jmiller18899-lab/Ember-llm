# Ember v0.0.44 system-line activation calibration

v0.0.43 held at **84/90 (93.3%)** with every no-regression gate passing. It also
produced the clearest negative result of the series: of three new user-request
framings tested on three disjoint synthetic folds, only one was accepted, and on
the held-out battery it changed *which* `path/plain_leaf` case failed rather than
adding one. Path stayed 9/10 and the global score did not move.

The conclusion is that **user-request wording is tapped out**. The global
requirement is still **86/90 (95%)**, so placement learning is still blocked, and
the remaining six failures are stable:

| Kind | Result | Residual failures |
| --- | ---: | --- |
| `short_code` | 8/10 | 2 |
| `long_code` | 7/10 | 3 |
| `path` | 9/10 | 1 |
| every other kind | 10/10 | 0 |

Five of the six are code-shaped. v0.0.44 therefore changes a different variable.

## What is frozen

The entire v0.0.43 user-prompt stack, byte for byte, including the selection that
run `34286232002` froze when it measured 84/90:

| Subtype | Frozen user prompt |
| --- | --- |
| `short_code/len4` | v0.0.42 baseline |
| `short_code/len5` | v0.0.42 baseline |
| `long_code/4x4` | v0.0.42 baseline |
| `long_code/3x5` | v0.0.42 baseline |
| `path/plain_leaf` | v0.0.43 `web_lookup` |
| `url/one_mixed` | v0.0.41 `quoted_text` |
| `url/two_segment` | v0.0.42 `tool_query` |
| `mixed/upper` | v0.0.41 `literal_query` |

This is asserted, not assumed: `test_frozen_user_stack_reproduces_the_measured_v043_prompts_exactly`
rebuilds the v0.0.43 battery and requires all 90 prompts to be identical when
every system line is the baseline. If a later edit disturbs the stack, that test
fails before any CPU time is spent.

## What changes

Exactly one thing: the system line, and only for the four code-shaped residual
subtypes. `path/plain_leaf` already consumed its change in v0.0.43 and is not a
code lookup, so it stays frozen and cannot move in either direction.

The baseline system line is the v0.0.37 schema line:

> You are Ember. Use web_search with JSON arguments. Its only required argument is query, a string.

The three challengers describe *when* web_search applies to a code or identifier
lookup, and keep the schema half of the sentence verbatim:

1. `code_activation`: `You are Ember. Use {tool} with JSON arguments when the user asks to look up a code or identifier. Its only required argument is {field}, a string.`
2. `identifier_capability`: `You are Ember. {tool} looks up codes and identifiers. Call it with JSON arguments. Its only required argument is {field}, a string.`
3. `reference_lookup`: `You are Ember. Requests to look up an exact reference code are handled by {tool} with JSON arguments. Its only required argument is {field}, a string.`

## Why none of them may teach the placement

`slot_exact` is 0/84 and is the intended target of a later learning phase. A
system line that told Ember to copy the value into `query` could raise
`slot_exact` by prompting rather than by learning, and would destroy the metric's
value as evidence. So the runner refuses to start if any challenger contains a
placement instruction, and `test_a_placement_instruction_is_rejected_rather_than_silently_accepted`
proves the check fires by feeding it one. Two further invariants hold:

- every challenger must end with the identical schema sentence, so the change is
  activation and not schema;
- within a calibration case the user request is byte-identical across all four
  candidates, so nothing but the system line varies.

## Selection

The v0.0.43 three-fold rule is carried over unchanged: six deterministic synthetic
values per fold per subtype, disjoint from the 90 held-out values *and* from
v0.0.43's own folds. A challenger may replace the baseline only if it loses on no
fold, strictly wins at least two of three, and gains at least two cases combined.

The rule is deliberately not tightened. v0.0.43's `web_lookup` was a robust
three-fold synthetic winner that produced zero held-out gain, which is evidence
about *transfer*, not about the threshold. v0.0.44 therefore measures that
directly: `transfer` in the report records each subtype's synthetic gain beside
its held-out result, so the question is recorded either way rather than argued.

## Gates

Unchanged from v0.0.43, as required:

- exact v0.0.8 reference control: 4/4 canonical envelopes and correct tool names;
- global envelope/tool baseline: at least 86/90 (95%);
- kind floors: `short_code >= 8`, `long_code >= 7`, `url >= 10`, `path >= 9`, `mixed >= 10`;
- subtype floors: `short_code/len4 >= 5`, `short_code/len5 >= 3`, `long_code/4x4 >= 5`,
  `long_code/3x5 >= 2`, `url/one_mixed >= 4`, `url/two_segment >= 3`,
  `path/plain_leaf >= 3`, `mixed/upper >= 3`.

`slot_exact` remains diagnostic only. `training_authorized=false`,
`gpu_training_authorized=false`, and `production_authorized=false`.

## How to read the outcome

A PASS at 86/90 would establish an envelope baseline strong enough to design a
separate value-placement learning phase. It would not train, promote, deploy, or
integrate anything.

A FAIL that holds at 84/90 with the gates green would be the second independent
null after v0.0.43. Read together with v0.0.43's transfer failure, that would be
reasonable evidence that prompt space — request *and* system line — is exhausted
for these five code-shaped cases, and that the remaining gap is a property of the
checkpoint rather than of the prompt. It would not on its own establish a capacity
limit; the checkpoint has never been trained on envelope-shaped completions.

A regression on any floor means the selected system line hurt a cohort that was
already passing, and the selector's zero-loss rule failed to predict it. That is
itself worth recording, because it would be the first measured case of a system
line moving a non-target cohort.

## Measured result

The formal compatibility-corrected run was GitHub Actions run `34288515663`, job
`102269563506`, on commit `003a3117cee090aab32c787c411a79f551f4d96e`.
The workflow returned **FAIL** only because the unchanged global 95% gate was not
met; the runner completed normally, wrote the report and summary, and uploaded
its evidence artifact.

- Focused/inherited tests: **207 passed**.
- Exact v0.0.8 reference: **4/4** canonical envelopes and **4/4** correct tool names.
- 90-case JSON-valid envelope: **84/90 (93.3%)**.
- 90-case correct tool name: **84/90 (93.3%)**.
- Correct tool conditional on a valid envelope: **84/84 (100%)**.
- `slot_exact`: **0/84 (0%)**.
- Right envelope + right tool + wrong value: **84/90 (93.3%)**.
- Clean stop: **90/90 (100%)**.
- Kind no-regression gate: **PASS**.
- Subtype no-regression gate: **PASS**.
- Artifact: `ember-v044-envelope-34288515663`, artifact ID `10080549791`, ZIP SHA256 `9717dbeb7a684cff6442f2c73d83ef29fbbd944da74d3de22ea255fd939fe9df`.

All four code-shaped subtype selectors retained the baseline system line:

| Subtype | Selected system | Synthetic gain | Held-out result |
| --- | --- | ---: | ---: |
| `short_code/len4` | baseline | +0 | 5/5 |
| `short_code/len5` | baseline | +0 | 3/5 |
| `long_code/4x4` | baseline | +0 | 5/5 |
| `long_code/3x5` | baseline | +0 | 2/5 |

The held-out kind totals therefore stayed exactly at the protected v0.0.43 level:
`short_code 8/10`, `long_code 7/10`, `url 10/10`, `path 9/10`, and
`mixed 10/10`; digits, model IDs, entities, and expressions remained 10/10.

## Conclusion

v0.0.44 is a second independent prompt-side null result. v0.0.43 found no global
gain from changing the user request; v0.0.44 found no selectable synthetic gain
from changing the system activation line. The remaining six envelope failures are
stable: two short-code cases, three long-code cases, and one path case.

Do **not** lower the 86/90 gate and do **not** launch placement training from this
result. The next useful CPU-only step is a residual activation/logit diagnostic on
the six failures versus matched passing cases while holding prompts completely
frozen. That can determine whether the failures are near the JSON/tool-call
decision boundary or are qualitatively in a different decoding regime before any
learning-phase design is authorized.

No optimizer, GPU job, training, promotion, deployment, or production integration
was run.
