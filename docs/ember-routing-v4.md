# Ember routing v4 CPU experiment

The frozen v2 assistant failed its consumed 100-request check: 82/100 combined
in full precision and 85/100 with the dequantized INT4 checkpoint. Parser v3
separately repaired arithmetic and location extraction. This experiment addresses
tool selection while retaining that parser and the original model weights.

## What changes

An opt-in `RoutingRuntime` predicts five routes directly: direct answer, weather,
calculator, web search, and current time. A ridge head combines text features with
the frozen Ember block-04 representation. A text-only control participates in the
same selection; ties favor less Ember contribution. A selected text-only arm
must be described as a text classifier, not evidence of improved LLM reasoning.

Training uses the original 384 requests plus 320 explicitly labeled development
requests. The added contrasts cover schedules, deadlines, service notices,
current conditions, local clocks, arithmetic, and ordinary writing or explanation
requests that mention tools. Sixteen template groups keep all city and number
variants together during four-fold selection. Vocabulary and normalization are
fitted within each training fold. Historical training stays in training only.
Grouped development accuracy is not an independent generalization estimate.

The original 100-request confirmation and the parser's 60-request confirmation
are excluded from fitting. The consumed 100 requests are then evaluated as
development regressions, with paired original-v2, parser-v3, and routing-v4 arms
on both checkpoints. Exact-argument and fixture-dispatch scoring are unchanged.

## Measurement and freeze

Run the new CPU workflow or reproduce its commands:

```bash
python -m tool_assistant.build --out baseline-candidate
python -m tool_assistant.build_v4 --base baseline-candidate --out routing-candidate
python -m tool_assistant.evaluate_v4 --bundle routing-candidate \
  --cases tool_assistant/data/confirmation-100.json --kind development \
  --report evidence/routing-v4-development.json
```

The baseline build requires the existing authorized Hugging Face token. Both
builders refuse to overwrite an existing candidate. Routing-head training does
not update Ember parameters. INT4 still means dequantized float32 CPU execution.

A fresh confirmation can be authored only after freezing source and fitted
candidate file hashes in `tool_assistant/data/routing-v4-source-lock.json`.
The evaluator verifies both, rejects previously recorded request strings, and
requires 20 requests per route. It refuses report overwrites and fails the job
unless both checkpoints score 100/100 combined. Once measured, that suite becomes
regression evidence; it must not be reused to claim another fresh confirmation.

## Measured result

The [CPU experiment](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34691874565)
completed on commit `70904a87bbef0cc520205208c82f98cdb8549156`. Routing v4
improved the consumed development check, but **failed the strict 100/100 bar**.

| Checkpoint | Path | Correct route / 100 | Combined / 100 |
| --- | --- | ---: | ---: |
| Full | Original v2 | 87 | 82 |
| Full | Parser v3 | 87 | 86 |
| Full | Routing v4 + parser v3 | 96 | 93 |
| INT4, dequantized | Original v2 | 90 | 85 |
| INT4, dequantized | Parser v3 | 90 | 88 |
| INT4, dequantized | Routing v4 + parser v3 | 98 | 95 |

Combined success requires the expected route, correct tool arguments, and one
successful fixture dispatch. The 20 direct requests measure routing only; no
generated answer is scored. These are the same previously consumed 100 requests,
so the improvement is development evidence, not a fresh confirmation result.

For full precision, grouped selection chose Ember feature weight 1.0 and ridge
1.0 at 287/320. For INT4, selection chose the **text-only control**, weight 0.0
and ridge 0.01 at 284/320. Its better 95/100 combined score comes from the
learned text router and parser; it does not establish improved Ember reasoning.
The frozen checkpoint is still loaded by the opt-in runtime, but its hidden
representation contributes zero to that selected route score.

The five failures in the strongest measured configuration are:

| Request | Remaining issue |
| --- | --- |
| “I need to know whether it is wet in Bergen at the moment.” | Weather is routed to current time. |
| “Search for current train service disruptions at Zurich Hauptbahnhof.” | Web search is routed to weather. |
| “I am stepping outside; what is the current temperature in Nuuk?” | Correct weather route; parser asks for clarification. |
| “I am packing a jacket; check the current temperature in Ulaanbaatar.” | Correct weather route; parser asks for clarification. |
| “I am about to call someone in Apia; what time is it there now?” | Correct time route; parser asks for clarification. |

The Bergen routing error is a regression from the original router. Better total
accuracy therefore does not justify replacing the default runtime. The next
bounded work is to improve weather/current-time and disruption/search contrasts,
and separately handle a single location expressed across clauses without
discarding ambiguous-location checks. Both checkpoints still need to clear a
new confirmation after a new candidate is frozen.

The [complete paired report](../reports/ember-routing-v4-development.json),
[selection report](../reports/ember-routing-v4-selection.json), and
[candidate manifest](../reports/ember-routing-v4-manifest.json) preserve the
measured evidence. The original fitted heads are in Actions artifact
`10297810403`, SHA-256
`56a5adc22bb5e261ea6c9c0d1a58874581e780f24103b8ff33df6acabdb91b74`.

## Validation and status

Local validation passed: **179 focused tests** and **671 repository tests**.
The focused checks cover grouped fitting, the text-only control, serialization,
parser and dispatch integration, candidate tampering, and consumed-suite rejection.
The [hosted validation run](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34691890057)
also passed all 671 repository tests and reproduced the parser-only 60/60 check.
An additional local reconstruction of the selected text-only control reproduced
the 98/100 routes, 95/100 combined score, and all five failed requests without
loading an Ember checkpoint. No new confirmation has been authored or consumed.

This remains research code with `production_ready: false`. Direct-answer
generation and live tool adapters are not implemented or qualified by this work.
The workflow preserves evidence and fitted heads; it does not publish a prototype.
The earlier candidate, failed report, source locks, and default runtime are unchanged.
