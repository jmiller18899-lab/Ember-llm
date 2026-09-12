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

## Status

Local validation passed: **179 focused tests** and **671 repository tests**.
The focused checks cover grouped fitting, the text-only control, serialization,
parser and dispatch integration, candidate tampering, and consumed-suite rejection.
Actual checkpoint measurements are pending. These tests do not establish better
routing. No new confirmation has been authored or consumed yet.

This remains research code with `production_ready: false`. Direct-answer
generation and live tool adapters are not implemented or qualified by this work.
The workflow preserves evidence and fitted heads; it does not publish a prototype.
The earlier candidate, failed report, source locks, and default runtime are unchanged.
