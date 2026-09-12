# Ember routing v5 and argument parser v4

This revision addresses the five misses in the strongest v4 development result:
two route confusions and three unnecessary clarifications on requests with a
context clause. The v4 code, fitted-candidate manifest, and 95/100 result remain
preserved. The new components are opt-in and do not change the default runtime.

## What changes

Routing v5 continues the text-only control that produced the strongest v4 result.
It fits one shared helper for both checkpoints; Ember representations do not
contribute to its route score. The 864 training requests comprise 384 historical
requests, 320 v4 development requests, and 160 new contrasts. The new examples
cover wet weather, transport interruptions and schedules, direct explanations
that mention those subjects, and current-time questions with a named antecedent.
All consumed confirmation requests are excluded from fitting.

Ridge selection holds out each of 24 complete template groups once over four
folds. Vocabulary and IDF are fitted inside each training fold. This is grouped
development selection, not independent confirmation. The model-free helper can
be checked locally. The integrated runtime additionally verifies the frozen
Ember bundle and retains its tokenizer/context limit. It performs no model
forward pass for routing and does not train Ember's parameters.

Parser v4 retains v3 arithmetic and ordinary single-clause parsing. It recognizes
one complete context clause followed by one weather or time question, separated
by a semicolon or an identifiable sentence boundary. A small set of complete
first-person context frames can describe stepping outside or packing clothing.
A call/contact/visit context can supply one explicit place for a following
question using “there.” Names, quoted names, and time zones retain v3 validation.

It does not choose between two places, discard arbitrary instructions, resolve
“here” or “my office,” or bypass the current-time limitation. Missing, unknown,
or ambiguous context still needs clarification. Place parsing is not geocoding.

## Evaluation contract

The CPU workflow compares v4, v4 routing with the revised parser, and v5 routing
with the revised parser on both checkpoints. The original 100 requests are
development regressions. The original 60 parser requests are also checked with
supplied routes. Exact arguments and one fixture dispatch remain required for
tool requests; the 20 direct requests score routing only.

Before writing any new confirmation, source and every fitted-candidate file must
be hashed in `tool_assistant/data/routing-v5-source-lock.json`. The new suite is
bound to that lock, must contain 20 requests per route, and must be disjoint by
normalized request text from training, consumed suites, and recorded literal
development tests. The evaluator refuses source or head changes and report
overwrites. A confirmation failure exits unsuccessfully and fails the workflow.
Repeating a consumed suite is regression evidence, not another fresh confirmation.

The workflow emits exact JSON evidence in bounded log chunks as well as keeping
an Actions artifact. This permits checksum-verified readback without repeating a
model experiment when an artifact download transport is unavailable.

## Current status

The local text helper passed **100/100 consumed development requests**, including
all five prior failures, and the revised parser retained **60/60** consumed parser
checks. Grouped development selection chose ridge 0.01 at **435/480**. These are
development results. Full-bundle CPU measurement and fresh confirmation are
pending; no prototype or production deployment is qualified by these results.
