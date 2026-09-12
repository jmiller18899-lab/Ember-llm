# Ember routing v5 and argument parser v4

This revision addresses the five misses in the strongest v4 development result:
two route confusions and three unnecessary clarifications on requests with a
context clause. The v4 code, fitted-candidate manifest, and 95/100 result remain
preserved. The new components are opt-in and do not change the default runtime.
V5 repairs all five on the consumed development suite. Its frozen new-request
confirmation scores **99/100 on both checkpoints**, leaving one routing error and
**failing the strict 100/100 gate**.

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

## Measured development result

The [full-bundle CPU run](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34693520160)
on source commit `c05a082ce0df944e16008c23c75b525ea5cc5e7e` repaired all five
misses from the strongest v4 configuration. These are the same 100 previously
consumed requests, so the result measures development regressions.

| Checkpoint | Path | Correct route / 100 | Combined / 100 |
| --- | --- | ---: | ---: |
| Full | Routing v4 + parser v3 | 96 | 93 |
| Full | Routing v4 + parser v4 | 96 | 96 |
| Full | Routing v5 + parser v4 | 100 | 100 |
| INT4, dequantized | Routing v4 + parser v3 | 98 | 95 |
| INT4, dequantized | Routing v4 + parser v4 | 98 | 98 |
| INT4, dequantized | Routing v5 + parser v4 | 100 | 100 |

The revised parser retained **60/60** consumed parser regressions with supplied
routes. Grouped development selection chose ridge 0.01 at **435/480**. Both
checkpoint integrations use the same text helper, so equal v5 scores are not two
independent confirmations of LLM capability. INT4 execution dequantizes to float32
on CPU; this run does not measure native quantized inference speed.

The [complete paired report](../reports/ember-routing-v5-development.json),
[selection report](../reports/ember-routing-v5-selection.json), and
[candidate manifest](../reports/ember-routing-v5-manifest.json) preserve this
measurement. The development artifact is `10298287301`, SHA-256
`32f0a943275c501dbe592c7b8a247dce6cff3fdf7789a04a5db4f50443441800`.

## Freeze and new confirmation

The [source lock](../tool_assistant/data/routing-v5-source-lock.json) records
30 source files and all 38 fitted-candidate files. It was committed as
`7cf5355ed5ae9a4b42c8566744b75b2b0c075072` before the
[100 new requests](../tool_assistant/data/routing-v5-confirmation.json) were
authored and committed in its child, `af509ac9b39b47662836f1ac25c9036d2d778202`.
The lock's SHA-256 is
`2a6bd995f55c5c1c34afb7477df5ab3d54826e5d7e8132105eb7519778413579`;
the new suite's SHA-256 is
`d49cc38f0d6b7d04c8b9ea4019c826ecf64c9cd1aefe781fbad7be7833bcb6c2`.

The suite contains 20 requests per route. Its normalized request strings do not
duplicate the recorded training, consumed suites, or literal development tests.
An assistant authored the suite after the freeze and checked the arithmetic gold
values independently before inference. This is finite contract coverage, not
blind human evaluation or a guarantee of semantic independence from training
templates. Once evaluated, these requests become consumed regression evidence.

The [confirmation CPU run](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34694061949)
completed on `af509ac9b39b47662836f1ac25c9036d2d778202`. It verified the source
and all fitted-file hashes against the prior freeze before and after scoring.
The repeated development measurements reproduced every prior case outcome.

These paired results use the same **new** 100 requests; they are separate from
the development table above:

| Checkpoint | Path | Correct route / 100 | Combined / 100 |
| --- | --- | ---: | ---: |
| Full | Routing v4 + parser v3 | 89 | 81 |
| Full | Routing v4 + parser v4 | 89 | 89 |
| Full | Routing v5 + parser v4 | 99 | 99 |
| INT4, dequantized | Routing v4 + parser v3 | 92 | 80 |
| INT4, dequantized | Routing v4 + parser v4 | 92 | 92 |
| INT4, dequantized | Routing v5 + parser v4 | 99 | 99 |

V5 passes 19/20 weather requests and 20/20 each for time, calculator, web search,
and direct routing. It has 79/80 correct tool arguments with successful fixture
dispatch and retains 60/60 parser regressions. Within this suite, no request that
passed either v4 control becomes a combined failure in v5.

The remaining request is `v5-new-weather-16`:

> I am contacting a friend in Tromsø; is it cold there currently?

Its expected route is `weather` with `{"location": "Tromsø"}`. V5 selects
`get_time`, then returns `needs_clarification` with `ambiguous_location`; no tool
is dispatched. Both older v4 controls also fail this request, selecting `direct`.
This is a remaining route error on a weather question following a contact
context. The 100/100 requirement is unmet, so the report records `status: FAIL`
and `strict_pass: false`, and the workflow ends unsuccessfully after preserving
the evidence. The frozen helper and test labels were not changed after scoring.

The [complete confirmation report](../reports/ember-routing-v5-confirmation.json)
has SHA-256
`bbe95471634480dadd216b6fd6fc3fcba36d071f6e27469845f4c2d577b9a78b`.
Its [matching run manifest](../reports/ember-routing-v5-confirmation-manifest.json)
has SHA-256
`5351d5fda07722d9bfea40e3c157b404b32fa06f130887bfaba45246e7c3ea23`.
The new manifest has its own build timestamp; all 38 candidate file hashes match
the pre-confirmation lock exactly. Both JSON files were recovered losslessly
from the job log and checked against their emitted checksums. The complete
Actions artifact is `10297993516`, SHA-256
`4241d707d2ca126964d6752f3e4b187b3ad2552b6d9abc4a688144c6de6d4a42`.

## Validation and scope

The experiment passed **231 focused tests**. The
[hosted repository validation](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34694063918)
passed **723 repository tests**, **22 packaged Ember tests**, and the historical
parser-v3 **60/60** check. These verify implementation and compatibility; they do
not replace the frozen confirmation gate.

The helper measures tool routing, validated arguments, and one deterministic
fixture dispatch. Calculator arguments are scored by numerical equivalence;
other tool arguments require exact field values. Direct requests score routing
only. Generated-answer quality and live service adapters remain untested.
The original model weights and default runtime are unchanged. This is opt-in
research code with `production_ready: false`; no prototype is released by this
work.

The next bounded experiment can address weather versus time routing after a
contact context. Any revised helper must preserve this failed report, treat
these 100 requests as consumed regressions, and freeze a new candidate before
authoring another confirmation.
