# Ember request routing and arithmetic v7

V7 addresses the six failures in the consumed v6 confirmation: four explanation
questions routed to weather/time, an explicit web-search request routed direct,
and a rejected sum of two English word powers. The optional revision retains
the v6 contact-context fix and all archived fitted files.

## Changes and boundaries

The arithmetic bug comes from treating the plus or minus before a second word
power as a unary sign even when a first word power is already an operand.
`3 squared plus 4 squared` becomes two adjacent expressions instead of a sum.
Parser v7 recognizes a preceding `squared` or `cubed` as an operand when deciding
whether that sign is binary. It retains the previous grammar, exact-fraction
validation, limits, precedence, unsupported-operation refusals, and whole-input
checks. Location and search argument parsing delegate to the original parser.

Routing v7 gives complete, explicit web-search commands priority over the
fitted text classifier. It recognizes commands naming the web, internet, or
online search and retains the full original query for the service. Generic
“look up” requests use the original route because they can request current time
or weather. A development regression caught that distinction before freezing.

Complete causal, mechanism, role, and purpose question frames can select direct
routing. The mechanism grammar uses a bounded set of verbs. Recognized unquoted
current-information cues, clock-reading forms, extra-instruction or sentence
boundaries, and malformed quotes cause these rules to abstain. Requests outside
the grammar follow v6 unchanged; this is not a general-purpose intent parser.
Quoted subject text does not become an instruction.

The new rules always run after v6 validation and checkpoint context limits.
Plans identify the routing source and use `margin_kind: not_applicable` for
deterministic choices. A direct route still returns `direct_answer_unavailable`;
no generated answer is claimed by this helper.

## Evaluation contract

The frozen v6 source and its 94/100 report remain intact. V7 is a separately
hashed source overlay over the same 38 archived v5 candidate files. No helper
head is fitted and no Ember weight is trained. Integrated checks restore the
original fitted bytes before loading either full or dequantized INT4 checkpoints.
Both integrations share one text classifier and the same source rules; equal
scores do not establish independent LLM capability or quantized inference speed.

Three paired arms separate the changes: frozen v6, v6 routing with parser v7
only, and the complete v7 revision. Each is scored on the three consumed
100-request suites, with a separate report for each suite. New parser behavior
must also retain the historical 60 supplied-route parser checks.

Before any new confirmation is authored, the source overlay, tests, workflow,
previous suites, and recorded development requests are hashed and committed.
The new 100 requests must be exactly disjoint by normalized text from recorded
training, consumed suites, development requests (including generated arithmetic
checks), and literal strings in the frozen source. The set is assistant-authored
finite contract coverage, not blind human testing or a guarantee of semantic
independence from earlier templates. It becomes consumed after measurement.

Strict confirmation requires 100/100 on each checkpoint integration: correct
route, exact non-arithmetic arguments or equivalent arithmetic value, and one
fixture dispatch. Direct requests score routing only. The workflow preserves
all reports and enforces failures after the measurements finish; intermediate
`continue-on-error` step conclusions are not proof of a passed gate.

The live suite includes all 14 v6 cases plus the six repaired requests on each
integration. It checks actual weather, time, Brave Search, local arithmetic,
and no dispatch for explanation questions. All 40 outcomes count. Recorded
first attempts remain distinct from bounded Open-Meteo timeout recoveries.

## Results

The [62-file source freeze](../tool_assistant/data/routing-v7-source-lock.json)
was committed as `e074778dcaa7b9ea23d10ebd840cbf87ca60348d` before the
[100 new requests](../tool_assistant/data/routing-v7-confirmation.json) were
authored in its child `7d6619e02af345869fc4e72040ba8411a6d3388d`.
The source lock SHA-256 is
`560f7a3edb36ae30695a6fdb109bff7ad70180f989d688e145d10b85610f6702`;
the suite SHA-256 is
`256b788d921585d9727ce117b8402480bd8653fc5e4b1e5df59c0f6e6c231da8`.
All 20 arithmetic gold values were computed independently with exact integer or
rational operations before inference; supplied IANA zone names were validated.

Local development passes all three consumed 100-request suites and all 60
historical parser cases. The original v6 control retains 94/100 on its last
suite. Focused routing/parser/freeze/service checks pass 207 tests. These local
checks rebuild a text helper; they do not substitute for the archived-bundle
integration measurement.

The [integrated run](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34702676899)
measured source `7d6619e02af345869fc4e72040ba8411a6d3388d` with the original
archived helper and model files. All six previously reported failures are
repaired. All 300 consumed requests pass on both checkpoint integrations.
The new confirmation scores **99/100 on both**, leaving one definition request
misrouted, so the strict overall gate correctly **fails**. The live suite
separately passes **40/40**. No code, labels, or fitted bytes were changed after
these results were observed.

| Suite | Frozen v6 | V6 route + parser v7 | Complete v7 |
| --- | ---: | ---: | ---: |
| Original consumed 100 | 100/100 | 100/100 | 100/100 |
| Consumed v5 confirmation | 100/100 | 100/100 | 100/100 |
| Consumed v6 confirmation | 94/100 | 95/100 | 100/100 |
| New v7 confirmation | 82/100 | 89/100 | 99/100 |

These paired combined scores are identical for full and dequantized INT4
integrations. The new suite contains different requests from the earlier v6
suite; its results must not be presented as a directly comparable 94-to-99
benchmark gain. On the same new requests, the parser repairs seven failures
and the routing rules repair ten more. No request that passed the frozen v6
control becomes a failure. The revised parser retains 60/60 supplied-route
regressions on every measured suite.

The new confirmation passes all **80/80 tool requests**, including exact
arguments and fixture dispatch, and **19/20 direct-routing requests**. The
remaining request is `v7-new-direct-18`, “What is photosynthesis?” The frozen
classifier selects `calculator`; both v6 and v7 ask for arithmetic clarification
and dispatch no tool. This request is outside the new explanation-frame rules.
It remains an unresolved routing limitation, not a successful generated answer.
The confirmation is now consumed; a later revision needs a new freeze and new
requests. The measured v7 source and this failed report remain fixed.

All reports were recovered losslessly from bounded log chunks, matched to their
emitted SHA-256 values, and independently recounted:

| Report | SHA-256 |
| --- | --- |
| [Original development](../reports/ember-routing-v7-development-original.json) | `3c556fcaaad8506f01beeaad93346a80495897da11b8c6fa2765b74df92e53d9` |
| [V5 development](../reports/ember-routing-v7-development-v5.json) | `865daec3624b69fe038f1276e1652ad9f9f4c3c2f079819e92d16c47c57601bf` |
| [V6 development](../reports/ember-routing-v7-development-v6.json) | `a10a2469b157396c8630c9fffec282bfecab171e1507639a9f0e3b4584c2589a` |
| [New confirmation](../reports/ember-routing-v7-confirmation.json) | `0d53ba5b32dc01b295e3ec044068a2bde7da7aa0a9dc5519fc47fd8aa2cc15f5` |

## Live checks and repository validation

The [live report](../reports/ember-live-services-v7-34702676899.json) records
40/40 final passes, with 38/40 first-attempt passes and two recovered geocoding
timeouts. The exact arithmetic repair executes locally and returns 25; the
exact Barbican search reaches Brave and returns results. Each repaired
explanation question avoids tool dispatch on both integrations. Existing
weather/time behavior, including the Tromsø context repair, still passes.

| Live check | Final result |
| --- | ---: |
| Weather | 6/6 |
| Time | 8/8 |
| Local arithmetic | 6/6 |
| Brave Search | 4/4 |
| Location/time guards and direct requests with no dispatch | 16/16 |

The live report SHA-256 is
`759b8bfd92d13b04968a198a32bfbe00caf62f0d1b5c8bea16d05a6bbf3334b4`.
The [restoration report](../reports/ember-live-restore-v7-34702676899.json) has
SHA-256 `1d3a635fc1e0268fd3f7dac57d41d39e0e06c7f801523b9ecffc8d6b5e811318`.
The actual measured manifest is byte-identical to the archived original, with
SHA-256 `df8a6bb074e157403f1c4142ce8c8eae4448ee72a94b7876d661ffc4364703b4`.
All 62 source files and all 38 candidate files were verified before and after
scoring. The v5 and v6 source freezes remain intact.

The evidence artifact is `10300484535`, named
`ember-routing-v7-34702676899-attempt-1`, with SHA-256
`e00f8220539c1a46fbbf66a978ffb693d3164239f807e9e8caf71b843596e9e9`.
The [validation run](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34702679444)
on the measured source passes **981 repository tests**, **22 packaged Ember
tests**, and the separate supplied-route parser check at **60/60**. The
integration experiment passes **207 focused tests**. These successes do not
override the 99/100 confirmation failure.

Generated-answer quality remains outside this test, and `production_ready`
remains false. This is an opt-in helper revision in a draft PR.

The later [definition-routing v8 repair](ember-routing-v8.md) fixes the
photosynthesis failure and preserves this frozen source and its measured report.
V8 passes all 400 consumed requests and 20/20 new definition questions; its
broader fresh suite records two separate location-context failures.
