# Ember definition routing v8

V8 fixes the remaining v7 failure: “What is photosynthesis?” selected the
calculator and returned an arithmetic clarification. The new optional rule
recognizes short definition questions and selects `direct`, with no tool call.
The original v7 source and its 99/100 measurement remain unchanged.

## Behavior and scope

The rule recognizes complete “What is”, “What are”, and “What's” questions,
including curly apostrophes, an optional polite prefix or suffix, and a quoted
term. A term contains one to eight words made from letters, with internal
hyphens or apostrophes. It does not enumerate definition subjects or special-case
photosynthesis.

Arithmetic words, numbers, service readings, recognized changing-information
cues, and additional clauses cause the new rule to abstain. Those requests keep
the exact v7 plan apart from the revision tag. This is deliberately bounded
definition recognition, not general intent understanding: some definitions
containing tool vocabulary or longer clauses still use the old classifier.

V7 routing always runs first, retaining input validation and checkpoint context
limits. The parser remains `argument-parser-v7`. Deterministic decisions identify
`routing_source: definition_question` and `margin_kind: not_applicable`.
A direct result still has status `direct_answer_unavailable`; this repair does
not add answer generation.

## Frozen comparison

The [74-file source lock](../tool_assistant/data/routing-v8-source-lock.json)
was committed in `494f856645b31551d7a923cce487686c8eb75837` before the
[100 new requests](../tool_assistant/data/routing-v8-confirmation.json) were
authored and committed in `04b1144a9c49c76ab2e22bfcd100c887c3bcff2f`.

- Source-lock SHA-256: `cf1b9b3d354836bbf857666d7f79f562eb69dcd28c46d3f608f00cdbd3d44c4c`.
- New-suite SHA-256: `f4bb2f9623ea69b7c52904720fdeea286d51b8acda4c7452e3efb23a279ce0be`.
- Original candidate manifest: `df8a6bb074e157403f1c4142ce8c8eae4448ee72a94b7876d661ffc4364703b4`.

The new suite contains 20 requests per route. Definition subjects, numeric
questions, and live-service requests cover the intended distinction. Normalized
text hashes are exactly disjoint from recorded training, prior suites, and
development requests. Two duplicate draft requests were replaced before any
inference. Arithmetic gold values were independently calculated with integer and
fraction operations; supplied IANA zones were checked before scoring.
These assistant-authored requests provide finite contract coverage, not blind
human testing or guaranteed semantic independence from earlier templates.

The comparison uses frozen v7 and the v8 overlay on the same requests, for both
full and dequantized INT4 checkpoint integrations. It restores all 38 original
candidate files and verifies their hashes, plus all nested source freezes,
before and after scoring. Both integrations use one archived text classifier;
equal scores do not establish independent LLM capability or INT4 inference speed.
No Ember model or routing head is trained for this repair.

Strict confirmation requires 100/100 on each integration: correct routing,
exact tool arguments or equivalent arithmetic, and one fixture dispatch.
Direct requests score routing only. Four previous 100-request suites are now
development regressions. The unchanged parser also retains its separate
60-case supplied-route check.

The live suite retains all 40 previous checks and adds the definition repair,
a “What is” word-power calculation, and division-by-zero refusal on each
integration, for 46 total checks. Every outcome counts, and first attempts are
reported separately from bounded Open-Meteo timeout recoveries.

## Measured results

The [integrated run](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34704176690)
verifies the exact photosynthesis repair and passes all **400 consumed requests**
on both checkpoint integrations. The fresh suite passes **20/20 definition
questions** and scores **98/100 overall**. Two location-context requests also
fail in the unchanged v7 control, so the strict overall gate remains **FAIL**.

| Same requests | Frozen v7 | Definition v8 |
| --- | ---: | ---: |
| Original consumed 100 | 100/100 | 100/100 |
| Consumed v5 confirmation | 100/100 | 100/100 |
| Consumed v6 confirmation | 100/100 | 100/100 |
| Consumed v7 confirmation | 99/100 | 100/100 |
| Fresh v8 confirmation | 82/100 | 98/100 |

Full and dequantized INT4 have the same results. On the same fresh requests,
v8 repairs 16 definition-routing failures and introduces no regressions.
Fresh routing is 99/100; exact tool arguments and fixture dispatch pass 78/80.
Calculator, search, and direct routing each pass 20/20; weather and time each
pass 19/20. The unchanged parser retains 60/60 supplied-route checks.
The earlier 99/100 and current 98/100 scores use different request sets and
must not be interpreted as a decline on the same benchmark.

The two remaining failures are preserved in the raw confirmation:

| Request | Expected | Observed in both v7 and v8 |
| --- | --- | --- |
| `v8-new-weather-06`: “I am contacting my cousin in Akureyri; is it cold there now?” | Weather for Akureyri | Time route; ambiguous-location clarification; no dispatch |
| `v8-new-get_time-09`: “We are phoning a client in Fes; what time is it there currently?” | Time for Fes | Correct route; ambiguous-location clarification; no dispatch |

The existing `PLACE_CONTEXT` grammar in resolver v4 recognizes a fixed set of
contact descriptions, including friends, colleagues, relatives, and customers.
It does not recognize “my cousin” or “a client”. That prevents the v6 context
rule and location parser from binding these antecedents. This is separate from
definition routing and is left unchanged in this measured revision. The fresh
suite is now consumed; any later repair needs a new source freeze and new
confirmation rather than editing these results.

The [live report](../reports/ember-live-services-v8-34704176690.json) passes
**46/46**, with **44 first-attempt passes** and two recovered geocoding timeouts.
The exact photosynthesis question selects direct with no dispatch or HTTP on
both integrations; the preserved v7 plan still selects calculator. The new
word-power boundary returns 25 and division by zero requests clarification.
Weather, time, Brave Search, and all previous live checks remain successful.
The trace records 32 HTTP 200 responses and two initial geocoding timeouts,
each recovered on the second attempt after the unchanged 0.5-second delay.

The [CPU validation run](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34704178402)
on the measured source passes **1,058 repository tests**, **22 packaged Ember
tests**, and the separate **60/60 parser check**. The integrated run passes
**154 focused tests**. These checks do not override the failed 100/100 gate.

## Evidence

All reports were recovered losslessly from the integrated run's 58 bounded
evidence chunks. Chunk indexes, counts, and SHA-256 values were verified.
An independent recount confirmed every reported route/argument/dispatch count,
paired gains, absence of measured regressions, and all live outcomes. The
restored manifest matches the original archive byte for byte. All nested source
freezes and all 38 candidate files remain intact.

| Report | SHA-256 |
| --- | --- |
| [Original development](../reports/ember-routing-v8-development-original.json) | `f22c830bae84374c6fed65e2181e2e440fca42a355843501ceceeafbd73b461d` |
| [V5 development](../reports/ember-routing-v8-development-v5.json) | `ea637356bc9d54d6ff6c2501d1d7b3dd89ae095b7d7d23c2ff3a49a17b444c3d` |
| [V6 development](../reports/ember-routing-v8-development-v6.json) | `41fb978814641aa92976bda2adfe066f63fae358f5dcf04b17e39fc716b68f6d` |
| [V7 development](../reports/ember-routing-v8-development-v7.json) | `3e0c95fd65991e17744fe1620636605af1e9ef7ffcf9d5d2bf11bec164b06a6f` |
| [Fresh confirmation](../reports/ember-routing-v8-confirmation.json) | `8851601d89f18ddcfdd1afa37083c21f7bbff27295707f2b97cf8eefca0e5970` |
| [Live services](../reports/ember-live-services-v8-34704176690.json) | `1c89c6339b0d14d3e585ebf477f3458b709af4ab9609a01286857b4efb08ac93` |
| [Candidate restoration](../reports/ember-live-restore-v8-34704176690.json) | `ac0ad9f7afc17f08a548be19506a3e985c886fb0b7af27afa7ccfde31791f141` |

Artifact `10301492413`, named `ember-routing-v8-34704176690-attempt-1`, has
SHA-256 `05ab9141588731cbed3335e23b1462980d47ea6d3345c985653b4fc54a1795ce`.
No code, labels, or model bytes were changed after observing these results.
Generated-answer quality remains untested and `production_ready` remains false.
This optional helper revision is saved in a draft PR and is not deployed.
