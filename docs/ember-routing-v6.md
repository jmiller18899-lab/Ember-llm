# Ember contextual routing v6

V6 addresses the consumed v5 confirmation miss:
“I am contacting a friend in Tromsø; is it cold there currently?”
The existing argument parser accepts this as weather for Tromsø, but the fitted
v5 helper chooses `get_time` and asks for clarification. The local text helper
also misroutes the isolated cold-weather reference question, so simply removing
the context and asking the same classifier again is insufficient.

The optional `RoutingV6Runtime` uses a complete named contact/visit context and
a complete current-place reference question to select weather or time. It reuses
the existing parser's grammar and requires successful parsing of the whole
request. Everything outside that bounded grammar follows the unchanged v5 path.
There are no added location exceptions, new parser rules, or fitted parameters.

Input validation and the checkpoint's context limit run before the contextual
choice. Plans identify their routing source; a deterministic choice has
`margin_kind: not_applicable` and uses zero as an API sentinel, not a confidence
score. The original v5 runtime and default runtime remain separate controls.

## Evaluation contract

The new source overlay is hashed separately from the original v5 bundle. All
38 archived candidate files, the original manifest, and all 30 frozen v5 source
files must remain byte-identical. The runtime loads the archived v5 head and
model bundle, plus the separately verified v6 source. Rebuilt helper bytes are
restored from retained Actions artifacts before any integrated measurement.

The two previous 100-request suites are now consumed development regressions.
Both v5 and v6 are scored on each, on full and dequantized INT4 checkpoint
integrations. Historical parser regressions still require 60/60. The new routing
confirmation requires 20 new requests per route, exact arguments for tool
requests, one fixture dispatch, and a strict 100/100 result on each integration.
Direct requests score routing only.

New requests are authored only after committing the new source freeze. Their
normalized text must be disjoint from recorded training, previous suites,
development examples, and literal strings in the frozen source/tests. This is
assistant-authored finite contract coverage, not blind human evaluation or a
claim of semantic independence from existing templates. Failed reports remain
preserved, and tests are consumed after measurement.

The live suite counts all 14 cases on each integration, including the original
weather failure and a paired time question. It checks real weather, time, search,
local arithmetic, and refusal behavior. Every attempt is recorded; first-attempt
results and bounded Open-Meteo timeout recovery stay separate. The original v5
plan for the failing request is also recorded without making a live call.

The CPU workflow starts when the subsequent confirmation file is pushed, or by
manual dispatch. Its source and runner are included in the freeze. Every measured
gate must pass; evidence is retained even if one fails. No GPU work is involved.

Both checkpoint integrations share one text head and the same deterministic
rule. Routing does not use Ember representations or run a model forward pass.
Equal scores therefore do not constitute two independent LLM confirmations.
Generated-answer quality is outside this test, and `production_ready` remains
false. This revision is an opt-in helper experiment, not a production deployment.

## Results

The [48-file source freeze](../tool_assistant/data/routing-v6-source-lock.json)
was committed as `5eb35772824ab48f691abe9e772a0c5bc1296f60` before authoring
the [100 confirmation requests](../tool_assistant/data/routing-v6-confirmation.json),
committed in its child `6961f3776de59415e6179c740a0f6b85a84dcfb1`.
The lock SHA-256 is
`25bad5f0334f355ce25fa5075534bd5cbb669f3f958ab948c10d2e6211467b94`;
the suite SHA-256 is
`4800a783c3c95cdca4bc32cfcb5a02d1f35eb8d430b94c54f29a152a1cf8dd30`.
All 20 arithmetic gold values were checked independently with exact fractions
before inference. Local focused validation passed 128 tests.

The [integrated CPU run](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34700726146)
on `6961f3776de59415e6179c740a0f6b85a84dcfb1` verified the original 38 candidate
files and all 48 source files before and after measurement. It repaired the
target weather/time confusion, passed the live suite, and **failed the overall
new-request confirmation at 94/100**. The strict gate correctly leaves the run
unsuccessful. No code, labels, or fitted bytes were changed after scoring.

Both checkpoint integrations produced these same paired results:

| Suite | Frozen v5 combined | Context v6 combined |
| --- | ---: | ---: |
| Original consumed 100 | 100/100 | 100/100 |
| Consumed v5 confirmation | 99/100 | 100/100 |
| New v6 confirmation | 91/100 | 94/100 |

On the new suite, v6 routes 95/100 correctly and passes exact arguments plus
fixture dispatch for 78/80 tool requests. It passes **20/20 weather and 20/20
time** cases. Three weather cases that fail in the paired v5 control now pass,
and no previously passing case becomes a failure. Historical parser coverage
remains 60/60. The six remaining failures also occur unchanged in the paired
v5 control:

| Request | Expected | Observed |
| --- | --- | --- |
| Calculate 3 squared plus 4 squared. | Calculator, result 25 | Correct route; parser rejects the expression |
| Search the web for this week’s events at the Barbican Centre. | Web search | Direct route |
| Why do some leaves change colour in autumn? | Direct | Time route with `autumn` as argument |
| How do roots help a plant absorb water? | Direct | Weather route; missing-location clarification |
| How does insulation reduce heat loss? | Direct | Weather route; missing-location clarification |
| What is the role of yeast in bread dough? | Direct | Time route with `bread dough` as argument |

These are newly measured limitations of the retained general classifier/parser,
not repaired by this bounded context rule. Direct-answer generation remains
untested. The new suite is now consumed regression evidence for any later
revision; a further confirmation needs a new freeze and new requests.

The [development report](../reports/ember-routing-v6-development.json) has SHA-256
`dc370ced5a1ecbb83876ab2b9bfdc163b8d9c7b4dd215f2487ae0865c5f33154`.
The [confirmation report](../reports/ember-routing-v6-confirmation.json) has SHA-256
`217eda9c5fb7768281f1bc1d40c1f30ec30b51fa1cfe3119e1952c91d448c760`.
Both were recovered losslessly from bounded log chunks, matched to their emitted
checksums, and independently recounted. The frozen v5 99/100 failure and all
earlier evidence remain preserved.

## Live result

The [live report](../reports/ember-live-services-v6-34700726146.json) records
**28/28 final passes**, with 26 first-attempt passes and two recovered timeouts.
All cases count toward the result, including the repaired request.

| Check | Final result |
| --- | ---: |
| Weather, including the exact Tromsø regression | 6/6 |
| Time, including the matching contact-context question | 8/8 |
| Local arithmetic | 4/4 |
| Brave Search | 2/2 |
| Ambiguous/unknown/future/direct guards | 8/8 |

The exact failing request now dispatches weather once for Tromsø on both
integrations and obtains valid current conditions. The full-checkpoint run
recovers a 12.193-second geocoding timeout with one retry; the INT4 run succeeds
on its first attempt. The paired time request geocodes the same place and
obtains the external current time for `Europe/Oslo` on both integrations.
The other retry recovers the INT4 ambiguous-London geocoding check, which then
correctly asks for clarification. Both failures remain visible in the trace.

The run records 30 successful HTTP responses and two geocoding timeouts. The
live report SHA-256 is
`e3723b9581706376c67afc0c324de02db7066f584c68b2812b8ca064ce0f12eb`.
The [restoration report](../reports/ember-live-restore-v6-34700726146.json) has SHA-256
`9fac6f394dd68673e57b4868ba59703a62af30cb8140f8938f86c62b977a094c`.
The restored manifest matches the original
`df8a6bb074e157403f1c4142ce8c8eae4448ee72a94b7876d661ffc4364703b4`.
The evidence artifact is `10299653847`, named
`ember-routing-v6-34700726146-attempt-1`, with SHA-256
`25c1c175af1d81f76b77124004eafcea345006b5515307a5603134a70f73291d`.

The [repository validation](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34700727650)
on the measured source passed **851 repository tests**, **22 packaged Ember
tests**, and the separate supplied-route parser check at **60/60**. The
experiment's **128 focused tests** also passed. These checks and live success
do not override the failed overall routing confirmation.
