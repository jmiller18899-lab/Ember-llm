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

Integrated confirmation and live measurements are pending. The frozen v5
99/100 failure and the earlier live-service evidence remain unchanged.
