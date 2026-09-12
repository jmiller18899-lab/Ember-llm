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
routing. The mechanism grammar uses a bounded set of verbs. Unquoted requests
for current/recent information, clock readings, extra instructions, multiple
sentences, and malformed quotes cause these rules to abstain. Requests outside
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

Local development passes all three consumed 100-request suites and all 60
historical parser cases. The original v6 control retains 94/100 on its last
suite. Focused routing/parser/freeze/service checks pass 207 tests. These local
checks rebuild a text helper; they do not substitute for the archived-bundle
integration measurement. Frozen new-request and live measurements are pending.

Generated-answer quality remains outside this test, and `production_ready`
remains false. This is an opt-in helper revision in a draft PR.
