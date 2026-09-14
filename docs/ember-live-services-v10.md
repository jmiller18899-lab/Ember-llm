# Country continuation and clock validation v10

V9 repaired the contact parser and passed its fixture confirmation, but the live
Fes request still needed a country. The geocoder returned its maximum 100 rows,
so one exact match in that incomplete list could not justify choosing a city.
The country-qualified follow-up reached the clock service, but its DST label
contradicted local ZoneInfo metadata even though both agreed on UTC+01:00.

V10 is an optional service layer over unchanged v9 routing. The original live
implementation, v9 source locks, candidate bytes, and failed reports are retained.

## Explicit country reply

`ServiceConversation` keeps one pending weather/time request per instance. A
geocoding ambiguity asks which country the named city is in. `reply("Morocco")`
resumes that same tool with `Fes, Morocco`, rather than routing the isolated word
Morocco as a new task. A new request, cancellation, or a completed reply clears
the pending state. Repeated replies do not duplicate the tool call. Invalid
country text dispatches nothing. Each conversation must have its own instance.
The service never selects a country automatically from provider suggestions.

```python
from tool_assistant.live_services_v10 import LiveServicesV10, ServiceConversation
from tool_assistant.runtime_v9 import RoutingV9Runtime

runtime = RoutingV9Runtime(bundle_path, "full")
conversation = ServiceConversation(runtime, LiveServicesV10())
initial = conversation.run(
    "We are phoning a client in Fes; what time is it there currently?"
)
# Display the country question and wait for an actual user reply.
result = conversation.reply("Morocco")
```

This is a library integration point, not a deployed chat UI. `run` starts a new
request; `reply` is only for answering the pending country question or cancelling.
State is in-memory and is not shared or persisted across conversations.

## Clock contract

The exact requested IANA zone, timezone-aware timestamp, actual UTC offset,
integer `utc_offset_seconds`, and 120-second freshness window must agree. Missing
or malformed DST labels are still rejected. A disagreement in the two valid
boolean DST labels no longer rejects an otherwise verified time; the result
retains `dst_active` with `dst_active_source="provider"`, plus
`zoneinfo_dst_active` and `dst_label_agreement` so the disagreement stays visible.
This deliberately changes only the acceptance rule for differing DST labels.

Python's [ZoneInfo documentation](https://docs.python.org/3/library/zoneinfo.html)
explains that it uses system time-zone data, falling back to the tzdata package.
The [datetime documentation](https://docs.python.org/3/library/datetime.html#datetime.tzinfo.utcoffset)
defines `utcoffset()` as the total UTC offset, including any DST adjustment.
The observed provider/local label mismatch does not by itself identify which
metadata convention or database version caused it. The new contract verifies the
actual offset and instant and records both interpretations without claiming they
agree. No other time-validation failure is converted into success.

## Evaluation contract

The service source was frozen before the live run. The runner restores the same
full and dequantized INT4 integrations and verifies all v9 source/candidate hashes
before and after measurement. No model weight or routing head is trained.

The new live contract has 52 turn checks: 48 unchanged service/guard checks and
four country-clarification turns (question plus explicit reply per integration).
The test supplies Morocco as the user reply; success is not autonomous guessing.
The entire original 50-check v9 contract is also recorded separately, including
its unqualified Fes failures. A new assisted-flow PASS must not overwrite or
relabel those original failures. First attempts and recovered retries remain
separate. This does not establish continuous uptime or direct-answer quality.

Local offline validation passed 98 service tests, including offset errors, stale
and future clocks, malformed labels, non-hour offsets, cancellation, isolation,
and duplicate-reply prevention. Live results follow once verified.

## Verified result

[Live run 34856109144](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34856109144)
passed the new contract: **52/52 final outcomes, 49/52 first attempts**. Three
eligible Open-Meteo timeouts recovered through the unchanged retry policy.
All four Fes clarification turns passed. Both qualified time results record
`dst_label_agreement=false` while returning a verified timezone, offset, and
fresh timestamp. The original unqualified 50-check contract still records
48/50; it is retained separately and is not relabeled as passed.

The source freeze is `f8a8ffe5574820b18d8dc0d2eb08f0fdd150b18d`; the subsequent
run trigger is `64075716fe77132f57cfe0c6ce04be2812eb06bc`.
The run passed 124 offline routing/service/restoration tests and verified all
nested v9 source locks and original candidate files before and after live calls.
Both full and dequantized INT4 integrations were exercised. These integrations
share the same archived routing helper; the run is not new LLM training.

The live and restoration JSON reports were recovered losslessly from indexed,
SHA-256-verified log chunks. The artifact is `10353625947`, with ZIP SHA-256
`750cb671ece9e6f7ad4fd571e1111b15a99cabe6de3d275339dbdb6bb6dc1890`.
The metrics scorecard now separates the new assisted-flow result from original
unqualified outcomes and retains the direct-answer learning failure.
Eleven scorecard tests passed, including rejection of rewritten original counts.

This optional service layer is saved in the draft tool-assistant PR. It has not
been deployed or wired into ClawAgent's chat UI. Its integration requires a
separate ServiceConversation per conversation and explicit handling of a country
reply. Direct-answer quality and broad production readiness remain unproven.
