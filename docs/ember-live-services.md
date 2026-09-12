# Ember live service smoke test

The frozen v5 helper previously had caller-supplied fixture handlers only. This
revision adds an opt-in service layer and a small end-to-end live test. It does
not modify routing, parser source, fitted heads, or model weights. The 99/100
routing confirmation remains a failed historical result.

## Service contract

| Tool | Implementation | Live check |
| --- | --- | --- |
| Weather | Open-Meteo geocoding and current weather model conditions | Two cities, including a Unicode name and a request with context; coordinates, units, fields, freshness, and provider attribution |
| Current time | TimeAPI.io with IANA zones; Open-Meteo geocoding for city names | UTC, a 45-minute offset, and city-to-zone resolution; external timestamp, offset, DST, and clock freshness |
| Calculator | Existing bounded arithmetic evaluator | Parentheses and a percentage, executed locally |
| Web search | Brave Search API | A real search returning provider URLs, when `BRAVE_SEARCH_API_KEY` is configured |

Official provider contracts are [Open-Meteo weather](https://open-meteo.com/en/docs),
[Open-Meteo geocoding](https://open-meteo.com/en/docs/geocoding-api),
[TimeAPI.io OpenAPI](https://timeapi.io/swagger/v1/swagger.json), and
[Brave web search](https://api-dashboard.search.brave.com/app/documentation/web-search/get-started).
Open-Meteo's current values are weather model conditions, not a guarantee of an
on-site sensor observation. Place lookup requires one exact populated-place name
after normalization; a country or full region name can qualify it. Aliases,
abbreviations, and uncertain or truncated results can need clarification.

The service layer calls the existing runtime's `run` method. A missing search
key produces `tool_unavailable`; ambiguous or unresolved places produce
`needs_clarification`; network errors, invalid data, and stale responses produce
`tool_error`. It does not fill failures with fabricated results. Requests have
timeouts and response-size limits, retain TLS verification, and make one attempt.
Authentication headers and raw exception details are omitted from evidence.

## Reproduce

Given the previously verified v5 bundle:

```bash
python -m pytest -q tests/test_live_services.py
python -m tool_assistant.live_smoke --bundle routing-v5-candidate \
  --report evidence/live-services-smoke.json
```

The GitHub Actions workflow rebuilds the preserved assets with the same pinned
CPU dependencies and existing `HF_TOKEN` secret. It retrieves the original fitted
helpers from the retained base, v4, and v5 evaluation artifacts, checks every
archived helper against the freeze, and restores those exact bytes before live
measurement. All other candidate files must already match. The original
manifest is restored after validation; model weights and the freeze are
unchanged. The workflow uses `BRAVE_SEARCH_API_KEY` from
repository secrets if available. No account, subscription, or paid compute is
provisioned. At most one search is requested per checkpoint integration.

The local `--head` option checks the fitted text-helper hash and all frozen source
hashes, then exercises the same routes, parser, and dispatch against live
services. It is labeled a preflight because it excludes checkpoint loading and
Ember's tokenizer/context limit. Hosted `--bundle` testing includes those checks
for both full precision and dequantized INT4. Both use the same text router.

Reports are never overwritten. The smoke test verifies the 30 frozen source
files and all 38 fitted-candidate files before and after hosted measurement.
Every HTTP attempt records its provider, URL, status, duration, and response hash
when available. The normalized service result records actual returned values,
timestamps, and source attribution. Exact JSON evidence is also emitted into logs
for checksum-verified readback.

## Scoring and limits

Each integration runs eight service checks and four guards: ambiguous city,
unknown city, future weather, and direct requests without dispatch. Calculator
execution is local; it is not counted as an external service. A missing search
key is `BLOCKED`, never a pass. An unexpected failed check gives `FAIL`; blocked
checks with no unexpected failures give `PARTIAL`. Both exit unsuccessfully after
writing evidence, so a red workflow can represent unavailable credentials.

The known Tromsø weather-to-time routing failure is recorded separately on the
unchanged helper. It is excluded from the live-service counts but remains a
release blocker. Injected timeout, HTTP 401/429/503, malformed response, and
staleness checks are explicitly offline tests; the smoke run does not deliberately
disrupt providers. Direct-answer generation, search-result relevance, service
uptime over time, and production deployment are outside this test.

## Measurement status

Local service contracts: **46 passed**. Archive restoration has three additional
checks for exact restoration, a mismatched archive, and changed model weights.

The [first hosted attempt](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34695406601)
stopped before any service calls because six rebuilt fitted files had different
checksums. All 32 other candidate files matched, including the original model
weights. The local text-helper rebuild also failed the same fitted-file check.
The mismatch does not establish changed model behavior; it prevents claiming an
identical candidate. The archive restoration step preserves the original freeze.
The initial manifest and blocked-run details are retained under
`reports/ember-live-rebuild-*.json`. The hosted repository check passed 769 tests.

The [completed live run](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34695925118)
measured source `db7161091eb1974acc1cf412572101e21d4dfec2` after restoring all six
archived helper files. All 30 frozen source hashes and all 38 candidate file
hashes matched before and after live testing.

| Check | Full precision | INT4 integration | Result |
| --- | ---: | ---: | --- |
| Weather | 2/2 | 1/2 | Three real weather responses; one geocoding timeout |
| Current time | 3/3 | 3/3 | Six real clock responses, including city lookup and a 45-minute offset |
| Calculator | 2/2 | 2/2 | Four correct local executions |
| Web search | 1 blocked | 1 blocked | No `BRAVE_SEARCH_API_KEY` was configured; no search HTTP request was made |
| Ambiguous/unknown city, future weather, direct dispatch guards | 4/4 | 4/4 | Expected clarification or no-dispatch behavior |

Of 24 service/guard checks, **21 passed, one failed, and two were blocked**.
The report and workflow correctly record **FAIL**. The trace contains 18 HTTP
200 responses (nine geocoding, three weather, six clock) and one geocoding
attempt without a response. Both checkpoint integrations use the same archived
text helper; these are separate requests, not independent LLM capability tests.

The failed live request was `weather_unicode` on the INT4 integration:

> What is the current weather in "Tromsø, Norway"?

The helper correctly selected `weather` and extracted `Tromsø, Norway`. The
geocoding request timed out after about 12.19 seconds, and the adapter returned
`tool_error` with `code: timeout`. It did not call the weather endpoint or invent
conditions. The same request succeeded earlier in the full-precision run. This
single-run difference is a service-call failure, not evidence of an INT4 routing
error. No retry result is substituted for this failure.

The prior “contacting a friend in Tromsø” weather-to-time error was also
reproduced on both integrations. Those two known failures are reported separately
from the 24 service/guard checks and remain a release blocker. Generated answers
remain untested.

The [live report](../reports/ember-live-services-smoke.json) has SHA-256
`55f8bba75adf8988348142239785a227fc5c260285053a30b70ea79560a76799`.
The [restoration record](../reports/ember-live-candidate-restore.json) has SHA-256
`5107a3c55fdcd4382091dc2bd1edb3c0db278fff1027aa51eb110c4f9a3bd5a4`.
The measured [candidate manifest](../reports/ember-routing-v5-manifest.json) is
byte-identical to the pre-confirmation freeze, SHA-256
`df8a6bb074e157403f1c4142ce8c8eae4448ee72a94b7876d661ffc4364703b4`.
All were recovered from exact log chunks and checksum-verified. Run artifact
`10299210462` has SHA-256
`95e2156a3f78086382fd0d99cb0463804da6fd62fe7fd5279dd2abe39acf7c3b`.

The [final source validation](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34695926852)
passed **772 repository tests**, **22 packaged Ember tests**, and the historical
parser-v3 **60/60** check. The live workflow separately passed **49** service and
restoration tests. The remaining live-test work is to configure a Brave Search
key and verify search, and to address intermittent geocoding availability while
preserving explicit failures. No production deployment is qualified by this run.
