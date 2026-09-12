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
socket timeouts and response-size limits and retain TLS verification. Open-Meteo
geocoding and weather GETs now retry timeout errors up to twice, waiting 0.5 and
1 second. Other errors and other providers keep one attempt. Each HTTP attempt,
including any failed attempt followed by recovery, remains in the evidence.
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
unchanged. The workflow uses `BRAVE_SEARCH_API_KEY` from repository secrets,
falling back to the repository variable with the same name when the secret is
empty. It also accepts `BRAVE_API_KEY` and `BRAVE_SEARCH_KEY`, in that order,
checking each name's secret before its variable. A short prerequisite job
reports only name availability and the selected source. If none is available,
it exits unsuccessfully with `BLOCKED` and skips checkpoint preparation.
The caller passes the selected value as a secret to the local reusable
workflow, so it is masked before the runner logs step environments. Both
workflow files are included in the report's source checksums. This follows
GitHub's [reusable workflow secret mapping](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows)
and [allowed expression contexts](https://docs.github.com/en/actions/reference/workflows-and-actions/contexts#context-availability).
No account, subscription, or paid compute is
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

Schema-2 reports additionally record a logical request ID, attempt number,
transport phase, outcome, sanitized error code, and any retry delay. A successful
retry completes the same GET and does not repeat the tool dispatch or previously
successful geocoding. The 12-second timeout applies to blocking socket operations,
as described by [Python's urllib documentation](https://docs.python.org/3/library/urllib.request.html#urllib.request.urlopen);
it is not a hard wall-clock deadline. At most three attempts are made per
Open-Meteo GET. A weather request can require both geocoding and forecast GETs.

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

Starting with schema 2, the service result may pass after a bounded timeout retry.
`first_attempt_counts` preserves success/failure before recovery, and
`retry_summary` counts affected cases, recovered cases, and additional HTTP
attempts. The scorer validates the complete attempt sequence: consecutive
attempt numbers, the same URL and request ID, only eligible timeout retries,
the allowed attempt limit, and a final complete HTTP 200 response with a body
checksum. It rejects skipped attempt numbers, altered retry URLs, retries of
permanent errors, and additional completed requests. Schema-1 reports retain their original
single-attempt outcomes.

## Measurement status

Initial local service contracts: **46 passed**. Archive restoration has three additional
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

## Credential follow-up, 2026-09-12

The [follow-up run](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34696749620)
measured source `00a38d987440746ff2915a46aa461e46270bb416` after the user reported
adding the Brave key. The workflow still received an empty
`BRAVE_SEARCH_API_KEY`. Both search checks returned `missing_credentials`, and
neither made an HTTP request. This establishes that the key was unavailable to
this job; it does not establish the name or scope of the saved secret. The
workflow binds the environment variable to `secrets.BRAVE_SEARCH_API_KEY` and
does not select a GitHub environment.

The service/guard result was again **21 passed, one failed, and two blocked**:
weather 3/4, current time 6/6, local calculator 4/4, search 0/2 verified, and
guards 8/8. The INT4 Tromsø weather request succeeded on this run. The INT4
Reykjavík weather request instead timed out at the forecast endpoint after
about 12.17 seconds, following successful geocoding. Its route and parsed
location were correct; the adapter returned an explicit `tool_error`. There
were 19 HTTP 200 responses (ten geocoding, three weather, six clock) and one
weather attempt without a response. The two known routing failures were
reproduced separately. The overall result remains **FAIL**.

All 30 frozen source files and 38 candidate files were verified before and after
measurement. All six rebuilt helper hashes also matched their archived hashes
on this run; the restoration step still verified the archived bytes and restored
the original manifest. Earlier mismatched rebuilds and measured failures remain
preserved. Artifact names now include the run attempt so a retry can retain its
own evidence.

The exact [follow-up report](../reports/ember-live-services-34696749620.json)
has SHA-256 `b4b32f8ad7e96d70f55699dfacfa13944c9f669c89d8268ef5ca6cd360de83fa`.
The [restoration record](../reports/ember-live-restore-34696749620.json) has
SHA-256 `b6671acba211db4e8dd731b60eddc9e3df7f323c3d66b4dd2f5e509e72bb31ac`.
Both were recovered from log chunks and checksum-verified. The measured
candidate manifest still matches the original `df8a6bb0…4703b4` manifest above.
Artifact `10299580023`, named `ember-live-services-34696749620-attempt-1`, has
SHA-256 `66b3360f1cf6d7537d2d217ed7a837bca21715483b63cebd9fa18a78004a9b28`.

[Source validation](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34696751001)
passed 772 repository tests, 22 packaged tests, and historical parser-v3 60/60.
The live workflow passed all 49 service/restoration tests. Completing search
verification requires resolving the saved secret's exact name and scope so
this job can receive it. No key value is recorded in the evidence.

## Repository-variable follow-up, 2026-09-12

The [variable-enabled run](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34697672365)
measured source `e6d86ceeee4780fa63c6c3f427f361cdd07ceeb1` after adding a
`vars.BRAVE_SEARCH_API_KEY` fallback through a local reusable workflow secret.
The runner still received an empty key. Both search checks remain **BLOCKED**
with no Brave HTTP requests. This does not determine the saved variable's name
or scope.

The result was **21 passed, one failed, and two blocked** service/guard checks:
weather 4/4, current time 5/6, local calculator 4/4, and guards 8/8. The INT4
Reykjavík time request timed out during geocoding after about 12.15 seconds;
the clock endpoint was not called for that request. The trace contains 18
HTTP 200 responses (nine geocoding, four weather, five clock) and one geocoding
attempt without a response. The two preserved routing failures remain separate,
and the overall result remains **FAIL**.

All 30 frozen source and 38 candidate checksums matched before and after testing.
The exact [report](../reports/ember-live-services-34697672365.json) has SHA-256
`284cde7bc959c3bea37ac8b5255a1c7dfb8b1fd37e9441162bcb173a57e64529`;
the [restoration record](../reports/ember-live-restore-34697672365.json) has
SHA-256 `af6f1c6a90812f8f6c170f265704de1761846d861dea8fc903c4e507be50397f`.
Both were recovered exactly from log chunks and checksum-verified. The original
candidate manifest still matches. Artifact `10299077135`, named
`ember-live-services-34697672365-attempt-1`, has SHA-256
`0fa5bf5794cd6bbb6eb52fa42986052def4bebc21e2249d6ffe86ce7b92ac097`.

[Validation](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34697673858)
passed 772 repository tests, 22 packaged tests, and historical parser-v3 60/60;
all 49 service/restoration tests also passed. The follow-up adds a quick
availability check for the three supported Brave credential names before
another checkpoint rebuild. It inspects only booleans, never the key values.

## Brave verified, 2026-09-12

The [credential-name check](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34698068507/job/103564886205)
found `vars.BRAVE_API_KEY`. The canonical `BRAVE_SEARCH_API_KEY` name was
unavailable in both Secrets and Variables. The supported alias resolved this
name mismatch and was passed as a workflow secret; the live job's environment
log shows `BRAVE_SEARCH_API_KEY: ***`.

The [live run](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34698068507)
measured source `cddf9e147576d36a7aaa65575064f1585089faf8` on both frozen
checkpoint integrations. Brave returned **HTTP 200 on both requests**, with
three URLs per response, including `https://docs.python.org/3/` and
`https://www.python.org/doc/`. The calls took about 1.04 and 0.39 seconds.
No key value is included in the report or credential-name record.

| Check | Full precision | INT4 integration |
| --- | ---: | ---: |
| Weather | 1/2 | 0/2 |
| Current time | 3/3 | 3/3 |
| Local calculator | 2/2 | 2/2 |
| Brave web search | 1/1 | 1/1 |
| Ambiguity, unknown city, future weather, direct dispatch guards | 4/4 | 4/4 |

The 24 service/guard checks recorded **21 passed, three failed, and none
blocked**. All three failures were correctly routed weather requests that
timed out: full-precision Tromsø at the forecast endpoint (12.23 seconds),
INT4 Tromsø at geocoding (12.13 seconds), and INT4 Reykjavík at the forecast
endpoint (12.05 seconds). Each returned an explicit `tool_error`. The trace
contains 18 HTTP 200 responses (nine geocoding, one weather, six clock, two
search) and three attempts without a response. The two known weather-to-time
routing failures remain separate. The overall result remains **FAIL**, and
`production_ready` remains false. No retry replaces any measured failure.

All 30 frozen source and 38 candidate hashes matched before and after testing.
The exact [live report](../reports/ember-live-services-34698068507.json) has
SHA-256 `3f29ebcc7dd7f6729b14177c53dd62dba5768b7bd69fe01fd80b0e31d63801ad`;
the [restoration record](../reports/ember-live-restore-34698068507.json) has
SHA-256 `3f9b663b534983b17e40fc32b125781efe59ace8ce84df863ea3adad2164caf4`.
Both were recovered from exact log chunks and checksum-verified. The measured
manifest remains byte-identical to the original freeze. The separate
[credential-name record](../reports/ember-brave-credential-34698068507.json)
is explicitly derived from the availability event and has SHA-256
`b38ce0e575ef5bc540b441aeed26f02ea3a65b9567a8957b06d7cb743953e25d`.
Artifact `10299461662`, named `ember-live-services-34698068507-attempt-1`, has
SHA-256 `fb9a041c5b510522a8390cee55cbe440aeedf7e2c45fec2a7d002239f49e9063`.

[Validation](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34698070777)
passed 772 repository tests, 22 packaged tests, and historical parser-v3 60/60.
The live job passed all 49 service/restoration tests. Brave search is now
verified on this bounded smoke check; weather availability and the preserved
routing error remain unresolved. Generated-answer quality remains untested.

## Weather timeout recovery

The earlier reports show the same valid Open-Meteo requests sometimes succeeding
and sometimes timing out, at both geocoding and forecast. They do not identify
whether the underlying delay occurred in connection setup, the network, or the
provider. The client previously abandoned each request after its first timeout.
The recovery change retries that individual GET up to twice and records every
attempt, including whether it failed while opening the response or reading its
body. Exhaustion still returns an explicit `tool_error`.

The change preserves the frozen router, parser, model files, exact place matching,
and current-weather validation. HTTP errors, malformed JSON, invalid units,
stale conditions, and ambiguous places retain their existing error or clarification
behavior. HTTP errors such as 429 and 503 are not retried by this policy.

Local verification passed **77 service/restoration tests**, including timeout
recovery, exhaustion, partial-body timeouts, one tool dispatch, one successful
geocoding call despite forecast retries, unchanged clock/search behavior, stale
data and ambiguity after recovery, and rejection of incomplete or misleading
HTTP traces. Live measurement records first-attempt and recovered results
separately.

The [live recovery run](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34699255050)
measured source `db7f4dfc512dbb76fa46699f78297583086db318` and passed all
**24 service/guard checks**. Before retries, **23 passed and one failed**.
Weather passed **4/4 after recovery**, compared with **3/4 on the first attempt**.

| Check | Full precision | INT4 integration |
| --- | ---: | ---: |
| Weather | 2/2 | 2/2, one recovered request |
| Current time | 3/3 | 3/3 |
| Local calculator | 2/2 | 2/2 |
| Brave web search | 1/1 | 1/1 |
| Ambiguity, unknown city, future weather, direct dispatch guards | 4/4 | 4/4 |

The INT4 Reykjavík weather request hit a real geocoding timeout after 12.127
seconds while opening the response. After the configured 0.5-second pause,
the same GET returned HTTP 200 in 0.428 seconds. The forecast then returned
HTTP 200 in 0.407 seconds, with current conditions for the correctly resolved
Reykjavík location. The original timeout remains in the report. The tool was
dispatched once, and the successful forecast was requested once. This run
demonstrates recovery from an actual external timeout, alongside the injected
offline tests; it does not establish uninterrupted provider availability.

The trace contains **22 complete HTTP 200 responses** (ten geocoding, four
weather, six clock, two search) and the one failed geocoding attempt. There
were no blocked checks. The live-service workflow is **PASS** under the
documented retry policy. The two known weather-to-time routing failures remain
separate, the frozen 99/100 routing confirmation still fails, and
`production_ready` remains false.

All 30 frozen source and 38 candidate hashes matched before and after testing.
The exact [live report](../reports/ember-live-services-34699255050.json) has
SHA-256 `3d40438368cd6178969469e6cc582a6611ed700e4a014dff82c85c0401ec36c1`;
the [restoration record](../reports/ember-live-restore-34699255050.json) has
SHA-256 `11c4e3de3d371c7dba7c27f07e892e9530777460672bac7613e17579d8a829a6`.
Both were recovered from exact log chunks and checksum-verified, including
independent recounts of first-attempt outcomes and recovery counts. The measured
manifest still matches the original freeze byte for byte. Artifact `10300305546`,
named `ember-live-services-34699255050-attempt-1`, has SHA-256
`aecb513ad0e2df6ac65d5c0ac08e07403cabddc8a9efcb6c9d55df8e75039c5b`.

[Source validation](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34699256827)
passed **800 repository tests**, **22 packaged tests**, and historical parser-v3
**60/60**. The live workflow passed all **77 service/restoration tests**.

## Context routing follow-up

The optional [v6 routing revision](ember-routing-v6.md) repairs the Tromsø
weather/time mix-up and passes 28/28 live checks on source
`6961f3776de59415e6179c740a0f6b85a84dcfb1`, including the exact previously failing
request and its matching time question. Two geocoding timeouts recover through
the existing bounded retry policy; first-attempt results are 26/28. All 40 new
weather/time requests pass, while six other cases leave the broader confirmation
at 94/100 and its strict gate failed. The frozen v5 measurements above remain
historical evidence for that unchanged helper.
