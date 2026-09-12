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

The GitHub Actions workflow rebuilds the candidate with the same pinned CPU
dependencies and existing `HF_TOKEN` secret. It uses `BRAVE_SEARCH_API_KEY` from
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

Local contract tests: **46 passed**. Live measurements are in progress.
