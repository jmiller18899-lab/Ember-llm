# Contact-location repair v9

The latest v8 confirmation failed two requests: contacting “my cousin” in
Akureyri for weather, and phoning “a client” in Fes for time. The older contact
grammar did not recognize those descriptions.

V9 adds a separate optional runtime overlay for `a/my/our/the cousin/client`
within the existing first-person contact/visit frames. It requires one complete
context clause, one recognized current-place question, and one validated place.
It uses the original input validation, checkpoint context limit, and place
validation. Unsupported or ambiguous requests retain the v8 fallback. The prior
source, model parameters, fitted helper, and measured reports stay intact.

The source freeze was committed as
`b88be6bbe0b5978ed29602759dc554d30b554386`. Its child
`a7617b269813d665b23fd0fe257d375f5087e26d` adds 100 new requests, 20 per route.
The confirmation checks normalized-text separation from recorded training,
consumed suites, and source literals. It is assistant-authored finite contract
coverage, not blind human evaluation or guaranteed template independence.

Local development checks passed all 500 consumed requests with no measured
regressions. The focused v8/v9 test run passed 90 tests. These are text-helper
checks, not checkpoint integration or new LLM training evidence.

The integrated runner restores all archived fitted candidate bytes, verifies
nested source freezes before and after evaluation, compares v8 and v9 on each
identical suite, and rechecks the earlier live service cases plus both repairs.
Direct routing continues to mean no tool dispatch, not correct answer generation.
This revision does not alter the direct-answer model or authorize deployment.

## Integrated measurements

[Run 34851973951](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34851973951)
restored all original candidate bytes and passed all five consumed 100-request
suites on both checkpoint integrations. On the same fresh requests, v8 scored
80/100 combined and v9 scored **100/100**. V9 routed 100/100 correctly and passed
80/80 exact tool arguments and fixture dispatches. Twenty contact requests were
repaired, with no passing control case regressing. The 60 supplied-route parser
checks also passed. The experiment's offline tests passed 100/100.

This is a real PASS under the frozen finite routing contract. Both integrations
share the same archived text head; these gains are not model-weight learning.
The new confirmation is now consumed and must not be reused as fresh evidence.

The original live suite remains **FAIL: 48/50 final, 46/50 first attempt**.
Two earlier service timeouts recovered. Both failed cases are the unqualified
Fes time request, once per integration. The v9 parser correctly dispatches
`get_time(timezone="Fes")`; the service asks for a country because the provider
returns a saturated 100-result geocoding list. One exact city among the returned
rows does not establish that no further exact cities were omitted.

A separately recorded local text-helper follow-up with **Fes, Morocco** resolved
the country and reached the clock service. It then failed the existing clock
response validation. A diagnostic read showed the provider's `dst_active=false`
while local ZoneInfo marked DST active; both agreed on UTC+01:00. This is an
additional service-contract issue, not another contact-parser failure. No guard
was removed and neither failed result was relabeled as success.

The integrated workflow is correctly red because its live gate fails, despite
its successful routing confirmation. Production readiness remains false.
The next service work is to handle a user's country clarification and validate
clock-provider DST semantics without weakening time-zone offset/freshness checks.
Direct-answer learning remains a separate failed gate and is unchanged.

All six evaluation reports and the live/restoration reports were recovered from
66 indexed evidence chunks with counts and SHA-256 verified. The evidence
artifact is `10351224190`; its ZIP SHA-256 is
`1a6dd41ab89b1aa4109d6a6464d84bb4e053db53f837deffa7ffe42b873c2859`.
The current scorecard points to the new routing and live evidence while retaining
the previous reports. Recounting these reports does not rerun live services.
