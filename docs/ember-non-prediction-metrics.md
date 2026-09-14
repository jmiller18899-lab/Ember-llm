# Ember non-prediction metrics

The v9 tool helper passes its latest paired fixture confirmation. The live
service gate remains failed for unqualified Fes. The generated-answer model has
not met its learning gate. Prediction benchmarks do not replace either measure.

| Metric | Latest archived evidence | Meaning |
| --- | --- | --- |
| Combined route, arguments, fixture dispatch | v8 80/100 → v9 100/100, same requests | 20 repaired, zero measured regressions; strict fixture gate passes |
| Correct tool route | 100/100 | Correct route on the latest finite confirmation |
| Exact tool arguments | 80/80 | Both contact forms repaired in fixture evaluation |
| Parser with supplied routes | 60/60 | Parser regression coverage; routing not assessed |
| Live services and guards, September 14 | 48/50 final, 46/50 first attempt | Two recovered timeouts; unqualified Fes requests require country clarification |
| Generated-answer task checks, September 11 | 0/24 → 2/24 | Learning gate fails; lexical checks overstate content quality |
| Answer development loss | 6.1853 → 3.6641 | Better loss alone does not demonstrate useful answers |
| Protected routing parameters during answer trial | Hashes and sampled features unchanged | Measured isolation preserved |

Both full and dequantized INT4 integrations use the same archived text helper.
These results do not show independent LLM reasoning or native INT4 speed.
Direct routing means selecting no tool, not generating a correct answer. Earlier
99/100 and later 98/100 confirmations use different suites and are not a paired
regression. The latest confirmation is now consumed development evidence.

## Reproduce the scorecard

```sh
python -m jobs.ember_metrics_health --check reports/ember-non-prediction-scorecard.json
```

This standard-library command verifies the registered report hashes and recounts
case outcomes, denominators, paired gains/regressions, first-attempt failures,
learning gates, and preserved parameter hashes. It audits recorded scorer flags;
it does not rerun model inference or independently judge generated text.
The scorecard retains original measurement timestamps. Running this command does
not refresh those timestamps or turn archived service results into live health.

CPU validation now runs this check, and report changes trigger validation too.
Passing CI means the evidence is internally consistent, even when capability
gates correctly remain failed. Missing reports, changed hashes, mismatched
aggregates, changed paired requests, or a stale scorecard fail this check.

When new evaluations finish, retain old reports, deliberately update
`config/ember_metrics_sources.json` to the new compatible reports and SHA-256
values, then regenerate with `--output reports/ember-non-prediction-scorecard.json`.
The adapter currently names comparison arms from the source registry; update its
tests for later report schemas. This is an explicit evidence registry, not an
automatic search for the latest file. It does not launch scheduled training or
live calls. All work remains on the experimental tool branch, separate from main.

## Next success gates

1. Preserve the v9 routing PASS and the live-service FAIL separately. Resolve
   country clarification for unqualified Fes and the separately observed clock
   DST-label mismatch before claiming end-to-end success.
2. Improve name, subject, fact, and requested-action preservation in generated
   answers. Inspect raw outputs; the existing lexical pass can invent a recipient.
   Require the existing learning gate before using the untouched confirmation.
3. Recheck live services before release. Production uptime, latency/cost, broad
   answer quality, and native INT4 speed still need their own measurements.

No new capability improvement is claimed by adding this scorecard. The code
makes the existing progress and remaining failures visible and reproducible.
