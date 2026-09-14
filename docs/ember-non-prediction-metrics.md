# Ember non-prediction metrics

The v9 tool helper passes its latest paired fixture confirmation. The new live
service contract passes with an explicit country reply for Fes. The generated-answer model has
not met its learning gate. Prediction benchmarks do not replace either measure.

| Metric | Latest archived evidence | Meaning |
| --- | --- | --- |
| Combined route, arguments, fixture dispatch | v8 80/100 → v9 100/100, same requests | 20 repaired, zero measured regressions; strict fixture gate passes |
| Correct tool route | 100/100 | Correct route on the latest finite confirmation |
| Exact tool arguments | 80/80 | Both contact forms repaired in fixture evaluation |
| Parser with supplied routes | 60/60 | Parser regression coverage; routing not assessed |
| Live services and guards, September 14, v10 assisted flow | 52/52 final, 49/52 first attempt | Three recovered timeouts; includes explicit country replies, not automatic Fes resolution |
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

1. Preserve v9 routing PASS and v10 assisted-service PASS, alongside the original
   unqualified 48/50 result. The new flow asks for a country; it does not infer one.
2. Improve name, subject, fact, and requested-action preservation in generated
   answers. Inspect raw outputs; the existing lexical pass can invent a recipient.
   Require the existing learning gate before using the untouched confirmation.
3. Recheck live services before release. Production uptime, latency/cost, broad
   answer quality, and native INT4 speed still need their own measurements.

No new capability improvement is claimed by adding this scorecard. The code
makes the existing progress and remaining failures visible and reproducible.

## Grounded direct-answer v2 update

The scorecard now retains the old `direct_answers` canary and adds a separate
`direct_grounded_v2` section. New development reference accuracy moved from 0/48
to 12/48, and the newly consumed confirmation scored 6/24. Every grounded pass was
a status label; named writing, rewrites and fact preservation scored zero.
The development learning gate passed but strict confirmation failed. Historical
lexical scores and losses on different suites must not be combined into a trend.
See `docs/ember-direct-grounded-v2.md` for raw failure examples and provenance.
