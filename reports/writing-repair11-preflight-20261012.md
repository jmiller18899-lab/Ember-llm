# Ember WR11 preparation — October 12, 2026

Status: diagnostic prepared and CPU-validated; no inference or training launched.
Repair2 remains the reference. No deployed model, training loss, checkpoint selector,
frozen grader, promotion rule or workflow was modified.

## Current evidence

The saved October 10 WR8 review has been superseded by two completed experiments:

| Run | Selected step | Reported benchmark | Benchmark copies | Result |
| --- | ---: | ---: | ---: | --- |
| Repair2 | reference | 714/744 | 5/30 | Reference |
| WR9 | 13 | 713/744 | 5/30 | Failed preset gates |
| WR10 | 39 | 713/744 | 5/30 | Failed preset gates |

Sources: [PR 35](https://github.com/jmiller18899-lab/Ember-llm/pull/35),
[PR 37](https://github.com/jmiller18899-lab/Ember-llm/pull/37), the committed WR9/WR10
verdicts, and the existing [WR10 job log](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/38078619738).
The log independently shows 52 optimizer steps, selection of step 39, and the
final-report marker. The private full per-case outputs were not retrieved, so the
exact repeated reminder omission is reported by the prior verdict, not reverified
from generated text in this session.

Both verdicts identify `fresh_writing/fresh-shorten-03` as the sole regression and
report no strict improvements. The three latest experiments do not justify another
loss adjustment without a discriminating diagnostic. WR9 and WR10 development copy
counts must not be compared as a fixed benchmark: their development prompts differ.

## Confirmed preparation gaps

1. All four WR10 shortening training wrappers explicitly ask to retain reminders.
   A neutral request such as `Shorten this note:` is absent from those wrappers.
   Whether this caused the regression remains a hypothesis.
2. `choose_selected` excludes the baseline from its ranking and maximizes
   `v3_pass - copies` after copy/force-count filters. A synthetic reproduction
   selects a candidate with 5 passes and 10 copies over a baseline with 6 passes
   and 11 copies. It also lacks a per-case pass-retention check. This proves a
   selection weakness, not that a better WR10 checkpoint exists.
3. WR10 corrects the old WR8 `candidate/` convention by saving after reloading
   the selected checkpoint. Do not apply the obsolete WR8 labeling diagnosis to
   WR10's candidate weights.

## Prepared diagnostic

`reports/probes/wr11_reminder_preflight.py` supplies 12 fresh sources, each under
neutral and explicit instructions: four remember cases, four do-not-forget cases,
and four ordinary note controls. All 24 prompts are development-only and marked
`training_allowed: false`. The exported prompts omit answer keys.

Probe SHA-256: `667a031d014b40bd68e2496af26974c7abd27cba47dff2d38b3fe0c83d3e3ffd`.
The existing v3 grader is unchanged at SHA-256
`77f1608c1dbaf4cb7ee27a9a57386babf8e00bed1d689f22c42a3379536e6195`.

Exact source/prompt overlap is zero against 2,288 generated WR3-WR10 writing rows
and the already-observed 744 benchmark records. This is not a semantic or all-history
overlap audit. Reserved final holdouts were not loaded. Paired prompts share sources,
so 24 outputs are not 24 independent test examples.

The saved-output comparator reports pass/fail/review, reminder drops, prompt
sensitivity, improvements and every baseline pass lost. An equal net score cannot
hide a regression. It requires complete output IDs, a matching probe hash, pinned
checkpoint identity, deterministic generation settings, and identical per-case
system prompts between models. Adapter hashes are supplied by the producer and
must be verified against downloaded bytes by that producer. A diagnostic result
does not authorize promotion or establish a training recommendation.

## Next executable step

Run inference only on these prompts with Repair2
`Jmiller18899/ember-qwen3.5-4b-repair2` at
`daf938bba5d4e6b650ec9d34a2d3ac56706cf549` and WR10's selected step 39. Resolve and
pin WR10's exact Hub revision and adapter hash from its existing saved evidence
first. Reuse the frozen router and generation settings from WR10. Save both
outputs privately with checkpoint identity, inference configuration and the exact
system prompt used for each row. No benchmark answer should enter training.

The generation runner has not been added or launched: authenticated access and
the exact WR10 artifact identity are still missing. The comparator's input schema:

```json
{
  "probe_sha256": "<hash from exported prompts>",
  "checkpoint": {
    "repo_id": "<actual model repo>",
    "revision": "<40-character immutable Hub commit>",
    "subfolder": "<actual adapter subfolder; . for root>",
    "adapter_sha256": "<64-character hash of the downloaded adapter>"
  },
  "generation": {"do_sample": false, "max_new_tokens": 96, "enable_thinking": false},
  "records": [{"id": "<probe id>", "output": "<actual model output>", "system_prompt": "<actual frozen system prompt>"}]
}
```

```bash
python3 reports/probes/wr11_reminder_preflight.py --export /tmp/wr11-prompts.json
python3 reports/probes/wr11_reminder_preflight.py \
  --baseline /path/to/repair2-outputs.json \
  --candidate /path/to/wr10-outputs.json \
  --output /path/to/comparison.json
```

If explicit prompts preserve reminders while neutral prompts lose them, test
ordinary training wrappers as the next isolated variable. If both fail, inspect
the per-token preference and retention behavior before picking a new loss. In
either case, agree on the WR11 experiment from actual generated outputs first.
The existing minimum 715/744, copy cap, protected-family and shortening-regression
rules remain in force. No automatic promotion or retry is proposed.

## Verification

The eight new tests failed before implementation and passed afterward. The full
focused run passed 54 tests:

```bash
python3 -m unittest tests.test_ember_wr11_preflight \
  tests.test_ember_writing_repair10 tests.test_ember_writing_repair9 \
  tests.test_ember_writing_repair8 tests.test_ember_meaning_preservation_v3
```

Prompt export and its overlap audit succeeded. `git diff --check` passed.
`python3 -m pytest -q` could not start because pytest is not installed; torch is
also absent from this environment. No full-suite, GPU or model-quality pass is claimed.
