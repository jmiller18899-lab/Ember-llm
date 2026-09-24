# Ember meaning-preservation grading repair — 2026-09-24

Grader version: `meaning-preservation-v2`.
Parent: `2844bde72b7b0af0a597ce980ad8dce4309ccdaa`, the writing-policy experiment.
Scope: evaluation grading only; no model training, runtime-policy change, or deployment.

## Root cause and correction

The previous shortening checks accepted any shorter answer containing selected facts. They did not test whether a request became a requirement. The completed job `Jmiller18899/6ab577df6b030d633f68f2dd` therefore awarded three false passes to rewrites using "must".

The evaluator now calls the same `grade_case` hook for every shortening row, including main-suite, probe, original-drafting, and fresh-writing rows. It keeps the original checker as `legacy_fresh_score`; pinned historical suite modules are not changed.

The additional grader distinguishes requests, requirements, recommendations, permission, optionality, prohibition, absence of obligation, and uncertainty. It checks protected qualifiers, negation, conditions, timing relations, and numeric values. An order-sensitive content comparison prevents selected keywords alone from proving equivalence. Numeric signs, currency, and percentage symbols are retained; "likely" and "unlikely" are not normalized into the same meaning.

Results are `pass`, `fail`, or `review`, with reasons. `review` is not counted as a pass. This is a conservative English shortening checker, not a general semantic-equivalence model. Unsupported but potentially valid paraphrases and complex modal scope require manual review.

## Score integrity and replay

Historical and corrected scores are recorded separately. Historical expected counts are used only to reproduce the old baseline. Candidate improvement and regression checks compare baseline and candidate graded under the same new rubric, not against stale 185/192 and 13/20 thresholds.

Saved `writing-results.json` now includes per-case source rows, outputs, legacy grades, corrected grades, and reasons. The caller must retain this job output file. CPU-only replay is available with:

```sh
python jobs/ember_drafting_repair_candidates_eval.py --rescore reports/meaning-preservation-regressions.json
```

Replay explicitly reports that its coverage is limited to the supplied records. It does not load a model, fetch the frozen evaluator, assume a full benchmark, or promote a candidate. The 24 writing cases already inspected are labelled previously observed development evidence, not new holdout data.

## Verified results

- Existing evaluator source was reconstructed locally and matched its GitHub blob SHA exactly: `3fa34978da9c2b0b95cf761566e0a29a1417129f`.
- Original writing-policy tests: 9 passed before the patch.
- Initial regression run: 50 tests, 29 expected failures with the old grader.
- Additional edge-case run exposed four false passes involving probability polarity, numeric sign, currency, and percent; these were fixed.
- Final scoped verification: 64 tests passed; pytest additionally reported 23 subtests passed. Both unittest discovery and pytest passed on Python 3.13.5.
- Python 3.11 syntax parsing passed; an actual Python 3.11 runtime was not available locally.
- All three recorded false passes now fail with `force_changed:request->requirement`. Their replay evidence is in `reports/meaning-preservation-regressions.json`.
- AST comparison confirms original policy-rule text, candidate configurations, historical thresholds, dataset generation, and routing functions are unchanged.
- Compile checks and `git diff --check` passed.

Verification commands:

```sh
python -m unittest discover -s tests -v
python -m pytest -q
python -m compileall -q jobs tests
```

Verification scope: the retrieved evaluator, its 9 existing tests, and 55 new grading/integrity tests. The full repository suite and GPU inference were not rerun. No new 744-case score is claimed; a corrected full baseline needs all original outputs or a fresh evaluation. Review was self-review, not an independent reviewer.

No frozen adapter, frozen-policy module, model prompt, model routing, or deployment was changed. No paid job was launched for this repair.
