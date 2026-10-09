# Meaning-preservation probe — 2026-10-09

The offline probe supports using the checker for experimental review flags, not as a hard runtime acceptance gate. No model was promoted, runtime behavior changed, or additional training or model inference launched.

## Method

Reused the existing meaning-preservation-v2 evaluator in jobs/ember_drafting_repair_candidates_eval.py. An experimental wrapper additionally flags reminder_dropped when an explicit reminder in the source disappears from the rewrite. The existing evaluator is unchanged; the wrapper is an archived development probe, not deployed product code.

The semantic checker was isolated using a legacy scorer that always returns true. Consequently these results are checker routing decisions, not full benchmark scores or measured semantic accuracy.

Inputs comprised:
- 20 assistant-authored development examples: 10 intended valid shortenings and 10 intentionally invalid rewrites, organized in pairs.
- 80 saved outputs for 40 paired shortening inputs: Repair2 before training and WR7 after training. These cover 32 older shortening inputs and 8 compact additions; the 32 recipient-message inputs were excluded.

The challenge examples were adjusted during the local pilot and are not a preregistered, independent, or human-adjudicated test set.

## Development challenge

| Expected class | Checker | Pass | Fail | Review |
| --- | --- | ---: | ---: | ---: |
| Valid (10) | Existing | 4 | 2 | 4 |
| Valid (10) | Experimental reminder rule | 4 | 2 | 4 |
| Invalid (10) | Existing | 1 | 6 | 3 |
| Invalid (10) | Experimental reminder rule | 0 | 7 | 3 |

The reminder rule removes one false acceptance. With the experimental rule, all 10 invalid examples are withheld from automatic acceptance, but only 7 receive explicit failure decisions; 3 remain unresolved reviews. Six of 10 valid examples are also withheld, including two explicit false rejections.

Examples of valid equivalents rejected by the checker include “prior to” → “before” and a consent condition expressed as “unless” → “without.” Other valid paraphrases remain in review because the checker relies heavily on token structure.

## Replay of saved shortening outputs

| Saved model outputs | Checker | Pass | Fail | Review |
| --- | --- | ---: | ---: | ---: |
| Repair2 (40) | Existing | 0 | 22 | 18 |
| Repair2 (40) | Experimental reminder rule | 0 | 24 | 16 |
| WR7 (40) | Existing | 1 | 17 | 22 |
| WR7 (40) | Experimental reminder rule | 1 | 19 | 20 |

These are not human correctness judgments. In particular, 39 withheld WR7 outputs do not imply 39 incorrect outputs. The replay includes valid rewrites that this checker rejects or sends to review.

The reminder rule adds explicit failures for wr5-dev-short-diagnostic-07-00 and wr5-dev-short-diagnostic-07-02 at both endpoints. The only automatically accepted WR7 output is wr5-dev-short-diagnostic-06-00.

The compact cases illustrate the tradeoff: an unchanged copy fails the shortening requirement and “should” → “must” triggers a force-change failure, while valid consent and “each and every” → “each” rewrites are also rejected.

## Decision and next review step

Keep Repair2 active and do not deploy this checker as an automatic blocker. Existing frozen scores remain unchanged: Repair2 714/744 and WR7 713/744. This probe does not establish general reliability.

Before considering another similar training run, review a separately adjudicated set of valid/invalid contrast pairs. A bounded follow-on could normalize specific equivalent timing and consent constructions, while retaining explicit checks for obligation, ownership, negation, quantities, and reminder actions. Compare false acceptance and false rejection rates, and preserve an abstain/review outcome for uncertain cases. That follow-on has not been implemented.

## Reproducibility

- Probe: reports/probes/meaning_probe_20261009.py
- Workflow: .github/workflows/ember-meaning-probe.yml
- Successful CPU-only workflow run: https://github.com/jmiller18899-lab/Ember-llm/actions/runs/37972857641
- Workflow commit: 0b4bfe869ea1c81bbff3c3a612ff8ba3548b56ef
- Saved WR7 evidence: Jmiller18899/ember-qwen3.5-4b-writing-repair7-20261009 at c5fd030e5a23a73a489a3630a50de40bc1a854f4
- Experiment branch: codex/ember-next-training-20261009

The workflow completed successfully as an experiment. Its success does not mean the checker met a deployment quality threshold.
