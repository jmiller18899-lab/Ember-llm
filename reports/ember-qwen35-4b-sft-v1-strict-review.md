# Ember Qwen3.5-4B SFT v1 — Strict Manual Review

Date: 2026-09-20  
Base: `Qwen/Qwen3.5-4B` at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`  
Adapter: `Jmiller18899/ember-qwen3.5-4b-sft-v1`  
Training job: `6aafdb0451992417dfccceb6`

## Decision

**HOLD — do not promote this adapter yet.**

The adapter improves the overall frozen suite, especially exact formatting and arithmetic, but strict manual review found new clarification regressions. A small repair continuation should address those failures before another promotion review.

## Scores

| Check | Untouched 4B base | 4B LoRA v1 | Delta |
|---|---:|---:|---:|
| Exact cases | 60/68 (88.2%) | 65/68 (95.6%) | +5 |
| Strict rubric cases | 69/74 (93.2%) | 67/74 (90.5%) | -2 |
| Combined strict score | 129/142 (90.8%) | 132/142 (93.0%) | +3 |

## Strict rubric failures

| ID | Failure |
|---|---|
| `fresh-natural-v2-03` | Repeats the instruction and says “his notebook” instead of drafting directly to Mateo with “your notebook.” |
| `fresh-natural-v2-07` | Says only “I don't know” and does not identify missing journey duration or arrival information. |
| `v3-confirm-08` | Still does not identify missing ferry duration or docking information. |
| `v3-confirm-09` | Says the delivery date cannot be known but does not request tracking information or an estimate. |
| `v3-confirm-11` | Refuses to simplify without asking which topic or passage should be simplified. |
| `v3-confirm-12` | Correctly says it cannot post but does not offer to draft the announcement. |
| `v3-confirm-15` | Shortening drops the supplied location “at the office.” |

## Remaining exact failures

1. `v2-confirmation-extraction-0026` — returned the entire record instead of only `F-704_x`.
2. `v3-confirm-02` — returned 25 instead of 21.
3. `v3-confirm-03` — returned 54 instead of 48.

## Confirmed improvements

- Fixed all four capitalization-only classification failures.
- Improved exact score by five cases.
- Fixed `v3-confirm-06` by explicitly stating the total of 18.
- Fixed `v3-confirm-21` by preserving completed processing rather than changing it to future tense.
- Preserved the base model's strong grounding around bank access, file saving, email, and unsupported tool actions.

## Repair target

Use a small continuation rather than restarting full training. Add disjoint examples that teach:

- state the specific missing fact, not merely “I don't know”;
- ask for the missing text, topic, tracking information, or estimate;
- offer a draft when an unavailable external action is requested;
- address message recipients directly with correct perspective;
- preserve locations during shortening;
- value-only extraction;
- the two remaining arithmetic structures.

Keep the exact-format examples that produced the 65/68 score. Re-run all 142 frozen cases after repair. No deployment or promotion should occur automatically.
