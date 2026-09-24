# Promotion suite v3: baseline of frozen policy v2

Suite: `jobs/ember_promotion_suite_v3.py` (192 cases, seed 20260925, fresh wording). Evaluator: `jobs/ember_promotion_suite_v3_eval.py`.
Run: GitHub Actions 35955649541, suite at commit `f381fee`. Weights: `ember-qwen3.5-4b-repair2@daf938bb` in every configuration. Eval only; no gate applied.

| Configuration | Score |
|---|---|
| Repair2 alone (base system prompt, no rules, no tool) | 147/192 |
| Policy v1 (tool + temporal/drafting rules) | 162/192 |
| **Policy v2 (frozen baseline: v1 + context rule)** | **164/192** |

v2 fixes 20 of Repair2's standalone failures: 16 via the tool and 4 via the rules.

## By family
| Family | Repair2 alone | v1 | **v2** |
|---|---|---|---|
| arith_tool_shapes | 9/24 | 21/24 | **21/24** |
| arith_beyond_tool | 24/24 | 24/24 | **24/24** |
| arith_distractor | 8/16 | 8/16 | **8/16** |
| time_calc | 15/24 | 19/24 | **19/24** |
| time_grounding | 16/16 | 16/16 | **16/16** |
| clarification | 14/16 | 14/16 | **14/16** |
| drafting | 13/20 | 12/20 | **12/20** |
| action_honesty | 12/12 | 12/12 | **12/12** |
| context | 20/24 | 20/24 | **22/24** |
| extraction | 16/16 | 16/16 | **16/16** |

## The 28 v2 failures, by cause
| Cause | Cases | Example |
|---|---|---|
| Irrelevant numbers break the model's arithmetic. The tool correctly declines because there is an extra number. | 8 (arith_distractor) | "A library founded in 2019 had 35 atlases. 25 donated, 7 removed" → 17 (gold 53) |
| Tool vocabulary gap: "members **leave**" is not a removal cue | 3 (arith_tool_shapes) | 103 + 40 − 9 → 63 (gold 134) |
| Tool vocabulary gap: "goes into the oven … **needs** N minutes" | 5 (time_calc) | 5:00 PM + 245 min → 8:05 PM (gold 9:05 PM) |
| Drafting gate misses "Text X that …". The model echoes the instruction or gives a non-answer. | 6 (drafting) | "Text Priya that I have her notebook…" → echoed verbatim |
| Shorten request returns the sentence at the same length | 2 (drafting) | "…gather **at** the loading dock before noon on Saturday." |
| Context rule on "Whose item is it?" answers "X's item", dropping the item | 2 (context) | "Nadia's item, and it can be collected on Monday." |
| Translation request with no text: the request itself is translated | 2 (clarification) | "Could you translate this into Spanish?" → "¿Podrías traducir esto al español?" |

## Rules that make answers worse
Repair2 alone passes these, v2 fails them:
- **Context rule:** `context-00` and `context-18` ("Whose item is it…" records) both become "X's item". The rule gives +4 on context elsewhere, so the family is +2 net.
- **Drafting rule:** `drafting-13` ("Make this shorter …") returns an unshortened sentence. The family is −1 net.

## Suite defects fixed before this baseline
The first run (Actions 35955269095, v2 161/192) exposed three suite errors, fixed in `f381fee`:
- A time_grounding template could produce an arrival before the departure.
- The drafting scorer rejected "the <object>", which is correct from the sender's side.

## What the next candidate must show on v3
Beat v2's 164/192 without lowering any v3 family, while keeping every v2 freeze check (see `PROMOTED_BASELINE.md`).
