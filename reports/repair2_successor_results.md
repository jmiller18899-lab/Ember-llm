# Repair2 successor: results

HF job `6ab481106b030d633f68cf7a` (L4, 222 steps, 7m08s training, final train loss 0.61).
Candidate: `Jmiller18899/ember-qwen3.5-4b-repair2`. **Gate: accepted** (all 11 checks pass).
The frozen baseline reproduced exactly in the same run (125/180 held-out strict, 151/200 promotion v2).
The frozen promoted model was not modified.

## Time held-out v1 (strict)
| Slice | Frozen | Repair2 | Change |
|---|---|---|---|
| same_hour_placeholder | 9 | **20** | +11 |
| carry_placeholder | 8 | **20** | +12 |
| on_the_hour | 11 | 12 | +1 |
| noon_crossing | 16 | 15 | **-1** |
| long_duration | 10 | 10 | 0 |
| midnight_crossing | 11 | 11 | 0 |
| same_hour_natural / carry_natural / twelve_to_one | 20 / 20 / 20 | 20 / 20 / 20 | 0 |
| **Overall** | **125** (content 132) | **148** (content 148) | +23 |

## Promotion v2
| Family | Frozen | Repair2 |
|---|---|---|
| time_reasoning | 1/30 | **23/30** |
| clarification | 17/20 | **20/20** |
| context_consistency | 11/15 | 12/15 |
| arithmetic | 27/40 | 28/40 |
| extraction / grounding / drafting / action_honesty | 30 / 25 / 25 / 15 | 30 / 25 / 25 / 15 |
| **Overall** | **151/200** | **178/200** |

## Reading
- **Placeholder copying is fixed:** `H:` answers went from 22 to 0, and both placeholder slices are now 20/20. This fix accounts for all +23 on the held-out eval and most of the promotion v2 time gain.
- **The real skill gaps did not move:** long durations 10→10, midnight 11→11, `:60` rollover 11→12 (8 answers are still `X:60`). Arithmetic went only 27→28. At learning rate 1.5e-6 for 222 steps the update was big enough to change output *format* but not these calculations.
- **Clarification is fixed** (20/20). Context consistency improved slightly (11→12).
- **Watch:** noon_crossing dropped by 1 (16→15). The gate only checks overall held-out scores, so a per-slice drop can pass it.
- No protected family dropped.
