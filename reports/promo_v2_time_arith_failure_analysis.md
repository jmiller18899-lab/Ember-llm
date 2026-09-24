# Promotion v2: time and arithmetic failure analysis (frozen promoted Ember)

Source: HF job `6ab46f596b030d633f68cd4b`, model `Jmiller18899/ember-qwen3.5-4b-consolidation1@62e5b58`.
Reproduce: `python jobs/analyze_promo_v2_time_arith_failures.py` (CPU only).

## Time reasoning (1/30)

| Cluster | Count | Notes |
|---|---|---|
| `H:` prefix + correct time (`H:8:30 AM`) | 14 | Correct result; format fails only |
| `H:00 AM` | 5 | 3 are right for times on the hour, 2 wrong |
| `H:<single number>` (`H:10 PM`) | 6 | Hour or minutes only |
| Literal `H:MM AM` | 4 | Template copied, nothing computed |
| Clean correct answer | 1 | time-25, the only pass |

- **Format:** all 29 failures start with the literal `H:` from `Answer H:MM AM`. The model treats the placeholder as text to copy.
- **Hour carry is the real skill gap:** the time is right in 10/11 same-hour cases but only 5/19 when the minutes carry into the next hour. 14 of the 15 degenerate outputs are carry cases.
- **AM/PM:** kept correctly in 30/30, but **no case crosses noon or midnight**, so 12-hour rollover and AM/PM flips are **not tested yet**.
- Lenient score (leading `H:` removed): 15/30.

## Arithmetic (27/40; 12 add-subtract failures + 1 multiply)

| Cluster | Count | Cases |
|---|---|---|
| Wrong operation or dropped term (output = a-b+c, a-b, a-c, a+b or a) | 6 | 05, 07, 08, 10, 13, 17 |
| Tens-digit slip (off by a multiple of 10) | 4 | 03, 04, 09, 12 |
| Other | 2 | 06, 14 |
| Multiply-then-add (`4x11+9` gave 55, expected 53) | 1 | mult-06 |

- Carries and borrows **do not** explain the failures (arith-07 and 13 have neither).
- The pattern tracks **operand size**: 11 of the 12 failures have a two-digit added amount (b of 11 or more, other than 10). With b of 10 or less, 7/8 pass.
- 10 of 13 errors come out too low, which fits the model subtracting the added amount or dropping a tens digit.
- The sample is small (13 failures), so treat these as hypotheses for the curriculum, not conclusions.
