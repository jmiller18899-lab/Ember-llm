# Time held-out v1: frozen promoted Ember baseline

Source: HF job via workflow run 35942160944 (`ember-time-heldout-v1-frozen`), model
`Jmiller18899/ember-qwen3.5-4b-consolidation1@62e5b58`, suite seed 20260924, weights unchanged.

**Overall:** 125/180 strict, 132/180 content.

| Slice | Strict | Main failure |
|---|---|---|
| same_hour_natural | 20/20 | none |
| carry_natural | 20/20 | none |
| twelve_to_one | 20/20 | none |
| noon_crossing | 16/20 | minute errors on 105-120 min durations; one `13:20 PM` |
| midnight_crossing | 11/20 | minutes not normalised (`12:65 PM`), PM not flipped to AM, `13:25 PM` |
| on_the_hour | 11/20 | all 9 failures are `X:60` (e.g. `7:60 AM` for 8:00 AM) |
| long_duration | 10/20 | hour/minute errors when the duration is 75-180 min |
| same_hour_placeholder | 9/20 | `H:` echo (`H:MM PM`, `H:06:55 AM`, `H:13:35 AM`) |
| carry_placeholder | 8/20 | `H:` echo, same as above |

## What this changes
- **Carrying into the next hour is not the weakness.** Asked in natural format, carry cases score 20/20. The promo v2 "hour carry 5/19" result came from the `H:MM` placeholder, not the arithmetic.
- **The placeholder wording is the biggest single cause.** It drops even trivial same-hour cases from 20/20 to 9/20.
- **Four real skill gaps remain:**
  1. Minute overflow: a sum of exactly 60 minutes comes out as `:60` instead of rolling to the next hour.
  2. Durations over an hour, where minutes have to be converted to hours plus minutes.
  3. Midnight: the PM→AM flip and 11→12→1 wrap, including overflow like `12:65`.
  4. 24-hour leakage (`13:xx`) near noon and midnight.
- **Already solid:** noon AM→PM flips and the 12→1 rollover.

Scorer note: `no_time` also counts impossible times like `7:60 AM`, because a minute value of 60 or more isn't parsed as a time.
