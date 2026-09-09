# Ember v0.0.42 cross-seed subtype calibration

v0.0.41 reached 82/90 canonical envelopes but failed protection because `short_code/len5` regressed from 3/4 to 2/4. Two v0.0.41 subtype changes were nevertheless real non-regressing gains: `url/one_mixed` reached 4/4 and `mixed/upper` reached 3/3.

v0.0.42 assembled those gains with the v0.0.40 safe stack, restored the short-code len5 fallback, and calibrated only six remaining failure-bearing subtypes. Candidate selection used two independent synthetic folds and required a challenger to beat the mandatory baseline on both folds before held-out replacement.

## Measured result

GitHub Actions run `34285257677` evaluated commit `4f0ac856c0382a7f4d9b0b482bce32df33db2ad7` after the v0.0.42 runner was decoupled from the malformed post-run v0.0.41 source file. The inherited/focused suite passed **184 tests**. The CPU measurement then correctly returned **FAIL** only because the unchanged 95% global threshold was not reached.

- Exact v0.0.8 reference control: **4/4 envelope and 4/4 correct tool**.
- 90-case JSON-valid envelope: **84/90 (93.3%)**.
- 90-case correct tool name: **84/90 (93.3%)**.
- Correct tool conditional on valid envelope: **84/84 (100%)**.
- Slot exact: **0/84 (0%)**.
- Right envelope + right tool + wrong value: **84/90 (93.3%)**.
- Clean stop: **90/90 (100%)**.
- Kind no-regression gate: **PASS**.
- Structural-subtype no-regression gate: **PASS**.

The two-fold selector froze:

| Subtype | Selection |
| --- | --- |
| `short_code/len4` | baseline |
| `short_code/len5` | baseline |
| `long_code/4x4` | baseline |
| `long_code/3x5` | baseline |
| `path/plain_leaf` | baseline |
| `url/two_segment` | `tool_query` |

Only `url/two_segment` cleared the two-fold replacement rule. That change improved the held-out subtype from 2/3 to **3/3** and raised the full URL cohort to **10/10**.

Held-out kind results are now:

| Kind | v0.0.40 | v0.0.41 raw | v0.0.42 | Protected floor |
| --- | ---: | ---: | ---: | ---: |
| `short_code` | 8/10 | 7/10 | **8/10** | 8/10 |
| `long_code` | 7/10 | 7/10 | **7/10** | 7/10 |
| `digits` | 10/10 | 10/10 | **10/10** | — |
| `model_id` | 10/10 | 10/10 | **10/10** | — |
| `url` | 7/10 | 9/10 | **10/10** | 9/10 |
| `path` | 9/10 | 9/10 | **9/10** | 9/10 |
| `entity` | 10/10 | 10/10 | **10/10** | — |
| `expression` | 10/10 | 10/10 | **10/10** | — |
| `mixed` | 9/10 | 10/10 | **10/10** | 10/10 |

The protected subtype floors also all held: short len4 5/6, short len5 3/4, long 4x4 5/6, long 3x5 2/4, URL one-mixed 4/4, URL two-segment 3/3, path plain-leaf 3/4, and mixed upper 3/3.

## Conclusion

v0.0.42 is the strongest fully regression-safe envelope baseline so far: **84/90 (93.3%)**, up from v0.0.40's 80/90 and only **two cases short** of the required 86/90 threshold.

The remaining six envelope failures are localized to only four areas:

- `short_code`: 2 failures (`len4` one, `len5` one)
- `long_code`: 3 failures (`4x4` one, `3x5` two)
- `path/plain_leaf`: 1 failure

URL and mixed are now fully solved at 10/10, and all four historically strong kinds remain 10/10. A next CPU-only experiment should freeze URL, mixed, and every passing subtype, then target only these six remaining failures with stronger cross-seed evidence or a prompt feature orthogonal to the already-tested wording.

No optimizer, GPU job, training, promotion, deployment, or production integration was run. This remains failing baseline evidence and does not authorize placement learning.
