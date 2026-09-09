# Ember v0.0.30 — finishing the run, and closing the seam that broke v0.0.29

## The defect was mine, and it was only in the reporter

v0.0.29 raised `KeyError: 'gate_distance'`. `v029_progress()` never attached
`band_flow` or `gate_distance` to its return value, while the final reporter read
both out of it. Both helpers existed. Both print statements existed. The line
joining them did not.

How it got there: the edit that would have attached them shared a patch script
with a second edit whose assertion failed, and that assertion aborted the script
*before* it wrote the file — discarding the first edit with it. The follow-up
patch fixed only the assertion that had errored and never re-applied what had
been rolled back.

Why the tests missed it: `test_v029_band_flow_...` and `test_v029_gate_distance_...`
called the helpers directly, and `test_v029_transform_targets_...` asserted the
print statements appear in the runtime source. Each half was tested. **The seam
between them was not.**

### Nothing was lost

In the transformed runtime:

```
1003  base.upload(... best.pt ...)
1004  base.upload(... latest.pt ...)
1005  base.upload(... evaluation/v0.0.29-report.json ...)
1015  base.upload(... run-state.json ...)
1016  print(f"EMBER_HF_V029_PROMOTION=...")
1020  print(json.dumps(report["progress"]["gate_distance"], ...))   ← the crash
```

Every artifact uploaded before the failure, and `run-state.json` carries
`status: evaluation_complete`, so v0.0.30 resolves the v0.0.29 checkpoint through
the normal path. **No rerun.**

## Where v0.0.29 left the model

| | v0.0.28 | v0.0.29 | gate |
| --- | ---: | ---: | ---: |
| legacy exact copy | 5/9 | **6/9** | ≥ 5/9 ✅ |
| legacy continuation | 62/69 | **65/69 = 0.9420** | ≥ 0.90 ✅ |
| expanded exact copy | 28/90 | 32/90 | 36/90 |
| expanded continuation | 632/730 | 642/730 | 657/730 |
| expanded first-token top-1 | 85/90 | 87/90 | — |
| full-span margin health | −0.5865 | **−0.4361** | — |

Both legacy gates pass for the first time. What remains is four cases and
fifteen token decisions.

### The two remaining gates are consistent, not competing

Mean continuation is 730/90 = 8.111 tokens. Under independent errors, 0.40 exact
copy needs per-token **0.89318 → 652/730**, five tokens short of the **657/730**
the continuation gate asks for. One improvement satisfies both. v0.0.28 delivered
+33 tokens; v0.0.29 delivered +10.

## What v0.0.30 changes: the source, and the reporter

Nothing else. Three reasons the objective should stay put:

1. **`k = 2` still matches the data.** 730 × (1 − 0.8795) = 88 wrong continuation
   decisions over 58 failing rows is 1.52, plus three first-token failures →
   **1.57 wrong decisions per failing row**, against the 1.66 that set `k = 2`.
2. **More steps would not help.** v0.0.29's best checkpoint was step **479 of
   600**, so the last 120 steps produced nothing better and the cosine schedule
   was near its floor. The gain has to come from a fresh schedule on the new
   checkpoint — which is exactly this run.
3. **Exact copy is still sitting on the independence curve.** 0.8795^8.111 =
   0.3528 against a measured 0.3556. No structural failure has re-appeared.

## The repair, and the test that keeps it repaired

`v030_progress()` attaches both fields. That is the one-line fix. Two things
matter more than the fix:

**The trailing prints now read defensively.** Every artifact uploads before them,
so a reporting gap must never again turn a completed run into a failed job. A
test asserts each defensive read sits after the last `base.upload(`.

**A contract test derives its requirements from the transform text.** It scans
the runtime source for `report["progress"]["…"]` accesses, calls `v030_progress`
for real, and fails on any key the reporter reads that progress does not supply.
A second test does the static equivalent for `baseline[…]` / `final[…]` against
`v030_diagnostic`'s return keys.

The first test in that file runs the same check against **v0.0.29** and asserts
it reports exactly `{gate_distance, band_flow}` missing. Without that, the other
tests would prove nothing — a contract check that cannot fail on a known-broken
input is not a check.

## Protection, and the 2/3 bar

Protection moves up to the v0.0.29 result, with one case of slack on the 90-case
counts and none on the token-level rates. Note that legacy exact copy at 6/9 is
the same trap the tolerance was built for in v0.0.29:

```
6/9 = 0.6666666666666666   stored bar 0.6666666667
raw >=            → False
with tolerance    → True
```

A test asserts both halves of that, so the guard cannot regress to a raw compare.

## Reading the result

- **Both expanded gates clear** — promotion PASS, and v0.0.30 is a candidate to
  replace v0.0.20 as authoritative.
- **Continuation clears 657/730 but exact copy lags 36/90** — errors have become
  correlated within rows again; look at `final_by_kind` and `final_by_length`
  before touching the objective.
- **Gain drops below +10 tokens** — the objective is saturating. The band
  transition matrix, which now actually reaches the report, says whether the
  reachable band is still refilling from below; if it is not, the next spend
  belongs on capacity or representation rather than shaping.

## Run order

1. `audit` — free.
2. `python -m pytest -q tests`.
3. `preflight` — CPU. Reads the v0.0.29 baseline, prints `EMBER_V030_GATE_DISTANCE`,
   `sequences_within_reach` and the worst-margin histogram.
4. `train` — one approved T4. Re-disarm `.github/ember-v030.trigger` afterwards.
