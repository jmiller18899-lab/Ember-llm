# Ember v0.0.26 — breaking the copy plateau

> **Result (HF job `6a9f5cbde686246ca69a9a49`): the plateau broke.** Expanded
> continuation 0.7534 → 0.8151, legacy continuation 0.8406 → 0.8696, expanded
> exact copy 0.1889 → 0.2222, clean stop 1.0. Promotion FAIL on the stricter
> gates, but this is the first run since v0.0.16 in which the model measurably
> learned. Full analysis in
> [`reports/ember-v0.0.26-result.json`](../reports/ember-v0.0.26-result.json);
> the follow-up phase is
> [`docs/ember-v0.0.27-sequence-completion.md`](ember-v0.0.27-sequence-completion.md).

## What v0.0.25 actually measured

v0.0.25 completed, protected the baseline, and reported a final result identical
to v0.0.20: exact-copy 44.44%, continuation top-1 84.06%, clean stop 100%,
promotion FAIL. The natural reading is "position coverage was not enough".

That reading is not supported by the run. Four separate problems make it
impossible for v0.0.25 to have tested its own hypothesis, and each is checkable
without a GPU. They are recorded in
[`reports/ember-v0.0.25-plateau-analysis.json`](../reports/ember-v0.0.25-plateau-analysis.json).

### 1. The v0.0.25 checkpoint *is* v0.0.20

`jobs/ember_hf_sft_v025.py` seeds the selector with the baseline score and
writes `best.pt` at `step=-1` from the just-loaded v0.0.20 model:

```python
best_score = score(baseline, -1e9)
best_step = -1
save_checkpoint(best_path, model=model, ..., step=-1, ...)
```

Because `score()`'s last element is `-val_loss`, the seeded baseline carries a
`+1e9` tiebreaker, so a checkpoint that merely ties on every diagnostic can
never replace it. Continuation dropped by one token at step 11 and never
recovered, so nothing ever beat the seed. `best.pt` therefore holds the
unmodified v0.0.20 weights, and the reported "final" numbers are the step-0
baseline numbers read back.

**A case-by-case CPU diff of the v0.0.25 model against v0.0.20 would compare a
model with itself.** It is the one diagnostic guaranteed to return nothing.

### 2. The curriculum cannot emit four of the five failing shapes

`jobs/ember_curriculum_audit_v026.py` compares held-out values against the
structure the curriculum can actually generate. On the v0.0.15 curriculum that
v0.0.20–v0.0.25 all trained on:

```
kind         template   field signature              rows   chars first unsupported character
model_id     NO         [6a]/[3a]-[1d]-[5a]             0   10/18 #10 '-' in ...enai/gpt-6-astra...
url          NO         [5a]://[7a].[4a]/[4Aad]         0   21/25 #21 'a' in ...le.test/a7Q9...
path         NO         /[3a]/[5a]/[4Ad]/[6a].[4a]      0   22/27 #22 '.' in ...4/result.json...
entity       NO         [10Aa] [6Aa]                    0   17/17 -
mixed        NO         [4a]_[4Aad]-[4d]                0    5/14 #5 'Q' in ...acct_Q7m4-583...
verdict: FORMAT_GAP
```

Every training `model_id` is `vendor/ember-xxxx-NNb`; the held-out one is
`openai/gpt-6-astra`. Every training `url` has two path segments; the held-out
one has a single mixed-case segment. Every training `path` ends in
`result-xxxx.json`; the held-out one ends in `result.json`. Every training
`mixed` stem is lower-cased; the held-out one is `Q7m4`.

Cross-tabulated against exact-copy results, four of the five failures sit on a
template with zero supporting rows:

| | exact-copy fails | exact-copy passes |
| --- | --- | --- |
| **template unsupported** | model_id, url, path, mixed | entity |
| **template supported** | digits | short_code, long_code, expression |

v0.0.25's own trigger record shows the dead end from the inside: 320 selected
`model_id` coverage rows yielded **6 distinct target token ids**. The generator
had nothing else to give. v0.0.24's `support_sparse` label was right but
under-read — the support is not sparse, it is structurally absent.

### 3. The learning rate was annealed until training became a no-op

| version | 0.0.15 | 0.0.16 | 0.0.17 | 0.0.18 | 0.0.19 | 0.0.20 | 0.0.21 | 0.0.22 | 0.0.23 | 0.0.25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr | 6e-6 | 2.5e-6 | 1.5e-6 | 8e-7 | 5e-7 | 3.2e-7 | 2.4e-7 | 1.8e-7 | 1.6e-7 | 1.8e-7 |

Each disappointing run was answered by lowering the step size again. At 1.8e-7
an AdamW step moves a weight by at most about the learning rate, so 420 steps
bound total displacement near 7.6e-5 against weights of order 1e-2. Identical
diagnostics at every step from 11 to 419, and a validation-loss drift of 0.022,
are exactly what that regime looks like.

There is a second-order problem. The v0.0.25 curriculum drew 65–90% of every
batch from "failure neighbour" rows built from the same rigid templates, so at a
learning rate that *did* bite, it would have sharpened the model onto `ember-`,
`result-` and two-segment URLs — the priors that make the held-out cases fail.
Raising the learning rate without fixing the curriculum is the one change likely
to make things measurably worse.

### 4. Nine cases cannot resolve what these runs are chasing

`exact_copy_rate` moves in steps of 1/9 = 11.1 points. The promotion gate
(0.5556) is exactly one case above the current 0.4444. The much-discussed
84.06% → 82.61% continuation change is 58/69 → 57/69: one token. A 95% interval
on 4/9 spans roughly 0.19–0.73.

Worse, `score()` selects the saved checkpoint by `exact_copy_rate` **on the same
nine values** that are then reported as the held-out result, so the battery is
the training objective for checkpoint selection, not a held-out measurement.

### 5. The evaluation prompt is off-distribution in both distractor slots

`prompt_for()` hard-codes `Ignore old=K2P8 and fallback=77291`. `K` is in the
alphabet half reserved for the *validation* split, and every training `digits`
value has eight digits, not five. All nine evaluation prompts therefore carry a
slot pattern the model never saw in training.

### What is genuinely a representation problem

Exactly one case: `digits`. Its template has 400 supporting rows, and v0.0.24
independently labelled it `deep_context_miss_with_support` (expected rank 59)
rather than `support_sparse`. The model chose the piece `84` where `4` was
expected — a segmentation decision, not missing structure. That is worth pursuing
as a tokenizer experiment, but it is worth at most 1 of 9 cases and explains none
of the other four failures.

## What v0.0.26 changes

| | v0.0.25 | v0.0.26 |
| --- | --- | --- |
| curriculum | one template per kind | 2–4 templates per kind, covering every held-out structure |
| audit | none | `--assert-parity` gate in CPU CI |
| train/val split | leading character (`2-9A-H` vs `J-Z`) | position in one deduplicated stream |
| battery | 9 cases | 90 cases (the legacy 9 kept verbatim) |
| eval prompt | fixed unreachable distractors | drawn from the training distribution |
| learning rate | 1.8e-7 | 1.2e-6 |
| batch composition | 65–90% failure-neighbour rows | uniform |
| selection | 9-case exact copy | 90-case exact copy, with the legacy 9 as a non-regression guard |

The split change matters on its own: v0.0.15 reserved `J-Z` for validation, so
**no training target ever began with a letter from the second half of the
alphabet**, while three of the nine held-out values do (`Q7M4`, `V9K2-4R7P`,
and `Q7M4` inside the held-out path).

The batch-composition change is deliberate. Hard mining and neighbour
oversampling were added across v0.0.21–v0.0.25 to compensate for structure the
curriculum did not have. With parity restored, one simpler run is worth more
than another run with four interacting mechanisms, because its result is
interpretable either way. The margin loss from v0.0.20 is kept, since v0.0.20 is
the best checkpoint and the margin term targets exactly the quantity that
matters — the gap between the correct copy token and the runner-up.

## Run order

Everything before step 4 is free.

1. **Audit the curriculum (local, seconds, no dependencies).**

   ```bash
   python jobs/ember_curriculum_audit_v026.py \
       --data jobs/ember_sft_data_v026.py \
       --diagnostics jobs/ember_sft_data_v026.py \
       --assert-parity
   ```

   Expected: `verdict: FORMAT_PARITY`. The same command against
   `jobs/ember_sft_data_v015.py` and `jobs/ember_hf_sft_v015.py` must still
   report `FORMAT_GAP` — that regression guard is what keeps the class of bug
   from coming back.

2. **Run the CPU tests.** `python -m pytest -q tests`.

3. **Run the CPU preflight**: Actions → *Ember v0.0.26 Jobs* → `preflight`. It
   loads v0.0.20, measures both batteries, asserts format parity and leakage
   safety, and prints the baseline. **Record the 90-case baseline** — the
   promotion gates for this run are gains over it, and it has never been
   measured.

4. **Only then**, if the preflight baseline looks sane, launch one T4 run:
   Actions → *Ember v0.0.26 Jobs* → `train`. Re-disarm
   `.github/ember-v026.trigger` to `bootstrap` afterwards, as with every prior
   version.

## Reading the result

v0.0.26 is informative whichever way it lands, which is the point.

- **Expanded exact copy rises and the legacy nine hold or improve** — the
  plateau was the format gap. Continue with the same curriculum discipline.
- **Expanded exact copy rises but the legacy nine stall at 4/9** — the four
  template failures are fixed in general but those specific strings need more
  than parity; look at per-case first-error offsets next.
- **Nothing moves at 1.2e-6 with parity restored** — that is the first run in
  the series entitled to conclude the representation is the limit, and the
  digits-style tokenizer experiment becomes the right next spend.

Keep v0.0.20 authoritative until a candidate reports `promotion_eligible: true`.
