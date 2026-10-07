# Ember Writing Repair 5: verdict

Status: **rejected**. Repair2 (`daf938bba5d4e6b650ec9d34a2d3ac56706cf549`)
stays the reference. No promotion, deployment, or holdout evaluation.

Evidence: GitHub run
[37142187149](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/37142187149),
Hugging Face job `6ac12906404719ba3762ebef`, committed under
`reports/evidence/writing-repair5-20261003/`. Copying replay:
[37149501386](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/37149501386),
committed under `reports/evidence/wr5-copying-20261003/`.

## Automatic gates

WR5 completed 128 optimizer steps (loss 0.277) from untouched Repair2.
The frozen 744-case benchmark moved **714 → 711** (26 → 29 fail, 4 review
unchanged). That misses the pre-registered floor of 715 and is a protected
writing regression.

| Gate | Repair2 | WR5 | Rule |
| --- | ---: | ---: | --- |
| Benchmark pass / 744 | 714 | 711 | ≥ 715 |
| Benchmark shortening copies / 30 | 5 | 8 | ≤ 5 |
| Benchmark not-shorter / 30 | 7 | 10 | no new shortening fails |
| Diagnostic copies / 32 | 1 | 1 | no new copies |
| Diagnostic mean length ratio | 0.698 | 0.722 | — |

Fresh development was saved before and after the optimizer. Manual semantic
review of those 64 pairs is not required for rejection: the automatic
benchmark already failed.

## What changed

Four cases flipped pass → fail; one probe improved.

- `promo_v2/draft-short-00` and `draft-short-05`: Repair2 answered
  "Please gather at the loading dock before 7/8." WR5 returned the source
  "Please make sure everyone gathers...".
- `fresh_writing/fresh-shorten-03`: Repair2 dropped "that"
  ("Please remember the spare key must remain..."). WR5 copied the source.
  Grader: `not_shorter`.
- `suite_v3/drafting-07`: Repair2 converted addressee "her notebook" →
  "your notebook" in a note to Priya. WR5 copied "her notebook".
- `probes/probe-context-00`: the one gain, a fuller owner/item/time answer.

Protected arithmetic, extraction, time, grounding, clarification, and the
180+24 time/context holdouts used in the 744 were unchanged. The two reserved
96-case final holdouts were not loaded.

## Copying replay

Inference-only checkpoints of the same candidate:

| Step | Benchmark copies | Not shorter | Shortening passed / 30 |
| ---: | ---: | ---: | ---: |
| 0 | 5 | 7 | 15 |
| 32 | 6 | 8 | 14 |
| 64 | 6 | 8 | 14 |
| 96 | 6 | 8 | 14 |
| 128 | 8 | 10 | 12 |

Teacher-forced likelihood still preferred the verbatim source over a valid
short rewrite on the loading-dock items at every step. On
`fresh-shorten-03` that preference widened. The last 32 steps added the
second pair of copies.

WR5 training templates averaged **0.577** of source length (frame dump).
Repair2's own shortening answers averaged **0.797** on the 30-row slice.
The uniform "if you please" little-cut slice shared one suffix. Aggressive
preamble removal did not transfer to unseen short hedges ("make sure
everyone", "remember that").

## Decision

WR3 (714 → 712), WR4 (714 → 713, copies 5 → 8), and WR5 (714 → 711,
copies 5 → 8) all used the same parent, learning rate, and 128 steps.
Further SFT that only dumps long frames or lightly trims the source is
not authorized. The next experiment has to teach the Repair2 hedge-removal
pattern that WR5 unlearned, and the addressee `her/his/their` → `your`
conversion, without repeating WR5's frames or the 30 benchmark sources.
