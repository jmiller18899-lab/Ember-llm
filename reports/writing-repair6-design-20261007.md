# Ember Writing Repair 6: design

Status: one isolated data-only experiment. Same parent, grader, routing,
and hyperparameters as WR3–WR5. Nothing here authorizes a retry, promotion,
or holdout evaluation.

WR5 is rejected (`reports/writing-repair5-verdict-20261007.md`): benchmark
714 → 711, shortening copies 5 → 8. WR5 dumped long preambles (mean target
0.577 of source) and unlearned Repair2's short-hedge rewrite
("Please make sure everyone gathers..." → "Please gather...") plus the
addressee conversion "her notebook" → "your notebook".

## What the WR5 evidence changes

- **Do not train wholesale frame deletion.** Long "I am writing with a
  polite request..." sources do not transfer to "Please make sure..." or
  "Please remember that...". WR6 targets **short hedge removal** that keeps
  `Please` and the inner action: `Please {hedge} {verb} ...` →
  `Please {verb} ...`.
- **First two tokens of the target must differ from the source.** WR5
  little-cuts shared a `Please check ...` prefix with their sources and
  used one suffix (`if you please`). Prefix-identical targets raise the
  chance of emitting the rest of the source.
- **Teach addressee conversion harder than third-party naming.**
  `drafting-07` failed because WR5 copied `her` from the prompt into a note
  *to* Priya. WR6 puts 48 pronoun-addressee rows on wrappers close to
  "Write a quick note to {name}:" and cuts third-party `her` → `Name's`
  rows so they cannot dominate.
- **Keep the four-token benchmark-frame ban.** Training sources may not
  share a 4-token run with the 30 frozen shortening sources, so
  `make sure everyone`, `remember that the`, `kindly asked to`, and
  `would appreciate it if` stay out of training.

## Data (512 rows)

Shortening, 128 rows:

- 96 **hedge removal**: source is `Please {hedge} ...` where hedge is
  `see that`, `confirm`, `check you`, `be certain`, `verify you`,
  `remember to`, and similar. The reference keeps `Please`, the inner
  verb, owners, counts, timing and modality, and is at most 0.80 of
  source length. First two normalised tokens differ.
- 16 **condition retention**: `only`, `only after`, `unless`, `exactly`,
  `at most` survive. Prefix break still required.
- 16 **compact hedges**: `Kindly note`, `Take note that`, `Do not forget
  to`, not the banned benchmark frames. No shared suffix across templates.
- Contracts: mean reference length in `[0.62, 0.78]`; no reference equal
  to its source; original action verb kept; no added `ensure`.

Messages, 128 rows:

- 48 **addressee pronoun**: prompt body uses `her`/`his`/`their` for an
  object owned by the addressee; the reference uses `your`. Wrappers
  include `Write a quick note to {name}:`.
- 24 **addressee named**, 24 **third-party pronoun**, 16 **third-party
  named**.
- 16 **modal contrasts** (`should` vs `will` on otherwise identical
  sources).
- Greetings still rotate; only meaning is graded.

Retention, 256 rows: the existing training-only arithmetic, extraction,
grounding, clarification, and direct rows.

## Decontamination

Exact prompt/source overlap check against the frozen benchmark and
WR1–WR5 writing sets. Four-token frame check against the 30 benchmark
shortening sources. The two 96-case final holdouts stay unloaded.

## Fixed experiment

Repair2 at `daf938bba5d4e6b650ec9d34a2d3ac56706cf549`, 512 rows, one
epoch, 128 optimizer steps, learning rate 7.5e-7, microbatch 1, gradient
accumulation 4, warmup 8, seed 431, maximum length 384 with no silent
truncation, `l4x1`, 90-minute timeout, checkpoints every 32 steps to a new
private repository. No automatic retry, holdout-based selection, or
deployment.

A fresh 64-case diagnostic is answered before any optimizer update and
again after training. Verbatim-copy count, not-shorter count, and mean
length ratio are saved for the diagnostic and the 30 benchmark shortening
rows.

## Acceptance, fixed before launch

1. Benchmark pass count of at least 715 of 744, with no new failure among
   the 30 shortening rows and no protected family losing a case.
2. Benchmark shortening verbatim copies no more than Repair2's 5.
3. Diagnostic manual review: at least +3 net over the Repair2 answers on
   the same 64 rows, with no new verbatim copy.
4. `drafting-07`-style addressee conversion must not newly fail.
5. Only a candidate meeting 1–4 may be evaluated on the final holdouts.

## Risk

WR3–WR5 each moved the 744-case set by three cases or fewer, always
down. If WR6 also lands at or below 714, stop this writing-SFT series and
use a runtime check that rejects a shortening answer that is not strictly
shorter than its source.
