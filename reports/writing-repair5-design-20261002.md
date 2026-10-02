# Ember Writing Repair 5: design

Status: design only. No data code, preflight, or GPU job exists yet, and
nothing here authorizes a launch.

WR4 was rejected (`reports/writing-repair4-verdict-20261002.md`): benchmark
714 → 713, fresh diagnostic 45 → 44. Its shortening regressions were the model
returning the source unchanged, not padding: verbatim copies went 1 → 4 of 32
diagnostic rows and 5 → 8 of 30 benchmark shortening rows.

## What the WR4 evidence changes

- **Strictly shorter references are already required.** The WR4 data builder
  rejects any reference that is not shorter. The problem is that they were only
  slightly shorter: training templates average 0.887 of source length,
  diagnostic references 0.870. Repair2 already cut more (0.789); WR4 moved it
  toward light trims (0.826) and sometimes none.
- **Near-unchanged references should be capped, not added.** WR4 already
  taught one-phrase edits ("no later than" → "by", "until Monday has arrived" →
  "before Monday"). Training on more of them is the likely cause of the copying.
  WR5 keeps a small capped slice so the model still handles sources with little
  to cut, but every reference must change the source.
- **References must keep the source's action verb.** Three WR4 templates
  produce "Please ensure", and the model began turning "Please check that" into
  "Please ensure", changing a request to verify into a request to cause.
- **Recipient "your" and negative scope are already 16/16.** Recipient rows
  can be reduced in favour of the remaining errors: third-party owners for
  his/their (1/2 and 1/4 named), the persistent "should" → "will", and "only"
  conditions (0/4).
- **The fixed "Hi {name}," greeting spent the update on style.** 10 of 23
  changed diagnostic answers only added "Hi ".

## Data (512 rows, same shape as WR4)

Shortening, 128 rows:

- 96 **frame removal**: the source wraps the instruction in a removable frame
  (for example "Please take care that", "It would help if", "Bear in mind
  that", "It is possible that", "subsequently"). The reference removes the
  whole frame and keeps the request marker, the original action verb,
  quantifiers, conditions, modality, ownership and timing. Each reference is
  at most 0.80 of source length.
- 16 **condition retention**: "only", "only if", "unless", "exactly", "all"
  and "at most" survive the cut. Structures differ from the diagnostic's
  "You are permitted to …, but only after …".
- 16 **little to cut**: an already compact source with one dispensable word
  or phrase. The reference removes it, is at most 0.92 of source length, and
  is never identical to the source. Capped at 12.5% of shortening rows.
- Contracts: mean reference length at most 0.75 of source; no reference
  equal to its source after normalisation; the source's action verb appears
  in the reference; no "ensure" unless the source contains it.

Messages, 128 rows:

- 48 third-party pronoun with ownership context: 16 each for her, his and
  their. The reference names the owner.
- 32 third-party named, 32 recipient (named and pronoun), each with paired
  targets as in WR4.
- 16 modal contrasts where "should" and "will" share otherwise identical
  sources, so the difference is the only signal.
- Greetings rotate between "Hi {name},", "{name}," and "Hello {name},", and
  only meaning is graded.

Retention, 256 rows: the same 256 training-only examples (arithmetic 64,
extraction 64, grounding 48, clarification 48, direct answers 32).

## Decontamination

Keep WR4's exact prompt/source overlap check against the frozen benchmark and
the WR1–WR4 training and development sets. Add a frame check: reject any
training source that shares a run of four or more tokens with any of the 30
benchmark shortening sources. Benchmark frames such as "make sure everyone",
"kindly asked" and "would appreciate it if" therefore stay out of training,
and the benchmark remains a test of generalisation to unseen frames. The two
96-case final holdouts stay unloaded.

## Fixed experiment

Same parent, base, grader, routing and settings as WR4, so the data is the
only change: Repair2 at `daf938bba5d4e6b650ec9d34a2d3ac56706cf549`, 512 rows,
one epoch, 128 optimizer steps, learning rate 7.5e-7, microbatch 1, gradient
accumulation 4, warmup 8, seed 431, maximum length 384 with no silent
truncation, `l4x1`, 90-minute timeout, checkpoints every 32 steps to a new
private repository. No automatic retry, holdout-based selection or deployment.

A fresh 64-case diagnostic (32 shortening, 32 messages, new structures) is
generated and answered before any optimizer update and again after training.
The evaluation job saves verbatim-copy count, not-shorter count and mean
length ratio for both the diagnostic and the 30 benchmark shortening rows, so
the copying failure is visible without manual inspection.

## Acceptance, fixed before launch

1. Benchmark pass count of at least 715 of 744, with no new failure among the
   30 shortening rows and no protected family losing a case.
2. Benchmark shortening verbatim copies no more than Repair2's 5.
3. Diagnostic manual review: at least +3 net over the Repair2 answers on the
   same 64 rows, with no new verbatim copy.
4. Only a candidate meeting 1–3 may be evaluated on the final holdouts.

## Risk and fallback

WR3 (714 → 712) and WR4 (714 → 713) each moved the benchmark by two cases or
fewer at this learning rate and step count. If WR5 also lands within ±2, further writing
SFT at this scale is unlikely to pay off. A cheaper fallback is a runtime check
in the existing policy layer: reject a shortening answer that is not strictly
shorter than its source, and ask again once.
