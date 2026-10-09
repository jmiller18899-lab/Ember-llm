# WR8 meaning-aware contrastive experiment — October 9, 2026

## Where this starts
The offline probe in `reports/meaning-preservation-probe-20261009.md` found that meaning-preservation-v2 is usable for review flags, not as a hard runtime gate. It falsely rejected valid timing and consent paraphrases (`prior to` → `before`, `unless ... consent` → `without ... consent`) and missed reminder-drops because leading `please remember` was stripped as framing. WR7 had already failed its numerical gates (714 → 713, copies 5 → 7), including a `should` → `must` rewrite. Repair2 stays the reference.

## Checker fix
`jobs/ember_meaning_preservation_v3.py` is a bounded follow-on, not a replacement of the frozen grader:

- Normalize `prior to` with `before`.
- Normalize the specific consent pair `unless ROLE has given/has consented` with `without ROLE consent`.
- Treat remember/recall/do-not-forget as reminder actions; dropping them is `reminder_dropped`.
- Keep explicit fails for obligation/force, negation, quantities, conditions, and ownership-token drift.
- Keep `review` for unrecognized paraphrases such as `assemble` → `meet`.
- Do not install this checker as a runtime blocker. Frozen 744-case scores remain on v2.

## Training change
WR5–WR7 repeated similar 128-step SFT from Repair2 and increased source copies. WR8 therefore changes method, not just the 16 little-cut rows.

- Parent: Repair2 `daf938bba5d4e6b650ec9d34a2d3ac56706cf549`.
- 96 fresh shortening pairs that v3 accepts, each with a v3-rejected contrast (`should`→`must`, reminder drop, `unless`→`if`, drop `exactly`/`only`, third-party `your`).
- Answer-only SFT on the preferred rewrite plus token unlikelihood on the rejected rewrite.
- 64 recipient/third-party notes and the original 256-row non-writing retention. Encoded budget remains 512 sequences / 128 steps / LR `7.5e-7` / `l4x1`.
- Checkpoint selection uses v3 on development shortening: a later checkpoint is eligible only if copies and force-change fails do not exceed the Repair2 diagnostic. If none qualify, selection stays at checkpoint-0.
- Final 96-case holdouts stay unloaded. No automatic retry or promotion.

## Acceptance, fixed before launch
- Frozen benchmark ≥715/744 on the selected checkpoint.
- ≤5 verbatim shortening copies on the frozen 30.
- No shortening or protected-family regressions.
- No new development copies versus Repair2.
- Compact diagnostic inspects reminder, modal force, consent, counts and ownership individually.
- v3 never becomes a production runtime gate.
