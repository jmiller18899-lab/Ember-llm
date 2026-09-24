# Ember 4B Promotion Freeze — 2026-09-24

Status: PROMOTED / FROZEN BASELINE (replaces the 2026-09-23 Consolidation1 freeze, which stays available for rollback)

## Frozen configuration
Repair2 weights + scoped temporal rule + scoped drafting rule + deterministic arithmetic/time tool.

| Component | Pin |
|---|---|
| Base | `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Adapter | `Jmiller18899/ember-qwen3.5-4b-repair2@daf938bba5d4e6b650ec9d34a2d3ac56706cf549` (no weight changes since its acceptance) |
| Scoped rules + system prompt | `jobs/ember_4b_repair2_original_promotion_eval.py` sha256 `48272629…29ffc` (same text as the Consolidation1 freeze) |
| Arithmetic tool | `jobs/ember_arith_tool.py` sha256 `d4dd1d38…1bac1` (commit `aa69888`) |
| Runtime entry point | `jobs/ember_runtime_policy.py`: `respond(prompt, generate)`: tool answer if it fires, else Repair2 with scoped rules |

Decoding: greedy, `max_new_tokens=96`, `enable_thinking=False`, bf16.

## Promotion evidence
GitHub Actions run 35952086798 (`jobs/ember_4b_repair2_arith_tool_eval.py`, eval only). One generation pass, scored with and without the tool.

| Gate | Consolidation1 freeze | Repair2 | **Repair2 + tool** |
|---|---|---|---|
| Original exact (benchmark v1 @`835243e1`, corrected gold) | 72/72 | 71/72 | **72/72** |
| Exact regressions | 0 | — | **0** |
| Temporal | 8/8 | 8/8 | **8/8** |
| Drafting | 7/8 | 7/8 | **7/8** |
| Promotion v2 (200) | 151 | 178 | **197** |
| Time held-out v1 (180, strict) | 125 | 148 | **180** |
| Rule leakage on 72 exact | 0 | 0 | **0** |

Promotion v2 by family (Repair2 → + tool): arithmetic 28→40/40, time_reasoning 23→30/30; extraction 30/30, grounding 25/25, clarification 20/20, drafting 25/25 and action_honesty 15/15 unchanged; context_consistency 12/15 unchanged.

- Tool changed 52 outputs: 52 fixed, **0 worsened**. It never fired on temporal, drafting or other non-arithmetic prompts.
- Repair2's standalone scores reproduced exactly in the same run (71 / 8 / 7 / 178 / 148).
- `passed: true`. `weights_changed: false`.

## What the scores do and do not show
- The tool answers the arithmetic and time families outright, so 40/40, 30/30 and 180/180 measure the tool plus its routing, not the model's arithmetic. All three suites use a few fixed phrasings.
- The tool fires only on three problem shapes (n×k+x, a+b−c, start time + duration) with explicit bare-number/bare-time format requests. Other phrasings fall back to Repair2. There, arithmetic stays at Repair2's level: roughly 28/40 on promotion v2, and the (n+1)×k confusion on multiply-then-add is still present.
- Independent coverage is limited to the hand-written paraphrase and negative tests in `tests/test_ember_arith_tool.py`. Real user traffic has not been measured.
- Remaining promotion-v2 misses: 3 context_consistency rubric cases. Drafting target: 7/8.
- `production_ready` stays false, as for the previous freeze. Deployment is a separate decision.

## Rollback
Previous baseline: `Jmiller18899/ember-qwen3.5-4b-consolidation1@62e5b58b78f823a6cd720a4ff53d0adda1624210`
with runtime policy at branch `freeze/ember-4b-consolidation1-promoted-2026-09-23` (commit `a3af05bee4fe641272ec7a69f4e4108a20148140`).
That freeze is unchanged. Its record says 71/72 exact at benchmark `9b080364`; at the corrected revision `835243e1` it scores 72/72.

## Freeze policy
Do not mutate the pinned adapter revision, the two pinned runtime files, or this record's commit. Future work must use a new branch/checkpoint and pass the same gates (72/72 with zero exact regressions, temporal 8/8, drafting ≥7/8, promotion v2 ≥197 with no family lower, time held-out not lower, zero rule leakage, tool never worsening an output) before replacing this baseline.
