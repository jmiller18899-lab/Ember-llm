# Ember 4B Promotion Freeze v2 — 2026-09-24

Status: PROMOTED / FROZEN BASELINE. It replaces v1 (freeze branch `freeze/ember-4b-repair2-arith-tool-promoted-2026-09-24`, commit `b19ffe2`). v1 and the 2026-09-23 Consolidation1 freeze both stay available for rollback.

## Frozen configuration
v1 plus one scoped context rule: Repair2 weights + scoped temporal rule + scoped drafting rule + **scoped context rule** + deterministic arithmetic/time tool.

| Component | Pin |
|---|---|
| Base | `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Adapter | `Jmiller18899/ember-qwen3.5-4b-repair2@daf938bba5d4e6b650ec9d34a2d3ac56706cf549` (unchanged) |
| Scoped temporal/drafting rules + system prompt | `jobs/ember_4b_repair2_original_promotion_eval.py` sha256 `48272629…29ffc` (unchanged) |
| Arithmetic tool | `jobs/ember_arith_tool.py` sha256 `d4dd1d38…1bac1` (unchanged) |
| **Context rule** | `jobs/ember_context_rule.py` sha256 `239a1d29…cb70` (commit `9640efe`) |
| Runtime entry point | `jobs/ember_runtime_policy_v2.py`: `respond(prompt, generate)`. The tool answers if it fires; otherwise Repair2 answers with scoped rules, including the context rule when its gate fires |

Decoding: greedy, `max_new_tokens=96`, `enable_thinking=False`, bf16.

## Why the context rule
Diagnostic (Actions run 35952932842): the three v1 context_consistency misses were all the `item=folder` records. The model answered "Theo owns **the item** …", writing the field name instead of its value. The rule tells the model to answer with the record's values, never its field names. It fires only on a `field=value` record followed by a who/when/what question. It never fires when the prompt asks to return or give a single field, so exact extraction prompts are unaffected.

## Promotion evidence
GitHub Actions run 35953355347 (`jobs/ember_4b_policy_v2_context_eval.py`, eval only). v1 and v2 were compared in the same run on the same weights.

| Gate | v1 (reproduced) | **v2** |
|---|---|---|
| Original exact (benchmark v1 @`835243e1`) | 72/72 | **72/72** |
| Exact regressions | — | **0** |
| Temporal | 8/8 | **8/8** |
| Drafting | 7/8 | **7/8** |
| Promotion v2 (200) | 197 | **200** |
| context_consistency | 12/15 | **15/15** |
| Time held-out v1 (180, strict) | 180 | **180** |
| Fresh record questions (24, `jobs/ember_context_holdout.py`) | 10/24 | **20/24** |
| Rule leakage (temporal/drafting on 72 exact; context on all 120 benchmark cases) | 0 | **0** |

- No output changed outside the context-rule prompts. v1's scores reproduced exactly in the same run. `passed: true`, `weights_changed: false`.
- The rule fires on exactly the 15 promotion-v2 context prompts. It fired on nothing in the 120-case benchmark, time held-out or temporal/drafting targets, and the arithmetic tool never fires on record prompts.

## What the scores do and do not show
- All v1 limits still apply: the arithmetic/time scores are computed by the tool on a few fixed phrasings, and other phrasings fall back to Repair2.
- **Still failing: 4 of 24 fresh record questions**, all in the "Lost property … Whose item is it and what day can it be picked up?" form. The model answers "Eli's item can be picked up on Thursday." It still uses the word from the question ("item") instead of the value when the question itself names the field. The rule helps (v1 got all 6 of that form wrong, v2 gets 2 right) but does not fix it.
- The promotion-v2 context family uses one template, so 15/15 is narrower evidence than the 20/24 on fresh wording.
- `production_ready` stays false. Deployment is a separate decision.

## Rollback
- **v1:** branch `freeze/ember-4b-repair2-arith-tool-promoted-2026-09-24` (commit `b19ffe2bfaf6b95c7e590ad3db4b67e0e7bd3840`), entry point `jobs/ember_runtime_policy.py`. Same weights; drop the context rule.
- **Consolidation1:** branch `freeze/ember-4b-consolidation1-promoted-2026-09-23` (commit `a3af05b`), adapter `ember-qwen3.5-4b-consolidation1@62e5b58`.

## Freeze policy
Do not mutate the pinned adapter revision, the three pinned runtime files, or this record's commit. A replacement must pass: 72/72 with zero exact regressions, temporal 8/8, drafting ≥7/8, promotion v2 200/200 with no family lower, time held-out not lower, fresh record questions ≥20/24, zero rule leakage, and the tool never worsening an output.

---

## Previous baseline (v1): Ember 4B Promotion Freeze — 2026-09-24

Status: PROMOTED / FROZEN BASELINE (replaces the 2026-09-23 Consolidation1 freeze, which stays available for rollback)

### Frozen configuration
Repair2 weights + scoped temporal rule + scoped drafting rule + deterministic arithmetic/time tool.

| Component | Pin |
|---|---|
| Base | `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Adapter | `Jmiller18899/ember-qwen3.5-4b-repair2@daf938bba5d4e6b650ec9d34a2d3ac56706cf549` (no weight changes since its acceptance) |
| Scoped rules + system prompt | `jobs/ember_4b_repair2_original_promotion_eval.py` sha256 `48272629…29ffc` (same text as the Consolidation1 freeze) |
| Arithmetic tool | `jobs/ember_arith_tool.py` sha256 `d4dd1d38…1bac1` (commit `aa69888`) |
| Runtime entry point | `jobs/ember_runtime_policy.py`: `respond(prompt, generate)`: tool answer if it fires, else Repair2 with scoped rules |

Decoding: greedy, `max_new_tokens=96`, `enable_thinking=False`, bf16.

### Promotion evidence
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

### What the scores do and do not show
- The tool answers the arithmetic and time families outright, so 40/40, 30/30 and 180/180 measure the tool plus its routing, not the model's arithmetic. All three suites use a few fixed phrasings.
- The tool fires only on three problem shapes (n×k+x, a+b−c, start time + duration) with explicit bare-number/bare-time format requests. Other phrasings fall back to Repair2. There, arithmetic stays at Repair2's level: roughly 28/40 on promotion v2, and the (n+1)×k confusion on multiply-then-add is still present.
- Independent coverage is limited to the hand-written paraphrase and negative tests in `tests/test_ember_arith_tool.py`. Real user traffic has not been measured.
- Remaining promotion-v2 misses: 3 context_consistency rubric cases. Drafting target: 7/8.
- `production_ready` stays false, as for the previous freeze. Deployment is a separate decision.

### Rollback
Previous baseline: `Jmiller18899/ember-qwen3.5-4b-consolidation1@62e5b58b78f823a6cd720a4ff53d0adda1624210`
with runtime policy at branch `freeze/ember-4b-consolidation1-promoted-2026-09-23` (commit `a3af05bee4fe641272ec7a69f4e4108a20148140`).
That freeze is unchanged. Its record says 71/72 exact at benchmark `9b080364`; at the corrected revision `835243e1` it scores 72/72.

### Freeze policy
Do not mutate the pinned adapter revision, the two pinned runtime files, or this record's commit. Future work must use a new branch/checkpoint and pass the same gates (72/72 with zero exact regressions, temporal 8/8, drafting ≥7/8, promotion v2 ≥197 with no family lower, time held-out not lower, zero rule leakage, tool never worsening an output) before replacing this baseline.
