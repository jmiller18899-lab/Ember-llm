# Ember 4B Promotion Freeze — 2026-09-23

Status: PROMOTED / FROZEN BASELINE

## Frozen model
- Hugging Face repo: Jmiller18899/ember-qwen3.5-4b-consolidation1
- Exact model revision: 62e5b58b78f823a6cd720a4ff53d0adda1624210
- Base: Qwen/Qwen3.5-4B
- Weights changed by runtime fixes: no

## Frozen runtime policy
- Git commit: cf16a26e194d9151ff45ed395ecfc31678e6b8d7
- Freeze branch: freeze/ember-4b-consolidation1-promoted-2026-09-23
- Includes scoped temporal-grounding and drafting-transformation rules.

## Promotion evidence
Final Hugging Face job: 6ab4556152d0dbd7f1d87ec8
- Fresh exact: 71/72
- Exact regressions: 0
- Temporal: 8/8
- Drafting: 7/8
- Temporal rule leakage on 72 exact cases: 0
- Drafting rule leakage on 72 exact cases: 0
- promotion_freeze_pass: true

## Freeze policy
Do not mutate this model revision or this runtime-policy commit. Future training and policy experiments must use a new branch/checkpoint and must pass the same regression gates before replacing this baseline.
