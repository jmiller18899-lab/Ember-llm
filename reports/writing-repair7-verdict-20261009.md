# WR7 completion review — October 9, 2026

## Verdict
Training and evaluation completed, but WR7 fails the preset promotion gates. Retain Repair2 as the reference; no promotion, deployment, additional training, or final-holdout evaluation was performed by this review.

## Evidence
- HF job: 6ac92478fee2c9007017ad59, stage COMPLETED.
- Private output: Jmiller18899/ember-qwen3.5-4b-writing-repair7-20261009.
- Evidence revision: c5fd030e5a23a73a489a3630a50de40bc1a854f4.
- Read-only collection: https://github.com/jmiller18899-lab/Ember-llm/actions/runs/37971556683 (success).
- All 128 optimizer steps completed; training loss 0.2845345064997673.
- No active jobs returned by the latest-50 job inspection.

## Frozen runtime benchmark
Repair2 714/744 -> WR7 713/744 (minimum required 715).
Three regressions: promo_v2/draft-short-05, suite_v3/drafting-07, fresh_writing/fresh-shorten-03.
Two improvements: probes/probe-context-00 and probes/probe-context-04.
Other suite pass counts: exact72 72->72, temporal8 8->8, drafting8 4->4, promo_v2 200->199, heldout 180->180, ctx_holdout 24->24, suite_v3 181->180, probes 28->30, fresh_writing 17->16.
These are frozen runtime scores with routing, not pure model-only measurements.

Benchmark shortening copies: 5->7 out of 30 (maximum allowed 5).
Benchmark answers not shorter: 7->9 out of 30.
Development shortening copies: 2->3 out of 40.
Development answers not shorter: 2->4 out of 40.
The development aggregate includes the original 32 shortening cases plus 8 new compact-edit cases, so it must not be directly compared to WR5's 32-case total.

## Targeted semantic inspection
The collector compared all 72 before/after development IDs and prompts, returning changed outputs and all 8 new compact cases. This review is a targeted assistant review, not a complete independently graded 72-case semantic score.
The new courier reminder remains a verbatim copy. The ensemble example changes 'should' to 'must', which is a meaning error despite shortening. The apprentices answer improves by preserving 'All' and 'exactly nine'. The field-guide answer restores the omitted qualifier 'field'. A previously shortened permission example becomes a full source copy, and another example changes 'only if' into 'if', weakening the necessary-condition statement.
The numerical score and copying gates already fail; no complete semantic score is needed to establish non-promotion. The stored final report still correctly says manual_review_completed=false.

## Next step
Do not simply repeat this SFT recipe or promote WR6 based on its aggregate gain: WR6 also failed its copying gates. Investigate whether a meaning-aware candidate selection/checking layer can reject unchanged shortening answers while also protecting modal force, conditions, ownership and counts. A length-only check would miss 'should' -> 'must'. Any later training method or data change should be evaluated as a separate experiment using the retained Repair2 reference.
