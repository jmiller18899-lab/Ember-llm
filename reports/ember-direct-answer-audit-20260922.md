# Ember direct-answer audit — September 22, 2026

## Scope and status

Evaluation instrumentation and saved-evidence audit are complete. The live cutoff replay was submitted but, at the last inspection for this report, Hugging Face still reported SCHEDULING / waiting for requested hardware. No live cutoff conclusion or model-quality improvement is claimed. No Ember training, weight update, promotion, deployment, or merge was performed.

## Verified saved evidence

Model: `Jmiller18899/ember-qwen3.5-4b-sft-v1`.
Audited model-repository revision: `23431a2ac366f2e6c2c69657ae572fcb12f68cdc`.
Training source: `1a4988596b46f07ebb321ce85f18804e12a767de`.

The CPU audit downloaded `evaluation/base_4b.json`, `evaluation/final.json`, and `manifest.json` at that pinned revision. It paired cases by ID, required identical prompts and grading definitions, and recalculated every exact score from the saved output.

- 142 paired cases: 68 exact checks and 74 rubric cases.
- Base: 60/68 exact passes. Adapter: 65/68 exact passes.
- Five exact improvements, zero exact regressions, three remaining exact failures.
- The three remaining exact failures were also present in the base: `v2-confirmation-extraction-0026`, `v3-confirm-02`, and `v3-confirm-03`.
- Historical outputs lack complete raw-token and stopping evidence. A saved short answer alone does not prove whether generation reached its output cap.

Source verification: GitHub Actions run `35755243788`, job `106839240131`, `SAVED_AUDIT` log entry. Earlier independent Hugging Face verification job `6aafdf2b52d0dbd7f1d742af` reported the same 60/68 and 65/68 counts and no training rerun.

The earlier conversational claims of 46/50 versus 43/50, two epochs, and an unsupported upload `retry` argument were incorrect. The original training log records 116 steps and one epoch, followed by a Trackio bucket/Xet upload error after training. This audit does not modify the original training job or its recorded exit status.

## Answer-quality triage

Review of saved paired outputs found a clear clarification regression in `v3-confirm-11`: the base requested the missing topic/text, while the adapter refused simplification. Other targets include identifying the specific missing information instead of a bare unknown answer, preserving location when shortening text, and writing from the correct sender/recipient perspective. These qualitative notes do not silently replace historical rubric scores.

Two positive controls retain already-correct arithmetic and arrival-time answers. Reviewed examples are consumed regression diagnostics, not independent unseen test evidence. Detailed private-model outputs were retained in the conversation's local triage artifact rather than copied into this public report.

## Implemented diagnostic

`jobs/ember_direct_answer_audit.py` preserves raw token IDs, decoded text with special tokens, the exact scored text, output length, EOS IDs, and stop reason. An EOS at the token limit is classified as a normal stop, not truncation. Existing strict exact scoring is unchanged.

The replay compares the pinned base and adapter on 14 selected cases, at output limits 96 and 192: 56 planned generations. It checks reproduction of each saved 96-token answer before interpreting the longer-limit comparison. It freezes model parameters and has no training path. Reports are saved after every generation and uploaded as separate diagnostic JSON with hash readback. `--upload-report PATH` supports retrying only an evidence upload without loading the model.

## Verification

Tested code commit: `41e7b675115a1072ffed875bae32f94a6143c391`.
GitHub Actions CPU run: `35756073955`; job: `106842051538`.

- Repository-wide `python -m pytest -q`: **1152 passed in 62.37s**.
- Separate audit-contract check: **13 passed in 0.02s** (these 13 are included in the 1152 above).
- Local TDD: 13 failing tests against the insufficient implementation, then all 13 passing; compilation also passed.
- Initial broad-test attempts were blocked by missing PyTorch, then NumPy/JSON Schema dependencies. The final CPU workflow installs them and completed successfully; no old tests were weakened or skipped to obtain the passing result.

## Pending live evaluation

Hugging Face job: `6ab2aea651992417dfcd3b7f`.
Submitted: September 22, 2026, 16:36:54 UTC.
Hardware: `l4x1`. Job timeout: 900 seconds. No duplicate job was launched.
Last observed stage for this report: **SCHEDULING**, waiting for hardware; no generation results available.

Continue by reading this existing job's status and logs. Do not rerun training or launch another diagnostic merely because the job remains queued. If the 96-token replay differs from historical output, disclose that reproduction issue. If it reproduces and ends with EOS well before the cap, unchanged answers at 192 support an answer-quality issue for that selected case. A length stop followed by a corrected longer response instead supports a generation-budget issue. Neither outcome by itself establishes broad generalization.

Next model-quality gate: improve clarification and fact preservation without losing the five exact gains, then test on fresh prompts excluded from training. No further paid training was started as part of this milestone.
