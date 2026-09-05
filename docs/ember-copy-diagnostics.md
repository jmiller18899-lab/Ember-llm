# Ember copy-conditioning diagnostics

## Purpose

This CPU-only diagnostic separates four failure modes that were previously mixed together by loss-only and structural tool-call evaluation:

1. tokenizer round-trip / fragmentation problems;
2. weak conditioning on the current prompt value;
3. a correct copy signal that is overwhelmed by a stronger learned response prior at generation time; and
4. continuation / EOS control after a correct copy start.

It is read-only. It downloads private checkpoints, performs inference and teacher-forced scoring on CPU, and writes a JSON report. It never trains, uploads, promotes, or mutates a model repository.

Run it from **Actions → Ember Copy Diagnostics → Run workflow**. The repository `HF_TOKEN` secret is used only to read the private Ember checkpoints. The report is retained as a GitHub Actions artifact for seven days.

## Battery

The suite compares the authoritative `ember-v0.0.7-t4` checkpoint with the accepted experimental `ember-v0.0.9-t4` checkpoint on nine deterministic held-out copy shapes: short code, long code, digits, model ID, URL, filesystem path, natural-language entity, arithmetic expression, and mixed identifier. Each prompt also contains distractors.

For each case it records:

- tokenizer round-trip and fragmentation metrics;
- EOS-aware greedy output and exact target containment;
- first expected copy-token rank against the full vocabulary;
- the log-probability gap between the model's top token and the expected first token; and
- teacher-forced NLL for the correct completion versus a same-shape corrupted completion.

The last two measurements are intentionally separate. A model can know that the prompted value is better than another candidate once the answer is teacher-forced, yet still assign a generic response opening much higher probability at the actual first generation step.

## Verified run — 2026-09-04

Successful CPU-only GitHub Actions run: `33937438412`.

| Metric | v0.0.7 | v0.0.9 |
| --- | ---: | ---: |
| Tokenizer round-trip | 9/9 | 9/9 |
| Exact-copy greedy output | 0/9 | 0/9 |
| Target contained in greedy output | 0/9 | 0/9 |
| Clean EOS stop | 0/9 | 9/9 |
| Correct completion beats corrupted completion | 1/9 | 8/9 |
| Expected first copy token in top 5 | 0/9 | 0/9 |
| Expected first copy token in top 20 | 0/9 | 0/9 |
| Mean expected first-token rank | 1042.6 | 1273.4 |
| Mean top-token log-prob gap over expected token | 10.30 | 6.96 |
| Mean corrupted-minus-correct completion NLL margin | -0.290 | +0.535 |

### Interpretation

The tokenizer is **not** the root cause: all nine values round-trip exactly under both checkpoints.

v0.0.9 clearly learned useful prompt/value information. The correct held-out completion beats a same-shape wrong completion in 8/9 cases, versus only 1/9 for v0.0.7. Its mean NLL margin improves by about +0.824.

However, that copy signal is not activated strongly enough at generation start. Across all nine v0.0.9 cases, the expected first copied token is not even in the top 20. Its average vocabulary rank is about 1,273. Greedy generation therefore chooses a much stronger memorized opening such as `A ...` or `Uploads ...`, then produces a learned generic response pattern. This is not a bug in `argmax`: greedy decoding is accurately exposing the model's dominant response prior.

v0.0.9 also fixes EOS control relative to v0.0.7 (9/9 clean stops versus 0/9), so EOS is no longer the main blocker.

## Root-cause verdict

**`copy_signal_overwhelmed_by_response_prior`**

The current evidence does not support throwing away v0.0.9 or blaming SentencePiece. It also does not support treating this as merely a decoding-parameter problem. The model has acquired semantic discrimination under teacher forcing, but the learned generic first-response distribution is much stronger than the requested literal-copy behavior.

## Recommended next training phase

Keep v0.0.9 as the source checkpoint and make the next corrective phase explicitly attack the first-token response prior before returning to mixed tool SFT:

1. Start with a **direct literal-copy warmup** where the completion begins immediately with the target value and is only `target + EOS`.
2. Use broad target shapes (codes, numbers, URLs, paths, model IDs, entities, expressions), not one synthetic identifier family.
3. Include distractors and require the current `TARGET=` value, so memorized template tokens are consistently wrong.
4. Measure **first expected-token rank/top-k rate** during training, not only completion loss. Promotion from the warmup should require a large improvement from the current 0/9 top-20 baseline.
5. Only after copy-start behavior improves should tool-call JSON and tool-result responses be mixed back in.
6. Keep the semantic-fidelity promotion gate: exact arguments and grounded tool-result facts matter more than structural JSON validity or loss reduction alone.

The existing v0.0.14 literal-copy work points in the right direction, but this diagnostic suggests the curriculum should emphasize direct copy-start behavior first instead of splitting the initial corrective phase evenly across tool calls, direct answers, and tool-result responses.

## Suggested gate before another broad SFT phase

For the same nine diagnostic cases, require at minimum:

- tokenizer round-trip: 9/9;
- expected first token in top 20: at least 7/9;
- expected first token in top 5: at least 5/9;
- correct completion beats corrupted completion: at least 8/9;
- clean EOS stop: 9/9; and
- exact greedy copy: at least 5/9 before reintroducing the full mixed tool curriculum.

These are diagnostic gates, not final ClawAgent promotion criteria. The final model must still pass held-out tool selection, exact tool arguments, tool-result grounding, direct responses, clean stopping, non-regression, and INT4 validation.
