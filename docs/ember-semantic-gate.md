# Ember semantic-v1 CPU gate

v0.0.31 step 479 passed the copy promotion and the historical ClawAgent
promotion contract. Those are legitimate results. That contract accepts a
correct tool name with any nonempty arguments and mostly checks that direct
answers are nonempty. It does not establish semantic correctness or production
readiness.

This separate gate evaluates the same checkpoint and its existing INT4 export.
It never trains, runs generated tools, changes deployment configuration,
updates `run-state.json`, or overwrites `evaluations/latest.json`.

## Frozen contract

`config/ember_semantic_v1.json` contains 36 cases: 12 tool calls, 12 direct
answers, and 12 responses to supplied tool results. The 12 historical prompts
are unchanged and labeled `legacy_recheck`. The 24 `new_challenge` prompts were
authored after the candidate's training, with new values, negation, missing
information, tool failures, empty results, signs, units, and arithmetic results.
They have not been supplied to a training job. This is not an assertion that
every phrase has been exhaustively checked against all earlier corpora.

Every case in every group must pass on both full weights and reconstructed
INT4 weights. A single failure blocks this bounded semantic gate. The report
also separates semantic-content failures from output-format/stopping failures.
Neither a lower validation loss nor the legacy PASS can override those checks.

- Tool calls need one complete native `name`/`arguments` JSON envelope. Duplicate
  keys, non-finite JSON constants, prefixes, suffixes, extra keys, wrong fields,
  wrong types, and wrong values fail. Full values are compared against explicit
  alternatives, not substrings or a blob of argument values.
- Responses must match an explicitly accepted complete answer after only case
  and whitespace normalization. Numbers, units, signs, punctuation, and negation
  remain significant. Keyword soup, appended noise, invented owners, false
  success claims, and incorrect facts fail.
- A real model-emitted `<|endoftext|>` is required. Missing EOS, extra role
  markers, repeated EOS, and any text after EOS fail when scoring supplied raw
  completions. The scorer never salvages a passing prefix.

These are conservative finite-answer fixture oracles, not a general semantic
judge. Valid unlisted paraphrases can fail. Review raw outputs before deciding
what to train; never silently widen this version's answer sets after seeing a
candidate's failures. If a rule needs correction, version it and rerun both
variants. A PASS clears this test only; `production_authorized` stays false.

## Decoding and evidence

The old `model.generate` always decodes the full token budget, even after EOS.
Fresh evaluation uses deterministic argmax on CPU and stops only when the
model emits its actual dedicated EOS token. This is decoding behavior, not a
post-processing cleanup: it does not insert EOS, trim output at a convenient
JSON brace, or discard an unfinished answer. The full generated token IDs,
text, and stop reason are saved for every case. Budget exhaustion fails.

The recorded legacy raw generations are also rescored in full, without
trimming, under `legacy_raw_recheck`. That audit is reported separately from
the fresh EOS-aware result so decoder overruns are not mistaken for text that
a correctly stopping runtime would actually serve.

The runner resolves one immutable Hugging Face repository revision before
downloading anything and uses it for all model/evaluation files. It checks the
model version, run ID, step, legacy evaluation path and PASS, then verifies
every INT4 tensor against deterministic quantization of the full checkpoint.
The report records the repository revision, both file hashes, evaluator/spec
hashes, package hash, runtime, and code commit. Reconstructed INT4 inference
tests quantized-weight quality; it does not benchmark a native INT4 kernel.

## Run and persistence

The `Ember Semantic CPU Gate` workflow uses a GitHub-hosted CPU runner and the
existing `H_F2` repository secret. It does not submit any Hugging Face GPU or
CPU Jobs. Run `evaluate` once, then return the separate semantic trigger to
`bootstrap`. The v0.0.31 promotion/training triggers remain unchanged.

For an existing CPU environment with PyTorch, SentencePiece, huggingface-hub,
and jsonschema installed:

```bash
python -m pytest -q tests/test_ember_semantic_gate.py tests/test_v008_readiness.py
python jobs/ember_semantic_gate.py --output-dir semantic-results --publish
```

`HF_TOKEN` must provide read access to the private candidate and, when using
`--publish`, write access for reports. No token is printed. Reports are written
locally before upload, including FAIL results, and published atomically under
`evaluations/semantic-v1/` with a separate `latest.json`. GitHub Actions also
preserves the JSON as an artifact when the model fails the gate. Evaluation
returns a nonzero exit code for FAIL; infrastructure failures do not produce
a false PASS.
