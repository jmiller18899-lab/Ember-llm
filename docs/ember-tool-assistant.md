# Frozen Ember tool-assistant candidate

This combines the strongest observed direct/tool router and text-family classifier
with extraction helper v2. The runtime loads saved weights without fitting,
cross-validation, network access, or access to evaluation answers.

**Current result: FAIL — 82/100 full precision and 85/100 INT4.**
See the [measured results and repair priorities](ember-tool-assistant-results.md).
No passing prototype is available.

This is a candidate until its combined confirmation passes. The workflow packages
a prototype only after all 100 new requests pass in both precisions, including
all 80 exact argument checks and fixture dispatches. A failed run preserves
its evidence and exits unsuccessfully. A consumed confirmation cannot be reused
as fresh evidence for a modified candidate.

## Pinned components

- Router source: `13bcf0eb645c5b1f1fd5423982f6b89ab8c923f2`.
- Extraction source: `10683d1f62fb9c2eff04f9f522c4f8ffb3907414`.
- Full step-9 checkpoint SHA256: `700e4259b2e724c8fa384e23ba1a6e67636c4f110863e887499f06951622c9ad`.
- INT4 checkpoint SHA256: `d77c466fb8a9e602796aed3dffa4070b55cae8b492057b8f59a672ab6a1c1c78`.
- The model repository revision, tokenizer/model package hashes, dependency
  versions, fitted-head selection, and every bundle file hash are recorded in
  `manifest.json`. Optimizer state is omitted from the inference bundle.

The original 384 training requests are stored with their original order and
binary/family memberships. Feature selection uses training-only cross-validation.
The copied implementations reproduce the original family selection: hybrid,
ridge 0.01, 256/256 CV, vocabulary 2026. Source hashes and a source lock are saved.

## Run a passing bundle

Extract the prototype archive into an empty directory, then run:

```sh
python -m pip install -r requirements.txt
python -m tool_assistant 'Calculate 46 squared.' --bundle .
```

The CLI returns a proposed tool call. Python callers can use
`Runtime(bundle).run(request, handlers)` to execute through an explicit mapping
of tool names to implementations. Missing handlers and tool exceptions return
structured statuses. Direct requests return `direct_answer_unavailable` because
language quality remains a separate blocker.

The four argument shapes are `weather(location)`, `calculator(expression)`,
`web_search(query)`, and `get_time(timezone)`. The existing time resolver returns
place names as well as IANA names; an external adapter must resolve a place name
unambiguously before using a clock API. No external service credentials or live
weather/search providers are bundled.

## What the test establishes

The confirmation includes natural paraphrases, places with punctuation,
broader current-information requests, decimals, negative numbers, and grouped
arithmetic. Location/time/search arguments require exact dictionary equality;
calculator expressions must have the correct arithmetic value under a bounded
AST evaluator. Correctly predicting a tool name alone does not count as success.
Direct cases measure routing only. Fixture dispatch tests the call boundary,
not live service availability, tool-result language, or direct-answer quality.

The INT4 checkpoint is dequantized to float32 for CPU inference, as in the
original experiments. This does not claim packed INT4 speed or memory benefits.
The runtime remains single-turn, with Ember's original context limit and a fixed
system prompt. Passing this finite suite does not establish production readiness.

## Reproduce the build and evaluation

From the repository checkout, with authorized Hugging Face read access:

```sh
python -m tool_assistant.build --out frozen-candidate
python -m tool_assistant.evaluate --bundle frozen-candidate \
  --cases tool_assistant/data/confirmation-100.json \
  --report evidence/combined-confirmation.json
```

Reproduction of this same frozen candidate is allowed; it is not another
independent confirmation. Preserve the first result, including failures.
The build refuses to overwrite an existing bundle; evaluation refuses to
overwrite an existing report. Ember language weights, production pointers,
and ClawAgent remain unchanged.
