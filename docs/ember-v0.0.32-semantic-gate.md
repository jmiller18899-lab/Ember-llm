# Ember v0.0.32 — a gate that can actually fail

## v0.0.31 is promoted

| | v0.0.30 | **v0.0.31** | gate |
| --- | ---: | ---: | ---: |
| expanded exact copy | 34/90 | **43/90** | 36/90 ✅ |
| expanded continuation | 651/730 | **662/730** | 657/730 ✅ |
| legacy exact copy | 6/9 | 6/9 | ≥ 5/9 ✅ |
| legacy continuation | 65/69 | 65/69 | ≥ 0.90 ✅ |
| first-token top-1 | 88/90 | 89/90 | — |
| full-span margin health | −0.3204 | **−0.2449** | — |

Plus INT4 export, `promotion_eligible: true`, and a legacy tool-routing
evaluation at 1.0 across every rate with validation loss 4.458 → 3.581 (−19.7%).
Nine within-reach cases crossed into correct and none fell out. That is the
milestone the project has been chasing since v0.0.7.

The promotion record itself carries the caveat:

> formal legacy evaluator passes, but raw completions remain noisy after the
> first scored segment; require a stricter semantic quality gate before
> production ClawAgent integration

## Why a perfect score is compatible with bad completions

`score_case` in `jobs/ember_hf_eval.py` accepts a tool call when four things
hold: a `<|tool|>` marker exists, the first balanced JSON object after it
parses, the name matches, and `arguments` is a **non-empty dict or string**.

That is the whole rubric. Nothing inspects argument keys. Nothing checks whether
the values name anything from the prompt. Nothing looks past the closing brace.
A response case passes on "no `<|tool|>` anywhere" plus **three visible
characters**. And nothing requires generation to stop rather than run out of
budget.

So this scores a clean pass today:

```
<|tool|>{"name": "weather", "arguments": {"x": 1}}
<|user|> what about tomorrow <|assistant|> the weather in the weather in
```

Right marker, valid JSON, right tool name, non-empty arguments. It calls the
wrong location, invents a user turn, and degenerates into a loop.

**Grounding is the sharpest omission.** Getting `"location": "Detroit"` out of a
prompt that says Detroit is precisely the capability v0.0.15 through v0.0.31
were built for — 43/90 exact copies of held-out strings is that capability —
and the promotion evaluator never tests it.

## What v0.0.32 grades

| | legacy | v0.0.32 |
| --- | --- | --- |
| tool marker | present anywhere | present, and **at most one** |
| JSON | first object parses | same |
| tool name | matches | same |
| arguments | non-empty dict or string | **an object, with required keys** |
| argument values | *not checked* | **must carry the prompt's entity** |
| after the answer | *not checked* | **no trailing text, no invented turns** |
| termination | *not checked* | **EOS inside the budget** |
| repetition | *not checked* | **4-gram repeat ratio ≤ 0.35** |
| response length | ≥ 3 characters | **≥ 12 characters and ≥ 3 words** |

Every gate is set at 1.0. This is a pre-deployment bar, not a trend line.

## The gate is proven offline before anything is paid for

A gate that cannot fail on a known-bad input proves nothing — the same principle
that made the v0.0.29 contract test start by failing against v0.0.29.

`tests/test_ember_v032_semantic_gate.py` drives **eleven completions the legacy
rubric accepts**, asserts the legacy rubric really does accept each one, asserts
the strict rubric rejects it, and asserts the *exact set* of checks it fails on:
ungrounded arguments, the wrong city, opaque string arguments, an invented user
turn, trailing prose, no EOS, two tool calls, a three-character reply, a
degenerate loop, an invented conversation, and never stopping. Two more assert
that a genuinely good completion passes **both** rubrics, so the gate is not
simply rejecting everything.

The scoring functions import nothing beyond the standard library, so the whole
rubric runs in CI with no token, no model and no GPU. The workflow runs that
suite before it will submit anything.

The evaluator is read-only. A test asserts its source contains no `upload_file`,
`save_checkpoint`, `loss.backward`, `optimizer` or `cuda`.

## What this does not do

It does not weaken the legacy evaluator or any copy gate, and it does not
re-open the promotion v0.0.31 already earned. It is an **additional** bar for
deployment. Whether v0.0.31 replaces v0.0.20 as authoritative on the strength of
the promotion it already has is your call and does not depend on this gate.

## Reading the result

- **The strict gate passes too** → the noise note was conservative, and Ember is
  ready for ClawAgent integration on this battery.
- **Grounding fails while structure passes** → the copy capability is not
  reaching the tool-argument slot. That is a training target, and the curriculum
  work that got exact copy to 43/90 is the tool for it.
- **Structure passes and everything after the answer fails** → the model answers
  and then keeps talking. That is an EOS/stop problem, and clean-stop is already
  1.0 on the copy battery, so it would be specific to the agent format.
- **`legacy_passes_but_strict_fails` is empty** → the two rubrics agree on this
  checkpoint and the caveat on the promotion record can be closed.

## Run order

1. `rubric` — free, no token. Runs the offline suite and proves the gate can fail.
2. `evaluate` — `cpu-upgrade`. Generates the twelve cases from the promoted
   checkpoint and scores both rubrics side by side.
3. Read `EMBER_V032_SEMANTIC_GATE` and
   `EMBER_V032_LEGACY_PASSES_BUT_STRICT_FAILS`.

No mode in this workflow submits GPU work.
