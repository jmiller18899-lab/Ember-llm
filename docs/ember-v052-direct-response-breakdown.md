# Ember v0.0.52 direct-response failure breakdown

Read from the saved evaluation artifact
(`Jmiller18899/ember-v0.0.52-t4`, `evaluations/latest.json`, run
`ember-v052-trust-20260910T160958Z`). No model was run; the evaluator already
recorded both sub-checks for every case.

## Every failure is a spurious tool call. None is an empty generation.

The direct-response check is `passed = ("<|tool|>" not in completion) and
(len(readable) >= 3)` (`jobs/ember_hf_eval.py:139-146`). Of four cases, two
failed, and both failed the same half:

| Case | Prompt | `no_extra_tool_call` | `nonempty` |
| --- | --- | --- | --- |
| `direct_greeting` | Say hello in a friendly sentence. | **false** | true |
| `direct_rewrite` | Rewrite this title clearly: website thing not working. | **false** | true |
| `direct_explain` | Explain what a checkpoint is in one sentence. | true | true |
| `direct_plan` | Give me two short steps for testing a login form. | true | true |

`all_generations_nonempty` is true across the whole battery. The regression is
entirely that the model opens with a tool call where none was asked for.

## It is one memorised call, not context-sensitive over-triggering

Both failures emit the same bytes, at response position zero:

    <|tool|>
    {"arguments":{"location":"Austin"},"name":"weather"}

That is byte-identical to the completion of the `tool_weather` case and to the
INT4 smoke test. A greeting and a title rewrite both produce a weather lookup for
Austin. This is one high-frequency continuation being emitted regardless of
prompt, which is a different fault from a model that judges a tool to be needed.

## It did not fall. It sits below an absolute floor.

The evaluator's own checks record `direct_response_non_regression: true` — the
candidate is at or above its baseline — and `absolute_direct_response_gate:
false`. The threshold is `minimum_direct_response_rate: 0.75`
(`config/ember_v0.0.8_eval.json`), and the baseline is **ember-v0.0.7-t4**, not
the promoted v0.0.31.

So the promotion block is "0.5 is below the 0.75 floor", not "direct responses
got worse". Any claim that this training run damaged direct-response behaviour is
unsupported by this artifact: the comparison it ran says otherwise, and it was
run against an old baseline.

## The measurement is four cases

`direct_response_rate` 0.5 is 2 of 4. With a 0.75 floor the gate permits exactly
one failure, so a single case flipping moves the metric 25 points and changes the
promotion decision. Nothing here separates 50% from 75% at any useful confidence.

## Two evaluator limitations worth recording

**No truncation at `<|endoftext|>`.** Every completion in the battery runs past
its first `<|endoftext|>` into hallucinated `<|user|>`/`<|assistant|>` turns, and
`no_extra_tool_call` scans that entire runaway text. A case that answers
correctly and then rambles into a tool call would be scored as a direct-response
failure. It does not change this result — both failures emit their tool call at
position zero — but `tool_web_search` does contain two further `<|tool|>` calls
inside its runaway text, so the exposure is real.

**The gate does not measure answer quality.** It requires three readable
characters and no tool marker. Both passing answers are template filler:
"A checkpoint is a practical mechanism used to make a system easier to operate
reliably" and "1. Prepare a controlled example and validate a broken link". The
same filler clause appears in nearly every completion in the battery. A 20.1%
validation-loss improvement coexisting with this is consistent: neither number
tracks response quality.

## What follows

- The failure is one measurable quantity — the rate at which a tool marker opens
  a no-tool prompt — not a broad behavioural regression.
- Fixing it is a rehearsal question (no-tool prompts in the preservation set),
  not a structural one.
- The four-case battery cannot support a promotion decision on direct response.
  Widening it is cheaper than any further training run.
- Truncating generation at the first `<|endoftext|>` before scoring would remove
  a scoring fragility that will eventually produce a wrong verdict.

No model was run, nothing was promoted, and no checkpoint was modified.
