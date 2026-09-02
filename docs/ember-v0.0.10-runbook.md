# Ember v0.0.10 semantic-fidelity runbook

Ember v0.0.10 is a cost-gated completion-only supervised fine-tune. It starts
from the accepted v0.0.9 checkpoint, which learned tool routing but not
argument values, tool-result facts, or a clean stop. Merging these readiness
changes does **not** launch a Hugging Face Job. Every evaluation and training
job must be selected manually from the **Ember Hugging Face Jobs** workflow.

v0.0.9 remains the experimental development checkpoint. It is not ClawAgent's
default model.

## Why the previous gate was not enough

The v0.0.8 evaluation spec still used by `eval-v009` scores a tool call as
valid when it has the `<|tool|>` marker, parseable JSON, the expected tool
name, and any non-empty arguments. Direct and tool-result answers pass when
they are nonempty and do not make another tool call.

That let v0.0.9 report 100% on all four tool-call cases while calling
`weather(Austin)` for Detroit, `calculator(330)` for `347 × 28`,
`web_search` for SQLite instead of Python, and `get_time` with
`America/Anchorage` for Tokyo. After a correct calculator result of `9716`,
the model could still answer with an unrelated product and pass.

`config/ember_v0.0.10_eval.json` keeps the same 12 held-out prompts and
changes the meaning of a pass:

- tool calls must include the requested argument values;
- tool-result answers must mention the supplied facts;
- generation must include `<|endoftext|>` and must not continue afterward.

A `weather(Austin)` completion for the Detroit prompt is now an automatic
failure.

## Fixed experiment contract

- Source package: `ember-v0.0.7-hf-ready.zip`, verified by SHA-256 before use.
- Source checkpoint: the evaluated v0.0.9 `best.pt` in
  `Jmiller18899/ember-v0.0.9-t4`, which must report `technical_pass: true` and
  `promotion_eligible: true` under the original structural spec.
- Architecture and tokenizer: unchanged.
- Objective: completion-only loss over 1,152 train and 144 validation
  examples that copy requested entities, quote tool-result facts, and stop at
  `<|endoftext|>`.
- Official promotion prompts: excluded from training and validation.
- Schedule: 400 optimizer steps, 20 warmup steps, batch size 8, gradient
  accumulation 2, learning rate `2e-5`, and context length 256.
- Durable recovery point: `resume/latest.pt` is uploaded every 200 steps with
  a checksum and matching manifest. A rerun may resume only a v0.0.10
  checkpoint produced by the same pinned configuration.

## Execution order

1. Merge the readiness PR only after **Ember CPU Validation** is green.
2. Manually dispatch `preflight-v010`. This is a CPU job. Confirm the SFT
   data, held-out split, tokenizer boundaries, and masked completion loss.
3. Obtain explicit approval for one paid `t4-small` run. Do not select
   `sft-v010` before that approval.
4. Manually dispatch `sft-v010` exactly once. Do not start a parallel copy.
   The launcher has a run-state lock and refuses a recent live run or a model
   already marked training-complete.
5. If the job is interrupted, confirm there is no live job, then dispatch
   `sft-v010` again. It will validate and resume the latest durable v0.0.10
   checkpoint rather than start over.
6. After training reaches `training_complete_pending_eval`, manually dispatch
   `eval-v010`. This is a CPU job. It scores the candidate with the semantic
   specification and rescores the stored v0.0.9 completions with that same
   rubric before the non-regression comparison.
7. Promote the candidate only when `evaluations/latest.json` reports
   `promotion_eligible: true` under the v0.0.10 specification. ClawAgent
   integration is a separate, later change.

## Evaluation and promotion gates

v0.0.10 is testing whether Ember can learn the information inside the agent
structure that v0.0.9 already demonstrated. A lower loss is not enough.

Both v0.0.9 (rescored) and v0.0.10 run the same 12 held-out deterministic
prompts: four tool-call requests, four direct responses, and four responses
to supplied tool results. Generation uses seed 1337 and greedy decoding
(`top_k: 1`). Promotion requires:

- tool selection stays essentially perfect: at least 3/4 held-out tool names
  correct, with no regression against rescored v0.0.9 routing;
- tool-argument accuracy rises dramatically: at least 2/4 argument sets
  match the request (v0.0.9 rescored at 0/4);
- tool-result grounding is reliable: at least 3/4 answers use the supplied
  facts and do not call another tool;
- no extra invented dialogue after `<|endoftext|>`: at least 9/12 cases stop
  cleanly;
- no material regression in direct responses (absolute floor 3/4, and no drop
  versus rescored v0.0.9);
- fixed held-out validation loss does not regress more than 5% versus
  v0.0.9; a large loss improvement without the behavior gates still fails; and
- INT4 still loads and generates non-empty output.

These are minimum promotion gates, not a claim that the model is production
ready.

## Private artifacts

Training writes to `Jmiller18899/ember-v0.0.10-t4`:

- `run-state.json` - current lock and lifecycle status;
- `resume/latest.pt` and `resume/manifest.json` - interruption recovery;
- `checkpoints/<run-id>/best.pt` - full checkpoint;
- `checkpoints/<run-id>/best.int4.pt` - portable INT4 checkpoint;
- `config/ember_agent_tool_sft_v0.0.10.json` - exact run configuration;
- `evaluations/<timestamp>-candidate-v0.0.10.json` and
  `evaluations/latest.json` - evaluation and promotion decision; and
- `trackio/ember.db` - experiment metrics.

The structural v0.0.9 evaluation remains in `Jmiller18899/ember-v0.0.9-t4`.
The historical v0.0.8 specification file is unchanged.

## Recovery rules

- `running` state newer than two hours: assume a live job; do not relaunch.
- stale `running` or `error` state: verify Hugging Face Jobs has no active
  copy, then rerun the same `sft-v010` mode to resume.
- checkpoint without a manifest, manifest without a checkpoint, checksum
  mismatch, or configuration mismatch: stop for manual review.
- `training_complete_pending_eval`, `evaluation_complete`, or `complete`:
  never relaunch training.
- failed semantic promotion gate: keep v0.0.9 as the development checkpoint
  and review the evaluation; do not connect v0.0.10 to ClawAgent.
