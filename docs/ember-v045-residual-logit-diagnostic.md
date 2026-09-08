# Ember v0.0.45 residual envelope-entry/logit diagnostic

v0.0.43 and v0.0.44 are two independent prompt-side nulls. With the prompt stack frozen, the pinned v0.0.31 step-479 checkpoint remains at **84/90 (93.3%)** canonical envelopes and correct tool names. The six failures are stable:

- `short_code/len4`: 1 failure out of 6;
- `short_code/len5`: 1 failure out of 4;
- `long_code/4x4`: 1 failure out of 6;
- `long_code/3x5`: 2 failures out of 4;
- `path/plain_leaf`: 1 failure out of 4.

The exact failure IDs are pinned in the v0.0.45 config. If the deterministic source checkpoint no longer reproduces them, this diagnostic stops rather than silently analyzing a different cohort.

## What v0.0.45 changes

Nothing in the model or prompt.

The v0.0.44 user/system stack is reconstructed exactly with all four v0.0.44 code-system selectors on their measured baseline choice. The diagnostic then selects **all 24 held-out cases in the five failure-bearing structural subtypes**. This produces the six failures plus all 18 same-subtype passes as matched controls; no passing examples are hand-picked.

The four exact v0.0.8 tool-call controls are also regenerated and must remain 4/4 before the diagnostic is accepted.

## First-token logit probe

At the `<|assistant|>` boundary, before generating any completion token, v0.0.45 records:

- the rank of the atomic `<|tool|>` token;
- its logit and softmax probability;
- its margin against the best non-tool token;
- the best competing token and its decoded text;
- EOS rank/probability;
- next-token entropy;
- the top eight next-token candidates;
- prompt and target token counts.

The central quantity is `tool_margin_vs_best_other`:

- positive means `<|tool|>` is the greedy choice;
- slightly negative means the model is close to entering the tool envelope but another token barely wins;
- strongly negative or a deep tool rank indicates a qualitatively different first-token regime.

The thresholds in the config are descriptive heuristics only. They do not create a promotion or training gate.

## One-token forced-marker rescue

For every one of the 24 matched cases, the diagnostic performs a second decode in which **exactly one token — `<|tool|>` — is forced as the first assistant token**. After that one token, ordinary greedy decoding resumes with the same budget.

This is deliberately not a prompt and not a training target. The target value is never inserted, copied, teacher-forced, or otherwise supplied. The forced completion is scored only to answer one question:

> If Ember is placed just inside the tool envelope, does its existing downstream decoder already produce a canonical JSON envelope with the correct tool?

A high rescue rate on the six failures would localize the remaining problem to envelope entry. A low rescue rate would show that the residual cases differ deeper in the decoding path.

Forced-marker successes are never counted toward the 84/90 baseline and cannot authorize placement learning.

## Matched comparison

v0.0.45 reports the six failures versus all 18 same-subtype passes, including median/mean tool rank, tool margin, tool probability, target token count, and forced-envelope success. It also reports the same quantities per subtype.

The heuristic regime labels are:

- `downstream_intact_near_envelope_entry_boundary`;
- `downstream_intact_tool_token_competitive_but_suppressed`;
- `downstream_intact_different_first_token_regime`;
- `deeper_decoding_difference_after_envelope_entry`.

These labels summarize evidence; they do not authorize a learning phase.

## Safety / authorization boundary

`training_authorized=false`, `gpu_training_authorized=false`, `production_authorized=false`, and `placement_learning_authorized=false` are hard requirements in the config and runner.

There is no optimizer, backward pass, CUDA execution path, checkpoint write, promotion, deployment, or integration action in v0.0.45.

The useful output is a localization decision for the next experiment: whether future learning should target the envelope-entry token decision specifically, or whether the residual cases require a broader decoding intervention.
