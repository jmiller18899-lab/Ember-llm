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

## Measured result

The formal corrected diagnostic was GitHub Actions run `34289689685`, job `102273238774`, on commit `92ea5b1030c39f638827c274f30ce02b5821409b`.

- Focused/inherited guards: **99 passed**.
- Restored v0.0.41 runner: Python compile **PASS**.
- Exact v0.0.8 reference controls: **4/4** before the residual diagnostic was accepted.
- Residual-subtype cohort: **24 cases** — the six stable failures plus all 18 same-subtype passes.
- The exact six historical failure IDs reproduced.
- Diagnostic status: **COMPLETE**.
- Artifact: `ember-v045-logit-34289689685`, artifact ID `10080894928`, ZIP SHA256 `9648818041ceae00053ea6f13fd1e7a85343d90d2cb752aebfb43bf357fa7522`.

### Failure group versus matched passes

| Measurement | Six failures | 18 matched passes |
| --- | ---: | ---: |
| median `<|tool|>` rank | **19** | **1** |
| mean `<|tool|>` rank | 56.33 | 1.00 |
| median tool margin vs best other | **-1.4068** | **+2.0095** |
| mean tool margin | -1.4480 | +1.6840 |
| median tool probability | 0.003977 | 0.013332 |
| median target token count | 6.0 | 6.5 |
| valid correct-tool envelope after one forced marker | **2/6** | **18/18** |

The matched passing group is extremely clean: all 18 cases rank `<|tool|>` first and all 18 remain valid after the one-token intervention. The failure group is materially different, but not in one uniform way.

### Six residual cases

| Case | Subtype | Tool rank | Tool margin | One-marker rescue | Interpretation |
| --- | --- | ---: | ---: | --- | --- |
| `system_target_short_code_02` | `short_code/len5` | 1 | +1.6712 | no | already enters the tool envelope; failure is downstream |
| `system_target_short_code_03` | `short_code/len4` | 3 | -1.6754 | no | tool token is competitive, but downstream decoding also fails |
| `system_target_long_code_02` | `long_code/3x5` | 226 | -3.7581 | **yes** | strong entry suppression; downstream tool decoder is intact |
| `system_target_long_code_05` | `long_code/4x4` | 72 | -3.9198 | **yes** | strong entry suppression; downstream tool decoder is intact |
| `system_target_long_code_08` | `long_code/3x5` | 35 | -1.1382 | no | entry suppression plus a deeper decoding failure |
| `system_target_path_05` | `path/plain_leaf` | 1 | +0.1325 | no | tool marker is already greedy; failure is downstream |

Forced-marker rescue on the six failures was only **2/6 (33.3%)**. Two cases were classified near the configured boundary and three were competitive by the broader heuristic, but the overall regime is:

`deeper_decoding_difference_after_envelope_entry`

## Conclusion

v0.0.45 rejects the idea that the remaining 84/90 plateau is one simple tool-entry problem.

There are at least two distinct mechanisms:

1. **Envelope-entry suppression with intact downstream decoding** — `long_code_02` and `long_code_05`. Their `<|tool|>` ranks are 226 and 72, yet forcing exactly that one marker immediately restores a canonical correct-tool envelope.
2. **Downstream envelope-construction failure** — the other four residual cases. Most importantly, `short_code_02` and `path_05` already rank `<|tool|>` first, so an entry-only learning objective cannot solve them. `short_code_03` and `long_code_08` also remain invalid after the forced marker.

That means a narrow objective that only teaches Ember to choose `<|tool|>` would be incomplete. It could plausibly fix two residual long-code cases but would leave four structural failures unresolved.

Do **not** authorize placement learning or GPU training from v0.0.45. The next useful experiment is a CPU-only v0.0.46 structural-prefix diagnostic: hold prompts and weights frozen, teacher-force only non-value canonical envelope structure (`<|tool|>`, JSON opening/name/tool/arguments/slot-key boundaries) on the four non-rescued failures versus matched passes, and measure the token-by-token divergence. The target argument values must remain unforced so `slot_exact` stays an honest future learning target. The two rescued long-code cases should be kept as a separate entry-suppression cohort and their first-token competitors recorded.

No optimizer, GPU job, training, promotion, deployment, or production integration was run.
