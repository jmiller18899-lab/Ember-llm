# Ember v0.0.36 single-target agent-envelope preflight

v0.0.34 and v0.0.35 are preserved as failed prompt-baseline experiments. Neither authorized training or GPU work.

The v0.0.35 result made routing much cleaner once JSON existed (44/45 JSON-valid envelopes named the correct tool), but envelope JSON validity itself was only 45/90. Inspection of the original v0.0.32 CPU report showed why: its clearest baseline examples of `right envelope + right tool + wrong argument value` were single-target literal requests. The preflight revisions had added `old` and `fallback` distractors that were not required by the proposed baseline and changed the behavior under measurement.

v0.0.36 removes that extra variable. It performs two CPU-only measurements against the pinned v0.0.31 step-479 checkpoint.

First, it runs the four exact tool-call prompts from `config/ember_v0.0.8_eval.json` without rewriting them: weather/Detroit, calculator/347×28, web search/Python release, and get time/Tokyo. All four must produce canonical JSON tool envelopes and all four must name the expected tool. If this 4/4 reference gate fails, the supposed agent-style starting point is not present in the pinned checkpoint and the experiment stops.

Second, it runs the same 90 historical held-out values used by the copy diagnostics, balanced at ten cases for each of nine value kinds. Each prompt contains only one requested value:

- entity → `weather(location)` using `What is the weather in <value> right now?`
- digits and expression → `calculator(expression)` using `Calculate <value>.`
- short_code, long_code, model_id, url, path, and mixed → `web_search(query)` using `Find current information about <value>.`

The system lines for those three tools are copied exactly from the v0.0.8 evaluation config. There is no `TARGET=...`, no literal-copy system role, and no old/fallback distractor.

The 90-case gate requires at least 86/90 canonical JSON envelopes and at least 86/90 correct tool names. `slot_exact` is diagnostic only. If the two envelope gates pass while `slot_exact` remains low, that is the intended large-sample reproduction of the v0.0.32 finding: envelope and routing are established, and the remaining error is value placement.

`training_authorized=false`, `gpu_training_authorized=false`, and `production_authorized=false`. A PASS does not authorize any model update; it only establishes a valid baseline for a separately designed next phase.
