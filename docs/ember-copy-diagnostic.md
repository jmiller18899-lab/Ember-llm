# Direct-answer copy diagnosis

Run: https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34862254165
Source: e52ecfcd824166aec7d7419d447daa499d53533d

This read-only CPU comparison evaluated the original full model and the unchanged
v2 adapter on the same 32 diagnostic requests. No weights were trained, fresh
confirmation consumed, or production changes made. These probes are now consumed
development evidence, not an independent promotion benchmark.

| Cohort | Original model | V2 adapter |
| --- | ---: | ---: |
| Training-seen entities in diagnostic task frames | 0/8 | 2/8 |
| Previously consumed confirmation entities | 0/8 | 0/8 |
| Novel diagnostic entities | 0/8 | 0/8 |
| Four exact training requests | 0/4 | 4/4 |
| Three classifications plus exact greeting | 0/4 | 3/4 |

The adapter passed 9/32 overall, but this heterogeneous total is not comparable
to the prior 48/24 suites. All 18 exact-copy/extraction probes failed. The two
training-entity passes were the thank-you and issue-title tasks. Classification
passed 3/3; the exact greeting failed. The four training replays are four samples,
not evidence of perfect recall of all 312 training examples.

For “Repeat exactly: Nora”, the model added a definition. For “Repeat exactly:
Nora reviewed 23 invoice records.”, it appended “in 2 minutes.” Extracting either
the name or the number produced a full sentence. These failures persist with
training-seen tokens. Previously consumed and novel entity sets also failed all
8 probes each. The untrained baseline failed every probe, so this run does not
show copying was lost during v2 training; it shows v2 did not establish it.

Inference: the candidate can reproduce sampled familiar answer patterns and
classify statuses, while failing to obey narrow copy/extraction instructions.
This supports investigating task control and copying together. It does not
identify an architectural cause or prove that more steps will fix the problem.
Exact normalized matching is conservative, but the examples above have actual
extra or altered content rather than harmless paraphrases.

Next bounded experiment: train paired copy/extract requests that use the same
names, subjects and numbers with different required output lengths. Balance this
against the previous classification and grounded-writing tasks. Require separate
copy, extraction, classification and content-preservation gates, rather than an
aggregate that classification alone can satisfy. Keep routing layers protected
and freeze a new confirmation set before any future candidate evaluation.

The raw outputs are in reports/ember-copy-diagnostic-34862254165.json. Adapter
SHA-256 matched the recorded v2 artifact; the model loader verified base bundle
hashes. The selected adapter keys were restricted to block 5 and output
normalization. CPU repository validation passed for the diagnostic source.
