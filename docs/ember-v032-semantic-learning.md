# Ember v0.0.32 semantic learning experiment

The repaired data passed CPU validation, but no model weights changed in that
preflight. This experiment tests whether Ember can learn the corrected answers
while retaining the copy behavior learned through v0.0.31.

The source is the pinned v0.0.31 step-479 checkpoint. The repaired dataset is
pinned to private revision `dd99fb5df322ffc8716fa8050ccb33f578239454`; all source
and split hashes are checked before optimization. Historical generators and
the frozen semantic evaluator remain unchanged.

| Setting | CPU canary | Conditional T4 experiment |
| --- | --- | --- |
| Semantic training examples | 96; two complete pairs per family | All 2,880 |
| Development loss examples | 48; one validation pair per family | All 576 |
| Copy replay examples | 36; four per kind | 3,600 |
| Optimizer steps | At most 120 | At most 600 |
| Semantic / copy batch | 6 / 2 | 12 / 4 |
| Learning rate | 0.00001 with warmup and cosine decay | Same |
| Objective | 75% semantic CE + 25% copy CE | Same |
| Precision | FP32 | FP32 |
| Training loop time limit | 15 minutes | 35 minutes |

Completion-only labels end at the real EOS. Prompt and padding targets are
ignored. Checkpoints include optimizer and RNG state, and are saved privately
at each validation interval. Best-checkpoint selection uses only development
loss. Trackio metrics are retained with the private experiment artifacts.

The CPU canary must satisfy every condition fixed in the config before the run:

- Training-probe loss falls by at least 10%, and development loss falls.
- Training-probe exact answers increase by at least three, including a gain in
  both tool calls and tool-result responses.
- Development-probe exact answers increase by at least one, with no category
  losing correct answers.
- The existing copy gate passes, all previously correct protected copy cases
  remain correct with EOS, and legacy/expanded exact-copy, continuation, and
  clean-stop rates do not decrease.

The canary is a learning test, not a held-out promotion result. Its 96 training
and 48 validation probes are from the development data. The separate frozen
36-case semantic gate is never used for canary training or checkpoint selection.

On a canary PASS, `begin` may submit one `t4-small` job with a one-hour hard
timeout. Both the launcher and GPU trainer validate the same immutable canary
report against trainer/config/data hashes and recomputed gate checks. A private
launch reservation prevents duplicate paid submissions, including workflow
retries. A failed canary submits no GPU job.

The T4 run starts afresh from the pinned v0.0.31 checkpoint. After selecting its
best development checkpoint, the job runs the unchanged semantic gate and copy
protections on full weights and reconstructed INT4 weights. INT4 and reports
are retained even if the candidate fails. No production deployment or rewrite
of v0.0.31's promotion evidence is performed.

The connected Hugging Face credential lacks repository write scope, so the
existing GitHub `H_F2` secret is used for private checkpoint persistence and the
conditional Jobs submission. The Hugging Face connector can inspect that job.

## Launched experiment

[CPU learning workflow 34215725680](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34215725680)
was started from code `d2ca7fcd18c0ee7b36e0dc032f79a8eea306a25d` in `begin` mode.
All 125 focused tests passed locally and on that runner. The separate
[repository CPU validation](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34215762092)
also passed. The CPU learning measurement is pending; this launch record is not
a canary or semantic PASS.

The trigger has returned to `bootstrap` for subsequent pushes. The active
workflow retains its original `begin` mode: a passing canary can submit its one
guarded T4 job, whose script then evaluates the selected full and INT4 checkpoints.
The run's `publication.json` and, if submitted, `gpu-submission.json` provide the
immutable report revision and actual Hugging Face job ID.
