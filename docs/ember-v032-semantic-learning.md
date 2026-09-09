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

## Completed CPU canary: FAIL

[CPU learning workflow 34215725680](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34215725680)
ran code `d2ca7fcd18c0ee7b36e0dc032f79a8eea306a25d` in `begin` mode and completed
all 120 optimizer steps. Step 120 had the lowest development loss and was
selected. The model state changed, and the checkpoint and report were saved.
The canary returned measured `FAIL` (exit code 2), not an execution error or
timeout. The conditional T4 submission step was skipped.

All 125 focused tests passed locally and on the runner. The separate
[repository CPU validation](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34215762092)
also passed. Recomputing the learning and copy gates from the downloaded report
reproduced its failure, and the GPU launcher guard rejected that actual report.

| Measurement | Pinned v0.0.31 source | CPU step 120 |
| --- | --- | --- |
| Training-probe loss | 3.39537 | 0.82626 (75.7% lower) |
| Development loss | 3.40897 | 1.71859 (49.6% lower) |
| Training-probe exact answers | 0/96 | 23/96 |
| Development-probe exact answers | 0/48 | 2/48 |
| Legacy exact copies | 6/9 | 4/9 |
| Expanded exact copies | 43/90 | 23/90 |
| Legacy continuation top-1 | 65/69 | 63/69 |
| Expanded continuation top-1 | 662/730 | 624/730 |
| Copy clean-stop rate, both sets | 100% | 100% |

The only failed top-level learning check was `copy_preserved`. The candidate
failed the existing copy gate, lost two previously correct legacy cases and 21
previously correct expanded cases, and gained only one expanded case. These
sets overlap and should not be added together. For example, a previously exact
copy of `53*19+7` became `4*19+7`, and `https://example.test/a7Q9` became
`https://example.test/a7Q7`.

The 23 correct training answers comprise 14 tool calls, three direct responses,
and six tool-result responses. Development results were 0/16 tool calls, 0/16
direct responses, and 2/16 tool-result responses. The two development successes
were the fixed fallbacks `Owner unavailable.` and `No results found.`; they do
not establish accurate handling of new argument or result values. All 144
semantic probe generations reached EOS, but many still contained wrong values
or incoherent content. These remain development probes, not the frozen
held-out semantic promotion evaluation.

Evidence is retained in the private canary repository at
[immutable revision c091819](https://huggingface.co/Jmiller18899/ember-v0.0.32-canary/commit/c0918190230a18c42f1dedb7c14c9fb8fe64c7c7):

- Report: `runs/ember-v032-canary-20260908T103020.731652Z/report.json`
- Report SHA256: `bbaa9fe8f68bb7935572d40bab691154407a3ab49cd1c4deeb97fa4f1970b442`
- Best-checkpoint SHA256: `68633ff9ad28690c941d9621d2176d0f9d6e5ea8eb99f5461927379b5be21827`
- GitHub artifact: `ember-v032-learning-34215725680` (ID `10051998400`).

The tested 75% semantic / 25% copy objective with only 36 replay examples did
not preserve the source's copy behavior. A useful next CPU experiment would
increase copy replay coverage and weight and record copy diagnostics at each
validation interval. Those are hypotheses to test, not established fixes.
It should start again from the pinned v0.0.31 source, retain every existing gate,
and use a separate recorded configuration. No retry or GPU run was launched
while diagnosing this failure. The trigger remains `bootstrap`.
