# Ember v0.0.33 CPU copy-replay canary

The v0.0.32 canary learned 23/96 training answers and 2/48 development answers,
but expanded exact copying fell from 43/90 to 23/90. The copy-preservation gate
rejected it and the T4 submission was skipped. Its code and failed evidence
remain preserved in [PR #23](https://github.com/jmiller18899-lab/Ember-llm/pull/23).

This separate CPU experiment tests whether broader and more heavily weighted
copy replay can preserve copying while still learning semantic answers. It
starts afresh from the same pinned v0.0.31 step-479 checkpoint. The failed
v0.0.32 checkpoint is not used as the source.

| Setting | Failed v0.0.32 canary | v0.0.33 canary |
| --- | --- | --- |
| Copy replay pool | 36; four per kind | 360; 40 per kind |
| Copy loss weight | 25% | 75% |
| Semantic loss weight | 75% | 25% |
| Copy batch | 2 | 6 |
| Semantic batch | 6 | 6 |
| Semantic training / development probes | 96 / 48 | Same examples and seed |
| Optimizer steps | 120 | 120 |
| Learning rate | 0.00001, warmup and cosine decay | Same |
| Precision | FP32 | FP32 |
| Copy diagnostics | Before and after training | Also at steps 40, 80, 120 |

Replay remains restricted to the historical generator's training split; it
excludes all its diagnostic values. The larger pool includes the old replay
pool and stays balanced across nine kinds. Sampling is with replacement, and
the report records which replay examples were actually seen. This tests the
combined replay recipe; it does not isolate the effect of each changed setting.

All three checkpoints, their hashes, optimizer/RNG state, and copy diagnostics
are saved privately in `Jmiller18899/ember-v0.0.33-canary`. Best-checkpoint
selection still uses only development loss. Copy diagnostics are observations
and final acceptance checks; they do not select a more favorable checkpoint.
The selected checkpoint's copy evidence is reproduced after reloading it.

The runner imports the unchanged v0.0.32 learning and copy gates. All learning
thresholds, the semantic dataset revision, the model implementation, and the
v0.0.31 source remain pinned. Lower loss alone cannot pass: exact answers must
improve, the existing copy gate must pass, every previously correct protected
copy case must remain correct, and protected copy rates must not decrease.
The 36-case frozen semantic promotion benchmark is excluded from this trial.

The loop has a 20-minute limit including interval diagnostics and saving,
within a 40-minute GitHub job limit. The workflow uses the existing `H_F2`
secret and pinned CPU dependencies. It has no GPU submission step. A PASS is
only a CPU learning-canary result; it does not authorize production use.

The GitHub run summary will show the failed checks and per-checkpoint copy
results, including when the training command exits with a measured FAIL. JSON
reports are retained as workflow artifacts, while weight files remain in the
private model repository. This experiment has not produced a measured result yet.

Local verification passed all 134 focused tests, including actual optimizer
updates toward the more heavily weighted copy target, checkpoint/optimizer/RNG
persistence, checkpoint selection independent of copy results, unchanged gate
checks, and replay separation. The workflow repeats these tests before training.

## Launched CPU experiment

[Workflow 34219463932](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34219463932)
started from code `06b9e19011de53be90dc280f571f63b8174e92d1` in `canary` mode.
The measured result is pending. The trigger has returned to `bootstrap` for
future pushes; the active workflow keeps its originally checked-out canary
request. The implementation is published in
[draft PR #24](https://github.com/jmiller18899-lab/Ember-llm/pull/24).
