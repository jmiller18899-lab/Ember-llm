# Semantic repair data v1

The frozen semantic gate found wrong tool arguments and incorrect answers in
v0.0.31 step 479, including a generic checkpoint definition and weather facts
replaced with unrelated values. The historical v0.0.9 generator also emits one
generic definition across concepts and only healthy service statuses.

This adds a separate, versioned dataset. Historical pinned generators, the
promotion result, and the frozen semantic evaluator remain unchanged.

| Split | Tool calls | Direct responses | Tool-result responses | Total |
| --- | ---: | ---: | ---: | ---: |
| Training | 960 | 960 | 960 | 2,880 |
| Validation | 192 | 192 | 192 | 576 |

Each category contains eight equally represented families. Examples require
literal requested arguments, correct selection between alternatives, concrete
definitions and plans, preserved facts, negative and decimal results, correct
temperature units, unhealthy services, errors, empty results, and missing owners.
Every two-example pair has different correct answers and stays in one split.
City and timezone pairs stay together across wording variants.

The validator independently reads each request and tool result, checks strict
field types, recomputes arithmetic, and rejects another pair member's answer and
post-EOS noise. These negative probes are never exported as training labels.
It does not trust an answer field in the record. Definitions, plans, and rewrites
use explicit reviewed reference tables rather than an automated quality judge.

The validation split is for development: templates and some concept answers are
shared. It is not an unseen-concept benchmark. Known frozen benchmark inputs and
exact tool-argument sets are excluded, including normalization and system-prompt
changes. Generic concepts and answer labels such as healthy/unhealthy and
unavailable remain trainable. This is not exhaustive semantic decontamination.

## CPU checks

Run structural and semantic checks without private model access:

```sh
python jobs/ember_semantic_data_preflight.py --data-only --output-dir semantic-data-results
```

That produces `DATA_ONLY_PASS`, not a complete preflight PASS. For the complete
check, use the CPU workflow or run the same command without `--data-only` with
`HF_TOKEN` available. The actual pinned v0.0.31 BPE must encode every example with
stable completion boundaries, one final EOS, no truncation, and space for a
96-token generation budget within the 256-token context. Completion-only labels
include the first answer token and EOS, while prompt and padding targets are
ignored. `encode_row` is the checked adapter for future training integration.

One sample per family per split receives a CPU forward pass; training samples
also receive a backward pass to check finite gradients. No optimizer is created
or stepped. Every model-state tensor is compared with the original checkpoint
afterward. Sample losses are compatibility evidence, not a quality evaluation.

`--publish` requires the complete preflight and writes only synthetic data and
reports to the private `Jmiller18899/ember-semantic-data-v1` dataset. A manifest
records data, source, evaluator, tokenizer, and checkpoint hashes. The workflow
also retains artifacts on failure. No model checkpoint is written, no training
job starts, and no ClawAgent deployment occurs.

A data PASS means these examples are ready for a separate training decision.
It does not change v0.0.31's semantic FAIL or its legitimate legacy promotion
PASS. A future trained candidate must rerun the frozen semantic gate and receive
human review of any valid answer rejected by its conservative finite oracle.

## Verified result

The [complete CPU preflight](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34191668495)
passed on code `f05b0034d1c59a2b6f488859d79c2729e6777496`.
The [repository CPU validation](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34191671392)
also passed. The dedicated runner passed all 107 focused data and frozen-gate tests.

| Check | Observed result |
| --- | --- |
| Valid examples / normalized unique inputs | 3,456 / 3,456 |
| Rejected corrupted targets | 6,912 |
| Frozen benchmark input / argument matches | 0 / 0 |
| Actual BPE vocabulary / EOS token ID | 16,384 / 6 |
| Data window / model context capacity | 256 / 512 tokens |
| Longest prompt / complete sequence | 113 / 129 tokens |
| Truncated examples | 0 |
| Examples supervising the final EOS | 3,456 |
| Prompt / padding loss tokens | 0 / 0 |
| CPU forward / backward samples | 48 / 24 |
| Finite gradient tensors checked | 1,056 |
| Optimizer steps / model-state changes | 0 / 0 |

The actual tokenizer exposed a trailing newline after EOS in the initial data.
Final targets end exactly at EOS, so there is no post-EOS training token. The
preflight also distinguishes the 256-token data window from the model's larger
512-token capacity; the examples are checked against the smaller window.

The private dataset was saved at
[revision dd99fb5](https://huggingface.co/datasets/Jmiller18899/ember-semantic-data-v1/commit/dd99fb5df322ffc8716fa8050ccb33f578239454).
Its content prefix is
`ember-semantic-data-v1/776c565486efc36d5bd535cf144a4348b163ac63cfa760cf4adc2bd95a654e74/`.
Training and validation JSONL files are at that prefix; the full report and
manifest are under `runs/34191668495/`. The Actions artifact contains the same
bytes plus publication details. Downloaded artifact hashes were verified against
both the manifest and a deterministic local rebuild.

| File | SHA256 |
| --- | --- |
| train.jsonl | `fc66cb1e116861fbc5f13499d56d63a8e4d6db3d43262a888df1907251b1d100` |
| validation.jsonl | `a51623cea837deda0131dd7fa8f0fc05e61ee8aae60b6e404ccc59c476df2761` |

The trigger is returned to `bootstrap`. Training configuration remains
unauthorized, and neither the old model repository nor the promotion state was
modified by this data job.
