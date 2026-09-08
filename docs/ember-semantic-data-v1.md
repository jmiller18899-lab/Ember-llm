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
