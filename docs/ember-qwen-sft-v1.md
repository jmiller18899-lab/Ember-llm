# Qwen adapter experiment v1

The original Qwen3.5-2B benchmark preserved exact copying but missed warning
labels and sometimes answered a requested thank-you as its recipient. A global
prompt repair harmed copying. This experiment retains the original short system
instruction and teaches the distinction through task-specific examples.

User authorization: one bounded paid training experiment. The launcher requests
one L4 GPU at the listed $0.80/hour, with a 7,200-second timeout (maximum $1.60
compute at that rate). It refuses workflow retries and atomically reserves a
single launch in the private output repository before requesting a GPU. An
ambiguous submission is not retried automatically. The existing writable
GitHub `HF_TOKEN` is used because the connected OAuth credential cannot write
model checkpoints. Credentials are never printed.

Base: `Qwen/Qwen3.5-2B`, revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`, text-only BF16 loading. This is distinct
from the earlier Q4_K_M inference benchmark. Only rank-8 LoRA parameters are
trained, for 120 optimizer steps at 2e-5 learning rate, batch one with eight
accumulations, maximum sequence length 256. No silent truncation is allowed.
The tokenizer disables thinking and masks all prompt tokens from training loss.

The deterministic synthetic data contains 480 training examples, 48 development
examples, and 60 confirmation examples, balanced over six families: greeting,
copying, extraction, classification, drafting, and constrained short writing.
Entities and requests are disjoint across splits. Examples deliberately include
copying warning/error text without classifying it, as well as both drafting a
thank-you and replying to someone else's thanks. These narrow synthetic tests
do not establish broad conversational ability or independent real-world quality.

Before paid submission, GitHub runs the actual tokenizer and collator on all
splits and two optimizer steps through a tiny random hybrid Qwen architecture.
This catches dependency, completion-mask and forward/backward failures; it does
not prove full-size GPU memory use or checkpoint loading will succeed.

The same BF16 runtime generates confirmation answers before and after the fixed
training schedule. Development loss is logged every 30 steps; no adaptive search
or confirmation-driven checkpoint selection occurs. Exact scores for flexible
drafting are diagnostic only and require human review. Strict-family regressions
are reported separately. No automatic model promotion occurs.

Checkpoints are pushed every 30 steps to the private repository
`Jmiller18899/ember-qwen3.5-2b-sft-v1`. Trackio logs locally, while portable metrics,
the data manifest, baseline/final answers and comparison are uploaded to the same
repository. The original Ember checkpoint and prior benchmark evidence are not
overwritten. A successful adapter experiment still needs review and a separate
quantized deployment test on the 4 GB DigitalOcean machine.

References:
- https://huggingface.co/docs/hub/jobs-pricing
- https://huggingface.co/docs/transformers/model_doc/qwen3_5
- https://huggingface.co/docs/trl/sft_trainer
