# Ember v0.0.49 protected replay CPU canary — result

Formal widened-replay run: `34310937571`, job `102337282047`, commit `54c7e7cc7ee13e03c729164504b8679c628f8835`.

The run restarted from the untouched v0.0.31 step-479 checkpoint. All focused and inherited guards passed before learning: **125 tests passed**. Training was CPU-only with 80 optimizer steps, learning rate `3e-7`, gradient clip `0.5`, and loss weights 15% entry / 15% placement / 50% tool replay / 20% copy replay. No GPU, promotion, deployment, or production integration was authorized.

Artifact: `ember-v049-protected-34310937571`, artifact ID `10088576972`, ZIP SHA256 `f520d051b0011f54458de9e2446c4a00151f8fcc4440b1650da5f54d6fd0c115`.

## Target learning

The target skills remained learnable under heavy replay protection:

- Entry development top-1: **18/24 -> 24/24** (+6).
- Entry mean loss: **5.1308 -> 0.0841** (98.36% reduction).
- Placement exact top-1: **0/24 -> 2/24** (+2).
- Placement token top-1: **53.49% -> 69.77%** (+16.28 points).
- Placement mean loss: **3.1154 -> 2.0415** (34.47% reduction).

All configured learning-gain checks passed.

## Replay preservation

Hard-token replay substantially reduced catastrophic forgetting relative to v0.0.48:

- Tool replay token top-1: **100% -> 97.79%** — replay gate PASS (>=95%).
- Copy replay token top-1: **100% -> 96.73%** — replay gate PASS (>=95%).
- Exact historical reference controls: **4/4** — PASS.

However, the existing copy-diagnostic protection still failed. Both legacy and expanded exact-copy/continuation measures regressed enough to trip the protected copy gate.

## Familiar 90-case regression

The familiar evaluation-only battery recovered dramatically from v0.0.48's 33/90 correct-tool result, but remained below the protected 84/90 floor:

- canonical JSON: **77/90**;
- correct tool: **77/90**.

Correct-tool totals by kind after v0.0.49:

| Kind | Result |
| --- | ---: |
| digits | 10/10 |
| entity | 10/10 |
| expression | 10/10 |
| mixed | 10/10 |
| path | 10/10 |
| url | 9/10 |
| model_id | 8/10 |
| short_code | 6/10 |
| long_code | 4/10 |

Protected subtype losses were concentrated in:

- `short_code/len4`: **3** (floor 5);
- `long_code/4x4`: **2** (floor 5);
- `url/one_mixed`: **3** (floor 4).

`short_code/len5`, `long_code/3x5`, `url/two_segment`, `path/plain_leaf`, and `mixed/upper` all met their historical subtype floors.

## Conclusion

v0.0.49 is a **scientific FAIL**, not a harness failure. Hard replay successfully turns the v0.0.48 catastrophic collapse into a much narrower 77/90 retention problem while preserving measurable entry and placement learning. The candidate must not be promoted, GPU-trained, deployed, or used as the source checkpoint for the next experiment.

The remaining issue is no longer basic replay coverage: replay sequences themselves retain >96% token top-1, yet borderline familiar tool decisions still move. That indicates the next protection mechanism should preserve the source model's probability/logit distribution or margins, not only its greedy target tokens.

v0.0.50 should restart again from untouched v0.0.31 step 479 and use frozen-teacher distribution preservation (or an equivalent source-logit/margin distillation term) plus minimal-delta checkpoint selection. Candidate selection must use only disjoint synthetic development/replay evidence; the familiar 90 cases and four historical controls remain final evaluation-only regression gates.
