# Ember candidate-generated prefix closure: prespecified CPU experiment

Authorized after the JSON-closing projection result. One CPU experiment, starting
from the immutable v0.0.31 step-479 source. No checkpoint export or promotion.

The previous projection protected punctuation at source prefixes, but candidate
argument text changed. This test instead adds cross-entropy on pure closing
punctuation after candidate-generated argument bodies. It does not supervise
those bodies or claim their values are correct.

## Fixed design

- Original 40-update rung-0 AdamW recipe and original placement/tool-KL/copy-KL
  batches, LR 1e-7, original weights 0.20/0.55/0.25, clip 0.25, weight decay 0.01.
- Add closing-punctuation CE with fixed weight 0.20 before shared clipping.
  No gradient projection. This is an additional objective, not an optimizer
  trajectory replication or a matched-gradient-norm ablation.
- Eight fresh source-structurally-valid training prompts: six four-character and
  two five-character codes. Six separate fresh source-valid holdout prompts:
  four four-character and two five-character codes. Namespace and value exclusions
  include earlier closure attempts, older first-update attempts and historical
  data. Selection uses source structure only, never candidate evaluation results.
- Regenerate all eight training prompts before updates 1, 6, 11, 16, 21, 26, 31,
  36. Use all usable examples for the next five updates. Preserve generated token
  prefixes exactly; no retokenization of generated argument text.
- Recognize the expected tool envelope and a JSON string body. A closing quote,
  unescaped newline or EOS marks an observed boundary. Reject invalid escapes,
  unrecognized scaffolding, unbounded truncated bodies, and boundaries that
  straddle tokens. Log every accepted and rejected generation. Abort if a refresh
  has no usable examples. Do not substitute holdout or familiar cases.
- Copy only pure closing punctuation tokens from that prompt's source suffix,
  ending before lexical tool-name content. Verify identical decoding in the new
  context. Label these tokens only; all argument and prompt targets are masked.
- Probe steps 1, 12, 13, 23, 40. Full familiar/copy/reference/KL source and endpoint
  evaluations. Frozen evaluations do not supply gradients, stopping decisions,
  weights, target selection or refresh timing.
- Endpoint requires original learning gates (one additional exact placement case
  and at least 3 percentage points in placement token top-1), all original
  preservation gates, all 84 source-passing familiar cases individually retained,
  fresh training and holdout structural retention, and 40 nonzero finite updates.
- CPU float32, deterministic, two threads. Script time bound 30 minutes;
  workflow limit 40 minutes. No output weights; restore source and verify teacher.

Report structural validity separately from exact arguments. Placement is the
existing teacher-forced probe, not end-to-end task accuracy. Report whether
refreshes actually encountered changed and unclosed bodies: if not, the proposed
repair mechanism was not exercised. Intermediate probes do not establish the
first failure time or rule out every possible operating point.

Local validation: 63 tests passed, including escaping, boundary alignment,
wrong-argument label masking, gradients only at supervised output positions,
data exclusion and inherited preservation/optimizer contracts.
