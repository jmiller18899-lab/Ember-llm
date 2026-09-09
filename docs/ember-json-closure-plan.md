# Ember JSON-closing protection experiment

The verified original trajectory lost a valid short-code tool call at step 13
when its JSON-closing token lost to a newline. The original placement-learning
gate was reached only at step 40, after two structural losses. This bounded
experiment tests protection aimed specifically at the closing-token decision.

## Fixed design

Use the same SHA256-verified v0.0.31 step-479 source, original rung-0 learning and
KL data, deterministic batch schedule, 40 AdamW steps, LR 1e-7, gradient clip 0.25,
weight decay 0.01, and weights 0.20 placement / 0.55 tool KL / 0.25 copy KL.
CPU float32, two threads, zero dropout, deterministic algorithms.

Before training, construct eight fresh structurally valid short-code responses
from the source: six four-character and two five-character targets. At each
response's first argument-closing quote, compute the source closing-token versus
strongest alternative logit-margin gradient. Accept only closing tokens that
start at the quote, so a token overlapping the value's final characters is never
used. JSON string parsing handles escaped quotes. Require eight independent
directions and fix their orthonormal basis before all frozen evaluations.

After each ordinary AdamW step, project the **actual parameter proposal** onto
the orthogonal complement of those directions. AdamW's internal moments remain
those of the unprojected gradients; the next gradient is evaluated at the
projected model. Check mathematical non-expansion/orthogonality, a nonzero actual
float32 update and actual residual fraction <=0.001 at every step. Record how
much proposal norm is retained. No extra loss term or learning-rate change.

This is a first-order constraint at the source, not a guarantee of nonlinear
decoding behavior. Measure actual closing margins and free-generated responses
as well. Argument values in source responses may be wrong; only their valid tool
structure is protected. No value token is directly supervised by this mechanism.

Prepare six separate fresh structurally valid holdout responses (four four-
character and two five-character targets). These never provide gradients or
projection directions. Reject values from the available historical corpora,
current ladder data, and all attempted values in the earlier first-update
diagnostic; accepted anchor and holdout cohorts are mutually disjoint. The fixed
namespace and 48-attempt cap per quota are committed before execution.

## Evaluation and limits

Require the original baseline: 84/90 familiar JSON and correct tool, 4/4 reference,
1/24 exact placement and 88/170 placement tokens. Observe short-code retention,
placement, actual closure margins and both fresh cohorts at steps 1, 12, 13, 23
and 40. Run full familiar/copy/reference/KL guards on the source and at step 40.
This reduced observation schedule avoids rerunning the entire previous trace.
It cannot identify the first failure at an unobserved step or exclude every
intermediate operating point.

The **preselected step-40 endpoint** must pass every original learning and
preservation check, individual-case retention, fresh anchor/holdout structural
retention, continued top-1 closing tokens at the protected prefixes, and all 40
projection mechanism checks. No thresholds are relaxed. A completed run can have
a failed scientific endpoint.

Compare with verified original run `34369464055`. The intervention changes both
allowed direction and the retained proposal norm; there is no norm-matched
control here, so a benefit would not isolate those effects. The original
development set is reused for the controlled comparison, not a new promotion
evaluation. Frozen familiar cases never enter training or anchor construction.

Hard limit: 30 minutes of script execution, 40 minutes including runner setup.
Check state hashes around observation and verify unchanged teacher/pristine
restore at the end. Save report, data, summary and hashes as the workflow artifact.
No checkpoint export, GPU job, promotion, deployment or ClawAgent integration.
