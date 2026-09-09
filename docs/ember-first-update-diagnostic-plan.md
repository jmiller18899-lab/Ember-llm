# Ember bounded first-update diagnostic

Authorized scope: one bounded CPU diagnostic, with one first AdamW proposal
evaluated without restriction and after an explicit direction restriction.

## Starting point and limits

- Published code lineage: `50064f0e07b035538b053b316bc3fff1cba0696e`.
- Source: untouched Ember v0.0.31 step 479, immutable Hub revision and SHA256
  from `config/ember_semantic_data_v1.json`; the existing loader verifies both.
- Exactly one AdamW proposal, empty optimizer state, LR `1e-7`, weight decay
  `0.01`, gradient clip `0.25`, CPU float32, two threads, deterministic
  algorithms, and source dropout required to be zero.
- Published v0.0.51 loss weights: 0.05 entry, 0.20 placement, 0.50 tool KL,
  0.25 copy KL. These are not the unpublished ladder's reported 0.80
  preservation mixture. Its later rung-0 report/code/data were unavailable;
  this diagnostic does not claim to reconstruct that run or its exact failures.
- Hard script limit: 25 minutes; workflow limit including setup: 35 minutes.
  At most 32 teacher-generation attempts per preservation quota.
- Fixed data, projection and thresholds; no hyperparameter search or further
  optimizer steps. No checkpoint persistence, GPU run, promotion or deployment.

## Fresh data and identical learning proposal

Use a separate deterministic namespace and reject values in the available
historical/familiar lists and published synthetic corpora through v0.0.51.
Template, learning, development and preservation targets are mutually disjoint.
Exclusion cannot be verified against the unpublished ladder's missing values.

Require eight accepted four-character short codes, two five-character short
codes, and two examples for each of the eight other kinds. Accepted tool rows
meet the existing preservation definition: source-generated canonical JSON and
correct tool name. Argument-copy correctness is recorded, but is not required
or replaced with a gold response. Copy rows use the published direct-response
filter, one fresh example per kind. No generated error is relabeled as a correct
answer. Quota exhaustion ends the bounded diagnostic with an explicit error.

The learning batch contains one entry example per target subtype and two
placement examples per target subtype. Half of the eight-row tool KL batch is
four-character short codes. Both arms share the same actual AdamW proposal,
including optimizer scaling and weight decay.

## Direction restriction

For each of eight fresh four-character code tool responses, find its smallest
source winner-versus-runner-up logit margin along the source-generated response.
Compute that margin's parameter gradient at the pristine source. These are
first-order sensitivity directions, not KL or L2 restoring gradients.

Orthonormalize with two-pass modified Gram-Schmidt. Project the actual AdamW
parameter change onto the orthogonal complement of this span. Check
non-expansion and mathematical projection residual before evaluation. This
changes allowed directions rather than uniformly scaling the entire proposal.

Recompute actual parameter changes after applying the float32 projection.
Report residual after rounding, cosine with the raw proposal, changes by tensor,
and the eight source margins. The mechanism check requires actual residual
fraction <=0.001. First-order margin preservation does not guarantee preservation
of all decoding decisions; the behavioral guards remain mandatory.

## Evaluation

Fix all batches, directions and proposals before the frozen evaluations. Run
the source and both arms through the original helpers for:

- fresh 24-case entry and 24-case placement development probes;
- familiar 90-case JSON and correct-tool battery, every kind/subtype floor,
  and retention of every individual source-passing case ID;
- four reference controls;
- legacy/expanded copy diagnostic and every original copy-protection check;
- teacher KL and token retention on fresh preservation rows.

Record generated responses and first changed tokens. For any lost familiar
case, measure source/candidate margins at the same first-divergence prefix.
All frozen batteries remain excluded from training gradients and projection
construction. Evaluation results never tune or extend the experiment.

Execution COMPLETE is separate from scientific PASS. An operating point still
requires every original synthetic gate (+1 entry, +1 exact placement, +2
percentage points placement tokens, both KL budgets and token retention floors),
all preservation checks, individual-case retention and a nonzero update. No
extra run is triggered if neither arm passes. Finish by verifying the pristine
source restore and frozen teacher state hash.

Evidence: `first-update-results/report.json`, `data.json`, and `summary.md`,
persisted as the GitHub Actions artifact. Model weights and credentials are
excluded from these artifacts.
