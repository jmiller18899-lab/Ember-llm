# Ember bounded first-update CPU diagnostic result

Execution completed on 2026-09-09. Exactly one CPU AdamW proposal was evaluated in two arms. The diagnostic took **341.85 seconds (5 minutes 42 seconds)**. Its 24 local/runner contract tests passed.

**Scientific result: no operating point found.** Both arms preserved the previously passing behavior, but neither improved the discrete placement measurements after one step.

| Measurement | Source | Unrestricted | Projected |
| --- | ---: | ---: | ---: |
| Familiar correct tool | 84/90 | 84/90 | 84/90 |
| Source-passing familiar cases retained | 84 | 84/84 | 84/84 |
| Lost familiar case IDs | — | none | none |
| Historical kind/subtype floors | reproduced | PASS | PASS |
| Reference controls | 4/4 | 4/4 | 4/4 |
| Full copy-protection guard | baseline | PASS | PASS |
| Exact placement on fresh development cases | 1/24 | 1/24 | 1/24 |
| Correct placement tokens, identical case set | 87 | 87 | 87 |
| Existing synthetic learning gate | — | FAIL | FAIL |

Eight fresh teacher-correct four-character code examples supplied eight independent source-margin sensitivity directions. Their source margins ranged from 0.1253 to 1.2735. The raw optimizer proposal had L2 norm 0.0004860536; the projected proposal had norm 0.0004791781 (98.59% of the raw norm). Both actual updates were nonzero. The projected arm passed its actual float32 projection-residual check (fraction <=0.001).

The source reproduction gates and final teacher/pristine-restore integrity checks are enforced by the runner before it can emit COMPLETE. Both arms passed individual-case retention, all familiar floors, references and every original copy-protection check. Thus preservation failure did **not** occur on the first update in this diagnostic. This does not establish where the unpublished ladder's two failures occurred, and does not establish a useful learning/preservation operating point.

This run uses the published v0.0.51 objectives (0.05 entry, 0.20 placement, 0.50 tool KL, 0.25 copy KL), with LR 1e-7 and fresh values disjoint from the available published corpora. The later ladder's source files and exact data were unavailable; its reported 0.80 preservation mixture is not reconstructed here. No second optimizer step, additional experiment, checkpoint persistence, promotion, GPU job or deployment was performed.

[Completed run](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34353485215) · [Full report, data and summary artifact](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34353485215/artifacts/10104961019)

Evaluated commit: `9f232d050991acc37cef1759fb2ed64406e539b2`. Artifact ID: `10104961019`; SHA256: `a1d5caee7852c2568a968c0c3a15125b397c5ee29834c33761dd4a67ed796105`.

This note and the event JSON were verified from the completed job log. The artifact was successfully uploaded by GitHub; its returned attachment URL could not be downloaded into this workspace (HTTP 403), so unlogged raw metrics are not represented here as independently inspected.

Follow-up evidence check (2026-09-09): the artifact was downloaded successfully
and its report/data/summary were inspected. The raw report confirms the logged
retention and placement results above. Weighted step-0 gradient norms were
22.7292 entry, 16.7970 placement, and 0.0000688187 combined preservation; these
belong to this separate diagnostic and must not be substituted for the ladder's
27.7029 placement gradient. The ladder source is now available and is used by
the separate rung-0 trajectory diagnostic.
