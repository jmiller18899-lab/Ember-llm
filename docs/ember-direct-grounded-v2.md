# Grounded direct-answer learning v2

The previous direct-answer trial improved its old 24 lexical checks from 0 to 2,
but one passing thank-you invented a recipient. Lower development loss was not
sufficient evidence of useful content. This trial keeps the original evaluation
and raw answers while adding a conservative source-grounded measurement.

## Data and scoring

The training split contains 240 new examples and the 72 unchanged historical
training examples. The new tasks cover named thank-you messages, issue-title
rewrites, preservation of names/counts/durations in facts, and status labels.
Forty-eight new development examples and 24 new confirmation examples use separate
requests, entity sets, and request frames. Split validation rejects exact
normalized prompt overlap and protects old development/confirmation requests.
Related task templates still share structure: this is assistant-authored finite
coverage, not a blind human test or a broad instruction corpus.

New development/confirmation answers must match a supplied reference after
case/punctuation normalization and must terminate correctly. Numbers, signs,
names, subjects, order, and extra claims matter. A valid paraphrase may fail this
conservative metric. Its scope is bounded content fidelity; it is not a general
semantic judge. The old lexical scores are retained on the same new cases, and
all old 24 quality generations are reported separately before and after training.

## Bounded learning

The CPU run starts from the original frozen full-precision model, not the
previous experimental adapter. It trains only block 5 and output normalization
for at most 240 steps, using batch size 2 and learning rate 0.0002. The earlier
trial used different development data and fewer steps, so its loss is not a
like-for-like trend and any gain cannot be attributed to data alone.

Checkpoints at steps 80, 160, and 240 compete with the unchanged baseline on
answer-token development loss. Greedy generation is evaluated only after that
selection. The learning gate requires at least a 10% loss improvement, at least
three additional reference-matched development answers, and no decrease in the
old 24 lexical task-check score relative to this run's baseline. The 24 new
confirmation cases are generated only if that gate passes. The separate strict
confirmation gate requires 24/24; passing a small development gate is not release
readiness. The original 12-case confirmation is untouched by this experiment.

All protected parameter hashes and sampled block_04 features must remain exactly
unchanged. The selected adapter is saved/reloaded before final measurement.
Experiment source/data hashes are verified before and after the run. Tool-routing
code, the optional v10 service flow, and original checkpoints remain unchanged.
No production promotion or INT4 answer export is part of this trial.

The exact frozen setup passed 20 data/scoring/isolation tests. The source freeze
is `fff4a131be59e3fe03b2b4c4ed99fa9eec2293d9`; the subsequent CPU run trigger is
`78e5a3f5d56b298d12ac86db39b0b227021e779c`.

## Measured result

[CPU trial 34857841073](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34857841073)
completed training and saved its artifact. The final quality assertion failed as
intended because strict confirmation failed. The separate source CPU validation
run 34857848039 passed; this does not establish model readiness.

| Measurement | Baseline | Selected step 240 |
|---|---:|---:|
| New 48-case development reference match | 0/48 | 12/48 |
| New development answer-token loss | 6.172916 | 1.883521 |
| Historical 24-case lexical task check | 0/24 | 4/24 |
| New confirmation reference match | Not run | 6/24 |

| Family | Development | Confirmation |
|---|---:|---:|
| Named thank-you | 0/12 | 0/6 |
| Issue-title rewrite | 0/12 | 0/6 |
| Name/count/duration preservation | 0/12 | 0/6 |
| Status classification | 12/12 | 6/6 |

The development gate passed, so the 24 new confirmation requests were consumed.
They are now historical evidence, not a fresh test for another candidate. The
original 12-case confirmation remains untouched. All protected parameter hashes
and sampled block_04 features were unchanged. No production promotion or INT4
answer export occurred.

Raw confirmation answers show substantive content failures, beyond the scorer's
paraphrase limitation. A thumbnail-not-opening request became “Invoice does not
open.” Nadia's 137 reservation records in 37 minutes became “Noraia reviewed 22
profile records in 10 minutes.” These outputs suggest learned sentence templates
without reliable source copying; this is a diagnosis to test, not proof of the
underlying mechanism.

The historical 4/24 score is also misleading if read as four correct answers.
Two writing checks accepted invented content: the hello request produced “Thank
you, Evan, for fixing the archive.” The code-review thank-you switched the subject
to calendar and invented Evan. Those frozen lexical outcomes are preserved,
explicitly separated from grounded reference scores, and are not evidence that
free-form writing improved.

The unchanged raw report is
`reports/ember-direct-grounded-v2-34857841073.json`. Its compressed log chunks were
reassembled with index/count and SHA-256 verification. Artifact `10352838270`
contains the selected adapter, checkpoints, optimizer state and raw reports;
ZIP SHA-256 is
`45621ccf7f559d35a88eaef6b4015f642ec431cd3970acc1f5101aac702ed3c9`.
The metrics registry pins the raw report's hash and recounts every outcome,
including recomputing reference checks from generated text.

The next bounded investigation should measure copying independently of sentence
style: single supplied names, subjects and numbers, then their combinations,
with seen/unseen entities separated. Keep routing protected, use the consumed v2
cases only for diagnosis, and freeze a new confirmation cohort before scoring a
future candidate. More training alone is not yet justified by this result.
