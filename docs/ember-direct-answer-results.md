# Direct-answer canary: insufficient improvement

The [completed CPU trial](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34565875001)
at `b2b6960b311ecb8e9824c26ab5fc9cd191337406` trained the final block and output
normalization while preserving the model parameters that supply routing features.

| Measurement | Baseline | Selected checkpoint |
| --- | ---: | ---: |
| Development answer-token loss | 6.1853 | 3.6641 |
| Task-aware checks on the old 24 cases | 0/24 | 2/24 |
| Unchanged historical quality rule | 0/24 | 1/24 |

Checkpoint selection used development loss alone, including the unchanged
baseline as a candidate. Step 80 was selected. At step 120, training loss kept
falling while development loss rose to 3.7886; this small trial already showed
limited generalization. All 72 training examples were used, and 3,212,800
parameters were trainable. The run completed all 120 steps.

The hashes of every frozen parameter matched before and after training.
Observed block_04 features also matched exactly. Router heads were unchanged.
The selected adapter was saved and reloaded before final answer evaluation.

## Why this is not ready

The task-aware score permits an exact one-word classification label. The
unchanged historical generic check requires at least two words; both scores
are retained so this exception does not silently change past evidence.

Even the small automatic gain overstates answer quality. One response passed
the lexical thank-you check but said “Thank you, Calculator” despite the prompt
giving no recipient name. Other answers substituted unrelated subjects, such as
rewriting a checkout-button problem as a search-box problem. The model learned
some response patterns without reliably preserving the requested content.

The development progress gate required at least three more task-check passes
and a 10% reduction in development loss. It failed. The 12 new direct-answer
confirmation requests were therefore **not scored**. There was no INT4 export
or production promotion.

The full [report](../reports/ember-direct-answer-canary.json) retains all
before/after generations, per-checkpoint losses, parameter hashes, training
order, and the selected adapter digest. Experimental adapters, optimizer/RNG
state, and logs remain in artifact `10186018606` from the completed run.

An earlier launch stopped on a missing evaluator import before any optimizer
updates. The dependency was corrected, the actual quality helpers were covered
by tests, and the completed run above supplies the reported evidence.

## Next direct-answer work

Expand training data that requires preserving names, subjects, facts, and
requested actions across genuinely different instructions. Include development
checks for invented entities and wrong-subject substitutions. Use the frozen
tool router independently from any answer-model experiment, and require gains
in actual generated content before spending the untouched confirmation.

The result does not establish that larger pretraining is necessary, nor that
more steps on the same 72 examples would solve the problem. It supports a
broader, better-grounded direct-answer learning experiment with explicit
generalization checks.
