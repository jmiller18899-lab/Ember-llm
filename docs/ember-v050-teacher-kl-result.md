# Ember v0.0.50 frozen-teacher KL minimal-delta CPU canary — result

Formal legitimate retry: `34312165618`, job `102340886299`, commit `4893bd98ae3beba82bd292d6718e2b898615df4d`.

The first v0.0.50 attempt (`34312062689`) was a lineage-compatibility harness error before model loading or any optimizer step and is not a scientific result. The retry imported the widened v0.0.49 replay compatibility patch, passed **131 tests**, and completed the full CPU canary.

Artifact: `ember-v050-teacher-kl-34312165618`, artifact ID `10088992534`, ZIP SHA256 `989e3b260ebac9bd3db01b867de3194154ab3cb7686c83c79abda61360334e06`.

## Minimal-delta synthetic selection

The student restarted from untouched v0.0.31 step 479. A frozen teacher copy supplied full-distribution KL preservation on fresh disjoint tool/copy trajectories. Candidate selection inspected only fresh synthetic entry/placement development sets and teacher-distillation drift at steps 10/20/30/40.

No checkpoint satisfied the full synthetic selection gate:

- step 10: entry +0, placement exact +0, placement token top-1 +0; teacher retention PASS.
- step 20: entry +1, placement exact +0, placement token top-1 +0; teacher retention PASS.
- step 30: entry +5, placement exact +0, placement token top-1 +0; teacher retention PASS.
- step 40: entry +6, placement exact +0, placement token top-1 +0; teacher retention PASS.

At step 40, teacher preservation remained extremely strong:

- tool distillation token top-1: **99.67%**;
- tool mean teacher KL: **0.010259**;
- copy distillation token top-1: **99.80%**;
- copy mean teacher KL: **0.000269**.

The final synthetic selector therefore correctly failed only the two placement-learning checks. Entry learning and all teacher-preservation checks passed.

## Final evaluation-only regression evidence

Even though the synthetic selector failed, the frozen step-40 state was evaluated for diagnosis after the candidate step was fixed. Retention is substantially better than v0.0.49 and fully clears the preservation gates:

- familiar canonical JSON: **85/90**;
- familiar correct tool: **85/90**;
- familiar regression gate: **PASS**;
- exact historical controls: **4/4 PASS**;
- existing historical copy-protection gate: **PASS**;
- model state changed: yes.

Correct-tool totals by kind:

| Kind | Result |
| --- | ---: |
| short_code | 8/10 |
| long_code | 8/10 |
| digits | 10/10 |
| model_id | 10/10 |
| url | 10/10 |
| path | 9/10 |
| entity | 10/10 |
| expression | 10/10 |
| mixed | 10/10 |

All historical protected subtype floors passed:

- short_code/len4 5;
- short_code/len5 3;
- long_code/4x4 6;
- long_code/3x5 2;
- url/one_mixed 4;
- url/two_segment 3;
- path/plain_leaf 3;
- mixed/upper 3.

The full historical copy-protection check passed every subcheck, including legacy/expanded exact-copy, continuation-top1, clean-stop, and case-retention checks.

## Conclusion

v0.0.50 is a **scientific FAIL because placement did not improve on fresh synthetic development data**, not because of retention. This is a major advance over v0.0.48/v0.0.49: frozen-teacher KL plus a much smaller update preserves Ember's historical behavior, including an 85/90 familiar score, while still teaching envelope entry (+6 synthetic top-1 cases).

The next experiment should keep the frozen-teacher preservation mechanism and restart again from untouched v0.0.31 step 479, but shift a small amount of optimization budget toward placement. Entry pressure can be reduced because entry improved strongly even under v0.0.50's conservative recipe. Candidate selection must remain synthetic-only and early-stop at the first checkpoint with real placement gain plus teacher preservation. Familiar 90/reference/copy gates remain final evaluation-only checks.

No v0.0.50 candidate is authorized for promotion, GPU training, deployment, or production integration.
