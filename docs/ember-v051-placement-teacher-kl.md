# Ember v0.0.51 placement-focused frozen-teacher KL CPU canary

v0.0.50 established that frozen-teacher distribution preservation solves the catastrophic-retention problem: the final diagnostic state scored 85/90 on the familiar battery, passed every historical kind/subtype floor, kept references 4/4, and passed the full historical copy guard. Its only scientific failure was zero placement gain on fresh synthetic development cases.

v0.0.51 keeps the same preservation mechanism and restarts from untouched v0.0.31 step 479. It changes only the learning balance and uses entirely fresh target/distillation values disjoint from the familiar 90 and every prior v0.0.48-v0.0.50 synthetic value.

The placement objective receives 20% of total loss and batch size 8; entry is reduced to 5% because v0.0.50 learned entry strongly even under conservative pressure. Frozen-teacher preservation remains 75% of total loss (50% tool KL + 25% copy KL). Learning rate is `1.5e-7`, gradient clip `0.25`, and the run is bounded to at most 60 CPU optimizer steps.

Candidate selection remains synthetic-only and early. At steps 10/20/30/40/50/60 the candidate must gain at least one entry top-1 case, at least one placement exact case, at least two placement token-top1 percentage points, retain >=99% teacher-generated tokens on both tool/copy distillation cohorts, and keep mean teacher KL <=0.04 on each. The first passing synthetic checkpoint is frozen.

Only after candidate selection is frozen are the familiar 90-case battery, four historical reference controls, historical kind/subtype floors, and existing copy diagnostic evaluated. Final PASS still requires >=84/90 familiar JSON and correct tool, all historical floors, 4/4 references, and copy-protection PASS.

No GPU training, checkpoint promotion, deployment, or production integration is authorized by v0.0.51. If a candidate reaches >=86/90 on the familiar evaluation battery, it must then pass a fresh fully disjoint 90-case confirmation battery before any GPU decision.
