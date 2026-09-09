# Ember v0.0.50 frozen-teacher KL minimal-delta CPU canary

v0.0.49 proved that broad hard-token replay can preserve its own teacher-generated sequences above 96% token top-1 while still allowing enough probability-margin drift to reduce the familiar 90-case correct-tool score to 77/90. v0.0.50 therefore changes the preservation objective rather than adding more hard replay volume.

The source remains the untouched Ember v0.0.31 step-479 checkpoint. A frozen deep-copied teacher is created before any optimizer step. The student is penalized with full-distribution KL divergence against that frozen teacher at response-token positions on fresh synthetic tool and direct-copy trajectories. Entry and value-placement objectives remain separately measured.

All v0.0.50 template, train, development, and distillation values are generated from new deterministic seeds and are disjoint from the familiar 90-case battery and prior v0.0.48/v0.0.49 synthetic values.

The update is deliberately smaller than v0.0.49: learning rate `1e-7`, at most 40 CPU steps, gradient clip `0.25`, with 80% of the loss assigned to teacher-distribution preservation (55% tool KL + 25% copy KL) and only 10% each to entry and placement.

Candidate selection is minimal-delta and does not inspect the familiar 90, reference controls, or historical copy diagnostic. At steps 10/20/30/40, the run evaluates only fresh synthetic development objectives and teacher-distillation drift. It stops at the first checkpoint that gains at least one entry top-1 case, one placement exact case, at least three placement token-top1 percentage points, retains >=99% of teacher-generated distillation tokens for both tool and copy cohorts, and keeps mean teacher KL <=0.08 on both cohorts.

Only after the candidate step is frozen are the familiar 90-case battery, four exact historical controls, protected kind/subtype floors, and existing copy diagnostic evaluated. Final PASS still requires >=84/90 familiar JSON and correct tool, every historical kind/subtype floor, 4/4 references, and copy-protection PASS.

A PASS is CPU canary evidence only. It does not authorize GPU training, promotion, deployment, or production integration. If a later candidate reaches >=86/90 on the familiar battery, a fresh fully disjoint 90-case confirmation battery is still required before any GPU decision.
