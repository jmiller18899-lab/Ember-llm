# Ember v0.0.51 placement-focused teacher-KL CPU canary

Formal run: `34313183490`, job `102343876586`, evaluated commit `c116015e2e37803ebb25f0806d3c799c17afbd70`.

The guard suite passed: **136 tests**. The run restarted from untouched Ember v0.0.31 step 479, used fresh synthetic target/distillation values, CPU only, and did not authorize GPU training, promotion, deployment, or production integration.

## Result

v0.0.51 is a scientific **FAIL**, not a harness failure.

The independent placement-probe check had already established that the placement metric is healthy: the control reproduced v0.0.48's published `1/24` exact and approximately 52% token-top1 baseline, the token gate was reachable with only a handful of flips, and the error was distributed rather than dominated by easy continuation positions.

v0.0.51 confirms that diagnosis directly:

- step 10: entry +1; placement exact +0; placement token gain +0.0 points;
- step 20: entry +3; placement exact +0; placement token gain **+3.05 points** — the token gate cleared;
- step 30: placement token gain **+4.27 points**;
- step 40: placement token gain **+6.10 points**, but tool-distribution top1 retention dropped below the selector floor;
- step 50: placement token gain **+6.71 points**; both tool and copy top1 retention were below selector floors;
- step 60: entry **+6**, placement exact **+1**, placement token gain **+7.32 points**.

At step 60 the placement-learning gates were finally all satisfied, but the preservation window had already closed. The final familiar correct-tool score was **79/90**, below the protected 84/90 floor. Frozen-teacher drift remained numerically moderate (`tool KL 0.03261`, `copy KL 0.001359`), showing that the relevant preservation failure is a small number of decision-boundary flips rather than broad distribution collapse.

Artifact: `ember-v051-placement-kl-34313183490`, artifact ID `10089399157`, ZIP SHA256 `bc88dcf8100fdb373aa94f34070ce59585ad651966fd2b02a6baae9adb5b644a`.

## Interpretation

The placement instrument is trustworthy and placement does respond once the effective update is large enough. The unresolved problem is now a narrow timing/tradeoff problem: placement token movement begins by step 20 while exact placement needs substantially more accumulated update, and source behavior begins leaving the protected envelope before exact gain arrives.

The next experiment should therefore avoid simply increasing steps or learning rate. It should seek more placement-efficient gradients inside the step-20-to-step-30 preservation window—for example by concentrating placement supervision on currently wrong target positions / first-token boundary positions while retaining frozen-teacher KL—then use the same fresh synthetic selector and final 84/90 + copy/reference hard gates.
