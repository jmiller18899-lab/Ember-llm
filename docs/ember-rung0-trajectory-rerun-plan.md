# Rung-0 trajectory re-run: full guards from step 12

## Why

The verified trajectory (`reports/ember-rung0-trajectory-verified.json`, run
`34369464055`) observed short codes and placement at every step, but ran the
full familiar/copy/reference/KL battery at only two states: step 1 and step 40.

That leaves the onset of the copy-protection failure unmeasured. The result doc
records it as failing at step 40 and holding at step 1, and correctly declines to
call step 12 preserved. Between those two states the battery was never run, so
"the first preservation failure is the short-code loss at step 13" is an
observation about short codes only — a copy or familiar regression could have
occurred earlier and gone unrecorded.

## What this run changes

Nothing in the optimization. Same pinned v0.0.31 step-479 source, same published
ladder config (SHA-pinned), same seed, same 40-step batch schedule, same LR 1e-7
and 0.20/0.55/0.25 weights. The only change is which states are measured:

    --full-eval-steps 12,13,23

giving full batteries at steps 1, 12, 13, 23 and 40.

Step 12 is the last state before the first short-code loss, step 13 is that loss,
and step 23 is the second one. Together they answer: did copy protection and the
familiar battery survive to step 12, and does the step-13 short-code loss coincide
with any other preservation failure?

## Why the added measurements cannot change the trajectory

Every probe runs inside `observation()`, which restores the torch and Python RNG
state, restores training mode, and raises if the model state digest moved. The
batch schedule is drawn once, before any evaluation. The run then re-asserts the
published endpoint: placement 2/24 and 95/170, familiar 82/90, tool KL
0.00075252, copy KL 0.00025024, and `expanded_cases_retained` as the sole copy
failure. If the extra batteries perturbed anything, that reproduction check fails
and the run reports `REPLICATION_MISMATCH`.

## Cost

The verified run took 1221 s with three batteries (baseline, step 1, step 40).
Three more are added, so the fixed CPU bound is raised from 1800 s to 3000 s and
the job timeout from 40 to 60 minutes. The bound is still enforced by SIGALRM.

## Scope

CPU only. No GPU training, no checkpoint export, no promotion, no deployment, no
integration. The source checkpoint is restored and hash-verified at exit, as
before.
