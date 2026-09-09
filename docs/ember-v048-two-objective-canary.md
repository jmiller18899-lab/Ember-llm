# Ember v0.0.48 two-objective CPU canary

v0.0.47 established one stable value-free generated-token envelope prefix across 18/18 successful sources and 13/13 leave-one-out matched controls. v0.0.48 is the first optimizer canary allowed to use that representation.

It trains only on synthetic values disjoint from the familiar 90-case battery and prior calibration values. Two objectives are measured independently:

1. **Envelope entry** — supervise only the first assistant `<|tool|>` token on short/long code subtypes.
2. **Value-boundary placement** — after the observed value-free prefix is supplied, supervise only the requested value continuation tokens. No familiar held-out value is used for learning.

The canary is bounded to 120 CPU optimizer steps at learning rate `1.2e-6`, with equal entry and placement loss weights.

Before any PASS is accepted, all of the following must hold: both synthetic development objectives improve by their configured margins, the model state actually changes, the four historical controls remain 4/4, the existing copy diagnostic is protected, and the familiar 90-case envelope battery does not regress below 84/90 or any protected kind/subtype floor.

The familiar 90 cases are evaluation-only and are not a fresh promotion set. A PASS does not authorize GPU training, promotion, deployment, or production integration. If the familiar battery reaches or exceeds 86/90, a fresh fully disjoint secondary 90-case confirmation battery is still required before any later GPU decision.

## Measured result

The first legitimate optimizer run was GitHub Actions run `34306129230`, job `102323084883`, on commit `c59d654c5d6117260074784ff8e145d981b8e910`. Two earlier attempts stopped before the first optimizer step because of value-continuation and launcher-path harness issues; neither changed model weights.

- Focused/inherited guards: **118 passed**.
- CPU optimizer steps completed: **120/120**.
- Template discovery: **6 successful sources**, **1 unique value-free boundary template**, 100% dominance, 7 prefix tokens.
- Entry development set: 24 fully disjoint synthetic cases.
- Placement development set: 24 fully disjoint synthetic cases.
- Model state changed: **yes**.
- Artifact: `ember-v048-canary-34306129230`, artifact ID `10086897118`, ZIP SHA256 `c947e9c90c538a48e266a7e198a907177d49f3498e90de7e9e26cd5b22b5f789`.

### Learning worked on both target objectives

| Objective | Before | After | Change |
| --- | ---: | ---: | ---: |
| Entry top-1 | 19/24 | 24/24 | **+5** |
| Entry mean loss | 4.7894 | 0.000871 | **99.98% reduction** |
| Placement exact top-1 | 1/24 | 9/24 | **+8** |
| Placement token top-1 | 90/171 (52.6%) | 147/171 (86.0%) | **+33.3 points** |
| Placement mean loss | 3.0862 | 1.0662 | **65.45% reduction** |

The optimizer itself behaved as a real learning run. Entry batch loss fell from 4.0101 at step 1 to 0.00120 at step 120; placement batch loss fell from 3.5510 to 0.4902. Therefore v0.0.48 is not a failure to learn the new objectives.

### Catastrophic protected-behavior regression

The candidate failed the canary because broad existing behavior collapsed:

- Familiar 90 canonical JSON: **84/90 → 36/90**.
- Familiar 90 correct tool: **84/90 → 33/90**.
- Exact v0.0.8 reference controls: **4/4 → 3/4**; `get_time` lost its valid envelope.
- Existing copy protection: **FAIL**.
- Expanded copy exact rate: **47.8% → 30.0%**.
- Expanded continuation top-1: **90.68% → 87.81%**.

Familiar correct-tool totals after the update were:

| Kind | Before | After |
| --- | ---: | ---: |
| short_code | 8/10 | **0/10** |
| long_code | 7/10 | **0/10** |
| mixed | 10/10 | **0/10** |
| url | 10/10 | **1/10** |
| path | 9/10 | **2/10** |
| model_id | 10/10 | **4/10** |
| digits | 10/10 | **8/10** |
| entity | 10/10 | **8/10** |
| expression | 10/10 | 10/10 |

Every protected short/long, URL, path, and mixed subtype floor failed. The canary decision was therefore **FAIL** even though every configured learning-gain check passed.

## Conclusion

v0.0.48 proves that both identified behaviors are learnable, but a plain 50/50 two-objective update causes severe catastrophic interference. The candidate must not be promoted, GPU-trained, deployed, or used as the base for the next experiment.

The next CPU experiment should restart from the untouched v0.0.31 step-479 source and focus on **protection against interference**, not stronger learning. A v0.0.49 canary should add broad synthetic envelope/copy replay or another explicit preservation term, use a substantially smaller effective update, and keep entry and placement gains separately measured. The familiar 90 and four historical controls must remain evaluation-only hard regression gates.
