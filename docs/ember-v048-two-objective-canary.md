# Ember v0.0.48 two-objective CPU canary

v0.0.47 established one stable value-free generated-token envelope prefix across 18/18 successful sources and 13/13 leave-one-out matched controls. v0.0.48 is the first optimizer canary allowed to use that representation.

It trains only on synthetic values disjoint from the familiar 90-case battery and prior calibration values. Two objectives are measured independently:

1. **Envelope entry** — supervise only the first assistant `<|tool|>` token on short/long code subtypes.
2. **Value-boundary placement** — after the observed value-free prefix is supplied, supervise only the requested value continuation tokens. No familiar held-out value is used for learning.

The canary is bounded to 120 CPU optimizer steps at learning rate `1.2e-6`, with equal entry and placement loss weights.

Before any PASS is accepted, all of the following must hold: both synthetic development objectives improve by their configured margins, the model state actually changes, the four historical controls remain 4/4, the existing copy diagnostic is protected, and the familiar 90-case envelope battery does not regress below 84/90 or any protected kind/subtype floor.

The familiar 90 cases are evaluation-only and are not a fresh promotion set. A PASS does not authorize GPU training, promotion, deployment, or production integration. If the familiar battery reaches or exceeds 86/90, a fresh fully disjoint secondary 90-case confirmation battery is still required before any later GPU decision.
