# Ember v0.0.41 structural-subtype calibration

v0.0.40 established the strongest regression-safe baseline so far at 80/90, with perfect tool routing conditional on a canonical envelope and the historical floor preserved for every weak kind. Ten envelope failures remain; six recoveries are needed to reach the 86/90 global gate.

v0.0.41 stops treating each weak kind as one homogeneous prompt family. The v0.0.26 generator already defines multiple structural variants: short codes have 4- and 5-character forms; long codes have 4-4 and 3-5 forms; URLs have two-segment, one-segment mixed-case, and one-segment code forms; paths have result-suffix, plain-leaf, and numbered-leaf forms; mixed identifiers have lowercase, mixed-case, and uppercase stems.

For every `(kind, structural variant)` pair, v0.0.40's best-safe prompt is the mandatory fallback. Six deterministic synthetic values of exactly that variant are generated, disjoint from all held-out values. Two challengers are compared with the baseline: a typed exact-search request and an explicit query-literal request. A challenger may replace the fallback only if it beats it by at least 2/6 cases on both canonical envelope and correct tool name.

The 90 held-out cases are then measured once using the frozen subtype-specific selections. The same three gates remain: 4/4 exact v0.0.8 reference control, at least 86/90 global envelope and tool-name rate, and no regression below the v0.0.40 historical floors (`short_code` 8, `long_code` 7, `url` 7, `path` 9, `mixed` 9).

`slot_exact` remains diagnostic. This phase contains no optimizer, model write, GPU submission, promotion, deployment, or production integration.
