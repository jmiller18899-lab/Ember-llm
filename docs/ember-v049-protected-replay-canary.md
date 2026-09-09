# Ember v0.0.49 protected replay CPU canary

v0.0.49 restarted from the untouched Ember v0.0.31 step-479 checkpoint after v0.0.48 proved that entry and placement learning are possible but caused catastrophic interference.

The v0.0.49 update was deliberately smaller and preservation-dominant: 80 CPU optimizer steps, learning rate 3e-7, gradient clip 0.5, with 15% entry loss, 15% placement loss, 50% source-generated tool-envelope replay, and 20% source-generated direct-copy replay. All target and replay values were synthetic and disjoint from the familiar 90-case battery and earlier calibration values. No GPU, promotion, deployment, or production integration was authorized.

## Formal result

The first legitimate optimizer run was GitHub Actions run `34310937571`, job `102337282047`, on commit `54c7e7cc7ee13e03c729164504b8679c628f8835`.

Two earlier v0.0.49 attempts stopped before optimization: the first had too few successful synthetic template sources, and the second had too few valid long-code tool-replay sources. Neither changed model weights. The final run widened only the synthetic discovery/replay pools while retaining the same quality thresholds.

- Inherited/focused guards: **125 passed**.
- CPU optimizer steps: **80/80**.
- Stable value-free template sources: **11 successful sources**, one unique template, 100% dominance, 7 prefix tokens.
- Tool replay: **118** valid source envelopes spanning all nine kinds.
- Copy replay: **48** source direct-response sequences spanning all nine kinds.
- Artifact: `ember-v049-protected-34310937571`, artifact ID `10088576972`, ZIP SHA256 `f520d051b0011f54458de9e2446c4a00151f8fcc4440b1650da5f54d6fd0c115`.

## Target learning still worked

| Objective | Before | After | Change |
| --- | ---: | ---: | ---: |
| Entry top-1 | 18/24 | 24/24 | **+6** |
| Entry mean loss | 5.1308 | 0.0841 | **98.36% reduction** |
| Placement exact top-1 | 0/24 | 2/24 | **+2** |
| Placement token top-1 | 92/172 (53.5%) | 120/172 (69.8%) | **+16.3 points** |
| Placement mean loss | 3.1154 | 2.0415 | **34.47% reduction** |

Every configured entry and placement learning-gain check passed.

## Replay materially reduced interference

| Preservation measure | Before | After |
| --- | ---: | ---: |
| Source tool-replay token top-1 | 100% | **97.79%** |
| Source copy-replay token top-1 | 100% | **96.73%** |
| Historical reference controls | 4/4 | **4/4** |

Both explicit replay-retention gates (minimum 95%) passed. This is a large improvement over v0.0.48, which broadly erased tool-envelope behavior.

## But protected behavior still regressed too far

The familiar 90-case envelope battery fell from **84/90 to 77/90** for both canonical JSON and correct tool name, below the hard 84/90 preservation floor.

| Kind | Before | After |
| --- | ---: | ---: |
| short_code | 8/10 | **6/10** |
| long_code | 7/10 | **4/10** |
| digits | 10/10 | 10/10 |
| model_id | 10/10 | **8/10** |
| url | 10/10 | **9/10** |
| path | 9/10 | **10/10** |
| entity | 10/10 | 10/10 |
| expression | 10/10 | 10/10 |
| mixed | 10/10 | 10/10 |

Protected subtype floors failed for short_code/len4, long_code/4x4, and url/one_mixed. Long_code/3x5, short_code/len5, URL/two_segment, mixed/upper, and path/plain_leaf remained at or above their floors.

The existing copy protection also failed despite the source-copy replay gate passing:

- legacy exact copy rate: **66.7% → 55.6%**;
- legacy continuation top-1: **94.20% → 92.75%**;
- expanded exact copy rate: **47.78% → 38.89%**;
- expanded continuation top-1: **90.68% → 89.59%**.

Thus a 95% teacher-replay retention rate is not strong enough to protect all boundary-sensitive legacy behaviors.

## Conclusion

v0.0.49 is a scientific **FAIL**. It demonstrates that broad source replay can sharply reduce catastrophic interference while retaining useful entry and placement learning, but the remaining parameter movement is still too large or insufficiently targeted. The candidate must not be promoted, GPU-trained, deployed, or used as the next source checkpoint.

The next experiment should again restart from untouched v0.0.31 step 479 and focus on tighter preservation. A v0.0.50 canary should test either a substantially smaller/shorter update with stronger replay weighting, or a parameter-local update strategy that limits which weights may move. The 84/90 familiar floor, 4/4 references, copy protection, and replay-retention gates must remain hard constraints rather than being relaxed.
