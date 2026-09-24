# Policy v3 candidates vs frozen policy v2

Run: GitHub Actions 35956600052 (`jobs/ember_policy_v3_candidates_eval.py`, components at `76f2b4c`, evaluator at `eb5b738`). All configurations use Repair2 weights `daf938bb`, and outputs are shared across configurations (484 generations). Eval only; nothing was frozen.

Components, each a new file (v2's pinned files are unchanged):
- **tool3** (`jobs/ember_arith_tool_v3.py`): masks label numbers (years, room/gate/bus/locker ids); adds leave/quit/cancel as removals, "goes into … at" as a start, "needs/requires N" as a duration.
- **draft3**: drafting gate also fires on "Text/Tell X that", "Let X know", "Write/Send a note to X".
- **ctx3**: context rule plus a "whose item" sentence.
- **missing**: asks for the text when a transformation request includes none.

## Scores
| Configuration | 72 exact | Temporal | Drafting | Promo v2 | Time held-out | Context held-out | **Suite v3** (dev) | **Fresh probes** |
|---|---|---|---|---|---|---|---|---|
| v2 (frozen, reproduced) | 72 | 8 | 7 | 200 | 180 | 20/24 | 164/192 | 20/36 |
| **full (all four)** | 72 | 8 | 7 | 200 | 180 | **24/24** | **185/192** | **28/36** |
| full − tool3 | 72 | 8 | 7 | 200 | 180 | 24 | 169 | 24 |
| full − draft3 | 72 | 8 | 7 | 200 | 180 | 24 | 184 | 25 |
| full − ctx3 | 72 | 8 | 7 | 200 | 180 | 20 | 183 | 28 |
| full − missing | 72 | 8 | 7 | 200 | 180 | 24 | 183 | 27 |

All five candidates pass every check: all v2 freeze checks, no exact regression, above v2 on suite v3 with no family lower, probes not lower, tool v3 never wrong when it fires, and no gate leakage on the 72 exact cases. **Best: full**, and every component contributes. Removing any one lowers suite v3, the probes or the context held-out.

**Contribution of each component** (full minus it):

| Component | Suite v3 | Probes | Context held-out |
|---|---|---|---|
| tool3 | +16 | +4 | 0 |
| draft3 | +1 | +3 | 0 |
| ctx3 | +2 | 0 | +4 |
| missing | +2 | +1 | 0 |

## Per family, v2 → full
Suite v3:
| Family | v2 | Full |
|---|---|---|
| arith_tool_shapes | 21 | 24/24 |
| arith_distractor | 8 | 16/16 |
| time_calc | 19 | 24/24 |
| clarification | 14 | 16/16 |
| drafting | 12 | 13/20 |
| context | 22 | 24/24 |

The rest are unchanged at full marks.

Probes:
| Family | v2 | Full |
|---|---|---|
| arith_distractor | 4 | 6/8 |
| arith_tool_shapes | 2 | 3/3 |
| time_calc | 2 | 3/3 |
| drafting | 3 | 6/8 |
| context | 3 | 3/6 |
| clarification | 6 | 7/8 |

- No output got worse: every changed answer went from wrong to right.
- The four "Whose item" answers in the context held-out that v2 got wrong are now right ("It is Eli's thermos …").

## What full still gets wrong (suite v3 7, probes 8)
- **Drafting perspective (5 in suite v3):** the new gate makes the model write a real message instead of echoing ("Hi Omar, I have his charger …"), but it keeps the third-person pronoun. The drafting rule does not convert his/her/their reliably for the "Text X that" form.
- **Shortening (2 in suite v3, 2 probes):** rewrites come back at the same length.
- **Context probes (3):** records that are not `key=value` with a question mark (arrow/colon logs, prose notes, "Tell me whose…") never reach the context gate, so ctx3 cannot help. The answers read "Ines's item; Friday."
- **Arithmetic outside the tool (2 probes):** "got off" and "opened … with 70 lamps" are not tool shapes, and the model gets them wrong.
- **Proofreading (1 probe):** "Their going to the park tomorow" came back with only the spelling fixed.

## Caveats
- Suite v3 is a development set: the components were designed on its failures, so 185/192 overstates generalisation.
- The fresh probes (20 → 28/36) are the better estimate. I wrote them after seeing the failures, in different wording, but they are not blind.
- The 36 probes are small. A candidate freeze should be read as "clearly better on these failure types", not as a measured general improvement.
