# Ember Qwen3.5-4B Baseline Review

Date: 2026-09-20  
Candidate: `Qwen/Qwen3.5-4B`  
Candidate revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`  
Evidence commit: `164af780483b7c84892b22e09a34cebe4cabcc43`

## Decision

**PASS — approve Qwen3.5-4B as the base for Ember's next LoRA experiment.**

Training has not started yet. The next adapter must be a new 4B LoRA; the 2B adapter is not shape-compatible.

## Quantitative results

| Check | Ember v3 (2B LoRA) | Qwen3.5-4B base | Delta |
|---|---:|---:|---:|
| Exact cases | 57/68 (83.8%) | 60/68 (88.2%) | +3 / +4.4 points |
| Rubric cases, strict manual review | not rescored here | 69/74 (93.2%) | — |
| Full evaluation cases | 142 | 142 | — |

The 4B base improved five exact families and regressed two classification families. All four classification regressions were capitalization-only outputs such as `Success` instead of required lowercase `success`.

## High-value improvements over Ember v3

- Refused to invent parcel, train, and package arrival times when duration was missing.
- Did not request private bank account details.
- Corrected the pen-total grounding case from 13 to 7.
- Corrected the ticket calculation to `42 - 13 + 6 = 35`.
- Fixed the Rowan greeting identity-adoption failure.
- Improved exact arithmetic, copy fidelity, and missing-information clarification.

## Strict manual-review failures to target

1. `fresh-natural-v2-03` — perspective error: says “his notebook” while addressing Mateo instead of “your notebook.”
2. `v3-confirm-06` — computes 9 blue and 9 yellow but does not explicitly state the required total of 18.
3. `v3-confirm-08` — says it does not know the ferry docking time without explicitly identifying missing duration or arrival information.
4. `v3-confirm-15` — shortening drops the supplied fact “at the office.”
5. `v3-confirm-21` — changes completed processing into future wording: “to be processed.”

## Exact failures to target

- Four classification answers used title case rather than the required lowercase label.
- One extraction case returned the complete record instead of only `F-704_x`.
- Three arithmetic cases remained wrong: expected 29, 21, and 48.

## Training gate

Proceed with a conservative Qwen3.5-4B LoRA only if the training set adds targeted examples for:

- exact lowercase status labels;
- value-only extraction;
- multi-step add/subtract arithmetic;
- explicit totals after component calculations;
- recipient perspective (`your` vs. `his/her`);
- preserving location and completion tense;
- explicit missing-information explanations.

Retain the existing grounding and clarification examples that produced the major 4B improvements.
