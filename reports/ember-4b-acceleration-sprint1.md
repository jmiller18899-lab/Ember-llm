# Ember 4B acceleration sprint 1

Starting checkpoint: `Jmiller18899/ember-qwen3.5-4b-sft-v1-repair1b`.

Verified entry gate:
- historical exact suite: 68/71 after targeted repair
- zero exact regressions in repair1b
- all five previously protected gains preserved
- extraction case v2-confirmation-extraction-0026 repaired
- two known arithmetic exact failures remain
- production promotion remains disabled

## Parallel lanes

1. Arithmetic/reasoning: mixed-operation composition and multiplication-plus-loose-item failures.
2. Grounding/clarification: ask for the specific missing input, answer when sufficient facts exist, never invent completed actions.
3. Conversation/drafting: preserve supplied facts, sender/recipient perspective, concise rewrites without dropping locations.

Each lane must use disjoint training prompts and a shared fresh holdout. A lane is rejected if it loses any currently correct exact case. No lane replaces repair1b automatically.

## Sprint gate

Candidate comparison requires:
- replay of all 71 exact cases;
- zero regression from repair1b exact passes;
- preservation of the five historical protected gains plus the repaired extraction case;
- fresh exact and rubric holdout results saved separately;
- human review for rubric cases;
- candidate checkpoint preserved before acceptance decision.

Only candidates that pass the gate may enter consolidation testing. Consolidation and deployment are separate decisions.
