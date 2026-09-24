# Repair2 vs frozen Consolidation1: original 72 exact cases

HF job `6ab4926b52d0dbd7f1d88d6c` (`jobs/ember_4b_repair2_exact_diff.py`, eval only, no weights written).
Consolidation1 `@62e5b58`, Repair2 `@daf938bb`, benchmark `@9b080364`. Same base model, same promotion system prompt, greedy decoding.
It reproduces the promotion scores exactly: **Consolidation1 71/72, Repair2 70/72**. Rule leakage: 0.

| Case | Gold | Consolidation1 | Repair2 | Note |
|---|---|---|---|---|
| `arith-pack-13` "4 sealed bundles with 9 screws in each bundle and 4 extra screws" | 40 | 40 ✅ | **45** ❌ | Only case Repair2 lost |
| `time-14` "10:50 AM + 75 minutes … Answer as H:MM AM." | 12:05 AM | 12:05 PM ❌ | 12:05 PM ❌ | Both fail; **the gold answer looks wrong** |

## arith-pack-13 is a near miss with a recurring cause
- Repair2 answers 45 = 4×9+9 = (4+1)×9: it reads the 4 extra screws as one more full bundle.
- The frozen model's only multiply miss on promotion v2 was the same error: `mult-06`, 4×11+9 gave 55 = (4+1)×11.
- It is a near miss. Scoring the correct answer "40" token by token, the model gets "4" right, then ranks "0" third, 0.875 nats behind its top choice "5". The correct answer's log-probability is −1.96 under Repair2 and −1.32 under Consolidation1.
- The repair (`jobs/ember_4b_repair3_train.py`) therefore targets only multiply-then-add with loose items, and leaves time reasoning alone.

## time-14 benchmark gold
10:50 AM + 75 min = **12:05 PM**. Both models give 12:05 PM. The gold "12:05 AM" appears to copy the `H:MM AM` format hint.
This case should be corrected in the benchmark, not trained toward. Until it is corrected, 71/72 is the effective ceiling for any correct model.
