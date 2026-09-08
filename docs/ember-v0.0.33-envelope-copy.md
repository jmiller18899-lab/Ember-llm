# Ember v0.0.33 — moving the copy target into the argument slot

## The evidence this phase is built on

The v0.0.32 semantic gate ran the promoted v0.0.31 checkpoint. All four tool
calls produced a **correct envelope, a correct tool name and a correct argument
key**, and failed **exactly one check** — `arguments_grounded`:

| asked for | emitted |
| --- | --- |
| Detroit | `{"arguments":{"location":"Austin"},"name":"weather"}` |
| 347 × 28 | `{"arguments":{"expression":"337"},"name":"calculator"}` |
| latest Python release | `{"arguments":{"query":"recent WebAssembly announcements"},...}` |
| Tokyo | `{"arguments":{"timezone":"America/Anchorage"},"name":"get_time"}` |

The same checkpoint reproduces **43 of 90** held-out literal strings exactly,
with first-token top-1 at 89/90. The copy skill exists; it does not reach the
slot.

Every copy phase since v0.0.15 trained completions whose **first token is the
target** — prompt says `TARGET=<value>`, completion is the bare value. In a tool
call the copied span begins about fifteen tokens in, after
`{"arguments":{"location":"`. No phase has ever trained a copy that starts
anywhere but position zero.

## The one change

Completions become the envelope the checkpoint already emits, with the value in
the slot:

```
<|user|>   Ignore old=… and fallback=…. TARGET=Q7M4. Call weather for TARGET.
<|assistant|>
<|tool|>
{"arguments":{"location":"Q7M4"},"name":"weather"}
<|endoftext|>
```

`copy_token_weight` lands on the **value's tokens**; the fixed scaffolding gets
`envelope_token_weight` (1.0), so the objective still measures the copy and not
the punctuation. Key order matches the checkpoint's own output, so the envelope
being taught is the one it already produces.

**Held fixed:** the worst-k=2 boundary-focused hinge that produced 43/90, LR
1.2e-6, 600 steps, batch 8×2, every token weight, the band settings, the mining
policy, the curriculum's values, and baseline-protected `best.pt`. A test
compares the config against v0.0.31's key by key and fails on any drift.

Tools are paired with kinds semantically — `entity`→weather/location,
`expression`,`digits`→calculator/expression, `model_id`,`short_code`→
web_search/query, plus lookup/fetch_url/read_file — so routing, which the
checkpoint already gets right, is reinforced rather than confused. The first
three pairs are the ones the evaluation battery actually uses.

## The assumption that cannot be checked here

Whether the tokenizer merges across the value's boundaries inside a JSON string
is only answerable with the real tokenizer, which lives inside the checkpoint.
So `encode_envelope_row` verifies the prefix property at all three boundaries
and returns a **reason** instead of a mis-encoded row, `encode_envelope_rows`
counts the rejections, and the preflight fails below
`minimum_envelope_encodable_fraction` (0.90) and prints
`EMBER_V033_TRAIN_ENCODING` with the exact breakdown.

An untestable assumption surfaces on CPU for free rather than inside a paid run.
The tests drive this with a character-level tokenizer, where the arithmetic is
exact, plus one that rewrites the boundary token the way a real merge would.

## What gets measured

Three batteries per checkpoint (hence `eval_interval` 20 → 30):

| battery | purpose |
| --- | --- |
| legacy 9 | continuity back to v0.0.15 |
| expanded 90, bare value | **protects** the 43/90 already achieved |
| **envelope 90** | `envelope_slot_exact_rate` — the signal |

`envelope_slot_exact_rate` is the `arguments_grounded` check from the v0.0.32
gate, measured on 90 held-out cases instead of 4. Selection leads on
`envelope_slot_margin_health` — the continuous statistic, for the same reason
v0.0.28 stopped leading on a 90-case count — but **only after** the protection
gates pass, so a checkpoint that wins the slot by losing bare-value copying is
disqualified before its slot metrics are compared.

## Reading the result

- **Slot exact rises, bare value holds** → placement was the whole problem, and
  the v0.0.32 gate should be rerun against the new checkpoint.
- **Slot exact rises, bare value falls** → the capability is moving rather than
  generalising. Protection will have refused the checkpoint; the next question
  is a mixed curriculum, not more steps.
- **Neither moves and the encodable fraction is low** → the tokenizer merges at
  the slot boundary, and the envelope needs a different delimiter before any of
  this is testable.
- **Neither moves and the encodable fraction is high** → placement was not the
  limit. That would be the first real evidence for capacity or representation
  work rather than curriculum work.

## Run order

1. `audit` — free. The carried curriculum must still report `FORMAT_PARITY`.
2. `python -m pytest -q tests`.
3. `preflight` — CPU. **Read `EMBER_V033_TRAIN_ENCODING` first**; then
   `EMBER_V033_BASELINE_SLOT_EXACT`, which is the slot rate before any training.
4. `train` — one approved T4, only if the encodable fraction is healthy.
   Re-disarm `.github/ember-v033.trigger` afterwards.

v0.0.31 remains the promoted checkpoint throughout. Nothing here promotes.
