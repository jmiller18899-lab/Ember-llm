# Pretrained candidate selection

This freezes one CPU comparison of Qwen3.5-0.8B and Qwen3.5-2B using Unsloth's
Q4_K_M GGUF conversions. Model revisions, file digests and llama.cpp b10964 are
pinned in benchmarks/qwen_candidates/models.json. No current Ember training is
launched, and no model is integrated or promoted by the benchmark.

Both candidates receive identical system/user messages and chat-template
settings: thinking disabled, greedy decoding, 96 output tokens, 2048-token
context, a single request at a time and two CPU threads. Each server runs inside
a two-CPU, 3 GiB Docker limit with swap disabled. This reserves nominal headroom
on a 4 GB host, but is not a test on the user's DigitalOcean instance or with its
other workloads. CPU identities, load time, full-response latency, decode rate,
server high-water RSS and cgroup peak memory are recorded.

There are 40 controlled cases: eight greetings, eight exact copies, eight field
extractions, eight short writing tasks and eight status classifications. All
five families must pass 8/8 under a conservative case/punctuation-normalized
reference check. It rejects changed names, numbers and extra content. This can
reject valid paraphrases and does not measure broad language quality. Eight
additional natural requests have no automatic pass label and require independent
review of raw answers for relevance, factual fidelity and invented details.

The smallest candidate clearing the controlled and manual checks is preferred
for a subsequent practical trial. Failure is preserved; there is no automatic
training trigger or production promotion. The suite includes previous Ember
failure examples, so this is model selection on development evidence, not blind
confirmation. Candidate totals must not be compared as a like-for-like trend to
Ember's earlier different 32/48/24-case suites.

Public sources:
- https://huggingface.co/Qwen/Qwen3.5-0.8B
- https://huggingface.co/Qwen/Qwen3.5-2B
- https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF
- https://huggingface.co/unsloth/Qwen3.5-2B-GGUF
- https://github.com/ggml-org/llama.cpp/releases/tag/b10964

## Initial measured result and bounded follow-up

Run 34863846441 completed on source 348d84635019142cca37c6168c11450de33462df.
The 0.8B candidate scored 30/40 and 2B scored 37/40. Both passed greetings and
exact copying at 8/8 each. Neither passed the strict automated gate. All original
raw responses are retained in reports/qwen-{0.8B,2B}-34863846441.json.

Manual inspection found the extraction prompt asking for the “subject” ambiguous:
its expected answer was the record category, but the grammatical subject is the
person's name. This case is unreliable evidence and is explicitly flagged, not
silently relabeled. Four 0.8B writing failures were reasonable paraphrases; its
other writing failures and two non-label classification outputs were real issues.
2B mislabeled two warning conditions as success. Both mishandled the natural
thank-you request in the unscored review set.

A single bounded prompt calibration now tests both unchanged weight files with
explicit warning/error definitions and an instruction to draft requested
thank-you messages rather than answer as the recipient. The ambiguous extraction
request is clarified. All 48 cases are already consumed, so calibration is not
fresh confirmation. This cheap inference-only customization precedes any proposal
to fine-tune weights. The original cases.json remains unchanged; calibration.json
records the changed system instruction and single clarified request.

## Final selection decision

Prefer **Qwen3.5-2B Q4_K_M with the original short system instruction** as the next
development starting point. No candidate satisfies the complete frozen gate,
so `qualified_model` remains null and production readiness remains false. No
fine-tuning, deployment, replacement of the custom Ember checkpoint, or new
Hugging Face model release occurred.

| Controlled family | Original 0.8B | Original 2B | Calibrated 0.8B | Calibrated 2B |
| --- | ---: | ---: | ---: | ---: |
| Greeting format | 8/8 | 8/8 | 8/8 | 8/8 |
| Exact copying | 8/8 | 8/8 | 4/8 | 5/8 |
| Extraction | 6/8 | 7/8 | 6/8 | 8/8 |
| Short writing | 2/8 | 8/8 | 3/8 | 7/8 |
| Status labels | 6/8 | 6/8 | 6/8 | 8/8 |
| Total | 30/40 | 37/40 | 27/40 | 36/40 |

Calibration run 34864387438 used source
7a91e329f8ac0a1630e3f2cb6824cc63040ff436. Its gains in 2B warning classification
came with three copy regressions. For instance, a repeat-exactly request received
“success.” Reject the calibration for both candidates. This was one bounded
prompt trial on consumed cases, not new model learning or confirmation.

Assistant review of the eight open requests scored original 0.8B at 6/8 and 2B
at 7/8; calibrated scores were 3/8 and 6/8. The original 2B answered a thank-you
drafting request with “You're welcome! Glad to hear Maya's calendar review was
helpful.” The calibrated response omitted Maya and the calendar. Natural drafting
therefore remains a concrete gap. Each review judgment and its rationale is
recorded separately in reports/ember-qwen-selection.json; it is not a blind
human benchmark and does not overwrite the frozen automatic outcomes.

### Resource measurements

| Original setup | Model file | Server high-water RSS | Cgroup charged peak | Median full response | Median decode rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.8B Q4_K_M | 533 MB | 1.69 GiB | 1.20 GiB | 0.93 s | 24.9 tokens/s |
| 2B Q4_K_M | 1.28 GB | 2.69 GiB | 1.51 GiB | 1.69 s | 17.6 tokens/s |

Both runs completed under a 3 GiB container limit with swap disabled and a
2-CPU quota. RSS counts resident mapped/shared pages, while the cgroup metric
counts charged memory; these measures need not match. Cache state and hosted
CPU variation affect results. The model preparation stage is outside the limit.
Short 2048-context, single-request text inference is all that was tested. This
supports a cautious 2B trial on a 4 GB machine but does not establish fit alongside
ClawAgent and a browser, long conversations, or the user's actual server speed.

The useful next customization targets are warning classification and natural
message drafting, with exact-copy regressions protected. Avoid putting task-
specific label rules into the global prompt again. Any further prompt or adapter
change needs new, unambiguous confirmation requests before release. Keep the
current custom Ember model archived; the selected candidate has a different
architecture and tokenizer and needs its own runtime integration.

All four raw reports were recovered from compressed log chunks, checksum
verified, recounted against their frozen input files and retained unchanged.
Source CPU CI passed for both benchmark revisions. reports/ember-qwen-selection.json
pins report hashes and records the provisional choice and rejected calibration.
