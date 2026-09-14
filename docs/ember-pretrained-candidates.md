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
