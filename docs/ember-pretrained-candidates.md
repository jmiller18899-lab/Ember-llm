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
