# Repair2 GGUF test build

Packaging only. No training run was started. The result is not release-qualified.

Hugging Face repo: `Jmiller18899/ember-repair2-gguf` (private, same visibility as Repair2).

| File | Size | Download |
| --- | --- | --- |
| `ember-repair2-q4_k_m.gguf` | 2,708,804,000 bytes (2.52 GiB) | https://huggingface.co/Jmiller18899/ember-repair2-gguf/resolve/main/ember-repair2-q4_k_m.gguf |
| `ember-repair2-q8_0.gguf` | 4,482,402,720 bytes (4.17 GiB) | https://huggingface.co/Jmiller18899/ember-repair2-gguf/resolve/main/ember-repair2-q8_0.gguf |

A private resolve URL requires a Hugging Face token. Q4_K_M is the iPad file. Q8_0 is under the 6 GiB cutoff and is published for a device that can spare the RAM.

Source: Repair2 adapter `daf938bba5d4e6b650ec9d34a2d3ac56706cf549` merged into `Qwen/Qwen3.5-4B` `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` as a text causal LM. Vision weights were not merged. llama.cpp `23b0202a189c44a54625aadcb37a946dd1d6278d`, `--no-mtp`. Job `6acaaecbfee2c9007018e030` on `cpu-xl` completed 2026-10-10 21:32:00Z–21:37:12Z.

Smoke test: `llama-completion`, context 2048, up to 48 new tokens, temperature 0, 8 threads, thinking prefilled closed.

- Q4 hello: `Hello!`
- Q4 arithmetic: `42`
- Q4 shorten: `The night clerk at the riverside ferry office must lock both gates after the last boat departs.`
- Q8 hello: `Hello!`

All four loads returned exit code 0.
