# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","accelerate==1.15.0","huggingface-hub==1.31.0"]
# ///
"""Merge the Repair2 LoRA into Qwen3.5-4B and publish an iPad GGUF test build.

Packaging only. This script has no trainer, no optimizer, and submits one CPU
job. It must not be pointed at a GPU flavor.
"""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

BASE = "Qwen/Qwen3.5-4B"
BASE_REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ADAPTER = "Jmiller18899/ember-qwen3.5-4b-repair2"
ADAPTER_REV = "daf938bba5d4e6b650ec9d34a2d3ac56706cf549"
OUT = "Jmiller18899/ember-repair2-gguf"
LLAMA_CPP_REV = "23b0202a189c44a54625aadcb37a946dd1d6278d"
FLAVOR = "cpu-xl"  # 16 vCPU, 124 GB RAM, 1000 GB disk. Not a training GPU.
TIMEOUT = "3h"
Q8_MAX_BYTES = 6 * 1024 ** 3
Q4_NAME = "ember-repair2-q4_k_m.gguf"
Q8_NAME = "ember-repair2-q8_0.gguf"
THINKING_OFF_OLD = (
    "{%- if enable_thinking is defined and enable_thinking is false %}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "    {%- else %}\n"
    "        {{- '<think>\\n' }}\n"
    "    {%- endif %}"
)
THINKING_OFF_NEW = (
    "{%- if enable_thinking is defined and enable_thinking is true %}\n"
    "        {{- '<think>\\n' }}\n"
    "    {%- else %}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "    {%- endif %}"
)
SYSTEM = "You are Ember. Answer directly and briefly."
PROMPTS = (
    ("hello", "Say hello in one short sentence."),
    ("arithmetic", "What is 17 + 25? Reply with the number only."),
    ("shorten", "Make this shorter without changing the facts: The night clerk at the riverside ferry office has to lock both gates after the last boat leaves."),
)


def thinking_off(template):
    if THINKING_OFF_OLD not in template:
        raise RuntimeError("Chat template is missing the expected thinking branch")
    patched = template.replace(THINKING_OFF_OLD, THINKING_OFF_NEW, 1)
    if "enable_thinking is true" not in patched:
        raise RuntimeError("Thinking-off patch did not apply")
    return patched


def chat_prompt(user):
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def check():
    source = Path(__file__).read_text()
    banned = ("Training" + "Arguments", "Tra" + "iner(", "l4" + "x1", "SFT" + "Trainer")
    found = [item for item in banned if item in source]
    if found:
        raise SystemExit(f"Packaging script contains training markers: {found}")
    if FLAVOR != "cpu-xl" or FLAVOR.startswith(("t4", "a10", "a100", "l4", "h100", "h200")):
        raise SystemExit(f"Refusing flavor {FLAVOR}")
    sample = "prefix\n" + THINKING_OFF_OLD + "\nsuffix\n"
    patched = thinking_off(sample)
    if THINKING_OFF_OLD in patched or "enable_thinking is false" in patched.split("prefix", 1)[-1]:
        raise SystemExit("Thinking-off patch kept the old default")
    if "17 + 25" not in chat_prompt(PROMPTS[1][1]):
        raise SystemExit("Prompt builder dropped the user text")
    print("GGUF_PACKAGING_CHECK_OK", flush=True)


def sh(cmd, cwd=None, timeout=None, env=None):
    print("RUN", " ".join(str(part) for part in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, timeout=timeout, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(str(part) for part in cmd)}")


def bin_env(binary):
    env = os.environ.copy()
    lib = str(Path(binary).resolve().parent)
    env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def find_bin(dest, name):
    matches = [path for path in Path(dest).rglob(name) if path.is_file() and path.stat().st_mode & 0o111]
    matches.sort(key=lambda path: (0 if path.parent.name == "bin" else 1, len(path.parts)))
    if not matches:
        raise RuntimeError(f"{name} was not produced")
    return matches[0]


def redact(text):
    import re
    return re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", text or "")


def launch():
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    who = api.whoami()
    if who.get("name") != "Jmiller18899":
        raise SystemExit("Wrong Hugging Face account")
    parent = api.model_info(ADAPTER, revision=ADAPTER_REV)
    private = bool(parent.private)
    if api.repo_exists(OUT):
        info = api.model_info(OUT)
        if bool(info.private) != private:
            raise SystemExit("Existing GGUF repo visibility does not match Repair2")
        files = set(api.list_repo_files(OUT))
        if "packaging-complete.json" in files or any(name.endswith(".gguf") for name in files):
            raise SystemExit("GGUF repo already has a package; not submitting another job")
        if "packaging-job.json" in files:
            prior = json.loads(Path(hf_hub_download(OUT, "packaging-job.json", token=token)).read_text())
            stage = getattr(api.inspect_job(job_id=prior["job_id"]).status, "stage", None)
            if stage not in ("ERROR", "CANCELED", "CANCELLED"):
                raise SystemExit(f"Packaging job already {stage}")
    else:
        api.create_repo(OUT, repo_type="model", private=private, exist_ok=False)
    job = api.run_uv_job(
        script=str(Path(__file__).resolve()),
        python="3.11",
        flavor=FLAVOR,
        timeout=TIMEOUT,
        env={"TOKENIZERS_PARALLELISM": "false", "PACKAGING_ONLY": "1"},
        secrets={"HF_TOKEN": token},
        labels={"name": "ember-repair2-gguf-packaging"},
    )
    receipt = {
        "job_id": job.id,
        "url": job.url,
        "flavor": FLAVOR,
        "timeout": TIMEOUT,
        "training": False,
        "purpose": "merge Repair2 and publish an iPad GGUF test build",
        "adapter": f"{ADAPTER}@{ADAPTER_REV}",
        "base": f"{BASE}@{BASE_REV}",
        "llama_cpp": LLAMA_CPP_REV,
        "private": private,
    }
    print("GGUF_JOB_SUBMITTED", json.dumps(receipt), flush=True)
    api.upload_file(
        repo_id=OUT,
        path_in_repo="packaging-job.json",
        path_or_fileobj=json.dumps(receipt, indent=2).encode(),
        commit_message="Record Repair2 GGUF packaging job",
    )


def stage(name):
    print(f"GGUF_STAGE {name}", flush=True)


def merge(work):
    import gc
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    from huggingface_hub import snapshot_download

    stage("download")
    base_dir = Path(snapshot_download(BASE, revision=BASE_REV, local_dir=work / "base"))
    adapter_dir = Path(snapshot_download(
        ADAPTER, revision=ADAPTER_REV, local_dir=work / "adapter",
        allow_patterns=["adapter_config.json", "adapter_model.safetensors", "chat_template.jinja",
                        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
    ))
    template_path = adapter_dir / "chat_template.jinja"
    if not template_path.exists():
        template_path = base_dir / "chat_template.jinja"
    patched = thinking_off(template_path.read_text())
    stage("merge")
    torch.set_num_threads(os.cpu_count() or 4)
    base = Qwen3_5ForCausalLM.from_pretrained(
        base_dir, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True,
        key_mapping={r"^model.language_model\.": "model."},
    )
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    out = work / "merged"
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    tok.chat_template = patched
    tok.save_pretrained(out)
    (out / "chat_template.jinja").write_text(patched)
    del merged, base
    gc.collect()
    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text())
    print("GGUF_MERGED_CONFIG", json.dumps({
        "architectures": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "mtp_num_hidden_layers": cfg.get("mtp_num_hidden_layers"),
        "tie_word_embeddings": cfg.get("tie_word_embeddings"),
        "vocab_size": cfg.get("vocab_size"),
    }), flush=True)
    if cfg.get("architectures") != ["Qwen3_5ForCausalLM"]:
        raise RuntimeError(f"Merged config architectures are {cfg.get('architectures')}")
    if cfg.get("num_hidden_layers") != 32:
        raise RuntimeError(f"Merged config has {cfg.get('num_hidden_layers')} layers")
    if "enable_thinking is true" not in (out / "chat_template.jinja").read_text():
        raise RuntimeError("Saved chat template does not default thinking off")
    shutil.rmtree(base_dir)
    shutil.rmtree(adapter_dir)
    return out


def build_llama(work):
    stage("llama.cpp")
    sh(["apt-get", "update"])
    sh(["apt-get", "install", "-y", "--no-install-recommends", "build-essential", "cmake", "git", "ca-certificates"])
    dest = work / "llama.cpp"
    if not dest.exists():
        sh(["git", "init", str(dest)])
        sh(["git", "remote", "add", "origin", "https://github.com/ggml-org/llama.cpp"], cwd=dest)
        sh(["git", "fetch", "--depth", "1", "origin", LLAMA_CPP_REV], cwd=dest)
        sh(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=dest)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dest, text=True).strip()
    if head != LLAMA_CPP_REV:
        raise RuntimeError(f"llama.cpp HEAD {head} != {LLAMA_CPP_REV}")
    sh(["cmake", "-S", ".", "-B", "build", "-DGGML_NATIVE=ON", "-DLLAMA_CURL=OFF",
        "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=OFF"], cwd=dest)
    sh(["cmake", "--build", "build", "-j", str(os.cpu_count() or 4),
        "--target", "llama-quantize", "llama-completion"], cwd=dest)
    return dest, find_bin(dest, "llama-quantize"), find_bin(dest, "llama-completion")


def convert_and_quantize(work, merged, llama_dir, quantize):
    stage("convert")
    f16 = work / "ember-repair2-f16.gguf"
    sh([sys.executable, str(llama_dir / "convert_hf_to_gguf.py"), str(merged),
        "--outfile", str(f16), "--outtype", "f16", "--no-mtp"], cwd=llama_dir)
    shutil.rmtree(merged)
    outputs = {}
    for name, qtype in ((Q4_NAME, "Q4_K_M"), (Q8_NAME, "Q8_0")):
        stage(f"quantize {qtype}")
        dest = work / name
        sh([str(quantize), str(f16), str(dest), qtype], env=bin_env(quantize))
        outputs[name] = dest.stat().st_size
        print(f"GGUF_QUANT {qtype} {outputs[name]}", flush=True)
    f16.unlink()
    return outputs


def visible_reply(text):
    reply = text.split("</think>")[-1] if "</think>" in text else text
    return reply.replace("[end of text]", "").strip()


def smoke(completion, model, cases):
    threads = "8"
    results = []
    for name, user in cases:
        stage(f"smoke {model.name} {name}")
        prompt = chat_prompt(user)
        cmd = [str(completion), "-m", str(model), "-c", "2048", "-n", "48", "-t", threads,
               "--temp", "0", "--top-k", "1", "-no-cnv", "--single-turn", "--no-warmup",
               "--reverse-prompt", "<|im_end|>", "-p", prompt]
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True, timeout=900, env=bin_env(completion))
            answer = redact(proc.stdout.strip())
            code = proc.returncode
            err = redact(proc.stderr[-800:])
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            raw_out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            raw_err = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            answer = redact(raw_out.strip())
            code = None
            err = redact(raw_err[-800:])
            timed_out = True
        reply = visible_reply(answer)
        result = {
            "case": name,
            "model": model.name,
            "returncode": code,
            "timed_out": timed_out,
            "load_ok": code == 0,
            "answer": answer[:1200],
            "reply": reply[:800],
            "stderr_tail": err,
        }
        if name == "arithmetic":
            result["contains_42"] = "42" in reply or "42" in answer
        print("GGUF_SMOKE", json.dumps({k: result[k] for k in result if k != "stderr_tail"}), flush=True)
        results.append(result)
    return results


def readme(private, sizes, q4_smoke, q8_smoke, published):
    def gb(n):
        return f"{n / (1024 ** 3):.2f} GiB"
    lines = [
        "---",
        "license: apache-2.0",
        "base_model: Qwen/Qwen3.5-4B",
        "library_name: gguf",
        "pipeline_tag: text-generation",
        "tags:",
        "- gguf",
        "- qwen3.5",
        "- ember",
        "- test-build",
        "---",
        "",
        "# Ember Repair2 GGUF",
        "",
        "**TEST BUILD. Not release-qualified.** This is a packaging of the current Repair2 reference so it can be tried on an iPad. It is not a promoted Ember release, and it is not WR8 or WR9.",
        "",
        "Text only. The Qwen3.5 vision tower was not merged. Repair2 is a LoRA on the text model.",
        "",
        "## Files",
        "",
    ]
    for name in published:
        lines.append(f"- `{name}` — {gb(sizes[name])} — https://huggingface.co/{OUT}/resolve/main/{name}")
    if Q8_NAME not in published and Q8_NAME in sizes:
        lines.append(f"- Q8_0 was built ({gb(sizes[Q8_NAME])}) and left out of this repo because it is above the {Q8_MAX_BYTES // (1024 ** 3)} GiB iPad cutoff.")
    lines += [
        "",
        f"Repo visibility: **{'private' if private else 'public'}**. A private file still uses the resolve URL above and requires a Hugging Face token.",
        "",
        "## iPad settings",
        "",
        "Use **Q4_K_M** in PocketPal or LLM Farm. Q8_0 is only for a device that can spare the extra RAM.",
        "",
        "- Context length: 4096. Use 8192 only when the device still has free memory. Do not request the full 262144 training context.",
        "- Temperature: 0, to match the greedy check used for this test build.",
        "- Stop string: `<|im_end|>`",
        "- Thinking: off. This test build's embedded chat template closes an empty think block unless `enable_thinking` is explicitly true.",
        "",
        "## Chat template",
        "",
        "```",
        "<|im_start|>system",
        "You are Ember. Answer directly and briefly.<|im_end|>",
        "<|im_start|>user",
        "{message}<|im_end|>",
        "<|im_start|>assistant",
        "<think>",
        "",
        "</think>",
        "",
        "```",
        "",
        "The model should continue with the answer and stop at `<|im_end|>`.",
        "",
        "## Provenance",
        "",
        f"- Adapter: `{ADAPTER}` revision `{ADAPTER_REV}`",
        f"- Base: `{BASE}` revision `{BASE_REV}`",
        "- Merge: `Qwen3_5ForCausalLM` with `model.language_model.` mapped to `model.`, then PEFT `merge_and_unload`.",
        f"- Converter: llama.cpp `{LLAMA_CPP_REV}` with `--no-mtp` and `--outtype f16`, then `llama-quantize`.",
        "",
        "## Smoke test",
        "",
        "llama-completion, context 2048, up to 48 new tokens, temperature 0, 8 threads, thinking prefilled closed. These prompts are not evaluation-set items.",
        "",
    ]
    for result in q4_smoke + q8_smoke:
        lines.append(f"### {result['model']} / {result['case']}")
        lines.append("")
        lines.append(f"Load ok: {result['load_ok']}. Return code: {result['returncode']}. Timed out: {result.get('timed_out', False)}.")
        if "contains_42" in result:
            lines.append(f"Contains 42: {result['contains_42']}.")
        lines.append("")
        lines.append("```")
        lines.append(result.get("reply") or result["answer"] or "(empty)")
        lines.append("```")
        lines.append("")
    lines.append("This smoke test only checks that the GGUF loads and produces a short answer. It is not a quality gate.")
    return "\n".join(lines) + "\n"


def upload(private, sizes, q4_smoke, q8_smoke, include_q8):
    from huggingface_hub import HfApi
    stage("upload")
    api = HfApi(token=os.environ["HF_TOKEN"])
    if api.whoami().get("name") != "Jmiller18899":
        raise SystemExit("Wrong Hugging Face account")
    published = [Q4_NAME]
    if include_q8 and sizes[Q8_NAME] <= Q8_MAX_BYTES:
        published.append(Q8_NAME)
    else:
        print(f"GGUF_SKIP_Q8 {sizes[Q8_NAME]} include={include_q8}", flush=True)
    root = Path("/tmp/ember-repair2-gguf")
    for name in published:
        api.upload_file(repo_id=OUT, path_in_repo=name, path_or_fileobj=str(root / name),
                        commit_message=f"Add Repair2 {name} test build")
    card = readme(private, sizes, q4_smoke, q8_smoke, published)
    api.upload_file(repo_id=OUT, path_in_repo="README.md", path_or_fileobj=card.encode(),
                    commit_message="Document Repair2 GGUF test build")
    report = {
        "training": False,
        "release_qualified": False,
        "private": private,
        "published": published,
        "sizes": sizes,
        "q4_smoke": q4_smoke,
        "q8_smoke": q8_smoke,
        "adapter": ADAPTER_REV,
        "base": BASE_REV,
        "llama_cpp": LLAMA_CPP_REV,
        "urls": {name: f"https://huggingface.co/{OUT}/resolve/main/{name}" for name in published},
    }
    api.upload_file(repo_id=OUT, path_in_repo="packaging-complete.json",
                    path_or_fileobj=json.dumps(report, indent=2).encode(),
                    commit_message="Record Repair2 GGUF smoke test")
    print("GGUF_PACKAGING_COMPLETE", json.dumps(report), flush=True)


def package():
    accel = os.environ.get("ACCELERATOR", "")
    print(f"GGUF_ACCELERATOR {accel!r}", flush=True)
    gpu_marks = ("t4", "a10", "a100", "l4x", "h100", "h200", "rtx")
    if any(mark in accel.lower() for mark in gpu_marks):
        raise SystemExit(f"This packaging job must not run on a GPU ({accel})")
    print("GGUF_PACKAGING_NOT_TRAINING", flush=True)
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    private = bool(api.model_info(OUT).private)
    work = Path("/tmp/ember-repair2-gguf")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    llama_dir, quantize, completion = build_llama(work)
    merged = merge(work)
    sizes = convert_and_quantize(work, merged, llama_dir, quantize)
    q4 = smoke(completion, work / Q4_NAME, PROMPTS)
    required = [item for item in q4 if item["case"] in ("hello", "arithmetic")]
    if not all(item["load_ok"] and item.get("reply") for item in required):
        raise SystemExit("Q4_K_M did not load and answer the short prompts; not uploading")
    if not any(item["case"] == "arithmetic" and item.get("contains_42") for item in q4):
        raise SystemExit("Q4_K_M arithmetic smoke did not contain 42; not uploading")
    include_q8 = sizes[Q8_NAME] <= Q8_MAX_BYTES
    q8 = smoke(completion, work / Q8_NAME, (PROMPTS[0],)) if include_q8 else []
    if q8 and not (q8[0]["load_ok"] and q8[0].get("reply")):
        print("GGUF_Q8_SMOKE_FAILED", flush=True)
        include_q8 = False
    upload(private, sizes, q4, q8, include_q8)


def refresh_card():
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=os.environ["HF_TOKEN"])
    report = json.loads(Path(hf_hub_download(OUT, "packaging-complete.json", token=os.environ["HF_TOKEN"])).read_text())
    if report.get("training") or report.get("release_qualified"):
        raise SystemExit("Refusing to refresh a card that is not a test build")
    card = readme(report["private"], report["sizes"], report["q4_smoke"], report["q8_smoke"], report["published"])
    api.upload_file(repo_id=OUT, path_in_repo="README.md", path_or_fileobj=card.encode(),
                    commit_message="Correct the Repair2 GGUF test-build card")
    print("GGUF_CARD_REFRESHED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true")
    group.add_argument("--launch", action="store_true")
    group.add_argument("--refresh-card", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
    elif args.launch:
        launch()
    elif args.refresh_card:
        refresh_card()
    else:
        package()


if __name__ == "__main__":
    main()
