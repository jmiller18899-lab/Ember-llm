"""Read-only first-token routing-logit diagnostic for Ember v0.0.52.

Compares the saved v0.0.52 trust-region candidate against its v0.0.31 step-479
source on the four direct-response prompts and four explicit tool-call prompts from
the fixed promotion eval. Measures the probability/rank/margin of the <|tool|>
continuation token and the strongest non-special continuation.

No weights are changed, saved, promoted, or uploaded.
"""
from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile

from huggingface_hub import HfApi, hf_hub_download
import torch

from jobs import ember_hf_eval as ev

CANDIDATE_NAME = "ember-v0.0.52-t4"
SOURCE_NAME = "ember-v0.0.31-t4"
OUT = Path("v052-first-token-logits")
TOP_K = 12


def load_model(repo_id: str, checkpoint_path: str, work: Path, package_root: Path):
    from src.checkpoint import load_checkpoint
    from src.model import EmberGPT, ModelConfig
    from src.tokenizer import tokenizer_from_state_dict

    token = os.environ["HF_TOKEN"].strip()
    local = Path(hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=checkpoint_path,
        token=token,
        local_dir=work / repo_id.replace("/", "__"),
    ))
    checkpoint = load_checkpoint(local, device="cpu")
    tokenizer = tokenizer_from_state_dict(checkpoint["tokenizer"])
    model = EmberGPT(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to("cpu").eval()
    return model, tokenizer, checkpoint


def marker_id(contract: dict, marker: str) -> int:
    signature = contract["signatures"][marker]
    if len(signature) != 1:
        raise RuntimeError(f"{marker} is not atomic in continuation position: {signature}")
    return int(signature[0])


def display_token(tokenizer, token_id: int, shared_prefix_id: int | None) -> str:
    candidates = []
    for ids in ([token_id], [shared_prefix_id, token_id] if shared_prefix_id is not None else []):
        if not ids:
            continue
        try:
            text = tokenizer.decode(ids)
        except Exception:
            continue
        if text not in candidates:
            candidates.append(text)
    text = candidates[-1] if candidates else ""
    return text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def inspect_case(model, tokenizer, case: dict) -> dict:
    prompt_ids = tokenizer.encode(case["prompt"])
    x = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.inference_mode():
        logits, _ = model(x, None)
    next_logits = logits[0, -1].float().cpu()
    probs = torch.softmax(next_logits, dim=-1)

    contract = ev.special_token_contract(tokenizer)
    if not contract["atomic"] or not contract["unique"]:
        raise RuntimeError("special-token continuation contract is not atomic+unique")
    shared_prefix = contract.get("shared_prefix_id")
    special_ids = {marker_id(contract, marker) for marker in ev.SPECIAL_TOKENS}
    tool_id = marker_id(contract, "<|tool|>")

    tool_logit = float(next_logits[tool_id])
    tool_prob = float(probs[tool_id])
    tool_rank = 1 + int((next_logits > next_logits[tool_id]).sum().item())

    sorted_ids = torch.argsort(next_logits, descending=True).tolist()
    non_special_id = next(i for i in sorted_ids if int(i) not in special_ids)
    non_special_logit = float(next_logits[non_special_id])
    non_special_prob = float(probs[non_special_id])

    top_ids = sorted_ids[:TOP_K]
    top = []
    for rank, tid in enumerate(top_ids, start=1):
        tid = int(tid)
        top.append({
            "rank": rank,
            "id": tid,
            "token": display_token(tokenizer, tid, shared_prefix),
            "logit": float(next_logits[tid]),
            "probability": float(probs[tid]),
            "is_tool": tid == tool_id,
            "is_special": tid in special_ids,
        })

    special_mass = float(sum(float(probs[i]) for i in special_ids))
    lexical_mass = max(0.0, 1.0 - special_mass)
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-30))).sum().item())
    argmax = int(sorted_ids[0])

    return {
        "prompt_tokens": len(prompt_ids),
        "argmax_id": argmax,
        "argmax_token": display_token(tokenizer, argmax, shared_prefix),
        "argmax_is_tool": argmax == tool_id,
        "tool_token_id": tool_id,
        "tool_logit": tool_logit,
        "tool_probability": tool_prob,
        "tool_rank": tool_rank,
        "best_non_special_id": int(non_special_id),
        "best_non_special_token": display_token(tokenizer, int(non_special_id), shared_prefix),
        "best_non_special_logit": non_special_logit,
        "best_non_special_probability": non_special_prob,
        "tool_minus_best_non_special_logit": tool_logit - non_special_logit,
        "special_probability_mass": special_mass,
        "lexical_probability_mass": lexical_mass,
        "entropy_nats": entropy,
        "top_tokens": top,
    }


def source_checkpoint_path(api: HfApi, repo_id: str, token: str, work: Path) -> str:
    state_path = Path(hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename="run-state.json",
        token=token,
        local_dir=work / "source-state",
    ))
    state = json.loads(state_path.read_text())
    if int(state.get("best_step", -1)) != 479:
        raise RuntimeError(f"source best_step is not 479: {state.get('best_step')}")
    run_id = str(state["run_id"])
    return f"checkpoints/{run_id}/best.pt"


def candidate_checkpoint_path(api: HfApi, repo_id: str) -> str:
    files = api.list_repo_files(repo_id, repo_type="model")
    return ev.latest_checkpoint_path(files, "/best.pt")


def collect(model, tokenizer, cases: list[dict]) -> dict[str, dict]:
    rows = {}
    for case in cases:
        row = inspect_case(model, tokenizer, case)
        rows[case["id"]] = row
        print(json.dumps({
            "event": "first_token_probe",
            "case": case["id"],
            "kind": case["kind"],
            "argmax": row["argmax_token"],
            "argmax_is_tool": row["argmax_is_tool"],
            "tool_probability": row["tool_probability"],
            "tool_rank": row["tool_rank"],
            "tool_margin": row["tool_minus_best_non_special_logit"],
            "best_non_special": row["best_non_special_token"],
        }), flush=True)
    return rows


def summarize(report: dict) -> str:
    lines = [
        "# Ember v0.0.52 first-token logits diagnostic",
        "",
        "Read-only comparison of v0.0.52 trust-region candidate vs v0.0.31 step-479 source.",
        "Positive margin means <|tool|> beats the strongest non-special first token.",
        "",
        "| Case | Kind | v52 argmax | v52 P(tool) | v52 tool rank | v52 margin | v31 P(tool) | Δ tool logit |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["cases"]:
        c = row["candidate"]
        s = row["source"]
        lines.append(
            f"| {row['id']} | {row['kind']} | `{c['argmax_token']}` | "
            f"{c['tool_probability']:.6f} | {c['tool_rank']} | "
            f"{c['tool_minus_best_non_special_logit']:+.4f} | "
            f"{s['tool_probability']:.6f} | {row['delta_tool_logit']:+.4f} |"
        )
    lines += ["", "## Direct-response detail", ""]
    for row in report["cases"]:
        if row["kind"] != "direct_response":
            continue
        c = row["candidate"]
        lines.append(
            f"- **{row['id']}**: argmax `{c['argmax_token']}`, P(tool)={c['tool_probability']:.6f}, "
            f"tool rank={c['tool_rank']}, best lexical=`{c['best_non_special_token']}`, "
            f"tool-vs-lexical margin={c['tool_minus_best_non_special_logit']:+.4f}, "
            f"Δtool-logit from v31={row['delta_tool_logit']:+.4f}."
        )
    lines += [
        "",
        "No training, checkpoint write, promotion, export, or integration occurred.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    torch.set_num_threads(2)
    torch.manual_seed(1337)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=token)
    owner = api.whoami()["name"]
    candidate_repo = f"{owner}/{CANDIDATE_NAME}"
    source_repo = f"{owner}/{SOURCE_NAME}"

    with tempfile.TemporaryDirectory(prefix="ember-v052-first-token-") as td:
        work = Path(td)
        archive = ev.download_verified(ev.PACKAGE_URL, work / "ember.zip", ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        package_root = work / "src" / "ember"
        import sys
        sys.path.insert(0, str(package_root))

        spec = json.loads(Path("config/ember_v0.0.8_eval.json").read_text())
        cases = [c for c in spec["cases"] if c["kind"] in {"direct_response", "tool_call"}]
        if len(cases) != 8:
            raise RuntimeError(f"expected 8 routing cases, got {len(cases)}")

        candidate_path = candidate_checkpoint_path(api, candidate_repo)
        source_path = source_checkpoint_path(api, source_repo, token, work)

        candidate_model, candidate_tok, candidate_ckpt = load_model(candidate_repo, candidate_path, work, package_root)
        candidate_rows = collect(candidate_model, candidate_tok, cases)
        candidate_contract = ev.special_token_contract(candidate_tok)
        del candidate_model, candidate_ckpt
        gc.collect()

        source_model, source_tok, source_ckpt = load_model(source_repo, source_path, work, package_root)
        source_rows = collect(source_model, source_tok, cases)
        source_contract = ev.special_token_contract(source_tok)
        del source_model, source_ckpt
        gc.collect()

        if candidate_contract["signatures"] != source_contract["signatures"]:
            raise RuntimeError("candidate/source special-token contracts differ")

        report_cases = []
        for case in cases:
            cid = case["id"]
            c = candidate_rows[cid]
            s = source_rows[cid]
            report_cases.append({
                "id": cid,
                "kind": case["kind"],
                "candidate": c,
                "source": s,
                "delta_tool_logit": c["tool_logit"] - s["tool_logit"],
                "delta_tool_probability": c["tool_probability"] - s["tool_probability"],
                "delta_tool_margin": c["tool_minus_best_non_special_logit"] - s["tool_minus_best_non_special_logit"],
            })

        direct = [r for r in report_cases if r["kind"] == "direct_response"]
        tool = [r for r in report_cases if r["kind"] == "tool_call"]
        report = {
            "schema_version": 1,
            "diagnostic": "ember-v052-first-token-logits-v1",
            "candidate_repo": candidate_repo,
            "candidate_checkpoint": candidate_path,
            "source_repo": source_repo,
            "source_checkpoint": source_path,
            "tool_token_id": marker_id(candidate_contract, "<|tool|>"),
            "cases": report_cases,
            "aggregates": {
                "direct_argmax_tool_count": sum(int(r["candidate"]["argmax_is_tool"]) for r in direct),
                "tool_argmax_tool_count": sum(int(r["candidate"]["argmax_is_tool"]) for r in tool),
                "mean_direct_tool_probability": sum(r["candidate"]["tool_probability"] for r in direct) / len(direct),
                "mean_toolcase_tool_probability": sum(r["candidate"]["tool_probability"] for r in tool) / len(tool),
                "mean_direct_delta_tool_logit": sum(r["delta_tool_logit"] for r in direct) / len(direct),
                "mean_toolcase_delta_tool_logit": sum(r["delta_tool_logit"] for r in tool) / len(tool),
            },
            "read_only": True,
        }
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (OUT / "summary.md").write_text(summarize(report))
        print(json.dumps({"event": "first_token_complete", **report["aggregates"]}), flush=True)


if __name__ == "__main__":
    main()
