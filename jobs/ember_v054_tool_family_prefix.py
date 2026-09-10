"""Read-only localization of Ember's earliest tool-family decision.

The v0.0.53 held-out failures usually enter <|tool|> correctly but then choose
an argument schema belonging to the wrong tool (often `query`, i.e. web_search)
before the later `name` field is generated. This diagnostic scores the four
possible argument-key continuations immediately after:

    <|tool|>\n{"arguments":{

for true tool requests. It compares held-out cases, the original fixed tool
cases, and the separate v0.0.54 development tool cases. No weights are changed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import time
import zipfile

from huggingface_hub import HfApi
import torch
import torch.nn.functional as F

from jobs import ember_hf_eval as ev
from jobs import ember_v053_heldout_routing as held
from jobs import ember_v054_routing_repair as v54

OUT = Path("v054-tool-family-prefix")
PREFIX = '<|tool|>\n{"arguments":{'
KEYS = {
    "weather": '"location":',
    "calculator": '"expression":',
    "web_search": '"query":',
    "get_time": '"timezone":',
}


def sequence_score(model, tokenizer, prompt: str, continuation: str) -> dict:
    prefix_ids = list(tokenizer.encode(prompt + PREFIX))
    full_ids = list(tokenizer.encode(prompt + PREFIX + continuation))
    split = 0
    for a, b in zip(prefix_ids, full_ids):
        if a != b:
            break
        split += 1
    targets = full_ids[split:]
    context = full_ids[:split]
    if not targets or not context:
        raise RuntimeError("continuation tokenization produced empty context/target")
    seq = context + targets
    if len(seq) > int(model.cfg.block_size):
        raise RuntimeError("family-prefix probe exceeds context window")
    x = torch.tensor([seq[:-1]], dtype=torch.long)
    with torch.inference_mode():
        logits, _ = model(x, None)
    logp = F.log_softmax(logits[0], dim=-1)
    start = len(context) - 1
    vals = []
    for offset, tid in enumerate(targets):
        vals.append(float(logp[start + offset, int(tid)]))
    return {
        "tokens": len(targets),
        "sum_logprob": sum(vals),
        "mean_logprob": sum(vals) / len(vals),
        "token_logprobs": vals,
    }


def inspect_case(model, tokenizer, case: dict) -> dict:
    expected_tool = case["expected_tool"]
    if expected_tool not in KEYS:
        raise RuntimeError(f"unsupported expected tool {expected_tool}")
    scores = {}
    for tool_name, key_text in KEYS.items():
        scores[tool_name] = {
            "key": key_text,
            **sequence_score(model, tokenizer, case["prompt"], key_text),
        }
    ranked = sorted(scores, key=lambda name: scores[name]["sum_logprob"], reverse=True)
    expected_score = scores[expected_tool]["sum_logprob"]
    best_wrong = max(
        (scores[name]["sum_logprob"] for name in scores if name != expected_tool),
        default=-float("inf"),
    )
    return {
        "id": case["id"],
        "expected_tool": expected_tool,
        "expected_key": KEYS[expected_tool],
        "predicted_tool_family": ranked[0],
        "predicted_key": KEYS[ranked[0]],
        "correct_top1": ranked[0] == expected_tool,
        "expected_rank": ranked.index(expected_tool) + 1,
        "expected_minus_best_wrong_sum_logprob": expected_score - best_wrong,
        "scores": scores,
        "ranking": ranked,
    }


def aggregate(rows: list[dict]) -> dict:
    by_expected = {}
    confusion = {}
    for row in rows:
        expected = row["expected_tool"]
        pred = row["predicted_tool_family"]
        bucket = by_expected.setdefault(expected, {"correct": 0, "total": 0, "margins": []})
        bucket["total"] += 1
        bucket["correct"] += int(row["correct_top1"])
        bucket["margins"].append(float(row["expected_minus_best_wrong_sum_logprob"]))
        confusion[f"{expected}->{pred}"] = confusion.get(f"{expected}->{pred}", 0) + 1
    for bucket in by_expected.values():
        bucket["mean_margin"] = sum(bucket["margins"]) / len(bucket["margins"])
        del bucket["margins"]
    nonsearch = [r for r in rows if r["expected_tool"] != "web_search"]
    query_wrong = sum(r["predicted_tool_family"] == "web_search" for r in nonsearch)
    return {
        "correct": sum(int(r["correct_top1"]) for r in rows),
        "total": len(rows),
        "by_expected": by_expected,
        "confusion": confusion,
        "nonsearch_predicted_as_web_search": query_wrong,
        "nonsearch_total": len(nonsearch),
    }


def summary(report: dict) -> str:
    lines = [
        "# Ember v0.0.54 earliest tool-family prefix diagnostic", "",
        "Read-only exact v0.0.53 step-9 checkpoint.",
        f"Scored candidate argument keys immediately after `{PREFIX}`.",
        "",
        "| Set | Correct family key | Non-search → web_search |",
        "| --- | ---: | ---: |",
    ]
    for name in ("heldout", "fixed", "development"):
        a = report["sets"][name]["aggregate"]
        lines.append(
            f"| {name} | {a['correct']}/{a['total']} | "
            f"{a['nonsearch_predicted_as_web_search']}/{a['nonsearch_total']} |"
        )
    lines += ["", "## Held-out cases", "", "| Case | Expected | Predicted | Rank | Margin |", "| --- | --- | --- | ---: | ---: |"]
    for r in report["sets"]["heldout"]["rows"]:
        lines.append(
            f"| {r['id']} | {r['expected_tool']} | {r['predicted_tool_family']} | "
            f"{r['expected_rank']} | {r['expected_minus_best_wrong_sum_logprob']:+.3f} |"
        )
    lines += ["", "No optimizer step, checkpoint save, export, promotion, or integration occurred."]
    return "\n".join(lines) + "\n"


def main():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN required")
    torch.set_num_threads(2)
    torch.manual_seed(20260910)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="ember-family-prefix-") as td:
        work = Path(td)
        archive = ev.download_verified(ev.PACKAGE_URL, work / "ember.zip", ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        import sys
        sys.path.insert(0, str(work / "src" / "ember"))

        api = HfApi(token=token)
        owner = api.whoami()["name"]
        repo = f"{owner}/{held.MODEL_NAME}"
        model, tokenizer, checkpoint = held.load_full(repo, work, token)
        model.eval()

        heldout_cases = [c for c in held.CASES if c["kind"] == "tool_call"]
        fixed_spec = json.loads(Path("config/ember_v0.0.8_eval.json").read_text())
        fixed_cases = [c for c in fixed_spec["cases"] if c["kind"] == "tool_call"]
        dev_cases = [c for c in v54.TRAIN if c["kind"] == "tool_call"]

        sets = {}
        for name, cases in (("heldout", heldout_cases), ("fixed", fixed_cases), ("development", dev_cases)):
            rows = [inspect_case(model, tokenizer, c) for c in cases]
            sets[name] = {"rows": rows, "aggregate": aggregate(rows)}
            print(json.dumps({"event":"family_prefix_set", "set":name, **sets[name]["aggregate"]}), flush=True)
            for row in rows:
                print(json.dumps({
                    "event":"family_prefix_case", "set":name,
                    "id":row["id"], "expected":row["expected_tool"],
                    "predicted":row["predicted_tool_family"],
                    "rank":row["expected_rank"],
                    "margin":row["expected_minus_best_wrong_sum_logprob"],
                    "ranking":row["ranking"],
                }), flush=True)

        report = {
            "schema_version": 1,
            "diagnostic": "ember-v054-earliest-tool-family-prefix-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_repo": repo,
            "checkpoint": held.BEST_PATH,
            "checkpoint_sha256": held.BEST_SHA256,
            "prefix": PREFIX,
            "candidate_keys": KEYS,
            "sets": sets,
            "read_only": True,
            "optimizer_steps": 0,
            "elapsed_seconds": time.monotonic() - started,
        }
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (OUT / "summary.md").write_text(summary(report))
        print(json.dumps({
            "event":"family_prefix_complete",
            "heldout":sets["heldout"]["aggregate"],
            "elapsed_seconds":report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
