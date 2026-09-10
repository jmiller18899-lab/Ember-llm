"""Read-only post-tool schema-key localization for Ember routing.

Compares the exact saved v0.0.53 step-9 checkpoint with a deterministic
reproduction of the best safe v0.0.54 two-stage state (Stage A 8e-7 step 1,
Stage B 3.2e-6 step 10). For the eight development tool cases, measure:
  * free greedy continuation immediately after <|tool|>;
  * args-first vs name-first JSON-prefix preference;
  * the first family/schema fork under {"arguments":{ among
    location, expression, query, timezone.

No optimizer state is saved, no checkpoint is written, and no production state
is changed. CPU-only diagnostic.
"""
from __future__ import annotations

import copy
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
from jobs import ember_v053_first_token_anchor as base
from jobs import ember_v054_routing_repair as v54
from jobs import ember_v054_two_stage_route_prefix as ts
from jobs import ember_alternating_trust_region as trust

OUT = Path("v054-schema-key-logits")
EXPECTED_KEY = {
    "weather": "location",
    "calculator": "expression",
    "web_search": "query",
    "get_time": "timezone",
}
KEY_SUFFIX = {
    "location": '"location":',
    "expression": '"expression":',
    "query": '"query":',
    "timezone": '"timezone":',
}
TOPLEVEL_SUFFIX = {
    "arguments_first": '{"arguments":{',
    "name_first": '{"name":',
}


def lcp_many(seqs: list[list[int]]) -> int:
    if not seqs:
        return 0
    limit = min(len(s) for s in seqs)
    n = 0
    for i in range(limit):
        value = int(seqs[0][i])
        if any(int(s[i]) != value for s in seqs[1:]):
            break
        n += 1
    return n


def continuation_scores(model, tokenizer, base_text: str, suffixes: dict[str, str]):
    full = {name: [int(x) for x in tokenizer.encode(base_text + suffix)] for name, suffix in suffixes.items()}
    common = lcp_many(list(full.values()))
    if common < 1:
        raise RuntimeError("candidate encodings have no usable common prefix")

    common_ids = list(next(iter(full.values())))[:common]
    x = torch.tensor([common_ids], dtype=torch.long)
    with torch.inference_mode():
        logits, _ = model(x, None)
    first_logits = logits[0, -1]
    first_probs = F.softmax(first_logits, dim=-1)
    predicted_id = int(torch.argmax(first_logits).item())

    rows = {}
    for name, ids in full.items():
        tail = ids[common:]
        if not tail:
            rows[name] = {
                "first_token_id": None,
                "first_token_text": "",
                "first_probability": 1.0,
                "first_rank": 1,
                "tail_tokens": 0,
                "tail_log_probability": 0.0,
                "tail_mean_log_probability": 0.0,
            }
            continue
        tid = int(tail[0])
        prob = float(first_probs[tid].item())
        rank = int((first_logits > first_logits[tid]).sum().item()) + 1

        # Score the entire candidate-only tail from the shared token prefix.
        seq = common_ids + tail
        inp = torch.tensor([seq[:-1]], dtype=torch.long)
        with torch.inference_mode():
            all_logits, _ = model(inp, None)
        start = len(common_ids) - 1
        selected = all_logits[0, start:start + len(tail)]
        targets = torch.tensor(tail, dtype=torch.long)
        token_logp = F.log_softmax(selected, dim=-1).gather(1, targets[:, None]).squeeze(1)
        total = float(token_logp.sum().item())
        rows[name] = {
            "first_token_id": tid,
            "first_token_text": tokenizer.decode([tid]),
            "first_probability": prob,
            "first_rank": rank,
            "tail_tokens": len(tail),
            "tail_log_probability": total,
            "tail_mean_log_probability": total / len(tail),
        }

    return {
        "common_token_count": common,
        "predicted_next_token_id": predicted_id,
        "predicted_next_token_text": tokenizer.decode([predicted_id]),
        "candidates": rows,
    }


def greedy_after_tool(model, tokenizer, prompt: str, max_new: int = 18):
    forced = prompt + "<|tool|>\n"
    ids = tokenizer.encode(forced)
    x = torch.tensor([ids], dtype=torch.long)
    available = int(model.cfg.block_size) - len(ids)
    n = min(max_new, max(1, available))
    with torch.inference_mode():
        y = model.generate(x, max_new_tokens=n, temperature=1.0, top_k=1)
    generated = y[0].tolist()[len(ids):]
    return tokenizer.decode(generated)


def probe_state(model, tokenizer, state, label: str):
    model.load_state_dict(state)
    model.eval()
    rows = []
    key_correct_first = 0
    key_correct_mean = 0
    args_first = 0
    query_first_wins = 0
    for c in [r for r in held.CASES if r["kind"] == "tool_call"]:
        expected_tool = c["expected_tool"]
        expected_key = EXPECTED_KEY[expected_tool]
        forced = c["prompt"] + "<|tool|>\n"
        top = continuation_scores(model, tokenizer, forced, TOPLEVEL_SUFFIX)
        args_score = top["candidates"]["arguments_first"]
        name_score = top["candidates"]["name_first"]
        top_choice = max(top["candidates"], key=lambda k: top["candidates"][k]["first_probability"])
        args_first += int(top_choice == "arguments_first")

        args_base = forced + '{"arguments":{'
        keys = continuation_scores(model, tokenizer, args_base, KEY_SUFFIX)
        first_choice = max(keys["candidates"], key=lambda k: keys["candidates"][k]["first_probability"])
        mean_choice = max(keys["candidates"], key=lambda k: keys["candidates"][k]["tail_mean_log_probability"])
        key_correct_first += int(first_choice == expected_key)
        key_correct_mean += int(mean_choice == expected_key)
        query_first_wins += int(first_choice == "query")

        row = {
            "id": c["id"],
            "expected_tool": expected_tool,
            "expected_key": expected_key,
            "free_after_tool": greedy_after_tool(model, tokenizer, c["prompt"]),
            "toplevel": top,
            "toplevel_first_choice": top_choice,
            "toplevel_arguments_probability": args_score["first_probability"],
            "toplevel_name_probability": name_score["first_probability"],
            "schema_keys": keys,
            "schema_first_choice": first_choice,
            "schema_mean_choice": mean_choice,
        }
        rows.append(row)
        print(json.dumps({
            "event": "schema_key_case",
            "model": label,
            "id": c["id"],
            "expected_tool": expected_tool,
            "expected_key": expected_key,
            "free_after_tool": row["free_after_tool"][:120].replace("\n", "\\n"),
            "toplevel_choice": top_choice,
            "args_p": args_score["first_probability"],
            "name_p": name_score["first_probability"],
            "schema_first_choice": first_choice,
            "schema_mean_choice": mean_choice,
            "schema_first_probabilities": {k: v["first_probability"] for k, v in keys["candidates"].items()},
            "schema_first_ranks": {k: v["first_rank"] for k, v in keys["candidates"].items()},
        }), flush=True)

    return {
        "label": label,
        "metrics": {
            "cases": len(rows),
            "args_first_preferred": args_first,
            "schema_key_correct_by_first_token": key_correct_first,
            "schema_key_correct_by_mean_tail_logp": key_correct_mean,
            "query_first_token_wins": query_first_wins,
        },
        "cases": rows,
    }


def reproduce_best(model, tokenizer, selected, template):
    contract, tool_id, special_ids = v54.routing_contract(tokenizer)
    copy_tool_id, _ = trust.objectives.token_contract(tokenizer)
    copy_examples = [
        trust.objectives.supervised_example(tokenizer, c, "placement", template, copy_tool_id)
        for c in selected
    ]

    ts.configure_groups(model, ts.A_GROUPS)
    opt_a = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=8e-7, weight_decay=0.0)
    a_stats = ts.projected_step(
        model, opt_a,
        lambda: ts.stage_a_backward(model, tokenizer, tool_id, special_ids),
        copy_examples,
    )
    a_route = v54.routing_counts(model, tokenizer, held.CASES)
    a_place = base.placement_probe(model, tokenizer, selected, template)
    if int(a_route["direct_ok"]) != 3 or int(a_route["tool_ok"]) != 8 or int(a_place["token_top1"]) != 22:
        raise RuntimeError(f"Stage-A reproduction mismatch: route={a_route['direct_ok']}/{a_route['tool_ok']} copy={a_place['token_top1']}")

    ts.configure_groups(model, ts.B_GROUPS)
    opt_b = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3.2e-6, weight_decay=0.0)
    b_stats = []
    for _ in range(10):
        b_stats.append(ts.projected_step(
            model, opt_b,
            lambda: ts.stage_b_backward(model, tokenizer, tool_id, special_ids),
            copy_examples,
        ))
    b_route = v54.routing_counts(model, tokenizer, held.CASES)
    b_place = base.placement_probe(model, tokenizer, selected, template)
    if int(b_route["direct_ok"]) != 10 or int(b_route["tool_ok"]) != 8 or int(b_place["token_top1"]) != 22:
        raise RuntimeError(f"Stage-B reproduction mismatch: route={b_route['direct_ok']}/{b_route['tool_ok']} copy={b_place['token_top1']}")
    return {
        "stage_a": {"routing": a_route, "placement": a_place, "stats": a_stats},
        "stage_b": {"routing": b_route, "placement": b_place, "last_stats": b_stats[-1]},
    }


def summary_md(report):
    lines = [
        "# Ember v0.0.54 post-tool schema-key logits", "",
        "Read-only comparison of saved v0.0.53 and reproduced best-safe v0.0.54 two-stage state.",
        "The diagnostic locates the family decision under the existing args-first JSON path.", "",
        "| Model | args-first preferred | correct schema key (first token) | correct schema key (mean tail) | query wins |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key in ("v053", "v054_best"):
        m = report[key]["metrics"]
        lines.append(
            f"| {key} | {m['args_first_preferred']}/{m['cases']} | "
            f"{m['schema_key_correct_by_first_token']}/{m['cases']} | "
            f"{m['schema_key_correct_by_mean_tail_logp']}/{m['cases']} | "
            f"{m['query_first_token_wins']}/{m['cases']} |"
        )
    lines += ["", "## Per-case first-token schema choice", ""]
    for i, c in enumerate(report["v053"]["cases"]):
        b = report["v054_best"]["cases"][i]
        lines.append(
            f"- **{c['id']}** expected `{c['expected_key']}`: v0.0.53 `{c['schema_first_choice']}` → v0.0.54-best `{b['schema_first_choice']}`."
        )
    lines += ["", f"Interpretation: {report['interpretation']}", "", "No checkpoint was saved or promoted.\n"]
    return "\n".join(lines)


def main():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN required")
    torch.set_num_threads(2)
    torch.manual_seed(20260910)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="ember-v054-schema-key-") as td:
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
        if checkpoint.get("train_config", {}).get("version") != "0.0.53":
            raise RuntimeError("source is not exact saved v0.0.53")
        model.eval()
        pristine = copy.deepcopy(model.state_dict())
        pristine_hash = trust.trace.state_digest(model)

        cfg, template, template_report, selected, selected_ids, *_ = base.build_placement_fixture(work)
        base_place = base.placement_probe(model, tokenizer, selected, template)
        if int(base_place["token_top1"]) != 22:
            raise RuntimeError("saved v0.0.53 copy baseline drifted")

        v053 = probe_state(model, tokenizer, pristine, "v0.0.53")
        model.load_state_dict(pristine); model.eval()
        reproduction = reproduce_best(model, tokenizer, selected, template)
        best_state = copy.deepcopy(model.state_dict())
        best_hash = trust.trace.state_digest(model)
        v054_best = probe_state(model, tokenizer, best_state, "v0.0.54-best")

        m0 = v053["metrics"]
        m1 = v054_best["metrics"]
        if m0["query_first_token_wins"] >= 4 and m0["schema_key_correct_by_first_token"] < 8:
            interpretation = (
                "The family error is already visible at the args-first argument-schema key. "
                "A bounded schema-key anchor is the next direct repair target."
            )
        else:
            interpretation = (
                "The schema-key fork alone does not explain enough failures; inspect the earlier post-tool JSON-prefix decision before training."
            )
        report = {
            "schema_version": 1,
            "diagnostic": "ember-v054-schema-key-logits-v1",
            "status": "PASS",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_state_sha256": pristine_hash,
            "reproduced_best_state_sha256": best_hash,
            "reproduction": reproduction,
            "v053": v053,
            "v054_best": v054_best,
            "interpretation": interpretation,
            "read_only": True,
            "elapsed_seconds": time.monotonic() - started,
        }
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (OUT / "summary.md").write_text(summary_md(report))
        print(json.dumps({
            "event": "schema_key_complete",
            "v053": m0,
            "v054_best": m1,
            "interpretation": interpretation,
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
