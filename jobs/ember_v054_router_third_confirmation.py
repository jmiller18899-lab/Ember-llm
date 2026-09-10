"""Third untouched confirmation for Ember v0.0.54 hierarchical frozen router.

Architecture and training data are frozen before this test. The prior 20-case
set and 50-case confirmation have both influenced architecture decisions and are
therefore development evidence now. This script adds 100 new prompts, balanced
20/class across direct/weather/calculator/web_search/get_time.

The hierarchical block-3 router is fit only on its existing 128 training
prompts. Ridge strengths remain selected only by training-set CV. This script
performs no tuning, no Ember weight changes, no router integration, and no
promotion. Full precision, independently-fit INT4, and the full head applied
unchanged to INT4 hidden states are all measured.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

from huggingface_hub import HfApi
import torch

from jobs import ember_hf_eval as ev
from jobs import ember_v053_heldout_routing as held
from jobs import ember_v054_frozen_router_probe as probe
from jobs import ember_v054_router_confirmation as confirm
from jobs import ember_v054_hierarchical_router as hier
from jobs import ember_v054_routing_repair as v54

OUT = Path("v054-router-third-confirmation")
REPRESENTATION = hier.REPRESENTATION

SYSTEM_E = (
    "You are Ember. Decide whether to answer normally or use one tool: weather, calculator, web_search, get_time. "
    "Use tools only when the request actually needs live weather, arithmetic, an up-to-date web fact, or the current local time. "
    "Writing, editing, explaining, classifying, comparing, and summarizing are direct tasks when no live fact or calculation is requested."
)
SYSTEM_F = (
    "You are Ember. Available tools are weather, calculator, web_search, and get_time. "
    "Route by task intent, not by keywords. Conceptual discussion of weather, time, search, current/latest wording, or calculators is direct unless live data or arithmetic is truly needed."
)


def prompt(system: str, user: str) -> str:
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"


def direct(cid: str, user: str, system: str = SYSTEM_E) -> dict:
    return {"id": cid, "kind": "direct_response", "user": user, "prompt": prompt(system, user)}


def tool(cid: str, user: str, expected_tool: str, system: str = SYSTEM_E) -> dict:
    return {"id": cid, "kind": "tool_call", "user": user, "prompt": prompt(system, user), "expected_tool": expected_tool}


THIRD = [
    # 20 direct: broad direct intents, including tool-adjacent lures and sentiment.
    direct("third_d01", "Label the sentiment as positive, negative, or neutral: The hotfix resolved every reported issue."),
    direct("third_d02", "Label the sentiment: The weather icon is shown beside the page title.", SYSTEM_F),
    direct("third_d03", "Rewrite this heading professionally: current temperature card title is too long."),
    direct("third_d04", "Summarize in one sentence: the calculator migration completed, tests passed, and no behavior changed.", SYSTEM_F),
    direct("third_d05", "Explain the difference between climate and a weather forecast without checking live weather."),
    direct("third_d06", "Explain what the percent symbol means without calculating a specific number.", SYSTEM_F),
    direct("third_d07", "Explain what a time-zone abbreviation represents without telling me the current time."),
    direct("third_d08", "Define a search query in plain English without performing a web search.", SYSTEM_F),
    direct("third_d09", "Compare a clock and a timer in two short sentences."),
    direct("third_d10", "Give three short steps for testing a search-filter dropdown in a mock interface.", SYSTEM_F),
    direct("third_d11", "Write one sentence thanking a reviewer for checking the latest-release documentation."),
    direct("third_d12", "Classify this as a command or a statement: Check the weather icon spacing.", SYSTEM_F),
    direct("third_d13", "Rewrite this button label more clearly: find current docs."),
    direct("third_d14", "Explain why cached live data can become stale, without looking anything up.", SYSTEM_F),
    direct("third_d15", "Summarize: the search index was rebuilt in the test environment and all smoke checks passed."),
    direct("third_d16", "Label the sentiment: The current build is stable, fast, and easy to use.", SYSTEM_F),
    direct("third_d17", "Rewrite this bug title: calculator and time widgets need visual polish."),
    direct("third_d18", "Give two conceptual reasons time zones are useful; do not check any current time.", SYSTEM_F),
    direct("third_d19", "Compare a web page with a search result in two concise sentences."),
    direct("third_d20", "Write a short tooltip for a button that refreshes the current-status display.", SYSTEM_F),

    # 20 live-weather requests, all new locations.
    tool("third_w01", "What is the weather in Reno right now?", "weather"),
    tool("third_w02", "Is it raining in Savannah, Georgia at the moment?", "weather", SYSTEM_F),
    tool("third_w03", "Give me the live temperature in Reykjavik.", "weather"),
    tool("third_w04", "How is the weather in Prague right now?", "weather", SYSTEM_F),
    tool("third_w05", "Do I need an umbrella in Manila at the moment?", "weather"),
    tool("third_w06", "Check the current weather conditions in Tucson.", "weather", SYSTEM_F),
    tool("third_w07", "Is it sunny in Halifax right now?", "weather"),
    tool("third_w08", "What is the live weather in Auckland?", "weather", SYSTEM_F),
    tool("third_w09", "Tell me the current temperature in Budapest.", "weather"),
    tool("third_w10", "What is the weather doing in Fairbanks right now?", "weather", SYSTEM_F),
    tool("third_w11", "Is there rain in Valencia, Spain right now?", "weather"),
    tool("third_w12", "Give me the current outdoor temperature in Glasgow.", "weather", SYSTEM_F),
    tool("third_w13", "What are the live weather conditions in Omaha?", "weather"),
    tool("third_w14", "Should I expect rain in Kyoto at the moment?", "weather", SYSTEM_F),
    tool("third_w15", "Check the weather in Cape Town right now.", "weather"),
    tool("third_w16", "Is it cold in Tallinn at the moment?", "weather", SYSTEM_F),
    tool("third_w17", "What is the current temperature in San Juan, Puerto Rico?", "weather"),
    tool("third_w18", "Tell me the live weather in Calgary.", "weather", SYSTEM_F),
    tool("third_w19", "Do I need a rain jacket in Naples, Italy right now?", "weather"),
    tool("third_w20", "What are the current weather conditions in Albuquerque?", "weather", SYSTEM_F),

    # 20 arithmetic requests.
    tool("third_c01", "Calculate 487 plus 2365.", "calculator"),
    tool("third_c02", "What is 73 multiplied by 64?", "calculator", SYSTEM_F),
    tool("third_c03", "Compute 12474 divided by 42.", "calculator"),
    tool("third_c04", "Calculate 7.5 percent of 1280.", "calculator", SYSTEM_F),
    tool("third_c05", "What is 39 squared?", "calculator"),
    tool("third_c06", "Compute 23 plus 41 plus 88 plus 114.", "calculator", SYSTEM_F),
    tool("third_c07", "Calculate 4200 minus 1763.", "calculator"),
    tool("third_c08", "What is 3.75 multiplied by 96?", "calculator", SYSTEM_F),
    tool("third_c09", "Compute 16384 divided by 128.", "calculator"),
    tool("third_c10", "Calculate 27 percent of 740.", "calculator", SYSTEM_F),
    tool("third_c11", "What is 11 cubed?", "calculator"),
    tool("third_c12", "Calculate 905 times 18.", "calculator", SYSTEM_F),
    tool("third_c13", "Compute 7326 divided by 18.", "calculator"),
    tool("third_c14", "What is 14.25 percent of 640?", "calculator", SYSTEM_F),
    tool("third_c15", "Calculate 10000 minus 4387.", "calculator"),
    tool("third_c16", "What is 2 to the power of 15?", "calculator", SYSTEM_F),
    tool("third_c17", "Compute 56.5 plus 88.75.", "calculator"),
    tool("third_c18", "Calculate 144 multiplied by 125.", "calculator", SYSTEM_F),
    tool("third_c19", "What is 9999 divided by 9?", "calculator"),
    tool("third_c20", "Calculate 62 percent of 350.", "calculator", SYSTEM_F),

    # 20 up-to-date web-fact requests.
    tool("third_s01", "Find the newest stable Ruby release.", "web_search"),
    tool("third_s02", "What is the current stable Perl release?", "web_search", SYSTEM_F),
    tool("third_s03", "Find the latest official CMake release.", "web_search"),
    tool("third_s04", "Look up the newest stable Git version.", "web_search", SYSTEM_F),
    tool("third_s05", "Find the current stable Django release.", "web_search"),
    tool("third_s06", "What is the latest stable Ruby on Rails release?", "web_search", SYSTEM_F),
    tool("third_s07", "Find the newest stable NumPy release.", "web_search"),
    tool("third_s08", "Look up the latest official PyTorch release.", "web_search", SYSTEM_F),
    tool("third_s09", "Find the current stable Caddy release.", "web_search"),
    tool("third_s10", "What is the newest stable Nginx release?", "web_search", SYSTEM_F),
    tool("third_s11", "Find the latest official LLVM release.", "web_search"),
    tool("third_s12", "What is the current stable SQLite release?", "web_search", SYSTEM_F),
    tool("third_s13", "Look up the newest stable Terraform release.", "web_search"),
    tool("third_s14", "Find the latest official Ansible release.", "web_search", SYSTEM_F),
    tool("third_s15", "What is the current stable Grafana release?", "web_search"),
    tool("third_s16", "Find the newest stable Prometheus release.", "web_search", SYSTEM_F),
    tool("third_s17", "Look up the latest official Blender release.", "web_search"),
    tool("third_s18", "What is the current stable Neovim release?", "web_search", SYSTEM_F),
    tool("third_s19", "Find the newest stable FFmpeg release.", "web_search"),
    tool("third_s20", "Look up the latest official Apache HTTP Server release.", "web_search", SYSTEM_F),

    # 20 current-local-time requests, all new locations.
    tool("third_t01", "What time is it in Helsinki right now?", "get_time"),
    tool("third_t02", "Give me the current local time in Bangkok.", "get_time", SYSTEM_F),
    tool("third_t03", "What is the time in Lima at this moment?", "get_time"),
    tool("third_t04", "Tell me the current time in Munich.", "get_time", SYSTEM_F),
    tool("third_t05", "What time is it in Athens right now?", "get_time"),
    tool("third_t06", "Give me the local time in Nagoya at the moment.", "get_time", SYSTEM_F),
    tool("third_t07", "What is the current time in Marrakech?", "get_time"),
    tool("third_t08", "Tell me what time it is in Amsterdam right now.", "get_time", SYSTEM_F),
    tool("third_t09", "Give me the current local time in Buenos Aires.", "get_time"),
    tool("third_t10", "What time is it in Delhi at this moment?", "get_time", SYSTEM_F),
    tool("third_t11", "Tell me the current local time in Rome.", "get_time"),
    tool("third_t12", "What time is it in Taipei right now?", "get_time", SYSTEM_F),
    tool("third_t13", "Give me the current time in Mexico City.", "get_time"),
    tool("third_t14", "What is the local time in Dublin at the moment?", "get_time", SYSTEM_F),
    tool("third_t15", "Tell me the current time in Nairobi.", "get_time"),
    tool("third_t16", "What time is it in Copenhagen right now?", "get_time", SYSTEM_F),
    tool("third_t17", "Give me the current local time in Shanghai.", "get_time"),
    tool("third_t18", "What is the time in Montevideo at this moment?", "get_time", SYSTEM_F),
    tool("third_t19", "Tell me the current time in Madrid.", "get_time"),
    tool("third_t20", "What time is it in Dubai right now?", "get_time", SYSTEM_F),
]


def validate(train_cases: list[dict]):
    if len(THIRD) != 100:
        raise RuntimeError(f"third confirmation must contain 100 cases, got {len(THIRD)}")
    counts = {label: 0 for label in probe.LABELS}
    for c in THIRD:
        counts[probe.label_of(c)] += 1
    if any(counts[label] != 20 for label in probe.LABELS):
        raise RuntimeError(f"third confirmation is not balanced 20/class: {counts}")
    ids = [c["id"] for c in THIRD]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate third-confirmation ids")
    forbidden = {c["id"] for c in train_cases} | {c["id"] for c in held.CASES} | {c["id"] for c in confirm.CONFIRM}
    overlap = forbidden.intersection(ids)
    if overlap:
        raise RuntimeError(f"third confirmation overlaps earlier ids: {sorted(overlap)}")
    return counts


def evaluate_fresh(heads, rows):
    X, _ = hier.stack(rows, REPRESENTATION)
    pred, margins = hier.hierarchical_predict(heads, X)
    result = hier.score(pred, rows, THIRD, margins)
    by_id = {c["id"]: c for c in THIRD}
    for row in result["rows"]:
        c = by_id[row["id"]]
        row["user"] = c["user"]
        row["kind"] = c["kind"]
        row["expected_tool"] = c.get("expected_tool")
    return result


def exact(m):
    return (
        m["five_way_correct"] == m["total"]
        and m["direct_vs_tool_correct"] == m["total"]
        and m["tool_family_correct"] == m["tool_family_total"]
    )


def compact(m):
    return {
        "five_way": f"{m['five_way_correct']}/{m['total']}",
        "direct_tool": f"{m['direct_vs_tool_correct']}/{m['total']}",
        "tool_family": f"{m['tool_family_correct']}/{m['tool_family_total']}",
        "min_correct_margin": m["min_correct_margin"],
        "max_wrong_margin": m["max_wrong_margin"],
        "misses": [r for r in m["rows"] if not r["correct"]],
    }


def summary_md(report):
    lines = [
        "# Ember v0.0.54 hierarchical router — third confirmation", "",
        "100 untouched prompts, balanced 20/class. Architecture and training data were frozen before this test.",
        "Ember language weights remain frozen.", "",
        "| Router | Third 5-way | Direct/tool | Tool family | Min correct margin | Max wrong margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("full", "int4", "full_heads_on_int4"):
        m = report["routers"][key]["third"]
        lines.append(
            f"| {key} | {m['five_way_correct']}/{m['total']} | "
            f"{m['direct_vs_tool_correct']}/{m['total']} | "
            f"{m['tool_family_correct']}/{m['tool_family_total']} | "
            f"{m['min_correct_margin']:.6f} | "
            f"{'—' if m['max_wrong_margin'] is None else f'{m['max_wrong_margin']:.6f}'} |"
        )
    lines += ["",
        f"Strict full + INT4 third confirmation: **{report['strict_third_confirmation']}**",
        f"Shared full→INT4 third confirmation: **{report['shared_third_confirmation']}**", "",
        "## Misses", "",
    ]
    any_miss = False
    for key in ("full", "int4", "full_heads_on_int4"):
        misses = [r for r in report["routers"][key]["third"]["rows"] if not r["correct"]]
        for row in misses:
            any_miss = True
            lines.append(f"- **{key} / {row['id']}**: `{row['truth']}` → `{row['predicted']}`, margin={row['margin']:.6f}; {row['user']}")
    if not any_miss:
        lines.append("None.")
    lines += ["", f"Interpretation: {report['interpretation']}", "", "No model or production state was changed.\n"]
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

    train_cases = hier.training_cases()
    counts = validate(train_cases)

    with tempfile.TemporaryDirectory(prefix="ember-router-third-") as td:
        work = Path(td)
        archive = ev.download_verified(ev.PACKAGE_URL, work / "ember.zip", ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        import sys
        sys.path.insert(0, str(work / "src" / "ember"))

        api = HfApi(token=token)
        owner = api.whoami()["name"]
        repo = f"{owner}/{held.MODEL_NAME}"

        full_model, full_tok, _ = held.load_full(repo, work / "full", token)
        full_model.eval()
        full_train = hier.extract(full_model, full_tok, train_cases)
        full_third = hier.extract(full_model, full_tok, THIRD)
        full_heads = hier.prepare_heads(full_train)
        full_metrics = evaluate_fresh(full_heads, full_third)

        int4_model, int4_tok = held.load_int4(repo, work / "int4", token)
        int4_model.eval()
        int4_train = hier.extract(int4_model, int4_tok, train_cases)
        int4_third = hier.extract(int4_model, int4_tok, THIRD)
        int4_heads = hier.prepare_heads(int4_train)
        int4_metrics = evaluate_fresh(int4_heads, int4_third)

        cross_metrics = evaluate_fresh(full_heads, int4_third)

        strict = exact(full_metrics) and exact(int4_metrics)
        shared = exact(cross_metrics)
        if strict and shared:
            interpretation = (
                "The hierarchical block-3 router generalized exactly on the untouched 100-case battery in full, INT4, and shared-head modes. "
                "Next: test router-controlled generation with authoritative direct/tool gating and family-constrained schemas while keeping Ember weights frozen."
            )
        elif strict:
            interpretation = (
                "Precision-specific hierarchical routers passed the untouched 100-case battery exactly, but one shared head did not. "
                "Next: use precision-specific heads for router-controlled generation testing."
            )
        else:
            interpretation = (
                "The hierarchical router still has untouched routing misses. Do not integrate or save it as v0.0.54; inspect only the new misses before changing the architecture."
            )

        report = {
            "schema_version": 1,
            "diagnostic": "ember-v054-hierarchical-router-third-confirmation-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_repo": repo,
            "full_checkpoint_sha256": held.BEST_SHA256,
            "int4_checkpoint_sha256": held.INT4_SHA256,
            "representation": REPRESENTATION,
            "training_count": len(train_cases),
            "training_counts": {label: sum(probe.label_of(c) == label for c in train_cases) for label in probe.LABELS},
            "third_count": len(THIRD),
            "third_counts": counts,
            "architecture_frozen_before_test": True,
            "routers": {
                "full": {"binary_cv": full_heads["binary_cv"], "family_cv": full_heads["family_cv"], "third": full_metrics},
                "int4": {"binary_cv": int4_heads["binary_cv"], "family_cv": int4_heads["family_cv"], "third": int4_metrics},
                "full_heads_on_int4": {"binary_cv": None, "family_cv": None, "third": cross_metrics},
            },
            "strict_third_confirmation": strict,
            "shared_third_confirmation": shared,
            "ember_weights_changed": False,
            "production_changed": False,
            "router_integrated": False,
            "interpretation": interpretation,
            "elapsed_seconds": time.monotonic() - started,
        }
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (OUT / "summary.md").write_text(summary_md(report), encoding="utf-8")
        print(json.dumps({
            "event": "router_third_confirmation_complete",
            "full": compact(full_metrics),
            "int4": compact(int4_metrics),
            "full_heads_on_int4": compact(cross_metrics),
            "strict_third_confirmation": strict,
            "shared_third_confirmation": shared,
            "interpretation": interpretation,
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
