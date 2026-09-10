"""Ember v0.0.54 extractive executor v2 with a sixth fresh confirmation set.

Fixes two generic parser defects discovered by the first extractive executor:
- arithmetic phrases ending in sentence punctuation left a trailing dot attached
  to the final number;
- location extraction treated an internal period (for example St. John's) as
  the end of the location.

Ember v0.0.53 INT4 remains frozen. The already-established training-selected
multi-layer router is refit from the same 160 router-training prompts. The v2
resolver is tested on the prior fourth and fifth tool sets plus a new sixth
balanced 50-case set. The sixth set is not used for fitting or selection.

Strict gate:
- prior four router sets still exact (170/170);
- fifth routing still exact (50/50);
- sixth routing exact (50/50);
- fourth/fifth/sixth tool arguments all exact (120/120).

No Ember checkpoint or production pointer is changed. Candidate router/resolver
artifacts are written only to the workflow artifact if every gate passes.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import zipfile

from huggingface_hub import HfApi
import torch

from jobs import ember_hf_eval as ev
from jobs import ember_v053_heldout_routing as held
from jobs import ember_v054_frozen_router_probe as probe
from jobs import ember_v054_router_confirmation as confirm
from jobs import ember_v054_router_calibration as cal
from jobs import ember_v054_multilayer_router as multi
from jobs import ember_v054_int4_controlled_generation as control
from jobs import ember_v054_int4_extractive_executor as v1

OUT = Path("v054-extractive-v2")

SYSTEM_K = (
    "You are Ember. Route the request to a direct reply or exactly one tool: weather, calculator, web_search, get_time. "
    "Use tools only for live weather, requested arithmetic, current web facts, or current local time. "
    "Writing, explaining, summarizing, comparing, and classifying are direct tasks."
)
SYSTEM_L = (
    "You are Ember. Available tools are weather, calculator, web_search, and get_time. "
    "Choose based on the user's task rather than keywords; answer directly unless live information or arithmetic is actually needed."
)


def p(system: str, user: str) -> str:
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"


def direct(cid: str, user: str, system: str = SYSTEM_K) -> dict:
    return {"id": cid, "kind": "direct_response", "user": user, "prompt": p(system, user)}


def tool(cid: str, user: str, name: str, expected, system: str = SYSTEM_K) -> dict:
    return {"id": cid, "kind": "tool_call", "user": user, "prompt": p(system, user), "expected_tool": name, "expected": expected}


SIXTH = [
    direct("sixth_d01", "Label the sentiment: The current weather card looks much better now.", SYSTEM_L),
    direct("sixth_d02", "Rewrite this title: latest calculator layout cleanup."),
    direct("sixth_d03", "Explain what a time zone is without checking the current time.", SYSTEM_L),
    direct("sixth_d04", "Summarize: the search page changed color and no live lookup behavior changed."),
    direct("sixth_d05", "Compare a spreadsheet and a calculator in two short sentences.", SYSTEM_L),
    direct("sixth_d06", "Classify this as a question or statement: The weather icon is blue."),
    direct("sixth_d07", "Define current status as a user-interface phrase.", SYSTEM_L),
    direct("sixth_d08", "Give two short steps for testing a cancel button."),
    direct("sixth_d09", "Rewrite this heading clearly: web search settings information."),
    direct("sixth_d10", "Write a friendly sentence saying the latest draft is ready.", SYSTEM_L),

    tool("sixth_w01", "What is the weather in St. Louis right now?", "weather", "St. Louis", SYSTEM_L),
    tool("sixth_w02", "Is it raining in Fort Worth at the moment?", "weather", "Fort Worth"),
    tool("sixth_w03", "Give me the live temperature in Brasília.", "weather", "Brasília", SYSTEM_L),
    tool("sixth_w04", "How is the weather in Washington, D.C. right now?", "weather", "Washington, D.C."),
    tool("sixth_w05", "Do I need an umbrella in Port of Spain at the moment?", "weather", "Port of Spain", SYSTEM_L),
    tool("sixth_w06", "Check the current weather in Kansas City.", "weather", "Kansas City"),
    tool("sixth_w07", "Is it sunny in Salt Lake City right now?", "weather", "Salt Lake City", SYSTEM_L),
    tool("sixth_w08", "What are the live weather conditions in Québec City?", "weather", "Québec City"),
    tool("sixth_w09", "Tell me the current temperature in Santa Fe.", "weather", "Santa Fe", SYSTEM_L),
    tool("sixth_w10", "What is the weather doing in Ho Chi Minh City right now?", "weather", "Ho Chi Minh City"),

    tool("sixth_c01", "Calculate 917 plus 1386.", "calculator", 917 + 1386, SYSTEM_L),
    tool("sixth_c02", "What is 93 multiplied by 54?", "calculator", 93 * 54),
    tool("sixth_c03", "Compute 9360 divided by 24.", "calculator", 9360 / 24, SYSTEM_L),
    tool("sixth_c04", "Calculate 17 percent of 840.", "calculator", 0.17 * 840),
    tool("sixth_c05", "What is 71 squared?", "calculator", 71 ** 2, SYSTEM_L),
    tool("sixth_c06", "Compute 64 plus 91 plus 135.", "calculator", 64 + 91 + 135),
    tool("sixth_c07", "Calculate 5100 minus 2187.", "calculator", 5100 - 2187, SYSTEM_L),
    tool("sixth_c08", "What is 12.75 multiplied by 32?", "calculator", 12.75 * 32),
    tool("sixth_c09", "Compute 15360 divided by 120.", "calculator", 15360 / 120, SYSTEM_L),
    tool("sixth_c10", "Calculate 38 percent of 725.", "calculator", 0.38 * 725),

    tool("sixth_s01", "Find the newest stable OCaml release.", "web_search", ("ocaml", "release"), SYSTEM_L),
    tool("sixth_s02", "What is the current stable NixOS release?", "web_search", ("nixos", "release")),
    tool("sixth_s03", "Find the latest official Argo CD release.", "web_search", ("argo", "cd", "release"), SYSTEM_L),
    tool("sixth_s04", "Look up the newest stable Envoy release.", "web_search", ("envoy", "release")),
    tool("sixth_s05", "Find a recent official NIST announcement.", "web_search", ("nist", "announcement"), SYSTEM_L),
    tool("sixth_s06", "What is the latest stable Meson release?", "web_search", ("meson", "release")),
    tool("sixth_s07", "Find the current stable CockroachDB release.", "web_search", ("cockroachdb", "release"), SYSTEM_L),
    tool("sixth_s08", "Look up the latest official Audacity release.", "web_search", ("audacity", "release")),
    tool("sixth_s09", "Find the newest stable LibreWolf release.", "web_search", ("librewolf", "release"), SYSTEM_L),
    tool("sixth_s10", "What is the current stable RHEL release?", "web_search", ("rhel", "release")),

    tool("sixth_t01", "What time is it in New Delhi right now?", "get_time", "New Delhi", SYSTEM_L),
    tool("sixth_t02", "Give me the current local time in São Paulo.", "get_time", "São Paulo"),
    tool("sixth_t03", "What is the time in San José at this moment?", "get_time", "San José", SYSTEM_L),
    tool("sixth_t04", "Tell me the current time in Abu Dhabi.", "get_time", "Abu Dhabi"),
    tool("sixth_t05", "What time is it in Ulaanbaatar right now?", "get_time", "Ulaanbaatar", SYSTEM_L),
    tool("sixth_t06", "Give me the local time in New York City at the moment.", "get_time", "New York City"),
    tool("sixth_t07", "What is the current time in Addis Ababa?", "get_time", "Addis Ababa", SYSTEM_L),
    tool("sixth_t08", "Tell me what time it is in Buenos Aires right now.", "get_time", "Buenos Aires"),
    tool("sixth_t09", "Give me the current local time in Panama City.", "get_time", "Panama City", SYSTEM_L),
    tool("sixth_t10", "What time is it in Kuala Lumpur at this moment?", "get_time", "Kuala Lumpur"),
]


def validate_sixth(train_cases):
    if len(SIXTH) != 50:
        raise RuntimeError(f"sixth set must contain 50 cases, got {len(SIXTH)}")
    counts = {label: sum(probe.label_of(c) == label for c in SIXTH) for label in probe.LABELS}
    if any(counts[label] != 10 for label in probe.LABELS):
        raise RuntimeError(f"sixth set is not 10/class: {counts}")
    ids = [c["id"] for c in SIXTH]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate sixth ids")
    forbidden = (
        {c["id"] for c in train_cases}
        | {c["id"] for c in held.CASES}
        | {c["id"] for c in confirm.CONFIRM}
        | {c["id"] for c in cal.THIRD_CONFIRM}
        | {c["id"] for c in multi.FOURTH_CONFIRM}
        | {c["id"] for c in v1.FIFTH}
    )
    overlap = set(ids) & forbidden
    if overlap:
        raise RuntimeError(f"sixth set overlaps prior ids: {sorted(overlap)}")
    return counts


def normalize(text: str) -> str:
    text = text.casefold().replace("’", "'")
    text = re.sub(r"[^a-z0-9.+*/%' -]+", " ", text)
    return " ".join(text.split())


def extract_location_v2(user: str) -> str | None:
    text = user.strip()
    # Stop only at an explicit temporal suffix or sentence-ending punctuation.
    # Internal dots/apostrophes/commas remain part of the place name.
    match = re.search(
        r"\b(?:in|for)\s+(.+?)(?=\s+(?:right now|at the moment|at this moment|currently|today|now)\b|[?!]\s*$|\.\s*$|$)",
        text,
        flags=re.I,
    )
    if not match:
        return None
    value = match.group(1).strip(" ,?!")
    return value or None


def extract_expression_v2(user: str) -> str | None:
    text = user.casefold().replace(",", "").strip()
    text = re.sub(r"[?.!]+\s*$", "", text)
    num = r"([0-9]+(?:\.[0-9]+)?)"

    m = re.search(num + r"\s+percent\s+of\s+" + num, text)
    if m:
        return f"({m.group(1)}/100)*{m.group(2)}"
    m = re.search(num + r"\s+squared\b", text)
    if m:
        return f"{m.group(1)}**2"
    m = re.search(num + r"\s+to\s+the\s+power\s+of\s+" + num, text)
    if m:
        return f"{m.group(1)}**{m.group(2)}"

    normalized = text
    normalized = re.sub(r"\bmultiplied\s+by\b", " * ", normalized)
    normalized = re.sub(r"\bdivided\s+by\b", " / ", normalized)
    normalized = re.sub(r"\btimes\b", " * ", normalized)
    normalized = re.sub(r"\bplus\b", " + ", normalized)
    normalized = re.sub(r"\bminus\b", " - ", normalized)
    tokens = re.findall(r"[0-9]+(?:\.[0-9]+)?|[+*/-]", normalized)
    if len(tokens) < 3 or len(tokens) % 2 == 0:
        return None
    for index, token in enumerate(tokens):
        if index % 2 == 0 and not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", token):
            return None
        if index % 2 == 1 and token not in {"+", "-", "*", "/"}:
            return None
    return "".join(tokens)


def resolve_v2(case: dict, routed_tool: str) -> dict:
    user = case["user"]
    if routed_tool == "weather":
        key, value = "location", extract_location_v2(user)
    elif routed_tool == "get_time":
        key, value = "timezone", extract_location_v2(user)
    elif routed_tool == "calculator":
        key, value = "expression", extract_expression_v2(user)
    elif routed_tool == "web_search":
        key, value = "query", user.strip().rstrip("?").strip()
    else:
        raise ValueError(routed_tool)
    payload = {"name": routed_tool, "arguments": {key: value}} if value else None
    return {"tool": routed_tool, "key": key, "value": value, "payload": payload}


def check(case: dict, resolved: dict) -> dict:
    truth = case["expected_tool"]
    if resolved["tool"] != truth or not resolved["value"]:
        return {"passed": False, "reason": "wrong_tool_or_missing_value"}
    value = resolved["value"]
    if truth in {"weather", "get_time"}:
        expected = normalize(str(case["expected"]))
        observed = normalize(value)
        return {"passed": expected in observed, "expected": case["expected"], "observed": value}
    if truth == "calculator":
        observed = control.safe_arithmetic(value)
        expected = float(case["expected"])
        passed = observed is not None and math.isclose(float(observed), expected, rel_tol=1e-9, abs_tol=1e-9)
        return {"passed": passed, "expression": value, "expected_result": expected, "observed_result": observed}
    if truth == "web_search":
        observed = normalize(value)
        required = [normalize(term) for term in case["expected"]]
        return {"passed": all(term in observed for term in required), "query": value, "required": required}
    raise ValueError(truth)


def evaluate_args(cases, routes, label: str):
    rows = []
    by_family = {name: {"passed": 0, "total": 0} for name in control.TOOL_KEYS}
    passed = 0
    for case, route in zip(cases, routes):
        if case["kind"] != "tool_call":
            continue
        resolved = resolve_v2(case, route)
        result = check(case, resolved)
        ok = bool(result["passed"])
        passed += int(ok)
        truth = case["expected_tool"]
        by_family[truth]["total"] += 1
        by_family[truth]["passed"] += int(ok)
        row = {"id": case["id"], "truth": truth, "route": route, "resolved": resolved, "check": result, "passed": ok}
        rows.append(row)
        if not ok:
            print(json.dumps({"event": "v2_argument_miss", "set": label, **row}, ensure_ascii=False), flush=True)
    return {"passed": passed, "total": len(rows), "by_family": by_family, "rows": rows}


def fourth_cases():
    return v1.fourth_expected_cases()


def routes_for(heads, rows):
    routes, margins = multi.hierarchical_predict(heads, rows)
    return routes, margins


def routing_score(cases, routes, margins):
    truth = [probe.label_of(c) for c in cases]
    rows = [
        {"id": c["id"], "truth": t, "route": r, "correct": t == r, "margin": float(m)}
        for c, t, r, m in zip(cases, truth, routes, margins)
    ]
    return {"correct": sum(int(r["correct"]) for r in rows), "total": len(rows), "rows": rows}


def summary(report):
    lines = [
        "# Ember v0.0.54 extractive executor v2",
        "",
        "Frozen v0.0.53 INT4 base + exact multi-layer router + deterministic resolver v2.",
        "Resolver fixes terminal punctuation in arithmetic and internal punctuation in place names.",
        "",
        "| Gate | Result |",
        "| --- | ---: |",
        f"| Prior router regression | {report['prior_router_cases']}/170 |",
        f"| Fifth routing regression | {report['fifth_routing']['correct']}/50 |",
        f"| Sixth fresh routing | {report['sixth_routing']['correct']}/50 |",
        f"| Fourth arguments | {report['fourth_arguments']['passed']}/40 |",
        f"| Fifth arguments | {report['fifth_arguments']['passed']}/40 |",
        f"| Sixth fresh arguments | {report['sixth_arguments']['passed']}/40 |",
        "",
        "## Sixth arguments by family",
        "",
        "| Family | Passed |",
        "| --- | ---: |",
    ]
    for family, row in report["sixth_arguments"]["by_family"].items():
        lines.append(f"| {family} | {row['passed']}/{row['total']} |")
    lines += [
        "",
        f"Strict v2 executor pass: **{report['strict_pass']}**",
        f"Interpretation: {report['interpretation']}",
        "",
        "No Ember checkpoint or production pointer was changed.",
    ]
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

    train_cases = cal.training_cases()
    sixth_counts = validate_sixth(train_cases)

    with tempfile.TemporaryDirectory(prefix="ember-extractive-v2-") as td:
        work = Path(td)
        archive = ev.download_verified(ev.PACKAGE_URL, work / "ember.zip", ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        import sys
        sys.path.insert(0, str(work / "src" / "ember"))

        api = HfApi(token=token)
        owner = api.whoami()["name"]
        repo = f"{owner}/{held.MODEL_NAME}"
        model, tokenizer = held.load_int4(repo, work / "int4", token)
        model.eval()

        train_rows = multi.extract(model, tokenizer, train_cases)
        old_rows = multi.extract(model, tokenizer, held.CASES)
        second_rows = multi.extract(model, tokenizer, confirm.CONFIRM)
        third_rows = multi.extract(model, tokenizer, cal.THIRD_CONFIRM)
        fourth_rows = multi.extract(model, tokenizer, multi.FOURTH_CONFIRM)
        fifth_rows = multi.extract(model, tokenizer, v1.FIFTH)
        sixth_rows = multi.extract(model, tokenizer, SIXTH)
        heads = multi.prepare_heads(train_rows)

        prior_eval = multi.evaluate_all(heads, old_rows, second_rows, third_rows, fourth_rows)
        if not multi.exact_all(prior_eval):
            raise RuntimeError("INT4 router no longer reproduces exact 170/170 prior routing")

        fifth_routes, fifth_margins = routes_for(heads, fifth_rows)
        sixth_routes, sixth_margins = routes_for(heads, sixth_rows)
        fifth_routing = routing_score(v1.FIFTH, fifth_routes, fifth_margins)
        sixth_routing = routing_score(SIXTH, sixth_routes, sixth_margins)

        fourth_routes_all, _ = routes_for(heads, fourth_rows)
        fourth_tool_routes = [r for r, c in zip(fourth_routes_all, multi.FOURTH_CONFIRM) if c["kind"] == "tool_call"]
        fourth_args = evaluate_args(fourth_cases(), fourth_tool_routes, "fourth")
        fifth_args = evaluate_args(v1.FIFTH, fifth_routes, "fifth")
        sixth_args = evaluate_args(SIXTH, sixth_routes, "sixth")

        strict = (
            fifth_routing["correct"] == 50
            and sixth_routing["correct"] == 50
            and fourth_args["passed"] == 40
            and fifth_args["passed"] == 40
            and sixth_args["passed"] == 40
        )

        if strict:
            multi.save_heads(OUT / "router-int4-extractive-v2.pt", heads, "int4")
            resolver_spec = {
                "schema_version": 2,
                "kind": "ember-extractive-tool-argument-resolver",
                "supported_tools": control.TOOL_KEYS,
                "location_strategy": "extract after in/for; terminate only at explicit temporal suffix or sentence-ending punctuation; preserve internal dots/apostrophes/commas",
                "calculator_strategy": "strip terminal punctuation; normalize percent/squared/power and explicit plus/minus/multiplied-by/divided-by/times phrases to a safe expression",
                "web_search_strategy": "use complete user request as query",
                "generation_required": False,
            }
            (OUT / "resolver-v2-spec.json").write_text(json.dumps(resolver_spec, indent=2, sort_keys=True) + "\n")
            interpretation = (
                "The frozen INT4 router plus extractive resolver v2 cleared 100/100 fifth+sixth routing and 120/120 tool arguments across three independent tool sets. "
                "The tool-execution path is now strong enough for an adversarial resolver check. Direct-response language quality remains a separate blocker before Ember promotion."
            )
        else:
            routing_misses = [r["id"] for r in sixth_routing["rows"] if not r["correct"]]
            arg_misses = [r["id"] for r in sixth_args["rows"] if not r["passed"]]
            interpretation = (
                "Resolver v2 fixed the known parser defects but at least one strict routing/argument gate remains. "
                f"Sixth routing misses={routing_misses}; sixth argument misses={arg_misses}. Keep Ember frozen and repair only the router/resolver."
            )

        report = {
            "schema_version": 2,
            "diagnostic": "ember-v054-int4-extractive-executor-v2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_repo": repo,
            "int4_checkpoint_sha256": held.INT4_SHA256,
            "router": {
                "binary_representation": heads["binary"]["representation"],
                "family_representation": heads["family"]["representation"],
                "binary_cv": heads["binary_selected"]["cv_best"],
                "family_cv": heads["family_selected"]["cv_best"],
            },
            "prior_router_cases": 170,
            "fifth_routing": fifth_routing,
            "sixth_routing": sixth_routing,
            "sixth_counts": sixth_counts,
            "fourth_arguments": fourth_args,
            "fifth_arguments": fifth_args,
            "sixth_arguments": sixth_args,
            "strict_pass": strict,
            "candidate_artifacts_written": strict,
            "ember_weights_changed": False,
            "production_changed": False,
            "interpretation": interpretation,
            "elapsed_seconds": time.monotonic() - started,
        }
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        (OUT / "summary.md").write_text(summary(report), encoding="utf-8")
        print(json.dumps({
            "event": "extractive_v2_complete",
            "router": {"binary": heads["binary"]["representation"], "family": heads["family"]["representation"]},
            "fifth_routing": f"{fifth_routing['correct']}/50",
            "sixth_routing": f"{sixth_routing['correct']}/50",
            "fourth_arguments": f"{fourth_args['passed']}/40",
            "fifth_arguments": f"{fifth_args['passed']}/40",
            "sixth_arguments": f"{sixth_args['passed']}/40",
            "sixth_by_family": sixth_args["by_family"],
            "strict_pass": strict,
            "interpretation": interpretation,
            "elapsed_seconds": report["elapsed_seconds"],
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
