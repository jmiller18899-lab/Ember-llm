"""INT4 router + extractive tool-argument executor for Ember v0.0.54.

The previous controlled-generation diagnostic proved the frozen INT4 router is
exact on 170/170 routing cases, but Ember's LM head generated 0/40 correct tool
argument values. This diagnostic stops asking the LM head to invent/copy those
values.

Architecture under test:
- frozen Ember INT4 hidden states -> training-selected multi-layer router;
- router chooses direct/weather/calculator/web_search/get_time;
- deterministic extractive resolver copies/normalizes tool arguments from the
  user's request;
- no Ember generation is used for tool arguments.

The resolver is generic for these four schemas and never receives expected test
answers. It is checked on the prior fourth-set 40 tools plus a NEW fifth balanced
50-case set (10 direct, 10 per tool family). The fifth set is not used for router
fitting or representation selection.

No Ember checkpoint or production pointer is changed. If strict gates pass, the
INT4 router head and resolver specification are written only to the workflow
artifact as a candidate component.
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

OUT = Path("v054-int4-extractive-executor")

SYSTEM_I = (
    "You are Ember. Route the request to a direct response or one tool: weather, calculator, web_search, get_time. "
    "Use tools only for live weather, explicit arithmetic, current web facts, or current local time. "
    "Writing, explaining, comparing, and classifying are direct tasks."
)
SYSTEM_J = (
    "You are Ember. Available tools are weather, calculator, web_search, and get_time. "
    "Choose by task semantics rather than keywords; answer directly unless live information or arithmetic is required."
)


def prompt(system: str, user: str) -> str:
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"


def direct(cid: str, user: str, system: str = SYSTEM_I) -> dict:
    return {"id": cid, "kind": "direct_response", "user": user, "prompt": prompt(system, user)}


def tool(cid: str, user: str, name: str, expected, system: str = SYSTEM_I) -> dict:
    return {
        "id": cid,
        "kind": "tool_call",
        "user": user,
        "prompt": prompt(system, user),
        "expected_tool": name,
        "expected": expected,
    }


FIFTH = [
    direct("fifth_d01", "Label the sentiment: The weather card redesign looks excellent.", SYSTEM_J),
    direct("fifth_d02", "Rewrite this heading clearly: current clock panel copy cleanup."),
    direct("fifth_d03", "Explain what division means without calculating a specific example.", SYSTEM_J),
    direct("fifth_d04", "Summarize: the search input moved right and the network behavior did not change."),
    direct("fifth_d05", "Compare a timer and a calendar in two short sentences.", SYSTEM_J),
    direct("fifth_d06", "Classify this as a question or statement: The latest build is ready."),
    direct("fifth_d07", "Define a weather alert without checking live weather.", SYSTEM_J),
    direct("fifth_d08", "Give two short steps for testing a submit button."),
    direct("fifth_d09", "Rewrite this title: calculator keyboard spacing issue.", SYSTEM_J),
    direct("fifth_d10", "Write one friendly sentence thanking someone for testing the app."),

    tool("fifth_w01", "What is the weather in Tulsa right now?", "weather", "Tulsa", SYSTEM_J),
    tool("fifth_w02", "Is it raining in Aberdeen at the moment?", "weather", "Aberdeen"),
    tool("fifth_w03", "Give me the live temperature in Kobe.", "weather", "Kobe", SYSTEM_J),
    tool("fifth_w04", "How is the weather in Granada right now?", "weather", "Granada"),
    tool("fifth_w05", "Do I need an umbrella in Dunedin at the moment?", "weather", "Dunedin", SYSTEM_J),
    tool("fifth_w06", "Check the current weather in Pittsburgh.", "weather", "Pittsburgh"),
    tool("fifth_w07", "Is it sunny in Nice right now?", "weather", "Nice", SYSTEM_J),
    tool("fifth_w08", "What are the live weather conditions in Winnipeg?", "weather", "Winnipeg"),
    tool("fifth_w09", "Tell me the current temperature in Bologna.", "weather", "Bologna", SYSTEM_J),
    tool("fifth_w10", "What is the weather doing in Moncton right now?", "weather", "Moncton"),

    tool("fifth_c01", "Calculate 835 plus 1264.", "calculator", 835 + 1264, SYSTEM_J),
    tool("fifth_c02", "What is 79 multiplied by 68?", "calculator", 79 * 68),
    tool("fifth_c03", "Compute 8640 divided by 36.", "calculator", 8640 / 36, SYSTEM_J),
    tool("fifth_c04", "Calculate 23 percent of 780.", "calculator", 0.23 * 780),
    tool("fifth_c05", "What is 63 squared?", "calculator", 63 ** 2, SYSTEM_J),
    tool("fifth_c06", "Compute 59 plus 83 plus 147.", "calculator", 59 + 83 + 147),
    tool("fifth_c07", "Calculate 4200 minus 1735.", "calculator", 4200 - 1735, SYSTEM_J),
    tool("fifth_c08", "What is 11.5 multiplied by 26?", "calculator", 11.5 * 26),
    tool("fifth_c09", "Compute 12288 divided by 96.", "calculator", 12288 / 96, SYSTEM_J),
    tool("fifth_c10", "Calculate 44 percent of 650.", "calculator", 0.44 * 650),

    tool("fifth_s01", "Find the newest stable Haskell GHC release.", "web_search", ("haskell", "ghc", "release"), SYSTEM_J),
    tool("fifth_s02", "What is the current stable Gentoo release information?", "web_search", ("gentoo", "release")),
    tool("fifth_s03", "Find the latest official Pulumi release.", "web_search", ("pulumi", "release"), SYSTEM_J),
    tool("fifth_s04", "Look up the newest stable Mosquitto release.", "web_search", ("mosquitto", "release")),
    tool("fifth_s05", "Find a recent official CDC announcement.", "web_search", ("cdc", "announcement"), SYSTEM_J),
    tool("fifth_s06", "What is the latest stable Clang release?", "web_search", ("clang", "release")),
    tool("fifth_s07", "Find the current stable Neo4j release.", "web_search", ("neo4j", "release"), SYSTEM_J),
    tool("fifth_s08", "Look up the latest official GIMP release.", "web_search", ("gimp", "release")),
    tool("fifth_s09", "Find the newest stable Tor Browser release.", "web_search", ("tor", "browser", "release"), SYSTEM_J),
    tool("fifth_s10", "What is the current stable CentOS Stream release?", "web_search", ("centos", "stream", "release")),

    tool("fifth_t01", "What time is it in Tallinn right now?", "get_time", "Tallinn", SYSTEM_J),
    tool("fifth_t02", "Give me the current local time in Kathmandu.", "get_time", "Kathmandu"),
    tool("fifth_t03", "What is the time in Medellín at this moment?", "get_time", "Medellín", SYSTEM_J),
    tool("fifth_t04", "Tell me the current time in Sofia.", "get_time", "Sofia"),
    tool("fifth_t05", "What time is it in Muscat right now?", "get_time", "Muscat", SYSTEM_J),
    tool("fifth_t06", "Give me the local time in Busan at the moment.", "get_time", "Busan"),
    tool("fifth_t07", "What is the current time in Lagos?", "get_time", "Lagos", SYSTEM_J),
    tool("fifth_t08", "Tell me what time it is in Strasbourg right now.", "get_time", "Strasbourg"),
    tool("fifth_t09", "Give me the current local time in Guayaquil.", "get_time", "Guayaquil", SYSTEM_J),
    tool("fifth_t10", "What time is it in Hyderabad at this moment?", "get_time", "Hyderabad"),
]


def validate_fifth(train_cases):
    if len(FIFTH) != 50:
        raise RuntimeError(f"fifth set must have 50 cases, got {len(FIFTH)}")
    counts = {label: sum(probe.label_of(c) == label for c in FIFTH) for label in probe.LABELS}
    if any(counts[label] != 10 for label in probe.LABELS):
        raise RuntimeError(f"fifth set not 10/class: {counts}")
    ids = [c["id"] for c in FIFTH]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate fifth ids")
    forbidden = (
        {c["id"] for c in train_cases}
        | {c["id"] for c in held.CASES}
        | {c["id"] for c in confirm.CONFIRM}
        | {c["id"] for c in cal.THIRD_CONFIRM}
        | {c["id"] for c in multi.FOURTH_CONFIRM}
    )
    overlap = set(ids) & forbidden
    if overlap:
        raise RuntimeError(f"fifth ids overlap prior sets: {sorted(overlap)}")
    return counts


def normalize(text: str) -> str:
    text = text.casefold().replace("’", "'")
    text = re.sub(r"[^a-z0-9.+*/%' -]+", " ", text)
    return " ".join(text.split())


def extract_location(user: str) -> str | None:
    text = user.strip()
    patterns = [
        r"\b(?:in|for)\s+(.+?)(?=\s+(?:right now|at the moment|at this moment|now|today|currently)\b|[?.!]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            value = match.group(1).strip(" ,.? !")
            return value or None
    return None


def number(text: str) -> str:
    return text.strip().replace(",", "")


def extract_expression(user: str) -> str | None:
    t = user.casefold().replace(",", "").strip()
    num = r"([0-9]+(?:\.[0-9]+)?)"
    m = re.search(num + r"\s+percent\s+of\s+" + num, t)
    if m:
        return f"({number(m.group(1))}/100)*{number(m.group(2))}"
    m = re.search(num + r"\s+squared\b", t)
    if m:
        return f"{number(m.group(1))}**2"
    m = re.search(num + r"\s+to\s+the\s+power\s+of\s+" + num, t)
    if m:
        return f"{number(m.group(1))}**{number(m.group(2))}"

    words = re.sub(r"[^a-z0-9.+-]+", " ", t)
    tokens = words.split()
    out = []
    ops = {"plus": "+", "minus": "-", "times": "*"}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", tok):
            out.append(tok)
        elif tok in ops:
            out.append(ops[tok])
        elif tok == "multiplied" and i + 1 < len(tokens) and tokens[i + 1] == "by":
            out.append("*"); i += 1
        elif tok == "divided" and i + 1 < len(tokens) and tokens[i + 1] == "by":
            out.append("/"); i += 1
        i += 1
    expr = "".join(out)
    if not re.fullmatch(r"[0-9.]+(?:[+*/-][0-9.]+)+", expr):
        return None
    return expr


def extract_query(user: str) -> str:
    # A complete user request is a valid search query and avoids lossy rewriting.
    return user.strip().rstrip("?").strip()


def resolve(case: dict, routed_tool: str) -> dict:
    user = case["user"]
    if routed_tool == "weather":
        value = extract_location(user)
        key = "location"
    elif routed_tool == "get_time":
        value = extract_location(user)
        key = "timezone"
    elif routed_tool == "calculator":
        value = extract_expression(user)
        key = "expression"
    elif routed_tool == "web_search":
        value = extract_query(user)
        key = "query"
    else:
        raise ValueError(routed_tool)
    payload = {"name": routed_tool, "arguments": {key: value}} if value else None
    return {"tool": routed_tool, "key": key, "value": value, "payload": payload}


def check_case(case: dict, resolved: dict) -> dict:
    tool_name = case["expected_tool"]
    if resolved["tool"] != tool_name or resolved["value"] is None:
        return {"passed": False, "reason": "wrong_tool_or_missing_value"}
    value = resolved["value"]
    if tool_name in {"weather", "get_time"}:
        expected = normalize(str(case["expected"]))
        observed = normalize(value)
        return {"passed": expected in observed, "expected": case["expected"], "observed": value}
    if tool_name == "calculator":
        observed = control.safe_arithmetic(value)
        expected = float(case["expected"])
        passed = observed is not None and math.isclose(float(observed), expected, rel_tol=1e-9, abs_tol=1e-9)
        return {"passed": passed, "expected_result": expected, "observed_result": observed, "expression": value}
    if tool_name == "web_search":
        observed = normalize(value)
        required = [normalize(term) for term in case["expected"]]
        return {"passed": all(term in observed for term in required), "required": required, "query": value}
    raise ValueError(tool_name)


def evaluate_resolver(cases, routes):
    rows = []
    passed = 0
    by_family = {name: {"passed": 0, "total": 0} for name in control.TOOL_KEYS}
    for case, route in zip(cases, routes):
        if case["kind"] != "tool_call":
            continue
        resolved = resolve(case, route)
        check = check_case(case, resolved)
        ok = bool(check["passed"])
        passed += int(ok)
        truth = case["expected_tool"]
        by_family[truth]["total"] += 1
        by_family[truth]["passed"] += int(ok)
        rows.append({"id": case["id"], "truth": truth, "route": route, "resolved": resolved, "check": check, "passed": ok})
        print(json.dumps({"event": "extractive_case", "id": case["id"], "truth": truth, "route": route,
                          "value": resolved["value"], "check": check, "passed": ok}, ensure_ascii=False), flush=True)
    return {"passed": passed, "total": len(rows), "by_family": by_family, "rows": rows}


def fourth_expected_cases():
    rows = []
    for case in multi.FOURTH_CONFIRM:
        if case["kind"] != "tool_call":
            continue
        copied = dict(case)
        if case["expected_tool"] in {"weather", "get_time"}:
            copied["expected"] = control.EXPECTED_LOCATION[case["id"]]
        elif case["expected_tool"] == "calculator":
            copied["expected"] = control.EXPECTED_CALC[case["id"]]
        else:
            copied["expected"] = control.EXPECTED_SEARCH_TERMS[case["id"]]
        rows.append(copied)
    return rows


def summary_markdown(report):
    lines = [
        "# Ember v0.0.54 INT4 extractive executor",
        "",
        "Frozen INT4 Ember + training-selected multi-layer router + deterministic extractive argument resolver.",
        f"Prior router verification: {report['prior_router_cases']}/170 exact.",
        "",
        "| Gate | Result |",
        "| --- | ---: |",
        f"| Fifth-set routing | {report['fifth_routing']['correct']}/50 |",
        f"| Prior fourth-set argument extraction | {report['fourth_arguments']['passed']}/40 |",
        f"| Fresh fifth-set argument extraction | {report['fifth_arguments']['passed']}/40 |",
        "",
        "## Fifth arguments by family",
        "",
        "| Family | Passed |",
        "| --- | ---: |",
    ]
    for name, row in report["fifth_arguments"]["by_family"].items():
        lines.append(f"| {name} | {row['passed']}/{row['total']} |")
    lines += [
        "",
        f"Strict executor pass: **{report['strict_pass']}**",
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
    fifth_counts = validate_fifth(train_cases)

    with tempfile.TemporaryDirectory(prefix="ember-extractive-executor-") as td:
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
        fifth_rows = multi.extract(model, tokenizer, FIFTH)
        heads = multi.prepare_heads(train_rows)

        prior_eval = multi.evaluate_all(heads, old_rows, second_rows, third_rows, fourth_rows)
        if not multi.exact_all(prior_eval):
            raise RuntimeError("INT4 router no longer reproduces exact prior four-set routing")

        fifth_routes, fifth_margins = multi.hierarchical_predict(heads, fifth_rows)
        fifth_truth = [probe.label_of(c) for c in FIFTH]
        fifth_correct = sum(int(a == b) for a, b in zip(fifth_routes, fifth_truth))
        fifth_routing = {
            "correct": fifth_correct,
            "total": 50,
            "rows": [
                {"id": c["id"], "truth": truth, "route": route, "correct": truth == route, "margin": float(margin)}
                for c, truth, route, margin in zip(FIFTH, fifth_truth, fifth_routes, fifth_margins)
            ],
        }
        for row in fifth_routing["rows"]:
            if not row["correct"]:
                print(json.dumps({"event": "fifth_routing_miss", **row}), flush=True)

        # Prior fourth argument set: use its already-validated routes, but generic resolver only.
        fourth_tool_cases = fourth_expected_cases()
        fourth_tool_rows = [row for row, case in zip(fourth_rows, multi.FOURTH_CONFIRM) if case["kind"] == "tool_call"]
        fourth_routes_all, _ = multi.hierarchical_predict(heads, fourth_rows)
        fourth_tool_routes = [route for route, case in zip(fourth_routes_all, multi.FOURTH_CONFIRM) if case["kind"] == "tool_call"]
        fourth_args = evaluate_resolver(fourth_tool_cases, fourth_tool_routes)

        fifth_args = evaluate_resolver(FIFTH, fifth_routes)

        strict = fifth_correct == 50 and fourth_args["passed"] == 40 and fifth_args["passed"] == 40
        if strict:
            multi.save_heads(OUT / "router-int4-extractive.pt", heads, "int4")
            resolver_spec = {
                "schema_version": 1,
                "kind": "ember-extractive-tool-argument-resolver",
                "supported_tools": control.TOOL_KEYS,
                "location_strategy": "extract span after in/for before temporal suffix/punctuation",
                "calculator_strategy": "normalize explicit arithmetic language into safe expression",
                "web_search_strategy": "use full user request as query",
                "generation_required": False,
            }
            (OUT / "resolver-spec.json").write_text(json.dumps(resolver_spec, indent=2, sort_keys=True) + "\n")
            interpretation = (
                "The frozen INT4 router plus extractive resolver cleared both the prior 40-tool set and a fresh 40-tool set while routing the fresh 50/50 exactly. "
                "This is a viable candidate executor architecture. Next: run one final adversarial/phrasing confirmation and direct-response quality check before integration."
            )
        else:
            misses = [r["id"] for r in fifth_routing["rows"] if not r["correct"]]
            arg_misses = [r["id"] for r in fifth_args["rows"] if not r["passed"]]
            interpretation = (
                "The extractive executor improved argument correctness dramatically but did not clear every fresh gate. "
                f"Inspect routing misses {misses} and argument misses {arg_misses}; keep Ember frozen and repair only the router/resolver component."
            )

        report = {
            "schema_version": 1,
            "diagnostic": "ember-v054-int4-extractive-executor-v1",
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
            "prior_router_evaluation": prior_eval,
            "fifth_counts": fifth_counts,
            "fifth_routing": fifth_routing,
            "fourth_arguments": fourth_args,
            "fifth_arguments": fifth_args,
            "strict_pass": strict,
            "candidate_artifacts_written": strict,
            "ember_weights_changed": False,
            "production_changed": False,
            "interpretation": interpretation,
            "elapsed_seconds": time.monotonic() - started,
        }
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        (OUT / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
        print(json.dumps({
            "event": "extractive_executor_complete",
            "router": {"binary": heads["binary"]["representation"], "family": heads["family"]["representation"]},
            "fifth_routing": f"{fifth_correct}/50",
            "fourth_arguments": f"{fourth_args['passed']}/40",
            "fifth_arguments": f"{fifth_args['passed']}/40",
            "fifth_by_family": fifth_args["by_family"],
            "strict_pass": strict,
            "interpretation": interpretation,
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
