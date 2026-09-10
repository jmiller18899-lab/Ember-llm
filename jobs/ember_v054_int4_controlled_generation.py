"""INT4 router-controlled generation diagnostic for Ember v0.0.54.

This is the first end-to-end test of the new architecture after the frozen
multi-layer router achieved exact routing across all four INT4 evaluation sets.
Ember's language weights remain frozen.

Pipeline:
1. Refit the training-selected INT4 router on the fixed 160 routing-training
   prompts and require exact 170/170 routing across old20 + second50 + third50 +
   fourth50 before any controlled generation is scored.
2. DIRECT route: greedily generate a normal reply while masking tool/control
   markers and stopping at <|endoftext|>.
3. TOOL route: the router supplies the tool family and deterministic JSON
   scaffold. Ember generates only the argument VALUE from the user's prompt.
   The scaffold is then closed externally. This isolates argument extraction
   from the already-solved tool-family routing problem.

The untouched fourth 50-case set is used for the generation test: 10 direct and
40 tool cases. Tool argument values are checked semantically by family.

No Ember checkpoint, router artifact, or production pointer is saved or changed.
CPU-only diagnostic.
"""
from __future__ import annotations

import ast
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

OUT = Path("v054-int4-controlled-generation")
MAX_DIRECT_TOKENS = 64
MAX_VALUE_TOKENS = 32

TOOL_KEYS = {
    "weather": "location",
    "calculator": "expression",
    "web_search": "query",
    "get_time": "timezone",
}

EXPECTED_LOCATION = {
    "fourth_w01": "Fargo",
    "fourth_w02": "Cork",
    "fourth_w03": "Sendai",
    "fourth_w04": "Seville",
    "fourth_w05": "Christchurch",
    "fourth_w06": "Cleveland",
    "fourth_w07": "Marseille",
    "fourth_w08": "Edmonton",
    "fourth_w09": "Naples",
    "fourth_w10": "St. John's",
    "fourth_t01": "Oslo",
    "fourth_t02": "Ho Chi Minh City",
    "fourth_t03": "Asunción",
    "fourth_t04": "Bucharest",
    "fourth_t05": "Doha",
    "fourth_t06": "Fukuoka",
    "fourth_t07": "Dakar",
    "fourth_t08": "Luxembourg",
    "fourth_t09": "La Paz",
    "fourth_t10": "Chennai",
}

EXPECTED_CALC = {
    "fourth_c01": 692 + 1448,
    "fourth_c02": 87 * 62,
    "fourth_c03": 7560 / 24,
    "fourth_c04": 0.16 * 875,
    "fourth_c05": 58 ** 2,
    "fourth_c06": 48 + 76 + 125,
    "fourth_c07": 3100 - 1276,
    "fourth_c08": 9.25 * 28,
    "fourth_c09": 10752 / 84,
    "fourth_c10": 0.36 * 825,
}

EXPECTED_SEARCH_TERMS = {
    "fourth_s01": ("scala", "release"),
    "fourth_s02": ("opensuse", "release"),
    "fourth_s03": ("helm", "release"),
    "fourth_s04": ("nats", "release"),
    "fourth_s05": ("usgs", "announcement"),
    "fourth_s06": ("gcc", "release"),
    "fourth_s07": ("couchdb", "release"),
    "fourth_s08": ("inkscape", "release"),
    "fourth_s09": ("vivaldi", "release"),
    "fourth_s10": ("almalinux", "release"),
}


def marker_ids(tokenizer):
    contract = ev.special_token_contract(tokenizer)
    if not contract["atomic"] or not contract["unique"]:
        raise RuntimeError("special-token contract is not atomic+unique")
    ids = {marker: int(contract["signatures"][marker][0]) for marker in ev.SPECIAL_TOKENS}
    return ids


def stop_at_unescaped_quote(text: str):
    escaped = False
    for i, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            return text[:i], True
    return text, False


def normalize_text(text: str) -> str:
    text = text.casefold()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9.+*/%^ -]+", " ", text)
    return " ".join(text.split())


def generate_direct(model, tokenizer, prompt: str, specials: dict) -> dict:
    context = [int(x) for x in tokenizer.encode(prompt)]
    generated = []
    eot = specials["<|endoftext|>"]
    blocked = {
        token_id
        for marker, token_id in specials.items()
        if marker != "<|endoftext|>"
    }
    stopped = False
    with torch.inference_mode():
        for _ in range(MAX_DIRECT_TOKENS):
            if len(context) >= int(model.cfg.block_size):
                break
            x = torch.tensor([context], dtype=torch.long)
            logits, _ = model(x, None)
            next_logits = logits[0, -1].float().clone()
            for token_id in blocked:
                next_logits[token_id] = -float("inf")
            token_id = int(torch.argmax(next_logits).item())
            if token_id == eot:
                stopped = True
                break
            generated.append(token_id)
            context.append(token_id)
    text = tokenizer.decode(generated).strip()
    return {
        "text": text,
        "nonempty": len(ev.visible_text(text)) >= 3,
        "stopped_at_eot": stopped,
        "tokens": len(generated),
        "passed": len(ev.visible_text(text)) >= 3 and "<|tool|>" not in text,
    }


def generate_argument_value(model, tokenizer, prompt: str, tool_name: str, specials: dict) -> dict:
    key = TOOL_KEYS[tool_name]
    forced_prefix = f'<|tool|>\n{{"arguments":{{"{key}":"'
    context = [int(x) for x in tokenizer.encode(prompt + forced_prefix)]
    generated = []
    blocked = set(specials.values())
    closed = False
    raw_text = ""
    with torch.inference_mode():
        for _ in range(MAX_VALUE_TOKENS):
            if len(context) >= int(model.cfg.block_size):
                break
            x = torch.tensor([context], dtype=torch.long)
            logits, _ = model(x, None)
            next_logits = logits[0, -1].float().clone()
            for token_id in blocked:
                next_logits[token_id] = -float("inf")
            token_id = int(torch.argmax(next_logits).item())
            generated.append(token_id)
            context.append(token_id)
            raw_text = tokenizer.decode(generated)
            value, closed = stop_at_unescaped_quote(raw_text)
            if closed:
                break
    value, closed = stop_at_unescaped_quote(raw_text)
    value = value.strip()
    payload = {
        "name": tool_name,
        "arguments": {key: value},
    }
    return {
        "tool": tool_name,
        "key": key,
        "value": value,
        "raw_generated": raw_text,
        "closed_quote_seen": closed,
        "tokens": len(generated),
        "payload": payload,
        "payload_json": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    }


def safe_arithmetic(text: str):
    value = text.strip().lower().replace("×", "*").replace("÷", "/").replace("^", "**")
    match = re.fullmatch(r"\s*([+-]?[0-9]+(?:\.[0-9]+)?)\s*%\s*(?:of|\*)\s*([+-]?[0-9]+(?:\.[0-9]+)?)\s*", value)
    if match:
        return float(match.group(1)) / 100.0 * float(match.group(2))
    match = re.fullmatch(r"\s*([+-]?[0-9]+(?:\.[0-9]+)?)\s+percent\s+of\s+([+-]?[0-9]+(?:\.[0-9]+)?)\s*", value)
    if match:
        return float(match.group(1)) / 100.0 * float(match.group(2))
    try:
        tree = ast.parse(value, mode="eval")
    except SyntaxError:
        try:
            return float(value)
        except ValueError:
            return None

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = visit(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div): return a / b
            if isinstance(node.op, ast.Pow):
                if abs(b) > 10 or abs(a) > 1e6:
                    raise ValueError("unsafe exponent")
                return a ** b
        raise ValueError("unsupported arithmetic")

    try:
        result = visit(tree)
        return float(result) if math.isfinite(float(result)) else None
    except Exception:
        return None


def argument_pass(case: dict, tool_name: str, value: str) -> dict:
    cid = case["id"]
    normalized = normalize_text(value)
    if tool_name in {"weather", "get_time"}:
        expected = EXPECTED_LOCATION[cid]
        expected_norm = normalize_text(expected)
        passed = expected_norm in normalized
        return {"passed": passed, "expected": expected, "normalized": normalized}
    if tool_name == "calculator":
        expected = float(EXPECTED_CALC[cid])
        observed = safe_arithmetic(value)
        passed = observed is not None and math.isclose(float(observed), expected, rel_tol=1e-9, abs_tol=1e-9)
        return {"passed": passed, "expected_result": expected, "observed_result": observed, "normalized": normalized}
    if tool_name == "web_search":
        terms = EXPECTED_SEARCH_TERMS[cid]
        passed = all(term in normalized for term in terms)
        return {"passed": passed, "required_terms": list(terms), "normalized": normalized}
    raise ValueError(tool_name)


def router_predictions(heads, rows):
    predicted, margins = multi.hierarchical_predict(heads, rows)
    return predicted, margins


def verify_router(heads, old_rows, second_rows, third_rows, fourth_rows):
    evaluation = multi.evaluate_all(heads, old_rows, second_rows, third_rows, fourth_rows)
    if not multi.exact_all(evaluation):
        compact = {
            name: metrics["five_way_correct"]
            for name, metrics in evaluation.items()
        }
        raise RuntimeError(f"INT4 multi-layer router no longer reproduces exact four-set result: {compact}")
    return evaluation


def native_fourth(model, tokenizer) -> dict:
    results = []
    direct_pass = tool_pass = tool_name_correct = 0
    for case in multi.FOURTH_CONFIRM:
        completion = ev.generate_completion(model, tokenizer, torch, case["prompt"], {
            "max_new_tokens": 64,
            "temperature": 1.0,
            "top_k": 1,
            "seed": 20260910,
        })
        score = ev.score_case(case, completion)
        results.append({"id": case["id"], "kind": case["kind"], "completion": completion, "score": score})
        if case["kind"] == "direct_response":
            direct_pass += int(score["passed"])
        else:
            tool_pass += int(score["passed"])
            tool_name_correct += int(score.get("tool_name_matches", False))
    return {
        "direct_pass": direct_pass,
        "direct_total": 10,
        "tool_pass": tool_pass,
        "tool_total": 40,
        "tool_name_correct": tool_name_correct,
        "results": results,
    }


def summary_markdown(report):
    controlled = report["controlled"]
    native = report["native_baseline"]
    lines = [
        "# Ember v0.0.54 INT4 router-controlled generation",
        "",
        "Exact saved INT4 v0.0.53 step-9 base; Ember weights frozen.",
        f"Router: binary={report['router']['binary_representation']}, family={report['router']['family_representation']}; verified exact on 170/170 prior routing cases.",
        "Tool calls use router-selected deterministic structure; Ember generates only the argument value.",
        "",
        "| Metric | Native INT4 | Router-controlled |",
        "| --- | ---: | ---: |",
        f"| Direct routing/generation | {native['direct_pass']}/10 | {controlled['direct_pass']}/10 |",
        f"| Correct tool family/name | {native['tool_name_correct']}/40 | {controlled['tool_family_correct']}/40 |",
        f"| Valid structured tool payload | {native['tool_pass']}/40 | {controlled['structured_payload_valid']}/40 |",
        f"| Semantically correct argument value | — | {controlled['argument_pass']}/40 |",
        "",
        "## Argument extraction by family",
        "",
        "| Family | Passed |",
        "| --- | ---: |",
    ]
    for family, row in controlled["by_family"].items():
        lines.append(f"| {family} | {row['passed']}/{row['total']} |")
    lines += [
        "",
        f"Controlled-generation strict pass: **{report['strict_pass']}**",
        f"Interpretation: {report['interpretation']}",
        "",
        "No checkpoint, router, or production pointer was changed.",
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

    with tempfile.TemporaryDirectory(prefix="ember-int4-controlled-") as td:
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
        heads = multi.prepare_heads(train_rows)
        router_eval = verify_router(heads, old_rows, second_rows, third_rows, fourth_rows)

        predicted, margins = router_predictions(heads, fourth_rows)
        specials = marker_ids(tokenizer)
        controlled_rows = []
        direct_pass = tool_family_correct = structured_valid = argument_ok = 0
        by_family = {name: {"passed": 0, "total": 0} for name in ("weather", "calculator", "web_search", "get_time")}

        for case, route, margin_value in zip(multi.FOURTH_CONFIRM, predicted, margins):
            truth = probe.label_of(case)
            if case["kind"] == "direct_response":
                generated = generate_direct(model, tokenizer, case["prompt"], specials)
                passed = route == "direct" and generated["passed"]
                direct_pass += int(passed)
                controlled_rows.append({
                    "id": case["id"], "kind": "direct_response", "router": route,
                    "router_margin": float(margin_value), "generation": generated, "passed": passed,
                })
                print(json.dumps({"event":"controlled_direct", "id":case["id"], "router":route,
                                  "router_margin":float(margin_value), "passed":passed,
                                  "text":generated["text"][:180]}), flush=True)
            else:
                family_correct = route == truth
                tool_family_correct += int(family_correct)
                generated = generate_argument_value(model, tokenizer, case["prompt"], route, specials)
                structural = bool(generated["value"]) and generated["closed_quote_seen"] and route in TOOL_KEYS
                structured_valid += int(structural)
                arg = argument_pass(case, route, generated["value"]) if family_correct else {"passed": False, "reason": "wrong_router_family"}
                arg_passed = family_correct and structural and bool(arg["passed"])
                argument_ok += int(arg_passed)
                by_family[truth]["total"] += 1
                by_family[truth]["passed"] += int(arg_passed)
                controlled_rows.append({
                    "id": case["id"], "kind": "tool_call", "truth": truth,
                    "router": route, "router_margin": float(margin_value),
                    "family_correct": family_correct, "generation": generated,
                    "argument_check": arg, "passed": arg_passed,
                })
                print(json.dumps({"event":"controlled_tool", "id":case["id"], "truth":truth,
                                  "router":route, "router_margin":float(margin_value),
                                  "value":generated["value"], "closed":generated["closed_quote_seen"],
                                  "argument_check":arg, "passed":arg_passed}), flush=True)

        native = native_fourth(model, tokenizer)
        controlled = {
            "direct_pass": direct_pass,
            "direct_total": 10,
            "tool_family_correct": tool_family_correct,
            "tool_total": 40,
            "structured_payload_valid": structured_valid,
            "argument_pass": argument_ok,
            "by_family": by_family,
            "rows": controlled_rows,
        }
        strict = (
            direct_pass == 10
            and tool_family_correct == 40
            and structured_valid == 40
            and argument_ok == 40
        )
        if strict:
            interpretation = (
                "The exact INT4 router plus constrained scaffold produced correct direct/tool behavior and semantically correct arguments on all 50 fourth-set cases. "
                "Next: repeat on a fifth fresh generation set, then package the router+decoder as a separate candidate component without altering Ember weights."
            )
        else:
            failed = [r["id"] for r in controlled_rows if not r["passed"]]
            interpretation = (
                "Routing is solved, but constrained argument extraction still fails on some tool requests. "
                f"Localize only the failed value spans ({', '.join(failed[:12])}) before any integration; keep Ember and the router frozen."
            )

        report = {
            "schema_version": 1,
            "diagnostic": "ember-v054-int4-router-controlled-generation-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_repo": repo,
            "int4_checkpoint": held.INT4_PATH,
            "int4_checkpoint_sha256": held.INT4_SHA256,
            "router": {
                "binary_representation": heads["binary"]["representation"],
                "family_representation": heads["family"]["representation"],
                "binary_cv": heads["binary_selected"]["cv_best"],
                "family_cv": heads["family_selected"]["cv_best"],
                "routing_verification": router_eval,
                "routing_cases_verified": 170,
            },
            "generation_set": "v054-multilayer-fourth-confirmation",
            "native_baseline": native,
            "controlled": controlled,
            "strict_pass": strict,
            "ember_weights_changed": False,
            "router_saved": False,
            "production_changed": False,
            "interpretation": interpretation,
            "elapsed_seconds": time.monotonic() - started,
        }
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        (OUT / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
        print(json.dumps({
            "event":"int4_controlled_generation_complete",
            "router_binary":heads["binary"]["representation"],
            "router_family":heads["family"]["representation"],
            "native":{"direct":native["direct_pass"], "tool_name":native["tool_name_correct"], "tool_pass":native["tool_pass"]},
            "controlled":{"direct":direct_pass, "tool_family":tool_family_correct,
                          "structured":structured_valid, "argument":argument_ok, "by_family":by_family},
            "strict_pass":strict,
            "interpretation":interpretation,
            "elapsed_seconds":report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
