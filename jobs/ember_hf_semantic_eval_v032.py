# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["huggingface-hub>=1.4", "sentencepiece>=0.2", "torch>=2.4"]
# ///
"""Ember v0.0.32: a semantic quality gate for the promoted v0.0.31 checkpoint.

v0.0.31 passed the legacy promotion evaluator with valid_tool_call_rate,
direct_response_rate and tool_result_response_rate all at 1.0, and its own
promotion record still says: "raw completions remain noisy after the first
scored segment; require a stricter semantic quality gate before production
ClawAgent integration."

That note is precise, and the legacy rubric explains it. For a tool call it
checks that <|tool|> appears, that the first balanced JSON object after it
parses, that the tool name matches, and that ``arguments`` is a non-empty dict
or string. Nothing checks what the arguments *say*, and nothing looks at the
text after the closing brace. For a direct response it checks that no <|tool|>
appears and that at least three visible characters were produced.

So a completion like::

    <|tool|>{"name": "weather", "arguments": {"x": 1}}
    <|user|> what about tomorrow <|assistant|> the weather in the weather in

scores a perfect pass today: right marker, valid JSON, right name, non-empty
arguments. It calls the wrong location, invents a user turn, and degenerates.

This job grades the four things the legacy rubric leaves out:

* **grounding** -- the argument values must carry the entity the prompt asked
  about, which is the capability v0.0.15 through v0.0.31 were built for;
* **argument shape** -- required keys, not merely a non-empty object;
* **termination** -- the generation must stop at EOS inside the budget rather
  than being cut off at max_new_tokens; and
* **what follows** -- no trailing visible text after a tool call, no invented
  conversation turns, no degenerate repetition.

It is read-only. It downloads a checkpoint, generates on CPU, and writes a JSON
report. It never trains, never uploads a model, and never promotes anything.

The scoring functions carry no third-party imports so the whole rubric can be
exercised offline; ``tests/test_ember_v032_semantic_gate.py`` runs it against
synthetic completions, including ones the legacy scorer accepts.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

# `hf jobs uv run` uploads only this script, so the runner has no repository
# checkout. Every asset therefore comes from a commit-pinned raw URL, the same
# way jobs/ember_hf_eval.py and the v0.0.16+ trainers fetch theirs. The local
# --spec / --legacy-spec flags exist for offline use and for the test suite.
ASSET_COMMIT = "41c1b294bf411623b45faa859a23e53f1ad7b9bb"
RAW = f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{ASSET_COMMIT}"
SPEC_URL = f"{RAW}/config/ember_semantic_quality_v0.0.32.json"
LEGACY_SPEC_URL = f"{RAW}/config/ember_v0.0.8_eval.json"
PACKAGE_URL = (
    "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/main/"
    "ember-v0.0.7-hf-ready.zip"
)

SPECIAL_TOKENS = (
    "<|system|>", "<|user|>", "<|assistant|>",
    "<|tool|>", "<|tool_result|>", "<|endoftext|>",
)
EOT = "<|endoftext|>"


# --------------------------------------------------------------------------
# rubric  (stdlib only, so the tests can drive every branch without a model)
# --------------------------------------------------------------------------

def visible_text(text: str) -> str:
    for token in SPECIAL_TOKENS:
        text = text.replace(token, " ")
    return " ".join(text.split())


def extract_json_object(text: str):
    """The first balanced JSON object in text, or None. Mirrors the legacy job."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start:index + 1])
                except json.JSONDecodeError:
                    return None
                return (value, index + 1) if isinstance(value, dict) else None
    return None


def tool_name(payload: dict) -> str:
    direct = payload.get("name") or payload.get("tool")
    if isinstance(direct, str):
        return direct
    function = payload.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return ""


def tool_arguments(payload: dict):
    if "arguments" in payload:
        return payload["arguments"]
    function = payload.get("function")
    if isinstance(function, dict):
        return function.get("arguments")
    return None


def normalise_arguments(arguments):
    """Arguments as a dict when possible; a JSON string is parsed first."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def argument_text(arguments) -> str:
    """Every value in the arguments, flattened, for grounding checks."""
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        return arguments
    if isinstance(arguments, dict):
        return " ".join(argument_text(value) for value in arguments.values())
    if isinstance(arguments, (list, tuple)):
        return " ".join(argument_text(value) for value in arguments)
    return str(arguments)


def repeated_ngram_ratio(text: str, n: int) -> float:
    """Share of n-grams that are repeats. Catches degenerate loops."""
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def split_at_eos(completion: str) -> tuple:
    """(answer, text_after_eos, stopped_at_eos).

    These are three different facts and the first version of this rubric
    collapsed them. `clean_stop` returned True whenever EOS appeared anywhere,
    and every other check ran on `before_eos`, which discards the text after it.
    So a completion that answered, emitted EOS, and then invented four more
    conversation turns scored clean_stop and no_trailing_noise at 1.0 -- the
    exact "scores the first segment and ignores the rest" flaw this gate was
    written to catch in the legacy evaluator.

    The evaluation harness generates the full token budget without halting at
    EOS, so text after the first EOS is a property of the harness, not of what a
    caller that stops at EOS would ever see. It is therefore reported as
    evidence rather than gated, while *whether the model emitted EOS at all*
    is a real property of the model and is gated.
    """
    stopped = EOT in completion
    if not stopped:
        return completion, "", False
    answer, after = completion.split(EOT, 1)
    return answer, after, True


def extra_turn_markers(text: str, forbidden) -> list:
    return [marker for marker in forbidden if marker in text]


def numbers_in(text: str) -> list:
    return re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))


def unfaithful_numbers(answer: str, prompt: str, tolerated) -> list:
    """Numbers the answer states that appear nowhere in the prompt.

    The prompt carries the user's request and the tool result, so any other
    number in the answer was invented. This is what catches "587 multiplied by
    27 equals 672" against a tool result of 9716, and "45 degrees" against a
    tool result of 72.
    """
    available = set(numbers_in(prompt)) | set(str(value) for value in tolerated)
    return [value for value in numbers_in(answer) if value not in available]


def score_tool_call(case: dict, completion: str, quality: dict, prompt: str = "") -> dict:
    body, after_eos, stopped = split_at_eos(completion)
    markers = body.count("<|tool|>")
    after = body.split("<|tool|>", 1)[1] if markers else body
    found = extract_json_object(after)
    payload, end = (found if found else (None, 0))

    trailing = visible_text(after[end:]) if payload is not None else visible_text(after)
    arguments = tool_arguments(payload or {})
    normalised = normalise_arguments(arguments)
    flattened = argument_text(arguments)

    required = list(case.get("required_argument_keys", []))
    any_of = list(case.get("required_argument_keys_any_of", []))
    keys = set(normalised or {})
    keys_present = (
        all(key in keys for key in required)
        and (not any_of or bool(keys & set(any_of)))
    )

    haystack = flattened if case.get("grounding_case_sensitive") else flattened.casefold()
    grounded_missing = [
        value for value in case.get("grounded_values", [])
        if (value if case.get("grounding_case_sensitive") else value.casefold()) not in haystack
    ]

    checks = {
        "marker_present": markers >= 1,
        "single_tool_marker": markers <= int(quality["max_tool_markers"]),
        "json_valid": payload is not None,
        "tool_name_matches": bool(payload) and tool_name(payload) == case["expected_tool"],
        "arguments_are_object": normalised is not None,
        "required_keys_present": keys_present,
        "arguments_grounded": not grounded_missing,
        "stopped_at_eos": stopped if quality.get("require_stop_at_eos", True) else True,
        "no_trailing_noise": len(trailing) <= int(quality["max_trailing_visible_chars"]),
        "no_extra_turn_markers": not extra_turn_markers(body, quality["forbid_extra_turn_markers"]),
        "no_degenerate_repetition": repeated_ngram_ratio(
            visible_text(body), int(quality["repetition_ngram"])
        ) <= float(quality["max_repeated_ngram_ratio"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "advisory": advisory_after_eos(after_eos, quality),
        "answer": visible_text(body)[:200],
        "tool_markers": markers,
        "argument_keys": sorted(keys),
        "grounded_missing": grounded_missing,
        "trailing_visible": trailing[:120],
    }


def advisory_after_eos(after_eos: str, quality: dict) -> dict:
    """Reported, never gated. See split_at_eos for why."""
    if not quality.get("report_text_after_eos", True):
        return {}
    visible = visible_text(after_eos)
    return {
        "no_text_after_eos": not visible,
        "text_after_eos_chars": len(visible),
        "invented_turns_after_eos": sorted(set(
            marker for marker in ("<|user|>", "<|assistant|>", "<|system|>", "<|tool|>")
            if marker in after_eos
        )),
        "extra_eos_count": after_eos.count(EOT),
    }


def score_response(case: dict, completion: str, quality: dict, prompt: str = "") -> dict:
    body, after_eos, stopped = split_at_eos(completion)
    readable = visible_text(body)
    lowered = readable.casefold()

    facts = list(case.get("tool_result_facts", []))
    missing_facts = [fact for fact in facts if fact.casefold() not in lowered]

    wanted = list(case.get("addresses_request_any", []))
    addresses = (not wanted) or any(item.casefold() in lowered for item in wanted)

    invented = []
    if case.get("numeric_faithfulness"):
        if not prompt:
            raise RuntimeError(
                f"case {case['id']} declares numeric_faithfulness; scoring it needs the prompt"
            )
        invented = unfaithful_numbers(
            readable, prompt, quality.get("numeric_faithfulness_tolerated_values", [])
        )

    checks = {
        "no_tool_call": "<|tool|>" not in body,
        "stopped_at_eos": stopped if quality.get("require_stop_at_eos", True) else True,
        "no_extra_turn_markers": not extra_turn_markers(body, quality["forbid_extra_turn_markers"]),
        "long_enough": len(readable) >= int(quality["direct_min_visible_chars"]),
        "enough_words": len(readable.split()) >= int(quality["direct_min_words"]),
        "no_degenerate_repetition": repeated_ngram_ratio(
            readable, int(quality["repetition_ngram"])
        ) <= float(quality["max_repeated_ngram_ratio"]),
        # The content checks the first version of this rubric did not have.
        "uses_tool_result": not missing_facts,
        "addresses_request": addresses,
        "no_invented_numbers": not invented,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "advisory": advisory_after_eos(after_eos, quality),
        "answer": readable[:200],
        "missing_tool_result_facts": missing_facts,
        "invented_numbers": invented,
        "visible_chars": len(readable),
        "repeated_ngram_ratio": repeated_ngram_ratio(readable, int(quality["repetition_ngram"])),
    }


def score_case(case: dict, completion: str, quality: dict, prompt: str = "") -> dict:
    if case["kind"] == "tool_call":
        return score_tool_call(case, completion, quality, prompt)
    if case["kind"] in {"direct_response", "tool_result_response"}:
        return score_response(case, completion, quality, prompt)
    raise ValueError(f"unsupported case kind: {case['kind']}")


def score_legacy(case: dict, completion: str) -> bool:
    """The legacy rubric, reproduced so reports can show the difference."""
    if case["kind"] == "tool_call":
        marker = "<|tool|>" in completion
        after = completion.split("<|tool|>", 1)[1] if marker else completion
        found = extract_json_object(after)
        payload = found[0] if found else None
        arguments = tool_arguments(payload or {})
        return bool(
            marker and payload is not None
            and tool_name(payload) == case["expected_tool"]
            and isinstance(arguments, (dict, str)) and bool(arguments)
        )
    return "<|tool|>" not in completion and len(visible_text(completion)) >= 3


def aggregate(rows: list, spec: dict) -> dict:
    def rate(kind, predicate):
        selected = [r for r in rows if r["kind"] == kind]
        if not selected:
            raise RuntimeError(f"spec has no {kind} cases")
        return sum(bool(predicate(r)) for r in selected) / len(selected)

    strict = [r for r in rows if r["strict"]["passed"]]
    responses = [r for r in rows if r["kind"] != "tool_call"]
    metrics = {
        "grounded_tool_call_rate": rate("tool_call", lambda r: r["strict"]["passed"]),
        "direct_quality_rate": rate("direct_response", lambda r: r["strict"]["passed"]),
        "tool_result_quality_rate": rate("tool_result_response", lambda r: r["strict"]["passed"]),
        # Content, separated from shape: does the answer use what it was given
        # and answer what was asked, without inventing numbers?
        "faithful_response_rate": sum(
            all(r["strict"]["checks"][name] for name in
                ("uses_tool_result", "addresses_request", "no_invented_numbers"))
            for r in responses
        ) / len(responses) if responses else 0.0,
        # The model emitting EOS, which is distinct from what the harness
        # generated after it. The latter is advisory; see split_at_eos.
        "stop_at_eos_rate": sum(r["strict"]["checks"]["stopped_at_eos"] for r in rows) / len(rows),
        "no_trailing_noise_rate": sum(
            r["strict"]["checks"].get("no_trailing_noise", True) for r in rows
        ) / len(rows),
        "no_text_after_eos_rate": sum(
            bool(r["strict"].get("advisory", {}).get("no_text_after_eos")) for r in rows
        ) / len(rows),
        "overall_strict_pass_rate": len(strict) / len(rows),
        "legacy_pass_rate": sum(bool(r["legacy_passed"]) for r in rows) / len(rows),
    }
    gates = {
        name.replace("minimum_", ""): metrics[name.replace("minimum_", "")] >= float(threshold) - 1e-9
        for name, threshold in spec["gates"].items()
        if name.replace("minimum_", "") in metrics
    }
    return {
        "metrics": metrics,
        "gates": gates,
        "passed": all(gates.values()),
        "legacy_passes_but_strict_fails": [
            r["id"] for r in rows if r["legacy_passed"] and not r["strict"]["passed"]
        ],
        "advisory_note": (
            "no_text_after_eos is reported, not gated: the harness generates the full token "
            "budget without halting at EOS, so text after the first EOS is a property of the "
            "harness rather than of what a caller stopping at EOS would see. stopped_at_eos is "
            "gated because emitting EOS is a property of the model."
        ),
    }


# --------------------------------------------------------------------------
# job
# --------------------------------------------------------------------------

def load_spec(local_path: str, url: str) -> dict:
    """A local file when one is given, otherwise the pinned raw URL."""
    if local_path:
        return json.loads(Path(local_path).read_text())
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_checkpoint(api, repo: str, token: str, work: Path, hf_hub_download) -> tuple:
    """The candidate's best.pt, from run-state when it is usable.

    run-state.json names the run whose checkpoint was selected, which is the
    right answer. Listing the repository is the fallback for a checkpoint whose
    state file is missing or incomplete -- the same fallback the legacy
    evaluator uses -- so a reporting gap in an earlier run cannot block a
    read-only evaluation.
    """
    run_id = ""
    try:
        state = json.loads(Path(hf_hub_download(
            repo_id=repo, repo_type="model", filename="run-state.json",
            token=token, local_dir=work / "state",
        )).read_text())
        if state.get("status") == "evaluation_complete":
            run_id = str(state.get("run_id", "")).strip()
    except Exception:
        run_id = ""

    if run_id:
        return f"checkpoints/{run_id}/best.pt", {"resolved_via": "run-state", "run_id": run_id}

    candidates = sorted(
        path for path in api.list_repo_files(repo_id=repo, repo_type="model")
        if path.startswith("checkpoints/") and path.endswith("/best.pt")
    )
    if not candidates:
        raise RuntimeError(f"{repo} has no checkpoints/*/best.pt")
    return candidates[-1], {"resolved_via": "repository listing", "run_id": candidates[-1]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="", help="local spec override; default fetches the pinned SPEC_URL")
    parser.add_argument("--legacy-spec", default="", help="local override; default fetches the pinned LEGACY_SPEC_URL")
    parser.add_argument("--out", default="", help="write the JSON report here")
    parser.add_argument("--completions", default="", help="score a JSON map of id -> completion instead of generating")
    args = parser.parse_args()

    spec = load_spec(args.spec, SPEC_URL)
    legacy = load_spec(args.legacy_spec, LEGACY_SPEC_URL)
    prompts = {case["id"]: case["prompt"] for case in legacy["cases"]}
    quality = spec["quality"]

    if args.completions:
        completions = json.loads(Path(args.completions).read_text())
    else:
        completions = generate_all(spec, prompts)

    rows = []
    for case in spec["cases"]:
        if case["id"] not in completions:
            raise RuntimeError(f"no completion for case {case['id']}")
        text = completions[case["id"]]
        rows.append({
            "id": case["id"],
            "kind": case["kind"],
            "completion": text,
            "strict": score_case(case, text, quality, prompts[case["id"]]),
            "legacy_passed": score_legacy(case, text),
        })

    report = {
        "version": spec["version"],
        "phase": spec["phase"],
        "candidate": spec["candidate_model_name"],
        "cases": rows,
        **aggregate(rows, spec),
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
    if not args.completions:
        publish_report(spec, report)
    print(f"EMBER_V032_SEMANTIC_GATE={'PASS' if report['passed'] else 'FAIL'}", flush=True)
    print(
        "EMBER_V032_LEGACY_PASSES_BUT_STRICT_FAILS="
        + (",".join(report["legacy_passes_but_strict_fails"]) or "none"),
        flush=True,
    )
    return 0


def publish_report(spec: dict, report: dict) -> None:
    """Persist the report to the candidate repo under evaluations/.

    A detached job's stdout lives only in its log stream, so a report that is
    printed and nowhere else cannot be read back later. jobs/ember_hf_eval.py
    writes its results to evaluations/ for the same reason; this follows it.
    Nothing else in the repository is touched: no checkpoint, no run-state, no
    promotion.
    """
    import os
    import tempfile
    from datetime import datetime, timezone

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("EMBER_V032_REPORT_PUBLISHED=SKIPPED_NO_TOKEN", flush=True)
        return
    api = HfApi(token=token)
    repo = f"Jmiller18899/{spec['candidate_model_name']}"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    body = json.dumps(report, indent=2) + "\n"
    with tempfile.TemporaryDirectory(prefix="ember-v032-report-") as td:
        local = Path(td) / "report.json"
        local.write_text(body)
        for remote in (
            f"evaluations/v032-semantic-gate-{stamp}.json",
            "evaluations/v032-semantic-gate-latest.json",
        ):
            api.upload_file(
                repo_id=repo, repo_type="model",
                path_or_fileobj=str(local), path_in_repo=remote,
                commit_message="Ember v0.0.32 semantic quality gate report",
            )
            print(f"EMBER_V032_REPORT={repo}/{remote}", flush=True)
    print("EMBER_V032_REPORT_PUBLISHED=PASS", flush=True)


def generate_all(spec: dict, prompts: dict) -> dict:
    """Download the candidate checkpoint and generate every case on CPU."""
    import os
    import sys
    import tempfile
    import urllib.request
    import zipfile

    from huggingface_hub import HfApi, hf_hub_download
    import torch

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required to read the candidate checkpoint")
    api = HfApi(token=token)
    api.whoami()  # fail before downloading anything if the token is not live
    repo = f"Jmiller18899/{spec['candidate_model_name']}"

    with tempfile.TemporaryDirectory(prefix="ember-v032-") as td:
        work = Path(td)
        package = work / "ember.zip"
        urllib.request.urlretrieve(PACKAGE_URL, package)
        with zipfile.ZipFile(package) as archive:
            archive.extractall(work / "src")
        sys.path.insert(0, str(work / "src" / "ember"))
        from src.checkpoint import load_checkpoint
        from src.model import EmberGPT, ModelConfig
        from src.tokenizer import tokenizer_from_state_dict

        remote, provenance = resolve_checkpoint(api, repo, token, work, hf_hub_download)
        print(f"EMBER_V032_CHECKPOINT={repo}/{remote} ({provenance['resolved_via']})", flush=True)
        checkpoint = Path(hf_hub_download(
            repo_id=repo, repo_type="model", filename=remote,
            token=token, local_dir=work / "model",
        ))
        loaded = load_checkpoint(checkpoint, device="cpu")
        tokenizer = tokenizer_from_state_dict(loaded["tokenizer"])
        model = EmberGPT(ModelConfig(**loaded["model_config"]))
        model.load_state_dict(loaded["model_state"])
        model.eval()

        generation = spec["generation"]
        torch.manual_seed(int(generation["seed"]))
        out = {}
        for case in spec["cases"]:
            prompt = prompts[case["id"]]
            ids = tokenizer.encode(prompt)
            budget = min(
                int(generation["max_new_tokens"]),
                int(model.cfg.block_size) - len(ids),
            )
            x = torch.tensor([ids], dtype=torch.long)
            with torch.inference_mode():
                y = model.generate(
                    x, max_new_tokens=budget,
                    temperature=float(generation["temperature"]),
                    top_k=int(generation["top_k"]),
                )
            out[case["id"]] = tokenizer.decode(y[0].tolist()[len(ids):])
        return out


if __name__ == "__main__":
    raise SystemExit(main())
