"""Compatibility entry point for v0.0.47 safe-prefix depth handling.

A successful generated token may span both structural JSON text and the first
character(s) of the query value. The base v0.0.47 extractor correctly refuses
to force such a token, but its stage annotator initially assumed every safe
prefix still reached all JSON keys. This shim removes that assumption and adds
a separate coverage guard: failure localization is accepted only if successful
source prefixes can safely reach the query-value boundary without consuming any
value-bearing token.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import ember_envelope_template_v047 as v047

_base_extract = v047.extract_template
_base_build = v047.build_template_library
_base_interpret = v047.interpret
_base_summary = v047.summary_markdown


def stage_milestones(prefix_text: str) -> list[tuple[int, str]]:
    milestones: list[tuple[int, str]] = []
    marker = prefix_text.find(v047.TOOL)
    if marker < 0:
        raise ValueError("template prefix lacks tool marker")
    marker_end = marker + len(v047.TOOL)
    milestones.append((marker_end, "tool_marker"))
    cursor = marker_end
    for pattern, name in (
        (r'"name"', "name_key"),
        (r'"web_search"', "tool_name"),
        (r'"arguments"', "arguments_key"),
        (r'"query"', "query_key"),
    ):
        match = re.search(pattern, prefix_text[cursor:])
        if match is None:
            break
        end = cursor + match.end()
        milestones.append((end, name))
        cursor = end
    if len(prefix_text) > milestones[-1][0]:
        milestones.append((len(prefix_text), "safe_nonvalue_boundary"))
    return milestones


def extract_template(row: dict, tokenizer) -> dict:
    result = _base_extract(row, tokenizer)
    result["reaches_value_boundary"] = result["prefix_char_end"] == result["value_char_start"]
    result["max_safe_stage"] = result["stages"][-1]
    return result


def build_template_library(rows: list[dict], tokenizer) -> list[dict]:
    templates = _base_build(rows, tokenizer)
    total = sum(len(template["source_ids"]) for template in templates)
    boundary = sum(
        len(template["source_ids"]) for template in templates
        if template.get("reaches_value_boundary") is True
    )
    v047._safe_depth_stats = {
        "sources": total,
        "value_boundary_sources": boundary,
        "value_boundary_source_rate": boundary / total if total else 0.0,
    }
    return templates


def interpret(rows: list[dict], cfg: dict) -> dict:
    result = _base_interpret(rows, cfg)
    stats = getattr(v047, "_safe_depth_stats", {"sources": 0, "value_boundary_sources": 0, "value_boundary_source_rate": 0.0})
    threshold = float(cfg["matched_pass_leave_one_out_complete_rate_minimum"])
    depth_stable = stats["value_boundary_source_rate"] >= threshold
    result["safe_prefix_depth"] = stats
    result["safe_value_boundary_stable"] = depth_stable
    if not depth_stable:
        result["regime"] = "successful_tokenization_entangles_structure_with_value_boundary"
        result["meaning"] = (
            "Successful envelopes do not provide enough value-free generated-token prefixes at the query boundary "
            "for deeper structural localization; no learning phase is authorized."
        )
    return result


def summary_markdown(report: dict) -> str:
    text = _base_summary(report)
    depth = report["interpretation"]["safe_prefix_depth"]
    extra = (
        "\n## Safe-prefix depth\n\n"
        f"- Successful source prefixes reaching the query-value boundary without a value-bearing token: "
        f"**{depth['value_boundary_sources']}/{depth['sources']} ({depth['value_boundary_source_rate']:.1%})**\n"
        f"- Boundary-depth stability: **{'PASS' if report['interpretation']['safe_value_boundary_stable'] else 'FAIL'}**\n"
    )
    return text + extra


v047._stage_milestones = stage_milestones
v047.extract_template = extract_template
v047.build_template_library = build_template_library
v047.interpret = interpret
v047.summary_markdown = summary_markdown


if __name__ == "__main__":
    raise SystemExit(v047.main())
