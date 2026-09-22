"""Package only evidence that passes the complete measured gate."""
import argparse
import json
from pathlib import Path
import shutil

from .runtime import sha256


def require_pass(report, manifest_sha256):
    if report.get("strict_pass") is not True or report.get("status") != "PASS":
        raise ValueError("Measured confirmation failed; packaging is blocked")
    if report.get("candidate_manifest_sha256") != manifest_sha256:
        raise ValueError("Confirmation belongs to a different candidate")
    if set(report.get("results", {})) != {"full", "int4"}:
        raise ValueError("Both precisions must pass")
    for result in report["results"].values():
        if (result.get("routing_correct"), result.get("routing_total"),
            result.get("exact_arguments_correct"), result.get("arguments_total"),
            result.get("combined_correct"), result.get("combined_total")) != (100, 100, 80, 80, 100, 100):
            raise ValueError("Incomplete combined confirmation")
        cases = result.get("cases", [])
        if len(cases) != 100 or len({c.get("id") for c in cases}) != 100 or not all(c.get("passed") is True for c in cases):
            raise ValueError("Per-request evidence is incomplete")
        tools = [c for c in cases if c.get("route") != "direct"]
        if len(tools) != 80 or not all(c.get("arguments_ok") is True and c.get("fixture_dispatch_ok") is True for c in tools):
            raise ValueError("Tool arguments or dispatch failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    require_pass(report, sha256(args.bundle / "manifest.json"))
    if args.out.with_suffix(".zip").exists():
        raise ValueError("Refusing to overwrite a prototype")
    root = Path(__file__).parent.parent
    shutil.copyfile(root / "docs/ember-tool-assistant.md", args.bundle / "README.md")
    shutil.copyfile(root / "tool_assistant/requirements.txt", args.bundle / "requirements.txt")
    shutil.copyfile(args.report, args.bundle / "confirmation.json")
    shutil.make_archive(str(args.out), "zip", args.bundle)


if __name__ == "__main__":
    main()
