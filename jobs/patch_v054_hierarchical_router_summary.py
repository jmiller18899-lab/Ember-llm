"""Patch the diagnostic-only hierarchical router summary formatting before execution.

The model/router logic is unchanged; this replaces two nested f-string display
expressions with precomputed strings so the diagnostic module parses on Python 3.11.
"""
from pathlib import Path

path = Path("jobs/ember_v054_hierarchical_router.py")
lines = path.read_text(encoding="utf-8").splitlines()
out = []
inserted = replaced_binary = replaced_family = False
for line in lines:
    out.append(line)
    if line.strip() == 'new = row["evaluation"]["confirmation"]':
        indent = line[: len(line) - len(line.lstrip())]
        out.append(indent + 'bcv_text = "—" if bcv is None else f"{bcv[\'accuracy\']:.1%}"')
        out.append(indent + 'fcv_text = "—" if fcv is None else f"{fcv[\'accuracy\']:.1%}"')
        inserted = True
        continue
    if 'f"| {key} | {\'—\' if bcv is None' in line:
        out[-1] = line[: len(line) - len(line.lstrip())] + 'f"| {key} | {bcv_text} | "'
        replaced_binary = True
    elif 'if fcv is None' in line and 'accuracy' in line and line.lstrip().startswith('f"'):
        out[-1] = line[: len(line) - len(line.lstrip())] + 'f"{fcv_text} | "'
        replaced_family = True

if not (inserted and replaced_binary and replaced_family):
    raise RuntimeError(
        f"summary patch mismatch: inserted={inserted} binary={replaced_binary} family={replaced_family}"
    )
path.write_text("\n".join(out) + "\n", encoding="utf-8")
print("HIERARCHICAL_ROUTER_SUMMARY_PATCH=PASS")
