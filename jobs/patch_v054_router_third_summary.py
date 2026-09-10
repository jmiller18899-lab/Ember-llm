"""Patch only nested f-string display formatting in third router confirmation."""
from pathlib import Path

# Report-only patch. This comment is also the deterministic workflow trigger.
path = Path("jobs/ember_v054_router_third_confirmation.py")
lines = path.read_text(encoding="utf-8").splitlines()
out = []
inserted = replaced = False
for line in lines:
    if line.strip() == 'm = report["routers"][key]["third"]':
        out.append(line)
        indent = line[: len(line) - len(line.lstrip())]
        out.append(indent + 'wrong_text = "—" if m["max_wrong_margin"] is None else f"{m[\'max_wrong_margin\']:.6f}"')
        inserted = True
        continue
    if "max_wrong_margin" in line and "if m['max_wrong_margin'] is None" in line:
        indent = line[: len(line) - len(line.lstrip())]
        out.append(indent + 'f"{wrong_text} |"')
        replaced = True
        continue
    out.append(line)
if not (inserted and replaced):
    raise RuntimeError(f"third summary patch mismatch: inserted={inserted} replaced={replaced}")
path.write_text("\n".join(out) + "\n", encoding="utf-8")
print("ROUTER_THIRD_SUMMARY_PATCH=PASS")
