import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"jobs"))
import ember_runtime_policy as R

def fake(system,prompt): return f"MODEL[{len(system)}]"

def test_tool_route_skips_model():
    p="There are 4 sealed bundles with 9 screws in each bundle and 4 extra screws. Total screws? Number only."
    assert R.respond(p,fake)==("40","tool:multiply_add")

def test_model_route_uses_scoped_rules():
    t=R.RULES.TEMP[0]["prompt"]; d=R.RULES.DRAFT[0]["prompt"]; plain="Summarize this for me."
    assert R.respond(t,lambda s,p:s)[0]==R.RULES.SYSTEM+" "+R.RULES.TEMPORAL_RULE
    assert R.respond(d,lambda s,p:s)[0]==R.RULES.SYSTEM+" "+R.RULES.DRAFT_RULE
    assert R.respond(plain,fake)==(f"MODEL[{len(R.RULES.SYSTEM)}]","model")

def test_frozen_hashes_match_files():
    import hashlib
    for rel,h in R.PINNED.items(): assert hashlib.sha256((R.ROOT/rel).read_bytes()).hexdigest()==h
