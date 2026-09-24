import hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"jobs"))
import ember_runtime_policy as V1, ember_runtime_policy_v2 as V2

def test_context_prompt_gets_context_rule():
    p="Context record 9: owner=Jon; item=folder; return_day=Monday. Now answer: Who owns the item and when is it returned?"
    s,route=V2.respond(p,lambda s,q:s)
    assert route=="model" and s==V2.RULES.SYSTEM+" "+V2.CONTEXT.CONTEXT_RULE

def test_v2_equals_v1_everywhere_else():
    prompts=[V2.RULES.TEMP[0]["prompt"],V2.RULES.DRAFT[0]["prompt"],"Summarize this for me.",
             "Record: user=Nora | ticket=ZX-123_a | priority=low. Return only the ticket value.",
             "There are 4 sealed bundles with 9 screws in each bundle and 4 extra screws. Total screws? Number only."]
    echo=lambda s,q:s
    for p in prompts: assert V2.respond(p,echo)==V1.respond(p,echo), p

def test_pins_match():
    for rel,h in V2.PINNED.items(): assert hashlib.sha256((V2.ROOT/rel).read_bytes()).hexdigest()==h
