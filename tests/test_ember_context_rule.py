import sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"jobs"))
import ember_context_rule as C, ember_context_holdout as HO
def mod(rel):
    m=types.ModuleType(rel); exec(compile((ROOT/rel).read_text(),rel,"exec"),m.__dict__); return m
P=mod("jobs/ember_4b_promotion_v2_frozen_eval.py"); H=mod("jobs/ember_time_heldout_v1.py"); O=mod("jobs/ember_4b_repair2_original_promotion_eval.py")

def test_gate_fires_only_on_context_family_in_promo_v2():
    fired={r["family"] for r in P.build() if C.context_gate(r["prompt"])}
    assert fired=={"context_consistency"}
    assert sum(C.context_gate(r["prompt"]) for r in P.build())==15

def test_gate_silent_on_extraction_time_temporal_drafting():
    for p in [r["prompt"] for r in H.build()]+[c["prompt"] for c in O.TEMP+O.DRAFT]:
        assert not C.context_gate(p), p
    for p in ["Record: user=Nora | ticket=ZX-123_a | priority=low. Return only the ticket value.",
              "Record | owner=Maya | ref=AK-1234 | state=open. Return only ref.",
              "Ticket | assignee=Dara | id=PA-1448 | priority=low. Give only the id.",
              "Set x=5 and y=7 in the config.",
              "Who owns the notebook? Lena said it is hers."]:
        assert not C.context_gate(p), p

def test_gate_fires_on_every_holdout_prompt_and_rule_has_no_suite_values():
    assert all(C.context_gate(r["prompt"]) for r in HO.build())
    assert len(HO.build())==24
    for v in ("folder","Theo","Jon","Monday","notebook","badge","charger","umbrella"):
        assert v not in C.CONTEXT_RULE

def test_holdout_scoring():
    r=HO.build()[0]; o,i,d=r["values"]
    assert HO.score(r,f"{o} owns the {i}, and it is returned {d}.")
    assert not HO.score(r,f"{o} owns the item and it is returned {d}.")

def test_rule_example_does_not_overlap_holdout():
    for r in HO.build():
        for v in r["values"]: assert v not in ("Ana","scarf","Sunday"), r
