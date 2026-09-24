import sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"jobs"))
import ember_policy_v3_rules as R, ember_context_rule as C, ember_policy_v3_probes as PR
O=types.ModuleType("o"); exec((ROOT/"jobs/ember_4b_repair2_original_promotion_eval.py").read_text(),O.__dict__)
P=types.ModuleType("p"); exec((ROOT/"jobs/ember_4b_promotion_v2_frozen_eval.py").read_text(),P.__dict__)
H=types.ModuleType("h"); exec((ROOT/"jobs/ember_time_heldout_v1.py").read_text(),H.__dict__)
g=lambda p:R.drafting_gate_v3(p,O.drafting_gate)

def test_drafting_gate_v3_extends_v2_only_on_message_requests():
    for p in ["Text Omar that I have his charger.","Let Maria know that I found her wallet.","Tell Kofi that I still have his umbrella.",
              "Write a quick note to Sam: I have their keys.","Send a note to Alex: I have their laptop."]: assert g(p), p
    for p in ["Tell me how to fix my bike.","Text messages cost 10 cents each. How much for 5? Number only.","Let me know the time.",
              "Record: owner=Theo; item=binder; return_day=Monday. Tell me whose item it is."]: assert not g(p), p
    for r in P.build()+H.build(): assert g(r["prompt"])==bool(O.drafting_gate(r["prompt"])), r["id"]

def test_missing_text_gate():
    for p in ["Could you translate this into Spanish?","Please fix the grammar in my paragraph.","Can you proofread my cover letter?",
              "Summarize this article for me, please.","Make this email sound more polite, thanks."]: assert R.missing_text_gate(p), p
    for p in ["Fix the grammar: 'She go to school.'","Translate into French: 'Thank you'","Tell me how to fix my bike.","Summarize the plot of Hamlet.",
              "How do I make my car more efficient?","Turn these notes into one sentence: milk, eggs, bread needed today.",
              "Shorten this without dropping place or time: 'Please ensure all visitors meet at the front desk before 9.'"]: assert not R.missing_text_gate(p), p
    fired={r["family"] for r in P.build()+H.build() if R.missing_text_gate(r["prompt"])}
    assert fired<={"clarification"}

def test_context_rule_v3_extends_v2_text_and_example_is_not_a_probe_value():
    t=R.context_rule_v3(C.CONTEXT_RULE); assert t.startswith(C.CONTEXT_RULE) and "whose" in t.lower()
    vals={v for r in PR.build() for v in r.get("values",[])}
    assert not vals&{"Ana","scarf"}

def test_probes_shape():
    rows=PR.build(); assert len(rows)==36 and len({r["prompt"] for r in rows})==36
