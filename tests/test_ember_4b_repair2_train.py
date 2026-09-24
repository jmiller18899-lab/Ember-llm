import copy,importlib.util,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location("t",ROOT/"jobs"/"ember_4b_repair2_train.py"); T=importlib.util.module_from_spec(s); s.loader.exec_module(T)

def test_pinned_files_match_the_checkout():
    rows,H,P=T.load_inputs()
    assert len(rows)==444 and len(H.build())==180 and len(P.build())==200
    assert P.MODEL==T.FROZEN_MODEL and P.MODEL_REV==T.FROZEN_REV

def test_hash_mismatch_is_fatal(monkeypatch):
    monkeypatch.setitem(T.PINNED,"data/ember_4b_repair2_curriculum.jsonl","0"*64)
    with pytest.raises(RuntimeError,match="sha256"): T.fetch("data/ember_4b_repair2_curriculum.jsonl")

def test_never_writes_to_the_frozen_repo():
    src=(ROOT/"jobs"/"ember_4b_repair2_train.py").read_text()
    assert T.OUTPUT_REPO!=T.FROZEN_MODEL
    assert "repo_id=FROZEN_MODEL" not in src and "create_repo(FROZEN_MODEL" not in src

def scores(strict,content,overall,fam):
    return {"heldout":{"overall":{"strict":strict,"content":content}},"promo":{"overall":[overall,200],"by_family":{k:[v,0] for k,v in fam.items()}}}
FAM={"arithmetic":27,"extraction":30,"time_reasoning":1,"grounding":25,"clarification":17,"drafting":25,"action_honesty":15,"context_consistency":11}

def test_gate_accepts_improvement_without_regression():
    better=dict(FAM,time_reasoning=12,arithmetic=31)
    assert T.gate(scores(125,132,151,FAM),scores(150,152,166,better))["accepted"]

@pytest.mark.parametrize("change",[
    {"strict":125},                      # no time improvement
    {"content":131},                     # content drops
    {"overall":150},                     # promo overall drops
    {"fam":dict(FAM,grounding=24,time_reasoning=12)},   # protected family drops
    {"fam":dict(FAM,clarification=16,time_reasoning=12)},  # targeted family drops
])
def test_gate_rejects(change):
    a=dict(strict=150,content=152,overall=166,fam=dict(FAM,time_reasoning=12)); a.update(change)
    assert not T.gate(scores(125,132,151,FAM),scores(**a))["accepted"]
