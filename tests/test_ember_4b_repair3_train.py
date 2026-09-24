import importlib.util,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("r3",ROOT/"jobs/ember_4b_repair3_train.py"); R=importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
H=R.module("jobs/ember_time_heldout_v1.py","h"); P=R.module("jobs/ember_4b_promotion_v2_frozen_eval.py","p"); O=R.module("jobs/ember_4b_repair2_original_promotion_eval.py","o")
EXC=R.promo_mult_triples(P)

def test_rows_deterministic_and_sized():
    a=R.build(P,EXC); b=R.build(P,EXC)
    assert R.rows_digest(a)==R.rows_digest(b) and len(a)==128
    assert sum(x["slice"]=="multiply_add_target" for x in a)==64
    assert len({x["id"] for x in a})==128

def test_targets_are_correct_and_never_n_plus_one_times_k():
    for x in R.target_rows(EXC|R.KNOWN_BENCH_TRIPLES):
        n,k,xx=x["operands"]; assert x["answer"]==str(n*k+xx) and xx!=k and int(x["answer"])!=(n+1)*k
        assert tuple(x["operands"]) not in EXC|R.KNOWN_BENCH_TRIPLES
        assert R.numbers(x["prompt"])==(n,k,xx)

def test_no_time_training_beyond_replay_and_no_rule_firing_on_targets():
    rows=R.build(P,EXC); time=[x for x in rows if x["family"]=="time_reasoning"]
    assert all(x["id"].startswith("repair3-replay-") for x in time) and len(time)==26  # replay only: placeholder 8, rollover 6, long 5, midnight 5, maintenance 2
    assert all(P.system_for(x["prompt"])==P.SYSTEM for x in rows if x["slice"]=="multiply_add_target")

def test_no_contamination_with_local_benchmarks_or_holdout():
    rows=R.build(P,EXC); ho=R.fresh_mult_holdout(EXC|R.KNOWN_BENCH_TRIPLES|{tuple(x["operands"]) for x in rows if "operands" in x and x["slice"]=="multiply_add_target"})
    bench=[r["prompt"] for r in P.build()]+[r["prompt"] for r in H.build()]+[c["prompt"] for c in O.TEMP+O.DRAFT]+[r["prompt"] for r in ho]
    assert R.contamination(rows,bench)=={"same_prompt":[],"same_numbers":[]}
    assert R.contamination(rows,["There are 4 sealed bundles with 9 screws in each bundle and 4 extra screws. Total screws? Number only."])["same_numbers"]==[]
    fake=[x["prompt"] for x in rows if x["slice"]=="multiply_add_target"][:1]
    assert R.contamination(rows,fake)["same_prompt"]

def test_gates():
    ex={f"c{i}":True for i in range(70)}; ex["arith-pack-13"]=False; ex["time-14"]=False
    b={"exact":70,"exact_pass":ex,"temporal":8,"drafting":7,"mult_holdout":[30,40]}
    good={**b,"exact":71,"exact_pass":{**ex,"arith-pack-13":True},"mult_holdout":[35,40]}
    assert R.fast_gate(b,good)["passed"]
    bad={**good,"exact_pass":{**good["exact_pass"],"c3":False}}
    g=R.fast_gate(b,bad); assert not g["passed"] and g["exact_regressions"]==["c3"]
    assert not R.fast_gate(b,{**good,"temporal":7})["passed"]
    fam={"arithmetic":[28,40],"time_reasoning":[23,30]}
    fb={"promo":{"overall":[178,200],"by_family":fam},"heldout":{"overall":{"strict":148,"content":148}}}
    assert R.full_gate(fb,{**fb,"promo":{"overall":[179,200],"by_family":{**fam,"arithmetic":[29,40]}}})["passed"]
    assert not R.full_gate(fb,{**fb,"promo":{"overall":[178,200],"by_family":{"arithmetic":[29,40],"time_reasoning":[22,30]}}})["passed"]
