import importlib.util,json,re
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name):
    s=importlib.util.spec_from_file_location(name,ROOT/"jobs"/f"{name}.py"); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
C=load("ember_4b_repair2_curriculum"); H=load("ember_time_heldout_v1")
ROWS=[json.loads(l) for l in (ROOT/"data"/"ember_4b_repair2_curriculum.jsonl").read_text().splitlines()]
TIME=[x for x in ROWS if x["family"]=="time_reasoning"]

def test_committed_corpus_is_the_frozen_builder_output():
    assert (ROOT/"data"/"ember_4b_repair2_curriculum.jsonl").read_text()==C.serialize(C.build())
    assert len(ROWS)==400 and len({x["id"] for x in ROWS})==400 and len({x["prompt"] for x in ROWS})==400

def test_weights():
    fam=Counter(x["family"] for x in ROWS)
    assert fam=={"time_reasoning":200,"arithmetic":100,"context_consistency":50,"clarification":50}
    t=Counter(x["slice"] for x in TIME)
    assert {k:v/200 for k,v in t.items()}=={"long_duration":.30,"midnight":.25,"rollover_60":.20,"placeholder":.20,"maintenance":.05}

def test_time_answers_and_slice_definitions():
    for x in TIME:
        s=H.parse12(x["start"]); d=x["duration_min"]; C.check_time(s,d,x["answer"])
        assert re.fullmatch(r"(1[0-2]|[1-9]):[0-5]\d (AM|PM)",x["answer"])
        crosses_midnight=s+d>=1440
        if x["slice"]=="midnight": assert crosses_midnight and x["start"].endswith("PM") and x["answer"].endswith("AM")
        else: assert not crosses_midnight
        if x["slice"]=="rollover_60": assert x["answer"].split(":")[1].startswith("00")
        if x["slice"]=="long_duration": assert d>=65
        if x["slice"]=="placeholder": assert re.search(r"\b[Hh]:[Mm][Mm]\b",x["prompt"])
        else: assert x["answer"]!=C.NATURAL_EXAMPLE and not re.search(r"\b[Hh]:[Mm][Mm]\b",x["prompt"])

def test_disjoint_from_both_evaluation_suites():
    pairs,arith,eval_prompts=C.eval_exclusions()
    assert len(pairs)==210  # 180 held-out v1 + 30 promotion v2
    assert not {(H.parse12(x["start"]),x["duration_min"]) for x in TIME}&pairs
    for x in ROWS:
        if x["family"]=="arithmetic":
            key=tuple(x["operands"]) if x["slice"]=="add_subtract" else ("mul",*x["operands"])
            assert key not in arith
        assert x["prompt"] not in eval_prompts
        for eval_phrase in ("Travel time is","formatted like 6:15 PM","Context record","Request #","A bin has","A bus departs","A meeting starts","A work shift begins","A train leaves"):
            assert eval_phrase not in x["prompt"],(eval_phrase,x["prompt"])

def test_arithmetic_targets_the_failure_pattern():
    add=[x for x in ROWS if x.get("slice")=="add_subtract"]; mul=[x for x in ROWS if x.get("slice")=="multiply_add"]
    assert all(int(x["answer"])==x["operands"][0]+x["operands"][1]-x["operands"][2] for x in add)
    assert all(int(x["answer"])==x["operands"][0]*x["operands"][1]+x["operands"][2] for x in mul)
    assert sum(x["operands"][1]>=11 for x in add)==64  # 80% two-digit added amounts

def test_targets_pass_the_frozen_promotion_scorer():
    ev=load("ember_4b_promotion_v2_frozen_eval")
    for x in ROWS:
        if x["family"]=="context_consistency":
            o=next(n for n in C.OWNERS if n in x["prompt"]); i=next(n for n in sorted(C.ITEMS,key=len,reverse=True) if n in x["prompt"]); d=next(n for n in C.DAYS if n in x["prompt"])
            row={"family":"context_consistency","id":"t","prompt":"","rubric":f"Answer that {o} owns the {i} and it is returned {d}; do not change or invent facts."}
            assert ev.rubric_pass(row,x["answer"]),x
        if x["family"]=="clarification":
            assert ev.rubric_pass({"family":"clarification","id":"t","prompt":"","rubric":""},x["answer"]),x
