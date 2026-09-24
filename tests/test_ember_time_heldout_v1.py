import importlib.util,re
from pathlib import Path
JOBS=Path(__file__).resolve().parents[1]/"jobs"
def load(name):
    s=importlib.util.spec_from_file_location(name,JOBS/f"{name}.py"); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
H=load("ember_time_heldout_v1")

def frozen_block(text):
    lines=text.splitlines()
    start=next(i for i,l in enumerate(lines) if l.startswith("BASE="))
    end=next(i for i,l in enumerate(lines) if l.startswith(" return SYSTEM+"))
    return lines[start:end+1]

def test_frozen_configuration_is_verbatim():
    heldout=(JOBS/"ember_time_heldout_v1.py").read_text()
    frozen=(JOBS/"ember_4b_promotion_v2_frozen_eval.py").read_text()
    assert frozen_block(heldout)==frozen_block(frozen)

def test_excludes_exactly_the_promo_v2_time_cases():
    suite=load("build_ember_promotion_suite_v2")
    keys=set()
    for r in suite.build():
        if r["family"]!="time_reasoning": continue
        h,m,ap,d=re.search(r"at (\d+):(\d+) (AM|PM)\. Travel time is (\d+)",r["prompt"]).groups()
        keys.add((H.parse12(f"{h}:{m} {ap}"),int(d)))
    assert keys==H.promo_v2_time_keys()

def test_preflight_and_determinism():
    assert H.preflight()==H.build()

def test_scorer_separates_format_from_content():
    row={"answer":"12:10 PM","format":"natural"}
    assert H.score(row,"12:10 PM")["strict"]
    s=H.score(row,"H:12:10 PM"); assert (s["strict"],s["content"],s["prefix_echo"])==(False,True,True)
    s=H.score(row,"12:10 AM"); assert not s["content"] and not s["meridiem_ok"] and s["hour_ok"]
