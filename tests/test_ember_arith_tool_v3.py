"""Tool v3 must keep every v2 behaviour except the intended changes, and answer exactly or not at all."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"jobs")); sys.path.insert(0,str(Path(__file__).resolve().parent))
import ember_arith_tool_v3 as T3
import test_ember_arith_tool as V2T

INTENDED={"The lot has 58 cars. 13 cars arrive, then 21 cars leave the lot... how many remain? Number only.":("add_subtract","50")}

def test_v2_positives_unchanged():
    for p,kind,ans in V2T.POSITIVE:
        exp=INTENDED.get(p,(kind,ans) if kind else None)
        assert T3.solve(p)==exp,(p,T3.solve(p))

def test_v2_negatives_still_decline():
    assert not [p for p in V2T.NEGATIVE if T3.solve(p)]

NEW_POSITIVE=[
 ("A library founded in 2019 had 35 atlases. 25 atlases were donated and 7 were removed. How many atlases does it have now? Number only.","add_subtract","53"),
 ("In room 183, there are 3 bins with 8 markers in each bin and 5 extra markers. How many markers are there? Number only.","multiply_add","29"),
 ("At gate 12, 4 carts hold 9 bags each, plus 3 loose bags. Total bags? Number only.","multiply_add","39"),
 ("Since 2015 the choir has had 30 singers. 6 new singers join and 4 quit. How many singers now? Number only.","add_subtract","32"),
 ("A club has 103 members. 40 new members join and 9 members leave the club. How many members remain? Number only.","add_subtract","134"),
 ("A roast goes into the oven at 5:00 PM and needs 245 minutes. When is it done? Time only.","clock_add","9:05 PM"),
 ("Bread goes in at 11:40 PM and requires 50 minutes. When is it done? Just the time.","clock_add","12:30 AM"),
 ("There are 5 crates with 6 apples each and 4 apples left over. How many apples? Number only.","multiply_add","34"),
]
NEW_NEGATIVE=[
 "In 2019 there were 40 chairs and in 2021 there were 55 chairs. How many more chairs? Number only.",   # a difference question
 "Room 12 has 30 desks and room 14 has 25 desks. How many desks in total? Number only.",               # two rooms, not the tool's shape
 "The 3:10 PM train goes into the tunnel at 3:20 PM. When does it arrive? Time only.",
 "A class had 25 students; 5 left early. What fraction left? Number only.",
]
def test_new_positive_and_negative():
    for p,k,a in NEW_POSITIVE: assert T3.solve(p)==(k,a),(p,T3.solve(p))
    assert not [(p,T3.solve(p)) for p in NEW_NEGATIVE if T3.solve(p)]
