"""The arithmetic tool must answer exactly or not at all. Prompts here are written independently of every eval suite."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"jobs"))
import ember_arith_tool as T

POSITIVE=[
 ("There are 4 sealed bundles with 9 screws in each bundle and 4 extra screws. Total screws? Number only.","multiply_add","40"),
 ("A florist makes 7 bouquets of 12 roses, plus 5 loose roses. How many roses altogether? Number only.","multiply_add","89"),
 ("5 shelves hold 8 books each and 3 books are loose. How many books in total? Just the number.","multiply_add","43"),
 ("We packed 6 boxes with 10 cans per box, and 2 spare cans. Total cans? Number only.","multiply_add","62"),
 ("There are 3 vans with 7 seats each, plus 4 folding seats. How many seats? Number only.","multiply_add","25"),
 ("A school has 12 classes of 25 students plus 8 additional students. How many students in all? Number only.","multiply_add","308"),
 ("A pantry starts with 40 cans. 15 cans are bought and 9 are eaten. How many cans are left? Number only.","add_subtract","46"),
 ("The lot has 58 cars. 13 cars arrive, then 21 cars leave the lot... how many remain? Number only.",None,None),  # "leave" is not a removal cue: decline
 ("A library had 120 books. 35 were donated, and 48 were borrowed. How many books remain? Number only.","add_subtract","107"),
 ("A jar contains 64 marbles. 20 marbles are removed. Then 11 more are added. How many marbles now? Number only.","add_subtract","55"),
 ("There were 200 tickets. 45 were sold. How many tickets are left? Number only.",None,None),                  # two numbers: not the 3-term shape
 ("A webinar starts at 9:40 AM and lasts 95 minutes. When does it end? Reply with only the time.","clock_add","11:15 AM"),
 ("The night bus departs at 11:35 PM; the trip takes 50 minutes. What time does it arrive? Just the time.","clock_add","12:25 AM"),
 ("A game kicks off at 12:30 PM and lasts 2 hours 15 minutes. When is it over? Time only.","clock_add","2:45 PM"),
 ("A shuttle leaves at 10:50 AM and the trip takes 75 minutes. What time does it arrive? Answer as H:MM AM.","clock_add","12:05 PM"),
 ("A plane takes off at 11:20 AM. The flight is 40 minutes long. When does it land? Format: h:mm AM/PM.","clock_add","12:00 PM"),
 ("A lecture begins at 11:05 PM and runs 3 hours. When does it finish? Give only the time, formatted like 6:15 PM.","clock_add","2:05 AM"),
]
NEGATIVE=[
 "My shipment was collected Wednesday. When was it delivered? Context: tonight.",
 "My bus left at 9:10 AM. I'm asking this afternoon. When did it arrive?",
 "Trip case 3: I only know my train departed at 9:20 AM. What exact time did it arrive?",
 "A ferry departed at 1:30 PM and traveled for 45 minutes. What time did it arrive?",           # no bare-time request: model answers
 "The bus arrives at 3:00 PM after a 40 minute trip. What time did it leave? Answer as H:MM PM.", # asks the start, not the end
 "The 3:10 PM train is running 15 minutes early. When will it arrive? Just the time.",
 "The meeting ends at 4:30 PM and lasted 45 minutes. When does it end? Time only.",
 "How long is a trip from 9:10 AM to 10:05 AM? Number only.",
 "There are 4 boxes with 6 eggs each and 3 eggs broke. How many eggs? Number only.",
 "There are 4 boxes with 6 eggs each plus one more full box. How many eggs? Number only.",
 "A car drives 3 hours at 50 km per hour. Total distance? Number only.",
 "Ana has 20 stickers and gives 5 to Ben who has 8. How many does Ben have? Number only.",
 "There are 12 boxes with 6 pens each. Total pens? Number only.",
 "There are 5 boxes with 4 pens each and 3 boxes with 2 pens each. Total pens? Number only.",
 "A bin has 50 parts. 12 are added and 7 are removed. How many parts remain?",                    # no number-only request
 "Record | owner=Maya | ref=AK-1234 | state=open. Return only ref.",
 "Ticket | assignee=Dara | id=PA-1448 | priority=low. Give only the id.",
 "Message Nina: I found their notebook and can return it Monday.",
 "Shorten this without dropping place or time: 'Please ensure all visitors meet at the front desk before 9.'",
 "Did you send that email? You do not have an email tool in this chat.",
 "What is 15% of 80? Number only.",
 "A recipe doubles 3 cups of flour and adds 2 cups of sugar. Total cups? Number only.",
]

def test_positive_answers_are_exact():
    for p,kind,ans in POSITIVE:
        got=T.solve(p)
        assert (got==(kind,ans)) if kind else got is None, (p,got)

def test_negatives_never_fire():
    fired=[(p,T.solve(p)) for p in NEGATIVE if T.solve(p) is not None]
    assert not fired, fired

def test_single_solver_only():
    for p,kind,_ in POSITIVE:
        if kind: assert sum(f(p) is not None for _,f in T.SOLVERS)==1, p
