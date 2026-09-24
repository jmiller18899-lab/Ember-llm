"""Fresh record-question prompts for checking the context rule beyond promotion v2's single template.

Field names, separators, question wording and values all differ from promotion v2. 'folder' is included
on purpose, because it is the value the model dropped. Scoring: every value appears in the answer, and no
field name appears in place of the value ("the item", "the object", ...).
"""
import re
NAMES=["Ines","Bilal","Chen","Dora","Eli","Farah","Gus","Hana"]
ITEMS=["folder","binder","stapler","folder","lanyard","folder","thermos","clipboard"]
DAYS=["Thursday","Monday","Wednesday","Friday","Tuesday","Saturday","Thursday","Monday"]
FORMS=[
 ("Context record: owner={o}; item={i}; return_day={d}. Now answer: Who owns the item and when is it returned?",("item",)),
 ("Loan log | borrower={o} | object={i} | due={d}. Who borrowed which object, and when is it due back?",("object",)),
 ("Lost property entry: finder=front desk; owner={o}; item={i}; pickup_day={d}. Whose item is it and what day can it be picked up?",("item",)),
 ("Asset: holder={o}, asset={i}, return_on={d}. Who holds the asset and when does it come back?",("asset",)),
]
def build():
    rows=[]
    for f,(form,fields) in enumerate(FORMS):
        for j in range(6):
            k=(f*6+j)%8; o,i,d=NAMES[k],ITEMS[(k+f)%8],DAYS[(k+2*f)%8]
            rows.append({"id":f"ctx-holdout-{f}{j}","prompt":form.format(o=o,i=i,d=d),"values":[o,i,d],"field_names":list(fields)})
    return rows
def score(row,out):
    o=out.lower()
    has=all(v.lower() in o for v in row["values"])
    swapped=any(re.search(rf"\bthe {f}\b",o) for f in row["field_names"]) and row["values"][1].lower() not in o
    return has and not swapped
