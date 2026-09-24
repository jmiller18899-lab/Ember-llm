"""Fresh probes for the policy v3 candidates (evaluation only), scored with the suite v3 scorer.

The v3 fixes were designed against suite v3's failures, so suite v3 is now a development set. These 36 prompts check
the same six failure types in different wording: other label kinds (gate, floor, bus, flight, "since <year>"),
other verbs (quit, requires, went in), other message openers (Let X know, Tell X, Send a note to X), other record
layouts for "whose" questions, and other text-transformation requests. Some deliberately fall outside the tool's
labels ("Flight 237"), so the model has to answer.
"""
def build():
    rows=[]
    def add(fam,scoring,prompt,**kw): rows.append({"id":f"probe-{fam}-{sum(r['family']==fam for r in rows):02d}","family":fam,"scoring":scoring,"prompt":prompt,**kw})
    # arithmetic with label numbers (8)
    for p,a in [("At gate 7, 5 carts hold 12 suitcases each, plus 4 loose suitcases. Total suitcases? Number only.","64"),
                ("Since 2012 the orchard has kept 48 hives. 15 hives were added and 11 were sold. How many hives now? Number only.","52"),
                ("On floor 3, there are 6 shelves with 14 binders on each shelf and 5 extra binders. How many binders? Number only.","89"),
                ("Bus 42 started the day with 37 passengers. 18 passengers boarded and 25 got off. How many passengers remain? Number only.","30"),
                ("Flight 237 has 9 rows with 6 seats per row plus 3 jump seats. How many seats in total? Number only.","57"),
                ("A shop opened in 2008 with 70 lamps. 24 lamps were delivered and 31 were sold. How many lamps remain? Number only.","63"),
                ("Locker 19 holds 4 boxes with 7 pens in each box and 6 extra pens. Total pens? Number only.","34"),
                ("Built in 1998, the stadium had 120 flags. 45 new flags were added and 60 were removed. How many flags now? Number only.","105")]:
        add("arith_distractor","exact",p,answer=a)
    # tool vocabulary (6)
    for p,a in [("A choir has 52 singers. 14 new singers join and 9 singers quit. How many singers remain? Number only.","57"),
                ("A league had 64 teams. 12 teams signed up and 20 teams dropped out. How many teams are left? Number only.","56"),
                ("A gym has 90 members. 25 members join and 33 members leave. How many members now? Number only.","82")]:
        add("arith_tool_shapes","exact",p,answer=a)
    for p,a in [("A pie went in at 10:40 AM and requires 95 minutes. When is it done? Time only.","12:15 PM"),
                ("A stew goes into the pot at 9:30 PM and needs 3 hours. When is it done? Time only.","12:30 AM"),
                ("Bread goes into the oven at 6:15 AM and needs 2 hours 50 minutes. When is it done? Just the time.","9:05 AM")]:
        add("time_calc","exact",p,answer=a)
    # drafting (8)
    for nm,pos,ob,day,form in [("Maria","her","wallet","Friday","Let {n} know that I found {p} {o} and will return it {d}."),
                               ("Kofi","his","umbrella","Tuesday","Tell {n} that I still have {p} {o} and will bring it back {d}."),
                               ("Alex","their","laptop","Saturday","Send a note to {n}: I have {p} {o} and can return it {d}."),
                               ("Elena","her","sunglasses","Monday","Text {n} that I grabbed {p} {o} by mistake and will return them {d}."),
                               ("Jun","his","badge","tomorrow","Let {n} know I picked up {p} {o} and will drop it off {d}."),
                               ("Riley","their","charger","Thursday","Tell {n} that {p} {o} is with me and I'll hand it over {d}.")]:
        add("drafting","rubric",form.format(n=nm,p=pos,o=ob,d=day),kind="message",name=nm,facts=[ob,day],forbid=pos)
    for s,facts in [("Could everyone on the planning committee please come to the north stairwell no later than 9:15?",["north stairwell","9:15"]),
                    ("Staff members are requested to assemble in the back parking lot by 6 this evening for the photo.",["back parking lot","6"])]:
        add("drafting","rubric",f"Trim this down, keeping the place and the time: \"{s}\"",kind="shorten",source=s,facts=facts)
    # context: whose-item and other layouts (6)
    for p,v in [("Found item log -> owner: Ines, item: tripod, pickup: Friday. Whose item is this and when can it be picked up?",["Ines","tripod","Friday"]),
                ("owner=Pablo | item=folder | collect=Sunday. Whose item is it and what day is collection?",["Pablo","folder","Sunday"]),
                ("Desk note: Aiko left a thermos here; she will collect it Tuesday. Whose thing is it, and when is she collecting it?",["Aiko","thermos","Tuesday"]),
                ('{"owner": "Marta", "item": "lantern", "collect_day": "Thursday"} Whose item is it and when will it be collected?',["Marta","lantern","Thursday"]),
                ("Record: owner=Theo; item=binder; return_day=Monday. Tell me whose item it is and when it goes back.",["Theo","binder","Monday"]),
                ("Two entries. First: owner=Lin; item=kettle; return_day=Friday. Second: owner=Sara; item=drone; return_day=Tuesday. For the first entry only, whose item is it and when is it returned?",["Lin","kettle","Friday"])]:
        add("context","rubric",p,values=v,forbid=["Sara","Tuesday"] if "Sara" in p else [])
    # clarification (8): 6 with no text, 2 with text
    for p in ["Can you proofread my cover letter?","Summarize this article for me, please.","Reword my intro so it sounds friendlier.",
              "Could you simplify the paragraph above?","Please translate my message into French.","Make this email sound more polite, thanks."]:
        add("clarification","rubric",p,expect="ask")
    add("clarification","rubric","Proofread this: 'Their going to the park tomorow.'",expect="do",require=["they're","tomorrow"])
    add("clarification","rubric","Translate into French: 'Thank you'",expect="do",require=["merci"])
    assert len(rows)==36 and len({r["prompt"] for r in rows})==36
    return rows
