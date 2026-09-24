"""Narrow deterministic arithmetic tool for Ember's runtime policy (no model weights involved).

solve(prompt) returns the exact answer string when the prompt is one of three deterministic problem shapes,
and None otherwise (the model then answers as usual):
  multiply_add  "N groups with K items each, plus X extra items"      -> N*K+X
  add_subtract  "start with A; B are added; C are removed"            -> A+B-C (any order of adds/removes)
  clock_add     "leaves at H:MM AM and takes D minutes; arrival?"     -> H:MM AM/PM (12-hour, rolls over noon/midnight)
It fires only when every number in the prompt is accounted for by one of those roles, each operation is named by
an unambiguous verb or cue, and the prompt explicitly asks for a bare number or a bare time. Anything else
(missing duration, extra numbers, rates, words instead of digits, no format request) returns None, so
grounding/abstention behaviour stays with the model and its scoped rules.
"""
import re

NUM=re.compile(r"(?<![\d:.])\d+(?![\d:]|\.\d)")
NUMBER_ONLY=re.compile(r"\b(number only|only the number|just the number|digits only|answer with a number only|reply with (?:only )?a number)\b",re.I)
TIME_ONLY=re.compile(r"(\b(?:answer|reply|respond)(?: as)?:? h:mm\b|\bformat(?:ted)?:?(?: as| like)? (?:h:mm|\d{1,2}:\d{2} ?[ap]m)\b|\breply with (?:only )?the time\b|\b(?:give|answer with|reply with|say) (?:only |just )?the time(?: only)?\b|\bjust give the time\b|\bonly the time\b|\btime only\b|\bjust the time\b)",re.I)
# a clock shown only as a formatting example ("formatted like 6:15 PM", "like 3:40 PM") is not problem data
EXAMPLE_CLOCK=re.compile(r"\b(?:like|e\.g\.,?|such as|for example|as in)\s+\d{1,2}:\d{2}\s*[AaPp]\.?\s*[Mm]\b\.?",re.I)
TOTAL_Q=re.compile(r"\b(total|in all|altogether|combined|how many)\b",re.I)
REMAIN_Q=re.compile(r"\b(remain|remaining|left|now|how many|in total|end up)\b",re.I)
SUB_CUE=re.compile(r"\b(remove[sd]?|removing|take[sn]?|took|taken|sold|sell[s]?|use[sd]?|lose[s]?|lost|gave|give[sn]?|broke|broken|spent|spend[s]?|ate|eaten|eat[s]?|drop(?:ped|s)?|discard(?:ed|s)?|minus|subtract(?:ed|s)?|shipped out|went out|borrowed|drained|poured out|carried out|taken out|leaked)\b",re.I)
ADD_CUE=re.compile(r"\b(add(?:s|ed|ing)?|arrive[sd]?|bought|buy[s]?|gain(?:s|ed)?|receive[sd]?|new|more|join(?:s|ed)?|deliver(?:s|ed|y)|donated|put in|brought in|carried in|came in|poured in|filled in|plus|restock(?:s|ed)?)\b",re.I)
START_CUE=re.compile(r"\b(has|had|have|holds?|held|contains?|contained|starts? with|started with|begins? with|began with|there (?:are|were)|stock of|opens? with)\b",re.I)
EACH_CUE=re.compile(r"\b(each|apiece|per)\b",re.I)
EXTRA_WORD=r"(?:extra|loose|spare|additional|leftover|single|separate|more)"

def _nums(p): return [(int(m.group()),m.start(),m.end()) for m in NUM.finditer(p)]

# ---- multiply_add ----
GROUP_EACH=re.compile(r"(\d+)\s+(?:[a-z-]+\s+){0,2}?[a-z-]+\b[^.?!\d]{0,40}?\b(\d+)\s+(?:[a-z-]+\s+){0,2}?[a-z-]+\s*(?:,\s*)?(?:each|apiece|per\s+[a-z-]+|in each(?:\s+[a-z-]+)?|in every\s+[a-z-]+)\b",re.I)
GROUP_OF=re.compile(r"(\d+)\s+(?:[a-z-]+\s+){0,1}?[a-z-]+\s+of\s+(\d+)\s+[a-z-]+",re.I)
EXTRA=re.compile(r"(?:\b(?:plus|and|also|with)\b[^.?!\d]{0,30}?)?\b(\d+)\s+(?:[a-z-]+\s+){0,1}?"+EXTRA_WORD+r"\b|\b(\d+)\s+(?:[a-z-]+\s+){0,2}?(?:left over|on the side|outside (?:the )?[a-z-]+|not in (?:a|any) [a-z-]+|(?:are|is|were|was|sit|sits|remain)\s+(?:loose|extra|spare|left over|separate))|\bplus\s+(\d+)\b",re.I)
def multiply_add(p):
    if not (NUMBER_ONLY.search(p) and TOTAL_Q.search(p)) or SUB_CUE.search(p): return None
    nums=_nums(p)
    if len(nums)!=3: return None
    g=GROUP_EACH.search(p) or GROUP_OF.search(p)
    if not g: return None
    used={g.start(1),g.start(2)}
    def at(m): i=next(i for i in (1,2,3) if m.group(i)); return m.start(i),int(m.group(i))
    extras={at(m) for m in EXTRA.finditer(p)}-{e for e in (at(m) for m in EXTRA.finditer(p)) if e[0] in used}
    if len(extras)!=1: return None
    (xs,x),=extras
    if {n[1] for n in nums}!=used|{xs}: return None
    n,k=int(g.group(1)),int(g.group(2))
    return str(n*k+x)

# ---- add_subtract ----
CLAUSE=re.compile(r"[.;,!?]|\b(?:and|then|but|after that|later)\b",re.I)
def add_subtract(p):
    if not (NUMBER_ONLY.search(p) and REMAIN_Q.search(p)) or EACH_CUE.search(p) or re.search(r"\b\d+\s+[a-z-]+\s+of\s+\d+",p,re.I): return None
    nums=_nums(p)
    if len(nums)<3 or len(nums)>5: return None
    bounds=[0]+[m.end() for m in CLAUSE.finditer(p)]+[len(p)]
    def clause(pos):
        lo=max(b for b in bounds if b<=pos); hi=min(b for b in bounds if b>pos) if any(b>pos for b in bounds) else len(p)
        return p[lo:hi]
    clauses=[clause(s) for _,s,_ in nums]
    if len(set(clauses))!=len(clauses): return None       # two numbers in one clause: ambiguous role
    first=clauses[0]
    if ADD_CUE.search(first) or SUB_CUE.search(first) or not START_CUE.search(first) and not re.match(r"\s*(?:a|an|the)?\s*[a-z -]*\b(?:starts?|begins?)\b",first,re.I) and not re.search(r"\b(?:starts?|began|begins?) (?:the day )?(?:with|at)\b",first,re.I):
        return None
    total=nums[0][0]
    for (v,_,_),c in zip(nums[1:],clauses[1:]):
        a,s=bool(ADD_CUE.search(c)),bool(SUB_CUE.search(c))
        if a==s: return None                               # no cue, or both: do not guess
        total+=v if a else -v
    return str(total) if total>=0 else None

# ---- clock_add ----
CLOCK=re.compile(r"\b(\d{1,2}):(\d{2})\s*([AaPp])\.?\s*[Mm]\b\.?")
START_VERB=re.compile(r"\b(leaves?|left|departs?|departed|starts?|started|begins?|began|opens?|kicks? off|sets? off|set out|takes? off|boards?|sails?|sailed|put|placed)\b",re.I)
DUR=re.compile(r"\b(?:(\d+)\s*(?:hours?|hrs?|h)\b(?:\s*(?:and\s*)?(\d+)\s*(?:minutes?|mins?|m)\b)?|(\d+)\s*(?:minutes?|mins?)\b)",re.I)
DUR_CUE=re.compile(r"\b(takes?|took|lasts?|lasted|runs?|ran|for|is|was|travel time|duration|ride|trip|long|later|after)\b",re.I)
END_VERB=re.compile(r"\b(arriv\w*|lands?|landed|ends?|ended|finish\w*|docks?|docked|expires?|expired|gets? (?:in|there)|reach\w*)\b",re.I)
ASKS_START=re.compile(r"\b(?:what time|when)\b[^.?!]{0,25}\b(?:leave|left|start|started|depart|departed|begin|began|set off|take off)\b",re.I)
END_Q=re.compile(r"\b(arrive|arrives|arrival|land|lands|end|ends|finish|finishes|done|over|get there|reach|dock|docks|expire|expires)\b",re.I)
def clock_add(p):
    if not TIME_ONLY.search(p) or not END_Q.search(p) or not START_VERB.search(p): return None
    p=EXAMPLE_CLOCK.sub(lambda m:" "*len(m.group()),p)   # keep offsets, drop the example
    clocks=list(CLOCK.finditer(p)); durs=list(DUR.finditer(p))
    if len(clocks)!=1 or len(durs)!=1: return None
    c,d=clocks[0],durs[0]
    pre=p[max(0,c.start()-60):c.start()]   # the clock must be the start: "leaves at 9:10 AM", not "arrives at 3:00 PM"
    if not START_VERB.search(pre) or END_VERB.search(pre) or ASKS_START.search(p): return None
    stray=[n for n in _nums(p) if not (c.start()<=n[1]<c.end() or d.start()<=n[1]<d.end())]
    if stray or not DUR_CUE.search(p[max(0,d.start()-40):d.start()]): return None
    h,m=int(c.group(1)),int(c.group(2))
    if not 1<=h<=12 or m>59: return None
    mins=(int(d.group(1))*60+int(d.group(2) or 0)) if d.group(1) else int(d.group(3))
    t=((h%12+(12 if c.group(3).upper()=="P" else 0))*60+m+mins)%1440
    return f"{(t//60-1)%12+1}:{t%60:02d} {'AM' if t<720 else 'PM'}"

SOLVERS=(("multiply_add",multiply_add),("add_subtract",add_subtract),("clock_add",clock_add))
def solve(prompt):
    """(kind, answer) for the single solver that fires, else None. Two solvers firing is treated as ambiguous."""
    hits=[(k,a) for k,f in SOLVERS if (a:=f(prompt)) is not None]
    return hits[0] if len(hits)==1 else None
