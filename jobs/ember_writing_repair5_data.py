"""Authored WR5 targets and training isolation; generation grades stay manual."""
from collections import Counter
from string import Formatter
import hashlib, json, random, re

VERSION="ember-writing-repair5-data-v1"
SEED=431
WRAPPERS=[
    'Shorten this without changing its meaning: "{s}"',
    'Give a briefer version, keeping the roles and conditions:\n{s}',
    'Make this sentence shorter without changing its force: {s}',
    'Trim this, keeping the facts, ownership and timing: "{s}"',
]
FRAME=[
 ("I am writing with a polite request for you to check the {obj} at the {place} before {time}.",
  "Please check the {obj} at the {place} before {time}.","check",("Please","before")),
 ("This is a request addressed to you in this message: please place {owner}'s {obj} in the {place} until {day}.",
  "Please place {owner}'s {obj} in the {place} until {day}.","place",("Please","until")),
 ("I have an instruction for you to remember for later: the {group} must keep exactly {q} {items} in the {place}.",
  "Remember: the {group} must keep exactly {q} {items} in the {place}.","keep",("Remember","must","exactly")),
 ("I would like to make a polite request in this message for you to inspect all {q} {items} in the {place} on {day}.",
  "Please inspect all {q} {items} in the {place} on {day}.","inspect",("Please","all")),
 ("I want to remind you of the following fact for your records: {owner}'s {obj} might arrive at the {place} by {time}.",
  "Remember: {owner}'s {obj} might arrive at the {place} by {time}.","arrive",("Remember","might","by")),
 ("Here is something that I want you to remember for later: you may carry exactly {q} {items} into the {place} after {time}.",
  "Remember: you may carry exactly {q} {items} into the {place} after {time}.","carry",("Remember","may","exactly","after")),
 ("I am sending you this message to make the following request: please tell the {group} to leave the {obj} at the {place} on {day}.",
  "Please tell the {group} to leave the {obj} at the {place} on {day}.","tell",("Please tell",)),
 ("This is something that I am asking you to check in this message: all the {group} must wait at the {place} until {time}.",
  "Please check: all the {group} must wait at the {place} until {time}.","check",("Please check","all","must","until")),
 ("Here is a request that I am sending to you about my belongings: please bring my {obj} to the {place} on {day}.",
  "Please bring my {obj} to the {place} on {day}.","bring",("Please","my")),
 ("I am explaining my plan to you in the words of this message: I will return {owner}'s {obj} before {time}.",
  "I will return {owner}'s {obj} before {time}.","return",("I will","before")),
 ("I want you to know the following information about my plans: I should leave the {obj} at the {place} on {day}.",
  "I should leave the {obj} at the {place} on {day}.","leave",("I should",)),
 ("For your information, I am explaining the situation in the following words: the {obj} may remain in the {place} until {day}.",
  "The {obj} may remain in the {place} until {day}.","remain",("may","until")),
]
CONDITION=[
 ("I am giving you this instruction in writing: you can move the {obj} only if the {role} agrees.",
  "You can move the {obj} only if the {role} agrees.","move",("can","only if","agrees")),
 ("Here is my instruction for you to follow: you must not open the {obj} unless the {role} gives permission.",
  "You must not open the {obj} unless the {role} gives permission.","open",("must not","unless","permission")),
 ("This is the request I want to make to you: please collect exactly {q} {items} from the {place}.",
  "Please collect exactly {q} {items} from the {place}.","collect",("Please","exactly")),
 ("I am telling you this so that you remember it: all the {group} should inspect the {obj} on {day}.",
  "Remember: all the {group} should inspect the {obj} on {day}.","inspect",("Remember","all","should")),
 ("The following is the request I am making to you: please carry at most {q} {items} into the {place}.",
  "Please carry at most {q} {items} into the {place}.","carry",("Please","at most")),
 ("I am setting out this instruction for you: you may use the {obj} only after {time}.",
  "You may use the {obj} only after {time}.","use",("may","only after")),
 ("I am giving you the following information to remember: only the {group} may enter the {place} before {time}.",
  "Remember: only the {group} may enter the {place} before {time}.","enter",("Remember","only","may","before")),
 ("Here is the instruction I am asking you to follow: please leave at least {q} {items} in the {place} until {day}.",
  "Please leave at least {q} {items} in the {place} until {day}.","leave",("Please","at least","until")),
]
LITTLE=[
 ("Please check {owner}'s {obj} on {day}, if you please.",
  "Please check {owner}'s {obj} on {day}.","check",("Please",)),
 ("Please bring my {obj} before {time}, if you please.",
  "Please bring my {obj} before {time}.","bring",("Please","my","before")),
 ("Please return the {obj} to the {place}, if you please.",
  "Please return the {obj} to the {place}.","return",("Please",)),
 ("Please inspect all {q} {items} on {day}, if you please.",
  "Please inspect all {q} {items} on {day}.","inspect",("Please","all")),
 ("Please leave {owner}'s {obj} until {time}, if you please.",
  "Please leave {owner}'s {obj} until {time}.","leave",("Please","until")),
 ("Please keep exactly {q} {items} in the {place}, if you please.",
  "Please keep exactly {q} {items} in the {place}.","keep",("Please","exactly")),
 ("Please move the {obj} only after {time}, if you please.",
  "Please move the {obj} only after {time}.","move",("Please","only after")),
 ("Please record at most {q} {items} by {time}, if you please.",
  "Please record at most {q} {items} by {time}.","record",("Please","at most","by")),
]
DEV_SHORT=[
 ("My request is that you verify exactly {q} {items} at the {place} before {time}.",
  "Please verify exactly {q} {items} at the {place} before {time}.","verify",("Please","exactly","before")),
 ("The instruction I want to give is this: keep {owner}'s {obj} in the {place} until {day}.",
  "Keep {owner}'s {obj} in the {place} until {day}.","keep",("until",)),
 ("You can examine the {obj} at the {place}, with permission required from the {role}.",
  "You can examine the {obj} at the {place} with the {role}'s permission.","examine",("can","permission")),
 ("My plan is not a commitment: I should bring the {obj} to the {place} on {day}.",
  "I should bring the {obj} to the {place} on {day}; no commitment.","bring",("should","no commitment")),
 ("I am asking you to check this fact: nobody has moved {owner}'s {obj} from the {place}.",
  "Please check: nobody has moved {owner}'s {obj} from the {place}.","check",("Please check","nobody has moved")),
 ("This is my instruction: only if the {role} agrees may the {group} take exactly {q} {items}.",
  "Only if the {role} agrees may the {group} take exactly {q} {items}.","take",("Only if","agrees","may","exactly")),
 ("My request concerns all the {group}: please ask them to wait at the {place} until {time}.",
  "Please ask all the {group} to wait at the {place} until {time}.","ask",("Please ask","all","until")),
 ("The rule I want you to remember is that the {obj} must not leave the {place} before {day}.",
  "Remember: the {obj} must not leave the {place} before {day}.","leave",("Remember","must not","before")),
]
TRAIN_MESSAGES=[
 "I found {owned} and {force} return it on {day}.",
 "I wrapped {owned} and {force} bring it to the {place} before {time}.",
 "I collected {owned} and {force} leave it at the {place} on {day}.",
 "I repaired {owned} and {force} hand it back after {time}.",
 "I checked {owned} at the {place} on {day}; I have not moved it elsewhere.",
 "I stored {owned} in the {place} on {day}; nobody has moved it elsewhere.",
 "I can return {owned} only after {time}.",
 "I will bring {owned} to the {place} on {day} only if the {role} agrees.",
]
MODAL_MESSAGES=[
 "I located {owned}; I {force} return it on {day}.",
 "I packaged {owned}; I {force} bring it to the {place} before {time}.",
 "I retrieved {owned}; I {force} leave it at the {place} on {day}.",
 "I fixed {owned}; I {force} hand it back after {time}.",
]
DEV_MESSAGES=[
 "I catalogued {owned} and {force} send it back on {day}.",
 "I examined {owned} and {force} deliver it to the {place} before {time}.",
 "I secured {owned} at the {place} on {day}; I have not handed it to anyone.",
 "I labelled {owned} at the {place} on {day}; nobody has handed it to anyone.",
]
FORCES=("can","should","might","will")
GREETINGS=("Hi {n}, ","{n}, ","Hello {n}, ")

def norm(text):
    return " ".join(re.findall(r"\w+",text.casefold()))

def values(split,i):
    if split=="train":
        people=[("Leona","her"),("Darius","his"),("Emery","their"),("Tamsin","her"),
                ("Bastian","his"),("Quinn","their"),("Vera","her"),("Lucian","his")]
        owners={"her":["Alina","Esme","Liora","Yvette"],"his":["Matteo","Nico","Hugo","Cedric"],
                "their":["River","Jules","Avery","Skyler"]}
        objects=["brass compass","purple thermos","silver stapler","amber lantern",
                 "checked blanket","teal satchel","leather pouch","orange flask"]
        groups=["weavers","archivists","bakers","menders","potters","rangers","gardeners","docents"]
        places=["map alcove","brick kiosk","fern patio","linen depot","north annex","stone loft","rear atrium","copper gallery"]
    else:
        people=[("Fiona","her"),("Gareth","his"),("Auden","their"),("Imogen","her")]
        owners={"her":["Elodie","Marisol"],"his":["Pascal","Renato"],"their":["Ellis","Marlow"]}
        objects=["violet lunchbox","steel binoculars","woven cushion","pearl brooch"]
        groups=["carvers","surveyors","florists","referees"]
        places=["birch pavilion","ceramic alcove","garden cabin","river kiosk"]
    pick=lambda seq:seq[i%len(seq)]
    n,pronoun=pick(people)
    owner_pronoun=("her","his","their")[i%3]
    return dict(n=n,pronoun=pronoun,owner=owners[owner_pronoun][(i//3)%len(owners[owner_pronoun])],
                owner_pronoun=owner_pronoun,obj=pick(objects),group=pick(groups),place=pick(places),
                day=pick(["Thursday","Saturday","Monday","Wednesday","Friday","Sunday","Tuesday"]),
                time=f"{1+i%8}:{(i*13+23)%60:02d} PM",q=str(3+i%11),
                items=pick(["tickets","pins","labels","reels"]),role=pick(["curator","foreman","captain","custodian"]))

def protected_fields(template,v):
    keys=dict.fromkeys(k for _,k,_,_ in Formatter().parse(template) if k)
    return [v[k] for k in keys]

def short_row(split,group,t,j,template,i):
    source,answer,verb,anchors=template
    v=values(split,i)
    return dict(id=f"wr5-{split}-short-{group}-{t:02}-{j:02}",split=split,training_allowed=split=="train",
                family="shortening",shortening_group=group,source=source.format(**v),
                prompt=WRAPPERS[(t+j)%4].format(s=source.format(**v)),answer=answer.format(**v),
                structure=source,action_verb=verb,protected=protected_fields(answer,v)+list(anchors),
                semantic_review_required=True)

def message_row(split,row_id,template,v,relation,representation,group,i,**meta):
    owner=v["n"] if relation=="recipient" else v["owner"]
    pronoun=v["pronoun"] if relation=="recipient" else v["owner_pronoun"]
    owned=owner+"'s "+v["obj"] if representation=="named" else pronoun+" "+v["obj"]
    ownership="your "+v["obj"] if relation=="recipient" else owner+"'s "+v["obj"]
    context="" if relation=="recipient" or representation=="named" else f"Owner entry: {owner} owns the {v['obj']}. "
    w=dict(v,owned=owned,ownership=ownership)
    src=template.format(**w)
    answer=template.format(**dict(w,owned=ownership))
    wrapper="Write my message to {n}: {s}" if split=="train" else "Compose my note addressed to {n}: {s}"
    return dict(id=row_id,split=split,training_allowed=split=="train",family="recipient",
                prompt=context+wrapper.format(n=v["n"],s=src),source=src,
                answer=GREETINGS[i%3].format(n=v["n"])+answer,recipient=v["n"],owner=owner,
                owner_pronoun=pronoun,object=v["obj"],relation=relation,representation=representation,
                message_group=group,structure=template+"|"+wrapper+"|"+relation+"|"+representation,
                protected=[v["n"],ownership]+[v[k] for _,k,_,_ in Formatter().parse(template) if k and k!="owned"],
                force=v.get("force") if "{force}" in template else None,semantic_review_required=True,**meta)

def writing(split):
    if split not in ("train","dev"):
        raise ValueError("Only training and fresh development are supported")
    rows=[]
    if split=="train":
        for group,templates,repeats,offset in [("frame",FRAME,8,0),("condition",CONDITION,2,101),("little",LITTLE,2,173)]:
            for t,template in enumerate(templates):
                for j in range(repeats):
                    rows.append(short_row(split,group,t,j,template,offset+t*8+j))
        for p,pronoun in enumerate(("her","his","their")):
            for j in range(16):
                v=values(split,j*3+p)
                v["force"]=FORCES[(j//4)%4]
                rows.append(message_row(split,f"wr5-train-owner-{pronoun}-{j:02}",TRAIN_MESSAGES[j%8],
                                        v,"third_party","pronoun","third_party_pronoun",j+p))
        for j in range(32):
            v=values(split,211+j);v["force"]=FORCES[(j//8)%4]
            rows.append(message_row(split,f"wr5-train-named-{j:02}",TRAIN_MESSAGES[j%8],v,
                                    "third_party","named","third_party_named",j))
        for j in range(32):
            v=values(split,307+j);v["force"]=FORCES[(j//8)%4]
            rows.append(message_row(split,f"wr5-train-recipient-{j:02}",TRAIN_MESSAGES[j%8],v,
                                    "recipient","named" if j%2==0 else "pronoun","recipient",j))
        for j in range(8):
            for force in ("should","will"):
                v=dict(values(split,401+j),force=force)
                rows.append(message_row(split,f"wr5-train-modal-{j:02}-{force}",MODAL_MESSAGES[j%4],v,
                                        "third_party","named","modal_contrast",j,pair_id=f"modal-{j:02}"))
    else:
        for t,template in enumerate(DEV_SHORT):
            for j in range(4):
                rows.append(short_row(split,"diagnostic",t,j,template,701+t*4+j))
        for j in range(16):
            v=values(split,809+j);v["force"]=FORCES[(j//4)%4]
            rows.append(message_row(split,f"wr5-dev-owner-{j:02}",DEV_MESSAGES[j%4],v,
                                    "third_party","pronoun","third_party_pronoun",j))
        for j in range(4):
            for force in ("should","will"):
                v=dict(values(split,907+j),force=force)
                rows.append(message_row(split,f"wr5-dev-modal-{j:02}-{force}",DEV_MESSAGES[j%2],v,
                                        "third_party","named","modal_contrast",j,pair_id=f"dev-modal-{j:02}"))
        for j in range(8):
            v=values(split,1009+j);v["force"]=FORCES[j%4]
            rows.append(message_row(split,f"wr5-dev-recipient-{j:02}",DEV_MESSAGES[j%4],v,
                                    "recipient","named" if j%2==0 else "pronoun","recipient",j))
    for row in rows:validate_reference(row)
    return rows

def validate_reference(row):
    answer=row["answer"];source=row["source"]
    if not all(x.casefold() in answer.casefold() for x in row["protected"]):
        raise ValueError("Required fact or force was dropped: "+row["id"])
    if row["family"]=="shortening":
        if norm(answer)==norm(source) or len(answer)>=len(source) or len(answer.split())>=len(source.split()):
            raise ValueError("Reference must actually shorten: "+row["id"])
        limit=0.92 if row["shortening_group"]=="little" else 0.8
        if row["split"]=="train" and len(answer)/len(source)>limit:
            raise ValueError("Reference exceeds shortening budget: "+row["id"])
        if row["action_verb"] not in norm(answer).split():
            raise ValueError("Original action verb changed")
        if "ensure" in norm(answer).split() and "ensure" not in norm(source).split():
            raise ValueError("Cannot introduce ensure")
        if re.findall(r"\d+",answer)!=re.findall(r"\d+",source):
            raise ValueError("Numeric information changed")
    else:
        if not any(answer.startswith(g.format(n=row["recipient"])) for g in GREETINGS):
            raise ValueError("Wrong addressee")
        if row.get("force") and not re.search(r"\b"+row["force"]+r"\b",answer):
            raise ValueError("Modal force changed")
        for claim in ("I have not moved it elsewhere","nobody has moved it elsewhere",
                      "I have not handed it to anyone","nobody has handed it to anyone",
                      "only after","only if"):
            if claim in source and claim not in answer:
                raise ValueError("Negative or condition scope changed")

def build_train_rows(retention):
    expected={"arithmetic":64,"extraction":64,"grounding":48,"clarification":48,"direct":32}
    if len(retention)!=256 or Counter(r["family"] for r in retention)!=expected:
        raise ValueError("Retention composition changed")
    if any(r.get("split")!="train" or r.get("training_allowed") is not True for r in retention):
        raise ValueError("Require training-only retention")
    rows=writing("train")
    short=[r for r in rows if r["family"]=="shortening"]
    if sum(len(r["answer"])/len(r["source"]) for r in short)/len(short)>0.75:
        raise ValueError("Mean reference length exceeds 0.75")
    rows.extend(dict(id="wr5-retain-"+r["id"],split="train",training_allowed=True,
                     family=r["family"],prompt=r["prompt"],answer=r["answer"]) for r in retention)
    random.Random(SEED+3).shuffle(rows)
    sft_bytes(rows)
    return rows

def sft_bytes(rows):
    ids,prompts=set(),set()
    for r in rows:
        if r.get("split")!="train" or r.get("training_allowed") is not True:
            raise ValueError("Evaluation rows cannot enter training")
        if not all(isinstance(r.get(k),str) and r[k].strip() for k in ("id","prompt","answer")):
            raise ValueError("Invalid training text")
        if r["id"] in ids or norm(r["prompt"]) in prompts:
            raise ValueError("Duplicate training row: "+r["id"]+" | "+r["prompt"])
        ids.add(r["id"]);prompts.add(norm(r["prompt"]))
    return "".join(json.dumps({k:r[k] for k in ("id","prompt","answer")},sort_keys=True,ensure_ascii=False)+"\n" for r in rows).encode()

def fourgrams(text):
    tokens=norm(text).split()
    return {tuple(tokens[i:i+4]) for i in range(len(tokens)-3)}

def audit(train,dev,historical,benchmark_sources):
    for field in ("id","prompt","source","structure"):
        a={norm(r[field]) for r in train if r.get(field)}
        b={norm(r[field]) for r in dev if r.get(field)}
        if a & b:raise ValueError("Cross-split overlap: "+field)
    history_prompts={norm(r["prompt"]) for r in historical}
    history_sources={norm(r["source"]) for r in historical if r.get("source")}
    new=[r for r in train if not r["id"].startswith("wr5-retain-")]+dev
    if any(norm(r["prompt"]) in history_prompts or (r.get("source") and norm(r["source"]) in history_sources) for r in new):
        raise ValueError("Previously observed writing prompt/source reused")
    benchmark_grams=set().union(*(fourgrams(s) for s in benchmark_sources)) if benchmark_sources else set()
    for r in train:
        if r["family"]=="shortening" and fourgrams(r["source"]) & benchmark_grams:
            raise ValueError("Training source shares a benchmark four-token frame: "+r["id"])
    short=[r for r in train if r["family"]=="shortening"]
    return dict(train=len(train),fresh_dev=len(dev),historical_rows_checked=len(historical),
                benchmark_shortening_sources_checked=len(benchmark_sources),exact_prompt_source_overlaps=0,
                cross_split_structure_overlap=0,benchmark_four_token_overlaps=0,
                mean_reference_length_ratio=round(sum(len(r["answer"])/len(r["source"]) for r in short)/len(short),6),
                scope="Observed history only; reserved final holdouts remain unloaded")

def shortening_source(row):
    if row.get("kind") in ("shorten","shortening") and row.get("source"):
        return row["source"]
    if str(row.get("id","")).startswith("draft-short-"):
        match=re.search(r"'([^']+)'",row.get("prompt",""))
        return match.group(1) if match else None
    return None

def length_stats(pairs):
    if not pairs:raise ValueError("No shortening cases")
    return dict(rows=len(pairs),verbatim_copy=sum(norm(out)==norm(src) for src,out in pairs),
                not_shorter=sum(len(out.strip())>=len(src) for src,out in pairs),
                mean_length_ratio=round(sum(len(out.strip())/len(src) for src,out in pairs)/len(pairs),6))
