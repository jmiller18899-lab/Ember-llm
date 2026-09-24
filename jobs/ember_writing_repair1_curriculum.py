"""Build Writing Repair 1 data. Standard library only; NEVER runs model training.

Train/development/final-test authoring patterns and scenarios are disjoint.
The 256 retention rows are NEW verified rehearsal, not recycled evaluation rows.
Reference auditing is bounded to this authored curriculum, not a general judge.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import random
import re

VERSION = 'writing-repair1-curriculum-v1'
SEED = 924431
BASELINE_COMMIT = 'e6c9d37120802349190d9d66eaf35e7301a64c2e'
GRADER_COMMIT = '25924014c0e5d5a580a296b2841a1e6f6cbe3bb4'
BASE_REV = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
MODEL_REV = 'daf938bba5d4e6b650ec9d34a2d3ac56706cf549'

# Each (people, action, gerund, condition) is an authored semantic scenario.
CONTEXTS = {
 'train': [
  ('technicians','store exactly two spare fuses in drawer 38 before 6:25 PM on Tuesday','storing exactly two spare fuses in drawer 38 before 6:25 PM on Tuesday','the supervisor approves'),
  ('tutors','place at least four worksheets on the green shelf by 2:35 PM on Monday','placing at least four worksheets on the green shelf by 2:35 PM on Monday','the coordinator agrees'),
  ('gardeners','water only the east beds after 7:10 AM on Thursday','watering only the east beds after 7:10 AM on Thursday','the ground is dry'),
  ('porters','carry no more than three cartons to bay 46 before 5:40 PM on Friday','carrying no more than three cartons to bay 46 before 5:40 PM on Friday','the lift is working'),
  ('illustrators','save each sketch in folder 52 by 11:15 AM on Wednesday','saving each sketch in folder 52 by 11:15 AM on Wednesday','the editor confirms'),
  ('stewards','check every ticket at door 63 until 8:20 PM on Saturday','checking every ticket at door 63 until 8:20 PM on Saturday','the venue stays open'),
  ('caterers','keep the blue cooler below 5 degrees until 4:10 PM on Sunday','keeping the blue cooler below 5 degrees until 4:10 PM on Sunday','the backup chiller works'),
  ('surveyors','leave exactly six markers beside gate 74 before 9:45 AM on Tuesday','leaving exactly six markers beside gate 74 before 9:45 AM on Tuesday','the site manager agrees'),
  ('curators','return both labels to cabinet 85 by 3:05 PM on Thursday','returning both labels to cabinet 85 by 3:05 PM on Thursday','the exhibit is closed'),
  ('mechanics','leave at most five washers in tray 96 after 1:50 PM on Friday','leaving at most five washers in tray 96 after 1:50 PM on Friday','the repair is finished'),
 ],
 'dev': [
  ('docents','stack exactly seven brochures on the oak counter before 12:40 PM on Sunday','stacking exactly seven brochures on the oak counter before 12:40 PM on Sunday','the visitor center is open'),
  ('weavers','hang only the finished tapestries beside window 117 after 10:55 AM on Monday','hanging only the finished tapestries beside window 117 after 10:55 AM on Monday','the frames are ready'),
  ('marshals','inspect each barrier at checkpoint 128 by 5:15 PM on Wednesday','inspecting each barrier at checkpoint 128 by 5:15 PM on Wednesday','the route is clear'),
  ('binders','leave at most eight covers in bin 139 until 6:35 PM on Thursday','leaving at most eight covers in bin 139 until 6:35 PM on Thursday','the glue is dry'),
 ],
 'test': [
  ('divers','put exactly nine masks in cage 241 before 8:55 AM on Saturday','putting exactly nine masks in cage 241 before 8:55 AM on Saturday','the safety lead is present'),
  ('florists','move only the white lilies to the side veranda after 12:15 PM on Friday','moving only the white lilies to the side veranda after 12:15 PM on Friday','the delivery cart is available'),
  ('carpenters','count every hinge in box 263 by 4:35 PM on Tuesday','counting every hinge in box 263 by 4:35 PM on Tuesday','the inventory sheet is ready'),
  ('potters','keep at least ten tiles on rack 274 until 9:20 AM on Wednesday','keeping at least ten tiles on rack 274 until 9:20 AM on Wednesday','the kiln is cool'),
 ]
}

# Fields: semantic force, verbose source, faithful reference. Never train on bad rewrites.
SHORT = {
 'train': [
  ('request','{S} are kindly asked to {a}.','{S}, please {a}.'),
  ('request','We would appreciate it if {s} could {a}.','{S}, please {a}.'),
  ('requirement','It is a requirement that {s} {a}.','{S} must {a}.'),
  ('requirement','Please remember that {s} must {a}.','{S} must {a}.'),
  ('recommendation','It is recommended that {s} {a}.','{S} should {a}.'),
  ('recommendation','{S} ought to {a}, as a recommendation.','{S} should {a}.'),
  ('permission','{S} are permitted to {a}.','{S} may {a}.'),
  ('optional','It is entirely optional for {s} to {a}.','{S} may choose to {a}.'),
  ('no_obligation','Please note that {s} do not have to {a}.','{S} do not have to {a}.'),
  ('no_obligation','{S} are under no obligation to {a}.','{S} need not {a}.'),
  ('prohibition','{S} are strictly forbidden to {a}.','{S} must not {a}.'),
  ('prohibition','Please remember that {s} must not {a}.','{S} must not {a}.'),
  ('possibility','There is a possibility that {s} will {a}.','{S} might {a}.'),
  ('probability','Please note that {s} will probably {a}.','{S} will probably {a}.'),
  ('conditional_permission','{S} may {a}, but only if {c}.','{S} may {a} only if {c}.'),
  ('conditional_requirement','Please remember that {s} must {a} unless {c}.','{S} must {a} unless {c}.'),
 ],
 'dev': [
  ('request','A request for {s}: please {a}.','{S}, please {a}.'),
  ('requirement','{S} have a duty to {a}.','{S} must {a}.'),
  ('recommendation','The advice to {s} is to {a}.','{S} should {a}.'),
  ('permission','{S} have permission to {a}.','{S} may {a}.'),
  ('no_obligation','{S} have no obligation to {a}.','{S} need not {a}.'),
  ('prohibition','{S} are prohibited from {g}.','{S} must not {a}.'),
  ('possibility','It could be the case that {s} will {a}.','{S} might {a}.'),
  ('conditional_permission','{S} have permission to {a}, provided that {c}.','{S} may {a} provided that {c}.'),
 ],
 'test': [
  ('request','For {s}, the request is as follows: {a}.','{S}, please {a}.'),
  ('requirement','The following is mandatory for {s}: {a}.','{S} must {a}.'),
  ('recommendation','{S}: our recommendation is that you {a}.','{S} should {a}.'),
  ('permission','Permission to {a} has been granted to {s}.','{S} may {a}.'),
  ('optional','{S} can decide for themselves whether to {a}.','{S} may choose to {a}.'),
  ('no_obligation','There is no obligation on {s} to {a}.','{S} need not {a}.'),
  ('prohibition','For {s}, {g} is forbidden.','{S} must not {a}.'),
  ('possibility','It is possible, though not certain, that {s} will {a}.','{S} might {a}.'),
  ('probability','It is probable that {s} will {a}.','{S} will probably {a}.'),
  ('conditional_permission','Provided that {c}, permission to {a} is granted to {s}.','Provided that {c}, {s} may {a}.'),
  ('conditional_requirement','The instruction to {s} is mandatory: {a}, unless {c}.','{S} must {a} unless {c}.'),
  ('optional','{S} are free to choose whether or not to {a}.','{S} may choose to {a}.'),
 ]
}
SHORT_REQUEST = {
 'train':['Shorten the following without changing its meaning: {source}',
          'Make this more concise, preserving all facts and the strength of the instruction: "{source}"',
          'Rewrite briefly. Keep the people, conditions, quantities, and timing unchanged.\n{source}',
          'Condense this text faithfully: {source}'],
 'dev':['Give a shorter equivalent of this sentence:\n{source}',
        'Trim the wording, not the meaning: "{source}"'],
 'test':['An editor needs fewer words but exactly the same message. Original: {source}',
         'Produce a concise version that keeps who does what, when, and under what conditions.\n{source}',
         'Reduce this notice without adding certainty or changing permissions: "{source}"']
}

# Fictional entity pools are disjoint across all three splits.
NAMES = {
 'train':['Alina','Belen','Celia','Dalia','Esme','Farah','Greta','Hana','Ilse','Jada','Kesia','Leona',
          'Alden','Boris','Cedric','Damon','Emrys','Fabian','Gideon','Hugo','Ivo','Jonas','Keir','Luca'],
 'dev':['Mireya','Nell','Oona','Petra','Quincy','Ronan','Seth','Tobin'],
 'test':['Uma','Veda','Willa','Xenia','Yara','Zelia','Arlo','Bence','Caspar','Declan','Eamon','Fintan']
}
OWNERS = {
 'train':['Amaya','Bruno','Corin','Danica','Efrain','Fern','Galen','Hester','Indra','Jules','Kellan','Lumi'],
 'dev':['Mavis','Nevin','Orla','Perrin'],
 'test':['Qadir','Roslyn','Stellan','Thalia','Upton','Viola']
}
OBJECTS = {
 'train':['raincoat','e-reader','flask','lanyard','sketchbook','headset','toolkit','satchel'],
 'dev':['stopwatch','beret','monocular','folio'],
 'test':['harmonica','sundial','ocarina','tuning fork','brooch','pedometer','sextant','metronome']
}
WHEN = {
 'train':['Tuesday at 6:25 PM','Monday at 2:35 PM','Thursday at 7:10 AM','Friday at 5:40 PM','Wednesday at 11:15 AM','Saturday at 8:20 PM','Sunday at 4:10 PM','Tuesday at 9:45 AM'],
 'dev':['Sunday at 12:40 PM','Monday at 10:55 AM','Wednesday at 5:15 PM','Thursday at 6:35 PM'],
 'test':['Saturday at 8:55 AM','Friday at 12:15 PM','Tuesday at 4:35 PM','Wednesday at 9:20 AM']
}

# Six train, two development, three final-test request constructions per ownership role.
# The recipient owns o for R. For T the explicitly named other person owns it.
PERSPECTIVE = {
 'train': {
  'recipient':[
   'Draft a text to {r}: I found {p} {o} and can return it {w}.',
   'Write a message addressed to {r} saying I have {p} {o} and can return it {w}.',
   'Tell {r} in a draft that I found {p} {o} and will bring it back {w}.',
   'Compose a note for {r}: I have {p} {o}, but I cannot return it before {w}.',
   'Help me write to {r}. I found {p} {o} and might be able to return it {w}.',
   'Prepare a message to {r} explaining that I have {p} {o} and can return it {w} only if the office is open.',
  ],
  'third_party':[
   'Draft a text to {r}: {owner} left {p} {o} with me and I can return it to {owner} {w}.',
   'Write a message addressed to {r} saying I have {owner}\'s {o} and can return it to {owner} {w}.',
   'Tell {r} in a draft that {owner} lost {p} {o}, I found it, and I will return it to {owner} {w}.',
   'Compose a note for {r}: I have {owner}\'s {o}, but I cannot return it to {owner} before {w}.',
   'Help me write to {r}. {owner} left {p} {o} with me and I might return it to {owner} {w}.',
   'Prepare a message to {r} explaining that I have {owner}\'s {o} and can return it to {owner} {w} only if the office is open.',
  ]},
 'dev': {
  'recipient':[
   'Recipient: {r}. Facts for a draft: the {o} belongs to {r}; it is with me; I can return it {w}. Write the message itself.',
   'Turn these notes into a message for {r}: found {p} {o}; I cannot bring it back before {w}.',
  ],
  'third_party':[
   'Recipient: {r}. Facts for a draft: the {o} belongs to {owner}; it is with me; I can return it to {owner} {w}. Write the message itself.',
   'Turn these notes into a message for {r}: found {owner}\'s {o}; I cannot bring it back to {owner} before {w}.',
  ]},
 'test': {
  'recipient':[
   'I am writing directly to {r}, whose {o} I found. Make a sendable message saying this and that I can return it {w}.',
   'Use these details to draft my reply to {r}: their {o} is with me; a return {w} is possible, but not certain.',
   'For a message addressed to {r}, express both facts: I have the {o} that belongs to {r}; returning it {w} depends on the office being open.',
  ],
  'third_party':[
   'I am writing directly to {r} about the {o} I found that belongs to {owner}. Make a sendable message saying this and that I can return it to {owner} {w}.',
   'Use these details to draft my reply to {r}: {owner}\'s {o} is with me; a return to {owner} {w} is possible, but not certain.',
   'For a message addressed to {r}, express both facts: I have the {o} that belongs to {owner}; returning it to {owner} {w} depends on the office being open.',
  ]}
}


def normalized(s: str) -> str:
    return ' '.join(re.findall(r"[\w]+(?:'[\w]+)?|[+-]?\d+(?::\d+)?",s.lower().replace('’',"'")))


def perspective_body(split, role, t, v):
    """Reference surface; all claims and recipients are explicit, never 'sent'."""
    owned = 'your '+v['o'] if role=='recipient' else v['owner']+"'s "+v['o']
    recipient = '' if role=='recipient' else ' to '+v['owner']
    event = ('found' if t in (0,2,4) else 'have') if split=='train' else ('have' if t==0 else 'found')
    if split=='test': event = 'found' if t==0 else 'have'
    if role=='third_party' and split=='train' and t in (0,4):
        opening = f"{v['owner']} left {owned} with me"
    elif role=='third_party' and split=='train' and t==2:
        opening = f"{v['owner']} lost {owned}, and I found it"
    else:
        opening = f"I {event} {owned}"
    uncertainty = (split=='train' and t==4) or (split=='test' and t==1)
    negative = (split=='train' and t==3) or (split=='dev' and t==1)
    conditional = (split=='train' and t==5) or (split=='test' and t==2)
    if negative: closing=f"but I cannot return it{recipient} before {v['w']}"
    elif uncertainty:
        modal='might be able to' if split=='train' and role=='recipient' else 'might'
        closing=f"and I {modal} return it{recipient} {v['w']}"
    elif split=='train' and t==2: closing=f"and I will return it{recipient} {v['w']}"
    else: closing=f"and I can return it{recipient} {v['w']}"
    if conditional: closing+=' only if the office is open'
    body=opening+(', ' if negative else ' ')+closing+'.'
    if role=='third_party' and split=='train' and t in (0,4): event='left'
    if role=='third_party' and split=='train' and t==2: event='lost_found'
    modality=('cannot' if negative else ('might_be_able_to' if split=='train' and role=='recipient' else 'might') if uncertainty else 'will' if split=='train' and t==2 else 'can')
    return body, {'uncertain':uncertainty,'negated':negative,'conditional':conditional,'event':event,'return_modality':modality}


def writing_rows(split: str):
    rows=[]
    for t,(force,source,target) in enumerate(SHORT[split]):
        for j,(people,action,gerund,condition) in enumerate(CONTEXTS[split]):
            v={'s':'all '+people,'S':'All '+people,'a':action,'g':gerund,'c':condition}
            src=source.format(**v); answer=target.format(**v)
            rows.append({'id':f'wr1-{split}-short-{t:02}-{j:02}', 'split':split, 'task':'shortening',
                'family':'drafting','slice':'faithful_shortening','kind':'shortening','scoring':'rubric',
                'blueprint':f'{split}-short-{t:02}', 'contrast_group':f'{split}-scenario-{j:02}',
                'prompt':SHORT_REQUEST[split][t%len(SHORT_REQUEST[split])].format(source=src),
                'source':src,'answer':answer,'require':[people],'negative_any':[],
                'meaning':{'force':force,'subject':'all '+people,'action':action,
                    'gerund':gerund,'condition':condition if force.startswith('conditional_') else None},
                'provenance':'authored synthetic; not copied from a benchmark'})
    for role,patterns in PERSPECTIVE[split].items():
        for t,pattern in enumerate(patterns):
            for j in range(8):
                i=t*8+j
                v={'r':NAMES[split][i%len(NAMES[split])], 'owner':OWNERS[split][i%len(OWNERS[split])],
                   'o':OBJECTS[split][j%len(OBJECTS[split])], 'w':WHEN[split][i%len(WHEN[split])],
                   'p':('her','his','their')[i%3]}
                body,flags=perspective_body(split,role,t,v)
                rows.append({'id':f'wr1-{split}-{role}-{t:02}-{j:02}', 'split':split,'task':'perspective',
                    'family':'drafting','slice':'recipient_perspective','kind':'recipient' if role=='recipient' else 'thirdparty',
                    'scoring':'rubric','blueprint':f'{split}-{role}-{t:02}',
                    'contrast_group':f'{split}-ownership-{t:02}-{j:02}',
                    'prompt':pattern.format(**v), 'answer':f"Hi {v['r']}, {body}",
                    'require':[v['r'],v['o'],v['w']]+([] if role=='recipient' else [v['owner']]),
                    'object':v['o'],
                    'meaning':{'owner_role':role,'recipient':v['r'], 'owner':v['r'] if role=='recipient' else v['owner'],
                               'object':v['o'],'time':v['w'],'source_pronoun':v['p'],**flags},
                    'provenance':'authored synthetic; not copied from a benchmark'})
    return rows


def retention_rows():
    rows=[]
    def add(family,blueprint,prompt,answer,oracle):
        rows.append({'id':f'wr1-train-retention-{len(rows):03}','split':'train','task':'retention',
                     'family':family,'slice':'new_retention_practice','kind':'direct','scoring':'exact',
                     'blueprint':'train-retention-'+blueprint,'prompt':prompt,'answer':str(answer),
                     'oracle':oracle,'provenance':'new synthetic rehearsal; gold independently computed/checked'})
    # 64 arithmetic; explicit formula + independent calculator-style oracle.
    for i in range(64):
        a=137+i*7; b=21+i%17; d=5+i%9; k=3+i%8; typ=i%4
        if typ==0:
            p=f'A supply counter starts at {a}. Add {b}, then remove {d}. Return the resulting integer only.'
            ans=a+b-d; op='add_subtract'; operands=[a,b,d]
        elif typ==1:
            p=f'At a packing station, {k} bundles each hold {b} clips. There are another {d} loose clips, not another bundle. Give the total as an integer.'
            ans=k*b+d; op='multiply_add'; operands=[k,b,d]
        elif typ==2:
            p=f'{k*b} counters are shared equally among {k} stations. Each station then receives {d} more counters. Give the count at one station, integer only.'
            ans=b+d; op='divide_add'; operands=[k*b,k,d]
        else:
            p=f'Compute ({a} - {b}) * {k}. Give only the integer answer.'
            ans=(a-b)*k; op='subtract_multiply'; operands=[a,b,k]
        add('arithmetic',op,p,ans,{'op':op,'operands':operands})
    for i in range(48):
        rec={'rack':f'R{i+401}','tag':f'WR-{i+701}-K','state':('queued','open','closed')[i%3]}
        key=('rack','tag','state')[i%3]
        if i%2: p='Record: '+'; '.join(k+'='+v for k,v in rec.items())+f'. Return only the {key} value.'
        else: p=json.dumps(rec,sort_keys=True)+f'\nExtract {key}. Output the value alone.'
        add('extraction',f'record-{i%2}-{key}',p,rec[key],{'op':'extract','record':rec,'key':key})
    for i in range(32):
        hour=(i*5+3)%24; minute=(i*13+7)%60; duration=37+i*11
        start=datetime(2000,1,1,hour,minute); end=start+timedelta(minutes=duration)
        clock=lambda t:f'{(t.hour-1)%12+1}:{t.minute:02} '+('AM' if t.hour<12 else 'PM')
        p=f'The scanner cycle begins at {clock(start)} and lasts {duration} minutes. Give its ending clock time only, using H:MM AM/PM.'
        add('time_reasoning','scanner-clock',p,clock(end),{'op':'clock','hour':hour,'minute':minute,'duration':duration})
    # Grounding is balanced, uses actual answers (not only label classification),
    # and does not infer a result from a start, submission or payment event.
    scenarios=[
        ('consignment','dispatched','delivery date','Thursday'),
        ('shuttle','departed','arrival time','3:42 PM'),
        ('repair order','opened','completion date','Monday'),
        ('print job','submitted','finish time','11:28 AM'),
        ('application','received','decision','approved'),
        ('support ticket','created','resolution','replacement issued'),
        ('reservation','requested','confirmation status','confirmed'),
        ('invoice','paid','shipping date','Wednesday'),
    ]
    for i in range(48):
        item,observed,requested,known=scenarios[i%8]; batch=i//8
        supplied=batch%2==1; ref=f'QH-{4200+i}'
        facts={observed:('Tuesday','Friday','Sunday')[batch//2]}
        if supplied: facts[requested]=known
        record='; '.join(k+': '+v for k,v in facts.items())
        p=f'{item.capitalize()} {ref}. Recorded facts: {record}. What is the {requested}? Use only these facts; say when the information is insufficient.'
        ans=known if supplied else f'There is not enough information to determine the {requested}.'
        add('grounding',f'evidence-{i%8}-{int(supplied)}',p,ans,{'op':'grounded_answer','record':facts,'key':requested})
    missing=[
      ('announcement','Make my announcement clearer and less wordy.'),
      ('paragraph','Could you proofread the paragraph I am referring to? No paragraph is included.'),
      ('letter','Please rewrite my letter with a friendlier tone.'),
      ('caption','Help me improve this caption. I have not pasted it yet.'),
      ('instructions','Simplify my instructions while keeping all warnings.'),
      ('speech','Make the speech more concise without losing its main point.'),
      ('email','Please correct the punctuation in the email. Its contents are not attached.'),
      ('review','Can you make my review sound more balanced?'),
      ('biography','Translate my biography into Italian. I have not provided the biography.'),
      ('proposal','Please turn my proposal into a one-paragraph summary.'),
      ('memo','Rewrite the memo for a nontechnical audience.'),
      ('story','Check the story for spelling errors. No story text is supplied.'),
      ('notice','Shorten my notice but keep the deadline and restrictions.'),
      ('invitation','Make the invitation warmer without adding details.'),
      ('description','Translate this description into German. The source description is missing.'),
      ('report','Condense my report to three sentences. The report itself is not in this message.'),
    ]
    for i,(noun,p) in enumerate(missing):
        ans=f'Please paste the {noun} you want me to work on.'
        add('clarification',f'missing-text-{i}',p,ans,{'op':'clarify','noun':noun})
    for i in range(16):
        text=f'parcel m{7300+i} is ready'; op=('uppercase','lowercase','replace','first_word')[i%4]
        if op=='uppercase':
            p=f'Change this text to uppercase and return it alone: "{text}"'; ans=text.upper()
        elif op=='lowercase':
            text=text.upper();p=f'Use lowercase only for this supplied sentence: "{text}"'; ans=text.lower()
        elif op=='replace':
            text=f'draft j{8100+i} approved';p=f'Replace "draft" with "final" in this text, with no other changes: "{text}"';ans=text.replace('draft','final')
        else:
            text=f'item{i+9300} awaits pickup';p=f'Return the first word of this supplied text, nothing else: "{text}"';ans=text.split()[0]
        add('clarification','supplied-'+op,p,ans,{'op':op,'text':text})
    for i in range(32):
        limit=45+i; value=35+i*2; kind=i%4
        if kind==0:
            p=f'Rule: label a reading HIGH only when it is greater than {limit}; otherwise label it LOW. The reading is {value}. Give the label alone.'
            ans='HIGH' if value>limit else 'LOW'; oracle={'op':'threshold','value':value,'limit':limit}
        elif kind==1:
            words=[f'K{(i*7)%19+20}',f'K{(i*3)%17+50}',f'K{(i*11)%23+80}']
            p=f'Return only the last entry of this ordered list: {", ".join(words)}.'
            ans=words[-1]; oracle={'op':'last','items':words}
        elif kind==2:
            values=[i+57,i+31,i+74]
            p=f'Put these integers in ascending order, separated by commas and spaces: {", ".join(map(str,values))}.'
            ans=', '.join(map(str,sorted(values)));oracle={'op':'sort','values':values}
        else:
            words=[f'N{i+64}',f'N{i+26}']
            p=f'Join these two codes with one hyphen, preserving their order: {words[0]}, {words[1]}. Return only the joined codes.'
            ans='-'.join(words);oracle={'op':'join','items':words}
        add('direct_answer',('threshold','last-item','sort','join')[kind],p,ans,oracle)
    return rows


def build():
    parts={s:writing_rows(s) for s in ('train','dev','test')}
    parts['train']+=retention_rows()
    # Training order changes only within train; final test order is fixed.
    random.Random(SEED).shuffle(parts['train'])
    return parts


def oracle_answer(o):
    """Independent retention check, intentionally separate from the generators."""
    op=o['op']; x=o.get('operands')
    if op=='add_subtract': return str(sum(x[:2])-x[2])
    if op=='multiply_add': return str(x[2]+x[0]*x[1])
    if op=='divide_add':
        if x[0]%x[1]: raise ValueError('non-integer division')
        return str(x[0]//x[1]+x[2])
    if op=='subtract_multiply': return str(x[0]*x[2]-x[1]*x[2])
    if op=='extract': return o['record'][o['key']]
    if op=='clock':
        total=(o['hour']*60+o['minute']+o['duration'])%1440
        return f'{(total//60-1)%12+1}:{total%60:02} '+('AM' if total<720 else 'PM')
    if op=='grounded_answer':
        return o['record'].get(o['key'], f"There is not enough information to determine the {o['key']}.")
    if op=='clarify': return f"Please paste the {o['noun']} you want me to work on."
    if op=='uppercase': return o['text'].upper()
    if op=='lowercase': return o['text'].lower()
    if op=='replace': return 'final'+o['text'][5:]
    if op=='first_word': return o['text'].partition(' ')[0]
    if op=='sort': return ', '.join(str(x) for x in sorted(o['values']))
    if op=='join': return o['items'][0]+'-'+o['items'][1]
    if op=='threshold': return ('LOW','HIGH')[o['value']>o['limit']]
    if op=='last': return o['items'][-1]
    raise ValueError('unknown oracle: '+op)


def allowed_references(row):
    """Closed-form reference checks from the explicit semantic ledger.

    This validates authored gold, not arbitrary model paraphrases. Alternative
    correct model outputs are to be judged semantically, NOT against this list.
    """
    m=row['meaning']
    if row['task']=='shortening':
        S=m['subject'][0].upper()+m['subject'][1:]; a=m['action']; f=m['force']; c=m['condition']
        middle={'request':', please ','requirement':' must ','recommendation':' should ',
                'permission':' may ','optional':' may choose to ','prohibition':' must not ',
                'possibility':' might ','probability':' will probably '}
        if f in middle: return [S+middle[f]+a+'.']
        if f=='no_obligation': return [S+' need not '+a+'.',S+' do not have to '+a+'.']
        if f=='conditional_requirement': return [S+' must '+a+' unless '+c+'.']
        if f=='conditional_permission':
            if 'only if' in row['source']:
                return [S+' may '+a+' only if '+c+'.']
            return [S+' may '+a+' provided that '+c+'.', 'Provided that '+c+', '+m['subject']+' may '+a+'.']
        raise ValueError('Unknown force')
    owned=('your' if m['owner_role']=='recipient' else m['owner']+"'s")+' '+m['object']
    to='' if m['owner_role']=='recipient' else ' to '+m['owner']
    event=m['event']
    opening={
        'found':'I found '+owned,'have':'I have '+owned,
        'left':m['owner']+' left '+owned+' with me',
        'lost_found':m['owner']+' lost '+owned+', and I found it',
    }[event]
    mode=m['return_modality'].replace('_',' ')
    if mode=='cannot': tail=', but I cannot return it'+to+' before '+m['time']
    else: tail=' and I '+mode+' return it'+to+' '+m['time']
    if m['conditional']: tail+=' only if the office is open'
    return ['Hi '+m['recipient']+', '+opening+tail+'.']


def reference_errors(row):
    """Validate authored references and their explicit semantic constraints.

    This is NOT a model-output scorer. Final answers require the frozen grader
    and blind semantic review; exact matching to one reference is insufficient.
    """
    errors=[]; answer=row['answer']; m=row.get('meaning',{}); low=answer.lower()
    if row['task']=='retention':
        if answer!=oracle_answer(row['oracle']): errors.append('oracle_mismatch')
        return errors
    if row['task']=='shortening':
        if len(answer)>=len(row['source']): errors.append('not_shorter')
        if m['action'] not in row['source'] and m['gerund'] not in row['source']:
            errors.append('source_payload_changed')
        if m['subject'] not in row['source'].lower(): errors.append('source_subject_changed')
        if m['subject'] not in low: errors.append('subject_lost')
        if m['action'] not in answer: errors.append('action_or_protected_fact_changed')
        expected={
            'request':', please ', 'requirement':' must ', 'recommendation':' should ',
            'permission':' may ', 'optional':' may choose to ', 'no_obligation':None,
            'prohibition':' must not ', 'possibility':' might ', 'probability':' will probably ',
            'conditional_permission':' may ', 'conditional_requirement':' must ',
        }[m['force']]
        if expected and expected not in answer: errors.append('force_changed')
        if m['force']=='no_obligation' and not (' need not ' in answer or ' do not have to ' in answer): errors.append('force_changed')
        if m['force'] not in ('requirement','conditional_requirement','prohibition') and re.search(r'\bmust\b',answer): errors.append('invented_requirement')
        if m['force']=='requirement' and ' must not ' in answer: errors.append('negation_added')
        if m['condition'] and m['condition'] not in answer: errors.append('condition_dropped')
    else:
        for fact in row['require']:
            if fact not in answer: errors.append('missing_fact:'+fact)
        owned='your '+m['object'] if m['owner_role']=='recipient' else m['owner']+"'s "+m['object']
        if owned not in answer: errors.append('wrong_owner')
        if m['owner_role']=='third_party' and 'your '+m['object'] in answer: errors.append('ownership_reassigned')
        if not answer.startswith('Hi '+m['recipient']+', '): errors.append('wrong_addressee')
        if m['negated'] and 'cannot return' not in answer: errors.append('negation_dropped')
        if m['uncertain'] and not re.search(r'might (?:be able to )?return',answer): errors.append('uncertainty_lost')
        if m['conditional'] and 'only if the office is open' not in answer: errors.append('condition_dropped')
    if answer not in allowed_references(row): errors.append('reference_semantic_ledger_mismatch')
    if re.search(r"\b(?:i sent|message sent|i have sent|i already texted)\b",low): errors.append('false_action_claim')
    return errors


def audit(parts, external=()):
    errors=[]; ids=set(); prompts={}; source_split={}; blueprints={}; totals={}; statuses=Counter()
    expected={'train':512,'dev':64,'test':96}
    if set(parts)!=set(expected): errors.append('split_names')
    for split,rows in parts.items():
        totals[split]=dict(Counter(r['task'] for r in rows))
        if len(rows)!=expected[split]: errors.append(split+':count')
        for row in rows:
            key=row.get('id'); p=normalized(row.get('prompt',''))
            if key in ids: errors.append('duplicate_id:'+str(key))
            ids.add(key)
            if p in prompts: errors.append('duplicate_prompt:'+str(key))
            prompts[p]=split
            if row.get('split')!=split: errors.append('wrong_split:'+str(key))
            if not all(isinstance(row.get(k),str) and row[k].strip() for k in ('id','prompt','answer','blueprint')): errors.append('invalid_schema:'+str(key));continue
            if row['blueprint'] in blueprints and blueprints[row['blueprint']]!=split: errors.append('blueprint_leak')
            blueprints[row['blueprint']]=split
            if 'source' in row:
                src=normalized(row['source'])
                if src in source_split and source_split[src]!=split: errors.append('source_leak:'+str(key))
                source_split[src]=split
            for err in reference_errors(row): errors.append(str(key)+':'+err)
            statuses['checked_references']+=1
    if totals.get('train')!={'shortening':160,'perspective':96,'retention':256}: errors.append('train_mix')
    for s,n in [('dev',32),('test',48)]:
        if totals.get(s)!={'shortening':n,'perspective':n}: errors.append(s+':mix')
    for split,n in [('train',48),('dev',16),('test',24)]:
        roles=Counter(r['meaning']['owner_role'] for r in parts[split] if r['task']=='perspective')
        if roles!={'recipient':n,'third_party':n}: errors.append(split+':role_balance')
    ext_prompts={normalized(r['prompt']) for r in external if isinstance(r.get('prompt'),str)}
    ext_sources={normalized(r['source']) for r in external if isinstance(r.get('source'),str)}
    for p in set(prompts)&ext_prompts: errors.append('external_prompt_overlap:'+p[:80])
    for p in set(source_split)&ext_sources: errors.append('external_source_overlap:'+p[:80])
    return {'version':VERSION,'passed':not errors,'errors':errors,'counts':{s:len(v) for s,v in parts.items()},
        'mix':totals,'reference_checks':statuses['checked_references'],'external_rows_checked':len(external),
        'exact_cross_split_prompt_duplicates':len(prompts)!=sum(map(len,parts.values())),
        'blueprint_counts':{s:len({r['blueprint'] for r in rows}) for s,rows in parts.items()},
        'training_started':False,'model_inference_performed':False,'independent_human_review':False}


def training_view(parts):
    return [{k:r[k] for k in ('id','prompt','answer')} for r in parts['train']]


def write_package(outdir: Path):
    parts=build(); report=audit(parts)
    if not report['passed']: raise ValueError(json.dumps(report['errors'],indent=2))
    outdir.mkdir(parents=True,exist_ok=True)
    files={}
    for name,rows in [('train',training_view(parts)),('train_metadata',parts['train']),('dev',parts['dev']),('test',parts['test'])]:
        raw=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows).encode()
        path=outdir/(name+'.jsonl'); path.write_bytes(raw)
        files[path.name]={'rows':len(rows),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    manifest={'version':VERSION,'seed':SEED,'files':files,'base_model':'Qwen/Qwen3.5-4B',
        'base_revision':BASE_REV,'adapter':'Jmiller18899/ember-qwen3.5-4b-repair2','adapter_revision':MODEL_REV,
        'baseline_commit':BASELINE_COMMIT,'grader_commit':GRADER_COMMIT,
        'training_input':'train.jsonl','evaluation_only':['dev.jsonl','test.jsonl'],
        'final_test_policy':'Do not load into training, tune prompts, or select checkpoints using test.jsonl. Evaluate once after candidate selection.',
        'retention_provenance':'256 newly authored/algorithmic verified rehearsal rows, not literal replay from prior training',
        'new_test_is_model_unseen':True,'synthetic_template_based':True,'independent_human_review':False,
        'training_authorized_by_this_script':False,'automatic_promotion':False}
    (outdir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (outdir/'local_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return manifest, report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=Path('data/writing_repair1'))
    a=ap.parse_args()
    manifest,report=write_package(a.output)
    print(json.dumps({'manifest':manifest,'audit':report},indent=2))

if __name__=='__main__': main()
