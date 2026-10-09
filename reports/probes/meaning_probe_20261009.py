"""Throwaway offline probe. Not a runtime gate or a new frozen benchmark."""
import sys,json,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'jobs'))
import ember_drafting_repair_candidates_eval as G
PAIRS=[
('modal','All cadets should inspect the silver bell prior to dawn.','All cadets should inspect the silver bell before dawn.','All cadets must inspect the silver bell before dawn.'),
('condition','You may borrow the green atlas only if the archivist has given consent.','You may borrow the green atlas only if the archivist has consented.','You may borrow the green atlas if the archivist agrees.'),
('reminder','Please remember that every pilot must wait beside the hangar until noon.','Please remember every pilot must wait beside the hangar until noon.','Every pilot must wait beside the hangar until noon.'),
('ownership',"Please return Nadia's wool cap back to the window alcove before dusk.","Please return Nadia's wool cap to the window alcove before dusk.","Please return your wool cap to the window alcove before dusk."),
('quantity','All of the keepers must retain exactly seven tokens until Monday.','All keepers must retain exactly seven tokens until Monday.','All keepers must retain seven tokens until Monday.'),
('uncertainty','Please note that the violet parcel might arrive on Wednesday.','The violet parcel might arrive Wednesday.','The violet parcel will arrive Wednesday.'),
('roles','Please note that Hugo lends Iris the red map on Tuesday.','Hugo lends Iris the red map Tuesday.','Iris lends Hugo the red map Tuesday.'),
('negation','It is important that nobody move the gold crate before Friday.','No one move the gold crate before Friday.','Move the gold crate before Friday.'),
('synonym','The apprentices must assemble in the marble foyer before noon on Thursday.','The apprentices must meet in the marble foyer before noon Thursday.','The apprentices must meet outside before noon Thursday.'),
('consent',"No one may lend Luca's field journal unless the custodian has given consent.","No one may lend Luca's field journal without custodian consent.","No one may lend Luca's field journal unless the custodian has given consent.")]
def guard(source,output):
 row=dict(id='probe',source=source,kind='shortening')
 old=G.grade_case(row,output,lambda r,o:True)
 new=dict(old,reasons=list(old['reasons']))
 reminder=r'\b(?:remember|recall|do not forget|don\x27t forget)\b'
 if re.search(reminder,source,re.I) and not re.search(reminder,output,re.I):
  new.update(passed=False,status='fail');new['reasons'].append('reminder_dropped')
 return old,new
records=[]
for name,source,good,bad in PAIRS:
 for label,answer in [('valid',good),('invalid',bad)]:
  old,new=guard(source,answer)
  records.append(dict(id=name+'-'+label,expected=label,source=source,answer=answer,existing=old,strict=new))


from collections import Counter
for label in ('valid','invalid'):
    print('CHALLENGE_SUMMARY',json.dumps({'expected':label,**{v:dict(Counter(r[v]['status'] for r in records if r['expected']==label)) for v in ('existing','strict')}}))
print('CHALLENGE_RESULTS',json.dumps(records))
from huggingface_hub import hf_hub_download
repo='Jmiller18899/ember-qwen3.5-4b-writing-repair7-20261009'
revision='c5fd030e5a23a73a489a3630a50de40bc1a854f4'
for filename in ('dev-before.json','dev-after.json'):
    rows=json.loads(Path(hf_hub_download(repo,'evidence/'+filename,revision=revision)).read_text())
    short=[r for r in rows if r['family']=='shortening']
    assert len(short)==40
    results=[]
    for row in short:
        old,new=guard(row['source'],row['output'])
        results.append(dict(id=row['id'],existing=old,strict=new))
    print('REPLAY_SUMMARY',json.dumps({'file':filename,'revision':revision,'rows':len(short),**{v:dict(Counter(r[v]['status'] for r in results)) for v in ('existing','strict')}}))
    print('REPLAY_DECISIONS',json.dumps({'file':filename,'decisions':results}))
