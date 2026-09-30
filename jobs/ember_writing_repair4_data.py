"""WR4 authored writing references; not a model-output grader.

Target compact requests, owner perspective, modal force, and negative-claim scope.
Only training and fresh development are available. The frozen grader and final
holdouts are not changed. Contracts check authored targets, not semantic ability.
"""
from collections import Counter
import hashlib
import json
import random
import re
from string import Formatter

VERSION = 'ember-writing-repair4-data-v1'
SEED = 431

# Independently authored wording for this experiment. These are synthetic data,
# not independent human adjudication or copies of the failing benchmark prompts.
# (compact source, reference, literal force/scope anchors)
TRAIN_SHORT = [
 ('Please take care to ensure that the {group} sign in at the {place} before {time}.',
  'Please ensure the {group} sign in at the {place} before {time}.', ('Please ensure', 'before')),
 ('Please make certain that the {obj} stays in the {place} until {day}.',
  'Please ensure the {obj} stays in the {place} until {day}.', ('Please ensure', 'until')),
 ('Please keep in mind that you should inspect the {obj} at the {place} on {day}.',
  'Please remember: you should inspect the {obj} at the {place} on {day}.', ('Please', 'should')),
 ('Do not forget that the {group} may take exactly {q} {items} after {time}.',
  'Remember: the {group} may take exactly {q} {items} after {time}.', ('Remember', 'may', 'exactly', 'after')),
 ('If the {role} gives permission, you can leave the {obj} at the {place} on {day}.',
  'With the {role}\'s permission, you can leave the {obj} at the {place} on {day}.', ('permission', 'can')),
 ('The {group} must examine the {obj} first, then carry it into the {place}.',
  'The {group} must examine the {obj} before carrying it into the {place}.', ('must', 'before', 'into')),
 ('The {obj} might arrive by {time}; it is not a confirmed delivery.',
  'The {obj} might arrive by {time}; delivery is unconfirmed.', ('might', 'by', 'unconfirmed')),
 ('I am able to return {owner}\'s {obj} on {day}, but not before {time}.',
  'I can return {owner}\'s {obj} on {day}, but not before {time}.', ('I can', 'not before')),
 ('I should bring the {obj} to the {place} on {day}; I am not promising I will.',
  'I should bring the {obj} to the {place} on {day}; this is not a promise.', ('I should', 'not a promise')),
 ('Please tell the {group} to leave exactly {q} {items} in the {place} until {time}.',
  'Please tell the {group}: leave exactly {q} {items} in the {place} until {time}.', ('Please tell', 'exactly', 'until')),
 ('Please see to it that the {group} wait at the {place} until {time}.',
  'Please ensure the {group} wait at the {place} until {time}.', ('Please ensure', 'until')),
 ('It is not necessary for you to move the {obj} from the {place} before {day}.',
  'You need not move the {obj} from the {place} before {day}.', ('need not', 'before')),
 ('The {group} must not use the {obj} in the {place} until {day} has arrived.',
  'The {group} must not use the {obj} in the {place} before {day}.', ('must not', 'before')),
 ('You can collect at least {q} {items} at the {place}, provided that the {role} agrees.',
  'You can collect at least {q} {items} at the {place} if the {role} agrees.', ('can', 'at least', 'if', 'agrees')),
 ('I put {owner}\'s {obj} in the {place}; I have not taken it anywhere else.',
  'I put {owner}\'s {obj} in the {place} and have not taken it elsewhere.', ('I put', 'have not taken it elsewhere')),
 ('The {obj} is in the {place}; nobody has taken it out of there.',
  'The {obj} is in the {place}; nobody has removed it.', ('nobody has removed it',)),
]
DEV_SHORT = [
 ('Please check that all the {group} are at the {place} no later than {time}.',
  'Please check that all the {group} are at the {place} by {time}.', ('Please check', 'all', 'by')),
 ('Please bear in mind that the {obj} must stay at the {place} through {day}.',
  'Please remember: the {obj} must stay at the {place} through {day}.', ('Please', 'must', 'through')),
 ('The {group} should take at most {q} {items}; that is advice, not a limit imposed on them.',
  'The {group} should take at most {q} {items}; advice, not an imposed limit.', ('should', 'at most', 'not an imposed limit')),
 ('It is possible that the {obj} will reach the {place} on {day}, but this is not certain.',
  'The {obj} might reach the {place} on {day}; this is uncertain.', ('might', 'uncertain')),
 ('You are permitted to check the {obj} at the {place}, but only after {time}.',
  'You may check the {obj} at the {place} only after {time}.', ('may', 'only after')),
 ('Before moving the {obj}, the {group} must first record exactly {q} {items}.',
  'The {group} must record exactly {q} {items} before moving the {obj}.', ('must', 'exactly', 'before')),
 ('I left {owner}\'s {obj} at the {place}; I did not subsequently move it.',
  'I left {owner}\'s {obj} at the {place} and haven\'t moved it since.', ('I left', 'haven\'t moved it since')),
 ('Please ask the {group} to check the {obj} only if the {role} says it is allowed.',
  'Please ask the {group} to check the {obj} only with the {role}\'s permission.', ('Please ask', 'only', 'permission')),
]
TRAIN_MODAL = [
 ('I located {owned} and {force} return it on {day}.',
  'I located {ownership} and {force} return it on {day}.'),
 ('I packed {owned} and {force} leave it at the {place} on {day}.',
  'I packed {ownership} and {force} leave it at the {place} on {day}.'),
 ('I repaired {owned} and {force} deliver it after {time}.',
  'I repaired {ownership} and {force} deliver it after {time}.'),
 ('I recovered {owned} and {force} bring it to the {place} by {time}.',
  'I recovered {ownership} and {force} bring it to the {place} by {time}.'),
]
DEV_MODAL = [
 ('I retrieved {owned} and {force} hand it over before {time}.',
  'I retrieved {ownership} and {force} hand it over before {time}.'),
 ('I inspected {owned} and {force} take it to the {place} on {day}.',
  'I inspected {ownership} and {force} take it to the {place} on {day}.'),
]
TRAIN_NEGATIVE = [
 ('I checked {owned} at the {place} on {day}; {negative}.',
  'I checked {ownership} at the {place} on {day}; {negative}.',
  {'speaker': 'I have not carried it elsewhere', 'global': 'nobody has carried it elsewhere'}),
 ('I stored {owned} in the {place} on {day}; {negative}.',
  'I stored {ownership} in the {place} on {day}; {negative}.',
  {'speaker': 'I did not give it to anyone', 'global': 'it has not been given to anyone'}),
]
DEV_NEGATIVE = [
 ('I photographed {owned} at the {place} on {day}; {negative}.',
  'I photographed {ownership} at the {place} on {day}; {negative}.',
  {'speaker': 'I have not sent it elsewhere', 'global': 'it has not been sent elsewhere'}),
]
FORMS = [('recipient', 'named'), ('recipient', 'pronoun'),
         ('third_party', 'named'), ('third_party', 'pronoun')]
FORCES = ['can', 'should', 'might', 'will']
WRAPPERS = ['Shorten this without changing its meaning: "{s}"',
            'Give a briefer version, keeping the roles and conditions:\n{s}',
            'Make this sentence shorter without changing its force: {s}',
            'Trim this, keeping the facts, ownership and timing: "{s}"']


def norm(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def values(split, i):
    if split == 'train':
        people = [('Celia','her'),('Jonas','his'),('Mira','her'),('Omar','his'),
                  ('Sasha','their'),('Theo','his'),('Ada','her'),('Robin','their')]
        others = [('Bianca','her'),('Dante','his'),('Elisa','her'),('Felix','his'),
                  ('Harper','their'),('Isaac','his'),('Lydia','her'),('Morgan','their')]
        objects = ['wool scarf','green binder','canvas toolkit','blue helmet',
                   'spare charger','ceramic mug','small speaker','red folder']
        groups = ['editors','guides','musicians','painters','cleaners','ushers','tailors','visitors']
        places = ['art room','south office','side hall','repair shop','glass studio','east room','quiet lounge','main foyer']
    else:
        people = [('Nadia','her'),('Elias','his'),('Rowan','their'),('Simone','her')]
        others = [('Keira','her'),('Malik','his'),('Parker','their'),('Soren','his')]
        objects = ['ticket wallet','mosaic panel','white headset','linen apron']
        groups = ['sculptors','translators','plumbers','singers']
        places = ['cedar shed','painting booth','front cabin','metal workshop']
    pick = lambda seq: seq[i % len(seq)]
    n, pronoun = pick(people)
    owner, owner_pronoun = pick(others)
    return dict(n=n, pronoun=pronoun, owner=owner, owner_pronoun=owner_pronoun,
                obj=pick(objects), group=pick(groups), place=pick(places),
                day=pick(['Friday','Sunday','Tuesday','Saturday','Thursday','Monday','Wednesday']),
                time=f'{2+i%5}:{(i*11+17)%60:02d} PM', q=str(5+i%9),
                items=pick(['badges','sheets','discs','cups']),
                role=pick(['steward','manager','instructor','organizer']))


def protected_fields(template, values):
    keys = dict.fromkeys(k for _, k, _, _ in Formatter().parse(template) if k)
    return [values[k] for k in keys]


def message_row(split, row_id, source, answer, v, relation, representation, **metadata):
    owner = v['n'] if relation == 'recipient' else v['owner']
    ownership = 'your '+v['obj'] if relation == 'recipient' else owner+"'s "+v['obj']
    context = ''
    if representation == 'named':
        owned = owner+"'s "+v['obj']
    elif relation == 'recipient':
        owned = v['pronoun']+' '+v['obj']
    else:
        # Bind the possessive before the addressee is introduced.
        context = (f"Ownership context: {owner} confirmed that {v['owner_pronoun']} own {v['obj']} belongs to {owner}. ")
        owned = 'the '+v['obj']
    w = dict(v, owned=owned, ownership=ownership)
    wrapper = 'Draft my message to {n}: {s}' if split == 'train' else 'Prepare a short note from me addressed to {n}: {s}'
    src = source.format(**w)
    return dict(id=row_id, split=split, training_allowed=split=='train', family='recipient',
                prompt=context+wrapper.format(n=v['n'],s=src), source=src,
                answer='Hi '+v['n']+', '+answer.format(**w), recipient=v['n'], owner=owner,
                object=v['obj'], relation=relation, representation=representation,
                structure=source+'|'+wrapper+'|'+relation+'|'+representation,
                protected=[v['n']]+protected_fields(answer,w), semantic_review_required=True,
                **metadata)


def writing(split):
    if split not in ('train','dev'):
        raise ValueError('Only training and fresh development are supported')
    rows = []
    short = TRAIN_SHORT if split == 'train' else DEV_SHORT
    for t,(source,answer,anchors) in enumerate(short):
        for j in range(8 if split=='train' else 4):
            v=values(split,t*8+j)
            src=source.format(**v)
            rows.append(dict(id=f'wr4-{split}-short-{t:02}-{j:02}',split=split,
                training_allowed=split=='train',family='shortening',prompt=WRAPPERS[(t+j)%4].format(s=src),
                source=src,answer=answer.format(**v),structure=source,
                protected=protected_fields(answer,v)+list(anchors),semantic_review_required=True))
    modal = TRAIN_MODAL if split=='train' else DEV_MODAL
    for t,(source,answer) in enumerate(modal):
        for j,force in enumerate(FORCES):
            forms = FORMS if split=='train' else [(rel,'pronoun' if (t+j)%2 else 'named') for rel in ('recipient','third_party')]
            for k,(relation,representation) in enumerate(forms):
                v=dict(values(split,17+t*7+j),force=force)
                rows.append(message_row(split,f'wr4-{split}-modal-{t:02}-{force}-{relation}-{representation}',
                    source,answer,v,relation,representation,force=force))
    negative = TRAIN_NEGATIVE if split=='train' else DEV_NEGATIVE
    for t,(source,answer,claims) in enumerate(negative):
        for scope,claim in claims.items():
            for j in range(4 if split=='train' else 2):
                for k,(relation,representation) in enumerate(FORMS):
                    v=dict(values(split,61+t*7+j),negative=claim)
                    rows.append(message_row(split,f'wr4-{split}-negative-{t:02}-{scope}-{j:02}-{relation}-{representation}',
                        source,answer,v,relation,representation,negative_scope=scope,negative_claim=claim))
    for r in rows:
        validate_reference(r)
    return rows


def validate_reference(row):
    """Authored-target contracts only; never used to grade model generations."""
    a=row['answer']
    if not all(f.casefold() in a.casefold() for f in row['protected']):
        raise ValueError('Required fact, perspective or force was dropped: '+row['id'])
    if row['family']=='shortening':
        if len(a.split())>=len(row['source'].split()) or len(a)>=len(row['source']):
            raise ValueError('Reference must actually shorten the compact source: '+row['id'])
        if re.findall(r'\d+',a)!=re.findall(r'\d+',row['source']):
            raise ValueError('Numeric information changed')
    else:
        if not a.startswith('Hi '+row['recipient']+', '):
            raise ValueError('Wrong addressee')
        if row.get('force') and not re.search(r'\b'+row['force']+r'\b',a):
            raise ValueError('Modal force changed')
        if row.get('negative_claim') and row['negative_claim'] not in a:
            raise ValueError('Negative claim scope changed')


def build_train_rows(retention):
    expected={'arithmetic':64,'extraction':64,'grounding':48,'clarification':48,'direct':32}
    if len(retention)!=256 or any(r.get('split')!='train' or r.get('training_allowed') is not True for r in retention):
        raise ValueError('Require the 256 pinned training retention rows')
    if Counter(r['family'] for r in retention)!=expected:
        raise ValueError('Retention composition changed')
    rows=writing('train')
    for r in retention:
        rows.append(dict(id='wr4-retain-'+r['id'],split='train',training_allowed=True,
                         family=r['family'],prompt=r['prompt'],answer=r['answer']))
    random.Random(SEED+3).shuffle(rows)
    sft_bytes(rows)
    return rows


def sft_bytes(rows):
    seen_ids,seen_prompts=set(),set()
    for r in rows:
        if r.get('split')!='train' or r.get('training_allowed') is not True:
            raise ValueError('Evaluation rows cannot enter the training export')
        if not all(isinstance(r.get(k),str) and r[k].strip() for k in ('id','prompt','answer')):
            raise ValueError('Invalid training fields')
        if r['id'] in seen_ids or norm(r['prompt']) in seen_prompts:
            raise ValueError('Duplicate training row')
        seen_ids.add(r['id']);seen_prompts.add(norm(r['prompt']))
    return ''.join(json.dumps({k:r[k] for k in ('id','prompt','answer')},sort_keys=True,ensure_ascii=False)+'\n' for r in rows).encode()


def audit(train,dev,historical):
    for field in ('id','prompt','source','structure'):
        a={norm(r[field]) for r in train if r.get(field)}
        b={norm(r[field]) for r in dev if r.get(field)}
        if a & b:
            raise ValueError('Cross-split overlap: '+field)
    history_prompts={norm(r['prompt']) for r in historical}
    history_sources={norm(r['source']) for r in historical if r.get('source')}
    new=[r for r in train if not r['id'].startswith('wr4-retain-')]+dev
    if any(norm(r['prompt']) in history_prompts or (r.get('source') and norm(r['source']) in history_sources) for r in new):
        raise ValueError('Writing overlaps a previously observed prompt/source')
    return dict(train=len(train),fresh_dev=len(dev),historical_rows_checked=len(historical),
                exact_prompt_source_overlaps=0,cross_split_structure_overlap=0,
                scope='Observed history only; final holdouts not loaded; no claim of semantic decontamination')


def fingerprint(rows):
    return hashlib.sha256(sft_bytes(rows)).hexdigest()
