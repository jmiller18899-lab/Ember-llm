"""WR3 authored references, not a semantic model-output grader.

Only training and fresh development are generated. No final holdout is loaded.
Each shortening target removes >=20% of source words while preserving its
explicit force, quantities, ownership and conditions. Synthetic/template data;
automated contracts do not substitute for independent semantic review.
"""
from collections import Counter
import hashlib
import json
import random
import re
from string import Formatter

VERSION = 'ember-writing-repair3-data-v1'
SEED = 431

# (source, reference, additional literal meaning anchors)
TRAIN_SHORT = [
 ('I am writing to ask you to please leave the {obj} in the {place} until {day}, and to keep it there unless the {role} approves its removal.',
  'Please leave the {obj} in the {place} until {day}, unless the {role} approves its removal.', ('Please', 'until', 'unless', 'approves')),
 ('The guidance for the {group} is as follows: they should check the {obj} at the {place} before {time}, and this is a recommendation rather than a requirement.',
  'The {group} should, but are not required to, check the {obj} at the {place} before {time}.', ('should', 'not required', 'before')),
 ('There is a mandatory rule for the {group}: they must bring exactly {q} {items} to the {place} by {time}; this quantity must be exact.',
  'The {group} must bring exactly {q} {items} to the {place} by {time}.', ('must', 'exactly', 'by')),
 ('Regarding the {obj} in the {place}, you are not required to move it before {day}. There is no obligation to move it before that day.',
  'You are not required to move the {obj} in the {place} before {day}.', ('not required', 'before')),
 ('Please take note of this prohibition: the {group} must not open the {obj} until {day}. Opening it earlier than that day is prohibited.',
  'The {group} must not open the {obj} until {day}.', ('must not', 'until')),
 ('Permission to collect the {obj} from the {place} is conditional. You may collect it only if the {role} is present, not when that person is absent.',
  'You may collect the {obj} from the {place} only if the {role} is present.', ('may', 'only if', 'present')),
 ('The arrival of the {obj} at the {place} before {time} on {day} is uncertain. It might arrive then, but its arrival then is not guaranteed.',
  'The {obj} might arrive at the {place} before {time} on {day}; this is not guaranteed.', ('might', 'before', 'not guaranteed')),
 ('For the {group}, bringing at least {q} {items} to the {place} is recommended. They should bring that minimum, but it is not mandatory.',
  'The {group} should bring at least {q} {items} to the {place}, but this is not mandatory.', ('should', 'at least', 'not mandatory')),
 ('The limit for the {group} is mandatory: they must bring at most {q} {items} to the {place}. They must not exceed that number.',
  'The {group} must bring at most {q} {items} to the {place}.', ('must', 'at most')),
 ('Here is the request I would like to make: please ask the {group} to meet at the {place} before {time} on {day}. That is my request.',
  'Please ask the {group} to meet at the {place} before {time} on {day}.', ('Please ask', 'before')),
 ('I want to let you know about my plans concerning the {obj}: I can bring it to the {place} on {day}, but I cannot bring it before {time}.',
  'I can bring the {obj} to the {place} on {day}, but not before {time}.', ('can', 'not before')),
 ('The following condition applies to moving the {obj} from cabinet {room}: you must keep it there until {day}, unless the {role} authorizes a move.',
  'You must keep the {obj} in cabinet {room} until {day}, unless the {role} authorizes a move.', ('must', 'until', 'unless')),
 ('Please remember this recommendation: you should not move the {obj} from the {place} before {day}. This is advice, not a prohibition.',
  'You should not move the {obj} from the {place} before {day}; advice, not a prohibition.', ('should not', 'before', 'not a prohibition')),
 ('I have an update about the inspection: the {group} checked exactly {q} {items} at the {place} on {day}. That is the exact number they checked.',
  'The {group} checked exactly {q} {items} at the {place} on {day}.', ('checked', 'exactly')),
 ('In terms of where things are located, the {obj} is in the {place}. It is not in cabinet {room}, despite that being another possible location.',
  'The {obj} is in the {place}, not cabinet {room}.', ('not',)),
 ('I am writing to explain who owns the {obj}: it belongs to {owner}, not {n}. {n} is not the owner of this item.',
  'The {obj} belongs to {owner}, not {n}.', ('belongs to', 'not')),
 ('The order of events is important here: the {group} must inspect the {obj} first and then move it to the {place}. Inspection must come before the move.',
  'The {group} must inspect the {obj} before moving it to the {place}.', ('must', 'before')),
 ('This is an optional activity for the {group}: they may check the {obj} at the {place} on {day}, but they do not have to do so.',
  'The {group} may check the {obj} at the {place} on {day}, but do not have to.', ('may', 'do not have to')),
 ('My request is that you please keep exactly {q} {items} in the {place} until {day}. Please keep that exact quantity there until that day.',
  'Please keep exactly {q} {items} in the {place} until {day}.', ('Please', 'exactly', 'until')),
 ('For clarity, the {group} will check the {obj} at the {place} on {day} if the {role} agrees. Without that agreement, they will not check it.',
  'The {group} will check the {obj} at the {place} on {day} if the {role} agrees; otherwise they will not.', ('will', 'if', 'agrees', 'otherwise', 'not')),
]
DEV_SHORT = [
 ('Checking the {obj} before {time} at the {place} is something the {group} should do. However, they are not obliged to carry out that check.',
  'The {group} should check the {obj} before {time} at the {place}, but are not obliged to.', ('should', 'before', 'not obliged')),
 ('The {obj} could be delivered to the {place} on {day}. Delivery on that day is a possibility, rather than something that is certain to happen.',
  'The {obj} could be delivered to the {place} on {day}, but this is not certain.', ('could', 'not certain')),
 ('As a request to the {group}, I would like to say: please keep the {obj} in cabinet {room} until {day}. That is what I am asking.',
  'Please, {group}, keep the {obj} in cabinet {room} until {day}.', ('Please', 'until')),
 ('The {group} must take exactly {q} {items} to the {place} on {day}. Taking that precise quantity is compulsory; neither a larger nor a smaller quantity will do.',
  'The {group} must take exactly {q} {items} to the {place} on {day}.', ('must', 'exactly')),
 ('Before {day}, moving the {obj} out of the {place} is not required. You do not have an obligation to move it out before that day.',
  'You need not move the {obj} out of the {place} before {day}.', ('need not', 'before')),
 ('Until {time}, you must not open the {obj} in the {place}. This is a ban on opening it earlier than that time, not merely advice.',
  'You must not open the {obj} in the {place} until {time}.', ('must not', 'until')),
 ('For the {group}, checking the {obj} at the {place} on {day} is permitted only if the {role} is present. That presence is necessary for the permission.',
  'The {group} may check the {obj} at the {place} on {day} only if the {role} is present.', ('may', 'only if', 'present')),
 ('A recommendation for the {group} concerns quantities: they should take at least {q} {items} to the {place}. This minimum is recommended, not required.',
  'The {group} should take at least {q} {items} to the {place}; this is not required.', ('should', 'at least', 'not required')),
 ('There is a quantity ceiling for the {group}. They must store at most {q} {items} in cabinet {room}, and storing more than that is not allowed.',
  'The {group} must store at most {q} {items} in cabinet {room}.', ('must', 'at most')),
 ('Ownership should be made clear: {owner} owns the {obj}. {n} does not own it, even though {n} has been mentioned in connection with it.',
  '{owner} owns the {obj}; {n} does not own it.', ('owns', 'does not own')),
 ('About my availability: I can collect the {obj} from the {place} on {day}. I cannot collect it any earlier than {time} on that day.',
  'I can collect the {obj} from the {place} on {day}, but not before {time}.', ('can', 'not before')),
 ('Inspection and movement have a required sequence: the {group} must inspect the {obj} before they take it to the {place}. The inspection must occur first.',
  'The {group} must inspect the {obj} before taking it to the {place}.', ('must', 'before')),
]
TRAIN_RECIPIENT = [
 ('Write a note to {n}: I found the {obj} belonging to {owner} and can return it on {day}.',
  'I found {ownership} and can return it on {day}.'),
 ('Tell {n} that I repaired {owner}\'s {obj}, but I cannot deliver it before {time}. Draft only.',
  'I repaired {ownership}, but I cannot deliver it before {time}.'),
 ('I want a message for {n}. I have {owner}\'s {obj} and might take it to the {place} on {day} if the {role} agrees.',
  'I have {ownership} and might take it to the {place} on {day} if the {role} agrees.'),
 ('Draft a text addressed to {n}, not to anyone else: I left the {obj} owned by {owner} in cabinet {room} until {day}.',
  'I left {ownership} in cabinet {room} until {day}.'),
 ('Write my note to {n} saying I can bring {owner}\'s {obj} to the {place} by {time}, unless the {role} asks me to wait.',
  'I can bring {ownership} to the {place} by {time}, unless the {role} asks me to wait.'),
 ('Please address {n} directly: I borrowed the {obj} from its owner, {owner}, and will return it on {day}. Write the message, do not send it.',
  'I borrowed {ownership} and will return it on {day}.'),
 ('I need a draft for {n}. {owner} owns the {obj}; I checked it at the {place} on {day} and have not moved it.',
  'I checked {ownership} at the {place} on {day} and have not moved it.'),
 ('Compose a message to {n}: the {obj} belongs to {owner}, I have it, and I should return it by {time} on {day}.',
  'I have {ownership} and should return it by {time} on {day}.'),
]
DEV_RECIPIENT = [
 ('My intended reader is {n}. Explain in my voice that I located {owner}\'s {obj} and can bring it back on {day}.',
  'I located {ownership} and can bring it back on {day}.'),
 ('To {n}, from me: I fixed the {obj} that {owner} owns. Delivery before {time} is not possible for me. Turn this into a short note.',
  'I fixed {ownership}, but I cannot deliver it before {time}.'),
 ('Create the wording I would send to {n}. I might collect the {obj} owned by {owner} from the {place} on {day}, provided the {role} agrees.',
  'I might collect {ownership} from the {place} on {day}, provided the {role} agrees.'),
 ('Use {n} as the addressee of a draft. I put {owner}\'s {obj} in cabinet {room}, and it must stay there until {day}.',
  'I put {ownership} in cabinet {room}, and it must stay there until {day}.'),
 ('A note for {n} is needed: I inspected the {obj} belonging to {owner} at the {place} on {day}; I have not transported it elsewhere.',
  'I inspected {ownership} at the {place} on {day} and have not transported it elsewhere.'),
 ('Help me word a message addressing {n}. I have the {obj}, whose owner is {owner}, and I should return it before {time}.',
  'I have {ownership} and should return it before {time}.'),
]
WRAPPERS = ['Shorten this without changing its meaning. Output only the rewrite:\n{s}',
            'Make this more concise, keeping every fact and condition:\n{s}',
            'Rewrite more briefly; do not turn advice into an order:\n{s}',
            'Trim unnecessary wording but preserve the message:\n{s}']


def norm(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def values(split, i):
    if split == 'train':
        names = ['Calista', 'Dorian', 'Elowen', 'Finley', 'Gianna', 'Hector', 'Ivo', 'Juniper']
        owners = ['Kellan', 'Leona', 'Marcel', 'Nerissa', 'Oren', 'Paloma', 'Quinlan', 'Renata']
        objects = ['folding easel', 'brass compass', 'canvas satchel', 'portable loom', 'wooden metronome', 'ceramic lantern', 'linen portfolio', 'copper kettle']
        groups = ['archivists', 'carpenters', 'couriers', 'decorators', 'editors', 'gardeners', 'guides', 'illustrators']
        places = ['north alcove', 'pottery room', 'map gallery', 'west pantry', 'garden shed', 'music studio', 'brick annex', 'weaving room']
    else:
        names = ['Sabina', 'Tobin', 'Ulric', 'Violetta']
        owners = ['Wendell', 'Xenia', 'Yorick', 'Zelda']
        objects = ['glass terrarium', 'velvet case', 'bamboo basket', 'stone figurine']
        groups = ['bookbinders', 'ceramists', 'engravers', 'weavers']
        places = ['east pavilion', 'orchard cabin', 'clock workshop', 'silver vestibule']
    pick = lambda seq: seq[i % len(seq)]
    return {'n': pick(names), 'owner': pick(owners), 'obj': pick(objects), 'group': pick(groups),
            'place': pick(places), 'day': pick(['Tuesday', 'Thursday', 'Saturday', 'Monday', 'Wednesday', 'Friday', 'Sunday']),
            'role': pick(['supervisor', 'caretaker', 'coordinator', 'curator']),
            'time': f'{8+i%4}:{(i*7)%60:02d} AM', 'q': str(13+i*3), 'room': str(210+i),
            'items': pick(['labels', 'ribbons', 'tiles', 'clips'])}


def writing(split):
    if split not in ('train', 'dev'):
        raise ValueError('Only training or fresh development is supported')
    rows = []
    templates = TRAIN_SHORT if split == 'train' else DEV_SHORT
    for t, (source, answer, anchors) in enumerate(templates):
        for j in range(8 if split == 'train' else 4):
            v = values(split, t*8+j)
            fields = list(dict.fromkeys(k for _, k, _, _ in Formatter().parse(answer) if k))
            src = source.format(**v)
            rows.append({'id': f'wr3-{split}-short-{t:02}-{j:02}', 'split': split,
                         'training_allowed': split == 'train', 'family': 'shortening',
                         'prompt': WRAPPERS[(t+j)%4].format(s=src), 'source': src,
                         'answer': answer.format(**v), 'protected': [v[k] for k in fields]+list(anchors),
                         'structure': source, 'semantic_review_required': True})
    frames = TRAIN_RECIPIENT if split == 'train' else DEV_RECIPIENT
    for t, (prompt, answer) in enumerate(frames):
        for j in range(6 if split == 'train' else 4):
            v = values(split, t*6+j+160)
            for relation in ('recipient', 'third_party'):
                owner = v['n'] if relation == 'recipient' else v['owner']
                ownership = 'your '+v['obj'] if relation == 'recipient' else owner+"'s "+v['obj']
                w = dict(v, owner=owner, ownership=ownership)
                fields = list(dict.fromkeys(k for _, k, _, _ in Formatter().parse(answer) if k))
                rows.append({'id': f'wr3-{split}-recipient-{t:02}-{j:02}-{relation}', 'split': split,
                             'training_allowed': split == 'train', 'family': 'recipient',
                             'prompt': prompt.format(**w), 'answer': 'Hi '+v['n']+', '+answer.format(**w),
                             'recipient': v['n'], 'owner': owner, 'object': v['obj'], 'relation': relation,
                             'protected': [v['n']]+[w[k] for k in fields], 'structure': prompt,
                             'semantic_review_required': True})
    for r in rows:
        validate_reference(r)
    return rows


def validate_reference(row):
    """Bounded authored-reference checks, never use to award model passes."""
    a = row['answer']
    if not all(f.casefold() in a.casefold() for f in row['protected']):
        raise ValueError('Protected fact or force was dropped')
    if row['family'] == 'shortening':
        s = row['source']
        if len(a.split()) > .8*len(s.split()) or len(a) >= len(s):
            raise ValueError('Reference must substantively shorten the source: '+row['id'])
        if re.findall(r'\d+', s) != re.findall(r'\d+', a):
            raise ValueError('Numeric information changed')
    elif not a.startswith('Hi '+row['recipient']+', '):
        raise ValueError('Wrong addressee')


def build_train_rows(retention):
    expected = {'arithmetic':64, 'extraction':64, 'grounding':48, 'clarification':48, 'direct':32}
    if len(retention) != 256 or any(r.get('split') != 'train' or r.get('training_allowed') is not True for r in retention):
        raise ValueError('Require exactly the 256 pinned training retention rows')
    if Counter(r['family'] for r in retention) != expected:
        raise ValueError('Retention composition changed')
    rows = writing('train')
    for r in retention:
        rows.append({'id': 'wr3-retain-'+r['id'], 'split': 'train', 'training_allowed': True,
                     'family': r['family'], 'prompt': r['prompt'], 'answer': r['answer']})
    random.Random(SEED+3).shuffle(rows)
    sft_bytes(rows)
    return rows


def sft_bytes(rows):
    seen_ids, seen_prompts = set(), set()
    for r in rows:
        if r.get('split') != 'train' or r.get('training_allowed') is not True:
            raise ValueError('Evaluation rows cannot enter the SFT export')
        if not all(isinstance(r.get(k), str) and r[k].strip() for k in ('id', 'prompt', 'answer')):
            raise ValueError('Invalid SFT fields')
        if r['id'] in seen_ids or norm(r['prompt']) in seen_prompts:
            raise ValueError('Duplicate SFT row')
        seen_ids.add(r['id']); seen_prompts.add(norm(r['prompt']))
    return ''.join(json.dumps({k:r[k] for k in ('id','prompt','answer')}, sort_keys=True, ensure_ascii=False)+'\n' for r in rows).encode()


def audit(train, dev, historical):
    for field in ('id', 'prompt', 'source', 'structure'):
        a = {norm(r[field]) for r in train if r.get(field)}
        b = {norm(r[field]) for r in dev if r.get(field)}
        if a & b:
            raise ValueError('Cross-split overlap: '+field)
    history_prompts = {norm(r['prompt']) for r in historical}
    history_sources = {norm(r['source']) for r in historical if r.get('source')}
    new_rows = [r for r in train if not r['id'].startswith('wr3-retain-')] + dev
    if any(norm(r['prompt']) in history_prompts or (r.get('source') and norm(r['source']) in history_sources) for r in new_rows):
        raise ValueError('New writing overlaps an observed historical prompt/source')
    return {'train':len(train), 'fresh_dev':len(dev), 'historical_rows_checked':len(historical),
            'exact_prompt_source_overlaps':0, 'cross_split_structure_overlap':0,
            'scope':'Pinned observed history only; final holdouts not generated; not semantic decontamination'}


def fingerprint(rows):
    return hashlib.sha256(sft_bytes(rows)).hexdigest()
