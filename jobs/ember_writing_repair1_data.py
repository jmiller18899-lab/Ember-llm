"""Writing Repair 1: reproducible synthetic data, CPU-only; never starts training.

Only train/train.jsonl may be passed to a trainer. Evaluation sets have different
source/prompt structures and entity pools, not random rows of training templates.
Author-reviewed references are checked against explicit facts and task logic.
These checks are data audits, not a replacement for the pinned baseline grader.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

VERSION = 'ember-writing-repair1-data-v1'
SEED = 20260924
BASELINE_COMMIT = 'e6c9d37120802349190d9d66eaf35e7301a64c2e'
GRADER_COMMIT = '25924014c0e5d5a580a296b2841a1e6f6cbe3bb4'
EXPECTED_COUNTS = {'train':512, 'dev':64, 'writing_holdout':96, 'model_holdout':96}
EXPECTED_TRAIN = {'shortening':160,'recipient':96,'arithmetic':64,'extraction':64,
                  'grounding':48,'clarification':48,'direct':32}

# S(source, reference, force, protected slots, ordered subject/action anchors).
def S(src, ans, force, protect, order):
    return {'source':src,'reference':ans,'force':force,'protect':protect,'order':order}

SHORT = {
 'train': [
 S('All {group} are kindly asked to deliver exactly {q} {items} to the {place} by {time}.',
   'All {group}, please deliver exactly {q} {items} to the {place} by {time}.','request',
   ['all','{group}','exactly {q} {items}','{place}','by {time}'],['{group}','deliver','{items}']),
 S('All {group} are required to deliver exactly {q} {items} to the {place} by {time}.',
   'All {group} must deliver exactly {q} {items} to the {place} by {time}.','requirement',
   ['all','{group}','exactly {q} {items}','{place}','by {time}'],['{group}','deliver','{items}']),
 S('Please remember that the {group} must keep the {obj} inside cabinet {room} until {day}.',
   'The {group} must keep the {obj} inside cabinet {room} until {day}.','requirement',
   ['{group}','{obj}','inside cabinet {room}','until {day}'],['{group}','keep','{obj}']),
 S('Please remember that the {group} are not required to keep the {obj} inside cabinet {room} until {day}.',
   'The {group} are not required to keep the {obj} inside cabinet {room} until {day}.','no_obligation',
   ['{group}','{obj}','inside cabinet {room}','until {day}'],['{group}','keep','{obj}']),
 S('Please note that the {group} should use the {obj} at the {place} after {time}.',
   'The {group} should use the {obj} at the {place} after {time}.','recommendation',
   ['{group}','{obj}','{place}','after {time}'],['{group}','use','{obj}']),
 S('Please note that the {group} are allowed to use the {obj} at the {place} after {time}.',
   'The {group} are allowed to use the {obj} at the {place} after {time}.','permission',
   ['{group}','{obj}','{place}','after {time}'],['{group}','use','{obj}']),
 S('Please note that the {obj} might arrive at the {place} before {time} on {day}.',
   'The {obj} might arrive at the {place} before {time} on {day}.','uncertainty',
   ['{obj}','{place}','before {time}','{day}'],['{obj}','arrive','{place}']),
 S('Please note that the {obj} will arrive at the {place} before {time} on {day}.',
   'The {obj} will arrive at the {place} before {time} on {day}.','certainty',
   ['{obj}','{place}','before {time}','{day}'],['{obj}','arrive','{place}']),
 S('It is important to remember that the {group} must not enter the {place} before {time}.',
   'The {group} must not enter the {place} before {time}.','prohibition',
   ['{group}','{place}','before {time}'],['{group}','enter','{place}']),
 S('It is important to remember that the {group} must enter the {place} before {time}.',
   'The {group} must enter the {place} before {time}.','requirement',
   ['{group}','{place}','before {time}'],['{group}','enter','{place}']),
 S('Please note that the {group} may collect the {obj} at the {place} only if the {role} is present.',
   'The {group} may collect the {obj} at the {place} only if the {role} is present.','may',
   ['{group}','{obj}','{place}','only if the {role} is present'],['{group}','collect','{obj}']),
 S('Please note that the {group} may collect the {obj} at the {place} unless the {role} is present.',
   'The {group} may collect the {obj} at the {place} unless the {role} is present.','may',
   ['{group}','{obj}','{place}','unless the {role} is present'],['{group}','collect','{obj}']),
 S('The {group} must bring a total of at least {q} {items} to the {place} by {time}.',
   'The {group} must bring at least {q} {items} to the {place} by {time}.','requirement',
   ['{group}','at least {q} {items}','{place}','by {time}'],['{group}','bring','{items}']),
 S('The {group} must bring a total of at most {q} {items} to the {place} by {time}.',
   'The {group} must bring at most {q} {items} to the {place} by {time}.','requirement',
   ['{group}','at most {q} {items}','{place}','by {time}'],['{group}','bring','{items}']),
 S('Please ensure that all {group} meet at the {place} before {time}.',
   'Please ensure all {group} meet at the {place} before {time}.','request',
   ['all {group}','{place}','before {time}'],['{group}','meet','{place}']),
 S('Please ensure that the {role} meets all {group} at the {place} before {time}.',
   'Please ensure the {role} meets all {group} at the {place} before {time}.','request',
   ['{role}','all {group}','{place}','before {time}'],['{role}','meets','{group}']),
 S('Please remember that all {group} should collect exactly {q} {items} from the {place} on {day}.',
   'All {group} should collect exactly {q} {items} from the {place} on {day}.','recommendation',
   ['all {group}','exactly {q} {items}','{place}','{day}'],['{group}','collect','{items}']),
 S('Please remember that some {group} should collect exactly {q} {items} from the {place} on {day}.',
   'Some {group} should collect exactly {q} {items} from the {place} on {day}.','recommendation',
   ['some {group}','exactly {q} {items}','{place}','{day}'],['{group}','collect','{items}']),
 S('Please note that the measured change for unit {room} on {day} was -{q}%.',
   'The measured change for unit {room} on {day} was -{q}%.','statement',
   ['unit {room}','{day}','-{q}%'],['measured change','unit {room}','-{q}%']),
 S('Please note that the measured change for unit {room} on {day} was +{q}%.',
   'The measured change for unit {room} on {day} was +{q}%.','statement',
   ['unit {room}','{day}','+{q}%'],['measured change','unit {room}','+{q}%']),
 ],
 'dev': [
 S('It would be appreciated if all {group} would bring the {obj} to the {place} by {time}.',
   'All {group}, please bring the {obj} to the {place} by {time}.','request',
   ['all {group}','{obj}','{place}','by {time}'],['{group}','bring','{obj}']),
 S('For the {group}, carrying the {obj} at the {place} is mandatory.',
   'The {group} must carry the {obj} at the {place}.','requirement',
   ['{group}','{obj}','{place}'],['{group}','carry','{obj}']),
 S('There is no requirement for the {group} to check cabinet {room} before {time}.',
   'The {group} are not required to check cabinet {room} before {time}.','no_obligation',
   ['{group}','cabinet {room}','before {time}'],['{group}','check','cabinet']),
 S('The {group} are forbidden from opening cabinet {room} until {day}.',
   'The {group} must not open cabinet {room} until {day}.','prohibition',
   ['{group}','cabinet {room}','until {day}'],['{group}','open','cabinet']),
 S('The {obj} could reach the {place} on {day}; this is only a possibility.',
   'The {obj} could reach the {place} on {day}.','uncertainty',
   ['{obj}','{place}','{day}'],['{obj}','reach','{place}']),
 S('The {group} may collect the {obj} at the {place}, but only if the {role} is present.',
   'The {group} may collect the {obj} at the {place} only if the {role} is present.','may',
   ['{group}','{obj}','{place}','only if the {role} is present'],['{group}','collect','{obj}']),
 S('The {group} should bring exactly {q} {items}, with the {place} as the destination and {time} as the deadline.',
   'The {group} should bring exactly {q} {items} to the {place} by {time}.','recommendation',
   ['{group}','exactly {q} {items}','{place}','{time}'],['{group}','bring','{items}']),
 S('Every member of the {group} is asked to bring the {obj} to the {place} on {day}.',
   'Each {group} member is asked to bring the {obj} to the {place} on {day}.','request',
   ['{group}','{obj}','{place}','{day}'],['{group}','member','bring','{obj}']),
 ],
 'writing_holdout': [
 S('A polite request for all {group}: please take the {obj} to the {place} before {time}.',
   'All {group}, please take the {obj} to the {place} before {time}.','request',
   ['all {group}','{obj}','{place}','before {time}'],['{group}','take','{obj}']),
 S('The {group} are obligated to deliver the {obj}. The destination is the {place}, and the deadline is {time}.',
   'The {group} must deliver the {obj} to the {place} by {time}.','requirement',
   ['{group}','{obj}','{place}','{time}'],['{group}','deliver','{obj}']),
 S('For the {group}, it is not mandatory to place the {obj} inside cabinet {room}.',
   'The {group} are not required to place the {obj} inside cabinet {room}.','no_obligation',
   ['{group}','{obj}','inside cabinet {room}'],['{group}','place','{obj}']),
 S('No member of the {group} is permitted to enter the {place} before {time}.',
   'No {group} member may enter the {place} before {time}.','prohibition',
   ['{group}','member','{place}','before {time}'],['{group}','member','enter','{place}']),
 S('One possibility is that the {obj} will reach the {place} on {day}; it is not certain.',
   'The {obj} might reach the {place} on {day}.','uncertainty',
   ['{obj}','{place}','{day}'],['{obj}','reach','{place}']),
 S('The recommendation for all {group} is to wait at the {place} until {time}.',
   'All {group} should wait at the {place} until {time}.','recommendation',
   ['all {group}','{place}','until {time}'],['{group}','wait','{place}']),
 S('Please be aware that only while the {role} is present are the {group} permitted to handle the {obj} at the {place}.',
   'The {group} are permitted to handle the {obj} at the {place} only while the {role} is present.','permission',
   ['{group}','{obj}','{place}','only while the {role} is present'],['{group}','handle','{obj}']),
 S('Please be aware that, except when the {role} is absent, the {group} must leave the {obj} at the {place} before {time}.',
   'The {group} must leave the {obj} at the {place} before {time}, except when the {role} is absent.','requirement',
   ['{group}','{obj}','{place}','before {time}','except when the {role} is absent'],['{group}','leave','{obj}']),
 S('The {role} is the person who must meet all {group} at the {place} on {day}.',
   'The {role} must meet all {group} at the {place} on {day}.','requirement',
   ['{role}','all {group}','{place}','{day}'],['{role}','meet','{group}']),
 S('It is required that no fewer than {q} {items} be delivered to the {place} by {time}.',
   'At least {q} {items} must be delivered to the {place} by {time}.','requirement',
   ['{q} {items}','{place}','by {time}'],['{items}','delivered','{place}']),
 S('The rule requires a maximum of {q} {items} to be stored inside cabinet {room}.',
   'At most {q} {items} must be stored inside cabinet {room}.','requirement',
   ['{q} {items}','inside cabinet {room}'],['{items}','stored','cabinet']),
 S('For unit {room}, the change measured on {day} was negative: specifically, -{q}%.',
   'Unit {room} had a measured change of -{q}% on {day}.','statement',
   ['unit {room}','{day}','-{q}%'],['unit {room}','measured change','-{q}%']),
 ]}

POOLS = {
 'train': {
  'names':'Anika Bruno Celia Diego Esme Farid Gita Hugo Ines Jonas Keira Luca Mira Nolan Opal Pavel Quinn Reva Silas Talia Uriel Willa Xander Yasmin'.split(),
  'objects':'telescope clarinet toolbox sundial raincoat sketchbook rucksack flask calculator recorder protractor binoculars'.split(),
  'groups':['archivists','surveyors','gardeners','curators','technicians','volunteers','organizers','musicians'],
  'places':['cedar pavilion','brick annex','map archive','canal depot','pottery studio','orchard cabin','south kiosk','copper atrium'],
  'roles':['steward','coordinator','conservator','caretaker','facilitator','instructor','registrar','curator'],
  'items':['sealed envelopes','ceramic cups','sample tubes','linen cloths','brass keys','paper maps','wooden blocks','glass beads']},
 'dev': {
  'names':'Amara Benoit Cassia Denzel Eira Florian Hadley Isolde Jovan Kiran Maelle Osric'.split(),
  'objects':['tripometer','ruler','metronome','harmonica','magnifier','ukulele','satchel','easel'],
  'groups':['editors','inspectors','sculptors','tutors','stewards','decorators','marshals','porters'],
  'places':['violet conservatory','granite landing','reed shelter','birch porch','marble foyer','rose pergola','stone alcove','silver rotunda'],
  'roles':['foreman','editor','guide','sculptor','tutor','inspector','marshal','porter'],
  'items':['cork stoppers','wool swatches','clay tiles','iron hooks','satin ribbons','plastic tags','felt pads','rubber rings']},
 'writing_holdout': {
  'names':'Aurelia Belen Cosima Darian Eamon Fenna Idris Jovita Leander Mireille Neriah Saffron'.split(),
  'objects':['barometer','ocarina','mandolin','sextant','trowel','abacus','tambourine','thimble'],
  'groups':['weavers','ceramists','librarians','cartographers','restorers','ushers','docents','cataloguers'],
  'places':['amber mezzanine','quartz belvedere','willow gallery','opal vestibule','fern arcade','basalt portico','ivy lodge','pearl cloister'],
  'roles':['weaver','librarian','docent','restorer','usher','cartographer','ceramist','cataloguer'],
  'items':['wax seals','copper pins','silk patches','slate plaques','leather loops','bamboo rods','tin tokens','velvet strips']},
 'model_holdout': {
  'names':'Alden Brisa Corin Delphine Elian Faye Galen Hester Ivo Jocelyn Maura Tobin'.split(),
  'objects':['astrolabe','planisphere','caliper','level','spatula','colander','sieve','ladle'],
  'groups':['bakers','cyclists','rowers','potters','tailors','glaziers','joiners','bookbinders'],
  'places':['ash lodge','cobalt porch','flint shed','elm hall','bronze bay','hazel room','moss hut','oak yard'],
  'roles':['baker','cyclist','rower','potter','tailor','glazier','joiner','bookbinder'],
  'items':['stone counters','blue counters','red counters','yellow counters','green counters','white counters','black counters','orange counters']}}
DAYS=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

RECIPIENT_FRAMES = {
 'train': [
 ('recipient','Draft a message to {n}: I found {n}\'s {obj} and can return it on {day}.',
  'Hi {n}, I found your {obj} and can return it on {day}.','I found','can return','{day}'),
 ('third_party','Draft a message to {n}: I found {owner}\'s {obj} and can return it to {owner} on {day}.',
  'Hi {n}, I found {owner}\'s {obj} and can return it to {owner} on {day}.','I found','can return','{day}'),
 ('recipient','Text {n} that I have {pos} {obj} and will leave it at the {place} after {time}. The {obj} belongs to {n}.',
  'Hi {n}, I have your {obj} and will leave it at the {place} after {time}.','I have','will leave','{place}|after {time}'),
 ('third_party','Text {n} that I have {owner}\'s {obj} and will leave it at the {place} after {time}.',
  'Hi {n}, I have {owner}\'s {obj} and will leave it at the {place} after {time}.','I have','will leave','{place}|after {time}'),
 ('recipient','Write a note for {n} saying I cannot return {n}\'s {obj} until {day}.',
  'Hi {n}, I cannot return your {obj} until {day}.','I cannot return','','until {day}'),
 ('third_party','Write a note for {n} saying I cannot return {owner}\'s {obj} until {day}.',
  'Hi {n}, I cannot return {owner}\'s {obj} until {day}.','I cannot return','','until {day}'),
 ('recipient','Tell {n} that I might bring {pos} {obj} to the {place} on {day}, but only if the {role} is present. The {obj} is {n}\'s.',
  'Hi {n}, I might bring your {obj} to the {place} on {day}, but only if the {role} is present.','I might bring','','{place}|{day}|only if the {role} is present'),
 ('third_party','Tell {n} that I might bring {owner}\'s {obj} to the {place} on {day}, but only if the {role} is present.',
  'Hi {n}, I might bring {owner}\'s {obj} to the {place} on {day}, but only if the {role} is present.','I might bring','','{place}|{day}|only if the {role} is present')],
 'dev': [
 ('recipient','Prepare a text addressed to {n}. Their {obj} is with me. Say that I can bring it on {day}.',
  'Hi {n}, I have your {obj} and can bring it on {day}.','I have','can bring','{day}'),
 ('third_party','Prepare a text addressed to {n}. {owner}\'s {obj} is with me. Say that I can bring it on {day}.',
  'Hi {n}, I have {owner}\'s {obj} and can bring it on {day}.','I have','can bring','{day}'),
 ('recipient','For a note to {n}, turn this into direct speech: I repaired the {obj} that belongs to {n}; I will return it before {time}.',
  'Hi {n}, I repaired your {obj}; I will return it before {time}.','I repaired','I will return','before {time}'),
 ('third_party','For a note to {n}, turn this into direct speech: I repaired the {obj} that belongs to {owner}; I will return it before {time}.',
  'Hi {n}, I repaired {owner}\'s {obj}; I will return it before {time}.','I repaired','I will return','before {time}'),
 ('recipient','Address {n} in a draft. Explain that I am keeping {n}\'s {obj} at the {place} until {day}.',
  'Hi {n}, I am keeping your {obj} at the {place} until {day}.','I am keeping','','{place}|until {day}'),
 ('third_party','Address {n} in a draft. Explain that I am keeping {owner}\'s {obj} at the {place} until {day}.',
  'Hi {n}, I am keeping {owner}\'s {obj} at the {place} until {day}.','I am keeping','','{place}|until {day}'),
 ('recipient','Compose a reply to {n}: I may be able to bring back their {obj} on {day}; do not promise it. The {obj} belongs to {n}.',
  'Hi {n}, I may be able to bring back your {obj} on {day}.','I may be able to bring back','','{day}'),
 ('third_party','Compose a reply to {n}: I may be able to bring back {owner}\'s {obj} on {day}; do not promise it.',
  'Hi {n}, I may be able to bring back {owner}\'s {obj} on {day}.','I may be able to bring back','','{day}')],
 'writing_holdout': [
 ('recipient','Recipient: {n}. Owner of the {obj}: {n}. My update: I located it and can bring it to the {place} on {day}. Write the message itself.',
  'Hi {n}, I located your {obj} and can bring it to the {place} on {day}.','I located','can bring','{place}|{day}'),
 ('third_party','Recipient: {n}. Owner of the {obj}: {owner}. My update: I located it and can bring it to the {place} on {day}. Write the message itself.',
  'Hi {n}, I located {owner}\'s {obj} and can bring it to the {place} on {day}.','I located','can bring','{place}|{day}'),
 ('recipient','I am replying to {n}, who owns the {obj}. My reply needs to say I cannot drop it at the {place} before {time}. Give only my reply.',
  'Hi {n}, I cannot drop your {obj} at the {place} before {time}.','I cannot drop','','{place}|before {time}'),
 ('third_party','I am replying to {n} about a {obj} owned by {owner}. My reply needs to say I cannot drop it at the {place} before {time}. Give only my reply.',
  'Hi {n}, I cannot drop {owner}\'s {obj} at the {place} before {time}.','I cannot drop','','{place}|before {time}'),
 ('recipient','Put this into a message from me to {n}: the {obj} is theirs; I will store it inside cabinet {room} until {day}.',
  'Hi {n}, I will store your {obj} inside cabinet {room} until {day}.','I will store','','inside cabinet {room}|until {day}'),
 ('third_party','Put this into a message from me to {n}: the {obj} belongs to {owner}; I will store it inside cabinet {room} until {day}.',
  'Hi {n}, I will store {owner}\'s {obj} inside cabinet {room} until {day}.','I will store','','inside cabinet {room}|until {day}'),
 ('recipient','Write the words I should send to {n}, not a claim of sending. I hope to return {n}\'s {obj} on {day}, provided that the {role} approves.',
  'Hi {n}, I hope to return your {obj} on {day}, provided that the {role} approves.','I hope to return','','{day}|provided that the {role} approves'),
 ('third_party','Write the words I should send to {n}, not a claim of sending. I hope to return {owner}\'s {obj} on {day}, provided that the {role} approves.',
  'Hi {n}, I hope to return {owner}\'s {obj} on {day}, provided that the {role} approves.','I hope to return','','{day}|provided that the {role} approves')]
}

def norm(s):
    return re.sub(r'\s+',' ',s.casefold().replace('’',"'")).strip()

def sha256(b): return hashlib.sha256(b).hexdigest()

def present(phrase, text):
    return bool(re.search(r'(?<!\w)'+re.escape(norm(phrase))+r'(?!\w)',norm(text)))

def values(split, i):
    p=POOLS[split]; offset=list(POOLS).index(split)
    v={k:p[k][i%len(p[k])] for k in ['objects','groups','places','roles','items']}
    return {'obj':p['objects'][(i+i//len(p['names']))%len(p['objects'])],'group':v['groups'],'place':v['places'],'role':v['roles'],
        'items':v['items'],'n':p['names'][i%len(p['names'])],
        'owner':p['names'][(i+5)%len(p['names'])], 'pos':['her','his','their'][i%3],
        'q':str(3+i%17+offset*19),'room':str(501+offset*100+i),
        'time':f'{1+(i*3+offset)%11}:{["05","25","40","55"][i%4]} '+('AM' if i%2 else 'PM'),
        'day':DAYS[(i*3+offset)%7]}

def row(split, family, template, i, prompt, answer, rubric, **extra):
    return {'id':f'wr1-{split}-{family}-{template:02}-{i:03}','split':split,
        'training_allowed':split=='train','family':family,
        'template_id':f'{split}/{family}/{template:02}',
        'prompt':prompt,'answer':answer,'rubric':rubric,
        'provenance':{'type':'authored_template_with_deterministic_slots','version':VERSION},**extra}

def writing(split):
    rows=[]; per={'train':8,'dev':4,'writing_holdout':4}[split]
    wrappers={'train':['Shorten this while preserving meaning: {s}','Make this shorter without losing facts: "{s}"',
        'Condense this without changing who does what: {s}','Rewrite this more briefly, keeping every condition: {s}'],
      'dev':['Please tighten the following sentence; preserve its meaning.\n{s}'],
      'writing_holdout':['Produce a shorter version of the passage below. Keep its meaning unchanged.\n{s}']}
    for t,spec in enumerate(SHORT[split]):
        for j in range(per):
            # Paired train contrasts share slots and stay in the same split.
            v=values(split,(t//2)*per+j if split=='train' else t*per+j)
            src=spec['source'].format(**v); ans=spec['reference'].format(**v)
            rubric={'task':'faithful_shortening','force':spec['force'],
                'protected':[p.format(**v) for p in spec['protect']],
                'ordered_anchors':[p.format(**v) for p in spec['order']],
                'manual_checks':['preserve subject and action roles','preserve scope, force and all facts',
                    'do not invent actions or completion','accept equivalent natural phrasing after review']}
            if split=='dev' and t==7: rubric['protected'].append('each')
            if split=='writing_holdout' and t in (9,10): rubric['protected'].append('at least' if t==9 else 'at most')
            p=wrappers[split][(t+j)%len(wrappers[split])].format(s=src)
            rows.append(row(split,'shortening',t,j,p,ans,rubric,source=src,kind='shortening',
                scoring='rubric',contrast_group=f'{split}/short/{t//2}/{j}',
                structural_signature=spec['source']+' -> '+spec['reference']))
    reps={'train':12,'dev':4,'writing_holdout':6}[split]
    for t,(relation,pt,at,action,commitment,facts) in enumerate(RECIPIENT_FRAMES[split]):
        for j in range(reps):
            v=values(split,(t//2)*reps+j)
            ownership='your '+v['obj'] if relation=='recipient' else v['owner']+"'s "+v['obj']
            rubric={'task':'recipient_perspective','recipient':v['n'],
                'owner':v['n'] if relation=='recipient' else v['owner'],'owner_relation':relation,
                'object':v['obj'],'ownership_phrase':ownership,'sender_action':action,
                'protected':[v['n'],ownership,action]+([commitment] if commitment else [])+
                    [x.format(**v) for x in facts.split('|')],
                'manual_checks':['sender remains I, recipient becomes you only when appropriate',
                    'third-party owner remains explicitly identified','preserve commitment and conditions',
                    'draft only; never claim to have sent it']}
            rows.append(row(split,'recipient',t,j,pt.format(**v),at.format(**v),rubric,
                scoring='rubric',contrast_group=f'{split}/recipient/{t//2}/{j}',
                structural_signature=pt+' -> '+at))
    return rows

ARITH_TRAIN = [
 ('add_sub','A supply box starts with {a} pieces. Another {b} are added and {c} are removed. How many remain? Number only.'),
 ('mult_add','There are {a} sealed packets containing {b} stamps each and {c} loose stamps. Total stamps? Number only.'),
 ('mult_sub','A rack holds {a} rows of {b} pots. After {c} pots are removed, how many are left? Number only.'),
 ('divide_add','{total} ribbons are shared equally among {a} teams. Each team then gets {c} additional ribbons. Per team? Number only.'),
 ('double_sum','One display contains {a} prints. A second contains {b} times as many. How many prints together? Number only.'),
 ('perimeter','A rectangle measures {a} units by {b} units. Find its perimeter. Number only.'),
 ('time','A timed session begins at {clock} and runs for {duration} minutes. Give its finish time as HH:MM in 24-hour notation.'),
 ('signed','A counter reads {negative}. It rises by {b}, then falls by {c}. What does it read now? Number only.')]
ARITH_HOLD = [
 ('add_sub','At closing, an inventory is calculated from an opening count of {a}, an incoming shipment of {b}, and {c} items issued. Report the closing count as one number.'),
 ('mult_add','Count the tokens: {a} tubes, {b} tokens per tube, and a separate pile of {c}. Respond with the total and nothing else.'),
 ('mult_sub','Before distribution, {a} trays each held {b} pastries; {c} pastries have since been handed out. State the remaining count only.'),
 ('divide_add','Each of {a} groups receives an equal share of {total} tiles plus a bonus of {c} tiles. Give one group\'s final allocation, number only.'),
 ('double_sum','The first batch has {a} units and the other batch is {b} times that size. Combine the batches. Return a bare number.'),
 ('perimeter','How far is one complete circuit around a rectangular plot with side lengths {a} and {b}? Output a number only.'),
 ('time','Start={clock}; elapsed={duration} minutes. Calculate the end clock reading, including midnight rollover. Use HH:MM (24-hour), with no explanation.'),
 ('signed','From a starting value of {negative}, apply an increase of {b} followed by a decrease of {c}. Final signed value only.')]

def arithmetic_expected(r):
    op=r['operation']; a,b,c=r['a'],r['b'],r['c']
    if op=='add_sub': return str(a+b-c)
    if op=='mult_add': return str(a*b+c)
    if op=='mult_sub': return str(a*b-c)
    if op=='divide_add': return str(r['total']//a+c)
    if op=='double_sum': return str(a*(1+b))
    if op=='perimeter': return str(2*(a+b))
    if op=='signed': return str(-a+b-c)
    mins=(r['start_minutes']+r['duration'])%1440
    return f'{mins//60:02d}:{mins%60:02d}'

def arithmetic_rows(split, n):
    frames=ARITH_TRAIN if split=='train' else ARITH_HOLD; out=[]
    rng=random.Random(SEED+(0 if split=='train' else 1000)); seen=set()
    for i in range(n):
        t=i%8; op,p=frames[t]
        while True:
            a=rng.randint(11,37) if split=='train' else rng.randint(41,67)
            b=rng.randint(3,9); c=rng.randint(1,10)
            key=(op,a,b,c)
            if key not in seen and c!=b: break
        seen.add(key)
        minutes=(23*60+45+i*7)%1440; duration=35+i*11
        v={'a':a,'b':b,'c':c,'total':a*b,'negative':-a,
           'clock':f'{minutes//60:02d}:{minutes%60:02d}','duration':duration}
        r={'task':'exact_arithmetic','operation':op,'a':a,'b':b,'c':c,'total':a*b,
           'start_minutes':minutes,'duration':duration}
        out.append(row(split,'arithmetic',t,i,p.format(**v),arithmetic_expected(r),r,
            scoring='exact',evaluation_mode='model_only',structural_signature=p))
    return out

def retention(split):
    out=arithmetic_rows(split,64 if split=='train' else 32)
    count=64 if split=='train' else 24
    for i in range(count):
        v=values(split,i); code=f'{"TR" if split=="train" else "MH"}-{1100+i}'; t=i%4
        record={'label':v['n'],'asset':v['obj'],'cabinet':v['room'],'code':code}
        field=['code','asset','label','cabinet'][i%4]
        if split=='train':
            prefix=['Read this JSON record and return only its {field}: {record}',
                    'Extract {field}; output only that value. Record: {record}',
                    'From the supplied record, copy the {field} value exactly, without a label: {record}',
                    'Give just the stored {field}, not any other field: {record}'][t]
            body=json.dumps(record,ensure_ascii=False)
        else:
            prefix=['Which value appears under {field}? Return it alone.\n{record}',
                    'A tab-separated record follows. Report its {field} field verbatim.\n{record}',
                    'Read the key/value card and emit only {field}.\n{record}',
                    'Ignore every field except {field}; copy its value.\n{record}'][t]
            body='\n'.join(k+'\t'+val for k,val in record.items())
        out.append(row(split,'extraction',t,i,prefix.format(field=field,record=body),record[field],
            {'task':'exact_extraction','field':field,'record':record},scoring='exact',
            evaluation_mode='model_only',structural_signature=prefix))
    count=48 if split=='train' else 16
    for i in range(count):
        v=values(split,i); t=i%4
        if split=='train':
            if t==0:
                p=f'The {v["obj"]} left the {v["place"]} on {v["day"]}. Its travel duration and arrival record are absent. Can its arrival time be determined?'
            elif t==1:
                p=f'The record for cabinet {v["room"]} says the {v["obj"]} was received on {v["day"]} at {v["time"]}. When was it received? Use only the record.'
            elif t==2:
                p=f'No messaging tool is connected. Have you already told {v["n"]} to collect the {v["obj"]}?'
            else:
                p=f'{v["n"]} owns the {v["obj"]}. Pickup is at the {v["place"]} on {v["day"]}. Who owns what, and where and when is pickup?'
        else:
            if t==0:
                p=f'Dispatch card: object={v["obj"]}; origin={v["place"]}; departure_day={v["day"]}. The card has no duration or arrival entry. What arrival time can you infer?'
            elif t==1:
                p=f'Use this receipt, not an estimate: {v["obj"]}, received {v["day"]}, {v["time"]}, cabinet {v["room"]}. State the recorded receipt time and day.'
            elif t==2:
                p=f'I need a factual status: was the collection message about the {v["obj"]} sent to {v["n"]}? In this conversation you have no send capability and no send confirmation.'
            else:
                p=f'Collection note | owner: {v["n"]} | article: {v["obj"]} | venue: {v["place"]} | date: {v["day"]}. Restate the collection details in one sentence.'
        if t==0: a='There is not enough information to determine the arrival time.'; facts=['not enough information','arrival time']
        elif t==1: a=f'It was received on {v["day"]} at {v["time"]}.'; facts=[v['day'],v['time']]
        elif t==2: a=f'No. I have not sent a message to {v["n"]}; no messaging tool is available here.'; facts=['not sent',v['n'],'no messaging tool']
        else:
            a=f'{v["n"]} owns the {v["obj"]}; pickup is at the {v["place"]} on {v["day"]}.'
            facts=[v['n'],v['obj'],v['place'],v['day']]
        out.append(row(split,'grounding',t,i,p,a,{'task':'grounded_response','subtype':t,'protected':facts},
            scoring='rubric',evaluation_mode='model_only',structural_signature=f'{split}/grounding-source-structure-{t}'))
    count=48 if split=='train' else 16
    styles=['brief','clear','polite','concise','direct','simple']
    for i in range(count):
        v=values(split,i); t=i%4; style=styles[(i//4)%6]
        if t%2==0:
            if split=='train':
                p=f'Please {"summarize" if t==0 else "proofread"} my {v["obj"]} description in a {style} style. I have not included the description.'
            else:
                p=f'Task: {"condense" if t==0 else "correct the wording of"} a description of the {v["obj"]}. No description text accompanies this task; use a {style} response.'
            a=f'Please paste the {v["obj"]} description you want me to {"summarize" if t==0 else "proofread"}.'
            rubric={'task':'missing_text','object':v['obj'],'protected':['please paste',v['obj'],'description']}
        else:
            text=f'The {v["obj"]} is at the {v["place"]}. It is ready on {v["day"]}.'
            if split=='train': p=f'Combine these supplied facts into one {style} sentence: {text}'
            else: p=f'Here is the description to work from, so do not ask for it again. Express both statements as one {style} sentence.\n{text}'
            a=f'The {v["obj"]} is at the {v["place"]} and is ready on {v["day"]}.'
            rubric={'task':'supplied_text','protected':[v['obj'],v['place'],v['day']],'source':text}
        out.append(row(split,'clarification',t,i,p,a,rubric,scoring='rubric',evaluation_mode='model_only',
                       structural_signature=f'{split}/clarification-source-structure-{t}'))
    count=32 if split=='train' else 8
    for i in range(count):
        v=values(split,i); t=i%4
        if t==0:
            literal=f'{v["n"]}|{v["room"]}|{v["obj"]}'
            p=(f'Copy the following string exactly, with nothing added: {literal}' if split=='train' else
               f'Literal reproduction task. Output the characters between the markers, and not the markers: <start>{literal}<end>')
            a=literal; rubric={'task':'copy','literal':literal}
        elif t==1:
            value=(21+i if split=='train' else 20+i*5); threshold=35
            p=(f'Use only this rule: a reading of {threshold} or more is HIGH; otherwise LOW. Reading: {value}. Output only the label.' if split=='train' else
               f'Classify the recorded value {value}. The two categories are HIGH (at least {threshold}) and LOW (below {threshold}). Give only the category.')
            a='HIGH' if value>=threshold else 'LOW'; rubric={'task':'classification','value':value,'threshold':threshold}
        elif t==2:
            nums=[i+12,i+3,i+8]
            p=(f'Sort these values smallest first. Reply with comma-separated numbers only: {nums[0]}, {nums[1]}, {nums[2]}.' if split=='train' else
               f'Arrange the measurements in increasing order: {nums[0]} / {nums[1]} / {nums[2]}. Use commas without words.')
            a=','.join(map(str,sorted(nums))); rubric={'task':'sort','values':nums}
        else:
            p=(f'Return a JSON object with exactly these facts: owner is {v["n"]}; item is {v["obj"]}. Use keys owner and item, no prose.' if split=='train' else
               f'Encode this record as JSON only, keeping precisely two keys (owner, item): {v["n"]} owns a {v["obj"]}.')
            record={'owner':v['n'],'item':v['obj']}; a=json.dumps(record); rubric={'task':'json','record':record}
        out.append(row(split,'direct',t,i,p,a,rubric,scoring='exact',evaluation_mode='model_only',
                       structural_signature=f'{split}/direct-source-structure-{t}'))
    return out

def build_splits():
    result={'train':writing('train')+retention('train'),'dev':writing('dev'),
            'writing_holdout':writing('writing_holdout'),'model_holdout':retention('model_holdout')}
    for i,rows in enumerate(result.values()): random.Random(SEED+i).shuffle(rows)
    return result


def output_force(s):
    s=norm(s)
    if re.search(r'\b(?:not required|not mandatory|do not have to|no requirement)\b',s): return 'no_obligation'
    if re.search(r'\b(?:must not|forbidden|not permitted|not allowed)\b',s) or re.search(r'^no .+ member may\b',s): return 'prohibition'
    if re.search(r'\b(?:must|mandatory|required|obligated)\b',s): return 'requirement'
    if re.search(r'\b(?:might|could|possibly|uncertain)\b',s): return 'uncertainty'
    if re.search(r'\bwill\b',s): return 'certainty'
    if re.search(r'\b(?:should|recommended)\b',s): return 'recommendation'
    if re.search(r'\b(?:allowed to|permitted to)\b',s): return 'permission'
    if re.search(r'\bmay\b',s): return 'may'
    if re.search(r'\b(?:please|asked to)\b',s): return 'request'
    return 'statement'


def validate_reference(r):
    """Check an authored target against task facts, NOT equality to itself.

    This bounded audit cannot certify arbitrary paraphrases. Semantic references
    and case rubrics still require author review and blinded model-output review.
    """
    errors=[]; a=r.get('answer'); p=r.get('prompt'); rub=r.get('rubric',{})
    if not isinstance(a,str) or not a.strip(): return ['empty_answer']
    if not isinstance(p,str) or not p.strip(): return ['empty_prompt']
    if not rub: return ['missing_rubric']
    for f in rub.get('protected',[]):
        if not present(f,a): errors.append('missing_fact:'+f)
    task=rub['task']
    if task=='faithful_shortening':
        source=r['source']
        if len(a)>=len(source): errors.append('not_shorter')
        if output_force(a)!=rub['force']: errors.append('force_changed')
        positions=[norm(a).find(norm(f)) for f in rub['ordered_anchors']]
        if any(x<0 for x in positions) or positions!=sorted(positions): errors.append('action_roles_changed')
        # Numeric strings include signs and time suffixes; no substring counting.
        nums=lambda s: re.findall(r'(?<!\w)[+-]?\d+(?::\d+)?(?:%|\s*[AP]M)?(?!\w)',s)
        if sorted(nums(source))!=sorted(nums(a)): errors.append('numeric_facts_changed')
        if re.search(r'\b(?:i sent|i have sent|message sent|i completed)\b',norm(a)): errors.append('invented_action')
    elif task=='recipient_perspective':
        if not a.startswith('Hi '+rub['recipient']+', '): errors.append('recipient_address')
        if not present(rub['ownership_phrase'],a): errors.append('ownership_changed')
        if not present(rub['sender_action'],a): errors.append('sender_or_commitment_changed')
        if rub['owner_relation']=='third_party' and present('your '+rub['object'],a): errors.append('third_party_reassigned')
        if re.search(r'\b(?:i sent|i have sent|message sent|i texted|i notified)\b',norm(a)): errors.append('claimed_sending')
    elif task=='exact_arithmetic':
        if a!=arithmetic_expected(rub): errors.append('wrong_arithmetic')
    elif task=='exact_extraction':
        if a!=rub['record'][rub['field']]: errors.append('wrong_extraction')
    elif task=='grounded_response':
        if rub['subtype']==0 and re.search(r'\d',a): errors.append('invented_arrival')
        if rub['subtype']==2 and re.search(r'(?<!not )\b(?:sent|delivered)\b',norm(a)): errors.append('claimed_action')
    elif task=='missing_text':
        if not a.endswith('?') and not a.startswith('Please paste '): errors.append('missing_clarification')
    elif task=='supplied_text':
        if re.search(r'\b(?:paste|share|provide)\b',norm(a)): errors.append('asked_for_existing_text')
    elif task=='copy':
        if a!=rub['literal']: errors.append('wrong_copy')
    elif task=='classification':
        if a!=('HIGH' if rub['value']>=rub['threshold'] else 'LOW'): errors.append('wrong_class')
    elif task=='sort':
        if a!=','.join(map(str,sorted(rub['values']))): errors.append('wrong_sort')
    elif task=='json':
        try:
            if json.loads(a)!=rub['record']: errors.append('wrong_json')
        except (ValueError,TypeError): errors.append('invalid_json')
    else: errors.append('unknown_task')
    return errors


def training_rows(rows):
    if len(rows)!=512: raise ValueError('Training export requires exactly 512 rows')
    if any(r.get('split')!='train' or r.get('training_allowed') is not True for r in rows):
        raise ValueError('Evaluation rows must never enter the training export')
    if len({r['id'] for r in rows})!=512: raise ValueError('Duplicate training id')
    return [{k:r[k] for k in ['id','prompt','answer']} for r in rows]


def prompt_skeleton(text):
    """Mask actual slot text; do not trust self-declared template identifiers."""
    text=norm(text)
    all_slots=[]
    for pool in POOLS.values():
        for category, values_ in pool.items():
            all_slots.extend((v,category) for v in values_)
    for v,category in sorted(set(all_slots),key=lambda p:-len(p[0])):
        text=re.sub(r'(?<!\w)'+re.escape(norm(v))+r'(?!\w)', '<'+category+'>', text)
    text=re.sub(r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b','<day>',text)
    text=re.sub(r'\b\d{1,2}:\d{2}\s*(?:am|pm)?\b','<time>',text)
    text=re.sub(r'(?<!\w)[+-]?\d+','<number>',text)
    return re.sub(r'\s+',' ',text).strip()


def audit_splits(splits, external_prompts=()):
    errors=[]; ids=set(); prompts={}; sources={}; templates={}; structures={}; groups={}; actual_structures={}
    external={norm(p) for p in external_prompts}
    if {k:len(v) for k,v in splits.items()}!=EXPECTED_COUNTS: errors.append('split_counts')
    if Counter(r['family'] for r in splits.get('train',[]))!=EXPECTED_TRAIN: errors.append('training_mix')
    for split, rows in splits.items():
        for r in rows:
            rid=r['id']
            if rid in ids: errors.append('duplicate_id:'+rid)
            ids.add(rid)
            if r['split']!=split or r['training_allowed']!=(split=='train'): errors.append('split_metadata:'+rid)
            pn=norm(r['prompt'])
            sk=prompt_skeleton(r.get('source',r['prompt']))
            if sk in actual_structures and actual_structures[sk]!=split: errors.append('cross_split_actual_structure:'+rid)
            actual_structures[sk]=split
            if pn in prompts: errors.append('duplicate_prompt:'+rid)
            prompts[pn]=split
            if pn in external: errors.append('external_prompt_overlap:'+rid)
            for field, store in [('source',sources),('template_id',templates),('structural_signature',structures),('contrast_group',groups)]:
                if field not in r: continue
                key=norm(r[field])
                if key in store and store[key]!=split: errors.append('cross_split_'+field+':'+rid)
                store[key]=split
            errors.extend(rid+':'+e for e in validate_reference(r))
    for name in ['dev','writing_holdout']:
        expected={'shortening':EXPECTED_COUNTS[name]//2,'recipient':EXPECTED_COUNTS[name]//2}
        if Counter(r['family'] for r in splits.get(name,[]))!=expected: errors.append('writing_balance:'+name)
    return {'version':VERSION,'passed':not errors,'errors':errors,'rows':len(ids),
        'split_counts':{k:len(v) for k,v in splits.items()},
        'family_counts':{k:dict(Counter(r['family'] for r in v)) for k,v in splits.items()},
        'template_counts':{k:len({r['template_id'] for r in v}) for k,v in splits.items()},
        'actual_cross_split_structures_checked':True,
        'external_prompt_count':len(external),
        'external_overlap_checked':bool(external),
        'author_review':'authored source/reference structures reviewed; not independent human review',
        'limitation':'Synthetic structured data; automatic checks are bounded, not a general semantic-equivalence proof',
        'training_started':False,'model_inference_performed':False}


def write_package(root, splits):
    root=Path(root); report=audit_splits(splits)
    if not report['passed']: raise ValueError(json.dumps(report['errors']))
    files={}
    for split,rows in splits.items():
        name=f'data/ember_writing_repair1/{"train" if split=="train" else "eval"}/{split}.jsonl'
        b=('\n'.join(json.dumps(r,ensure_ascii=False,sort_keys=True) for r in rows)+'\n').encode()
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        files[name]={'rows':len(rows),'bytes':len(b),'sha256':sha256(b)}
    name='data/ember_writing_repair1/train/sft_train_only.jsonl'
    b=('\n'.join(json.dumps(r,ensure_ascii=False,sort_keys=True) for r in training_rows(splits['train']))+'\n').encode()
    p=root/name;p.write_bytes(b); files[name]={'rows':512,'bytes':len(b),'sha256':sha256(b),'derived_copy':True}
    manifest={'version':VERSION,'seed':SEED,'baseline_commit':BASELINE_COMMIT,'grader_commit':GRADER_COMMIT,
        'files':files,'total_distinct_examples':sum(map(len,splits.values())),
        'train_rows':512,'evaluation_rows':256,'sft_export_is_not_additional_examples':True,
        'training_started':False,'model_inference_performed':False,
        'holdout_policy':'Do not train on evaluation files or use final holdout results to select checkpoints. Freeze these hashes before training.',
        'audit':report}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=Path('writing-repair1-package'))
    args=ap.parse_args();m=write_package(args.output,build_splits());print(json.dumps(m,indent=2))
