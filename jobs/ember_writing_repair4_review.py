"""Summarize the WR4 manual diagnostic review from committed evidence; CPU only, no model or Hub access."""
from collections import Counter
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'reports/evidence/writing-repair4-20260930/evidence'
REVIEW=ROOT/'reports/writing-repair4-diagnostic-review-20261002.json'
STATUSES=('pass','fail','review')


def norm(text):
    return re.sub(r'\s+',' ',text.strip().strip('"\'')).rstrip('.').lower()


def load(name):
    return json.loads((EVIDENCE/name).read_text())


def shortening_source(row):
    if row.get('kind') in ('shorten','shortening') and row.get('source'):
        return row['source']
    if str(row.get('id','')).startswith('draft-short-'):
        quoted=re.search(r"'([^']+)'",row.get('prompt',''))
        return quoted.group(1) if quoted else None
    return None


def length_stats(pairs):
    ratios=[len(out.strip())/len(src) for src,out in pairs]
    return {'rows':len(pairs),
            'verbatim_copy':sum(norm(out)==norm(src) for src,out in pairs),
            'not_shorter':sum(len(out.strip())>=len(src) for src,out in pairs),
            'mean_length_ratio':round(sum(ratios)/len(ratios),3)}


def benchmark_shortening(name):
    pairs,copies=[],[]
    for record in load(name)['records']:
        source=shortening_source(record['row'])
        if source:
            pairs.append((source,record['output']))
            if norm(record['output'])==norm(source):
                copies.append(record['suite']+'/'+record['id'])
    return {**length_stats(pairs),'verbatim_ids':copies}


def third_party_owner(rows,grades):
    named=Counter(); total=Counter()
    for row in rows:
        if not row['id'].endswith('third_party-pronoun'):
            continue
        pronoun=re.search(r'that (her|his|their) own',row['prompt']).group(1)
        total[pronoun]+=1
        named[pronoun]+='owner_not_resolved' not in grades[row['id']][1]
    return {p:[named[p],total[p]] for p in ('her','his','their')}


def summarize():
    review=json.loads(REVIEW.read_text())['rows']
    before,after=load('dev-before.json'),load('dev-after.json')
    ids=[r['id'] for r in before]
    if ids!=[r['id'] for r in after] or set(ids)!=set(review) or len(ids)!=64:
        raise ValueError('Review must grade exactly the 64 saved diagnostic rows')
    for grades in review.values():
        for status,reasons in grades.values():
            if status not in STATUSES or (status=='pass')!=(not reasons):
                raise ValueError('Pass rows carry no reasons; other rows need at least one')
    changed=[b['id'] for b,a in zip(before,after) if b['output']!=a['output']]
    greeting_only=[b['id'] for b,a in zip(before,after) if a['output']=='Hi '+b['output']]
    summary={'changed':len(changed),'greeting_only_changes':len(greeting_only),'families':{},'transitions':Counter(),
             'reasons':{},'third_party_owner_named':{},'dev_shortening_length':{},
             'benchmark_shortening_length':{'before':benchmark_shortening('baseline-744.json'),
                                            'after':benchmark_shortening('candidate-744.json')}}
    for label,rows in (('before',before),('after',after)):
        grades={r['id']:review[r['id']][label] for r in rows}
        summary['families'][label]={f:dict(Counter(grades[r['id']][0] for r in rows if r['family']==f))
                                    for f in ('shortening','recipient')}
        summary['reasons'][label]=dict(Counter(x for g in grades.values() for x in g[1]))
        summary['third_party_owner_named'][label]=third_party_owner(rows,grades)
        summary['dev_shortening_length'][label]=length_stats([(r['source'],r['output']) for r in rows if r['family']=='shortening'])
    for row_id in changed:
        summary['transitions'][review[row_id]['before'][0]+'->'+review[row_id]['after'][0]]+=1
    summary['transitions']=dict(summary['transitions'])
    summary['reference_length_ratio']=round(sum(len(r['answer'])/len(r['source']) for r in before if r['family']=='shortening')/32,3)
    return summary


if __name__=='__main__':
    print(json.dumps(summarize(),indent=2))
