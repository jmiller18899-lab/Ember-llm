"""Conservative reference checks for bounded, source-grounded direct responses."""
import re
from . import quality
from .learn import generate


def normalized(text):
    return tuple(re.findall(r"[+-]?[^\W_]+(?:[.'’+-][^\W_]+)*|[+-]", text.casefold()))


def score(text,row):
    allowed=[row['answer'],*row.get('alternatives',[])]
    return bool(normalized(text)) and any(normalized(text)==normalized(a) for a in allowed)


def evaluate(model,tokenizer,rows):
    output=[]
    for row in rows:
        generation=generate(model,tokenizer,row['user'])
        exact=score(generation['text'],row)
        generic=quality.generic_quality(generation['text'])
        semantic=quality.semantic_check(generation['text'],row['check'])
        output.append({'id':row['id'],'family':row['family'],'user':row['user'],
                       'reference':row['answer'],**generation,
                       'passed':generation['stopped_at_eot'] and exact,
                       'legacy_passed':generation['stopped_at_eot'] and generic['passed'] and semantic['passed'],
                       'content_reference_match':exact})
    return {'passed':sum(r['passed'] for r in output),'legacy_passed':sum(r['legacy_passed'] for r in output),
            'total':len(output),'rows':output}


def validate_data(data,tool_training):
    from .learn import validate_data as old_validate
    import json
    from pathlib import Path
    original=json.loads((Path(__file__).parent/'canary-data.json').read_text())
    old_validate(original,tool_training)
    groups=[data[k] for k in ('train','development','confirmation')]
    if list(map(len,groups))!=[312,48,24]:raise ValueError('Unexpected grounded split sizes')
    old_train=original['train']
    if data['train'][:72]!=old_train:raise ValueError('Historical replay changed')
    forbidden={normalized(r['user']) for r in quality.CASES+tool_training+original['development']+original['confirmation']}
    seen=set();ids=set()
    for group in groups:
        for row in group:
            key=normalized(row['user'])
            if key in seen or key in forbidden or row['id'] in ids:raise ValueError('Data overlap')
            if not row['answer'].strip():raise ValueError('Empty reference')
            seen.add(key);ids.add(row['id'])
    return True
