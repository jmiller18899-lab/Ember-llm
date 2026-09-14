"""Reproduce v9 live evidence and separately score explicit country continuation."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import torch
from .evidence_v5 import emit_json
from .freeze_v9 import verify_freeze
from .runtime import sha256
from .runtime_v9 import RoutingV9Runtime
from .live_smoke import check
from .live_smoke_v9 import measure
from .live_services_v10 import LiveServicesV10,ServiceConversation

FILES=('tool_assistant/live_services_v10.py','tool_assistant/live_smoke_v10.py',
       'tests/test_live_services_v10.py','.github/workflows/ember-live-v10.yml',
       '.github/workflows/ember-live-v10-run.yml')


def source_lock(root):
    lock=json.loads((root/'config/ember_live_v10_source_lock.json').read_text())
    if set(lock['files'])!=set(FILES):raise ValueError('Incomplete service source lock')
    for p,digest in lock['files'].items():
        if sha256(root/p)!=digest:raise ValueError('Service source changed: '+p)
    return lock


def row(case,output,events):
    outcome,checks=check(case,output,events)
    retries=sum(e.get('attempt',1)>1 for e in events)
    return {**case,'output':output,'http':events,'outcome':outcome,'checks':checks,
            'first_attempt_outcome':'FAIL' if retries else outcome,
            'additional_http_attempts':retries,'recovered_after_retry':outcome=='PASS' and retries>0}


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    if args.report.exists():raise ValueError('Refusing to overwrite measured evidence')
    root=Path(__file__).resolve().parents[1]
    freeze=verify_freeze(root,args.bundle);lock=source_lock(root)
    torch.set_num_threads(2)
    results={};original={};flows={}
    for precision in ('full','int4'):
        runtime=RoutingV9Runtime(args.bundle,precision)
        original[precision]=measure(runtime,LiveServicesV10())
        services=LiveServicesV10();conversation=ServiceConversation(runtime,services)
        case={'id':'fes_country_question','kind':'guard','route':'get_time','status':'needs_clarification',
              'error':'ambiguous_location','network_services':['geocoding'],'dispatches':1,
              'user':'We are phoning a client in Fes; what time is it there currently?'}
        out=conversation.run(case['user'])
        initial=row(case,out,list(services.client.events))
        start=len(services.client.events)
        reply={'id':'fes_country_reply','kind':'service','route':'get_time','timezone':'Africa/Casablanca',
               'network_services':['geocoding','clock'],'user':'Morocco',
               'scope':'explicit_user_country_reply'}
        out=conversation.reply(reply['user'])
        final=row(reply,out,services.client.events[start:])
        flows[precision]=[initial,final]
        # Preserve all original-contract failures separately. The new contract
        # requires user clarification, not automatic success on bare 'Fes'.
        results[precision]=[r for r in original[precision] if r['id']!='repaired_client_time']+[initial,final]
    if freeze!=verify_freeze(root,args.bundle) or lock!=source_lock(root):raise ValueError('Source/candidate changed')
    rows=[r for rs in results.values() for r in rs]
    old=[r for rs in original.values() for r in rs]
    counts=lambda rs,field:{k:sum(r[field]==k for r in rs) for k in ('PASS','FAIL','BLOCKED')}
    totals=counts(rows,'outcome')
    report={'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),
            'scope':'service_v10_with_explicit_country_reply','results':results,
            'original_v9_contract':{'results':original,'counts':counts(old,'outcome')},
            'country_clarification_flows':flows,'counts':totals,
            'first_attempt_counts':counts(rows,'first_attempt_outcome'),
            'status':'PASS' if totals['PASS']==len(rows)==52 else 'FAIL',
            'freeze':freeze,'service_source_lock':lock,
            'production_ready':False,'base_model_trained':False,'routing_changed':False,
            'direct_answer_quality_tested':False,
            'interpretation':'52 turn checks: 48 unchanged service/guard checks and 4 explicit clarification turns. Original unqualified Fes failures retained separately; the test supplies Morocco as the user reply, never an inferred country.'}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    emit_json(args.report,args.report.name)
    print(json.dumps({'event':'live_v10_summary','status':report['status'],'counts':totals,
                      'first_attempt_counts':report['first_attempt_counts'],'original_v9_counts':report['original_v9_contract']['counts']}))
    if report['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
