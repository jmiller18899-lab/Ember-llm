# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["huggingface-hub==1.31.0"]
# ///
"""Retrieve WR4 saved evidence only; no model loading, inference, or Hub writes."""
from dataclasses import asdict,is_dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from huggingface_hub import HfApi,hf_hub_download

REPO='Jmiller18899/ember-qwen3.5-4b-writing-repair4-20260930'
JOB='6abd34a9404719ba37613b46'
NAMESPACE='Jmiller18899'
FILES=['launch.json','evidence/launch-submission.json','evidence/run-spec.json',
       'evidence/progress.json','evidence/training-complete.json','evidence/final-report.json',
       'evidence/baseline-744.json','evidence/candidate-744.json','evidence/dev-before.json',
       'evidence/dev-after.json','evidence/training-metrics.json','development/fresh-dev.json']


def main():
    api=HfApi(token=os.environ['HF_TOKEN'])
    if api.whoami().get('name')!=NAMESPACE:
        raise RuntimeError('Unexpected Hugging Face account')
    info=api.model_info(REPO)
    revision=info.sha
    available=set(api.list_repo_files(REPO,revision=revision))
    folder=Path('wr4-saved-evidence')
    folder.mkdir(exist_ok=True)
    inspection={'job_id':JOB,'job_url':f'https://huggingface.co/jobs/{NAMESPACE}/{JOB}',
                'repo_id':REPO,'revision':revision,'private':info.private,
                'method':'Pinned saved-file retrieval; no weights or new inference',
                'candidate_weights_present':any(x.startswith('candidate/') and x.endswith('.safetensors') for x in available),
                'files':[],'missing_files':[]}
    job=api.inspect_job(job_id=JOB,namespace=NAMESPACE)
    status=job.status
    inspection['job_status']=asdict(status) if is_dataclass(status) else str(status)
    for key in ('created_at','started_at','finished_at','completed_at'):
        if getattr(job,key,None) is not None:
            inspection[key]=str(getattr(job,key))
    for name in FILES:
        if name not in available:
            inspection['missing_files'].append(name)
            continue
        source=Path(hf_hub_download(REPO,name,revision=revision,token=os.environ['HF_TOKEN']))
        raw=source.read_bytes()
        json.loads(raw)
        target=folder/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        inspection['files'].append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    (folder/'inspection.json').write_text(json.dumps(inspection,indent=2,default=str))
    summary={k:inspection[k] for k in ('job_id','revision','job_status','candidate_weights_present','missing_files')}
    report=folder/'evidence/final-report.json'
    if report.is_file():
        saved=json.loads(report.read_text())
        summary.update({k:saved.get(k) for k in ('training_completed','evaluation_complete','steps_completed','training_loss','candidate_commit','comparison')})
        for label in ('before','after'):
            scores=saved.get(label,{})
            summary[label]={metric:sum(x.get(metric,0) for x in scores.values()) for metric in ('pass','fail','review','total')}
    print('WR4_SAVED_EVIDENCE',json.dumps(summary,default=str),flush=True)


if __name__=='__main__':
    main()
