# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","accelerate==1.15.0","huggingface-hub==1.31.0","jinja2==3.1.6"]
# ///
"""One approved Writing Repair 1 epoch; frozen data/rules; no auto-promotion.

Only the 512-row SFT export enters training. New final holdouts are never run.
Checkpoints are saved to a separate private Hub repository during training and
before post-training evaluation. The frozen 744-case grader is unchanged.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import urllib.request

DATA_COMMIT='878a76ee7492504e648071626377a9524b4dec58'
GRADER_COMMIT='25924014c0e5d5a580a296b2841a1e6f6cbe3bb4'
BASE='Qwen/Qwen3.5-4B'
BASE_REV='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
SOURCE_MODEL='Jmiller18899/ember-qwen3.5-4b-repair2'
SOURCE_REV='daf938bba5d4e6b650ec9d34a2d3ac56706cf549'
OUTPUT_REPO='Jmiller18899/ember-qwen3.5-4b-writing-repair1-20260924'
SFT_SHA='3352bc50c505dc5cecafab88909b176be3385b45f3346d08e222189019995165'
SUITE_SHA='dcd42ca2c869f83f3bf46569baeca9199b6d0b0c38eab98ec8beefac0ac6fe32'
POLICY_SHA='e479dfc5b65e9b677c006dd58bf51e2113563bb6a16c9e3601bbbde5ce5d21fe'
PINS={
 'jobs/ember_writing_repair1_data.py':(DATA_COMMIT,'2c3c555833c6cc782b1266c3f88ae6616f91e4dd6154290c179eb61aead5f7bd'),
 'jobs/ember_writing_repair1_release.py':(DATA_COMMIT,'b2b4002bad823f308641b1a60a8b1975a30350002004fa5466d8223b656ff736'),
 'jobs/ember_drafting_repair_candidates_eval.py':(GRADER_COMMIT,'e134bf3919ea2871e7a10d3e90de877996f9c631414203762ab09c012bc42a9b'),
}
DATA_HASHES={
 'train/train.jsonl':'c716d547b4f7aec2f05dfbd1344f3a24bc48dd6ace2f2fdd24d91ab05b4af887',
 'train/sft_train_only.jsonl':SFT_SHA,
 'eval/dev.jsonl':'a8687aab3e256be3dee45ea25586f6aae64daef2cd1fef7e0c3de82c26af0e17',
 'eval/writing_holdout.jsonl':'4c380bc7258248a1f01fedc9cd6d2cb9e171782b5cbd2b74670004110d15e338',
 'eval/model_holdout.jsonl':'c5091e6f43a33de91ff57417bf6cc6db136077dbf6114e892568e9d8614e3d7d',
}
COUNTS={'exact72':72,'temporal8':8,'drafting8':8,'promo_v2':200,'heldout':180,'ctx_holdout':24,'suite_v3':192,'probes':36,'fresh_writing':24}
CORRECTED={'exact72':72,'temporal8':8,'drafting8':4,'promo_v2':200,'heldout':180,'ctx_holdout':24,'suite_v3':181,'probes':28,'fresh_writing':17}
EPOCHS=1
LR=1.5e-6
ACCUM=4
MAX_LEN=256
SEED=431


def run_spec():
    return {'experiment':'Writing Repair 1','data_version':'ember-writing-repair1-data-v1.2',
            'data_commit':DATA_COMMIT,'grader_commit':GRADER_COMMIT,'source_model':SOURCE_MODEL,
            'source_revision':SOURCE_REV,'base':BASE,'base_revision':BASE_REV,'output_repo':OUTPUT_REPO,
            'training_rows':512,'training_sha256':SFT_SHA,'epochs':EPOCHS,'learning_rate':LR,
            'microbatch':1,'gradient_accumulation':ACCUM,'optimizer_steps':expected_steps(),
            'warmup_steps':8,'max_length':MAX_LEN,'seed':SEED,'save_every_steps':32,
            'final_holdouts_evaluated':False,'automatic_promotion':False,'production_ready':False,
            'new_writing_scores_require_semantic_review':True,'answer_only_loss':True}


def expected_steps():
    return math.ceil(512/ACCUM)*EPOCHS


def verify_bytes(raw, expected):
    if hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError('Pinned file checksum mismatch')
    return raw


def validate_rows(rows):
    if len(rows)!=512: raise ValueError('Require exactly 512 training examples')
    ids=set();prompts=set()
    for r in rows:
        if set(r)!={'id','prompt','answer'}:raise ValueError('Only SFT id/prompt/answer fields allowed')
        if not all(isinstance(r[k],str) and r[k].strip() for k in r):raise ValueError('Invalid training text')
        if not r['id'].startswith('wr1-train-'):raise ValueError('Non-training split rejected')
        if r['id'] in ids or r['prompt'] in prompts:raise ValueError('Duplicate training entry')
        ids.add(r['id']);prompts.add(r['prompt'])
    return rows


def encode_ids(prefix,answer,end,max_len):
    if not isinstance(end,int) or not prefix or not answer:raise ValueError('Require prompt, target and end token')
    target=list(answer)+[end]
    if len(prefix)+len(target)>max_len:raise ValueError('No truncation allowed')
    return {'input_ids':list(prefix)+target,'labels':[-100]*len(prefix)+target}


def pad_batch(examples,pad):
    if not examples:raise ValueError('Empty batch')
    longest=max(len(x['input_ids']) for x in examples)
    batch={'input_ids':[],'labels':[],'attention_mask':[]}
    for x in examples:
        n=len(x['input_ids']);n_pad=longest-n
        if len(x['labels'])!=n:raise ValueError('Label alignment mismatch')
        batch['input_ids'].append(list(x['input_ids'])+[pad]*n_pad)
        batch['labels'].append(list(x['labels'])+[-100]*n_pad)
        batch['attention_mask'].append([1]*n+[0]*n_pad)
    return batch


def validate_output_repo(repo):
    if repo!=OUTPUT_REPO or repo==SOURCE_MODEL:raise ValueError('Only separate experimental output is allowed')


def verify_training(steps,loss,before_digest,after_digest):
    if steps!=expected_steps():raise ValueError('Training did not finish exactly one epoch')
    if not math.isfinite(loss):raise ValueError('Non-finite training loss')
    if before_digest==after_digest:raise ValueError('Adapter weights did not change')
    return True


def compare_records(before,after):
    def indexed(rows):
        out={}
        for r in rows:
            key=(r['suite'],r['id'])
            if key in out:raise ValueError('Duplicate evaluation identity')
            out[key]=r
        return out
    a,b=indexed(before),indexed(after)
    if set(a)!=set(b):raise ValueError('Evaluation coverage mismatch')
    return {'regressions':[list(k) for k in a if a[k]['ok'] and not b[k]['ok']],
            'improvements':[list(k) for k in a if not a[k]['ok'] and b[k]['ok']],
            'automatic_promotion':False,'semantic_review_required':True}


def fetch_pinned(rel):
    commit,sha=PINS[rel]
    local=Path(__file__).resolve().parents[1]/rel
    raw=local.read_bytes() if local.exists() else urllib.request.urlopen(
        f'https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{commit}/{rel}',timeout=60).read()
    return verify_bytes(raw,sha)


def load_inputs(work):
    jobs=work/'jobs';jobs.mkdir(parents=True,exist_ok=True)
    for rel in PINS:(jobs/Path(rel).name).write_bytes(fetch_pinned(rel))
    sys.path.insert(0,str(jobs))
    import ember_writing_repair1_release as D
    import ember_drafting_repair_candidates_eval as G
    splits=D.build_splits()
    data_audit=D.audit_splits(splits)
    if not data_audit['passed']:raise RuntimeError('Frozen data audit failed')
    D.write_package(work,splits)
    data_dir=work/'data/ember_writing_repair1'
    for rel,sha in DATA_HASHES.items():verify_bytes((data_dir/rel).read_bytes(),sha)
    # Only this exact export is permitted as train_dataset. Other splits are never concatenated.
    rows=validate_rows([json.loads(l) for l in (data_dir/'train/sft_train_only.jsonl').read_text().splitlines()])
    E,M,policy_sha=G.load_frozen()
    if policy_sha!=POLICY_SHA:raise RuntimeError('Frozen policy checksum mismatch')
    if (E.BASE,E.BASE_REV,E.MODEL,E.MODEL_REV)!=(BASE,BASE_REV,SOURCE_MODEL,SOURCE_REV):
        raise RuntimeError('Model pins changed')
    from huggingface_hub import hf_hub_download
    bench=json.loads(Path(hf_hub_download(E.BENCH,'candidate.json',revision=E.BENCH_REV)).read_text())
    exact=[r for r in bench if r.get('scoring')=='exact']
    suites=E.suites(M,exact)
    suites['fresh_writing']=[(r,G.legacy_fresh_score,r['family']) for r in G.fresh_cases()]
    manifest=[{'suite':s,'row':{**r,'id':r.get('id',str(i))}}
              for s,items in suites.items() for i,(r,_,_) in enumerate(items)]
    verify_bytes(json.dumps(manifest,sort_keys=True,ensure_ascii=False).encode(),SUITE_SHA)
    if {s:len(v) for s,v in suites.items()}!=COUNTS:raise RuntimeError('Suite coverage mismatch')
    audit=E.audit(M,[r['prompt'] for r in exact])
    if any(audit.values()):raise RuntimeError('Frozen routing audit failed')
    return rows,splits['dev'],G,E,M,suites


def encoded_rows(rows,tok,route,base_system):
    result=[]
    for r in rows:
        kind,value=route(r['prompt'])
        system=value if kind=='model' else base_system
        ids=tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':r['prompt']}],
            tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        result.append(encode_ids(ids,tok.encode(r['answer'],add_special_tokens=False),
                                 tok.convert_tokens_to_ids('<|im_end|>'),MAX_LEN))
    return result


def evaluate(suites,route,generate,G):
    records=[];scores={};cache={}
    for suite,items in suites.items():
        counts=Counter();families={};legacy=0
        for i,(row,scorer,family) in enumerate(items):
            kind,value=route(row['prompt'])
            key=(value,row['prompt'])
            if kind=='tool':output=value
            else:
                if key not in cache:cache[key]=generate(value,row['prompt'])
                output=cache[key]
            grade=G.grade_case(row,output,scorer)
            counts[grade['status']]+=1;legacy+=grade['legacy_passed']
            f=families.setdefault(family,[0,0]);f[0]+=grade['passed'];f[1]+=1
            records.append({'suite':suite,'id':row.get('id',str(i)),'row':row,'output':output,
                            'ok':grade['passed'],'legacy_ok':grade['legacy_passed'],'grading':grade})
        scores[suite]={'pass':counts['pass'],'fail':counts['fail'],'review':counts['review'],
                       'total':len(items),'legacy_pass':legacy,'families':families}
        print('WR1_EVAL_SUITE',json.dumps({'suite':suite,**scores[suite]}),flush=True)
    return scores,records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    work=Path(tempfile.mkdtemp(prefix='ember-wr1-train-'))
    rows,dev,G,E,M,suites=load_inputs(work)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(BASE,revision=BASE_REV);tok.pad_token=tok.eos_token
    route=G.build_route(E,M,'v3_baseline')
    encoded=encoded_rows(rows,tok,route,M.O.SYSTEM)
    info={**run_spec(),'max_training_tokens':max(len(x['input_ids']) for x in encoded),
          'baseline_manifest_sha256':SUITE_SHA,'trainer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'trainer_commit':os.environ.get('WR1_CODE_COMMIT'),'data_hashes':DATA_HASHES}
    print('WR1_TRAIN_PREFLIGHT_PASS',json.dumps(info),flush=True)
    if args.preflight:return

    import torch
    from huggingface_hub import HfApi,hf_hub_download
    from peft import PeftModel
    from transformers import Qwen3_5ForCausalLM,Trainer,TrainingArguments,TrainerCallback,set_seed
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    validate_output_repo(OUTPUT_REPO)
    api=HfApi(token=os.environ['HF_TOKEN'])
    if api.whoami().get('name')!='Jmiller18899':raise RuntimeError('Wrong training account')
    reservation=json.loads(Path(hf_hub_download(OUTPUT_REPO,'launch.json')).read_text())
    if reservation.get('data_sha256')!=SFT_SHA:raise RuntimeError('Unrecognized launch reservation')
    if api.file_exists(OUTPUT_REPO,'evidence/training-complete.json'):
        raise RuntimeError('Training already completed; refusing a duplicate run')
    if any(f.endswith('.safetensors') for f in api.list_repo_files(OUTPUT_REPO)):
        raise RuntimeError('Existing checkpoint found; do not silently retrain or overwrite')
    def upload_json(rel,payload):
        data=json.dumps(payload,ensure_ascii=False,indent=2).encode()
        result=api.upload_file(repo_id=OUTPUT_REPO,path_in_repo=rel,path_or_fileobj=data,
                               commit_message='Writing Repair 1 experimental evidence')
        return result.oid
    upload_json('evidence/run-spec.json',info)  # verify GPU-side write access before any optimizer step
    set_seed(SEED)
    base,loading=Qwen3_5ForCausalLM.from_pretrained(BASE,revision=BASE_REV,dtype=torch.bfloat16,
        device_map={'':0},output_loading_info=True,key_mapping={r'^model.language_model\.':'model.'})
    if loading['missing_keys'] or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise RuntimeError('Base model load mismatch')
    model=PeftModel.from_pretrained(base,SOURCE_MODEL,revision=SOURCE_REV,is_trainable=True)
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n for n,_ in trainable):raise RuntimeError('Only LoRA parameters may train')
    def digest():
        h=hashlib.sha256()
        for n,p in trainable:
            h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    before_digest=digest()
    def generate(system,prompt):
        model.eval()
        ids=tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':prompt}],
            tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=False).to(model.device)
        with torch.inference_mode():
            out=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,
                               do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
        return tok.decode(out[0,ids.shape[-1]:],skip_special_tokens=True).strip()
    def dev_outputs():
        out=[]
        for r in dev:
            kind,system=route(r['prompt'])
            if kind!='model':raise RuntimeError('Writing development route changed')
            out.append({'id':r['id'],'prompt':r['prompt'],'source':r.get('source'),
                        'output':generate(system,r['prompt']),'semantic_grade':None,'manual_review_pending':True})
        return out
    before,before_records=evaluate(suites,route,generate,G)
    upload_json('evidence/baseline-744.json',{'scores':before,'records':before_records})
    if {s:x['pass'] for s,x in before.items()}!=CORRECTED:
        raise RuntimeError('Corrected baseline did not reproduce; training not started')
    if sum(x['review'] for x in before.values())!=4:raise RuntimeError('Baseline review counts changed')
    before_dev=dev_outputs();upload_json('evidence/dev-before.json',before_dev)
    checkpoint0=work/'checkpoint-0';model.save_pretrained(checkpoint0);tok.save_pretrained(checkpoint0)
    api.upload_folder(repo_id=OUTPUT_REPO,folder_path=str(checkpoint0),path_in_repo='checkpoint-0',
                      commit_message='Preserve unchanged starting adapter before training')
    print('WR1_TRAINING_START',json.dumps({'rows':len(encoded),'optimizer_steps':expected_steps(),
        'trainable_parameters':sum(p.numel() for _,p in trainable),'starting_adapter_digest':before_digest,
        'baseline_reproduced':True,'final_holdouts_evaluated':False}),flush=True)
    class SaveToHub(TrainerCallback):
        def on_save(self,args,state,control,**kwargs):
            folder=Path(args.output_dir)/f'checkpoint-{state.global_step}'
            api.upload_folder(repo_id=OUTPUT_REPO,folder_path=str(folder),path_in_repo=f'checkpoints/step-{state.global_step}',
                              commit_message=f'Persist Writing Repair 1 step {state.global_step}')
            upload_json('evidence/progress.json',{'step':state.global_step,'total_steps':expected_steps(),
                        'checkpoint':f'checkpoints/step-{state.global_step}','automatic_promotion':False})
            print('WR1_CHECKPOINT_SAVED',state.global_step,flush=True)
    def collate(examples):
        return {k:torch.tensor(v,dtype=torch.long) for k,v in pad_batch(examples,tok.pad_token_id).items()}
    model.train();model.config.use_cache=False
    training_args=TrainingArguments(output_dir=str(work/'training'),num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,gradient_accumulation_steps=ACCUM,learning_rate=LR,
        warmup_steps=8,bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        save_strategy='steps',save_steps=32,save_total_limit=2,logging_steps=8,report_to=[],
        seed=SEED,data_seed=SEED,remove_unused_columns=False,dataloader_num_workers=0)
    trainer=Trainer(model=model,args=training_args,train_dataset=encoded,data_collator=collate,callbacks=[SaveToHub()])
    trained=trainer.train()
    after_digest=digest()
    verify_training(trainer.state.global_step,float(trained.training_loss),before_digest,after_digest)
    candidate=work/'candidate';model.save_pretrained(candidate);tok.save_pretrained(candidate)
    saved=api.upload_folder(repo_id=OUTPUT_REPO,folder_path=str(candidate),path_in_repo='candidate',
                             commit_message='Save one-epoch Writing Repair 1 candidate; not promoted')
    completion={**info,'steps_run':trainer.state.global_step,'training_loss':float(trained.training_loss),
                'starting_digest':before_digest,'candidate_digest':after_digest,'candidate_commit':saved.oid,
                'adapter_subfolder':'candidate','weights_changed':True,'training_completed':True,
                'evaluation_complete':False,'training_metrics':trained.metrics}
    upload_json('evidence/training-complete.json',completion)
    print('WR1_TRAINING_COMPLETE',json.dumps(completion),flush=True)
    model.gradient_checkpointing_disable();model.config.use_cache=True
    after,after_records=evaluate(suites,route,generate,G)
    comparison=compare_records(before_records,after_records)
    upload_json('evidence/candidate-744.json',{'scores':after,'records':after_records})
    upload_json('evidence/dev-after.json',dev_outputs())
    report={**completion,'evaluation_complete':True,'before':before,'after':after,**comparison,
            'baseline_runtime_unchanged':True,'final_holdouts_evaluated':False,'manual_review_completed':False,
            'candidate_selected':'fixed final checkpoint; not selected using holdout scores'}
    upload_json('evidence/final-report.json',report)
    card=('---\nbase_model: Qwen/Qwen3.5-4B\nlibrary_name: peft\n---\n'
          '# Ember Writing Repair 1\n\nExperimental one-epoch adapter in `candidate/`. Not deployed or promoted.\n'
          'See `evidence/final-report.json`. New writing needs semantic review; final holdouts remain untouched.\n')
    api.upload_file(repo_id=OUTPUT_REPO,path_in_repo='README.md',path_or_fileobj=card.encode(),commit_message='Document experimental candidate')
    print('WR1_FINAL_REPORT',json.dumps(report),flush=True)

if __name__=='__main__':main()
