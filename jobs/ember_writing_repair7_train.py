# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","accelerate==1.15.0","huggingface-hub==1.31.0","jinja2==3.1.6","trackio"]
# ///
"""One WR7 data-only experiment: fixed Repair2 parent; no retry or promotion.

Reuse the checksum-pinned WR2 benchmark engine, not its candidate weights.
Generate fresh development only; leave both final 96-case holdouts unused.
CPU --preflight performs no Hub writes. --launch uses the existing Actions secret.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from collections import Counter

BRANCH = 'codex/ember-next-training-20261009'
OUT = 'Jmiller18899/ember-qwen3.5-4b-writing-repair7-20261009'
SOURCE = 'Jmiller18899/ember-qwen3.5-4b-repair2'
SOURCE_REV = 'daf938bba5d4e6b650ec9d34a2d3ac56706cf549'
ENGINE_COMMIT = '43220b453807782091a9209384091e82fed086f8'
ENGINE_SHA = 'e20ccd3ac99de436c4fbd5e586e8bd303201e39965500b137bfcd2fa739d0eeb'
DATA_SHA = '67d328bc54e666db6a4f05cbce368f1510a6a2f39c9387d1e04efe1219e3a061'
FLAVOR, TIMEOUT = 'l4x1', '90m'
LR, STEPS, ACCUM, MAX_LEN, SEED = 7.5e-7, 128, 4, 384, 431
AUTO_PROMOTION = False
AUTO_RETRY = False
PREVIOUS_DATA_COMMIT = '68302b05c47825c33509ed9c5047f57c22f40842'
PREVIOUS_DATA_SHA = 'db60589c1a150d4f1feb15762e2bdc7914ee5dd6d7ed9b5fb8038019b4cbd1a1'


def verify(raw, sha):
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('Pinned source checksum mismatch')
    return raw


def validate_launch(env, output_exists):
    if env.get('GITHUB_REF') != 'refs/heads/'+BRANCH:
        raise ValueError('Launch must use the isolated WR7 branch')
    if env.get('GITHUB_RUN_ATTEMPT') != '1':
        raise ValueError('Automatic or manual workflow reruns cannot relaunch training')
    if not re.fullmatch('[0-9a-f]{40}', env.get('WR7_CODE_COMMIT', '')) or not env.get('HF_TOKEN'):
        raise ValueError('Pinned code and existing HF secret are required')
    if output_exists:
        raise ValueError('Output already exists; duplicate launch/retry refused')

def validate_storage(files, receipt):
    allowed = {'.gitattributes','storage-preflight.json'}
    if set(files) - allowed or 'storage-preflight.json' not in files:
        raise ValueError('Output has a launch or candidate; refusing duplicate training')
    expected = {'status':'PASS','output_repo':OUT,'source_model':SOURCE,
                'source_revision':SOURCE_REV,'training_started':False}
    if any(receipt.get(k) != v for k,v in expected.items()):
        raise ValueError('Private storage reservation mismatch')


def encode_ids(prefix, answer, end, max_len):
    if not prefix or not answer or not isinstance(end, int):
        raise ValueError('Missing prompt, answer or end token')
    target = list(answer)+[end]
    if len(prefix)+len(target) > max_len:
        raise ValueError('No truncation allowed')
    return {'input_ids':list(prefix)+target, 'labels':[-100]*len(prefix)+target}


def read_source(path, commit, sha):
    import urllib.request
    local = Path(__file__).resolve().parent / Path(path).name
    if local.is_file():
        return verify(local.read_bytes(), sha)
    url = 'https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/'+commit+'/'+path
    return verify(urllib.request.urlopen(url, timeout=60).read(), sha)


def load_module(name, raw, folder):
    path = folder/(name+'.py'); path.write_bytes(raw)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare():
    revision = os.environ.get('WR7_CODE_COMMIT', '')
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('WR7_CODE_COMMIT must be a pinned Git commit')
    work = Path(tempfile.mkdtemp(prefix='ember-wr7-'))
    engine = read_source('jobs/ember_writing_repair2_train.py', ENGINE_COMMIT, ENGINE_SHA)
    data_raw = read_source('jobs/ember_writing_repair7_data.py', revision, DATA_SHA)
    T = load_module('wr7_frozen_engine', engine, work)
    W = load_module('wr7_data', data_raw, work)
    prior = load_module('wr7_previous_data', read_source('jobs/ember_writing_repair3_data.py',PREVIOUS_DATA_COMMIT,PREVIOUS_DATA_SHA),work)
    os.environ['WR2_CODE_COMMIT'] = ENGINE_COMMIT
    old_rows, old_dev, G, E, M, suites = T.load_inputs(work/'history')
    if (T.SOURCE_MODEL,T.SOURCE_REV) != (SOURCE,SOURCE_REV):
        raise ValueError('Frozen parent mismatch')
    previous4 = load_module('wr7_wr4_history', read_source('jobs/ember_writing_repair4_data.py',
                            '5f2caf0360cd8e878500180abebc08aa4b1f4cf3',
                            'ea73fe0e4154778f1f1de8cf99e4b7ed876b42163780e5c67db33bf8bf9eaf66'), work)
    import ember_writing_repair1_data as D
    rows = W.build_train_rows(D.retention('train'))
    dev = W.writing('dev')
    history = [r for items in suites.values() for r,_,_ in items]+old_rows+old_dev+D.writing('train')+prior.writing('train')+prior.writing('dev')
    history += previous4.writing('train')+previous4.writing('dev')
    benchmark_sources = [W.shortening_source(r) for items in suites.values() for r,_,_ in items
                         if W.shortening_source(r)]
    if len(benchmark_sources) != 30:
        raise ValueError('Frozen shortening coverage changed')
    original = load_module('wr7_original_wr5', read_source('jobs/ember_writing_repair5_data.py',
        '4e406a094c0494e684eea9850c70242c441bc3ef',
        'a41ad5e8194e1280d310247450d2ec5d8d167927727103ffa8181dbc25c516fa'), work)
    original_rows = original.build_train_rows(D.retention('train'))
    if hashlib.sha256(original.sft_bytes(original_rows)).hexdigest() != '887c0df22cb668afb33e52c65b217e42597e437658790566a3d44ccb177255bf':
        raise ValueError('Original WR5 export failed to reproduce')
    changed = [(a,b) for a,b in zip(original_rows,rows) if a != b]
    if len(rows) != len(original_rows) or len(changed) != 16:
        raise ValueError('Exactly 16 rows must change')
    if any(a['id'] != b['id'] or a.get('shortening_group') != 'little' or b.get('shortening_group') != 'little' for a,b in changed):
        raise ValueError('Only the 16 small-cut rows may change, at the same positions')
    if original.writing('dev') != dev[:64] or len(dev) != 72:
        raise ValueError('Original development must remain identical')
    wr6 = load_module('wr7_wr6_history', read_source('jobs/ember_writing_repair6_data.py',
        'fb7b0cfe692cc409cde62ea5caf1cb5b6f391773',
        'a6cbaaf11625d616606bc9fd902c09c3cd9156a3d2576de89f2b5a12fc5fcf01'), work)
    new_rows = [b for a,b in changed]
    used = history + original.writing('train') + original.writing('dev') + wr6.writing('train') + wr6.writing('dev')
    used_prompts = {W.norm(r['prompt']) for r in used}
    used_sources = {W.norm(r['source']) for r in used if r.get('source')}
    if any(W.norm(r['prompt']) in used_prompts or W.norm(r['source']) in used_sources for r in new_rows):
        raise ValueError('Replacement overlaps used data or evaluation')
    audit = W.audit(rows,dev,history,benchmark_sources)
    audit.update(controlled_changed_rows=16,unchanged_rows=496,unchanged_row_order=True,
        replacement_history_checked_through='WR6',original_development_reused=True,additional_compact_development_rows=8,
        semantic_reference_review='Assistant-reviewed conservative edits; not independent human grading')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(T.BASE, revision=T.BASE_REV)
    tok.pad_token = tok.eos_token
    route = G.build_route(E,M,'v3_baseline')
    encoded = []
    for r in rows:
        kind,value = route(r['prompt'])
        if r['family'] in ('shortening','recipient') and kind != 'model':
            raise ValueError('Writing unexpectedly routed to deterministic tool')
        system = value if kind == 'model' else M.O.SYSTEM
        prefix = tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':r['prompt']}],
            tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        encoded.append(encode_ids(prefix,tok.encode(r['answer'],add_special_tokens=False),
                                  tok.convert_tokens_to_ids('<|im_end|>'),MAX_LEN))
    for r in dev:
        if route(r['prompt'])[0] != 'model':
            raise ValueError('Fresh development must test the model')
    raw = W.sft_bytes(rows)
    spec = {'experiment':'Writing Repair 7','data_version':W.VERSION,'code_commit':revision,
            'data_code_sha256':DATA_SHA,'frozen_engine_commit':ENGINE_COMMIT,'frozen_engine_sha256':ENGINE_SHA,
            'source_model':SOURCE,'source_revision':SOURCE_REV,'base':T.BASE,'base_revision':T.BASE_REV,
            'policy_sha256':T.POLICY_SHA,'grader_commit':T.GRADER_COMMIT,'benchmark_sha256':T.SUITE_SHA,
            'training_rows':len(rows),'fresh_development_rows':len(dev),'diagnostic_baseline_before_optimizer':True,'families':dict(Counter(r['family'] for r in rows)),
            'training_sha256':hashlib.sha256(raw).hexdigest(),'epochs':1,'learning_rate':LR,
            'optimizer_steps':STEPS,'microbatch':1,'gradient_accumulation':ACCUM,'warmup_steps':8,
            'max_length':MAX_LEN,'max_training_tokens':max(len(r['input_ids']) for r in encoded),
            'seed':SEED,'save_every_steps':32,'hardware':FLAVOR,'timeout':TIMEOUT,'output_repo':OUT,
            'automatic_retry':AUTO_RETRY,'automatic_promotion':AUTO_PROMOTION,'production_ready':False,
            'answer_only_loss':True,'final_holdouts_evaluated':False,'semantic_review_required':True,
            'candidate_selection':'Fixed final checkpoint; no holdout-based selection','data_audit':audit,
            'acceptance':{'minimum_benchmark_pass':715,'maximum_benchmark_verbatim_copies':5,
                          'protected_family_regressions_allowed':0,'shortening_regressions_allowed':0,
                          'minimum_manual_diagnostic_net_gain':3,'new_diagnostic_copies_allowed':0}}
    (work/'train.jsonl').write_bytes(raw)
    (work/'dev.json').write_text(json.dumps(dev,indent=2))
    print('WR7_PREFLIGHT_PASS',json.dumps(spec),flush=True)
    return work,T,W,G,E,M,suites,rows,dev,tok,route,encoded,spec


def launch(prepared):
    work,T,W,G,E,M,suites,rows,dev,tok,route,encoded,spec = prepared
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    validate_launch(os.environ,False)
    api = HfApi(token=os.environ['HF_TOKEN'])
    if api.whoami().get('name') != 'Jmiller18899':
        raise ValueError('Wrong Hugging Face account')
    if api.repo_exists(OUT):
        if not api.model_info(OUT).private:
            raise ValueError('Candidate output must stay private')
        receipt = json.loads(Path(hf_hub_download(OUT,'storage-preflight.json')).read_text())
        validate_storage(api.list_repo_files(OUT),receipt)
    else:
        api.create_repo(OUT,repo_type='model',private=True,exist_ok=False)
    raw_trainer = Path(__file__).read_bytes()
    manifest = dict(spec,trainer_sha256=hashlib.sha256(raw_trainer).hexdigest())
    api.create_commit(repo_id=OUT,operations=[
        CommitOperationAdd(path_in_repo='launch.json',path_or_fileobj=json.dumps(manifest,indent=2).encode()),
        CommitOperationAdd(path_in_repo='train/sft_train_only.jsonl',path_or_fileobj=(work/'train.jsonl').read_bytes()),
        CommitOperationAdd(path_in_repo='development/fresh-dev.json',path_or_fileobj=(work/'dev.json').read_bytes()),
        CommitOperationAdd(path_in_repo='source/ember_writing_repair7_train.py',path_or_fileobj=raw_trainer),
        CommitOperationAdd(path_in_repo='source/ember_writing_repair7_data.py',path_or_fileobj=Path(W.__file__).read_bytes()),
    ],commit_message='Reserve audited WR7 experiment; do not promote')
    job = api.run_uv_job(script=str(Path(__file__).resolve()),python='3.11',flavor=FLAVOR,timeout=TIMEOUT,
                        env={'WR7_CODE_COMMIT':spec['code_commit'],'TOKENIZERS_PARALLELISM':'false'},
                        secrets={'HF_TOKEN':os.environ['HF_TOKEN']},labels={'name':'ember-writing-repair7-20261009'})
    receipt = {'job_id':job.id,'url':job.url,'output_repo':OUT,'gpu_timeout':TIMEOUT,'automatic_retry':False}
    print('WR7_GPU_JOB_SUBMITTED',json.dumps(receipt),flush=True)
    api.upload_file(repo_id=OUT,path_in_repo='evidence/launch-submission.json',
                    path_or_fileobj=json.dumps(receipt,indent=2).encode(),commit_message='Record WR7 job submission')


def train(prepared):
    work,T,W,G,E,M,suites,rows,dev,tok,route,encoded,spec = prepared
    import torch
    from huggingface_hub import HfApi,hf_hub_download
    from peft import PeftModel
    from transformers import Qwen3_5ForCausalLM,Trainer,TrainingArguments,TrainerCallback,set_seed
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    api = HfApi(token=os.environ['HF_TOKEN'])
    if api.whoami().get('name') != 'Jmiller18899':
        raise ValueError('Wrong Hugging Face account')
    reservation = json.loads(Path(hf_hub_download(OUT,'launch.json')).read_text())
    for key,value in spec.items():
        if reservation.get(key) != value:
            raise ValueError('Launch reservation mismatch: '+key)
    verify(Path(__file__).read_bytes(),reservation['trainer_sha256'])
    if not api.model_info(OUT).private or any(f.endswith('.safetensors') for f in api.list_repo_files(OUT)):
        raise ValueError('Require fresh private output; refusing overwrite/retry')
    def upload(path,data):
        return api.upload_file(repo_id=OUT,path_in_repo=path,path_or_fileobj=json.dumps(data,ensure_ascii=False,indent=2).encode(),
                               commit_message='WR7 experimental evidence; no promotion')
    upload('evidence/run-spec.json',spec)
    set_seed(SEED)
    base,loading = Qwen3_5ForCausalLM.from_pretrained(T.BASE,revision=T.BASE_REV,dtype=torch.bfloat16,
        device_map={'':0},output_loading_info=True,key_mapping={r'^model.language_model\.':'model.'})
    if loading['missing_keys'] or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise RuntimeError('Base model weights did not load exactly')
    model = PeftModel.from_pretrained(base,SOURCE,revision=SOURCE_REV,is_trainable=True)
    trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n for n,p in trainable):
        raise ValueError('Only the existing LoRA adapter may change')
    def digest():
        h = hashlib.sha256()
        for n,p in trainable:
            h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    def generate(system,prompt):
        model.eval()
        ids = tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':prompt}],
            tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=False).to(model.device)
        with torch.inference_mode():
            output = model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,
                                    do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
        return tok.decode(output[0,ids.shape[-1]:],skip_special_tokens=True).strip()
    def dev_outputs():
        return [dict(r,output=generate(route(r['prompt'])[1],r['prompt']),semantic_grade=None,
                     manual_review_pending=True) for r in dev]
    def checkpoint(folder,path):
        model.save_pretrained(folder);tok.save_pretrained(folder)
        return api.upload_folder(repo_id=OUT,folder_path=str(folder),path_in_repo=path,
                                 commit_message='Persist isolated WR7 checkpoint; no promotion')
    before_digest = digest()
    before,before_records = T.evaluate(suites,route,generate,G)
    upload('evidence/baseline-744.json',{'scores':before,'records':before_records})
    if {s:x['pass'] for s,x in before.items()} != T.CORRECTED or sum(x['review'] for x in before.values()) != 4:
        raise RuntimeError('Frozen 714/744 baseline failed to reproduce; no training performed')
    baseline_dev = dev_outputs()
    upload('evidence/dev-before.json',baseline_dev)
    before_lengths = {'benchmark':W.length_stats([(W.shortening_source(r['row']),r['output'])
                       for r in before_records if W.shortening_source(r['row'])]),
                      'diagnostic':W.length_stats([(r['source'],r['output'])
                       for r in baseline_dev if r['family']=='shortening'])}
    upload('evidence/shortening-before.json',before_lengths)
    print('WR7_DIAGNOSTIC_BASELINE_SAVED',json.dumps({'cases':len(baseline_dev),'semantic_review_pending':True}),flush=True)
    checkpoint(work/'checkpoint-0','checkpoint-0')
    import trackio
    trackio.init(project='ember-writing',name='writing-repair7-20261009',space_id=None,embed=False,auto_log_gpu=False,auto_log_cpu=False,
        config={'model':SOURCE,'learning_rate':LR,'epochs':1,'training_rows':len(rows)})
    metrics = []
    class Save(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            values = {k:v for k,v in (logs or {}).items() if isinstance(v,(int,float))}
            if values:
                trackio.log(values,step=state.global_step)
                metrics.append(dict(step=state.global_step,**values))
                print('WR7_TRAINING_METRICS',json.dumps(metrics[-1]),flush=True)
        def on_save(self,args,state,control,**kwargs):
            folder = Path(args.output_dir)/f'checkpoint-{state.global_step}'
            api.upload_folder(repo_id=OUT,folder_path=str(folder),path_in_repo=f'checkpoints/step-{state.global_step}',
                              commit_message=f'Persist WR7 step {state.global_step}')
            upload('evidence/progress.json',{'step':state.global_step,'total_steps':STEPS,'automatic_promotion':False})
            upload('evidence/training-metrics.json',metrics)
            print('WR7_CHECKPOINT_SAVED',state.global_step,flush=True)
    def collate(examples):
        return {k:torch.tensor(v,dtype=torch.long) for k,v in T.pad_batch(examples,tok.pad_token_id).items()}
    model.train();model.config.use_cache=False
    args = TrainingArguments(output_dir=str(work/'training'),num_train_epochs=1,
        per_device_train_batch_size=1,gradient_accumulation_steps=ACCUM,learning_rate=LR,warmup_steps=8,
        bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        save_strategy='steps',save_steps=32,save_total_limit=2,logging_steps=8,report_to=[],
        seed=SEED,data_seed=SEED,remove_unused_columns=False,dataloader_num_workers=0)
    trainer = Trainer(model=model,args=args,train_dataset=encoded,data_collator=collate,callbacks=[Save()])
    print('WR7_TRAINING_START',json.dumps({'rows':512,'steps':STEPS,'baseline_reproduced':True}),flush=True)
    try:
        result = trainer.train()
    finally:
        trackio.finish()
        upload('evidence/training-metrics.json',metrics)
    after_digest = digest()
    T.verify_training(trainer.state.global_step,float(result.training_loss),before_digest,after_digest)
    saved = checkpoint(work/'candidate','candidate')
    completion = dict(spec,training_completed=True,steps_completed=trainer.state.global_step,
                      training_loss=float(result.training_loss),candidate_commit=saved.oid,adapter_subfolder='candidate',
                      starting_digest=before_digest,candidate_digest=after_digest,evaluation_complete=False)
    upload('evidence/training-complete.json',completion)
    print('WR7_TRAINING_COMPLETE',json.dumps(completion),flush=True)
    model.gradient_checkpointing_disable();model.config.use_cache=True
    candidate_dev = dev_outputs()
    upload('evidence/dev-after.json',candidate_dev)
    after,after_records = T.evaluate(suites,route,generate,G)
    upload('evidence/candidate-744.json',{'scores':after,'records':after_records})
    after_lengths = {'benchmark':W.length_stats([(W.shortening_source(r['row']),r['output'])
                      for r in after_records if W.shortening_source(r['row'])]),
                     'diagnostic':W.length_stats([(r['source'],r['output'])
                      for r in candidate_dev if r['family']=='shortening'])}
    upload('evidence/shortening-after.json',after_lengths)
    report = dict(completion,evaluation_complete=True,before=before,after=after,
                  comparison=T.compare_records(before_records,after_records),shortening_before=before_lengths,shortening_after=after_lengths,manual_review_completed=False)
    upload('evidence/final-report.json',report)
    card = '---\nbase_model: Qwen/Qwen3.5-4B\nlibrary_name: peft\n---\n# Ember Writing Repair 7\n\nExperimental candidate only. Not promoted or deployed.\nSee evidence/final-report.json. Fresh development needs semantic review. Final holdouts remain unused.\n'
    api.upload_file(repo_id=OUT,path_in_repo='README.md',path_or_fileobj=card.encode(),commit_message='Document WR7 experiment')
    print('WR7_FINAL_REPORT',json.dumps(report),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--preflight',action='store_true')
    group.add_argument('--launch',action='store_true')
    args = parser.parse_args()
    prepared = prepare()
    if args.preflight:
        return
    if args.launch:
        launch(prepared)
    else:
        train(prepared)


if __name__ == '__main__':
    main()


