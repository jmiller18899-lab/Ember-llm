"""Training-only CPU gradient and isolated AdamW balance diagnostic."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from jobs import ember_generated_closure as g
trace,diag,objectives,base=g.trace,g.diag,g.objectives,g.base
SNAPSHOT=ROOT/'config/ember_balance_training_snapshot.json'
SNAPSHOT_SHA='b78fed0f1eadb4e518b84db121d266fd0f4412fd39832f975a09cb0aadf06b86'
STEPS={1,13,23,40}
WEIGHTS=(0.,.025,.05,.10,.20)
LOSS_TOL=2e-6


def optimizer_digest(optimizer):
    h=hashlib.sha256()
    def visit(x):
        if hasattr(x,'detach'):
            h.update(str((x.dtype,tuple(x.shape))).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for key in sorted(x,key=str): h.update(str(key).encode()); visit(x[key])
        elif isinstance(x,(list,tuple)):
            for value in x: visit(value)
        else: h.update(repr(x).encode())
    visit(optimizer.state_dict())
    return h.hexdigest()


def stable_dot(a,b,torch):
    # Float32 reductions over millions of tiny parameter deltas lose precision.
    # Accumulate bounded chunks in float64; optimizer math stays unchanged.
    total=0.
    for start in range(0,a.numel(),1048576):
        total+=float(torch.dot(a[start:start+1048576].double(),b[start:start+1048576].double()))
    return total


def stable_norm(a,torch):
    return math.sqrt(max(0.,stable_dot(a,a,torch)))


def cosine(a,b,torch):
    denominator=stable_norm(a,torch)*stable_norm(b,torch)
    return stable_dot(a,b,torch)/denominator if denominator>1e-20 else None


def assign_gradient(model,vector):
    offset=0
    for p in model.parameters():
        n=p.numel();p.grad=vector[offset:offset+n].reshape_as(p).clone();offset+=n
    if offset != vector.numel(): raise ValueError('gradient vector mismatch')


def new_optimizer(model,torch,cfg,state=None):
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['control_learning_rate'],weight_decay=cfg['weight_decay'])
    if state is not None: opt.load_state_dict(copy.deepcopy(state))
    return opt


def component_losses(model,teacher,tokenizer,torch,cfg,examples,tools,copies,batch,structural):
    # Separate forward graphs permit independent gradients without graph retention.
    return {
        'placement':lambda:objectives.batch_loss(model,torch,examples,batch['placement']),
        'tool_kl':lambda:trace.distill.batch_teacher_kl(model,teacher,tokenizer,torch,tools,batch['tool'],cfg['distill_temperature']),
        'copy_kl':lambda:trace.distill.batch_teacher_kl(model,teacher,tokenizer,torch,copies,batch['copy'],cfg['distill_temperature']),
        'closing':lambda:objectives.batch_loss(model,torch,structural,list(range(len(structural))))}


def recommend(observations):
    """Prespecified local screen, not an evaluation or promotion gate."""
    eligible=[]
    for weight in WEIGHTS[1:]:
        passed=True
        for obs in observations:
            zero=next(t for t in obs['trials'] if t['weight']==0)
            trial=next(t for t in obs['trials'] if t['weight']==weight)
            gain=zero['loss_decrease']['placement']
            passed &= (gain>LOSS_TOL and trial['loss_decrease']['placement']>=.9*gain
                       and trial['loss_decrease']['closing']>=-LOSS_TOL
                       and trial['after']['tool_kl']<=.08 and trial['after']['copy_kl']<=.08)
        if passed: eligible.append(weight)
    complete=[r['step'] for r in observations]==sorted(STEPS)
    return {'criterion':'At all four states, at least 90% of zero-weight local placement loss decrease, no closing loss increase beyond 2e-6, both batch KL losses <=0.08.',
            'complete':complete,'eligible_weights':eligible if complete else [],
            'suggested_weight_for_next_full_test':min(eligible) if complete and eligible else None,
            'validated_training_balance':False}


def inspect_state(student,optimizer,teacher,tokenizer,torch,cfg,examples,snapshot,batch,structural,step):
    live_hash=trace.state_digest(student); opt_hash=optimizer_digest(optimizer)
    rng=torch.get_rng_state().clone()
    probe=copy.deepcopy(student).eval()
    state=copy.deepcopy(optimizer.state_dict())
    functions=component_losses(probe,teacher,tokenizer,torch,cfg,examples,snapshot['tools'],snapshot['copies'],batch,structural)
    gradients={}; before={}
    for name,fn in functions.items():
        loss=fn()
        if not bool(torch.isfinite(loss)): raise ValueError('nonfinite component loss')
        before[name]=float(loss.detach());gradients[name]=diag.gradient_vector(loss,probe,torch)
    gp,gt,gc,gs=(gradients[k] for k in ['placement','tool_kl','copy_kl','closing'])
    weighted={'placement':cfg['placement_loss_weight']*gp,'tool_kl':cfg['tool_kl_loss_weight']*gt,
              'copy_kl':cfg['copy_kl_loss_weight']*gc,'closing':g.CLOSURE_WEIGHT*gs}
    base_grad=weighted['placement']+weighted['tool_kl']+weighted['copy_kl']
    row={'step':step,'before':before,'raw_gradient_norms':{k:stable_norm(v,torch) for k,v in gradients.items()},
         'weighted_gradient_norms_at_020':{k:stable_norm(v,torch) for k,v in weighted.items()},
         'placement_closing_cosine':cosine(gp,gs,torch),
         'base_closing_cosine':cosine(base_grad,gs,torch),'trials':[]}
    theta=diag.flat_parameters(student,torch)
    zero_delta=None
    for weight in WEIGHTS:
        probe.load_state_dict(student.state_dict())
        po=new_optimizer(probe,torch,cfg,state)
        if optimizer_digest(po)!=opt_hash: raise ValueError('shadow optimizer history mismatch')
        combined=base_grad+weight*gs
        norm=stable_norm(combined,torch)
        assign_gradient(probe,combined)
        clip_norm=float(torch.nn.utils.clip_grad_norm_(probe.parameters(),cfg['gradient_clip'],error_if_nonfinite=True))
        po.step();probe.zero_grad(set_to_none=True)
        delta=diag.flat_parameters(probe,torch)-theta
        dn=stable_norm(delta,torch)
        if not math.isfinite(dn) or dn<=0: raise ValueError('invalid trial update')
        with torch.no_grad(): after={k:float(fn()) for k,fn in functions.items()}
        if not all(math.isfinite(x) for x in after.values()): raise ValueError('invalid trial loss')
        if zero_delta is None: zero_delta=delta.clone()
        matched_zero=zero_delta*(dn/stable_norm(zero_delta,torch))
        trial={'weight':weight,'combined_gradient_norm':norm,
               'optimizer_reported_gradient_norm':clip_norm,
               'clip_multiplier':min(1.,cfg['gradient_clip']/(clip_norm+1e-6)),
               'actual_update_l2':dn,'update_cosine_to_zero':cosine(delta,zero_delta,torch),
               'update_norm_ratio_to_zero':dn/stable_norm(zero_delta,torch),
               'predicted_loss_change':{k:stable_dot(v,delta,torch) for k,v in gradients.items()},
               'norm_matched_zero_predicted_loss_change':{k:stable_dot(v,matched_zero,torch) for k,v in gradients.items()},
               'after':after,'loss_decrease':{k:before[k]-after[k] for k in before}}
        row['trials'].append(trial)
        diag.event('balance_trial',step=step,weight=weight,placement_decrease=trial['loss_decrease']['placement'],closing_decrease=trial['loss_decrease']['closing'],update_cosine=trial['update_cosine_to_zero'])
        del po,combined,delta,matched_zero
    del probe,state,gradients,weighted,base_grad,theta,zero_delta,gp,gt,gc,gs
    torch.set_rng_state(rng)
    row['live_model_unchanged']=trace.state_digest(student)==live_hash
    row['live_optimizer_unchanged']=optimizer_digest(optimizer)==opt_hash
    if not row['live_model_unchanged'] or not row['live_optimizer_unchanged']: raise ValueError('diagnostic perturbed live replay')
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('training-balance-results'))
    output=parser.parse_args().output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    import torch
    cfg=trace.load_config();torch.set_num_threads(2);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True)
    raw=SNAPSHOT.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SNAPSHOT_SHA: raise ValueError('training snapshot changed')
    snapshot=json.loads(raw)
    if set(snapshot)-{'source','source_state_sha256','source_run','source_data_sha256','train_values','template','tools','copies','schedule','anchors','expected_updates','expected_refreshes'}: raise ValueError('unexpected snapshot content')
    report={'status':'ERROR','code_commit':os.getenv('GITHUB_SHA'),'snapshot_sha256':SNAPSHOT_SHA,'source':snapshot['source'],
            'diagnostic_steps':sorted(STEPS),'weights':list(WEIGHTS),'observations':[],'updates':[],'refreshes':[],
            'heldout_evaluations':0,'checkpoint_exported':False}
    started=time.monotonic();student=teacher=pristine=None
    def timeout(*args): raise TimeoutError('30-minute CPU diagnostic bound exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1800)
    try:
        with tempfile.TemporaryDirectory(prefix='ember-balance-') as td:
            student,tokenizer,source,splits,source_ref=base.load_inputs(json.loads(base.DEFAULT_CONFIG.read_text()),Path(td),torch)
            student.to('cpu').eval()
            if source_ref!=snapshot['source'] or trace.state_digest(student)!=snapshot['source_state_sha256'] or student.cfg.dropout!=0: raise ValueError('source mismatch')
            pristine=copy.deepcopy(student.state_dict());teacher=copy.deepcopy(student).eval()
            for p in teacher.parameters():p.requires_grad_(False)
            del source,splits
            train=g.data.v048d.build_cases({k:snapshot['train_values'][k] for k in sorted(g.data.PLACEMENT_SUBTYPES)},'v051_place_train')
            tid,_=objectives.token_contract(tokenizer)
            examples=[objectives.supervised_example(tokenizer,c,'placement',snapshot['template'],tid) for c in train]
            schedule=trace.batch_schedule(cfg,len(examples),len(snapshot['tools']),len(snapshot['copies']))
            if schedule!=snapshot['schedule']:raise ValueError('original batch schedule mismatch')
            optimizer=new_optimizer(student,torch,cfg)
            for step,batch in enumerate(schedule,1):
                if step in g.REFRESH_STEPS:
                    structural,audit=g.refresh_examples(student,tokenizer,torch,snapshot['anchors'],step)
                    expected=next(x for x in snapshot['expected_refreshes'] if x['step']==step)
                    if audit!=expected:raise ValueError(f'generated-prefix replay mismatch at step {step}')
                    report['refreshes'].append(audit)
                if step in STEPS:
                    row=inspect_state(student,optimizer,teacher,tokenizer,torch,cfg,examples,snapshot,batch,structural,step)
                    report['observations'].append(row)
                losses=g.optimizer_step(student,teacher,tokenizer,torch,cfg,examples,snapshot['tools'],snapshot['copies'],batch,optimizer,structural)
                expected=snapshot['expected_updates'][step-1]['losses']
                errors={k:abs(v-expected[k]) for k,v in losses.items()}
                if any(v>2e-5 for v in errors.values()):raise ValueError(f'original optimization replay mismatch at {step}: {errors}')
                report['updates'].append({'step':step,'losses':losses,'maximum_replay_error':max(errors.values())})
                diag.write_json(output/'report.json',report)
            report['recommendation']=recommend(report['observations']);report['status']='COMPLETE'
    except Exception as exc:
        report['error']={'type':type(exc).__name__,'message':str(exc)}
        raise
    finally:
        signal.alarm(0)
        if student is not None and pristine is not None:
            student.load_state_dict(pristine)
            report['source_restored']=trace.state_digest(student)==snapshot['source_state_sha256']
            report['teacher_unchanged']=teacher is not None and trace.state_digest(teacher)==snapshot['source_state_sha256']
            if not report['source_restored'] or not report['teacher_unchanged']:report['status']='ERROR'
        report['elapsed_seconds']=time.monotonic()-started;diag.write_json(output/'report.json',report)
        lines=['# Ember training balance diagnostic','',f"Status: {report['status']}",'',json.dumps(report.get('recommendation',report.get('error')),indent=2),
               '', 'Single-update training-loss measurements along the existing 0.20 trajectory; no held-out evaluation or checkpoint export.']
        (output/'summary.md').write_text('\n'.join(lines)+'\n')
    diag.event('complete',status=report['status'],recommendation=report.get('recommendation'),elapsed_seconds=report['elapsed_seconds'])
    return 0 if report['status']=='COMPLETE' else 1

if __name__=='__main__':raise SystemExit(main())
