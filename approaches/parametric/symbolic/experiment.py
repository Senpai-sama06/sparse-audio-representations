"""Resumable conditional symbolic regression on fixed 40 ms VMD IMFs.

Stages: verify coefficient fitting; synthesize functions and pretrain expression
generation; adapt through fitted-expression rewards; select on validation only;
compare frozen generators and random search on held-out speakers. Checkpoints,
all candidate formulas, source hashes and per-job results remain in the run.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from ..progress import Progress,atomic_json
from .expressions import evaluate,fit_expression,frontier
from .model import ExpressionGenerator,synthetic_example
from .search import search,reward
from .report import report
from util.io import atomic_checkpoint
from util.provenance import file_hash,source_fingerprint,environment


def known_answers():
    """Audit the continuous fitter separately from discrete formula discovery."""
    t=np.linspace(-1,1,640)
    tokens=['*','+','C','*','C','t','cos','+','+','*','C','t','*','C','square','t','C']
    constants=np.array([.3,.05,50,10,.3]);y=evaluate(tokens,constants,t)
    gain=np.sqrt(np.mean(y*y));initial=constants.copy();initial[:2]/=gain;initial*=1.001
    fit=fit_expression(tokens,y,restarts=3,max_nfev=1600,initial=initial)
    h=1e-6;_,jac=evaluate(tokens,constants,t,True);finite=np.stack([(evaluate(tokens,constants+np.eye(5)[i]*h,t)-evaluate(tokens,constants-np.eye(5)[i]*h,t))/(2*h) for i in range(5)],1)
    result=dict(scope='Known structure checks coefficient fitting and derivatives, not expression discovery.',
                chirped_amfm_nmse=fit.nmse,jacobian_max_error=float(np.max(np.abs(jac-finite))),fit=fit.record())
    if fit.nmse>1e-12 or result['jacobian_max_error']>1e-7:raise AssertionError(result)
    return result


def prepare(source,root,blocks,seed):
    """Reuse frozen sampled VMD targets; exclude residual, retain every mode.

Previous caches have eight float32 modes plus residual per block. Selection is
hash-based before fitting and independent of amplitude or reconstruction score.
"""
    ledger=json.loads((source/'data_manifest.json').read_text());items=[];records=[]
    import hashlib
    for role,count in zip(('train','validation','test'),blocks):
        sources=[r for r in ledger if r['role']==role]
        with np.load(source/'cache'/f'vmd_{role}.npz') as cache:data=cache['components']
        if len(data)!=len(sources) or data.shape[1:]!=(9,640):raise ValueError('Unexpected VMD cache layout')
        indices=sorted(range(len(sources)),key=lambda i:hashlib.sha256(f'{seed}:{role}:{sources[i]["utterance"]}:{sources[i]["start"]}'.encode()).hexdigest())[:count]
        if len(indices)!=count:raise ValueError('Insufficient cached blocks')
        for index in indices:
            for mode in range(8):
                item=f'{role}_b{index}_m{mode}';y=data[index,mode].astype(np.float64)
                # Silence remains visible, but near-zero modes cannot support
                # meaningful relative-error fitting; retain explicit exclusions.
                eligible=bool(np.mean(y*y)>1e-20)
                record=dict(**sources[index],item=item,cache_block=index,mode=mode,eligible=eligible)
                records.append(record)
                if eligible:items.append(dict(item=item,role=role,mode=mode,y=y))
    speakers={role:{r['speaker'] for r in records if r['role']==role} for role in ('train','validation','test')}
    if any(speakers[a]&speakers[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):raise ValueError('Speaker overlap')
    atomic_json(root/'data_manifest.json',dict(records=records,source=str(source),source_manifest_sha256=file_hash(source/'data_manifest.json'),
                caches={role:file_hash(source/'cache'/f'vmd_{role}.npz') for role in speakers},
                selection='Hash-based block selection, all 8 VMD modes; residual excluded; mode mean square <=1e-20 explicitly excluded.'))
    return items


def state(model,optimizer,step,rng,history,**extra):
    return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                training=history,**extra)


def restore(path,model,optimizer,rng):
    saved=torch.load(path,map_location=next(model.parameters()).device,weights_only=False)
    model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);rng.bit_generator.state=saved['rng']
    torch.set_rng_state(saved['torch_rng'].cpu())
    if saved.get('cuda_rng') is not None:torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
    return saved


def summarize(rows,jobs):
    """Aggregate per-IMF best observed fits at declared numerical budgets."""
    result=dict(scope='Conditional expression-generation pilot; finite search, no global optimality claim.',arms={})
    for arm in sorted({r['arm'] for r in rows}):
        group=[r for r in rows if r['arm']==arm];ids=sorted({r['item'] for r in group});entry=dict(evaluated_imfs=len(ids),budgets={})
        for budget in (4,8,16,32):
            errors=[]
            for item in ids:
                fits=[r['nmse'] for r in group if r['item']==item and r['parameters']<=budget]
                if fits:errors.append(min(fits))
            entry['budgets'][str(budget)]=dict(covered_imfs=len(errors),mean_nmse=float(np.mean(errors)) if errors else None,
                                              median_nmse=float(np.median(errors)) if errors else None)
        best=[min(r['nmse'] for r in group if r['item']==item) for item in ids]
        entry['best_any_size_mean_nmse']=float(np.mean(best))
        entry['nmse_thresholds']={}
        for threshold in (.1,.01,.001):
            sizes=[min(r['parameters'] for r in group if r['item']==item and r['nmse']<=threshold) for item in ids if any(r['item']==item and r['nmse']<=threshold for r in group)]
            entry['nmse_thresholds'][str(threshold)]=dict(successful_imfs=len(sizes),total_imfs=len(ids),median_parameters=float(np.median(sizes)) if sizes else None)
        info=[j for j in jobs if j['arm']==arm]
        entry.update(total_proposals=sum(j['proposals'] for j in info),total_unique=sum(j['unique'] for j in info),
                     invalid_fits=sum(j['invalid_fits'] for j in info),search_seconds=sum(j['seconds'] for j in info))
        result['arms'][arm]=entry
    return result


def run(source,output,*,pretrain_steps=1500,adapt_steps=80,blocks=(16,4,8),candidates=48,
        max_nfev=96,restarts=2,seed=43,device='auto',batch_size=24,val_every=20):
    pools=threadpool_limits(limits=1);torch.set_num_threads(1)
    if min(pretrain_steps,adapt_steps,candidates,max_nfev,restarts,batch_size,val_every,*blocks)<1:raise ValueError('All budgets must be positive')
    source=Path(source).resolve();root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True)
    for name in ('jobs','checkpoints','examples'): (root/name).mkdir(exist_ok=True)
    if device=='auto':device='cuda' if torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>1024**3 else 'cpu'
    settings=dict(source=str(source),pretrain_steps=pretrain_steps,adapt_steps=adapt_steps,blocks=list(blocks),candidates=candidates,
                  max_nfev=max_nfev,restarts=restarts,seed=seed,device=device,batch_size=batch_size,val_every=val_every,
                  vocabulary=['t','C','+','-','*','sin','cos','tanh','square'],node_budgets=[7,15,31],constant_bounds=[-2048,2048],
                  training_reward='-log10(max(NMSE,1e-8)) - 0.0005 * complete_description_bits',
                  adaptation='On-policy reward-weighted expression likelihood plus replay of best training expressions.',
                  validation='Mean best expression reward on at most 8 fixed validation IMFs, 12 proposals each; select checkpoint only.',
                  evaluation='Frozen random/pretrained/adapted arms, equal proposals and optimizer budgets, refine two frontier structures.',
                  output='Explicit prefix formula + float64 constants + output gain; no waveform latent decoder.',
                  source_fingerprint=source_fingerprint(Path(__file__).resolve().parents[3]))
    if (root/'manifest.json').exists() and json.loads((root/'manifest.json').read_text())!=settings:raise ValueError('Incompatible resume settings/source')
    atomic_json(root/'manifest.json',settings);atomic_json(root/'environment.json',environment())
    items=prepare(source,root,blocks,seed);train=[x for x in items if x['role']=='train'];val=[x for x in items if x['role']=='validation'];test=[x for x in items if x['role']=='test']
    if not train or not val or not test:raise ValueError('Empty eligible partition')
    total=1+pretrain_steps+adapt_steps+9+3*len(test);progress=Progress(root,total)
    progress.update(device=device,unit_definition='One known-function audit, pretraining update, adaptation update (including fitting), or held-out IMF/arm search. Costs differ substantially.',eta_seconds=None)
    history=[];rows=[];jobs=[];examples=[];completed=0
    def persist():report(root,rows,history,summarize(rows,jobs),examples)
    try:
        atomic_json(root/'known_answers.json',known_answers());completed=1;progress.update(jobs_completed=completed,stage='synthetic_pretraining')
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=ExpressionGenerator().to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-4);folder=root/'checkpoints';latest=folder/'pretraining_latest.ckpt';pretrained=folder/'pretrained.ckpt'
        start=0
        if latest.exists():
            saved=restore(latest,model,optimizer,rng);start=saved['step'];history=saved['training']
        persist()
        for step in range(start+1,pretrain_steps+1):
            model.train();batch=[synthetic_example(rng) for _ in range(batch_size)]
            y=torch.tensor(np.stack([x[0] for x in batch]),device=device);programs=[x[1] for x in batch]
            budgets=torch.tensor([next(b for b in (7,15,31) if b>=len(p)) for p in programs],device=device,dtype=torch.float32)
            optimizer.zero_grad();lp=model.log_prob(y,programs,budgets)
            loss=(-lp/torch.tensor([len(p) for p in programs],device=device)).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite pretraining loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            if step%25==0 or step==pretrain_steps:
                history.append(dict(stage='synthetic_pretraining',step=step,loss=float(loss.detach())))
                progress.update(jobs_completed=completed+step,current_job='synthetic_expression_generation',frames_completed=step,frames_total=pretrain_steps)
                print(f'PRETRAIN {step}/{pretrain_steps} loss={float(loss.detach()):.5f}',flush=True)
            if step%100==0 or step==pretrain_steps:
                atomic_checkpoint(latest,state(model,optimizer,step,rng,history));persist()
        if not pretrained.exists():atomic_checkpoint(pretrained,dict(model=model.state_dict(),step=pretrain_steps))
        completed+=pretrain_steps
        pretrained_model=copy.deepcopy(model).eval()
        pretrained_model.load_state_dict(torch.load(pretrained,map_location=device,weights_only=False)['model'])
        optimizer=torch.optim.Adam(model.parameters(),lr=5e-5);latest=folder/'adaptation_latest.ckpt';best_path=folder/'adapted_best.ckpt'
        start=0;best=-float('inf');replay={}
        if latest.exists():
            saved=restore(latest,model,optimizer,rng);start=saved['step'];history=saved['training'];best=saved['best'];replay=saved['replay']
        adaptation_started=time.monotonic()
        for step in range(start+1,adapt_steps+1):
            progress.update(stage='speech_adaptation',current_job=f'adaptation:{step}',frames_completed=0,frames_total=16,jobs_completed=completed+step-1)
            signals=[];programs=[];budgets=[];weights=[];rewards=[]
            for index in rng.choice(len(train),size=min(2,len(train)),replace=False):
                item=train[int(index)];y=item['y'];normal=y/max(float(np.sqrt(np.mean(y*y))),1e-12)
                outcome=search(y,model,seed=int(rng.integers(2**31)),candidates=8,restarts=restarts,max_nfev=max_nfev,refine=0)
                candidates_fit=outcome['records']
                if not candidates_fit:continue
                scores=np.array([reward(r) for r in candidates_fit]);center=float(np.median(scores));rewards.extend(scores.tolist())
                for row,score in zip(candidates_fit,scores):
                    signals.append(normal);programs.append(row['tokens']);budgets.append(row['budget'])
                    weights.append(float(np.clip(score-center,-2,2)))
                champion=max(candidates_fit,key=reward)
                if item['item'] not in replay or reward(champion)>reward(replay[item['item']]):replay[item['item']]=champion
                elite=replay[item['item']];signals.append(normal);programs.append(elite['tokens']);budgets.append(elite['budget']);weights.append(.25)
                progress.update(frames_completed=len(rewards))
            if not programs:raise RuntimeError('No valid expression fits in adaptation batch')
            model.train();optimizer.zero_grad()
            lp=model.log_prob(torch.tensor(np.stack(signals),device=device,dtype=torch.float32),programs,torch.tensor(budgets,device=device,dtype=torch.float32))
            loss=-(lp*torch.tensor(weights,device=device,dtype=torch.float32)).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite adaptation loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step()
            history.append(dict(stage='speech_adaptation',step=step,loss=float(loss.detach()),mean_reward=float(np.mean(rewards)),replay_items=len(replay)))
            print(f'ADAPT {step}/{adapt_steps} reward={np.mean(rewards):.5f} replay={len(replay)}',flush=True)
            if step%val_every==0 or step==adapt_steps:
                scores=[];indices=np.linspace(0,len(val)-1,min(8,len(val))).astype(int)
                for i,index in enumerate(indices):
                    progress.update(stage='validation_search',current_job=f'validation:{step}:{i}',frames_completed=i,frames_total=len(indices))
                    outcome=search(val[index]['y'],model,seed=seed+800000+i,candidates=12,restarts=restarts,max_nfev=max_nfev,refine=0)
                    scores.append(max([reward(r) for r in outcome['records']],default=-10.))
                score=float(np.mean(scores));history.append(dict(stage='validation',step=step,mean_reward=score))
                if score>best:best=score;atomic_checkpoint(best_path,dict(model=model.state_dict(),step=step,validation_reward=score))
                print(f'VALIDATION step={step} reward={score:.5f} best={best:.5f}',flush=True)
            atomic_checkpoint(latest,state(model,optimizer,step,rng,history,best=best,replay=replay))
            seconds_per_step=(time.monotonic()-adaptation_started)/(step-start)
            estimate=seconds_per_step*(adapt_steps-step+3*len(test)*candidates/16)
            atomic_json(root/'replay.json',replay);progress.update(jobs_completed=completed+step,eta_seconds=estimate,
                eta_note='Provisional extrapolation from adaptation fits; validation, report and refinement costs vary.');persist()
        selected=torch.load(best_path,map_location=device,weights_only=False);model.load_state_dict(selected['model']);model.eval();completed+=adapt_steps
        atomic_json(root/'selected_checkpoint.json',dict(step=selected['step'],validation_reward=selected['validation_reward'],checkpoint_sha256=file_hash(best_path)))
        # Unlike the initial fitter audit, these controls require discovery of
        # the structure. Known formulas are only used to produce test samples.
        t=np.linspace(-1,1,640)
        controls=dict(quadratic=.2+.1*t+.3*t*t,sinusoid=.3*np.sin(200*t+.3),
                      chirped_amfm=(.3+.05*t)*np.cos(50*t+10*t*t+.3))
        discovery_checks=[]
        for arm,proposer in [('random',None),('pretrained',pretrained_model),('adapted',model)]:
            for index,(name,y) in enumerate(controls.items()):
                path=root/'jobs'/f'known_discovery_{arm}_{name}.json'
                progress.update(stage='known_function_discovery',current_job=f'{arm}:{name}',jobs_completed=completed)
                if path.exists():outcome=json.loads(path.read_text())
                else:
                    outcome=search(y,proposer,seed=seed+700000+index,candidates=candidates,restarts=restarts,max_nfev=max_nfev)
                    atomic_json(path,outcome)
                best_fit=min(outcome['records'],key=lambda r:r['nmse']) if outcome['records'] else None
                discovery_checks.append(dict(arm=arm,function=name,best=best_fit,proposals=outcome['proposals'],unique=outcome['unique']))
                atomic_json(root/'synthetic_discovery.json',discovery_checks);completed+=1;progress.update(jobs_completed=completed)
        evaluation_started=time.monotonic();evaluations_finished=0
        for arm,proposer in [('random',None),('pretrained',pretrained_model),('adapted',model)]:
            for index,item in enumerate(test):
                path=root/'jobs'/f'{arm}_{item["item"]}.json'
                progress.update(stage='heldout_expression_search',current_job=f'{arm}:{item["item"]}',frames_completed=index,frames_total=len(test),jobs_completed=completed)
                if path.exists():outcome=json.loads(path.read_text())
                else:
                    started=time.monotonic();outcome=search(item['y'],proposer,seed=seed+900000+index,candidates=candidates,restarts=restarts,max_nfev=max_nfev)
                    outcome.update(seconds=time.monotonic()-started,arm=arm,item=item['item'],mode=item['mode'])
                    atomic_json(path,outcome)
                jobs.append(outcome)
                # Track two distinct Pareto relations: constants alone and the
                # complete formula description. Neither is silently substituted.
                front={(tuple(r['tokens']),tuple(r['constants'])) for r in outcome['frontier']}
                for r in outcome['records']:
                    on_parameters=not any(s['parameters']<=r['parameters'] and s['nmse']<=r['nmse'] and (s['parameters']<r['parameters'] or s['nmse']<r['nmse']) for s in outcome['records'])
                    rows.append(dict(**r,arm=arm,item=item['item'],mode=item['mode'],on_frontier=(tuple(r['tokens']),tuple(r['constants'])) in front,on_parameter_frontier=on_parameters))
                if index<8 and outcome['frontier']:
                    chosen=sorted(outcome['frontier'],key=lambda r:r['nmse'])[:2]
                    examples.append((f'{arm}_{item["item"]}',item['y'],[dict(**r,arm=arm) for r in chosen]))
                completed+=1;evaluations_finished+=1
                estimate=(time.monotonic()-evaluation_started)/evaluations_finished*(3*len(test)-evaluations_finished)
                progress.update(jobs_completed=completed,eta_seconds=estimate)
                print(f'TEST {arm} {index+1}/{len(test)} best_nmse={min((r["nmse"] for r in outcome["records"]),default=None)}',flush=True)
                persist()
        # Verify one representative expression in a clean process; no neural
        # model, original samples or optimization state is passed to evaluator.
        import subprocess,sys
        payload=next((root/'examples').glob('*.json'))
        subprocess.run([sys.executable,'-m','approaches.parametric.symbolic.payload',str(payload),str(root/'independent_expression.wav')],check=True)
        from .payload import reconstruct
        import soundfile as sf
        expected,rate=reconstruct(payload);observed,observed_rate=sf.read(root/'independent_expression.wav')
        if rate!=observed_rate or not np.allclose(expected,observed,rtol=1e-6,atol=1e-8):raise AssertionError('Independent expression evaluation mismatch')
        atomic_json(root/'independent_evaluation.json',dict(verified=True,payload=str(payload.relative_to(root)),
                    payload_sha256=file_hash(payload),max_float32_difference=float(np.max(np.abs(expected-observed)))))
        progress.update(jobs_completed=total,eta_seconds=0);progress.close('complete');persist()
        return root
    except BaseException as error:
        progress.update(error=f'{type(error).__name__}: {error}',failures=1);progress.close('failed');persist();raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--pretrain-steps',type=int,default=1500);p.add_argument('--adapt-steps',type=int,default=80)
    p.add_argument('--blocks',type=int,nargs=3,default=(16,4,8));p.add_argument('--candidates',type=int,default=48)
    p.add_argument('--max-nfev',type=int,default=96);p.add_argument('--restarts',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=24);p.add_argument('--val-every',type=int,default=20)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');p.add_argument('--seed',type=int,default=43)
    run(**vars(p.parse_args()))


if __name__=='__main__':main()
