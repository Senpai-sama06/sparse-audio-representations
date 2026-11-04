"""Train the first shared-representation pilot on speaker-disjoint 40 ms modes.

PCA and two conditional neural decoders are compared at explicit latent prefix
sizes. VMD is fixed here. Learned decomposition and RVQ are separate experiments.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import soundfile as sf
import torch
from threadpoolctl import threadpool_limits
from .backends import make_backend
from .learned_data import select_blocks
from .learned_models import SharedRepresentation,normalize
from .learned_payload import write_payload,decode_payload
from .learned_report import report
from .progress import Progress,atomic_json
from .ablation_report import write_table
from util.io import atomic_checkpoint
from util.provenance import file_hash,source_fingerprint,environment


def prepare(root,method,role,rows,progress,completed):
    """Cache complete independent blocks, with residuals explicitly included."""
    target=root/'cache'/f'{method}_{role}.npz';target.parent.mkdir(exist_ok=True)
    selected=[r for r in rows if r['role']==role]
    if target.exists():return completed+len(selected)
    backend=None if method=='direct' else make_backend(method)
    reference=[];components=[];current_path=None
    progress.update(stage='data_preparation',current_job=f'{method}:{role}',frames_total=len(selected),frames_completed=0)
    for i,row in enumerate(selected):
        if row['path']!=current_path:audio,rate=sf.read(row['path'],dtype='float32');current_path=row['path']
        x=audio[row['start']:row['start']+640]
        if len(x)!=640:raise ValueError('Incomplete source block')
        if backend is None:modes=x[None,:]
        else:
            decomposed=backend.decompose(x);modes=np.vstack([decomposed.modes,decomposed.residual]).astype(np.float32)
        reference.append(x.copy());components.append(modes)
        if i%16==0 or i+1==len(selected):progress.update(frames_completed=i+1,jobs_completed=completed+i+1)
    temp=target.with_suffix('.tmp.npz');np.savez(temp,reference=np.stack(reference),components=np.stack(components));temp.replace(target)
    return completed+len(selected)


def dataset(root,method,role):
    with np.load(root/'cache'/f'{method}_{role}.npz') as data:
        return dict(reference=torch.from_numpy(data['reference'].copy()),components=torch.from_numpy(data['components'].copy()))


@torch.no_grad()
def evaluate(model,data,budgets,method,activation,role,step,root,asset,device,basis=None):
    """Reconstruct entire blocks; never report individual mode MSE as block MSE."""
    if model is not None:model.eval()
    raw=data['components'];b,c,n=raw.shape;flat=raw.reshape(-1,n);normalized,means,gains=normalize(flat)
    rows=[]
    for dimension in budgets:
        predictions=[];codes=[]
        for start in range(0,len(flat),64):
            x=normalized[start:start+64].to(device)
            if basis is not None:
                code=x@basis[:,:dimension];shape=code@basis[:,:dimension].T
            else:
                code=model.encode(x,dimension)[:,:dimension]
                shape=model.decoder(torch.nn.functional.pad(code,(0,model.decoder.max_dim-dimension)))
            predictions.append(shape.cpu());codes.append(code.cpu())
        shape=torch.cat(predictions);code=torch.cat(codes)
        reconstructed=(shape*gains+means).reshape(b,c,n).sum(1)
        x=data['reference'];error=(x-reconstructed).square().sum(1);energy=x.square().sum(1);valid=energy>1e-10
        nmse=error[valid]/energy[valid]
        if not len(nmse):raise ValueError('All evaluation blocks are silent')
        spec=(torch.fft.rfft(x).abs()-torch.fft.rfft(reconstructed).abs()).square().sum(1)
        spec_energy=torch.fft.rfft(x).abs().square().sum(1)
        row=dict(method=method,activation=activation,role=role,step=step,latent_dimension=dimension,
                 parameters_per_block=c*(dimension+2),parameter_sample_ratio=c*(dimension+2)/640,
                 mean_nmse=float(nmse.mean()),median_nmse=float(nmse.median()),
                 spectral_nmse=float((spec[valid]/spec_energy[valid].clamp_min(1e-12)).mean()),
                 components=c,silent_blocks=int((~valid).sum()),blocks=b,
                 normalized_component_mse=float((shape-normalized).square().mean()))
        rows.append(row)
        for index in range(min(2,b)):
            prefix=f'{method}_{activation}_{role}_{index}'
            sf.write(root/'examples'/f'{prefix}_reference.wav',x[index].numpy(),16000,subtype='FLOAT')
            payload=root/'examples'/f'{prefix}_z{dimension}.json'
            details=write_payload(payload,asset,code[index*c:(index+1)*c],means[index*c:(index+1)*c],gains[index*c:(index+1)*c])
            if details['parameters']!=row['parameters_per_block']:raise RuntimeError('Code accounting mismatch')
            # All listening output comes from the independently loadable asset
            # and code file, not the in-memory encoder prediction.
            decoded,_=decode_payload(asset,payload)
            if not np.allclose(decoded,reconstructed[index].numpy(),rtol=2e-4,atol=2e-5):raise RuntimeError('Shared decoder round trip mismatch')
            sf.write(root/'examples'/f'{prefix}_z{dimension}_reconstruction.wav',decoded,16000,subtype='FLOAT')
    return rows


def run(corpus_root,output,*,steps=12000,budgets=(4,8,16,32,64),batch_size=32,eval_every=1000,
        seed=42,device='auto',train_speakers=64,eval_speakers=8,train_utterances=8,eval_utterances=4,blocks_per_utterance=8):
    pools=threadpool_limits(limits=1);torch.set_num_threads(1)
    if steps<1 or eval_every<1 or any(k<1 for k in budgets) or max(budgets)>64:raise ValueError('Invalid training budget')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True);(root/'examples').mkdir(exist_ok=True)
    if device=='auto':device='cuda' if torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>1024**3 else 'cpu'
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    settings=dict(corpus_root=str(Path(corpus_root).resolve()),steps=steps,budgets=list(budgets),batch_size=batch_size,
                  eval_every=eval_every,seed=seed,device=device,train_speakers=train_speakers,eval_speakers=eval_speakers,
                  train_utterances=train_utterances,eval_utterances=eval_utterances,blocks_per_utterance=blocks_per_utterance,
                  methods=['direct','vmd'],models=['pca','sine','snakebeta'],width=64,features=32,
                  loss='normalized component MSE + 1e-5 mean squared active code',
                  checkpoint_selection='minimum mean validation block NMSE averaged over all latent budgets',
                  samples=640,sample_rate=16000,quantization=False,rvq=False,learned_decomposition=False,
                  scope='Pilot shared representations; not the complete representation/symbolic search',
                  source=source_fingerprint(Path(__file__).resolve().parents[2]))
    if (root/'manifest.json').exists() and json.loads((root/'manifest.json').read_text())!=settings:raise ValueError('Incompatible resume settings/source')
    atomic_json(root/'manifest.json',settings);atomic_json(root/'environment.json',environment())
    if (root/'data_manifest.json').exists():rows=json.loads((root/'data_manifest.json').read_text())
    else:
        rows=select_blocks(corpus_root,seed,train_speakers,eval_speakers,train_utterances,eval_utterances,blocks_per_utterance)
        atomic_json(root/'data_manifest.json',rows)
    total=2*len(rows)+4*steps+8
    progress=Progress(root,total)
    progress.update(unit_definition='One data block preparation, optimizer step, PCA fit, or final model evaluation; ETA provisional',
                    device=device,training_stage='shared-representation experiment 1')
    history=json.loads((root/'history.json').read_text()) if (root/'history.json').exists() else []
    training=json.loads((root/'training_history.json').read_text()) if (root/'training_history.json').exists() else []
    assets=json.loads((root/'assets.json').read_text()) if (root/'assets.json').exists() else {}
    completed=0
    def persist():
        atomic_json(root/'history.json',history);atomic_json(root/'training_history.json',training);atomic_json(root/'assets.json',assets)
        report(root,history,training,assets)
    persist()
    try:
        for method in settings['methods']:
            for role in ('train','validation','test'):completed=prepare(root,method,role,rows,progress,completed)
        for method in settings['methods']:
            train=dataset(root,method,'train');validation=dataset(root,method,'validation')
            normalized,_,_=normalize(train['components'].reshape(-1,640));normalized=normalized.to(device)
            # PCA is fitted using train only. Test is evaluated once per model.
            folder=root/f'{method}_pca';folder.mkdir(exist_ok=True);asset=folder/'decoder.pt'
            if not (folder/'complete.json').exists():
                progress.update(stage='pca_fit',current_job=f'{method}:pca',frames_total=1,frames_completed=0)
                covariance=normalized.T@normalized/max(1,len(normalized));_,vectors=torch.linalg.eigh(covariance)
                basis=vectors[:,-max(budgets):].flip(1).contiguous()
                atomic_checkpoint(asset,dict(kind='pca',basis=basis.cpu()))
                history.extend(evaluate(None,validation,budgets,method,'pca','validation',0,root,asset,device,basis))
                history.extend(evaluate(None,dataset(root,method,'test'),budgets,method,'pca','test',0,root,asset,device,basis))
                assets[f'{method}/pca']=dict(decoder=str(asset.relative_to(root)),bytes=asset.stat().st_size,shared_learned_values=640*max(budgets))
                atomic_json(folder/'complete.json',dict(status='complete',test_evaluated=True));persist()
            completed+=2;progress.update(jobs_completed=completed)
            for activation in ('sine','snakebeta'):
                folder=root/f'{method}_{activation}';folder.mkdir(exist_ok=True)
                if (folder/'complete.json').exists():completed+=steps+1;progress.update(jobs_completed=completed);continue
                torch.manual_seed(seed)
                if device=='cuda':torch.cuda.manual_seed_all(seed)
                model=SharedRepresentation(activation,max(budgets)).to(device)
                optimizer=torch.optim.Adam(model.parameters(),lr=2e-4)
                first=0;best=float('inf');checkpoint=folder/'last.ckpt'
                if checkpoint.exists():
                    saved=torch.load(checkpoint,map_location=device,weights_only=True);model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
                    first=saved['step'];best=saved['best'];torch.set_rng_state(saved['cpu_rng'].cpu())
                    if device=='cuda':torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
                began=time.perf_counter();base=completed
                for step in range(first+1,steps+1):
                    model.train();indices=torch.randint(len(normalized),(batch_size,),device=device);x=normalized[indices]
                    dimension=budgets[(step-1)%len(budgets)]
                    optimizer.zero_grad(set_to_none=True);prediction,code=model(x,dimension)
                    mse=(prediction-x).square().mean();loss=mse+1e-5*code.square().mean()
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                    if step%100==0 or step==1:
                        training.append(dict(method=method,activation=activation,step=step,dimension=dimension,mse=float(mse.detach()),loss=float(loss.detach())))
                        remaining=max(0,total-(base+step));seconds=(time.perf_counter()-began)/max(1,step-first)
                        progress.update(stage='training',current_job=f'{method}:{activation}',frames_completed=step,frames_total=steps,
                                        jobs_completed=base+step,loss=float(loss.detach()),latent_dimension=dimension,eta_seconds=remaining*seconds)
                        print(f'TRAIN {method} {activation} step={step}/{steps} z={dimension} mse={float(mse.detach()):.6g}',flush=True)
                    if step%eval_every==0 or step==steps:
                        progress.update(stage='validation',current_job=f'{method}:{activation}',frames_completed=step)
                        asset=folder/'decoder_latest.pt'
                        atomic_checkpoint(asset,dict(kind='neural',spec=model.decoder.spec,state={k:v.cpu() for k,v in model.decoder.state_dict().items()}))
                        val=evaluate(model,validation,budgets,method,activation,'validation',step,root,asset,device)
                        score=float(np.mean([r['mean_nmse'] for r in val]));history.extend(val)
                        if score<best:
                            best=score;atomic_checkpoint(folder/'best.ckpt',dict(model=model.state_dict(),step=step,validation_score=score))
                        atomic_checkpoint(checkpoint,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,best=best,
                                                          cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if device=='cuda' else torch.get_rng_state()))
                        assets[f'{method}/{activation}']=dict(shared_decoder_trainable_values=sum(p.numel() for p in model.decoder.parameters()),
                                                             encoder_trainable_values=sum(p.numel() for p in model.encoder.parameters()),
                                                             latest_decoder_bytes=asset.stat().st_size,best_validation_score=best)
                        persist()
                selected=torch.load(folder/'best.ckpt',map_location=device,weights_only=True);model.load_state_dict(selected['model'])
                asset=folder/'decoder_best.pt';atomic_checkpoint(asset,dict(kind='neural',spec=model.decoder.spec,state={k:v.cpu() for k,v in model.decoder.state_dict().items()}))
                # Refresh validation examples from the selected checkpoint; test
                # has not participated in fitting or checkpoint selection.
                history.extend(evaluate(model,validation,budgets,method,activation,'validation',selected['step'],root,asset,device))
                history.extend(evaluate(model,dataset(root,method,'test'),budgets,method,activation,'test',selected['step'],root,asset,device))
                probe=next((root/'examples').glob(f'{method}_{activation}_test_0_z*.json'))
                subprocess.run([sys.executable,'-m','approaches.parametric.learned_payload',str(asset),str(probe),str(folder/'independent_decode.wav')],check=True)
                completed+=steps+1;progress.update(jobs_completed=completed)
                assets[f'{method}/{activation}'].update(decoder=str(asset.relative_to(root)),decoder_bytes=asset.stat().st_size,
                                                       selected_step=selected['step'],independent_decode_verified=True)
                atomic_json(folder/'complete.json',dict(status='complete',selected_step=selected['step'],test_evaluated=True));persist()
                del model,optimizer
            del normalized,train,validation
        progress.update(jobs_completed=total,eta_seconds=0);progress.close('complete');persist()
    except BaseException as error:
        progress.update(error=f'{type(error).__name__}: {error}',failures=progress.state['failures']+1);progress.close('failed');raise
    return history


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--corpus-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=12000);p.add_argument('--eval-every',type=int,default=1000);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--budgets',nargs='+',type=int,default=[4,8,16,32,64]);p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--train-speakers',type=int,default=64);p.add_argument('--eval-speakers',type=int,default=8)
    p.add_argument('--train-utterances',type=int,default=8);p.add_argument('--eval-utterances',type=int,default=4);p.add_argument('--blocks-per-utterance',type=int,default=8)
    run(**vars(p.parse_args()))
