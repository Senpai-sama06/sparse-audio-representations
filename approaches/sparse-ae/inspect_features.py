"""Run the fixed exploratory L1/dense channel inspection, then build its report."""
import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import soundfile as sf
import torch

from sae.interpret import (FEATURES, acoustic_features, channel_statistics,
                           decode_intervention, select_channels, select_examples)
from sae.metrics import audio_metrics
from sae.models import AudioSAE, ModelConfig

REPO = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_audio(path, wave):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wave, 16000, subtype='PCM_16')
    decoded, rate = sf.read(path, dtype='float32')
    assert rate == 16000 and decoded.shape == wave.shape and np.isfinite(decoded).all()
    return decoded, int(np.sum((wave < -1) | (wave >= 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=REPO/'runs/sparse_ae_local_audit_20261110')
    parser.add_argument('--output', type=Path, default=REPO/'runs/sparse_ae_features_20261110')
    args = parser.parse_args()
    root, out = args.inputs.resolve(), args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Output exists; choose an empty directory')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    cases = json.loads((root/'native_manifest.json').read_text())
    assert len(cases) == len({r['speaker'] for r in cases}) == 40
    checkpoints = {m:root/f'campaign/jobs/{m}/attempt_01/best_weights.pt' for m in ('l1','dense')}
    for method in checkpoints:
        selected = json.loads((checkpoints[method].parent/'data_selection.json').read_text())
        used = set(sum([selected[k] for k in ('train_ids','validation_ids','calibration_ids')], []))
        assert not used.intersection(r['id'] for r in cases)
    record = dict(run_id=out.name, code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_hashes={str(p.relative_to(REPO)):digest(p) for p in [Path(__file__), REPO/'approaches/sparse-ae/sae/interpret.py', REPO/'approaches/sparse-ae/sae/models.py', REPO/'approaches/sparse-ae/sae/metrics.py']},
        packages={p:importlib.metadata.version(p) for p in ('torch','numpy','scipy','soundfile','pesq','pystoi','matplotlib')},
        protocol=dict(stage='exploration', sample_rate=16000, seed=0, channels_per_model=8,
            examples_per_channel=3, crop_samples=16384, feature_names=FEATURES,
            selection='First eight SHA256(sae-features-v1:<method>:<channel>) channels with activity fraction from 0.01 through 0.99. Three strongest activations on distinct speakers per channel.',
            timing='Latent hop 256 samples (16 ms), nominal center 256*t+128; encoder receptive field 1276 samples (79.75 ms). Acoustic proxies use 1024-sample Hann windows; boundary and RMS<1e-4 frames are excluded from correlations.',
            association='Descriptive Pearson r across pooled frames. Other proxies control linearly for log10 RMS; RMS correlation is unadjusted. Pitch uses autocorrelation confidence >=0.6 over 50-400 Hz. Per-speaker r and sign agreement are also saved. No p-values or confirmatory labels.',
            sample_format='Mono PCM16 WAV; metrics calculated after decoding these exact WAV bytes. No gain adjustment, normalization, alignment or resampling.',
            ablation='Set one selected channel to zero at all time steps. Decode full-utterance latents, then crop 1.024 s around the selected peak. Control uses the unchanged code. No re-encoding.',
            limits='All 40 speakers appeared in development. No independent speaker-held-out confirmation. No phoneme annotations. Channels are not matched across independently trained models. Selected peak examples are not representative average ablation effects.'),
        cases=cases, models={}, baseline_cases=[], interventions=[], checks={})
    # Freeze case group and protocol before model inference.
    (out/'protocol.json').write_text(json.dumps(record,indent=2))
    waves=[]
    for case in cases:
        assert digest(case['path']) == case['sha256']
        x,sr=sf.read(case['path'],dtype='float32')
        assert sr==16000 and x.ndim==1
        x, clipped=save_audio(out/'audio'/f"{case['id']}_source.wav",x)
        assert clipped==0
        waves.append(x)
        record['baseline_cases'].append(dict(id=case['id'],speaker=case['speaker'],samples=len(x),start_sample=0,
            source_audio=f"audio/{case['id']}_source.wav",models={}))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    record['device']=device
    start=time.time()
    for method, checkpoint in checkpoints.items():
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert payload['model_config']['method']==method
        model=AudioSAE(ModelConfig(**payload['model_config']))
        model.load_state_dict(payload['model_state'],strict=True)
        model.to(device).eval()
        codes, acoustic, reconstruction=[],[],[]
        sham_max=0.0
        for i,(case,x) in enumerate(zip(cases,waves)):
            with torch.inference_mode():
                result=model(torch.from_numpy(x)[None,None].to(device),torch.tensor([len(x)],device=device))
                code=result['z']
                sham=decode_intervention(model,code,0,len(x),factor=1)
                sham_max=max(sham_max,float((sham-result['reconstruction']).abs().max()))
            assert np.all(result['valid'].cpu().numpy()), 'Unexpected padded latent steps'
            z=code[0].cpu().numpy(); codes.append(z)
            features,_=acoustic_features(x,len(z)); acoustic.append(features)
            y=result['reconstruction'][0,0].cpu().numpy()
            path=f"audio/{case['id']}_{method}.wav"
            y,clipped=save_audio(out/path,y); reconstruction.append(y)
            record['baseline_cases'][i]['models'][method]=dict(audio=path,clipped_samples=clipped,
                active_coefficients=int((z!=0).sum()),**audio_metrics(x,y,speech=True))
            print(f'{method} activations {i+1}/40',flush=True)
        assert sham_max<1e-6
        stats=channel_statistics(codes,acoustic)
        selected=select_channels(stats,method)
        summary=json.loads((checkpoint.parent/'summary.json').read_text())
        record['models'][method]=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
            best_epoch=summary['best_epoch'],selection_rule='Minimum validation total loss including native penalties',
            channels=stats,selected_channels=selected,sham_max_abs_error=sham_max,
            active_fraction=float(np.mean(np.concatenate(codes)!=0)))
        np.savez_compressed(out/f'{method}_activations.npz',**{case['id']:z for case,z in zip(cases,codes)})
        for channel in selected:
            for rank,(peak,index,frame) in enumerate(select_examples(codes,channel),1):
                case,x=cases[index],waves[index]
                code=torch.from_numpy(codes[index])[None].to(device)
                ablated=decode_intervention(model,code,channel,len(x))[0,0].cpu().numpy()
                center=frame*256+128
                crop_start=max(0,min(len(x)-16384,center-8192))
                crop_end=crop_start+min(16384,len(x))
                key=f'{method}_ch{channel:03d}_rank{rank}'
                source,_=save_audio(out/'audio'/f'{key}_source.wav',x[crop_start:crop_end])
                before,_=save_audio(out/'audio'/f'{key}_control.wav',reconstruction[index][crop_start:crop_end])
                after,clipped=save_audio(out/'audio'/f'{key}_ablated.wav',ablated[crop_start:crop_end])
                delta=after.astype('float64')-before
                effect=np.square(delta)
                window_energy=effect.reshape(-1,256).sum(axis=1) if len(effect)%256==0 else np.array([effect.sum()])
                concentrated=float(np.sort(window_energy)[-max(1,int(np.ceil(.1*len(window_energy)))):].sum()/max(effect.sum(),1e-30))
                relative=float(effect.sum()/max(np.square(before.astype('float64')).sum(),1e-30))
                trace=codes[index][:,channel]
                times=(np.arange(len(trace))*256+128)/16000-crop_start/16000
                mask=(times>=0)&(times< len(source)/16000)
                record['interventions'].append(dict(key=key,method=method,channel=channel,rank=rank,
                    id=case['id'],speaker=case['speaker'],start_sample=crop_start,samples=len(source),
                    peak_frame=frame,peak_activation=peak,trace_times=times[mask].tolist(),trace_values=trace[mask].tolist(),
                    source_audio=f'audio/{key}_source.wav',after_audio=f'audio/{key}_ablated.wav',
                    control_audio=f'audio/{key}_control.wav',clipped_samples=clipped,
                    after=audio_metrics(source,after,speech=True),control=audio_metrics(source,before,speech=True),
                    change_energy_relative_to_control=relative,change_energy_top_10_percent_frames=concentrated))
            print(f'{method} ablated channel {channel}: 3/3',flush=True)
        (out/'analysis.json').write_text(json.dumps(record,indent=2,allow_nan=False))
    record['elapsed_seconds']=time.time()-start
    record['checks']=dict(selected_baseline=40,completed_baseline=len(record['baseline_cases']),
        selected_interventions=48,completed_interventions=len(record['interventions']),failed=0,excluded=0,
        source_hashes_verified=True,utterance_selection_overlap=False,
        speaker_held_out=False,sham_decode_passed=True)
    (out/'analysis.json').write_text(json.dumps(record,indent=2,allow_nan=False))
    print('Completed',out/'analysis.json',flush=True)


if __name__=='__main__':
    main()
