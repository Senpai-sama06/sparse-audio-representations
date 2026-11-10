"""Independently score imported SAE checkpoints; never downloads or trains models.

Run after importing the campaign, references and native_manifest.json under --root.
PESQ/STOI use their installed reference packages directly. SI-SDR uses least
squares projection, independently of sae.metrics. Historical results stay intact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from pesq import pesq
from pystoi import stoi
from scipy.signal import resample_poly
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'approaches/sparse-ae'))
from sae.models import AudioSAE, ModelConfig
from sae.metrics import audio_metrics

KEYS = ('mse', 'nmse', 'si_sdr', 'pesq_wb', 'stoi')
METHODS = ('l1', 'kl', 'topk', 'jumprelu', 'gated', 'batchtopk', 'dense')


def independent(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    assert x.shape == y.shape and x.ndim == 1 and np.isfinite([x, y]).all()
    xc, yc = x - x.mean(), y - y.mean()
    projection = xc * np.linalg.lstsq(xc[:, None], yc, rcond=None)[0][0]
    return dict(mse=float(np.mean(np.square(x-y))),
                nmse=float(np.mean(np.square(x-y)) / np.mean(x*x)),
                si_sdr=float(10*np.log10(max(np.sum(projection**2), 1e-30) /
                                        max(np.sum((yc-projection)**2), 1e-30))),
                pesq_wb=float(pesq(16000, x, y, 'wb')),
                stoi=float(stoi(x, y, 16000, extended=False)))


def mean(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in KEYS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO/'runs/sparse_ae_local_audit_20261110')
    parser.add_argument('--report-only', action='store_true', help='Render existing audit results without recomputing')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.report_only:
        write_report(json.loads((root/'independent_audit.json').read_text()), root)
        return
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    historical = json.loads((REPO/'approaches/sparse-ae/reports/sparse_ae_pitchflower_best_metrics.json').read_text())
    cases = json.loads((root/'references/cases.json').read_text())
    output = {'device': device, 'torch': torch.__version__, 'independent_historical': {},
              'native_full_utterances': {}, 'checkpoint_hash_matches': {},
              'metric_implementation_max_errors': {k: 0.0 for k in KEYS},
              'native_limit': '40 unused dev-clean utterances, one per speaker; speakers were used for validation/calibration. Not a speaker-held-out test.',
              'selection': 'Minimum SHA256(sae-audit-20261110:<id>) per speaker after excluding all seven models training/validation/calibration selections and report cases.'}
    for method in METHODS:
        ckpt = root/f'campaign/jobs/{method}/attempt_01/best_weights.pt'
        checksum = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        output['checkpoint_hash_matches'][method] = checksum == historical['sae_checkpoints'][method]['sha256']
        rows = []
        for i, case in enumerate(cases):
            x, rate = sf.read(root/'references'/case['path'], dtype='float32')
            filename = f"{i:02d}_{case['id'].replace(':', '_')}_reconstruction.wav"
            y, yr = sf.read(root/'local_inference'/method/filename, dtype='float32')
            assert rate == yr == 16000
            scores = independent(x, y)
            existing = audio_metrics(x, y, speech=True)
            for k in KEYS:
                output['metric_implementation_max_errors'][k] = max(output['metric_implementation_max_errors'][k], abs(existing[k]-scores[k]))
            rows.append(dict(id=case['id'], **scores))
        saved = {r['id']: r['systems'][method] for r in historical['case_scores']}
        output['independent_historical'][method] = dict(mean=mean(rows), cases=rows,
            max_report_difference={k:max(abs(r[k]-saved[r['id']][k]) for r in rows) for k in KEYS})
        print('Independent historical scoring:', method, mean(rows), flush=True)
    # Score PitchFlower from its original 24 kHz embedded PCM source, without alignment or gain fitting.
    rows = []
    for case in cases:
        x, sr = sf.read(root/'references'/case['reference_24k_path'], dtype='float32')
        y, yr = sf.read(root/'references'/case['pitchflower_path'], dtype='float32')
        assert sr == yr == 24000
        x, y = resample_poly(x, 2, 3).astype('float32'), resample_poly(y, 2, 3).astype('float32')
        supplied, _ = sf.read(root/'references'/case['path'], dtype='float32')
        assert np.array_equal(x, supplied), 'SAE input differs from original reference resampling'
        rows.append(dict(id=case['id'], **independent(x, y)))
    output['pitchflower_independent'] = dict(mean=mean(rows), cases=rows)
    # Identity and deliberately degraded controls exercise score direction on real audio.
    rng = np.random.default_rng(0)
    output['controls'] = {name:independent(x, y) for name,y in {
        'identity':x, 'additive_noise':x+rng.normal(0, np.std(x), len(x)),
        'permuted_samples':rng.permutation(x)}.items()}
    native = json.loads((root/'native_manifest.json').read_text())
    for method in METHODS:
        payload = torch.load(root/f'campaign/jobs/{method}/attempt_01/best_weights.pt', map_location='cpu', weights_only=True)
        model = AudioSAE(ModelConfig(**payload['model_config']))
        model.load_state_dict(payload['model_state'], strict=True)
        model.to(device).eval()
        rows = []
        for i, case in enumerate(native):
            assert hashlib.sha256(Path(case['path']).read_bytes()).hexdigest() == case['sha256']
            x, rate = sf.read(case['path'], dtype='float32')
            assert rate == 16000 and x.ndim == 1
            with torch.inference_mode():
                result = model(torch.from_numpy(x)[None,None].to(device), torch.tensor([len(x)], device=device))
            y = result['reconstruction'][0,0].cpu().numpy()
            code = result['z'][result['valid']]
            row = dict(id=case['id'], samples=len(x), active_coefficients=int((code!=0).sum()),
                       coefficients_per_second=int((code!=0).sum())/(len(x)/rate),
                       output_peak=float(np.max(np.abs(y))), **independent(x,y))
            rows.append(row)
            # Float WAVs preserve the samples actually scored, including any overshoots.
            audio_dir=root/'native_outputs'/method
            audio_dir.mkdir(parents=True,exist_ok=True)
            sf.write(audio_dir/(case['id']+'.wav'),y,rate,subtype='FLOAT')
        output['native_full_utterances'][method] = dict(mean=mean(rows), cases=rows,
            mean_coefficients_per_second=float(np.mean([r['coefficients_per_second'] for r in rows])))
        (root/'independent_audit.json').write_text(json.dumps(output, indent=2))
        print('Native full utterances:',method,mean(rows),flush=True)
    print('Saved',root/'independent_audit.json',flush=True)
    write_report(output, root)


def write_report(output, root):
    report = REPO/'approaches/sparse-ae/reports/local_validation_20261110.md'
    lines = ['# Local checkpoint validation — 10 November 2026', '',
        'All seven best checkpoints were imported over SFTP from '
        '`kiran@172.16.56.8:/home/kiran/sparse-ae-work/sparse-ae/runs/librispeech_10h_parallel`.', '',
        '## Findings', '',
        'The historical reconstruction scores reproduce. No material scoring implementation error '
        'was found on these recordings. This does not make the original comparison a held-out '
        'test or establish compression performance.', '',
        f"All checkpoint hashes match the published report: {all(output['checkpoint_hash_matches'].values())}. "
        'The imported SAE Python source files were byte-identical to the local implementation. '
        'Local inference used the imported weights in evaluation mode with strict state loading; no training occurred.', '',
        'Independent MSE/NMSE calculations and least-squares zero-mean SI-SDR were compared with '
        '`sae.metrics`. PESQ-WB and STOI were called directly through the installed `pesq` and '
        '`pystoi` packages, bypassing the repository scoring wrapper. This checks their integration; '
        'it is not an independent implementation of PESQ/STOI themselves.', '',
        '## Historical 40-crop reproduction', '',
        'The original 1.504-second, peak-normalized references were used. Reconstruction WAVs '
        'were scored after PCM16 serialization, as in the published report. The supplied SAE '
        'input samples exactly matched resampling the original 24-kHz reference PCM. '
        'PitchFlower was independently rescored from its 24-kHz PCM audio.', '',
        '| Model | PESQ-WB | STOI | SI-SDR (dB) | NMSE |',
        '| --- | ---: | ---: | ---: | ---: |']
    for method, data in output['independent_historical'].items():
        m=data['mean']; lines.append(f"| {method} | {m['pesq_wb']:.6f} | {m['stoi']:.6f} | {m['si_sdr']:.4f} | {m['nmse']:.6f} |")
    m=output['pitchflower_independent']['mean']
    lines.append(f"| PitchFlower | {m['pesq_wb']:.6f} | {m['stoi']:.6f} | {m['si_sdr']:.4f} | {m['nmse']:.6f} |")
    lines += ['', 'Maximum absolute difference between independently recomputed local scores and the '
              'published SAE scores, across all 280 cases:', '']
    for k in KEYS:
        maximum=max(data['max_report_difference'][k] for data in output['independent_historical'].values())
        lines.append(f'- {k}: {maximum:.9g}')
    lines += ['', 'These are numerical reproduction differences; the per-case results are retained '
              'rather than asserted to be bit-identical across GPU/library environments.', '',
              '## Additional native-amplitude evaluation', '',
              '40 full dev-clean utterances totaling 267.735 seconds, one per speaker. Selection was '
              'fixed by SHA256 of `sae-audit-20261110:<id>` after excluding all seven models’ '
              'training, validation-selection and calibration-selection utterances, plus the original '
              'report utterances. Audio file hashes were verified on import and before inference. '
              'No peak normalization, time alignment, gain fitting or checkpoint reselection was applied. '
              'Scores use float outputs; the saved FLOAT WAVs preserve those samples.', '',
              '**These speakers were used in development. This is an unused-utterance diagnostic, '
              'not a fresh speaker-held-out test.** No test-clean directory was found in the searched '
              'server dataset locations.', '',
              '| Model | PESQ-WB | STOI | SI-SDR (dB) | NMSE | Active coefficients/s |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for method,data in output['native_full_utterances'].items():
        m=data['mean']; lines.append(f"| {method} | {m['pesq_wb']:.6f} | {m['stoi']:.6f} | {m['si_sdr']:.4f} | {m['nmse']:.6f} | {data['mean_coefficients_per_second']:.1f} |")
    lines += ['', '## Interpretation and limits', '',
              '- The original report’s 20 validation and 20 calibration utterances cannot support an independent test claim.',
              '- Peak-normalized short crops differ from the native-amplitude, full-utterance training-validation pipeline. Differences between the two tables cannot be attributed to amplitude alone: the recordings and durations also differ.',
              '- Sparsity levels differ substantially. Quality rankings alone do not compare equal rate or equal active-coefficient budgets.',
              '- No quantized sparse bitstream was evaluated. Active coefficients per second are not bits per second.',
              '- PESQ/STOI and samplewise SI-SDR measure different properties. Agreement of arithmetic does not establish perceived quality or statistical significance.', '',
              '## Reproduction artifacts', '',
              '- [Independent per-case scores and controls](../../../runs/sparse_ae_local_audit_20261110/independent_audit.json)',
              '- [Imported files, paths and SHA256 hashes](../../../runs/sparse_ae_local_audit_20261110/download_manifest.json)',
              '- [Additional recording selection](../../../runs/sparse_ae_local_audit_20261110/native_manifest.json)',
              '- [Local package versions](../../../runs/sparse_ae_local_audit_20261110/local_environment.json)',
              '- [Validation script](../../../tools/audit_sparse_ae_local.py)', '',
              'The raw local artifacts are under `runs/sparse_ae_local_audit_20261110/` and are not '
              'embedded in this Markdown report. Identity, additive-noise and sample-permutation '
              'metric controls are stored in the independent JSON. Historical reports were preserved.', '',
              '```powershell', 'python tools/audit_sparse_ae_local.py', '```', '']
    report.write_text('\n'.join(lines), encoding='utf-8')
    print('Report:', report, flush=True)


if __name__ == '__main__':
    main()
