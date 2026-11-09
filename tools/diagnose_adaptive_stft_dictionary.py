"""Frozen-payload diagnosis. No training, model updates, or new test selection.

Compare all 64 existing one-second test excerpts with stronger coefficient fits,
rank-32 complex SVD factors, and two oracle phase/magnitude interventions.
These test-set diagnostics are exploratory. They are not a new untouched test.
The SVD factors are serialized in the same format and size as the trained atoms.
Phase/magnitude interventions use untransmitted source information; their scores
are diagnostic controls, not achievable rates or proposed codecs.
"""
from pathlib import Path
from types import SimpleNamespace
import csv
import importlib.util
import json
import struct
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/adaptive_stft_dictionary_librispeech_20261107"
EVAL = RUN / "bitrate_quality_test_64"
OUT = RUN / "diagnosis_test_excerpts_64"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def unpack(path):
    raw = path.read_bytes(); n = struct.unpack("<I", raw[4:8])[0]
    h = json.loads(raw[8:8+n]); v = np.frombuffer(raw, dtype="<f4", offset=8+n).copy()
    f,k,t = h["bins"],h["atoms"],h["frames"]
    i=1; d=torch.complex(torch.from_numpy(v[i:i+f*k].reshape(f,k)),torch.from_numpy(v[i+f*k:i+2*f*k].reshape(f,k)))
    i+=2*f*k; a=torch.complex(torch.from_numpy(v[i:i+k*t].reshape(k,t)),torch.from_numpy(v[i+k*t:].reshape(k,t)))
    return d[None],a[None],torch.tensor([[float(v[0])]]),h


def least_squares(d,s):
    """Solve the exact real-linear Pi(D A) least-squares objective."""
    d64,s64=d[0].to(torch.complex128),s[0].to(torch.complex128)
    top=torch.cat((d64.real,-d64.imag),1)
    bottom=torch.cat((d64.imag,d64.real),1);bottom[0]=bottom[-1]=0
    matrix=torch.cat((top,bottom),0)
    target=torch.cat((s64.real,s64.imag),0)
    fit=torch.linalg.lstsq(matrix,target,driver="gelsd")
    k=d.shape[-1]
    a=torch.complex(fit.solution[:k],fit.solution[k:])[None].to(torch.complex64)
    return a,int(fit.rank),float(fit.singular_values[0]/fit.singular_values[-1])


def phase(z):
    return torch.where(z.abs()>1e-10,z/z.abs().clamp_min(1e-10),torch.ones_like(z))


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    m=load_module("diagnostic_dictionary_model",ROOT/"approaches/representations/adaptive_stft_dictionary.py")
    e=load_module("diagnostic_payload_codec",ROOT/"tools/evaluate_adaptive_stft_dictionary.py")
    protocol=json.loads((EVAL/"protocol.json").read_text())
    cfg=m.Settings(**protocol["training_settings"]);transform=m.AudioTransform(cfg)
    assert m.file_hash(ROOT/"approaches/representations/adaptive_stft_dictionary.py")==protocol["model_source_sha256"]
    OUT.mkdir(exist_ok=False);(OUT/"controls").mkdir();(OUT/"audio").mkdir()
    m.atomic_json(OUT/"protocol.json",dict(selection=protocol["selection"],checkpoint_sha256=protocol["checkpoint_sha256"],
                  modes=["one-second excerpts only"],scope="exploratory diagnosis of the existing test group; not untouched testing",
                  oracle_limits="Only phase/magnitude swaps use untransmitted source information; SVD factors are encoded",
                  model_updates=0))
    rows=[];bands=[(0,500),(500,2000),(2000,4000),(4000,8001)]
    for index,record in enumerate(protocol["selection"]):
        path=Path(protocol["dataset_root"])/record["relative"]
        x=m.read_excerpt(Path(protocol["dataset_root"]),record,record["offset"],cfg)[None]
        payload=EVAL/"payloads"/f"{record['utterance']}_excerpt.asd"
        d,a,gain,h=unpack(payload);s=transform.stft(x/gain)
        baseline,_=e.decode(payload);baseline=torch.from_numpy(baseline)[None]
        sy=transform.stft(baseline/gain)
        optimal,rank,condition=least_squares(d,s)
        u,singular,vh=torch.linalg.svd(s[0],full_matrices=False)
        k=cfg.atoms;svd_d=u[:,:k][None];svd_a=(singular[:k,None]*vh[:k])[None]
        outputs={"trained":baseline}
        for label,dc,ac in (("same_atoms_ls",d,optimal),("source_svd_atoms",svd_d,svd_a)):
            control=OUT/"controls"/f"{record['utterance']}_{label}.asd"
            e.encode(control,SimpleNamespace(dictionaries=[dc],coefficients=[ac],gain=gain),cfg,cfg.samples)
            wave,_=e.decode(control);outputs[label]=torch.from_numpy(wave)[None]
        outputs["oracle_source_phase"]=transform.inverse(sy.abs()*phase(s),cfg.samples)*gain
        outputs["oracle_source_magnitude"]=transform.inverse(s.abs()*phase(sy),cfg.samples)*gain
        # Re-analyzing the synthesized waveform can change its spectrum, but a
        # further inverse must reproduce the same audio (consistency control).
        consistent=transform.inverse(sy,cfg.samples)*gain
        consistency_wave_error=float((consistent-baseline).square().sum()/baseline.square().sum().clamp_min(1e-12))
        for label,y in outputs.items():
            spectrum=transform.stft(y/gain)
            residual=spectrum-s;freq=torch.fft.rfftfreq(cfg.n_fft,1/cfg.sample_rate)
            row=dict(utterance=record["utterance"],speaker=record["speaker"],method=label,
                     **m.waveform_metrics(x[0],y[0],cfg.n_fft//2),
                     **e.quality(x[0].numpy(),y[0].numpy(),cfg.sample_rate),
                     spectral_magnitude_nmse=float((spectrum.abs()-s.abs()).square().sum()/s.abs().square().sum()),
                     projected_ls_rank=rank,projected_ls_condition=condition,
                     waveform_consistency_control_nmse=consistency_wave_error)
            for lo,hi in bands:
                mask=(freq>=lo)&(freq<hi)
                row[f"band_{lo}_{hi}_complex_nmse"]=float(residual[:,mask].abs().square().sum()/s[:,mask].abs().square().sum().clamp_min(1e-12))
                row[f"band_{lo}_{hi}_energy_ratio"]=float(spectrum[:,mask].abs().square().sum()/s[:,mask].abs().square().sum().clamp_min(1e-12))
            rows.append(row)
            if index<3:sf.write(OUT/"audio"/f"{record['utterance']}_{label}.wav",y[0].numpy(),cfg.sample_rate,subtype="FLOAT")
        if index<3:sf.write(OUT/"audio"/f"{record['utterance']}_reference.wav",x[0].numpy(),cfg.sample_rate,subtype="FLOAT")
        m.append_json(OUT/"rows.jsonl",{"rows":rows[-5:]})
        print(f"Diagnosed {index+1}/64",flush=True)
    with (OUT/"metrics.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={}
    for method in outputs:
        group=[r for r in rows if r["method"]==method]
        summary[method]={"cases":len(group),"metrics":{}}
        for metric in ("nmse","pesq_wb","stoi","spectral_magnitude_nmse",*[f"band_{lo}_{hi}_{kind}" for lo,hi in bands for kind in ("complex_nmse","energy_ratio")]):
            values=[r[metric] for r in group if r[metric] is not None]
            summary[method]["metrics"][metric]={"mean":float(np.mean(values)) if values else None,"valid":len(values)}
    m.atomic_json(OUT/"summary.json",summary)
    diagnostic_protocol=json.loads((OUT/"protocol.json").read_text())
    diagnostic_protocol["diagnostic_source_sha256"]=m.file_hash(Path(__file__))
    diagnostic_protocol["all_control_payload_sizes_match_trained"]=all(
        (OUT/"controls"/f"{record['utterance']}_{label}.asd").stat().st_size
        == (EVAL/"payloads"/f"{record['utterance']}_excerpt.asd").stat().st_size
        for record in protocol["selection"] for label in ("same_atoms_ls","source_svd_atoms"))
    m.atomic_json(OUT/"protocol.json",diagnostic_protocol)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    positions=np.arange(4);width=.25
    for j,method in enumerate(("trained","same_atoms_ls","source_svd_atoms")):
        values=[summary[method]["metrics"][f"band_{lo}_{hi}_energy_ratio"]["mean"] for lo,hi in bands]
        axes[0].bar(positions+(j-1)*width,values,width,label=method)
    axes[0].axhline(1,color="black",linestyle="--");axes[0].set_xticks(positions,["0–0.5","0.5–2","2–4","4–8"])
    axes[0].set(xlabel="Frequency band (kHz)",ylabel="Output/reference spectral energy ratio",title="Mean across 64 fixed excerpts")
    axes[0].legend(fontsize=8)
    labels=list(summary);axes[1].barh(labels,[summary[k]["metrics"]["pesq_wb"]["mean"] for k in labels])
    axes[1].set(xlabel="Mean PESQ-WB",title="Phase/magnitude swaps use source information")
    fig.tight_layout();fig.savefig(OUT/"diagnosis.png",dpi=150);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
