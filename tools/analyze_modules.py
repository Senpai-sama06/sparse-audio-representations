"""Trace the frozen S-AF signal path on one reproducibly chosen development clip.

This is an illustrative module audit, not a new population evaluation. It records
feature shapes and offsets, residual updates, projection ranks, and an exact
decomposition of waveform error before/after pruning. No weights are changed.
"""
import argparse
import csv
from pathlib import Path
import json
from datetime import datetime, timezone
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lrac.data.preprocessing import load_audio
from lrac.models.autoencoder import ResidualBlock
from lrac.train import load_run
from lrac.utils import file_hash, write_json


def rms(x):
    return float(x.double().square().mean().sqrt())


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trace(network, x):
    """Record semantic block boundaries, not the no-op Identity modules."""
    features, residuals = [], []
    zero = torch.zeros_like(x)
    for index, module in enumerate(network):
        before, before_zero = x, zero
        x = module(x)
        zero = module(zero)
        if isinstance(module, torch.nn.Identity):
            continue
        features.append((str(index), type(module).__name__, x.detach().clone(), zero.detach().clone()))
        if isinstance(module, ResidualBlock):
            update = x-before
            a, b = before.double().reshape(-1), update.double().reshape(-1)
            signal_input = (before.double()-before_zero.double()).reshape(-1)
            signal_update = ((x.double()-zero.double()).reshape(-1)-signal_input)
            residuals.append(dict(module_index=index, input_rms=rms(before),
                update_rms=rms(update), output_rms=rms(x),
                update_to_input_norm=float(b.norm()/a.norm().clamp_min(1e-20)),
                input_update_cosine=float((a@b)/(a.norm()*b.norm()).clamp_min(1e-20)),
                signal_update_to_input_norm=float(signal_update.norm()/signal_input.norm().clamp_min(1e-20)),
                signal_input_update_cosine=float((signal_input@signal_update)/
                    (signal_input.norm()*signal_update.norm()).clamp_min(1e-20))))
    return features, residuals


def projection_summary(module):
    matrix = module.weight[:,:,0].detach().cpu().double().numpy()
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = float(max(matrix.shape)*np.finfo(np.float32).eps*singular[0])
    rank = int((singular > tolerance).sum())
    return dict(shape=list(matrix.shape), numerical_rank=rank,
        rank_tolerance=tolerance, singular_values=singular.tolist(),
        nonzero_condition_number=float(singular[0]/singular[rank-1]) if rank else None,
        input_nullity=matrix.shape[1]-rank)


def feature_map(ax, x, zero, label):
    values = (x-zero)[0].cpu().numpy()
    limit = max(float(np.quantile(np.abs(values),.995)),1e-12)
    ax.imshow(values,origin="lower",aspect="auto",extent=(0,2,-.5,len(values)-.5),
              interpolation="nearest",cmap="RdBu_r",vmin=-limit,vmax=limit)
    ax.set(title=f"{label}: {values.shape[0]} channels x {values.shape[1]} steps | f(x)-f(0)",ylabel="Channel ID")


def plot_trace(output, encoder, decoder, waveforms):
    """Features are drawn on their own time grids, never played as 16-kHz audio."""
    fig, axes = plt.subplots(8,1,figsize=(13,15),layout="constrained")
    t = np.arange(len(waveforms["reference"]))/16000
    axes[0].plot(t,waveforms["reference"],lw=.5)
    axes[0].set(title="Input waveform | 16 kHz | same two-second excerpt throughout",ylabel="Amplitude")
    stages = [entry for entry in encoder if entry[1]=="ResidualBlock"] + [encoder[-1]]
    labels = ["Encoder stage 1", "Encoder stage 2", "Encoder stage 3", "Encoder stage 4", "Latent projection"]
    for ax,(_,_,x,zero),label in zip(axes[1:6],stages,labels):
        feature_map(ax,x,zero,label)
    values = waveforms["latent_0.1"]
    limit = max(float(np.quantile(np.abs(values[values!=0]),.995)),1e-12)
    axes[6].imshow(values.T,origin="lower",aspect="auto",extent=(0,2,-.5,255.5),
                   interpolation="nearest",cmap="RdBu_r",vmin=-limit,vmax=limit)
    axes[6].set(title="Actual retained raw coefficients, global 10% support",ylabel="Channel ID")
    axes[7].plot(t,waveforms["reference"],lw=.6,alpha=.6,label="Reference")
    axes[7].plot(t,waveforms["support_0.1"],lw=.5,alpha=.8,label="10% reconstruction")
    axes[7].set(title="Output waveform | 16 kHz",ylabel="Amplitude")
    axes[7].legend(fontsize=8)
    for ax in axes:
        ax.set(xlim=(0,2),xlabel="Time (s)")
    fig.suptitle("One illustrative development clip | feature scales independent by stage; colors clipped at 99.5% absolute value",fontsize=11)
    fig.savefig(output/"saf_module_signal_flow.png",dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(6,1,figsize=(13,12),layout="constrained")
    _,_,x,zero=encoder[-1]
    feature_map(axes[0],x,zero,"Full-support latent values before decoder")
    entries=[decoder[0],*[e for e in decoder if e[1]=="ConvTranspose1d"][:-1]]
    for ax,(_,_,x,zero),label in zip(axes[1:5],entries,["Decoder projection","Upsampling 1","Upsampling 2","Upsampling 3"]):
        feature_map(ax,x,zero,label)
    axes[5].plot(t,waveforms["reference"],lw=.6,label="Reference",alpha=.6)
    axes[5].plot(t,waveforms["full"],lw=.5,label="Full-support output",alpha=.8)
    axes[5].set(title="Upsampling 4: one-channel waveform, 32,000 samples",ylabel="Amplitude")
    axes[5].legend(fontsize=8)
    for ax in axes:
        ax.set(xlim=(0,2),xlabel="Time (s)")
    fig.suptitle("Decoder trace at FULL support | feature scales independent; colors clipped at 99.5% absolute value",fontsize=11)
    fig.savefig(output/"saf_module_decoder_flow.png",dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,6),layout="constrained")
    axes[0].plot(t,waveforms["reference"]-waveforms["full"],lw=.5,label="x - full-support reconstruction")
    axes[0].plot(t,waveforms["full"]-waveforms["support_0.1"],lw=.5,alpha=.7,label="Full-support - 10% reconstruction")
    axes[0].set(title="Waveform error before pruning and change introduced by pruning",ylabel="Amplitude",xlabel="Time (s)")
    axes[0].legend(fontsize=8)
    from scipy import signal
    for name,key in [("Reference","reference"),("Full support","full"),("10% support","support_0.1"),("1% support","support_0.01")]:
        f,p=signal.welch(waveforms[key],16000,nperseg=2048,noverlap=1536)
        axes[1].plot(f,10*np.log10(np.maximum(p,1e-16)),label=name,lw=.8)
    axes[1].set(xlabel="Waveform frequency (Hz)",ylabel="PSD (dB/Hz)",title="Actual waveform spectra; feature-map colors are not spectra")
    axes[1].legend(fontsize=8)
    fig.savefig(output/"saf_module_reconstruction.png",dpi=150)
    plt.close(fig)


def analyze(args):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    run, output = Path(args.run), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    local = run/"analysis"/"modules"
    local.mkdir(parents=True, exist_ok=True)
    cfg, model, state = load_run(run,args.device)
    if cfg.model.activation != "identity" or cfg.model.kind != "sparse_autoencoder":
        raise ValueError("This audit requires the affine sparse autoencoder")
    pitch_protocol = json.loads((run/"analysis/pitch_intervention/protocol.json").read_text())
    if file_hash(run/"checkpoints/best.pt") != pitch_protocol["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the frozen study")
    cohort = json.loads((run/"analysis/pitch_intervention/cohort.json").read_text())
    chosen = cohort[0]["voiced"]  # First source, not selected by model responses.
    manifest = cfg.manifest(cfg.data.val_manifest)
    if file_hash(manifest) != pitch_protocol["manifest_sha256"]:
        raise ValueError("Development manifest differs from the frozen intervention")
    row = next(json.loads(line) for line in manifest.read_text().splitlines()
               if json.loads(line)["id"] == chosen["id"])
    audio = load_audio(row,cfg.data.sample_rate)
    n = 32000
    if len(audio) < n:
        raise ValueError("Selected source is shorter than the two-second trace")
    start = max(0,min(chosen["start_sample"]-(n-12288)//2,len(audio)-n))
    audio = audio[start:start+n]
    audio = (audio*(.05/max(np.sqrt(np.mean(audio.astype(float)**2)),1e-20))).astype(np.float32)
    x = torch.from_numpy(audio)[None,None].to(next(model.parameters()).device)
    model.eval()
    with torch.no_grad(),torch.backends.cudnn.flags(enabled=False,allow_tf32=False):
        encoder, er = trace(model.encoder,x)
        rep = model.encode_dense(x)
        zero_rep = model.encode_dense(torch.zeros_like(x))
        dense = model.decode(rep)
        decoder, dr = trace(model.decoder,rep.concatenate().transpose(1,2))
        if not torch.equal(encoder[-1][2].transpose(1,2),rep.concatenate()):
            raise ValueError("Encoder trace differs from actual representation")
        if not torch.equal(decoder[-1][2][...,:n],dense):
            raise ValueError("Decoder trace differs from actual reconstruction")
        ledger = []
        jump, receptive_field = 1,1
        for index,kind,y,zero in encoder:
            if kind == "Conv1d" and model.encoder[int(index)].kernel_size[0] == 9:
                receptive_field += 8*jump
                jump *= 4
            elif kind == "ResidualBlock":
                receptive_field += 4*jump
            ledger.append(dict(side="encoder",module_index=index,module=kind,
                channels=y.shape[1],time_steps=y.shape[2],values=y.numel(),
                values_per_input_sample=y.numel()/n,grid_rate_hz=16000/jump,
                input_receptive_field_samples=receptive_field,
                activation_rms=rms(y),zero_input_rms=rms(zero),
                signal_dependent_rms=rms(y-zero)))
        jump = 256
        for index,kind,y,zero in decoder:
            if kind == "ConvTranspose1d":
                jump //= 4
            ledger.append(dict(side="decoder",module_index=index,module=kind,
                channels=y.shape[1],time_steps=y.shape[2],values=y.numel(),
                values_per_input_sample=y.numel()/n,grid_rate_hz=16000/jump,
                input_receptive_field_samples=None,
                activation_rms=rms(y),zero_input_rms=rms(zero),
                signal_dependent_rms=rms(y-zero)))
        denominator = float(x.double().square().sum())
        dense_error = x.double()-dense.double()
        metrics, waveforms = [], dict(reference=audio,full=dense[0,0].cpu().numpy())
        for f in [.01,.1,1.]:
            selected = model.sparsify(rep,keep_fraction=f)
            reconstructed = model.decode(selected)
            zero_selected = model.sparsify(zero_rep,keep_fraction=f)
            zero_reconstructed = model.decode(zero_selected)
            pruning = dense.double()-reconstructed.double()
            dense_nmse = float(dense_error.square().sum())/denominator
            pruning_power = float(pruning.square().sum())/denominator
            interaction = float(2*(dense_error*pruning).sum())/denominator
            nmse = float((x.double()-reconstructed.double()).square().sum())/denominator
            if not np.isclose(dense_nmse+pruning_power+interaction,nmse,atol=1e-12):
                raise ValueError("Waveform error decomposition failed")
            metrics.append(dict(support=f,selected_coordinates=int(selected.selected_mask.sum()),
                nmse=nmse,snr_db=float(-10*np.log10(max(nmse,1e-20))),
                dense_nmse=dense_nmse,pruning_distortion_nmse=pruning_power,
                error_interaction_nmse=interaction,zero_input_output_rms=rms(zero_reconstructed),
                zero_input_retained_channel_types=int((zero_selected.concatenate()!=0).any(1).sum())))
            waveforms[f"support_{f:g}"] = reconstructed[0,0].cpu().numpy()
            if f == .1:
                waveforms["latent_0.1"] = selected.concatenate()[0].cpu().numpy()
    residuals = [dict(side=side,**record) for side,records in [("encoder",er),("decoder",dr)] for record in records]
    frequency=np.fft.rfftfreq(n,1/16000)
    weights=np.full(len(frequency),2.)
    weights[[0,-1]]=1
    reference_power=np.abs(np.fft.rfft(audio.astype(float)))**2*weights/n
    output_power=np.abs(np.fft.rfft(waveforms["full"].astype(float)))**2*weights/n
    error_power=np.abs(np.fft.rfft(audio.astype(float)-waveforms["full"].astype(float)))**2*weights/n
    high=frequency>=2000
    band_diagnostic=dict(cutoff_hz=2000,selection="Exploratory cutoff selected after inspecting the illustrative waveform PSD, not a preregistered test",
        input_energy_fraction=float(reference_power[high].sum()/reference_power.sum()),
        full_output_to_input_band_energy_ratio=float(output_power[high].sum()/reference_power[high].sum()),
        fraction_full_reconstruction_error_in_band=float(error_power[high].sum()/error_power.sum()),
        band_nmse=float(error_power[high].sum()/reference_power[high].sum()))
    save_csv(output/"saf_module_trace.csv",ledger)
    save_csv(output/"saf_module_residuals.csv",residuals)
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(),run=run.name,
        checkpoint_sha256=file_hash(run/"checkpoints/best.pt"),checkpoint_step=state["step"],
        manifest_sha256=file_hash(manifest),source_id=row["id"],speaker=row["speaker"],
        source_start_sample=start,samples=n,sample_rate=16000,input_rms=rms(x),
        scope="One illustrative development clip; not population evidence or a module-necessity test",
        precision="Native IEEE FP32 convolutions; cuDNN and CUDA matmul TF32 disabled",
        zero_decoder_definition="Decoder fed all-zero latent values, not zero waveform passed through encoder",
        identity_modules=sum(isinstance(m,torch.nn.Identity) for m in model.modules()),
        encoder_projection=projection_summary(model.encoder[-1]),
        decoder_projection=projection_summary(model.decoder[0]),
        metrics=metrics,exploratory_high_band=band_diagnostic,source_sha256=file_hash(Path(__file__)))
    write_json(output/"saf_module_trace.json",summary)
    np.savez_compressed(local/"trace.npz",**waveforms)
    plot_trace(output,encoder,decoder,waveforms)
    print(json.dumps({k:v for k,v in summary.items() if k not in ["encoder_projection","decoder_projection"]},indent=2))
    for name in ["encoder_projection","decoder_projection"]:
        print(name,{k:v for k,v in summary[name].items() if k!="singular_values"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default="runs/librispeech_sparse_autoencoder_affine_10h_20261027T114554258463Z_seed0")
    parser.add_argument("--output",default="documentation")
    parser.add_argument("--device",default="auto")
    analyze(parser.parse_args())
