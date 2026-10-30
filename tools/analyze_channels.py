"""Rank and characterize retained channels of the frozen affine audio model."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import csv
import json
from importlib import metadata
from pathlib import Path
import numpy as np
import torch
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lrac.channel_analysis import (PYIN, reference_voicing, affine_contributions,
    voicing_counts, binary_correlation, contribution_spectrum, spectral_summary)
from lrac.data.datasets import AudioDataset
from lrac.train import load_run
from lrac.utils import file_hash, write_json, seed_all


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_spectrograms(example, channels, statistics, path, indices):
    """Plot actual isolated reconstructions using one shared reference dB scale."""
    x, full, parts, labels, sr = [example[k] for k in ["x", "full", "parts", "labels", "sr"]]
    n = min(len(x), 8 * sr)  # Crop after whole-recording top-k and decoding.
    signals = [x[:n], full[:n], *[parts[i, :n] for i in indices]]
    titles = ["Reference", "Full reconstruction, 10% support", *[f"Channel {channels[i]}: g(z_c) - g(0)" for i in indices]]
    transforms = [signal.stft(y, sr, window="hann", nperseg=1024, noverlap=896,
                             boundary="zeros", padded=False) for y in signals]
    peak = max(float(np.abs(transforms[0][2]).max()), 1e-12)
    fig, axes = plt.subplots(len(signals), 1, figsize=(12, 1.7 * len(signals)), sharex=True, constrained_layout=True)
    for ax, (f, t, z), title in zip(axes, transforms, titles):
        db = 20 * np.log10(np.maximum(np.abs(z) / peak, 1e-4))
        image = ax.pcolormesh(t, f, db, shading="auto", vmin=-80, vmax=0, cmap="magma")
        ax.set(yscale="log", ylim=(60, 8000), ylabel="Hz", title=title)
        ax.set_yticks([100, 300, 1000, 3000, 8000], ["100", "300", "1k", "3k", "8k"])
    times = np.arange(len(labels["voiced"])) * 256 / sr
    for flag, color, name in [(labels["voiced"] & labels["audible"], "cyan", "pYIN voiced"),
                               (~labels["voiced"] & labels["audible"], "lime", "pYIN unvoiced")]:
        axes[0].scatter(times[flag], np.full(int(flag.sum()), 7000), s=6, color=color, label=name)
    axes[0].legend(loc="upper right", fontsize=7)
    axes[-1].set(xlim=(0, n / sr), xlabel="Time (s), from the start of the whole recording")
    fig.suptitle(f"{example['id']} | same amplitude scale for every row | zero-based channels", fontsize=11)
    fig.colorbar(image, ax=list(axes), fraction=.015, pad=.015, label="dB relative to reference STFT peak")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def analyze(args):
    seed_all(0)
    cfg, model, state = load_run(args.run, args.device)
    if cfg.model.activation != "identity" or cfg.model.kind != "sparse_autoencoder":
        raise ValueError("This analysis requires the affine sparse autoencoder")
    if any(isinstance(module, torch.nn.LeakyReLU) for module in model.modules()):
        raise ValueError("Decoder contains nonlinear activations")
    manifest = cfg.manifest(cfg.data.test_manifests[args.split])
    dataset = AudioDataset(manifest, cfg.data.sample_rate, limit=args.limit)
    frozen_path = Path("documentation/sparse_autoencoder_test_runs.json")
    if frozen_path.exists():
        for record in json.loads(frozen_path.read_text())["runs"]:
            if record["run_name"] == Path(args.run).name:
                if file_hash(Path(args.run)/"checkpoints/best.pt") != record["checkpoint_sha256"] or file_hash(manifest) != record["clean_manifest_sha256"]:
                    raise ValueError("Weights or data differ from the evaluated S-AF study")
    sr, width = cfg.data.sample_rate, cfg.model.latent_width
    device = next(model.parameters()).device
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    local = Path(args.run) / "analysis" / f"channels_{args.split}_0.1"
    if args.limit:
        local = local / f"limit_{args.limit}"
        if output == Path("documentation"):
            output = local / "report"
            output.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), run=Path(args.run).name,
        library_versions={name:metadata.version(name) for name in ["torch","librosa","numpy","scipy"]},
        checkpoint_sha256=file_hash(Path(args.run)/"checkpoints/best.pt"), checkpoint_step=state["step"],
        manifest_sha256=file_hash(manifest), split=args.split, examples=len(dataset), limit=args.limit,
        support_fraction=.1, channel_indices="zero-based", ranking="count of retained nonzero coefficients",
        pyin={**PYIN, "sample_rate":sr}, energy_gate="RMS >= max(1e-4, recording_peak_RMS * 10**(-35/20))",
        matched_energy="within recording, 3-dB RMS bins; >=5 V and 5 U frames; weight nV*nU/(nV+nU)",
        spectrum="duration-weighted Welch PSD; Hann 2048, overlap1536, no detrending; all contribution samples",
        encoder_cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32), decoder_cudnn_allow_tf32=False,
        decoder_cudnn_enabled=False,decoder_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        decoder_precision="Native IEEE FP32 convolutions for attribution; original encoder precision and support retained",
        script_sha256=file_hash(Path(__file__)), analysis_source_sha256=file_hash(Path("lrac/channel_analysis.py")),
        model_source_sha256=file_hash(Path("lrac/models/autoencoder.py")))
    write_json(local/"protocol.json", protocol)
    # Choose examples by source duration and speaker order, before inspecting channels.
    examples, seen = {}, set()
    for row in dataset.rows:
        if row["speaker"] not in seen and row["num_samples"] / row["sample_rate"] >= 6:
            examples[row["id"]] = None
            seen.add(row["speaker"])
            if len(examples) == 2:
                break
    counts = np.zeros(width, dtype=np.int64)
    recording_counts = np.zeros((len(dataset), width), dtype=np.int32)
    total_steps, total_seconds, selected_zeros = 0, 0., 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(reference_voicing, row, sr) for row in dataset.rows]
        for i, (x, row) in enumerate(dataset):
            with torch.no_grad():
                rep = model.sparsify(model.encode_dense(x[None].to(device)), keep_fraction=.1)
                z = rep.concatenate()[0]
                active = z != 0
                count = active.sum(0).cpu().numpy()
                selected_zeros += int((rep.selected_mask & (rep.concatenate() == 0)).sum())
            recording_counts[i] = count
            counts += count
            total_steps += len(z)
            total_seconds += x.shape[-1] / sr
            if (i + 1) % 100 == 0 or i + 1 == len(dataset):
                print(f"RANK {i+1}/{len(dataset)}", flush=True)
                write_json(local/"progress.json", dict(phase="ranking", completed=i+1, examples=len(dataset)))
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        channels = np.lexsort((np.arange(width), -counts))[:10].tolist()
        print("TOP_CHANNELS", channels, flush=True)
        ranking = [dict(rank=rank+1, channel=c, retained=int(counts[c]),
            activation_frequency=float(counts[c]/total_steps), share_retained=float(counts[c]/counts.sum()),
            recording_coverage=float(np.mean(recording_counts[:, c]>0)))
            for rank,c in enumerate(np.lexsort((np.arange(width), -counts)).tolist())]
        save_csv(output/"saf_channel_ranking.csv", ranking)
        size = len(channels)
        magnitudes, signed = [[] for _ in channels], [[] for _ in channels]
        spectral_power = np.zeros((size,1025))
        contribution_energy, delta_sum = np.zeros(size), np.zeros(size)
        reference_energy, full_energy, bias_energy = 0., 0., 0.
        precision_difference_energy, original_reconstruction_energy, precision_max_error = 0., 0., 0.
        per_speaker, category_totals = {}, {}
        check_errors = []
        for i, (x, row) in enumerate(dataset):
            with torch.no_grad():
                rep = model.sparsify(model.encode_dense(x[None].to(device)), keep_fraction=.1)
                z = rep.concatenate()[0, :, channels].cpu().numpy()
                original = model.decode(rep)[0,0].cpu().numpy()
                bias, full, parts = affine_contributions(model, rep, channels)
            labels = futures[i].result()
            if len(labels["voiced"]) != len(z):
                raise ValueError("Voicing and coefficient time grids differ")
            active = z != 0
            vc = voicing_counts(active, labels)
            speaker = str(row["speaker"])
            bucket = per_speaker.setdefault(speaker,{})
            for key,value in vc.items():
                bucket[key] = bucket.get(key,0) + value
                category_totals[key] = category_totals.get(key,0) + value
            for j in range(size):
                values = z[active[:,j],j]
                signed[j].append(values)
                magnitudes[j].append(np.abs(values))
            reference = x[0].numpy().astype(np.float64)
            e = full.astype(np.float64) - reference
            energy = float(reference @ reference)
            pc = parts.astype(np.float64)
            contribution_energy += np.sum(pc**2, axis=1)
            delta = np.sum(pc**2-2*pc*e,axis=1) / max(energy,1e-20)
            delta_sum += delta
            bucket["delta_sum"] = bucket.get("delta_sum",0) + delta
            bucket["recordings"] = bucket.get("recordings",0) + 1
            reference_energy += energy
            difference = full.astype(np.float64)-original
            precision_difference_energy += float(np.sum(difference**2))
            original_reconstruction_energy += float(np.sum(original.astype(np.float64)**2))
            precision_max_error = max(precision_max_error,float(np.max(np.abs(difference))))
            full_energy += float(np.sum((full.astype(np.float64)-bias)**2))
            bias_energy += float(np.sum(bias.astype(np.float64)**2))
            frequencies, psd = contribution_spectrum(parts, sr)
            spectral_power += psd * (len(reference)/sr)
            if row["id"] in examples:
                examples[row["id"]] = dict(x=reference, full=full, parts=parts, labels=labels, sr=sr, id=row["id"])
                for j,c in enumerate(channels):
                    changed = rep.concatenate().clone()
                    changed[...,c] = 0
                    with torch.no_grad(), torch.backends.cudnn.flags(enabled=False, allow_tf32=False):
                        removed = model.decode(rep.with_values(changed))[0,0].cpu().numpy()
                    check_errors.append(float(np.max(np.abs((full-removed)-parts[j]))))
            if (i+1)%100 == 0 or i+1 == len(dataset):
                print(f"CHARACTERIZE {i+1}/{len(dataset)}",flush=True)
                write_json(local/"progress.json",dict(phase="characterizing",completed=i+1,examples=len(dataset),channels=channels))
                if device.type=="cuda":torch.cuda.empty_cache()
    if file_hash(Path(args.run)/"checkpoints/best.pt") != protocol["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during analysis")
    # Probe fixed interior synthesis responses separately from data-driven spectra.
    values = torch.zeros(1,32,width,device=device)
    values[0,16,channels] = 1
    impulse_rep = model.make_representation(values,torch.zeros(1,1,8192,device=device),
                                           torch.tensor([8192],device=device),{"hop":256})
    _,_,impulses = affine_contributions(model,impulse_rep,channels)
    impulse_f = np.fft.rfftfreq(16384,1/sr)
    impulse_power = np.abs(np.fft.rfft(impulses,n=16384,axis=-1))**2
    rng=np.random.default_rng(0)
    speakers=list(per_speaker.values())
    draws=rng.integers(0,len(speakers),size=(2000,len(speakers)))
    adjusted=np.array([b["matched_difference_sum"]/b["matched_weight"] for b in speakers if b["matched_weight"]>0])
    if len(adjusted)!=len(speakers):raise ValueError("A speaker has no energy-matched V/U strata")
    bootstrap=adjusted[draws].mean(axis=1)
    lower,upper=np.percentile(bootstrap,[2.5,97.5],axis=0)
    statistics=[]
    for j,c in enumerate(channels):
        a=np.concatenate(magnitudes[j]);s=np.concatenate(signed[j]);v=category_totals
        if len(a) != counts[c]:
            raise ValueError("Channel retention changed between ranking and characterization")
        result=dict(rank=j+1,channel=c,retained=int(counts[c]),
            activation_frequency=float(counts[c]/total_steps),share_retained=float(counts[c]/counts.sum()),
            activations_per_second=float(counts[c]/total_seconds),recording_coverage=float(np.mean(recording_counts[:,c]>0)),
            abs_mean=float(a.mean()),abs_std=float(a.std()),signed_mean=float(s.mean()),negative_fraction=float(np.mean(s<0)))
        result.update({f"abs_p{q:02d}":float(np.percentile(a,q)) for q in [5,25,50,75,95,99]})
        result["abs_max"]=float(a.max())
        result.update(spectral_summary(frequencies,spectral_power[j]))
        result["impulse_peak_hz"]=float(impulse_f[np.argmax(impulse_power[j])])
        result["impulse_l2_gain"]=float(np.linalg.norm(impulses[j]))
        result.update(p_active_voiced=float(v["av"][j]/v["v"]),p_active_unvoiced=float(v["au"][j]/v["u"]),
            p_voiced_given_active=float(v["av"][j]/(v["av"][j]+v["au"][j])),
            phi_voiced=binary_correlation(v["av"][j],v["au"][j],v["v"],v["u"]),
            matched_rate_difference=float(adjusted[:,j].mean()),matched_ci_low=float(lower[j]),matched_ci_high=float(upper[j]),
            stable_p_active_voiced=float(v["stable_av"][j]/max(v["stable_v"],1)),
            stable_p_active_unvoiced=float(v["stable_au"][j]/max(v["stable_u"],1)),
            confident_p_active_voiced=float(v["confident_av"][j]/max(v["confident_v"],1)),
            confident_p_active_unvoiced=float(v["confident_au"][j]/max(v["confident_u"],1)),
            contribution_energy_over_bias_free_reconstruction=float(contribution_energy[j]/full_energy),
            ablation_delta_nmse_mean=float(delta_sum[j]/len(dataset)))
        result["adjusted_association"]="voiced" if lower[j]>0 else "unvoiced" if upper[j]<0 else "unclear"
        statistics.append(result)
    serializable={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in category_totals.items()}
    summary=dict(protocol=protocol,valid_time_steps=total_steps,seconds=total_seconds,
        total_retained=int(counts.sum()),selected_zero_values=selected_zeros,speakers=len(speakers),
        voicing_counts=serializable,baseline_voiced_fraction=category_totals["v"]/(category_totals["v"]+category_totals["u"]),
        bias_energy_over_reference=bias_energy/reference_energy,ablation_identity_max_absolute_error=max(check_errors),
        decoder_precision_change_relative_rms=float(np.sqrt(precision_difference_energy/original_reconstruction_energy)),
        decoder_precision_change_max_absolute=precision_max_error,
        examples=list(examples),channels=statistics)
    save_results(output,local,summary,examples,frequencies,spectral_power,impulses,impulse_f,impulse_power)
    write_json(local/"progress.json",dict(phase="complete",completed=len(dataset),examples=len(dataset),channels=channels))
    report = Path(__file__).resolve().parents[1]/"README.md" if output.resolve() == Path(__file__).resolve().parents[1]/"documentation" else output/"saf_channel_analysis.md"
    print(f"CHANNEL_REPORT={report}",flush=True)


def save_results(output,local,summary,examples,frequencies,spectral_power,impulses,impulse_f,impulse_power):
    statistics=summary["channels"]
    channels=[r["channel"] for r in statistics]
    size=len(channels)
    save_csv(output/"saf_channel_statistics.csv",statistics)
    write_json(output/"saf_channel_analysis.json",summary)
    np.savez_compressed(local/"spectra.npz",frequency=frequencies,power=spectral_power,impulse_frequency=impulse_f,
                        impulse_power=impulse_power,impulses=impulses,channels=channels)
    for index,ex in enumerate(examples.values()):
        if ex is None:continue
        if index==0:
            plot_spectrograms(ex,channels,statistics,output/"saf_channels_spectrograms_1.png",list(range(5)))
            plot_spectrograms(ex,channels,statistics,output/"saf_channels_spectrograms_2.png",list(range(5,10)))
        else:
            plot_spectrograms(ex,channels,statistics,output/"saf_channels_spectrograms_3.png",[0,4,9])
        np.savez_compressed(local/f"example_{index}.npz",x=ex["x"],full=ex["full"],parts=ex["parts"],
                            voiced=ex["labels"]["voiced"],audible=ex["labels"]["audible"],channels=channels)
    fig,axes=plt.subplots(2,1,figsize=(11,8),constrained_layout=True)
    for j,c in enumerate(channels):
        p=spectral_power[j];p=p/max(p.max(),1e-20)
        axes[0].plot(frequencies,10*np.log10(np.maximum(p,1e-8)),label=f"ch {c}")
    display_max=min(8000,max(600,np.ceil(max(r["energy95_hz"] for r in statistics)/100)*100))
    axes[0].set(xlim=(0,display_max),ylim=(-60,0),xlabel="Frequency (Hz)",ylabel="PSD (dB relative to own peak)",title="Actual contributions: duration-weighted Welch spectra, shape-normalized (dominant-band zoom)")
    axes[0].legend(ncol=5,fontsize=8)
    positions=np.arange(size)
    axes[1].bar(positions-.18,[100*r["p_active_voiced"] for r in statistics],.36,label="Voiced")
    axes[1].bar(positions+.18,[100*r["p_active_unvoiced"] for r in statistics],.36,label="Unvoiced")
    axes[1].set(xticks=positions,xticklabels=[str(c) for c in channels],xlabel="Channel (ranked by retention)",ylabel="Retention within reference state (%)",title="Raw state-conditional retention; energy-matched differences are reported separately")
    axes[1].legend()
    fig.savefig(output/"saf_channel_profiles.png",dpi=170);plt.close(fig)
    write_report(output,summary)


def write_report(output,summary):
    rows=summary["channels"];v=summary["voicing_counts"]
    lines=["# Ten most frequently retained S-AF channels at 10% support", "",
        f"Frozen S-AF checkpoint at step {summary['protocol']['checkpoint_step']}; all {summary['protocol']['examples']:,} test-clean recordings, "
        f"{summary['speakers']} speakers, {summary['seconds']/3600:.3f} hours. No retraining. Channels are **zero-based**.", "",
        "## Concrete result", "",
        f"The ten channels account for {100*sum(r['share_retained'] for r in rows):.2f}% of retained coefficients. "
        f"Minimum recording coverage is {100*min(r['recording_coverage'] for r in rows):.1f}%; individual retention is {100*min(r['activation_frequency'] for r in rows):.2f}–"
        f"{100*max(r['activation_frequency'] for r in rows):.2f}% of valid time positions. "
        "Their data-driven synthesis spectra are predominantly low-frequency, frequency-selective components. "
        f"Dominant PSD peaks span {min(r['peak_hz'] for r in rows):.1f}–{max(r['peak_hz'] for r in rows):.1f} Hz; "
        f"{sum(r['energy95_hz']-r['energy05_hz']<=40 for r in rows)} of ten have 90% of spectral energy within a band no wider than 40 Hz.", "",
        f"{sum(r['matched_ci_low']>0 for r in rows)} of ten have positive energy-matched voiced association: estimated retention-rate differences range from "
        f"{100*min(r['matched_rate_difference'] for r in rows):.2f} to {100*max(r['matched_rate_difference'] for r in rows):.2f} "
        "percentage points, with descriptive 95% speaker-bootstrap intervals shown below. "
        "They remain active in many unvoiced frames and should not be named exclusive voicing detectors. "
        "Other channels were not selected for this top-ten characterization.", "",
        f"For {sum(r['ablation_delta_nmse_mean']>0 for r in rows)} of ten channels, uncompensated removal increases mean waveform NMSE. The largest increase among them "
        f"is channel {max(rows,key=lambda r:r['ablation_delta_nmse_mean'])['channel']} "
        f"(+{max(r['ablation_delta_nmse_mean'] for r in rows):.5f}), "
        "showing that retention frequency and ablation importance are distinct rankings.", "",
        "## Retention and coefficient distributions", "",
        "Retention is the fraction of valid 16-ms latent time positions at which this channel has a nonzero retained coefficient. "
        "Ranking uses its count across the complete corpus, not the number of recordings containing the channel. "
        "Magnitude distributions condition on retained nonzero coefficients; the complement of retention is a point mass at zero. "
        "Absolute values are in learned latent units and are not comparable measures of importance across arbitrary encoder/decoder scalings.", "",
        "| Rank | Channel | Retained positions % | Share of all retained % | Absolute z: p05 / median / p95 | Absolute z: mean / SD | Negative % |",
        "|---:|---:|---:|---:|---|---|---:|"]
    for r in rows:
        lines.append(f"| {r['rank']} | {r['channel']} | {100*r['activation_frequency']:.2f} | {100*r['share_retained']:.2f} | "
            f"{r['abs_p05']:.5g} / {r['abs_p50']:.5g} / {r['abs_p95']:.5g} | {r['abs_mean']:.5g} / {r['abs_std']:.5g} | {100*r['negative_fraction']:.1f} |")
    lines += ["", "## Synthesis contribution and frequency content", "",
        "For the affine decoder g(z)=Az+b, the isolated contribution is **y_c=g(z_c)-g(0)**, not g(z_c). "
        "The full reconstruction is g(0)+sum_c y_c. The length-dependent bias waveform is included only once. "
        "All other retained values are fixed during channel ablation; top-k is not rerun. The reported ablation score is "
        "mean [(||xhat-y_c-x||²-||xhat-x||²)/||x||²]; positive values mean removal worsens waveform NMSE.", "",
        "Spectra use actual complete-recording contributions, duration-weighted Welch PSD (Hann 2048, overlap 1536, "
        "16 kHz, no detrending; 7.8125-Hz bins). The 5–95% spectral energy interval is a bandwidth summary, not a pitch range. "
        "Strong PSD peaks are local maxima within 12 dB of the largest peak, separated by at least three frequency bins. "
        "Interior unit-impulse peaks describe the decoder filter separately; actual peaks also depend on coefficient sequences. "
        "Channel energies are non-additive because these contributions need not be orthogonal.", "",
        "Attribution uses native FP32 convolutions with cuDNN/TF32 disabled in the decoder while retaining the original encoder precision and selected coefficients. "
        "This improves numerical consistency of the affine decomposition without changing weights. The correction pass checks "
        "the waveform difference from default decoding and reports it below.", "",
        "| Channel | Strong PSD peaks Hz (largest first) | Centroid Hz | 5–95% energy Hz | Energy <300 / 300–1000 / 1–3k / >3k % | Unit-impulse peak Hz | Ablation ΔNMSE |",
        "|---:|---:|---:|---|---|---:|---:|"]
    for r in rows:
        bands=" / ".join(f"{100*r['power_fraction_'+key]:.1f}" for key in ['below300','300to1000','1000to3000','above3000'])
        lines.append(f"| {r['channel']} | {r.get('peaks_within12db_hz',str(round(r['peak_hz'],1)))} | {r['centroid_hz']:.1f} | {r['energy05_hz']:.1f}–{r['energy95_hz']:.1f} | {bands} | {r['impulse_peak_hz']:.1f} | {r['ablation_delta_nmse_mean']:.6f} |")
    lines += ["", "## Voiced/unvoiced association", "",
        "Reference labels use [pYIN (Mauch & Dixon, 2014)](https://webspace.eecs.qmul.ac.uk/s.e.dixon/pub/2014/MauchDixon-PYIN-ICASSP2014.pdf) "
        "via [librosa 0.11](https://librosa.org/doc/0.11.0/generated/librosa.pyin.html): 50–600 Hz, 1024-sample centered frames and 256-sample hop. "
        "Low-energy frames below max(1e-4 RMS, peak frame RMS −35 dB) are excluded from the voiced/unvoiced comparison, not counted as unvoiced. "
        f"There are {v['v']:,} voiced, {v['u']:,} unvoiced and {v['low']:,} excluded low-energy valid time positions. "
        f"The baseline voiced fraction among included frames is {100*summary['baseline_voiced_fraction']:.2f}%.", "",
        "Raw phi is the point-biserial correlation between retention and the binary voiced flag, conditional on above-threshold energy. "
        "To reduce energy confounding, match frames within recording and 3-dB RMS bins with at least five frames of each state; "
        "weight each stratum by nV*nU/(nV+nU), aggregate within speaker, then average speakers equally. "
        f"Eligible strata contain {v['matched_v']:,} voiced and {v['matched_u']:,} unvoiced frames; "
        "the adjusted estimate therefore concerns their common energy support. "
        "Intervals are 2,000-resample paired-speaker percentile bootstrap intervals (seed 0) for this matched rate difference. "
        "They are descriptive, unadjusted for ten channel comparisons, and conditional on this fixed model and ranking. "
        "Positive difference means preferential retention in voiced frames at matched coarse energy, negative means unvoiced. "
        "These are exploratory associations with automatic labels, not ground-truth phonetic classes, independent features, or proof of a learned detector.", "",
        "| Channel | Retained in voiced % | Retained in unvoiced % | Raw phi | Energy-matched V−U difference, percentage points [95% CI] | Adjusted association |",
        "|---:|---:|---:|---:|---|---|"]
    for r in rows:
        lines.append(f"| {r['channel']} | {100*r['p_active_voiced']:.2f} | {100*r['p_active_unvoiced']:.2f} | {r['phi_voiced']:.3f} | "
            f"{100*r['matched_rate_difference']:+.2f} [{100*r['matched_ci_low']:+.2f}, {100*r['matched_ci_high']:+.2f}] | {r['adjusted_association']} |")
    lines += ["", "The CSV/JSON also retain sensitivity analyses: high-confidence pYIN labels (voiced probability ≥0.5 or unvoiced ≤0.1) "
        "and stable-state labels persisting ±4 latent frames. The encoder's 127.6-ms receptive field can span transitions, "
        "so a center-frame association does not establish a strictly localized phonetic response.", "",
        "![Contribution spectra and raw voicing rates](saf_channel_profiles.png)", "",
        "## Individual reconstructed-channel spectrograms", "",
        "Examples are selected by manifest speaker order and duration ≥6 s, before inspecting activations. "
        "Figures crop the first eight seconds **after** whole-recording 10% support selection and synthesis. "
        "Every row in each figure uses the same dB scale relative to the reference STFT peak; individual channels are not independently amplified. "
        "Frequency axes are logarithmic, 60–8000 Hz. Cyan/green dots on the reference mark automatic voiced/unvoiced frames. "
        "Spectrogram magnitudes do not add, even though the underlying affine contribution waveforms do.", "",
        "![Reference, reconstruction, and channels ranked 1–5](saf_channels_spectrograms_1.png)", "",
        "![Same recording, channels ranked 6–10](saf_channels_spectrograms_2.png)", "",
        "![Different speaker, channels ranked 1, 5 and 10](saf_channels_spectrograms_3.png)", "",
        f"Numerical isolated-versus-ablation identity: maximum absolute waveform discrepancy {summary['ablation_identity_max_absolute_error']:.3g} "
        "on the displayed full-recording examples. Differences arise from float32 arithmetic.", "",
        "Full distributions, recording coverage, band powers, bootstrap intervals and protocol hashes: "
        "[statistics CSV](saf_channel_statistics.csv), [analysis JSON](saf_channel_analysis.json), "
        "[all 256 channel rankings](saf_channel_ranking.csv). Raw example arrays and spectra stay under the run's analysis directory.", "",
        "Reproduce from the project root with `python tools/analyze_channels.py`; source/checkpoint paths can be changed with `--run`. "
        "This is a post-hoc characterization of the already evaluated test set, not a new independent test of an architecture selected using these findings."]
    if "decoder_precision_change_relative_rms" in summary:
        lines += ["", f"IEEE-FP32 versus default decoder: corpus relative RMS difference "
            f"{summary['decoder_precision_change_relative_rms']:.4g}; maximum absolute sample difference "
            f"{summary['decoder_precision_change_max_absolute']:.4g}. See "
            "[PyTorch's TF32 notes](https://docs.pytorch.org/docs/2.8/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)."]
    if "decomposition_max_absolute_error" in summary:
        lines += ["", f"Bias + ten isolated channels + the remaining channels also matches full decoding: "
                  f"maximum absolute discrepancy {summary['decomposition_max_absolute_error']:.3g} on the displayed recordings."]
    # Committed explanations live in the single README. Custom/run output
    # directories can still receive their automatically generated local report.
    if output.resolve() != Path(__file__).resolve().parents[1]/"documentation":
        (output/"saf_channel_analysis.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default="runs/librispeech_sparse_autoencoder_affine_10h_20261027T114554258463Z_seed0")
    parser.add_argument("--split",default="clean")
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--limit",type=int)
    parser.add_argument("--output",default="documentation")
    args=parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers and any --limit must be positive")
    analyze(args)
