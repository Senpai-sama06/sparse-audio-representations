"""Run the frozen S-AF pitch intervention on unused development recordings."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import numpy as np
import torch
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lrac.pitch_intervention import (SR, SAMPLES, CORE, SHIFTS, TARGETS, SUPPORTS,
    analysis_filters, prepare_speaker, measure, spectral_prediction, normalize)
from lrac.train import load_run
from lrac.utils import file_hash, write_json, seed_all


DEFAULT_RUN = "runs/librispeech_sparse_autoencoder_affine_10h_20261027T114554258463Z_seed0"


def csv_write(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def setup(args):
    run = Path(args.run)
    local = run/"analysis"/"pitch_intervention"
    if args.speakers != 40:
        local /= f"smoke_{args.speakers}"
    local.mkdir(parents=True, exist_ok=True)
    output = Path(args.output) if args.speakers == 40 else local/"report"
    output.mkdir(parents=True, exist_ok=True)
    return run, local, output


def prepare(args):
    run, local, output = setup(args)
    cfg, model, state = load_run(run, "cpu")
    del model
    manifest = cfg.manifest(cfg.data.val_manifest)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    validation = run/"evaluation/validation/limit_200/fraction_0.1/metrics.csv"
    with validation.open(newline="") as stream:
        excluded = {row["id"] for row in csv.DictReader(stream)}
    groups = {}
    for row in rows:
        if row["id"] not in excluded:
            groups.setdefault(str(row["speaker"]), []).append(row)
    speakers = sorted(groups, key=int)[:args.speakers]
    expected = next(r for r in json.loads(Path("documentation/sparse_autoencoder_test_runs.json").read_text())["runs"]
                    if r["run_name"] == run.name)["checkpoint_sha256"]
    if file_hash(run/"checkpoints/best.pt") != expected:
        raise ValueError("Checkpoint differs from the frozen study")
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        purpose="Exploratory controlled intervention; development data, no confirmatory test claims",
        run=run.name, checkpoint_sha256=expected, checkpoint_step=state["step"],
        manifest_sha256=file_hash(manifest), split="dev-clean", speakers=speakers,
        excluded_checkpoint_validation_ids=sorted(excluded), excluded_count=len(excluded),
        sample_rate=SR, samples=SAMPLES, core_samples=list(CORE),
        target_f0_hz=TARGETS.tolist(), primary_f0_range_hz=[120, 240],
        support_fractions=list(SUPPORTS), phase_shifts_samples=SHIFTS.tolist(),
        precision="Native IEEE FP32; cuDNN disabled and CUDA matmul TF32 disabled",
        topk="Global magnitude top-k over all 48 x 256 coefficients of each 0.768-s excerpt",
        averaging="Equal average of phase powers/rates, then equal source-speaker average",
        ranking="RMS of retained coefficients including zeros, at every support; retention reported separately",
        synthesis="WORLD: change voiced F0 only; spectral envelope and aperiodicity held exactly fixed; same 10-ms edge fade and whole-excerpt RMS 0.05",
        source_selection="First eligible region in first 30 unused utterances per speaker; WORLD voiced >=98%, F0 CV <=0.15 over central 256ms; independent pYIN >=90% voiced and F0 agreement within5%; no model responses used",
        negative_selection="WORLD >=98% unvoiced and pYIN <=10% voiced; audible core. Signal-derived labels, no phonetic annotation",
        qc="Independent pYIN F0 within5% of target and >=80% voiced; strict sensitivity also requires coarse-envelope shape RMSE <=3dB relative to sham",
        envelope_check="Welch 2048/512; FFT4096; log-power cepstrum retain quefrency<=2ms; gain-free shape RMSE over300-6000Hz",
        predictions="Unfitted fixed analysis-filter PSD overlap; ideal harmonic-envelope overlap (ignores aperiodicity); exact interior time-domain kernel check",
        negative_controls="Real unvoiced speech, envelope-matched WORLD aperiodic synthesis, seeded Gaussian broadband noise; no physical F0 assigned to controls",
        versions={p:metadata.version(p) for p in ["torch", "numpy", "scipy", "librosa", "pyworld"]},
        source_sha256={p:file_hash(Path(p)) for p in ["lrac/pitch_intervention.py", "tools/analyze_pitch.py"]})
    if (local/"protocol.json").exists():
        old = json.loads((local/"protocol.json").read_text())
        for k in ["checkpoint_sha256", "manifest_sha256", "target_f0_hz", "speakers"]:
            if old[k] != protocol[k]:
                raise ValueError(f"Existing protocol differs: {k}")
    else:
        write_json(local/"protocol.json", protocol)
    completed = {}
    tasks = []
    for speaker in speakers:
        path = local/f"{speaker}.json"
        if path.exists() and (local/f"{speaker}.npz").exists():
            completed[speaker] = json.loads(path.read_text())
        else:
            tasks.append((speaker, groups[speaker], str(local)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(prepare_speaker, task):task[0] for task in tasks}
        for future in as_completed(futures):
            speaker = futures[future]
            completed[speaker] = future.result()
            print(f"PREPARE {len(completed)}/{len(speakers)} speaker={speaker}", flush=True)
            write_json(local/"progress.json", dict(phase="prepare", completed=len(completed), total=len(speakers)))
    cohort = [completed[s] for s in speakers]
    selected = [c[k]["id"] for c in cohort for k in ["voiced", "unvoiced"] if c[k]]
    if excluded.intersection(selected):
        raise ValueError("Source overlaps checkpoint validation")
    write_json(local/"cohort.json", cohort)
    csv_write(output/"saf_pitch_quality.csv", [q for c in cohort for q in c["checks"]])
    print("PREPARATION COMPLETE", flush=True)


def analyze(args):
    run, local, output = setup(args)
    protocol = json.loads((local/"protocol.json").read_text())
    if file_hash(run/"checkpoints/best.pt") != protocol["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    seed_all(0)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    _, model, _ = load_run(run, args.device)
    if model.config.activation != "identity":
        raise ValueError("Frozen affine S-AF required")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    filters, bias = analysis_filters(model)
    np.savez_compressed(local/"filters.npz", analysis=filters, bias=bias)
    cohort = json.loads((local/"cohort.json").read_text())
    for i, source in enumerate(cohort):
        speaker = source["speaker"]
        result_path = local/f"response_{speaker}.npz"
        if result_path.exists() and not args.force:
            print(f"CACHED {i+1}/{len(cohort)} {speaker}", flush=True)
            continue
        data = np.load(local/f"{speaker}.npz")
        power, rate, centered, psd, harmonic, errors = [], [], [], [], [], []
        for j, x in enumerate(data["waveforms"]):
            result = measure(model, x, bias, filters if j == 0 else None)
            power.append(result["power"])
            rate.append(result["rate"])
            centered.append(result["centered_power"])
            if result["filter_check_max_abs"] is not None:
                errors.append(result["filter_check_max_abs"])
            f0 = TARGETS[j-3] if 3 <= j < 3+len(TARGETS) else source["voiced"]["world_f0_hz"]
            p, h = spectral_prediction(x, filters, data["envelope"], f0)
            psd.append(p)
            harmonic.append(h)
        controls = [data["waveforms"][-1]]
        names = ["aperiodic"]
        if source["unvoiced"]:
            controls.append(data["unvoiced"])
            names.append("unvoiced")
        noise = normalize(np.random.default_rng(int(speaker)).standard_normal(SAMPLES))
        controls.append(noise)
        names.append("noise")
        control_results = [measure(model, x, bias) for x in controls]
        control_power = [r["power"] for r in control_results]
        control_centered = [r["centered_power"] for r in control_results]
        control_core_rms = [float(np.sqrt(np.mean(x[CORE[0]:CORE[1]].astype(float)**2))) for x in controls]
        np.savez_compressed(result_path, power=power, rate=rate, centered=centered,
            psd=psd, harmonic=harmonic, control_names=names, control_power=control_power,
            control_centered=control_centered,control_core_rms=control_core_rms,
            filter_check_max_abs=errors)
        print(f"ANALYZE {i+1}/{len(cohort)} speaker={speaker}", flush=True)
        write_json(local/"progress.json", dict(phase="analyze", completed=i+1, total=len(cohort)))
    print("ANALYSIS COMPLETE", flush=True)


def correlation(x, y):
    if len(x) < 3 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return None
    return float(stats.spearmanr(x, y).statistic)


def report(args):
    run, local, output = setup(args)
    cohort = json.loads((local/"cohort.json").read_text())
    protocol = json.loads((local/"protocol.json").read_text())
    csv_write(output/"saf_pitch_sources.csv", [dict(speaker=s["speaker"],
        voiced_id=s["voiced"]["id"],voiced_start_sample=s["voiced"]["start_sample"],
        original_pyin_f0_hz=s["voiced"]["pyin_f0_hz"],original_world_f0_hz=s["voiced"]["world_f0_hz"],
        unvoiced_id=s["unvoiced"]["id"] if s["unvoiced"] else None,
        unvoiced_start_sample=s["unvoiced"]["start_sample"] if s["unvoiced"] else None)
        for s in cohort])
    results = [np.load(local/f"response_{s['speaker']}.npz") for s in cohort]
    power = np.stack([r["power"] for r in results])
    rate = np.stack([r["rate"] for r in results])
    centered = np.stack([r["centered"] for r in results])
    psd = np.stack([r["psd"] for r in results])
    harmonic = np.stack([r["harmonic"] for r in results])
    responses = np.sqrt(power.mean(0))
    rates = rate.mean(0)
    filters = np.load(local/"filters.npz")["analysis"]
    fft = np.abs(np.fft.rfft(filters, n=16384, axis=-1))**2
    freq = np.fft.rfftfreq(16384, 1/SR)
    peaks = freq[np.argmax(fft, axis=1)]
    order = np.lexsort((np.arange(256), peaks))
    checks = [c["checks"][3:3+len(TARGETS)] for c in cohort]
    qc = np.array([[q["pitch_pass"] for q in c] for c in checks])
    strict = np.array([[q["strict_pass"] for q in c] for c in checks])
    primary = (TARGETS >= 120) & (TARGETS <= 240)
    masks = {"all":np.ones_like(qc), "pitch_qc":qc, "strict_qc":strict}
    statistics = []
    bootstrap = np.random.default_rng(0).integers(0,len(cohort),size=(2000,len(cohort)))
    for c in range(256):
        row = dict(channel=c, analysis_peak_hz=float(peaks[c]))
        for label, mask in masks.items():
            mask = mask & primary[None]
            observed = centered[:, 3:3+len(TARGETS), c][mask]
            for name, predicted in [("psd", psd), ("harmonic", harmonic)]:
                pred = predicted[:, 3:3+len(TARGETS), c][mask]
                rho = correlation(observed, pred)
                row[f"{name}_{label}_spearman"] = rho
                row[f"{name}_{label}_relative_rmse"] = float(np.sqrt(np.mean((observed-pred)**2))/max(np.sqrt(np.mean(observed**2)), 1e-20)) if len(observed) else None
            row[f"{label}_observations"] = len(observed)
        # Within-source associations prevent between-speaker gain differences
        # from supplying all the apparent explanatory power.
        for name, predicted in [("psd", psd), ("harmonic", harmonic)]:
            rho = [correlation(centered[i, 3:3+len(TARGETS), c][primary & qc[i]],
                               predicted[i, 3:3+len(TARGETS), c][primary & qc[i]]) for i in range(len(cohort))]
            usable = [r for r in rho if r is not None]
            row[f"{name}_median_within_source_spearman"] = float(np.median(usable)) if usable else None
            row[f"{name}_within_source_speakers"] = len(usable)
            if usable:
                numeric_rho = np.array([r if r is not None else np.nan for r in rho])
                lo,hi = np.quantile(np.nanmedian(numeric_rho[bootstrap],axis=1),[.025,.975])
                row[f"{name}_within_source_median_ci_low"] = float(lo)
                row[f"{name}_within_source_median_ci_high"] = float(hi)
            else:
                row[f"{name}_within_source_median_ci_low"] = None
                row[f"{name}_within_source_median_ci_high"] = None
            rho_strict = [correlation(centered[i,3:3+len(TARGETS),c][primary & strict[i]],
                                     predicted[i,3:3+len(TARGETS),c][primary & strict[i]])
                          for i in range(len(cohort)) if (primary & strict[i]).sum() >= 8]
            usable_strict = [r for r in rho_strict if r is not None]
            row[f"{name}_strict_median_within_source_spearman"] = float(np.median(usable_strict)) if usable_strict else None
            row[f"{name}_strict_sources_at_least8_targets"] = len(usable_strict)
            for label, quality in [("pitch_qc",qc),("strict_qc",strict)]:
                squared_error, variation = 0., 0.
                for i in range(len(cohort)):
                    keep = primary & quality[i]
                    obs = centered[i,3:3+len(TARGETS),c][keep]
                    pred = predicted[i,3:3+len(TARGETS),c][keep]
                    if len(obs) >= 3:
                        squared_error += float(np.sum((obs-pred)**2))
                        variation += float(np.sum((obs-obs.mean())**2))
                row[f"{name}_{label}_unfitted_within_source_r2"] = 1-squared_error/variation if variation > 1e-30 else None
        nearest = np.maximum(1, np.round(peaks[c]/TARGETS)).astype(int)
        distance = np.abs(nearest*TARGETS-peaks[c])
        rho = [correlation(-distance[primary & qc[i]],
                           centered[i,3:3+len(TARGETS),c][primary & qc[i]]) for i in range(len(cohort))]
        usable = [r for r in rho if r is not None]
        row["nearest_harmonic_median_within_source_spearman"] = float(np.median(usable)) if usable else None
        statistics.append(row)
    csv_write(output/"saf_pitch_channels.csv", statistics)
    harmonic_rows = []
    for j, target in enumerate(TARGETS):
        keep = qc[:,j]
        for c in range(256):
            m = max(1, int(np.round(peaks[c]/target)))
            harmonic_rows.append(dict(channel=c, target_f0_hz=float(target),
                analysis_peak_hz=float(peaks[c]), nearest_harmonic_number=m,
                nearest_harmonic_hz=float(m*target), distance_hz=float(abs(m*target-peaks[c])),
                sources_pitch_qc=int(keep.sum()),
                centered_rms_all=float(np.sqrt(centered[:,j+3,c].mean())),
                centered_rms_pitch_qc=float(np.sqrt(centered[keep,j+3,c].mean())) if keep.any() else None,
                fixed_psd_rms_all=float(np.sqrt(psd[:,j+3,c].mean())),
                harmonic_comb_rms_all=float(np.sqrt(harmonic[:,j+3,c].mean()))))
    csv_write(output/"saf_pitch_harmonics.csv", harmonic_rows)
    # Full all-channel strengths/rates, all variants; not just the highlighted IDs.
    variants = cohort[0]["variants"]
    rows, ranks, agreements = [], [], []
    original_ranks = {}
    for s, support in enumerate(SUPPORTS):
        original_order = np.argsort(-responses[0, s], kind="stable")
        original_ranks[s] = stats.rankdata(-responses[0, s], method="average")
        for j, name in enumerate(variants):
            ranking = np.argsort(-responses[j, s], kind="stable")
            rank = stats.rankdata(-responses[j, s], method="average")
            paired = [correlation(stats.rankdata(-power[i,0,s],method="average"),
                                  stats.rankdata(-power[i,j,s],method="average")) for i in range(len(cohort))]
            paired = [p for p in paired if p is not None]
            sham_rank = stats.rankdata(-responses[1,s],method="average")
            agreements.append(dict(support=support, variant=name,
                rank_spearman=correlation(original_ranks[s], rank),
                sham_rank_spearman=correlation(sham_rank,rank),
                median_source_rank_spearman=float(np.median(paired)) if paired else None,
                p10_source_rank_spearman=float(np.quantile(paired,.1)) if paired else None,
                p90_source_rank_spearman=float(np.quantile(paired,.9)) if paired else None,
                top10_overlap=len(set(original_order[:10]) & set(ranking[:10])),
                top32_overlap=len(set(original_order[:32]) & set(ranking[:32]))))
            for c in range(256):
                rows.append(dict(variant=name, target_f0_hz=float(TARGETS[j-3]) if 3 <= j < 3+len(TARGETS) else "",
                    support=support, channel=c, coefficient_rms=float(responses[j,s,c]),
                    retention_rate=float(rates[j,s,c]), analysis_peak_hz=float(peaks[c])))
                ranks.append(dict(support=support, variant=name, channel=c, rank=float(rank[c]),
                    original_rank=float(original_ranks[s][c]), rank_change=float(rank[c]-original_ranks[s][c])))
    csv_write(output/"saf_pitch_activations.csv", rows)
    csv_write(output/"saf_pitch_rankings.csv", ranks)
    csv_write(output/"saf_pitch_rank_comparison.csv", agreements)
    # Main heatmap: row-normalization exposes response shapes, not relative gains.
    fig, axes = plt.subplots(1, 4, figsize=(17, 12), sharey=True, layout="constrained")
    arrays = [responses[3:3+len(TARGETS),2].T,
              np.sqrt(centered[:,3:3+len(TARGETS)].mean(0)).T,
              *[responses[3:3+len(TARGETS),s].T for s in range(2)]]
    for ax, data, title in zip(axes, arrays, ["Before top-k = full support: raw RMS", "Before top-k: bias-centered RMS", "1% support: retained RMS", "10% support: retained RMS"]):
        data = data / np.maximum(data.max(1, keepdims=True), 1e-15)
        image = ax.imshow(data[order], aspect="auto", origin="lower", extent=(77.5,362.5,-.5,255.5), vmin=0, vmax=1, cmap="magma")
        ax.set(title=title, xlabel="Target F0 (Hz)")
        ax.axvline(120, color="cyan", lw=.7)
        ax.axvline(240, color="cyan", lw=.7)
        ax.set_yticks(np.arange(0,256,32), [str(order[i]) for i in range(0,256,32)])
    axes[0].set_ylabel("All 256 channels, sorted by analysis-filter peak (tick = channel ID)")
    fig.colorbar(image, ax=list(axes), label="RMS / this channel's maximum across targets", fraction=.02)
    fig.suptitle(f"Frozen S-AF | {len(cohort)} development speakers | all manipulations shown; QC reported separately")
    fig.savefig(output/"saf_pitch_heatmap.png", dpi=160)
    plt.close(fig)
    # A separate retention view avoids conflating coefficient strength with rate.
    fig, axes = plt.subplots(1,3,figsize=(15,11),sharey=True,layout="constrained")
    for s, ax in enumerate(axes[:2]):
        image = ax.imshow(rates[3:3+len(TARGETS),s].T[order],origin="lower",aspect="auto",
                          extent=(77.5,362.5,-.5,255.5),vmin=0,vmax=1,cmap="viridis")
        ax.set(title=f"{SUPPORTS[s]:.0%} support: nonzero retained fraction",xlabel="Target F0 (Hz)")
    strict_centered = np.array([np.sqrt(centered[strict[:,j],j+3].mean(0)) if strict[:,j].any()
                               else np.full(256,np.nan) for j in range(len(TARGETS))]).T
    strict_centered /= np.maximum(np.nanmax(strict_centered,axis=1,keepdims=True),1e-15)
    strength_image = axes[2].imshow(strict_centered[order],origin="lower",aspect="auto",extent=(77.5,362.5,-.5,255.5),vmin=0,vmax=1,cmap="magma")
    axes[2].set(title="Strict QC: centered RMS / row max",xlabel="Target F0 (Hz)")
    axes[0].set_ylabel("Channels sorted by analysis-filter peak")
    fig.colorbar(image,ax=list(axes[:2]),fraction=.03,label="Nonzero retained fraction")
    fig.colorbar(strength_image,ax=axes[2],fraction=.05,label="Centered RMS / row maximum")
    fig.suptitle("Retention frequency and independent pitch + envelope quality sensitivity; full support selects all coordinates")
    fig.savefig(output/"saf_pitch_retention.png",dpi=160)
    plt.close(fig)
    # Four requested hypotheses; full harmonic comb handles multipeak filters.
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), layout="constrained")
    for k, c in enumerate([167,139,53,246]):
        ax = axes[k,0]
        observed = centered[:,3:3+len(TARGETS),c].mean(0)
        for name, values, style in [("Observed",observed,"-"), ("Fixed PSD overlap",psd[:,3:3+len(TARGETS),c].mean(0),"--"),
                                     ("Fixed envelope + harmonic comb",harmonic[:,3:3+len(TARGETS),c].mean(0),":")]:
            ax.plot(TARGETS, values/max(observed.max(),1e-20), style, label=name)
        for m in range(1, int(peaks[c]//80)+1):
            crossing = peaks[c]/m
            if 80 <= crossing <= 360:
                ax.axvline(crossing, color="grey", alpha=.3)
        ax.set(title=f"Channel {c}: analysis peak {peaks[c]:.1f} Hz", ylabel="Power / observed peak", xlabel="Target F0 (Hz)")
        ax.legend(fontsize=7)
        ax = axes[k,1]
        gain = fft[c]/max(fft[c].max(),1e-20)
        ax.plot(freq,10*np.log10(np.maximum(gain,1e-8)))
        ax.set(xlim=(0,1000), ylim=(-70,2), title="Effective analysis-filter gain", xlabel="Frequency (Hz)", ylabel="Relative dB")
    fig.savefig(output/"saf_pitch_harmonics.png", dpi=160)
    plt.close(fig)
    # Nearest-peak distance is a diagnostic; multiband filters require the full comb.
    fig, axes = plt.subplots(1,4,figsize=(16,4),layout="constrained")
    for ax,c in zip(axes,[167,139,53,246]):
        nearest = np.maximum(1,np.round(peaks[c]/TARGETS))
        distance = np.abs(nearest*TARGETS-peaks[c])
        observed = centered[:,3:3+len(TARGETS),c].mean(0)
        image = ax.scatter(distance,observed/max(observed.max(),1e-20),c=TARGETS,cmap="viridis",s=22)
        ax.set(title=f"Channel {c}",xlabel="Distance to nearest harmonic (Hz)",ylabel="Power / peak")
    fig.colorbar(image,ax=list(axes),label="Target F0 (Hz)",fraction=.025)
    fig.savefig(output/"saf_pitch_harmonic_distance.png",dpi=160)
    plt.close(fig)
    # Negative controls have no physical F0; compare their channel profiles.
    controls, control_gains, control_levels = {}, {}, {}
    for r in results:
        for index,(name, p) in enumerate(zip(r["control_names"],r["control_power"])):
            controls.setdefault(str(name),[]).append(p)
            if "control_centered" in r:
                level = float(r["control_core_rms"][index])
                control_gains.setdefault(str(name),[]).append(r["control_centered"][index]/max(level**2,1e-20))
                control_levels.setdefault(str(name),[]).append(level)
    fig, axes = plt.subplots(4,1, figsize=(14,12), sharex=True, layout="constrained")
    control_rows = []
    for s, support in enumerate(SUPPORTS):
        axes[s].plot(np.arange(256),responses[0,s,order], label="Original voiced speech")
        for name, p in controls.items():
            values = np.sqrt(np.mean(p,axis=0))[s]
            axes[s].plot(np.arange(256),values[order], label=f"{name} (n={len(p)})", alpha=.8)
            control_rows += [dict(control=name, sources=len(p), support=support, channel=c, coefficient_rms=float(values[c])) for c in range(256)]
        axes[s].set(title=f"{support:.0%} support", ylabel="Coefficient RMS")
        axes[s].legend(fontsize=8)
    original_levels = np.array([s["checks"][0]["core_rms"] for s in cohort])
    axes[3].plot(np.arange(256),np.sqrt(np.mean(centered[:,0]/original_levels[:,None]**2,axis=0))[order],label="Original voiced speech")
    for name, values in control_gains.items():
        axes[3].plot(np.arange(256),np.sqrt(np.mean(values,axis=0))[order],label=name,alpha=.8)
    axes[3].set(title="Before selection: bias-centered response / core input RMS",ylabel="Effective response gain")
    axes[3].legend(fontsize=8)
    axes[-1].set_xlabel("Channel position, sorted by analysis-filter peak (same order as heatmap)")
    fig.savefig(output/"saf_pitch_controls.png",dpi=160)
    plt.close(fig)
    csv_write(output/"saf_pitch_controls.csv", control_rows)
    if control_gains:
        csv_write(output/"saf_pitch_control_gains.csv",[dict(control=name,channel=c,
            response_gain=float(np.sqrt(np.mean(p,axis=0))[c]),
            white_noise_expected_gain=float(np.linalg.norm(filters[c])),
            mean_core_rms=float(np.mean(control_levels[name])))
            for name,p in control_gains.items() for c in range(256)])
    # Independent pitch and envelope checks show success/failure across the sweep.
    fig, axes = plt.subplots(1,3,figsize=(14,4), layout="constrained")
    measured = np.array([[q["pyin_f0_hz"] or np.nan for q in c] for c in checks])
    envelope = np.array([[q["envelope_shape_rmse_db"] for q in c] for c in checks])
    for row in measured:
        axes[0].plot(TARGETS,row,alpha=.15,color="tab:blue")
    axes[0].plot(TARGETS,TARGETS,"k--")
    axes[0].set(xlabel="Target F0 (Hz)",ylabel="Independent pYIN F0 (Hz)")
    axes[1].plot(TARGETS,qc.mean(0),label="Pitch QC")
    axes[1].plot(TARGETS,strict.mean(0),label="Pitch + envelope QC")
    axes[1].set(xlabel="Target F0 (Hz)",ylabel="Fraction passing",ylim=(0,1.02))
    axes[1].legend()
    axes[2].plot(TARGETS,np.median(envelope,axis=0))
    axes[2].fill_between(TARGETS,np.quantile(envelope,.1,axis=0),np.quantile(envelope,.9,axis=0),alpha=.2)
    axes[2].axhline(3,color="k",ls="--")
    axes[2].set(xlabel="Target F0 (Hz)",ylabel="Coarse envelope-shape RMSE (dB)",title="Median and 10–90% of sources")
    fig.savefig(output/"saf_pitch_quality.png",dpi=160)
    plt.close(fig)
    summary = dict(protocol=protocol, source_count=len(cohort), unvoiced_count=len(controls.get("unvoiced",[])),
        intervention_count=len(cohort)*len(TARGETS),
        pitch_qc_fraction=float(qc.mean()), primary_pitch_qc_fraction=float(qc[:,primary].mean()),
        strict_qc_fraction=float(strict.mean()), primary_strict_qc_fraction=float(strict[:,primary].mean()),
        primary_envelope_rmse_median_db=float(np.median(envelope[:,primary])),
        primary_f0_relative_error_median=float(np.nanmedian(np.abs(measured[:,primary]/TARGETS[primary]-1))),
        exact_filter_check_max_abs=float(max(r["filter_check_max_abs"].max() for r in results)),
        median_psd_within_source_spearman=float(np.median([r["psd_median_within_source_spearman"] for r in statistics if r["psd_median_within_source_spearman"] is not None])),
        median_harmonic_within_source_spearman=float(np.median([r["harmonic_median_within_source_spearman"] for r in statistics if r["harmonic_median_within_source_spearman"] is not None])),
        median_psd_strict_within_source_spearman=float(np.median([r["psd_strict_median_within_source_spearman"] for r in statistics if r["psd_strict_median_within_source_spearman"] is not None])),
        median_harmonic_strict_within_source_spearman=float(np.median([r["harmonic_strict_median_within_source_spearman"] for r in statistics if r["harmonic_strict_median_within_source_spearman"] is not None])),
        median_psd_pitch_qc_unfitted_within_source_r2=float(np.median([r["psd_pitch_qc_unfitted_within_source_r2"] for r in statistics if r["psd_pitch_qc_unfitted_within_source_r2"] is not None])),
        median_psd_strict_qc_unfitted_within_source_r2=float(np.median([r["psd_strict_qc_unfitted_within_source_r2"] for r in statistics if r["psd_strict_qc_unfitted_within_source_r2"] is not None])),
        channels_positive_psd_pitch_qc_unfitted_within_source_r2=sum(r["psd_pitch_qc_unfitted_within_source_r2"] > 0 for r in statistics if r["psd_pitch_qc_unfitted_within_source_r2"] is not None),
        channels_positive_psd_strict_qc_unfitted_within_source_r2=sum(r["psd_strict_qc_unfitted_within_source_r2"] > 0 for r in statistics if r["psd_strict_qc_unfitted_within_source_r2"] is not None),
        noise_gain_spearman=correlation(np.linalg.norm(filters,axis=1),np.sqrt(np.mean(control_gains["noise"],axis=0))) if control_gains else None,
        noise_gain_median_relative_error=float(np.median(np.abs(np.sqrt(np.mean(control_gains["noise"],axis=0))/np.linalg.norm(filters,axis=1)-1))) if control_gains else None,
        generated_utc=datetime.now(timezone.utc).isoformat(),
        execution_notes=["Two-speaker smoke run was separate from the 40-speaker experiment.",
            "Preparation initially stopped when WORLD changed its voicing decision after cropping. The same source-only eligibility check was moved into the candidate search; 37 already accepted sources were retained and the three unfinished speakers were completed. No model-response-based source selection.",
            "Preparation protocol is preserved with its original source hashes. Current measurement/report hashes are recorded separately."],
        analysis_source_sha256={p:file_hash(Path(p)) for p in ["lrac/pitch_intervention.py", "tools/analyze_pitch.py"]})
    write_json(output/"saf_pitch_analysis.json",summary)
    write_json(local/"progress.json",dict(phase="complete",completed=len(cohort),total=len(cohort)))
    print(json.dumps({k:v for k,v in summary.items() if k not in ["protocol","analysis_source_sha256"]},indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--stage", choices=["prepare","analyze","report","all"],default="all")
    parser.add_argument("--speakers", type=int,default=40)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--output",default="documentation")
    parser.add_argument("--force",action="store_true",help="Recompute cached model responses")
    args = parser.parse_args()
    for name, function in [("prepare",prepare),("analyze",analyze),("report",report)]:
        if args.stage in [name,"all"]:
            function(args)
