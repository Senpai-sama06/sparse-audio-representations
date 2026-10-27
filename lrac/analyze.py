from pathlib import Path
import csv
import numpy as np
import soundfile as sf
import torch
from scipy.optimize import linear_sum_assignment
from scipy.linalg import subspace_angles
from lrac.data.datasets import AudioDataset
from lrac.data.synthetic import generate
from lrac.data.preprocessing import load_audio
from lrac.losses import functional_interaction
from lrac.train import load_run
from lrac.utils import write_json


def acoustic_references(signal, sample_rate, hop=128):
    """Acoustic proxies, explicitly not ground truth feature labels."""
    import librosa
    x = np.asarray(signal)
    if len(x) < 512:
        x = np.pad(x, (0, 512 - len(x)))
    stft = np.abs(librosa.stft(x, n_fft=512, hop_length=hop))
    energy = librosa.feature.rms(S=stft, frame_length=512)[0]
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sample_rate)[0]
    flatness = librosa.feature.spectral_flatness(S=stft)[0]
    f0 = librosa.yin(x, fmin=65, fmax=min(600, sample_rate / 4), sr=sample_rate, frame_length=512, hop_length=hop)
    return {"energy": energy, "centroid": centroid, "flatness": flatness, "yin_f0_proxy": f0}


def _correlation(a, b):
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def analyze(run, split="validation", limit=None, device="auto", checkpoint="best.pt"):
    cfg, model, state = load_run(run, device, checkpoint)
    filename = cfg.data.val_manifest if split == "validation" else cfg.data.test_manifests[split]
    dataset = AudioDataset(cfg.manifest(filename), cfg.data.sample_rate, limit=limit or cfg.evaluation.analysis_examples)
    out = Path(run) / "analysis" / split
    if checkpoint != "best.pt":
        out = Path(run) / "analysis" / Path(checkpoint).stem / split
    (out / "audio").mkdir(parents=True, exist_ok=True)
    target_device = next(model.parameters(), next(model.buffers(), torch.empty(0))).device
    group_statistics, reference_correlations, interventions, factor_rows, component_alignment = [], [], [], [], []
    all_values, signatures = [], []
    for index in range(len(dataset)):
        signal, metadata = dataset[index]
        # Analysis is bounded, unlike full-utterance reconstruction evaluation.
        signal = signal[..., :round(cfg.data.crop_seconds * cfg.data.sample_rate)]
        x = signal[None].to(target_device)
        with torch.no_grad():
            representation = model.encode(x)
            values = representation.concatenate()
            baseline = model.decode(representation)
        all_values.append(values[0].cpu().numpy())
        refs = acoustic_references(signal[0].numpy(), cfg.data.sample_rate)
        known_components = generate(metadata)[1] if metadata["corpus"] == "synthetic" else {}
        coordinates = np.arange(values.shape[1]) * representation.timing["hop"] / cfg.data.sample_rate
        if "left" in representation.timing:
            coordinates += (cfg.model.frame_length / 2 - representation.timing["left"]) / cfg.data.sample_rate
        elif "block_length" in representation.timing:
            coordinates += cfg.model.frame_length / (2 * cfg.data.sample_rate)
            if representation.timing["tail_lengths"][0]:
                coordinates[-1] -= (cfg.model.frame_length - representation.timing["tail_lengths"][0]) / (2 * cfg.data.sample_rate)
        signature = []
        for k, (name, group) in enumerate(representation.groups.items()):
            activity = group[0].square().mean(-1).sqrt().cpu().numpy()
            stats = {"id": metadata["id"], "group": name, "rms": float(group.square().mean().sqrt()), "variance": float(group.var(unbiased=False)), "active_fraction": float((group.abs() > 1e-8).float().mean())}
            group_statistics.append(stats)
            row = {"id": metadata["id"], "group": name}
            for ref_name, ref in refs.items():
                interp = np.interp(coordinates, np.arange(len(ref)) * 128 / cfg.data.sample_rate, ref)
                row[ref_name] = _correlation(activity, interp)
            reference_correlations.append(row)
            signature.append([row[key] or 0 for key in refs])
            for operation in ["remove", "perturb"]:
                changed = {key: value.clone() for key, value in representation.groups.items()}
                if operation == "remove":
                    changed[name].zero_()
                else:
                    changed[name] += .1 * group.std(unbiased=False).clamp_min(1e-8)
                from lrac.models import Representation
                modified = Representation(changed, representation.lengths, representation.sample_rate, representation.original_samples, representation.timing)
                with torch.no_grad():
                    estimate = model.decode(modified)
                difference = float((estimate - baseline).square().mean())
                interventions.append({"id": metadata["id"], "group": name, "operation": operation, "change_mse": difference, "reference_mse": float((estimate - x).square().mean())})
                if operation == "remove":
                    contribution = (baseline - estimate)[0, 0].cpu().numpy()
                    for component_name, component in known_components.items():
                        # Correlations are descriptive; nonlinear contributions need
                        # not add up, and a matching label does not prove recovery.
                        reference = component[:len(contribution)]
                        if len(reference) == len(contribution):
                            component_alignment.append({"id": metadata["id"], "group": name, "component": component_name, "correlation": _correlation(contribution, reference)})
                if index < min(3, cfg.evaluation.audio_examples):
                    sf.write(out / "audio" / f"example_{index}_{name}_{operation}.wav", estimate[0, 0].cpu().numpy(), cfg.data.sample_rate, subtype="FLOAT")
        signatures.append(signature)
        if metadata["corpus"] == "synthetic":
            _, _, factors = generate(metadata)
            factor_rows.append({"id": metadata["id"], **factors, **{f"group_{k}_rms": float(v.square().mean().sqrt()) for k, v in enumerate(representation.groups.values())}})
    combined = np.concatenate(all_values)
    np.savez_compressed(out / "activations.npz", mean=combined.mean(0), variance=combined.var(0), active_fraction=(np.abs(combined) > 1e-8).mean(0), signatures=np.asarray(signatures).mean(0))
    details = {"checkpoint_step": state["step"], "examples": len(dataset), "inactive_coordinates": int(((np.abs(combined) > 1e-8).mean(0) < .001).sum()), "acoustic_references_are_proxies": True, "group_statistics": group_statistics, "acoustic_correlations": reference_correlations, "interventions": interventions, "synthetic_factors": factor_rows, "synthetic_component_alignment": component_alignment}
    if hasattr(model, "matrix"):
        matrix = model.matrix().detach().cpu().numpy()
        np.savez_compressed(out / "dictionary.npz", atoms=matrix)
        normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=0, keepdims=True), 1e-12)
        from sklearn.cluster import KMeans
        clusters = KMeans(n_clusters=cfg.model.groups, n_init=10, random_state=cfg.train.seed).fit_predict(normalized.T)
        np.savez_compressed(out / "atom_clusters.npz", labels=clusters)
        details["post_training_cluster_sizes"] = np.bincount(clusters, minlength=cfg.model.groups).tolist()
        gram = normalized.T @ normalized
        details["maximum_atom_coherence"] = float(np.max(np.abs(gram - np.eye(gram.shape[0]))))
        if cfg.model.kind == "convolutional":
            shifted = {}
            for shift in [1, cfg.model.conv_hop, cfg.model.conv_length // 2]:
                cross = normalized[shift:].T @ normalized[:-shift]
                shifted[str(shift)] = float(np.max(np.abs(cross)))
            details["translated_kernel_max_correlations"] = shifted
    elif cfg.model.groups > 1:
        # Diagnostic use only: small derivative windows; no full utterance Jacobian.
        z = values[:1].detach().requires_grad_(True)
        width = min(2048, representation.original_samples)
        with torch.enable_grad():
            functional = functional_interaction(lambda v: model.decode(representation.with_values(v))[..., :width], z, cfg.model.groups, cfg.train.functional_probes)
        details["functional_interaction_estimate"] = float(functional.detach())
    else:
        details["functional_interaction_estimate"] = None
        details["functional_interaction_status"] = "not applicable to an ungrouped representation"
    write_json(out / "diagnostics.json", details)
    print(f"ANALYSIS={out}", flush=True)
    return out


def compare_stability(run_a, run_b, split="validation"):
    a, b = Path(run_a) / "analysis" / split, Path(run_b) / "analysis" / split
    if (a / "dictionary.npz").exists() and (b / "dictionary.npz").exists():
        da, db = np.load(a / "dictionary.npz")["atoms"], np.load(b / "dictionary.npz")["atoms"]
        if da.shape[0] != db.shape[0]:
            return {"comparable": False, "reason": "Different signal spaces"}
        na = da / np.maximum(np.linalg.norm(da, axis=0), 1e-12)
        nb = db / np.maximum(np.linalg.norm(db, axis=0), 1e-12)
        correlations = np.abs(na.T @ nb)
        ia, ib = linear_sum_assignment(-correlations)
        # Full spans can coincide trivially; report the leading atom subspace as well.
        rank = min(32, da.shape[0], da.shape[1], db.shape[1])
        return {"comparable": True, "mean_matched_atom_correlation": float(correlations[ia, ib].mean()), "leading_subspace_angles_radians": subspace_angles(da[:, :rank], db[:, :rank]).tolist(), "full_span_angles_radians": subspace_angles(da, db).tolist()}
    sa, sb = np.load(a / "activations.npz")["signatures"], np.load(b / "activations.npz")["signatures"]
    distances = np.linalg.norm(sa[:, None] - sb[None], axis=-1)
    ia, ib = linear_sum_assignment(distances)
    return {"comparable": True, "group_signature_mean_distance": float(distances[ia, ib].mean()), "interpretation": "Diagnostic signature agreement, not identified-factor equivalence"}
