import numpy as np
import torch
from lrac.config import ModelConfig
from lrac.models.sparse_autoencoder import SparseAutoencoder
from lrac.channel_analysis import affine_contributions, voicing_counts, binary_correlation, spectral_summary


def test_affine_channels_sum_with_bias_and_equal_ablation():
    torch.manual_seed(3)
    model = SparseAutoencoder(ModelConfig(kind="sparse_autoencoder", groups=1,
        channels=[2, 2, 4, 4], latent_width=8, activation="identity"), 16000).eval()
    x = torch.randn(1, 1, 1031)
    rep = model.sparsify(model.encode_dense(x), keep_fraction=.1)
    original_tf32 = torch.backends.cudnn.allow_tf32
    bias, full, parts = affine_contributions(model, rep, list(range(8)), chunk_size=3)
    assert torch.backends.cudnn.allow_tf32 == original_tf32
    assert np.allclose(bias + parts.sum(0), full, atol=3e-6)
    z = rep.concatenate().clone()
    z[..., 3] = 0
    with torch.no_grad():
        removed = model.decode(rep.with_values(z))[0, 0].numpy()
    assert np.allclose(full - removed, parts[3], atol=3e-6)
    assert len(full) == 1031


def test_voicing_excludes_low_energy_and_controls_energy_confounding():
    # Activation depends only on energy bin; voiced frames occupy the loud bin.
    voiced = np.array([True]*15 + [False]*5 + [True]*5 + [False]*15 + [False]*2)
    rms = np.array([.1]*20 + [.01]*20 + [0.]*2)
    active = (rms > .05)[:, None]
    labels = dict(voiced=voiced, audible=rms > 0, rms=rms, probability=voiced.astype(float))
    result = voicing_counts(active, labels)
    assert result["low"] == 2 and result["v"] == result["u"] == 20
    assert binary_correlation(result["av"][0], result["au"][0], 20, 20) > 0
    assert result["matched_weight"] > 0
    assert result["matched_difference_sum"][0] == 0


def test_spectral_summary_identifies_frequency_and_band():
    f = np.arange(1025) * 16000 / 2048
    p = np.zeros(1025)
    p[64] = 1
    result = spectral_summary(f, p)
    assert result["peak_hz"] == result["centroid_hz"] == 500
    assert result["power_fraction_300to1000"] == 1
    p[32] = .3
    p[100] = .001
    assert spectral_summary(f, p)["peaks_within12db_hz"] == "500.0 / 250.0"
