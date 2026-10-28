import numpy as np
import torch
from lrac.config import ModelConfig
from lrac.models.sparse_autoencoder import SparseAutoencoder
from lrac.pitch_intervention import (analysis_filters, interior_indices, normalize,
                                     spectral_prediction, SAMPLES, SHIFTS, CORE, RADIUS)


def test_effective_analysis_filter_predicts_unseen_input_interior():
    torch.manual_seed(11)
    model = SparseAutoencoder(ModelConfig(kind="sparse_autoencoder", groups=1,
        channels=[2, 2, 4, 4], latent_width=5, activation="identity"), 16000).eval()
    filters, bias = analysis_filters(model)
    x = torch.randn(1, 1, SAMPLES)
    with torch.no_grad():
        z = model.encoder(x)[0].T.numpy()
    indices = interior_indices(0)
    windows = np.stack([x[0, 0, t*256-RADIUS:t*256+RADIUS+1].numpy() for t in indices])
    predicted = windows @ filters.T + bias
    assert np.allclose(predicted, z[indices], atol=2e-6)
    # Wrongly treating untied decoder atoms as encoder filters is not tested here:
    # the extracted gradient must reconstruct the actual analysis coefficients.


def test_phase_average_never_scores_outside_the_voiced_core():
    for shift in SHIFTS:
        positions = interior_indices(shift)*256-shift
        assert len(positions) >= 8
        assert np.all(positions-RADIUS >= CORE[0])
        assert np.all(positions+RADIUS < CORE[1])


def test_overlap_prediction_distinguishes_fixed_frequency_filters():
    sr = 16000
    t = np.arange(SAMPLES)/sr
    x = normalize(np.sin(2*np.pi*250*t))
    n = np.arange(-RADIUS, RADIUS+1)
    window = np.hanning(len(n))
    filters = np.stack([window*np.cos(2*np.pi*f*n/sr) for f in [250,700]])
    power = spectral_prediction(x, filters)
    assert power[0] > 1000*power[1]
    assert abs(np.sqrt(np.mean(x.astype(float)**2))-.05) < 1e-8
