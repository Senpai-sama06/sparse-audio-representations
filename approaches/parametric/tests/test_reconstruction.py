"""Boundary and numerical controls for the actual decomposition pipeline."""
import json

import numpy as np
import pytest
import soundfile as sf

from approaches.parametric.decomposition import Decomposition, ModeDecomposer, EMDDecomposer, VMDDecomposer
from approaches.parametric.reconstruction import WindowedReconstructor


class IdentityDecomposer(ModeDecomposer):
    name = "identity"

    def decompose(self, signal):
        return Decomposition(signal[None].copy(), np.zeros_like(signal))


@pytest.mark.parametrize("length,window,overlap", [(1, 32, .5), (17, 55, .5), (103, 32, 0),
                                                    (103, 32, .5), (103, 33, .9)])
def test_overlap_add_preserves_edges_length_and_gain(length, window, overlap):
    x = np.random.default_rng(6).normal(size=length)
    x[-1] = 3.0
    result = WindowedReconstructor(window, overlap).reconstruct(x, IdentityDecomposer())
    np.testing.assert_allclose(result.modes, x, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result.modes_plus_residual_control, x, rtol=1e-13, atol=1e-13)


def test_emd_native_residue_restores_the_input():
    pytest.importorskip("PyEMD")
    t = np.arange(641) / 16000
    x = .3 + .2 * np.sin(2 * np.pi * 370 * t) + .04 * np.sin(2 * np.pi * 1400 * t)
    result = EMDDecomposer().decompose(x)
    np.testing.assert_allclose(result.reconstruct(include_residual=True), x, atol=1e-12)
    assert result.diagnostics["residual_kind"] == "native_emd_residue"
    assert np.linalg.norm(result.residual) > 0
    silent = EMDDecomposer().decompose(np.zeros(321))
    np.testing.assert_array_equal(silent.reconstruct(include_residual=True), np.zeros(321))


def test_vmd_odd_tail_and_silence_are_preserved():
    pytest.importorskip("vmdpy")
    t = np.arange(321) / 16000
    x = .2 * np.sin(2 * np.pi * 300 * t)
    x[-1] = .8
    backend = VMDDecomposer(modes=3)
    result = backend.decompose(x)
    assert result.modes.shape == (3, len(x))
    np.testing.assert_allclose(result.reconstruct(include_residual=True), x, atol=1e-12)
    assert result.diagnostics["residual_kind"] == "input_minus_mode_sum"
    silent = backend.decompose(np.zeros(321))
    assert np.isfinite(silent.modes).all()
    assert not np.any(silent.reconstruct(include_residual=True))


def test_experiment_exports_measured_audio_and_separate_controls(tmp_path):
    pytest.importorskip("PyEMD")
    pytest.importorskip("vmdpy")
    from approaches.parametric.experiment import run
    from util.metrics import waveform_metrics
    import torch

    sr = 16000
    t = np.arange(4801) / sr
    x = (.5 + .2 * np.sin(2 * np.pi * 7 * t)) * .2 * np.sin(2 * np.pi * 270 * t)
    source = tmp_path / "input.wav"
    sf.write(source, x, sr, subtype="FLOAT")
    output = run(source, window_ms=(20, 40), vmd_modes=3, output=tmp_path / "experiment",
                 save_modes=True, speech_metrics=False)
    summary = json.loads((output / "summary.json").read_text())
    rows = summary["results"]
    assert len(rows) == 9  # self-reference plus 2 methods x 2 windows x 2 reconstructions
    reference, _ = sf.read(output / "reference.wav", dtype="float64")
    assert len(reference) == len(x)
    for row in rows:
        audio, rate = sf.read(output / row["audio"], dtype="float64")
        assert rate == sr and audio.shape == reference.shape
        expected = waveform_metrics(torch.from_numpy(reference)[None, None], torch.from_numpy(audio)[None, None], sr)
        assert row["nmse"] == expected["nmse"]
        assert row["pesq_wb_status"] == "disabled"
        if row["reconstruction"] == "modes_plus_residual_control":
            assert row["nmse"] < 1e-12
    assert (output / "emd/window_320_samples/modes/frame_000000.npz").is_file()
    assert (output / "index.html").is_file()
    assert json.loads((output / "progress.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        run(source, window_ms=(20,), output=output, speech_metrics=False)
