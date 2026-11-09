# Learned audio representations

Implementation code lives in `models/`, `training/`, and `evaluation/`.
Method-specific documentation and published results are grouped by study below.
Commands run from the repository root.

## Completed sparse-autoencoder baselines

The historical study compares sparse nonlinear (S-NL), dense-trained nonlinear
(D-NL), and sparse affine (S-AF) autoencoders. Its model implementation is
[models/sparse_autoencoder.py](models/sparse_autoencoder.py).

- [Model mathematics and experiment report](docs/sparse_autoencoder/sparse_autoencoder_models.tex)
- [Report bibliography](docs/sparse_autoencoder/sparse_autoencoder_models.bib)
- [Development result table](reports/sparse_autoencoder/sparse_autoencoder_results.tex)
- [Held-out test result table](reports/sparse_autoencoder/sparse_autoencoder_test_results.tex)
- [Archived result availability](reports/sparse_autoencoder/README.md)

Build the LaTeX report from `approaches/representations/docs/sparse_autoencoder/`.
Its table inputs refer to `../../reports/sparse_autoencoder/`; its bibliography
remains beside the source. The referenced PNG and additional numeric/provenance
exports are absent from this checkout. The source therefore is not a complete
standalone build package. The saved tables remain available without retraining.

`tools/export_model_results.py` and `tools/compare_models.py` now default to
`approaches/representations/reports/sparse_autoencoder/`. Re-exporting requires
the original completed runs; relocation does not recreate missing evidence.

This completed study is separate from the new planned rate-distortion benchmark
in [approaches/sparse-ae](../sparse-ae/README.md).

## Adaptive STFT dictionary

STFT means short-time Fourier transform. This study generates a spectral
dictionary for each input recording. Its implementation remains
[adaptive_stft_dictionary.py](adaptive_stft_dictionary.py).

- [Design and model description](docs/adaptive_stft_dictionary/adaptive_stft_dictionary_design.html)
- [Completed training run](docs/adaptive_stft_dictionary/adaptive_stft_dictionary_training_run.html)
- [Reconstruction diagnosis](docs/adaptive_stft_dictionary/adaptive_stft_dictionary_reconstruction_diagnosis.md)
- [Standalone 128-case listening and results report](reports/adaptive_stft_dictionary/adaptive_stft_dictionary_testcases.html)
- [Saved metrics](reports/adaptive_stft_dictionary/adaptive_stft_dictionary_metrics.json)
- [Verification record](reports/adaptive_stft_dictionary/adaptive_stft_dictionary_verification.json)

The completed report contains 64 selected test-clean utterances with an excerpt
case and a complete-recording case for each. It embeds all required audio,
images, and records. Share that HTML file itself (about 126.3 MB). The design,
training documentation, and diagnosis contain supporting source/run links and
are distinct from the standalone listening report.

Run `python tools/build_adaptive_stft_report.py` to rebuild from the saved local
payloads and original data. Its output directory is now
`approaches/representations/reports/adaptive_stft_dictionary/`. It does not train
the model. The existing published HTML, metrics, and verification record were
moved without changing their bytes or recorded build identities.

The model source hash is part of the frozen evaluation protocol. Its historical
optional design-note path in the docstring is preserved to keep that identity;
the design link above is the current location. Original run artifacts stay in
`runs/adaptive_stft_dictionary_librispeech_20261107/`. Relocating documentation
does not alter checkpoints, payloads, test selections, or measured scores.
