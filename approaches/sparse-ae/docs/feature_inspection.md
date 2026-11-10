# Sparse autoencoder feature inspection

This experiment measures what activates a latent channel. It also measures what
changes when that channel is removed. It compares L1 and dense checkpoints.
It does not retrain them.

## Scope

Run `sparse_ae_features_20261110` uses 40 full dev-clean recordings, one per
speaker. These are the unused utterances imported by the local checkpoint audit.
The speakers were used during development. This is an exploratory set. It is not
a final test set. No independent speaker-held-out corpus was found in the server
locations checked.

The input manifest is `runs/sparse_ae_local_audit_20261110/native_manifest.json`.
It records paths, original split, speaker IDs and audio hashes. The script checks
these hashes. It checks that the utterances do not occur in the two checkpoints'
training, validation-selection or calibration-selection lists.

## Steps

1. Run each model on each full recording. Save its valid latent activations.
2. Measure log RMS, spectral centroid, high-band power, periodicity and pitch.
3. Calculate channel associations across all recordings and within each speaker.
4. Remove a linear log-RMS effect from correlations with the other proxies.
5. Select eight channels per model by a fixed hash rule. Require activity on
   1–99% of valid latent frames. Do not use an ablation score for selection.
6. Find three peak examples from different speakers per channel.
7. Set the channel to zero. Decode the full latent sequence. Crop the result to
   1.024 s around the selected peak. Keep the unchanged reconstruction as control.
8. Save PCM16 audio. Calculate scores from its exact decoded samples.
9. Build the standalone HTML. Recalculate its scores and decode its assets.
10. Open the HTML locally. Check tables, images and native audio controls.

The encoder hop is 256 samples (16 ms). Its receptive field is 1,276 samples
(79.75 ms). Acoustic proxy windows have 1,024 samples (64 ms). Correlations
exclude incomplete windows and windows with RMS below 0.0001. Pitch uses a local
autocorrelation peak from 50–400 Hz and a confidence threshold of 0.6.
This pitch proxy can have octave errors. It is not a phoneme annotation.

## Commands

Run from the repository root. The first command requires the imported checkpoints
and input recordings. Choose a new output folder if the output already exists.

```powershell
python -m pytest approaches/sparse-ae/tests/test_interpret.py -q
python approaches/sparse-ae/inspect_features.py
python approaches/sparse-ae/report_features.py
```

`inspect_features.py` saves its protocol before inference. It saves channel
statistics, exact case and crop identities, activations, intervention scores,
source hashes and package versions. It tests a sham intervention with multiplier
one against the original decoder output. The analysis records are under
`runs/sparse_ae_features_20261110/`.

The published [HTML report](../reports/sparse_ae_feature_inspection.html),
[metric records](../reports/sparse_ae_feature_inspection_metrics.json), and
[verification record](../reports/sparse_ae_feature_inspection_verification.json)
are kept together. The HTML embeds all playback audio, images and required data.

## Limits and next action

Association is not a semantic label. The nonlinear decoder can combine channels.
An ablation can create an uncommon latent combination. Its effect does not prove
that the channel is necessary for a speech category. Peak examples do not measure
average ablation effects over the full corpus. Channel IDs do not match across
models. Correlation summaries are descriptive; no statistical significance is
claimed.

Use the exploration to choose candidate channels and proxy directions. Freeze
these hypotheses before evaluating speakers excluded from both training and
development. Add reliable phoneme annotations before testing phoneme claims.
