# Adaptive STFT dictionary: reconstruction diagnosis

Date: 2026-11-08. Run: `adaptive_stft_dictionary_librispeech_20261107`.

## Purpose and scope

Find why the saved model loses speech detail. Use the same 64 one-second LibriSpeech test-clean excerpts as the previous evaluation. All five comparisons completed for all 64 excerpts. No cases were excluded. No training or checkpoint changes were made.

These are exploratory checks on an existing test group. Use dev-clean for selecting changes. Use a fresh held-out group for the next final evaluation. These checks do not cover complete recordings or establish compression.

## Method

The short-time Fourier transform (STFT) represents each input as a complex matrix. The model represents this matrix with a dictionary D and coefficients A. The decoder reconstructs the spectrum from D times A, projects the imaginary DC and Nyquist values to zero, and applies the inverse STFT.

1. Decode each saved model payload. Keep its gain and source offset.
2. Keep D fixed. Fit A by a double-precision least-squares solve of the projected complex reconstruction error. Store the fitted coefficients as float32 values in a new payload. Decode this payload.
3. Use singular value decomposition (SVD) to fit 32 atoms and their coefficients to the source spectrum. Store both factors in a new payload. Decode this payload. Each control payload has the same byte count as its model payload.
4. Combine the model output magnitude with the source phase. Apply the inverse STFT.
5. Combine the source magnitude with the model output phase. Apply the inverse STFT.
6. Calculate scores on the resulting waveforms. Average each score over the same 64 excerpts.

Steps 4 and 5 use source information that is not transmitted. They are diagnostic controls. The SVD control sends both factors and does not need source information at the decoder.

## Results

Normalized mean squared error (NMSE) measures waveform error relative to source energy. Lower is better. Wideband perceptual evaluation of speech quality (PESQ-WB) and short-time objective intelligibility (STOI) are objective speech scores. Higher is better. Metric settings match the [previous evaluation](adaptive_stft_dictionary_training_run.html): 16 kHz mono audio, wideband PESQ, and standard STOI.

| Reconstruction | Waveform NMSE | PESQ-WB | STOI |
| --- | ---: | ---: | ---: |
| Trained model | 0.2242 | 1.748 | 0.849 |
| Same atoms; least-squares coefficients | 0.1062 | 1.691 | 0.883 |
| SVD; 32 atoms; same payload size | 0.0150 | 2.717 | 0.953 |
| Model magnitude; source phase | 0.1857 | 1.945 | 0.876 |
| Source magnitude; model phase | 0.1449 | 4.161 | 0.984 |

The trained output has about 11% of source spectral energy in the 2-4 kHz band and 14% in the 4-8 kHz band, on average. These energy ratios do not measure whether the remaining energy is correct. They show substantial energy loss in these bands.

The coefficient solve reduces waveform error, but PESQ falls slightly. Thus, better complex least-squares fitting alone does not establish better perceptual quality.

The SVD control improves all three scores at the same encoded size. The current model leaves a substantial fitting gap. More atoms are not the first change to test.

The source-magnitude control improves PESQ much more than the source-phase control. This points to missing or distorted magnitude as a major problem. Phase remains relevant. These interventions do not fully separate phase and magnitude effects: inverse STFT and subsequent analysis couple them.

Re-analysis and inverse STFT of the baseline output change its waveform by a maximum NMSE of about 1.5e-14. Thus, this consistency projection alone cannot repair the output. These single-excerpt tests have no chunk stitching.

## Next experiments

1. Add SVD as a deterministic encoder baseline. Compare learned initialization and SVD initialization before changing the atom count.
2. Inspect per-band residuals and the gradient contribution of each loss. The current model already has complex reconstruction loss, waveform L1 loss, and multi-resolution spectral loss. Test band-normalized or compressed-magnitude losses against the current loss. Keep a waveform fidelity term.
3. Test a stronger coefficient solve inside training. Compare direct inference, numerical inference, and learned refinement. Select settings with spectral diagnostics and speech scores, not waveform error alone.
4. Inspect each refinement stage. Check whether dictionary updates improve high-frequency detail or remove it. Record atom usage, dictionary conditioning, coefficient energy, and spectral error at each stage.

Run these comparisons on dev-clean first. Change one part at a time. Keep the atom count, payload format, and sample group fixed. New training has not started.

## Evidence and provenance

- [Diagnostic code](../../../../tools/diagnose_adaptive_stft_dictionary.py).
- [Protocol and source hash](../../../../runs/adaptive_stft_dictionary_librispeech_20261107/diagnosis_test_excerpts_64/protocol.json).
- [All 320 metric records](../../../../runs/adaptive_stft_dictionary_librispeech_20261107/diagnosis_test_excerpts_64/metrics.csv).
- [Mean scores](../../../../runs/adaptive_stft_dictionary_librispeech_20261107/diagnosis_test_excerpts_64/summary.json).
- [Diagnostic plot](../../../../runs/adaptive_stft_dictionary_librispeech_20261107/diagnosis_test_excerpts_64/diagnosis.png).
- [Earlier portable case report](../../reports/adaptive_stft_dictionary/adaptive_stft_dictionary_testcases.html).

The diagnostic worktree is based on revision `9e1ea9f795581323b424b50ca838792714b26392`, with local changes. The original training revision was not recorded. The protocol stores the selected checkpoint SHA-256: `5738d0475cfc65841527cba4fcd2480453ca78bffcd3d4813046c4a2891c5a8d`, at step 20,000. The diagnostic code verifies the model source hash against the previous evaluation protocol. Results remain under the local ignored `runs` folder.

This is a diagnostic document. It does not replace the portable case report. Objective scores are not a listening test. The cause of the fitting gap remains to be measured through the proposed ablations.
