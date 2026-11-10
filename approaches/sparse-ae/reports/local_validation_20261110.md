# Local checkpoint validation — 10 November 2026

All seven best checkpoints were imported over SFTP from `kiran@172.16.56.8:/home/kiran/sparse-ae-work/sparse-ae/runs/librispeech_10h_parallel`.

## Findings

The historical reconstruction scores reproduce. No material scoring implementation error was found on these recordings. This does not make the original comparison a held-out test or establish compression performance.

All checkpoint hashes match the published report: True. The imported SAE Python source files were byte-identical to the local implementation. Local inference used the imported weights in evaluation mode with strict state loading; no training occurred.

Independent MSE/NMSE calculations and least-squares zero-mean SI-SDR were compared with `sae.metrics`. PESQ-WB and STOI were called directly through the installed `pesq` and `pystoi` packages, bypassing the repository scoring wrapper. This checks their integration; it is not an independent implementation of PESQ/STOI themselves.

## Historical 40-crop reproduction

The original 1.504-second, peak-normalized references were used. Reconstruction WAVs were scored after PCM16 serialization, as in the published report. The supplied SAE input samples exactly matched resampling the original 24-kHz reference PCM. PitchFlower was independently rescored from its 24-kHz PCM audio.

| Model | PESQ-WB | STOI | SI-SDR (dB) | NMSE |
| --- | ---: | ---: | ---: | ---: |
| l1 | 2.874866 | 0.940978 | 14.3587 | 0.051345 |
| kl | 2.700868 | 0.928735 | 13.3096 | 0.062034 |
| topk | 2.057527 | 0.868755 | 8.8431 | 0.136707 |
| jumprelu | 2.101961 | 0.888191 | 10.4611 | 0.104217 |
| gated | 2.673929 | 0.931404 | 13.8102 | 0.055930 |
| batchtopk | 2.123457 | 0.885506 | 10.3034 | 0.108621 |
| dense | 2.879766 | 0.941592 | 14.4270 | 0.050383 |
| PitchFlower | 2.162867 | 0.904913 | -23.9552 | 1.973528 |

Maximum absolute difference between independently recomputed local scores and the published SAE scores, across all 280 cases:

- mse: 1.02125058e-07
- nmse: 9.46107403e-06
- si_sdr: 0.000703962302
- pesq_wb: 0.00402641296
- stoi: 0.000122433986

These are numerical reproduction differences; the per-case results are retained rather than asserted to be bit-identical across GPU/library environments.

## Additional native-amplitude evaluation

40 full dev-clean utterances totaling 267.735 seconds, one per speaker. Selection was fixed by SHA256 of `sae-audit-20261110:<id>` after excluding all seven models’ training, validation-selection and calibration-selection utterances, plus the original report utterances. Audio file hashes were verified on import and before inference. No peak normalization, time alignment, gain fitting or checkpoint reselection was applied. Scores use float outputs; the saved FLOAT WAVs preserve those samples.

**These speakers were used in development. This is an unused-utterance diagnostic, not a fresh speaker-held-out test.** No test-clean directory was found in the searched server dataset locations.

| Model | PESQ-WB | STOI | SI-SDR (dB) | NMSE | Active coefficients/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| l1 | 2.853340 | 0.951090 | 13.8492 | 0.051967 | 5007.8 |
| kl | 2.722390 | 0.941087 | 12.9609 | 0.062150 | 16022.2 |
| topk | 2.014759 | 0.884417 | 8.2868 | 0.151586 | 998.1 |
| jumprelu | 1.935574 | 0.882807 | 9.4126 | 0.121441 | 821.2 |
| gated | 2.686102 | 0.943707 | 13.5048 | 0.055453 | 4947.5 |
| batchtopk | 1.926529 | 0.889169 | 9.3488 | 0.123799 | 874.9 |
| dense | 2.944718 | 0.954807 | 14.1421 | 0.048611 | 7882.3 |

## Interpretation and limits

- The original report’s 20 validation and 20 calibration utterances cannot support an independent test claim.
- Peak-normalized short crops differ from the native-amplitude, full-utterance training-validation pipeline. Differences between the two tables cannot be attributed to amplitude alone: the recordings and durations also differ.
- Sparsity levels differ substantially. Quality rankings alone do not compare equal rate or equal active-coefficient budgets.
- No quantized sparse bitstream was evaluated. Active coefficients per second are not bits per second.
- PESQ/STOI and samplewise SI-SDR measure different properties. Agreement of arithmetic does not establish perceived quality or statistical significance.

## Reproduction artifacts

- [Independent per-case scores and controls](../../../runs/sparse_ae_local_audit_20261110/independent_audit.json)
- [Imported files, paths and SHA256 hashes](../../../runs/sparse_ae_local_audit_20261110/download_manifest.json)
- [Additional recording selection](../../../runs/sparse_ae_local_audit_20261110/native_manifest.json)
- [Local package versions](../../../runs/sparse_ae_local_audit_20261110/local_environment.json)
- [Validation script](../../../tools/audit_sparse_ae_local.py)

The raw local artifacts are under `runs/sparse_ae_local_audit_20261110/` and are not embedded in this Markdown report. Identity, additive-noise and sample-permutation metric controls are stored in the independent JSON. Historical reports were preserved.

```powershell
python tools/audit_sparse_ae_local.py
```
