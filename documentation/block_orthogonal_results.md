# First block-dictionary screening run — 2026-10-27

Run: `runs/synthetic_block_orthogonal_20261027T061328438852Z_seed0`.
Configuration: `configs/synthetic_block_orthogonal.yaml`.

## What was executed

- 512-dimensional orthonormal dictionary, initialized from DCT; four candidate groups.
- Nonoverlapping blocks; a global budget retaining 12.5% of waveform coefficients.
- 2,048 generated training recordings of 1.024 seconds, with separate validation
  and test seed ranges. These are controlled signals, not speech recordings.
- 200 minibatch Procrustes updates, batch size 16, on the local GPU.
- Validation every 20 updates on a fixed set of 32 recordings.
- Total training/validation time about 112 seconds; reported peak allocated GPU
  memory about 36.5 MiB. This is a pilot runtime, not a general performance benchmark.

## Result

The learned iterates did **not** improve on the initial DCT dictionary on the
validation selection objective. `best.pt` therefore correctly remains at step 0.
Normalized validation MSE was 0.02466 initially and 0.04564 at step 200. This
normalization uses fixed training-set signal power, not per-recording NMSE.

The reported best-checkpoint median waveform SNR at the 12.5% budget was 17.83 dB.
The routine worked, but this run provides no evidence of a better learned feature
space. Exact current-batch updates can overfit or change behavior across batches;
their mathematical property does not imply improved population reconstruction.

All logged same-batch updates reduced or preserved their objective, and the
largest logged dictionary orthogonality error was about 4.62e-12.

## Minimum-support comparison

The initial and final dictionaries were frozen and evaluated on the same 32
validation signals. Each recording has 16,384 valid coefficients before selection.
The table reports the median number needed to reach each waveform NMSE tolerance.

| Target NMSE | SNR target | Initial DCT | Final learned dictionary |
|---|---:|---:|---:|
| 0.1 | 10 dB | 554.5 (3.38%) | 880.5 (5.37%) |
| 0.01 | 20 dB | 2,585.5 (15.78%) | 4,830.5 (29.48%) |
| 0.001 | 30 dB | 4,929.5 (30.09%) | 8,243.0 (50.31%) |

Half-integer medians arise from an even number of signals. Every individual
support count is an integer. All 32 examples at all three tolerances, for both
checkpoints, passed numerical checks: predicted and measured error agreed, the
target was reached, and one fewer coefficient would exceed it.

This establishes conditional support optimality for the frozen orthonormal
transform and squared error. It does not establish a globally optimal dictionary
or perceptual-metric optimality. Coefficient arrays remain dense in memory with
explicit zero/support masks; these counts are not packed file sizes or bitrates.

## Software verification and remaining work

49 tests passed, including exhaustive tiny support searches, partial-block and
padding handling, same-batch Procrustes descent, checkpoint loading, and tolerance
evaluation. The eight-update smoke workflow also completed.

The full report, reference/reconstructed audio, metrics, initial/best/last
checkpoints, and training history are under the run directory. No final test set
was evaluated. No LibriSpeech or VCTK training was performed.

The next scientific questions are whether more representative/full-batch updates,
initializations, or a different structural assumption improve held-out sparsity.
Speech perceptual metrics and feature-meaning claims require separate validation.
