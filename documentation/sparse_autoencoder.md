# Sparse autoencoder baseline

This is the first implemented ungrouped sparse-autoencoder baseline. It learns
an unnamed representation directly from waveforms. It does not certify the
sparsest representation among all possible models.

## Model

`lrac/models/sparse_autoencoder.py` reuses the existing convolutional encoder and
mirrored decoder: four stride-four stages, channels 32/64/128/128, and 256 signed
latent channels. A two-second 16 kHz input has 125 x 256 latent values before
selection. The encoder receptive field is 2,041 samples, about 128 ms; global
selection does not make the candidate-value encoder global in time.

Magnitude selection retains a fraction or exact count across the whole example,
with exact zero values elsewhere. There is one representation group, no waveform
bypass, and no orthogonality or feature-label objective. Masked batch padding is
excluded from selection. Unequal example lengths are processed in length groups
so extra batch padding cannot alter a shorter example's code or reconstruction.
The decoder receives only latent values and reconstruction length metadata.

`model.keep_fraction` is a fraction in [0,1]. `model.sample_budget` optionally
overrides it with an exact retained-position budget. Zero means zero retained
values in this model. This differs from legacy per-time-step `code_budget=0`,
which means a dense code. Actual nonzero count can be less than the number of
selected positions if a selected value is exactly zero.

## Data and configuration

The speech runs use the inspected KaggleHub copy of train-clean-360, dev-clean,
and test-clean in place, without copying recordings. Adjust the absolute cache
path in `configs/data_librispeech_cached.yaml` on another workstation. The data
preparer supports both this nested layout and official archive extraction.

Balanced 1h/10h subsets cover all 921 training speakers if the source corpus is
complete; audits verify actual coverage, source intervals, metadata, speaker
separation, and duplicate decoded audio. Dev-clean and test-clean each contain
40 separate speakers. Test-other is not in this cache and is not required for
the initial clean experiment. Final test recordings are not used for selection.

```powershell
python run.py prepare --config configs/librispeech_sparse_autoencoder_10h.yaml
python run.py experiment --config configs/librispeech_sparse_autoencoder_1h.yaml --skip-prepare
python run.py experiment --config configs/librispeech_sparse_autoencoder_10h.yaml --skip-prepare
```

The first preparation indexes the cached training partition and writes both
balanced subsets. `--skip-prepare` reuses these manifests; the data audit still
runs before training. The source FLACs remain in the cache. Do not replace or
delete the cache while runs are active.

For a short synthetic workflow check:

```powershell
python run.py experiment --config configs/smoke_sparse_autoencoder.yaml
```

The larger controlled study is `configs/synthetic_sparse_autoencoder.yaml` and
trains a separate model on synthetic signals rather than mixing them into speech.

## Training and measurements

The main run is 20,000 updates with effective batch size eight, random two-second
crops, AdamW at 2e-4, and gradient clipping at one. The pilot uses 2,000 updates on
the 1h subset. Training samples fractions 0.01/0.02/0.05/0.1/0.25/0.5/1.0 uniformly
per microbatch. The loss is valid-sample waveform MSE divided by fixed estimated
training power. There is no extra l1, group, functional, or spectral penalty.

Checkpoint selection uses mean normalized waveform loss across the configured
training fractions and a fixed validation probe. The main probe covers 200
dev-clean utterances, balanced across its 40 speakers. Full-code and individual
fraction losses are retained in `validation.jsonl`. The code does not assume
monotone reconstruction error when increasing support in a nonlinear decoder.

`experiment` evaluates validation after training across the configured fractions,
including zero code. It saves per-example CSVs, speaker-macro summaries and
bootstrap intervals, audio, and an HTML/PNG report. It leaves final tests alone.

Metrics include actual nonzero counts, active values per second and per input
sample, MSE, MAE, NMSE, waveform SNR, SI-SDR, spectral/transient/high-band errors,
and timed inference. PESQ/STOI are optional speech metrics. Missing or invalid
measurements have explicit status fields; STOI warnings are not silently treated
as valid scores. ViSQOL is not integrated and has no computed score. Optional
packages can be installed with `pip install '.[speech-metrics]'`.

`support_search.csv` and `support_search.json` also record the smallest successful
actual support among the evaluated codes at NMSE 0.1/0.01/0.001, including unmet
and silent cases. This search does not assume a monotone error curve.

```powershell
python run.py evaluate --run runs/YOUR_RUN --split validation --limit 40 --keep-fractions 0,0.01,0.05,0.1,0.25,0.5,1
python run.py evaluate --run runs/YOUR_RUN --split validation --limit 40 --budgets 0,100,1000
```

All budgets refer to the whole recording. Float32 values, coordinate identifiers,
and length metadata are needed for reconstruction; scalar support is not an
encoded bitrate. The first results describe an achieved code, not a minimum-
support certificate. The matched dense-trained control is now training, as
described below. The DCT comparison remains to be evaluated; neither comparison
should be claimed complete before its results exist.

## Background runs and monitoring

From the project root, show the current comparison's progress with
`.\tools\status.ps1`. It reads each run's actual `progress.json`, not a cached
step count. `complete` means the report also exists; `training complete` means
only training is known complete. To watch the table, use:

```powershell
while ($true) { Clear-Host; .\tools\status.ps1; Start-Sleep -Seconds 5 }
```

Press Ctrl+C to stop watching; this does not stop training. For curves, run
`python -m tensorboard.main --logdir runs --host 127.0.0.1 --port 6006` and open
`http://127.0.0.1:6006`. Inspect `train/reconstruction`,
`validation/reconstruction`, and `train/active_fraction`. Use the step axis for
update-matched comparisons; concurrent runs have different wall-clock timings.
Per-budget validation losses are retained in `validation.jsonl`.

After preparing manifests, launch a persistent hidden process with:

```powershell
.\tools\start_training.ps1 -Config configs/librispeech_sparse_autoencoder_10h.yaml -SkipPrepare
```

The launcher prints the process ID and log paths and records them in
`runs/*.launch.json`. Standard output reports the actual timestamped run folder.
Its `progress.json`, `history.jsonl`, `validation.jsonl`, and `tensorboard/` show
progress. `checkpoints/initial.pt`, `best.pt`, and `last.pt` are saved. Checkpoints
are atomic; `completion.json` marks completed training. Evaluation/report creation
then follows in the same process. A failed run records an error rather than
claiming completion.

Resume training using a saved run configuration and `--resume` as documented in
the [command guide](running_experiments.md). Model/data settings must match the
checkpoint. Optimizer and model RNG are restored, but sampler position restarts;
resume is not an exact replay of uninterrupted data order.

## Run record: 27 October 2026

Validation: **59 tests passed**, followed by a complete eight-update synthetic
workflow at `runs/smoke_sparse_autoencoder_20261027T112656175786Z_seed0`.
An integration issue in the first smoke attempt was fixed: group-separation
derivatives are inapplicable to the ungrouped model and are now skipped explicitly.

Speech pilot: `runs/librispeech_sparse_autoencoder_1h_20261027T113227649577Z_seed0`.
It completed 2,000 updates in about 55.7 seconds after setup/auditing, using
about 195 MiB peak tensor allocation on CUDA. This is not total process VRAM.
It has 1,002,369 trainable parameters. The decoded-audio, source, interval, and
speaker audits passed. Both training subsets include 921 speakers; the 1h subset
has 948 source records and the 10h subset has 3,329. Validation and clean test
contain 40 separate speakers each. Final test reconstruction metrics remain unused.

The pilot's fixed-power-normalized validation loss averaged over seven budgets
decreased from 14.8563 to 0.3750. This loss is not per-example NMSE; the high initial
value also reflects the untrained network's output. The following are measured
on the same 40 full dev-clean utterances using the selected step-2000 checkpoint:

| Retained fraction | Median waveform SNR (dB) | Median NMSE | Mean STOI | Mean PESQ-WB |
|---|---:|---:|---:|---:|
| 0 | -0.004 | 1.00094 | 0.4414 | 1.0672 |
| 0.01 | 3.392 | 0.45799 | 0.5918 | 1.1074 |
| 0.02 | 4.065 | 0.39216 | 0.6289 | 1.1527 |
| 0.05 | 4.777 | 0.33288 | 0.6676 | 1.2141 |
| 0.10 | 5.025 | 0.31440 | 0.6871 | 1.2527 |
| 0.25 | 5.144 | 0.30598 | 0.7014 | 1.2827 |
| 0.50 | 5.158 | 0.30502 | 0.7051 | 1.2901 |
| 1.00 | 5.144 | 0.30601 | 0.7067 | 1.2935 |

These are limited pilot measurements, not high-fidelity reconstruction. Dense
quality is itself poor, so the small difference between 10% and full support
does not demonstrate that 10% is a sufficient high-accuracy representation.
Metrics on zero-code output are diagnostic numerical results, not evidence that
zero values preserve the utterance. The nonmonotone dense/50% result reinforces
that predicted-code error is not guaranteed to decrease with its budget.

Main run: `runs/librispeech_sparse_autoencoder_10h_20261027T113534514450Z_seed0`,
fresh training from seed 0 on the 10h subset for 20,000 updates. Launcher record:
`runs/librispeech_sparse_autoencoder_10h_20261027T113531946Z.launch.json`.
It runs independently in a hidden process and will perform validation measurement
and report generation after training. Follow its progress/checkpoints rather than
treat this launch record as evidence of completed training.

## Concurrent comparison runs

Status update: all three runs have completed 20,000 updates and their validation
reports exist. The status script reads this from each actual run directory.

The user requested concurrent experiments after observing low GPU memory use.
The original baseline continues unchanged. Two controlled alternatives are now
training on the same cached 10h subset, with identical source/speaker separation,
seed 0, batch size eight, 20,000 updates, waveform loss, and evaluation fractions.

| Run | Difference | Question |
|---|---|---|
| Original sparse autoencoder | Neural nonlinearities; training samples seven support fractions | Reference learned sparse representation |
| Dense-trained control | Identical neural model, but training retains the full code | Does sparse-budget training beat ordinary reconstruction training followed by pruning? |
| Affine control | Replace neural LeakyReLU functions in encoder and decoder with identity operations; retain sparse-budget training | Do nonlinear analysis/synthesis mappings improve on an affine learned transform? |

The affine control retains convolution biases, so its dense encoder and decoder
are affine rather than strictly homogeneous linear maps. Global support selection
is still nonlinear. Kernels, strides, latent dimensions, and parameter count are
unchanged. All three models have 1,002,369 trainable parameters. The affine and
nonlinear variants have identical initial parameter tensors under the same seed.

`model.activation: identity` selects the affine control. The default remains
`leaky_relu`. `train.validation_keep_fractions` optionally separates checkpoint
selection budgets from training budgets. The dense-trained control trains only
at fraction one, but selects its checkpoint over the same seven fractions as the
other runs. This keeps its validation criterion comparable.

Configurations:

- `configs/librispeech_sparse_autoencoder_dense_10h.yaml`
- `configs/librispeech_sparse_autoencoder_affine_10h.yaml`

Run directories:

- `runs/librispeech_sparse_autoencoder_dense_10h_20261027T114358039621Z_seed0`
- `runs/librispeech_sparse_autoencoder_affine_10h_20261027T114554258463Z_seed0`

All 60 tests pass, including equal parameter initialization/counts and numerical
affinity of the dense encoder/decoder. The affine variant completed a short
synthetic workflow before its speech training was launched.

Resource observations are saved in `runs/concurrent_training.json`. Before
concurrency, steady training was about 32.2 updates/s, excluding validation pauses.
A short two-job estimate was about 50.1 combined updates/s under the same
steady-step definition. Individual jobs slowed, while combined throughput improved.
Three-job GPU utilization reached about 87% at an initial reading with only
about 1.9 GB allocated. VRAM allocation alone is not a measure of spare compute.
These are resource probes, not claims of optimal scheduling or isolated model
throughput. The follow-up observation records a common wall-time window as well.

The 30.3-second three-job probe measured mean GPU utilization of 95.8%, maximum
allocation of 1,865 MiB, and 43.6 combined updates/s including validation pauses.
That last throughput definition differs from the steady-step estimates above.
All three data audits passed, their training-manifest hashes matched exactly,
and each validation record contained the same seven nonzero fraction scores.

Concurrency changes timing conditions: inference-time measurements collected
while other jobs use the GPU are contended measurements. Compare reconstruction
on identical examples and budgets; repeat isolated timing checks after training
jobs finish if architecture speed is part of a conclusion. Results are
single-seed exploratory comparisons, not final statistical evidence.
