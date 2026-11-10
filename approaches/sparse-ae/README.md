# Sparse autoencoder rate-distortion benchmark

LibriSpeech trainers are implemented for six sparse autoencoder families and a
matched dense control. Each entry script uses the same data and training machinery.
The seven-model 10-hour LibriSpeech campaign has completed on the remote GPU.
Its listening and score report is linked below. The current results are an
unquantized reconstruction comparison; sparse-AE bitrate has not been measured.

| Script | Model |
| --- | --- |
| `train_l1.py` | ReLU codes with L1 activity penalty |
| `train_kl.py` | Sigmoid codes with Bernoulli KL activity penalty |
| `train_topk.py` | Nonnegative Top-K per latent time step |
| `train_jumprelu.py` | Learned thresholds with JumpReLU/L0 gradient estimators |
| `train_gated.py` | Separate activation gating and magnitude estimation |
| `train_batchtopk.py` | Batch activity allocation; calibrated inference threshold |
| `train_dense.py` | Matched ReLU autoencoder without explicit sparsity penalty |

The shared implementation is under `sae/`. DCT is a fixed transform comparison;
it does not require a training script.

## Run every model on a remote GPU

`run_all.py` starts all seven trainers concurrently and detaches by default. It
keeps running across an ordinary SSH disconnect. See [remote setup and monitoring](docs/remote.md).

```bash
python run_all.py start --output runs/ada6000 --parallel 7 --device cuda --batch-size 4 --epochs 100 --patience 8
python run_all.py status --output runs/ada6000 --watch
```

These commands assume the current directory is `approaches/sparse-ae` or a copied
standalone copy of this entire folder. From the repository root, prefix the script
with `approaches/sparse-ae/`. Every trainer can also run independently.

Without `--root` or `--manifest-dir`, each trainer resolves existing local or
KaggleHub-cached LibriSpeech first. If absent, it installs the KaggleHub package
when needed and obtains the corpus through KaggleHub. Dataset setup is serialized
across trainers. Shared manifests are cached. The pinned fallback is
`pypiahmad/librispeech-asr-corpus/versions/1`; `--dataset` changes it. Set
`--offline` to prohibit installation and download.

## Prepare common data

Run these commands from the repository root. Install the project dependencies with
`python -m pip install -e .` if needed. Explicit manifest preparation remains
available when you want to choose your actual local LibriSpeech path.

```powershell
python approaches/sparse-ae/prepare_data.py --root "D:\datasets\LibriSpeech" --output runs/sparse_ae/manifests/pilot --train-split train-clean-100 --train-hours 10
```

The root can also be the directory that contains `LibriSpeech`. Expected splits
are `train-clean-100` and `dev-clean`. An optional existing `test-clean` split is
indexed, but training does not score it. `--train-split train-clean-360` selects
the larger training set. Omit `--train-hours` to use the entire selected split.

Preparation hashes the recordings and makes speaker-disjoint training,
validation, and calibration manifests. Half the dev-clean speakers are assigned
to validation and half to calibration. All models should reuse the same manifests.

## Train one model

```powershell
python approaches/sparse-ae/train_topk.py --manifest-dir runs/sparse_ae/manifests/pilot --output runs/sparse_ae/topk/k16_seed0 --k 16 --epochs 100 --patience 8 --min-delta 0.0001
```

Substitute any entry script from the table. Every script accepts `--help`. The
default device is CUDA when available, otherwise CPU. Defaults are batch size 8,
two-second training crops, learning rate 0.0003, and seed 0. Validation runs after
each epoch on a fixed speaker-balanced selection of up to 256 complete utterances.
Use `--validation-limit 0` for the whole validation partition.

An epoch is one shuffled pass over the selected utterances, using one random crop
per utterance. It is not a pass over every audio sample. Crop selection is stable
for a given seed, recording, and epoch, including after resumption.

Training stops when either condition is reached:

- The maximum epoch count is reached (`--epochs`, default 100).
- Validation fails to improve by more than `--min-delta` for `--patience`
  consecutive epochs (defaults 0.0001 and 8).

By default, the monitored quantity is validation total loss, including the
method's penalties. `--monitor reconstruction` selects waveform/spectral
reconstruction loss instead. Set `--patience 0` to disable early stopping.
`--min-epochs N` delays early stopping until epoch N. The improvement threshold
is absolute. Smaller improvements accumulate relative to the last significant
improvement. Best weights are saved for every new minimum, even if it is smaller
than `--min-delta`. Scores from different native loss definitions must not be
ranked directly against one another.

## Saved files and resumption

Each run directory contains:

| File | Purpose |
| --- | --- |
| `best_weights.pt` | Full model weights/configuration at the lowest monitored validation value |
| `decoder_best.pt` | Decoder weights/configuration from that same epoch |
| `last.pt` | Last complete epoch: model, optimizer, best snapshot, stopping counters, history, and RNG states |
| `best_validation.json` | Per-case reconstruction metrics and validation summary at the best epoch |
| `history.csv` | Training/validation losses, activity, metrics, and times by epoch |
| `summary.json` | Best epoch, stop reason, and checkpoint paths |
| `progress.json` | Live phase, epoch, batch, loss, best score, heartbeat, and process identity |
| `data_source.json` | Actual local/cache/KaggleHub data source |
| `config.json`, `initial_config.json` | Effective and original settings |
| `data_selection.json`, `source.json`, `environment.json` | Exact selected recordings, source hashes, and software/hardware details |

Set `--save-every 5` to retain additional periodic full checkpoints. Existing
nonempty output directories are protected against accidental replacement.

```powershell
python approaches/sparse-ae/train_topk.py --resume runs/sparse_ae/topk/k16_seed0/last.pt --epochs 150
```

Resume restores the saved settings. The corpus and architecture flags need not be
repeated. Architecture, training data, source hashes, and optimizer settings must
match. You may change the maximum epochs, stopping settings, device, worker count,
logging interval, and checkpoint interval. An already early-stopped run stays
stopped unless its stopping settings are changed to allow continuation, for
example `--patience 0`. An interrupted partial epoch is repeated from the last
complete epoch. Periodic checkpoints restore the best snapshot from their own
training history. Only load resumable checkpoints produced by a trusted run.

## Evaluate exported weights

Training logs NMSE and zero-mean SI-SDR on the validation cases. Add
`--speech-metrics` to compute PESQ-WB and STOI as well. Install their optional
dependencies with `python -m pip install -e ".[speech-metrics]"` first. Invalid
scores retain an explicit status and validity count.

After training, evaluate a declared held-out partition with:

```powershell
python approaches/sparse-ae/evaluate.py --checkpoint runs/sparse_ae/topk/k16_seed0/best_weights.pt --manifest runs/sparse_ae/manifests/pilot/test.jsonl --output runs/sparse_ae/topk/k16_seed0/test_evaluation --speech-metrics
```

This writes per-case JSON/CSV metrics, a summary, and reference/reconstruction
FLOAT WAVs for the first eight cases. All selected utterances are evaluated at
their full length. Test data does not select weights or calibrate thresholds.
Historical use of test-clean in this repository must be disclosed; these scripts
do not establish that it is a fresh confirmatory set.

These are the native-objective, unquantized training baselines. Their evaluation
does not claim a bitrate. Quantization-aware training and saved-bitstream
rate-distortion evaluation remain subsequent stages of the experiment plan.

## Completed 10-hour campaign report

- [Feature inspection: L1 and dense activations and ablations](reports/sparse_ae_feature_inspection.html) ([method](docs/feature_inspection.md))
- [Independent local validation of imported server checkpoints (10 November 2026)](reports/local_validation_20261110.md)
- [Standalone PitchFlower crop comparison](reports/sparse_ae_pitchflower_best_testcases.html)
- [Mean and per-case metrics, checkpoint hashes, and data overlap audit](reports/sparse_ae_pitchflower_best_metrics.json)
- [Score and report verification record](reports/sparse_ae_pitchflower_best_verification.json)
- [Embedded audio and score consistency audit](reports/sparse_ae_pitchflower_embedded_audio_audit.json)
- [Per-system, per-case CSV scores](reports/sparse_ae_pitchflower_best_case_scores.csv)

The report scores the saved best checkpoints for all seven methods on the same
40 1.504-second `dev-clean` crops embedded in the PitchFlower report. It is a
historical diagnostic set: 20 exact utterances appeared in the SAE validation
selection and 20 in the calibration selection. The crops are peak-normalized,
while SAE training used native LibriSpeech amplitude. See the report and audit
record before interpreting these scores. There is no bitrate comparison until
the latents are quantized and a complete bitstream is measured.

See [training details](docs/training.md) for the losses and method-specific
adaptations, and [the experiment plan](docs/experiment_plan.md) for the complete
benchmark protocol.

## Verification

```powershell
python -m pytest approaches/sparse-ae/tests -q
```

Tests use small synthetic FLAC recordings in a temporary LibriSpeech-like layout.
They do not download or train on the corpus.
