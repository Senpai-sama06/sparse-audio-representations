# Writing your own YAML

Run commands from the project root using the project Python environment.
Use spaces for indentation. Omitted settings use defaults in
[`lrac/config.py`](../lrac/config.py).

## Training

Save this as `configs/my_experiment.yaml`:

```yaml
extends: [data_synthetic.yaml, model_orthogonal.yaml]

data:
  manifest_dir: data/manifests/my_experiment

train:
  name: my_experiment
  stage: screen
  steps: 2000
  batch_size: 8
  learning_rate: 0.001

evaluation:
  audio_examples: 4
```

`extends` reuses existing settings; values written here override them.
Use `data` for dataset paths, `model` for representation settings, `train` for
learning, and `evaluation` for inspection settings. You can also copy
[`example.yaml`](example.yaml) for a self-contained starting point.

```powershell
python run.py experiment --config configs/my_experiment.yaml
```

This prepares manifests, audits/trains, and produces a validation report. It prints
the run directory. For training only, use `prepare` followed by `train` with the
same `--config` argument.

## Inference / reconstruction

Select the saved model through `--run`. The evaluator loads that run's
`checkpoints/best.pt` and its saved model configuration:

```powershell
python run.py evaluate --run runs/YOUR_RUN --split iid --limit 16
```

For a different dataset, save `configs/my_inference.yaml`, for example:

```yaml
extends: data_vctk.yaml
data:
  root: D:/datasets/vctk
  manifest_dir: data/manifests/my_vctk
  sample_rate: 16000
```

```powershell
python run.py prepare --config configs/my_inference.yaml
python run.py evaluate --run runs/YOUR_RUN --data-config configs/my_inference.yaml --limit 16
```

The inference YAML selects data; model weights and architecture come from the
saved run. Match its sampling rate. Results and reconstructed WAV files go under
that run's `evaluation` folder.

Ordinary paths are relative to the project root; absolute paths are also accepted.
`~` in ordinary paths expands to your user home directory. The cached LibriSpeech
configuration uses this to avoid a workstation-specific username.
Paths in `extends` are relative to the YAML file. Use a separate `manifest_dir`
when changing data preparation. `root` points to existing recordings;
downloads require an explicit `--download` on `prepare` or `experiment`.

The current sparse-autoencoder baseline uses
`librispeech_sparse_autoencoder_1h.yaml` for a pilot and
`librispeech_sparse_autoencoder_10h.yaml` for the main run. Edit the cache path
in `data_librispeech_cached.yaml`, prepare once, then use `experiment --skip-prepare`.
`train.keep_fractions` samples whole-example support fractions during training;
`evaluation.keep_fractions` sets the evaluation sweep. See the
[baseline guide](../documentation/sparse_autoencoder.md).

The matching `librispeech_sparse_autoencoder_dense_10h.yaml` trains the same
model with all values retained. Its `train.validation_keep_fractions` keeps
checkpoint selection on the common sparse budgets. The `*_affine_10h.yaml`
variant sets `model.activation: identity`, removing neural nonlinearities while
preserving the convolutional parameter budget.

The earlier dictionary reference is `synthetic_block_orthogonal.yaml`.
Its `keep_fraction` sets the fraction of coefficients retained across each whole
recording. It trains through SVD updates, so it uses no learning rate.
See [minimum-support evaluation](../documentation/block_orthogonal.md) for
`--nmse-targets` and comparisons with `--checkpoint initial.pt`.

## Studies without training

Use `study_*.yaml` for a numerical experiment that does not need a training run:

```yaml
name: my_search
kind: sparse_search
runs_dir: runs
parameters:
  seed: 0
```

```powershell
python run.py study --config configs/study_sparse_search.yaml
```

`kind` selects the experiment implementation; `name` labels its output directory.
`parameters` are the keyword arguments of `run(output, *, ...)` in the corresponding
`lrac/study_<kind>.py` module. Omitted arguments use its defaults, which are saved
in the run's `config.json`. Unknown arguments raise an error. The existing cue
study also accepts `beam_width`.

The same `extends` and project-relative path rules apply. Results go to a new
timestamped directory under `runs_dir`; `--output runs/my_check` selects a specific
new directory. An existing directory is rejected to preserve earlier results.
No `data`, `model`, or `train` section is required. See the
[codebase guide](../documentation/codebase.md) for adding a new study kind.

## L1 sparse-coding pilot

```powershell
python run.py study --config configs/study_l1_synthetic.yaml
python run.py study --config configs/study_l1_smoke.yaml
python run.py study --config configs/study_l1_librispeech.yaml
```

The first checks five synthetic signals; the second uses two speech clips; the
third uses three clips from each of 40 development speakers. The speech YAML's
`manifest` selects cached development data and `exclusions` points to the earlier
checkpoint-selection metrics CSV. Those utterances are excluded before selecting
clips. No downloads or neural training occur.

`targets` sets waveform NMSE limits and `quantization_bits` sets the coefficient
precision grid. `workers` controls CPU processes. Solver iteration/search limits
and tolerances are explicit in the speech YAML. Leave the mathematical protocol
fixed when reproducing a reported comparison; changed settings constitute a new
run. Results include `report.md`, `metrics.csv`, `rate_distortion.png`, solver
diagnostics and independently decodable files under `bitstreams/`.
