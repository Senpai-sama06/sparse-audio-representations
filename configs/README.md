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
