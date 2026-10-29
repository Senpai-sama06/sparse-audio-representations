# Contributing

Keep the research easy to inspect and reproduce. Use plain Python functions,
explicit model classes, and small YAML files. Keep experimental ideas separate
from the shared training and evaluation code.

## Add an experiment

1. Copy `configs/example.yaml`.
2. Rename it using `<dataset>_<model>[_<variant>][_<size>].yaml`, for example
   `synthetic_orthogonal.yaml` or `librispeech_autoencoder_functional_10h.yaml`.
3. Set `train.name` to the filename without `.yaml` and choose the appropriate
   stage: `smoke`, `reference`, `screen`, `main`, or `confirmation`.
4. Give genuinely different data preparation a separate `data.manifest_dir`.
5. Run `python run.py experiment --config YOUR_FILE.yaml` and record the question
   and comparison in `documentation/experiments.md`.

Use a self-contained configuration when that is easiest to read. For larger
studies, `extends` can reuse a dataset and model configuration; child values
override parent values. Paths inside configurations are relative to the project
root. Paths in `extends` are relative to that YAML file. Avoid long inheritance
chains and avoid putting settings into Python scripts.

The one-command workflow uses at most `train.val_examples` validation examples,
and `evaluation.analysis_examples` for feature inspection. Use `--limit N` to
change its validation limit. Final tests are a separate `evaluate` command after
model selection. No downloads happen without `--download`.

### Experiments without training

Use `python run.py study --config configs/study_<topic>.yaml`. The existing
`experiment` command remains the prepare/train/validation workflow.

1. Put reusable algorithms in `lrac/<topic>.py` and data generation/loading in
   `lrac/data/`. Reuse the existing metrics rather than making local copies.
2. Add `lrac/study_<topic>.py` with a plain `run(output, *, ...)` function. Keep
   the experimental loop, comparisons and study-specific tables here. Put tunable
   options in keyword arguments; fixed mathematical protocol details can stay in
   the study implementation and should be recorded in its results.
3. Add an explicit selection branch to `study_function` in `lrac/study.py`.
   No new base class, registry, training loop or script is required.
4. Add `configs/study_<topic>.yaml` with `name`, `kind` and `parameters`. The runner
   saves resolved defaults, environment/source hashes and completion/failure
   status, and refuses to overwrite an existing output directory.
5. Check the new behavior and document the research question and limitations.
   Promote selected results from `runs/` into `documentation/` deliberately;
   ordinary runs should not rewrite committed reports.

See [the codebase guide](documentation/codebase.md) for the layout and a minimal
configuration. The earlier S-AF analysis tools remain available; the two
deterministic search/cue scripts now only forward to this study command.

## Add or change a model

| Change | Location |
|---|---|
| Model implementation | `lrac/models/<model_name>.py` |
| Shared interface | `lrac/models/base.py` |
| Model selection | `build_model` in `lrac/models/__init__.py` |
| Allowed options and model kinds | `ModelConfig` and `validate` in `lrac/config.py` |
| Reusable model settings | `configs/model_<model_name>.yaml` |

Implement `encode(waveform, lengths)` and `decode(representation)` on
`ReconstructiveModel`. Waveforms are `(batch, 1, samples)`; representation groups
are `(batch, time, width)`. Keep decoding possible using only the representation
and model. Preserve original lengths, sampling rate, and padding metadata.

Use `is_dictionary=True` for alternating sparse inference/dictionary updates.
The training loop uses waveform MSE for dictionaries and waveform plus spectral
loss for neural models. Optional `post_step` handles constraints after updates;
`needs_fit` and `fit` support training-data-only fitting such as PCA. Start from
the closest existing model instead of introducing another training loop.

The `block_orthogonal` model instead declares `uses_procrustes=True` and implements
`update_dictionary` for its closed-form SVD step. It shares data audits,
validation, logging, and checkpoints with the same training loop. Its coefficient
and selected-support masks exclude padding from sparsity measurements.

Add a numerical test of the new behavior: reconstruction, shape/boundary handling,
gradients, or constraints as appropriate. Geometric and functional separation are
different claims; name their measurements explicitly.

## Add or change data

- Add generation or loading logic under `lrac/data/`.
- Keep corpus acquisition in `downloads.py`, manifest preparation in
  `manifests.py`, and signal reading/resampling in `preprocessing.py`.
- Add options in `DataConfig` and `configs/data_<dataset_name>.yaml`.
- Preserve the manifest fields used by `audit.py`: source identity, corpus,
  partition role, speaker identity where available, sample rate, and source extent.
- Extend the audit for any new dataset-specific assumptions. Verify actual
  identities, not just different output filenames.

Do not train or fit preprocessing on held-out data. Keep all crops from a source
recording in one role. Describe dataset exclusions and any known limitations.

## Names and small changes

- Python modules, functions, variables, and YAML keys use `snake_case`.
- Classes use `PascalCase`. Use the same model identifier in configuration,
  model selection, and experiment filenames.
- Use `train`, `validation`, and `test` for data roles.
- Put reusable implementation in `lrac`; put settings in `configs`.
- Keep these folders flat. Code subfolders are `lrac/models` and `lrac/data`;
  add ordinary pipeline functions to `train.py`, `evaluate.py`, `analyze.py`,
  `report.py`, or another clearly named module alongside them.
- Use `data_`, `model_`, and `sweep_` prefixes for reusable configurations.
- Use `study_` for no-training study configurations and their orchestration modules.
- Keep setup and numerical helper scripts together in `tools`. Reports and
  caches belong under `runs`; research notes belong in `documentation`.
- Keep `run.py` thin. Add commands to the existing CLI rather than making copies
  of training, analysis, or evaluation scripts.
- Add a helper only when it removes meaningful duplication. Avoid plugin
  registries, callback systems, or new configuration layers for a single use.
- Keep comments about assumptions and mathematics; avoid comments
  that only repeat the next line of code.

## Before sharing a change

Run `python -m pytest -q`. For a model or training change, also run a small
experiment using the relevant model. Document what changed, what was checked,
and what remains uncertain. Save conclusions as Markdown in `documentation`, with
run names and configuration details. Generated audio and checkpoints stay in
`runs/`, outside version control.
