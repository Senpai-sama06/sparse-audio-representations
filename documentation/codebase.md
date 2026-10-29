# Codebase and experiment guide

Keep the existing folders. Add a module when it has a clear job; use ordinary
functions instead of a framework for experiments.

## Where work belongs

| Responsibility | Location |
|---|---|
| Command-line entry point | `run.py` delegates to `lrac/cli.py` |
| Dataset acquisition, selection and generation | `lrac/data/` |
| Reusable optimization, signal processing and analysis | Named modules directly in `lrac/` |
| Trainable representations | `lrac/models/`, using the existing model interface |
| Training and saved-model evaluation | `lrac/train.py`, `evaluate.py`, `analyze.py`, `report.py` |
| Reconstruction and representation metrics | `lrac/metrics.py`; training objectives in `lrac/losses.py` |
| No-training run bookkeeping | `lrac/study.py` |
| A study's loop and comparisons | `lrac/study_<topic>.py` |
| Settings | Flat `configs/` directory |
| Generated results, caches and checkpoints | `runs/`, ignored by Git |
| Reviewed research notes and selected results | Flat `documentation/` directory |
| Tests | Flat `tests/` directory |
| Setup, export and compatibility scripts | Flat `tools/` directory |

Two existing workflows share the entry point:

```powershell
# Prepare data, train/fit, evaluate validation, analyze, report
python run.py experiment --config configs/example.yaml

# Run numerical inference/search without inventing a training job
python run.py study --config configs/study_sparse_search.yaml
```

Saved neural-model inference continues to use `evaluate --run ...`. These
workflows are separate because fitting a model and running a deterministic
optimizer are different operations. Both keep outputs under `runs/`.

## A small study configuration

```yaml
name: sparse_cues
kind: sparse_cues
runs_dir: runs
parameters:
  seed: 20261029
  beam_width: 8
```

`name` labels a run; `kind` selects an explicit implementation. Both use lower-case
snake_case. `parameters` maps directly to the study function's keyword arguments.
The runner rejects unknown keys and records omitted defaults. YAML inheritance
and relative-path rules are the same as in the training configurations. A study
that accepts its own paths should resolve them explicitly; the common runner
resolves `runs_dir` and `--output` against `project_root`.

```text
runs/sparse_cues_<timestamp>/
  config.json          # Resolved settings, including keyword defaults
  environment.json     # Python, library and device information
  source.json          # Hashes of the package source and dependency declarations
  progress.json        # running, complete, failed or interrupted
  completion.json      # Elapsed time and hashes/sizes of recorded artifacts
  sparse_cues_*.csv    # Study-specific measurements
  sparse_cues_*.json   # Witnesses, protocol and interpretation limits
```

The runner prints `RUN=...` before executing. Each run gets a new directory.
`--output runs/my_check` can select one explicitly; it must not already exist.
Failure status is retained and the original exception is raised. There is no
automatic resume mechanism for these studies.

## Adding an experiment

1. Write and check the reusable method independently of files, command-line
   arguments and plotting. Reuse data loaders and metric definitions.
2. Add a plain `run(output, *, option=default, ...)` function in
   `lrac/study_<topic>.py`. It assembles data, calls methods, compares results and
   writes study-specific artifacts under `output`. Keep numerical arrays and
   methods independent of this orchestration. It can also be called from Python.
3. Add one explicit branch in `study_function` and a `configs/study_<topic>.yaml`.
   No plugin discovery, generic experiment class or duplicated training code is
   necessary. Validate option values before expensive computation.
4. Check against a small independent reference, then record what was measured and
   what the experiment can establish. For audio studies, preserve source/speaker
   identities, split roles, crop definitions and any checkpoint/manifest hashes.
   The runner records code provenance; it cannot infer dataset provenance.
5. Inspect a fresh run before copying selected tables or figures into
   `documentation/`. Keep exploratory and confirmation claims explicit.

Experiment settings should be JSON-compatible YAML values. Return structured
measurements from reusable functions; perform serialization in the study module.
Use the shared `write_csv`/`write_json` helpers where appropriate. A metric is not
automatically applicable to every experiment: the exact rational search retains
its exact error checks; audio studies should use the shared waveform metrics.
Coefficient counts must not be labelled actual bitrates without a measured
encoding/decoding path.

## What this restructuring changes

- The exact-search and cue-comparison orchestration moved from `tools/` to
  `lrac/study_sparse_search.py` and `lrac/study_sparse_cues.py`.
- Their exact toy datasets moved to `lrac/data/sparse_examples.py`, removing the
  script-to-script import. The numerical optimizers themselves are unchanged.
- `tools/search_sparsest.py` and `tools/analyze_sparse_cues.py` remain thin
  compatibility commands. They now write fresh run directories by default.
  `--output` also requires a new directory; writing directly over `documentation/`
  is intentionally no longer the ordinary run workflow.
- Existing training configurations, saved checkpoints, training/evaluation
  commands and published measurements retain their formats.
- The older channel, pitch and module analysis tools retain their existing
  interfaces and cache behavior. They have not been migrated in this change.

No new dependency or folder was introduced by the restructuring. The workflow
now also hosts the [L1 audio experiment](l1_experiment.md): `linear.py` contains
matrix-free dictionaries, `l1.py` inference, `codec.py` serialization,
`data/l1_examples.py` cohort selection, and `study_l1.py` the experiment loop.
See [its measured results](l1_results.md).

The restructuring passed all 84 tests then present, including the existing training
and saved-model evaluation checks; the L1 extension brings the suite to 103 tests.
The moved search reproduces the 105 published
minimum-support results and complete error curves. A full cue-study rerun matches
all 1,440 comparisons, 135 summary rows, 96 feature rows, 288 continuous-construction
rows and exact witnesses, excluding elapsed execution times.
