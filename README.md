# Sparse reconstructive audio representations

Learn the fewest values needed to reconstruct each audio sample at the required
accuracy, without needing to name or explain the representation discovered.

The sparse-autoencoder baseline is implemented; see its
[training and measurement guide](documentation/sparse_autoencoder.md).
It uses the inspected LibriSpeech cache with speaker-balanced training subsets.
The [mathematical model and three-model comparison (PDF)](documentation/sparse_autoencoder_models.pdf)
documents the exact architecture, training objective, citations, and measured
development results. Its [LaTeX source](documentation/sparse_autoencoder_models.tex),
[references](documentation/sparse_autoencoder_models.bib), and
[numeric results](documentation/sparse_autoencoder_results.csv) are included.
The [full held-out test comparison](documentation/sparse_autoencoder_test_results.md)
covers all 2,620 test-clean recordings from 40 unseen speakers, including quality
curves, matched-error support comparisons, and paired speaker uncertainty. The
architecture audit identifies hidden compression to 16,000 values before the
32,000-slot latent projection for a two-second input; full-budget reconstruction
is therefore not a lossless reference.
The [S-AF channel characterization](documentation/saf_channel_analysis.md) ranks
all 256 channels at 10% support and analyzes the top ten: coefficient distributions,
bias-corrected synthesis spectra, voiced/unvoiced associations, ablations, and
individual-channel spectrograms.
The [controlled pitch intervention](documentation/saf_pitch_analysis.md) measures
all 256 frozen S-AF channels before and after top-k, with fixed-filter predictions,
independent manipulation checks, channel rankings, and unvoiced/noise controls.
The [module audit](documentation/saf_module_analysis.md) follows an example through
every block and lays out stage probes, reconstruction oracles and interventions
for understanding the architecture.
The [datasets and study protocol](documentation/experiments.md) and
[architecture comparison](documentation/implementation.md) record alternatives;
existing models remain earlier baselines for exploration.

The current research discussion is in
[finding the sparsest domain for a particular sample](documentation/adaptive_transforms.md),
following the [initial literature review](documentation/literature_review.md).
The [formal existence proofs](documentation/sparsest_representation.md) establish
when minimum support is attained and distinguish sample-adapted bases from
shared, stable reconstruction systems.
The [sparse optimization search](documentation/sparse_search.md) formulates joint
domain/coefficient optimization and provides a tiny exact rational search with
certified minimum-support results.
The [structural-cue investigation](documentation/sparse_cues.md) compares faster
deterministic searches, proves a pruning bound and an ideal constructive result,
and examines recurrence-based descriptions before further model training.
The [L1 experiment protocol](documentation/l1_experiment.md) specifies a sparse
inference baseline and definitions for decoded reconstruction quality and actual
bitrate. Run it with `python run.py study --config configs/study_l1_librispeech.yaml`;
the matching synthetic and two-speaker smoke configurations provide small checks.
The [first measured L1 results](documentation/l1_results.md) include 3,600 real
bitstreams from 120 development clips, solver certificates, quality/rate curves
and comparisons with DCT thresholding.
The [separate exhaustive L0 activity](documentation/l0_activity.md) enumerates
supports on a full speech clip without pruning or a time limit, with persistent
progress and resume commands.
[bibliography.md](bibliography.md) records the papers examined. Dictionaries and
orthogonality are candidate assumptions, not requirements of the research goal.

## Start here

On this workstation, set up and activate the Python environment:

```powershell
.\tools\bootstrap.ps1
.\.venv\Scripts\Activate.ps1
```

Run a small experiment from beginning to end:

```powershell
python run.py experiment --config configs/example.yaml
```

This prepares synthetic data manifests, audits and trains, evaluates **validation**
examples, analyzes the features, and creates `runs/<run_name>/report/index.html`.
The command prints the output path. It does not download speech or evaluate final
test sets automatically. A small example verifies the workflow, not the hypothesis.

If you prefer not to activate the environment, replace `python` with
`.\.venv\Scripts\python.exe`.

The example uses a sparse-coding baseline: an orthonormal block dictionary
with exact whole-recording coefficient selection and SVD updates. To run its
larger synthetic screening configuration:

```powershell
python run.py experiment --config configs/synthetic_block_orthogonal.yaml
```

See [the block-model guide](documentation/block_orthogonal.md) for the equations,
training assumptions, and minimum-support evaluation commands.

## The everyday workflow

1. Copy `configs/example.yaml` and give it a descriptive name.
2. Edit the `data`, `model`, `train`, and `evaluation` sections.
3. Run `python run.py experiment --config YOUR_CONFIG.yaml`.
4. Inspect the saved audio, metrics, feature diagnostics, and report.

For speech, use an existing LibriSpeech experiment and add `--download` only when
you want to acquire the corpus. For example:

```powershell
python run.py experiment --config configs/librispeech_orthogonal_1h.yaml --download
```

This is a longer screening experiment. The data audit checks speaker separation
before training. Without `--download`, speech recordings must already be present.

All individual steps remain available:

| Task | Command |
|---|---|
| Prepare data | `python run.py prepare --config CONFIG.yaml` |
| Check dataset separation | `python run.py audit --config CONFIG.yaml` |
| Train | `python run.py train --config CONFIG.yaml` |
| Evaluate a final test split | `python run.py evaluate --run runs/RUN_NAME --split clean` |
| Inspect validation features | `python run.py analyze --run runs/RUN_NAME` |
| Compare runs | `python run.py report --runs runs/RUN_A runs/RUN_B` |
| Prepare a parameter sweep | `python run.py sweep --config configs/sweep_latent_width.yaml` |

`--split clean` applies to LibriSpeech. Synthetic splits are `iid`, `combinations`,
`extrapolation`, and `families`. Run `python run.py --help` or
`python run.py COMMAND --help` for options. The installed `lrac` and
`python -m lrac` commands use the same implementation.

Experiments that need no training use the same entry point with `study`:

```powershell
python run.py study --config configs/study_sparse_search.yaml
python run.py study --config configs/study_sparse_cues.yaml
```

Each creates a new `runs/<name>_<timestamp>/` containing resolved settings,
environment/source records, results and completion status. See the
[codebase guide](documentation/codebase.md) for module responsibilities and how
to add a study. These commands repeat the existing toy experiments; the L1 audio
experiment remains a proposal awaiting review.

## Where things live

Seven working folders, with one level of code organization:

```text
run.py                 # Run the pipeline
lrac/                  # Python code: train.py, evaluate.py, analyze.py, report.py
  study.py             # Common bookkeeping for experiments without training
  study_<topic>.py     # One experiment loop; reusable algorithms stay separate
  models/              # Model implementations and their shared interface
  data/                # Signal generation, loading, and dataset audits
configs/               # All YAML settings in one place
data/                  # Downloaded recordings and generated manifests
runs/                  # Checkpoints, audio, metrics, reports, and caches
documentation/         # Research notes; original init/ documents stay here
tests/                 # All tests in one place
tools/                 # Setup, smoke check, environment lock, and MATLAB check
```

Configuration names distinguish their purpose without subfolders:

- `example.yaml` and `<dataset>_<model>*.yaml`: runnable experiments.
- `study_*.yaml`: experiments without model training.
- `data_*.yaml`: reusable dataset settings.
- `model_*.yaml`: reusable model settings.
- `sweep_*.yaml`: parameter sweeps.

See [writing your own YAML](configs/README.md) for brief training and inference examples.

## Build the mathematical report

With a LaTeX distribution providing `latexmk`, BibTeX, TikZ, and PGFPlots:

```powershell
cd documentation
latexmk -pdf -interaction=nonstopmode -halt-on-error sparse_autoencoder_models.tex
```

The committed PDF builds from the committed source and result table without
datasets or checkpoints. After repeating the local three-run study, regenerate
the numeric export, table, and provenance record from the project root with
`python tools/export_model_results.py --study runs/concurrent_training.json`.
That local study file lists the three run paths in S-NL, D-NL, S-AF order.
For new runs its minimal format is:

```json
{"runs": [
  {"name": "sparse nonlinear", "path": "runs/YOUR_SPARSE_RUN"},
  {"name": "dense nonlinear", "path": "runs/YOUR_DENSE_RUN"},
  {"name": "sparse affine", "path": "runs/YOUR_AFFINE_RUN"}
]}
```

Original training source and checkpoint hashes are preserved in
`documentation/sparse_autoencoder_runs.json`; code has subsequently received
documentation and portability fixes. `data/`, `runs/`, environments, and LaTeX
intermediates are ignored; source files and PDFs are committed.

## Evaluate the three frozen speech models

Evaluate each saved run on the full held-out set with:

```powershell
python run.py evaluate --run runs/YOUR_RUN --split clean --metric-workers 4
```

This uses the saved evaluation budgets and best development-selected checkpoint.
`--metric-workers` parallelizes optional PESQ/STOI on CPU; zero uses the original
serial calculation. It does not change the metric definitions or model weights.
Use `--resume` to reuse completed budgets after interrupting an evaluation with
the same frozen checkpoint and data. Run GPU evaluations sequentially if parallel
runs fill device memory; CPU metric workers can still run in parallel.
During the three-model study, `./tools/status.ps1 -Test` shows test progress.
After all three runs finish, `python tools/compare_models.py` generates the test
comparison, curves, numeric export, and paired speaker uncertainty in
`documentation/`. The local study file uses the same format shown above.

## Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for naming conventions and the specific
files to change when adding an experiment, model, or dataset. Prefer a readable
function and a small configuration over a new framework or duplicate script.

Run the checks with:

```powershell
python -m pytest -q
```

The complete [command reference](documentation/running_experiments.md)
covers installation elsewhere, external-corpus evaluation, checkpoints, sweeps,
and full-band studies. See also the [research specification](documentation/implementation.md)
and [experiment registry](documentation/experiments.md).
