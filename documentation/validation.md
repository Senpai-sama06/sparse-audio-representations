# Implementation validation — 2026-10-26

The Python research pipeline is implemented. This record concerns software and
numerical validation, not evidence that meaningful reconstructive speech factors
have already been discovered.

## Environment

- Windows, Python 3.11, PyTorch 2.8.0+cu128.
- NVIDIA RTX 2000 Ada Generation with 16 GB VRAM.
- Local `.venv` reuses the workstation's existing scientific packages; project
  installation and updated packaging tools reside in that environment.
- `requirements-lock.txt` records 58 distributions in the project dependency closure.

## Executed checks

`python -m pytest -q` in the local environment: **34 passed**.

Coverage includes exact transform boundaries, PCA mean handling, sparse inference,
convolution adjoints, dictionary gradients, neural lengths, functional gradients,
randomized versus exact Jacobian interactions, finite differences, speaker and
audio leakage rejection, archive traversal/link rejection, round-robin validation,
checkpoint reload/resume, rejection of changed-dataset resume, evaluation, and report generation.

`python -m compileall -q src` completed successfully.

MATLAB R2025b independently ran `tools/check_roundtrip.m`: relative waveform
reconstruction error **2.57e-14**.

## Prepared data

Deterministic full synthetic manifests have been generated and audited:

| Role | Examples |
|---|---:|
| Train | 10,000 |
| Validation | 1,000 |
| Test: new draws | 1,000 |
| Test: held-out combinations | 1,000 |
| Test: extrapolation | 1,000 |
| Test: unseen families | 1,000 |

Waveforms and known components are generated on demand. Seed intersections were
checked and the audit passed. A smaller, separate smoke dataset is also present.

## Executed model runs

- Learned orthogonal dictionary: eight GPU updates on the smoke dataset.
- Grouped waveform autoencoder: eight GPU updates, including differentiable
  functional separation, on the smoke dataset.
- Full DCT reference: validation and limited test evaluation with zero training updates.

The full DCT reference evaluated 16 signals per synthetic test regime. With all
coefficients retained, median ordinary SNR was approximately 100.07–101.18 dB.
This verifies the float32 analysis/synthesis path. It does not demonstrate compact
feature discovery: an exact reference transform retains all its coefficients.

The short learned-model runs are execution checks and are not trained research
models. In particular, eight autoencoder updates do not deliver near-accurate
reconstruction. Scores are retained honestly in the generated report.

Evaluation, group interventions, synthetic-component diagnostics, acoustic proxy
correlations, atom clustering, functional diagnostics, single-axis sweep resolution,
and HTML/Markdown/PNG reporting have been exercised.

The generated report is at `runs/reports/implementation_validation/index.html`. Generated
artifacts are intentionally ignored by Git. Plots use matching example-coverage
hashes and stage tags to avoid combining unlike evaluations into one curve.

## Work not executed

LibriSpeech and VCTK have **not** been downloaded. Corpus acquisition, integrity
checks, speaker audits, nested subsets, and external evaluation are implemented and
documented, but no claim is made that actual downloaded speech has passed the audit.

The 2,000/20,000-update studies, three-seed comparisons, 100-hour confirmation,
cross-domain conclusions, and 48 kHz study have not been run. Their configurations
and commands are available. Streaming, quantization, and entropy coding remain
outside this initial representation-discovery implementation.

## Researcher workflow update

The project now has one root entry point, `run.py`, and an `experiment` command
that prepares data, audits/trains, evaluates validation examples, analyzes features,
and writes a per-run report. Individual commands remain available. The default
workflow leaves final test evaluations untouched.

`configs/example.yaml` is a self-contained starting configuration.
The shorter README covers everyday use; `CONTRIBUTING.md` describes where to add
experiments, models, and datasets. The complete command reference is retained in
`documentation/running_experiments.md`.

Ambiguous `*_dictionary*.yaml` experiment names now use `*_orthogonal*.yaml`;
functional autoencoder experiment names explicitly include `autoencoder`.
Existing run directories and checkpoints were retained.

Validation after this update: **35 tests passed**. The documented example completed
all five steps using the local GPU and produced
`runs/example_20261026T084850606573Z_seed0/report/index.html`.

## Flat folder layout

The project now has seven working folders: `lrac`, `configs`, `data`, `runs`,
`documentation`, `tests`, and `tools`. The Python package has only two subfolders,
`models` and `data`; the other steps are plainly named modules. Configurations,
tests, and research notes no longer have category subfolders. The original
`documentation/init` documents stay in place.

`runs/reports` contains the previous top-level reports. Setup scripts and the MATLAB
check now share `tools`. The old notebook placeholder is a short documentation
page rather than a dedicated empty folder. Existing run directories and checkpoint
contents were not rewritten. `runs/layout_migration.json` records the file moves.

After the move, all **35 tests passed**, the full `configs/example.yaml` workflow
completed, and both renamed sweep specifications resolved. Dictionary and neural
checkpoints created before the reorganization still load and reconstruct signals.
The new example report is
`runs/example_20261026T092129716494Z_seed0/report/index.html`.
