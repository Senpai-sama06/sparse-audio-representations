# Running individual experiments

This is the detailed command reference. Run commands from the project root.
For the shorter starting workflow, see [README](../README.md).

This project asks whether speech and controlled signals admit reusable constituent
features that jointly reconstruct previously unseen samples accurately. It compares
dictionary atoms, grouped subspaces, convolutional patterns, and nonlinear latent
groups. Reconstruction, feature structure, separation, and generalization are
evaluated independently. The initial implementation uses continuous values; active
coefficient counts are **not transmitted bitrates**.

## Start on this workstation

Python 3.11 and CUDA-enabled PyTorch 2.8 are available. The local `.venv` reuses the
workstation's installed scientific packages through `--system-site-packages`; the
project and build tools are installed locally. It does not replace global packages.

```powershell
.\tools\bootstrap.ps1
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m lrac --help
```

For an independently installed environment on another Windows machine, create a
fresh venv, install `requirements-lock.txt`, then `pip install --no-deps -e .`.
The lock records the project's dependency closure on Windows/Python 3.11, including
the CUDA 12.8 PyTorch wheel. CPU/Linux environments need a platform-specific lock.

## Run the verified small pipeline

```powershell
.\.venv\Scripts\python.exe -m lrac prepare --config configs/smoke_orthogonal.yaml
.\.venv\Scripts\python.exe -m lrac train --config configs/smoke_orthogonal.yaml
.\.venv\Scripts\python.exe -m lrac train --config configs/smoke_autoencoder.yaml
```

Each training command prints `RUN=...`. Substitute that directory below:

```powershell
.\.venv\Scripts\python.exe -m lrac evaluate --run runs/YOUR_RUN --budgets 0,8,16
.\.venv\Scripts\python.exe -m lrac analyze --run runs/YOUR_RUN --split validation
.\.venv\Scripts\python.exe -m lrac report --runs runs/YOUR_RUN --output runs/reports/comparison
```

Budget `0` retains every coefficient; a positive budget retains that many coordinates
per time step, across all groups. Thresholding still applies. A full coefficient
budget does not undo sparse thresholding. Smoke runs and limited evaluations are
tagged explicitly. Their results verify execution, not the research hypothesis.

## Synthetic experiments

```powershell
.\.venv\Scripts\python.exe -m lrac prepare --config configs/data_synthetic.yaml
.\.venv\Scripts\python.exe -m lrac audit --config configs/data_synthetic.yaml
.\.venv\Scripts\python.exe -m lrac train --config configs/synthetic_dct.yaml
.\.venv\Scripts\python.exe -m lrac train --config configs/synthetic_orthogonal.yaml
```

Preparation writes deterministic manifests, not gigabytes of redundant WAV files.
Signals and known components are generated on demand. Defaults are 10,000 training,
1,000 validation, and 1,000 examples **per** test regime: new draws, held-out
combinations, extrapolation, and unseen chirp/AM-FM/damped families. Training and
test seed ranges do not overlap. Generators include both additive mixtures and
shared time-varying filtering interactions.

Other ready-to-run configurations cover PCA, random bases, grouped dictionaries,
overcomplete dictionaries, learned and random convolutional kernels, and the
autoencoder. Combine a dataset and model in YAML using `extends`.

## Prepare speech and audit the speakers

```powershell
.\.venv\Scripts\python.exe -m lrac prepare --config configs/data_librispeech.yaml --download
.\.venv\Scripts\python.exe -m lrac audit --config configs/data_librispeech.yaml
.\.venv\Scripts\python.exe -m lrac train --config configs/librispeech_orthogonal_1h.yaml
```

Downloads are explicit and checksum-verified. The LibriSpeech command acquires
`train-clean-100`, `dev-clean`, `test-clean`, and `test-other` from OpenSLR. If the
recordings already exist, place `LibriSpeech/` under `data/raw/librispeech/` and omit
`--download`. Archives are retained alongside extracted files. Allow roughly
20 GB of disk headroom; no speech corpus is bundled with this repository.

LibriSpeech's standard splits contain 251 training speakers, 40 validation speakers,
40 clean-test speakers, and 33 other-test speakers. The audit verifies counts,
speaker intersections, source IDs, source intervals, actual audio metadata,
filename metadata, and identical decoded audio across roles. Training fails on an
audit error. Hashing is deliberately enabled and can make the initial audit slow.

One-hour and ten-hour training manifests use equal per-speaker duration budgets
and deterministic source intervals, retaining all training speakers. Validation
uses round-robin speaker coverage rather than a prefix of speaker-sorted files.
The default bounded validation set is 64 utterances. Final evaluation uses whole
utterances and all available examples unless `--limit` is supplied.

## External corpus and transfer

```powershell
.\.venv\Scripts\python.exe -m lrac prepare --config configs/data_vctk.yaml --download
.\.venv\Scripts\python.exe -m lrac evaluate --run runs/YOUR_SPEECH_RUN --data-config configs/data_vctk.yaml
.\.venv\Scripts\python.exe -m lrac evaluate --run runs/YOUR_SPEECH_RUN --data-config configs/data_synthetic.yaml
.\.venv\Scripts\python.exe -m lrac evaluate --run runs/YOUR_SYNTHETIC_RUN --data-config configs/data_librispeech.yaml
```

These commands keep the model frozen. Dataset changes do not fine-tune it.
VCTK external evaluation selects up to ten `mic2` utterances per available speaker
and resamples them to 16 kHz. Its official archive is about 11 GB compressed;
allow additional extraction headroom. The corpus has separate licensing terms;
the downloaded archive retains its license and README.

For a separate full-band study, `configs/data_vctk_fullband.yaml` prepares an
80/10/remaining-percent speaker-disjoint split at 48 kHz using fixed seed 0.
`vctk_autoencoder_48k.yaml` trains a new model with gradient accumulation. Full-band
results are a separate study, not an upsampled version of 16 kHz reconstruction.

## Sweeps, checkpoints, and diagnostics

```powershell
.\.venv\Scripts\python.exe -m lrac sweep --config configs/sweep_functional_strength.yaml
.\.venv\Scripts\python.exe -m lrac sweep --config configs/sweep_functional_strength.yaml --execute
.\.venv\Scripts\python.exe -m lrac train --config YOUR_CONFIG.yaml --resume runs/YOUR_RUN/checkpoints/last.pt
```

Without `--execute`, a sweep writes resolved configurations only. Sweeps vary one
explicit axis and execute sequentially on the local GPU. Provided axes cover
sparsity, functional strength, latent width, and training seeds.

Checkpoints include model, optimizer, resolved configuration, training power, and
RNG states. Resume restores these states, but restarts random-crop/sampler position;
it is not a bitwise continuation of the data stream. Only load trusted local
checkpoints. Model selection uses validation reconstruction, never final test scores.

Analysis saves atom arrays and clusters, coordinate activity, group variance,
acoustic-proxy correlations, synthetic component alignments, and audio from group
removal and perturbation. Reports include speaker-level bootstrap intervals and
cross-run atom/subspace or group-signature comparisons. A correlation or a low
Jacobian interaction score alone does not establish factor recovery.

See [methodology](implementation.md),
[experiment registry](experiments.md), and
[validation record](validation.md).

## Layout

`lrac` holds reusable code; `configs` defines datasets, models, and runs;
`tests` checks numerical correctness and data isolation; `documentation` records
research decisions. Generated runs, reports, and caches live under `runs`, which
is ignored by Git along with bulk audio.
The original `documentation/init/init.tex` remains unchanged. MATLAB is optional
for numerical cross-checks, not a second training implementation.

## Primary references

- [LibriSpeech corpus and speaker partitions](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf)
- [LibriSpeech downloads and license](https://www.openslr.org/12/)
- [VCTK 0.92](https://datashare.ed.ac.uk/handle/10283/3443)
- [K-SVD and overcomplete dictionaries](https://freddy.cs.technion.ac.il/wp-content/uploads/2017/12/K-SVD-An-Algorithm-for-Designing-Overcomplete.pdf)
- [Efficient auditory coding](https://www.nature.com/articles/nature04485)
- [Functional orthogonality and its assumptions](https://proceedings.mlr.press/v306/simon26b.html)
