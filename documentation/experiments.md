# Experiment registry

## Sparse-autoencoder baseline: run protocol

Status: the sparse autoencoder is implemented, all 60 tests pass, and the new
1h speech pilot completed 2,000 updates. The 10h main run and matched dense-trained
and affine controls have each completed 20,000 updates and validation reports.
Final held-out test reconstruction remains separate. See the
[baseline guide and run record](sparse_autoencoder.md) for actual code,
configuration, checkpoints, and measurements. Earlier grouped-autoencoder smoke
runs are not this baseline. Invertible-model experiments remain deferred.

### Data

Use the existing LibriSpeech plan for the primary speech baseline and retain a
separate synthetic study. Do not mix their training examples or pool their scores.

| Role | Proposed data | Use |
|---|---|---|
| Engineering check | Small, separate synthetic smoke manifests | Verify code, gradients, support counting, boundaries, and checkpoint reload |
| Controlled synthetic baseline | Existing 10,000 training / 1,000 validation signals | Train a separate model on harmonic/noise/transient mixtures |
| Synthetic final evaluation | Four existing 1,000-example regimes | New draws, held combinations, extrapolation, and chirp/AM-FM/damped families; keep final sets out of selection |
| Speech pilot | Speaker-balanced 1-hour subset of cached train-clean-360 | Catch data, optimization, and memory problems before the main run |
| Main speech training | Speaker-balanced 10-hour subset of cached train-clean-360 | First trained speech baseline; preserve all 921 available training speakers |
| Speech validation | dev-clean | Checkpoint selection and development only |
| Final speech tests | test-clean and test-other, scored separately | Evaluate the frozen baseline after run choices are settled |
| Later external evaluation | VCTK mic2, frozen model | A separate generalization study after the first baseline |

The corpus paper reports 251 speakers in train-clean-100, 40 in dev-clean,
40 in test-clean, and 33 in test-other, with speaker-separated training,
development, and test partitions. Our code already builds deterministic
speaker-balanced 1h/10h subsets; actual durations and speaker coverage still
must be verified after acquisition. See [Panayotov et al.](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf).
Ten hours is a project subset, not an official corpus partition.

No transcripts, speaker embeddings, phoneme labels, or acoustic-feature targets
are model inputs. Speaker IDs are for sampling, auditing, and reporting only.
Keep every crop from an utterance within its official role. Audit speaker IDs,
source intervals, and duplicate decoded recordings before training. Report
training/validation quality separately; speaker separation does not eliminate
the possibility of overfitting.

Use native 16 kHz mono speech, float32 model inputs, and random two-second
training crops. Preserve the recording's amplitude; any normalization introduced
later must be reversible and its required parameters accounted for. Short
utterances are padded and masked; do not count padded samples as useful data.
The target is the provided waveform, including its existing recording detail.

Speech evaluation should reconstruct full utterances, with a global support
budget for each utterance. Also use a fixed matched-duration validation probe
to identify a possible train-crop versus full-utterance mismatch. Keep those
measurements separate. Freeze validation probe IDs/offsets; report exact coverage.

### KaggleHub cache inspection, 27 October 2026

The project data directory contains only synthetic manifests, but an inspection
of the user's KaggleHub cache found already downloaded speech elsewhere. This
corrects the earlier assumption that speech acquisition was still entirely needed.

Cache root: `C:/Users/RF AND SIMULATION/.cache/kagglehub/datasets/`.

| Cached dataset | Findings | Baseline relevance |
|---|---|---|
| `mathurinache/the-lj-speech-dataset/versions/1/LJSpeech-1.1` | 13,100 WAV files, sampled headers at 22,050 Hz mono; README identifies a single speaker | A different dataset from LibriSpeech; cannot support unseen-speaker evaluation by itself |
| `pypiahmad/librispeech-asr-corpus/versions/1/train-clean-360/LibriSpeech/train-clean-360` | 104,014 FLAC files, 921 speakers; metadata reports about 363.6h | Recommended existing source for balanced 1h/10h subsets, instead of acquiring train-clean-100 |
| Same LibriSpeech cache, `dev-clean/LibriSpeech/dev-clean` | 2,703 FLAC files, 40 speakers | Existing validation partition |
| Same LibriSpeech cache, `test-clean/LibriSpeech/test-clean` | 2,620 FLAC files, 40 speakers | Existing final clean evaluation partition |
| `dogrose/musan-dataset/versions/1/musan` | 2,016 WAV files across music/speech/noise; about 11.73 GiB | Possible later audio/noise study; keep separate from the first speech training |
| `omblivion/librispeech-small/1.archive` | 434,110,464-byte archive with a ZIP signature, but Python cannot read its ZIP directory; no extracted version/completion marker | Appears incomplete or damaged; do not rely on this copy |

The LibriSpeech folders' actual speaker-ID sets match their included SPEAKERS.TXT
metadata. Train-clean-360, dev-clean, and test-clean have zero pairwise speaker
overlap in the inspected cache. Three distributed files per split decoded as
finite, mono, 16 kHz audio. This is a file/metadata and sample-decode inspection,
not a checksum verification or a complete decoded-audio duplicate audit.
The inspection record is `runs/dataset_cache_inventory.json`.

Neither train-clean-100 nor test-other was found in this cache. The revised
recommendation is to use the already available train-clean-360 and preserve
speaker-balanced subset construction. This changed training speaker coverage
from the original proposed 251 to 921. The generated 1h subset has 948 source
records and the 10h subset has 3,329; both cover all 921 speakers. The table above
now records the cached 360h-source protocol actually selected.

Manifest preparation and auditing now support train-clean-360 and the nested
cache layout, with explicit expected speaker counts. The recordings are read
in place rather than copied to satisfy path assumptions.
Exclude intro-disclaimers, original-books, and raw-metadata from training audio.

No additional download is needed for the clean pilot. For the planned harder
evaluation, obtain `test-other.tar.gz` (listed as 328 MB) from
[OpenSLR's official LibriSpeech page](https://www.openslr.org/12).
Download `train-clean-100.tar.gz` (listed as 6.3 GB) only if we choose to preserve
that exact original training-source protocol. Verify downloaded archives against
the [official checksums](https://www.openslr.org/resources/12/md5sum.txt).

### Baseline model and scope

Implement an independent learned convolutional encoder/decoder as the first
sparse autoencoder. Reusing the existing temporal building blocks keeps the
comparison simple and permits full-utterance evaluation. It is a deliberately
local, shared-parameter baseline, not a claim to have selected the best architecture.

- Four encoder stages with channel widths 32/64/128/128 and stride four each;
  mirrored decoder. Kernel and residual-block details start from existing code.
- One ungrouped signed latent representation, width 256, total temporal stride
  256. For a two-second 16 kHz crop this gives 125 x 256 = 32,000 available values,
  matching its input scalar count before sparsification. Partial-tail/padding
  behavior must be tested rather than assumed.
- Retain values across the entire input, not independently at every latent time
  step. The existing `code_budget` behavior does not yet do this.
- No waveform bypass, attribute supervision, orthogonality, group-separation
  penalty, or functional-separation penalty.
- Begin with a budget-independent encoder and explicit global magnitude
  selection. Budget-conditioned encoders and learned support scores are later
  variants. Report this inference restriction rather than claim optimal support.

The existing layer pattern has an encoder receptive field of 2,041 samples
(about 128 ms at 16 kHz), computed from the four stride-four kernel-nine layers
and the two kernel-three convolutions per residual block. A global support
selection does not make these local candidate values depend on the whole
utterance. This is a material baseline limitation to state in results.

### Training stages and controls

| Stage | Proposed effort | Purpose |
|---|---|---|
| Smoke | A few updates on the tiny synthetic set | Numerical and integration checks only |
| Synthetic screening | 2,000 updates on the synthetic training set | First controlled support/reconstruction curves |
| Speech pilot | 2,000 updates on the balanced 1h subset | Throughput, optimization, and generalization probe |
| Main baseline | 20,000 updates on the balanced 10h subset | First substantial speech measurement |
| Confirmation | Repeat selected settings at seeds 1 and 2, then consider 100h | Assess repeatability and data scaling after the baseline works |

Proposed main defaults: seed 0, AdamW, learning rate 2e-4, weight decay 1e-4,
effective batch eight two-second clips, gradient clipping at one. Run a short
measured GPU throughput/memory probe before predicting duration or changing batch
size. The local RTX 2000 Ada has 16 GB VRAM; no wall-clock estimate has been
benchmarked for the new baseline. Record examples and audio seconds processed
as well as optimizer steps; update counts are not evidence of convergence.

For sparse training, sample a declared support fraction per batch from
1%, 2%, 5%, 10%, 25%, 50%, and 100%, applied over each example's valid latent
coordinates. This exposes the model to several resource levels and includes
full-budget reconstruction. Use an explicit zero-code diagnostic during
evaluation. An initial dense warm-up is optional and must be logged; it is not
an unreported change to the study.

Start with valid-sample waveform MSE normalized by fixed training-set power.
Log per-example NMSE separately. This keeps the first optimization target
explicit. Monitor spectral and perceptual measurements without adding their
weights to the first loss. Selection already enforces support; no additional
l1 or group penalty is required for this candidate.

Train a matching dense autoencoder with the same data, architecture, update
budget, and seed, then prune it at evaluation. Also evaluate a fixed
nonoverlapping DCT reference at matched whole-input coefficient counts. These
controls distinguish learning a sparse code from ordinary reconstruction ability
and from the usefulness of a standard transform. Record parameter counts and
actual support for every condition.

Use a fixed speaker-balanced dev-clean probe (proposed 200 utterances covering
all 40 speakers) for frequent validation, then evaluate all dev-clean recordings
on the selected checkpoint. Select a single shared checkpoint by a predefined
mean waveform validation loss over the declared nonzero budgets; also log dense
quality and the individual budget scores. Do not select separately on test data
or change metric weights after looking at final test results.

### Measurements

For each input, sweep the same declared fractions and a zero-code diagnostic.
Measure actual reconstruction; predicted sparse-code magnitude is not a quality
metric. Save:

- Actual nonzero values per utterance, per second, and as a fraction of the
  input scalar count; selected positions and required metadata are separate
  information to account for, not free hidden inputs.
- MSE, MAE, NMSE, waveform SNR, and SI-SDR. Ordinary SNR/NMSE retain amplitude
  sensitivity; SI-SDR alone is insufficient.
- Multi-resolution spectral loss, complex-STFT error, log-spectral distance,
  high-frequency error, and transient error using existing diagnostics.
- PESQ and STOI on valid speech examples. Their packages are installed, but
  these measurements still require integration into this baseline evaluator.
- ViSQOL if installed and integrated. It is currently unavailable; report that
  explicitly rather than create a score or silently omit it.
- Training/validation curves, parameter count, numerical precision, memory,
  inference time, representative reconstructions, and poor examples.

PESQ/STOI are speech diagnostics, not scores to interpret for synthetic chirps
or arbitrary noise. Record invalid or silent cases and failure reasons without
assigning fabricated quality values. Never pool unlike datasets or durations.

Report per-example distributions and speaker-macro summaries with speaker-level
bootstrap intervals. At NMSE targets 0.1, 0.01, and 0.001, record the smallest
successful **tested** support and the fraction of examples meeting each target.
These correspond to waveform SNRs of 10, 20, and 30 dB. If no tested code meets
the target, retain a failure entry. Do not assume practical error curves are
monotone or interpret this searched support as a global certificate.

The primary deliverable is support-versus-reconstruction curves, with dense,
sparse-trained, and DCT results on identical examples. The first synthetic and
speech pilots are exploratory measurements, not proof of a universal sparsest
representation or evidence for all audio domains.

### Dataset source reviewed

**47. Panayotov, Chen, Povey, and Khudanpur, LibriSpeech: an ASR corpus based on
public domain audio books** ([paper](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf)).
Corpus-structure and speaker-partition sections reviewed for this protocol;
the dataset was designed for read-speech recognition, not to establish general
audio reconstruction performance. Added to [bibliography.md](../bibliography.md).

## Earlier candidate studies

These are earlier candidate experiments, not an agreed architecture progression.
The [current architecture discussion](implementation.md) compares reconstruction
families and temporal assumptions. The earlier global sparse-bottleneck sketch
is deferred until those architecture choices are justified.
See the [adaptive-transform review](adaptive_transforms.md) for the current
research discussion and the [initial literature review](literature_review.md)
for background. Admissible adaptation, reconstruction information, and evidence
for minimum support must be settled before selecting the next model.

Freeze configuration and manifest hashes before consulting the final test sets.
Exploration on validation data must be logged as exploration. All tests below
admit a negative outcome; no feature family is assumed to exist in advance.

| ID | Hypothesis | Comparison and evidence |
|---|---|---|
| R0 | Analysis/synthesis implementation is correct | Full DCT, orthogonal, random, PCA round trips including boundaries |
| S1 | Learned dictionaries capture recurring structure | Learned vs fixed atoms at matched activity; synthetic controlled recovery |
| S2 | Geometric groups expose complementary families | Atom-only vs sparse-group inference; group ablation and seed stability |
| S3 | Translation reuse explains useful signal structure | Learned convolutional kernels vs random kernels with identical inference |
| S4 | Nonlinear groups add reconstructive flexibility | Autoencoder at widths 32/64/128/256, same training data |
| S5 | Functional separation improves factor organization | Matched nonlinear architecture, zero/nonzero Jacobian penalty; monitor dead groups |
| G1 | Results transfer to unseen speakers | Official LibriSpeech dev and test roles; speaker-macro scores and bootstrap CIs |
| G2 | Results transfer across recording conditions | Frozen VCTK mic2 evaluation, no adaptation or checkpoint selection |
| G3 | Results transfer across tested signal families | New draws, held combinations, extrapolation, held-out chirp/AM-FM/damped signals |
| G4 | Speech and synthetic structures transfer | Frozen speech->synthetic and synthetic->speech evaluations, reported separately |
| F1 | Selected results persist at full bandwidth | Separate 48 kHz VCTK speaker-disjoint training study |

## Progression

1. Pass numerical and audit tests; complete short dictionary and functional GPU runs.
2. Evaluate exact references on all synthetic regimes. Inspect fixed, random, and learned atoms.
3. Screen dictionary/grouped/convolutional variants; retain reconstruction/activity frontiers.
4. Compare nonlinear variants, first without and then with separation constraints.
5. Repeat selected configurations at three seeds and scale training to the full manifest.
6. Evaluate final LibriSpeech and external corpora once selections are frozen.
7. Write conclusions separating reconstruction from structure and generalization.

The default `configs/sweep_*.yaml` commands **only resolve configurations**. Add
`--execute` to run the finite axis sequentially. The initial study does not require
a cloud account or an unbounded parameter search.

## First implemented sparse-coding model

`configs/synthetic_block_orthogonal.yaml` now implements an l0-budgeted,
nonoverlapping orthonormal dictionary with minibatch Procrustes updates. See
[the model](block_orthogonal.md) and [pilot results](block_orthogonal_results.md).
Its exact coefficient-selection result is conditional on a fixed transform and
waveform squared error. The initial DCT outperformed the final learned iterate in
the first validation run; the final test sets remain untouched.
