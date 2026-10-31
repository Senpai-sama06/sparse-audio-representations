# Sparse audio representations

Find the fewest values that reconstruct an audio sample accurately, allowing the
representation to adapt without requiring a named transform or semantic features.
Minimum support, minimum L1 norm and minimum encoded bitrate are different goals.

## What we have established

- **Theory:** minimum support is attained whenever the declared feasible finite-code
  family is nonempty. A freely sample-adapted basis can make any nonzero finite
  waveform one-sparse, but describing that basis can carry the waveform itself.
  Existence therefore does not establish useful compression or an efficient search.
  [Full proofs](documentation/sparsest_representation.pdf) and
  [original problem](documentation/init/init.pdf) remain available in LaTeX/PDF.
- **Learned baselines:** sparse nonlinear (S-NL), dense-trained nonlinear (D-NL)
  and sparse affine (S-AF) autoencoders were trained; S-AF performed best among
  those three. Targets are the input waveforms themselves (self-supervised).
- **S-AF architecture:** four strided convolution/residual stages, a 256-channel
  latent projection, global magnitude top-k, and affine convolutional synthesis.
  Training used normalized waveform MSE, 20,000 steps and sampled support budgets.
  A two-second input passes through 16,000 values before its 32,000-slot latent
  projection; full support is not a lossless reference. See the
  [model mathematics](documentation/sparse_autoencoder_models.pdf).
- **Interpretation:** channels specialize in frequency bands and associate with
  voiced speech. Controlled pitch shifts support fixed-frequency filter behavior;
  they do not establish pitch/timbre slots. Channel 53 illustrates that retention
  frequency differs from reconstruction importance. Pitch-envelope preservation
  was imperfect, limiting causal claims. All-channel clustering and compensated
  ablations remain future work.
- **Deterministic studies:** tiny rational searches provide exact minima within a
  finite rotation family. Cue-guided searches, constructive rotations and recurrence
  examples show that the permitted family determines which structure is economical.
  These do not establish a universally sparsest audio domain.

## Data and measured results

Speech is mono 16 kHz LibriSpeech. The cached train-clean-360 contains 921 speakers;
training used a speaker-balanced 10-hour subset. Development and test-clean each
contain 40 speakers, disjoint from training. LJSpeech in the cache is a separate,
single-speaker corpus. Synthetic signals check controlled cases, not all signals.

Earlier whole-utterance test-clean results (2,620 recordings; means):

| Model, support | NMSE (lower) | PESQ-WB (higher) | STOI (higher) |
|---|---:|---:|---:|
| S-NL, 10% | 0.1830 | 1.589 | 0.791 |
| D-NL, 10% | 0.2801 | 1.119 | 0.688 |
| S-AF, 10% | 0.1065 | 2.027 | 0.873 |
| S-AF, full | 0.1032 | 2.472 | 0.899 |

These use unquantized codes: **S-AF bitrate has not been measured**.
[Full model scores](documentation/sparse_autoencoder_test_results.csv).

The L1 pilot solves minimum coefficient magnitude sum in the fixed unit-norm
impulse-plus-DCT dictionary, using float64 FISTA/lambda search. It encoded 120
centered two-second development clips: 20 calibration and 20 measurement speakers,
three unused utterances each, excluding 200 checkpoint-selection utterances.
All 360 fits met a 1% relative constrained primal-dual gap; 3,600 streams were
saved, decoded and scored. DCT thresholding used the same quantizer/coding rules.

Observed rate at common maximum final NMSE, on the 60 measurement clips:

| NMSE limit | L1 kb/s | DCT kb/s |
|---:|---:|---:|
| 0.1 | 39.25 | 24.16 |
| 0.01 | 78.89 | 73.74 |
| 0.001 | 148.26 | 126.80 |

L1 used more bytes on average in this setup; the rate difference at 0.01 has a
95% paired-speaker interval that includes zero. Actual errors can undershoot the
limits differently, so perceptual comparisons are not at identical achieved NMSE.
For L1 the corresponding NMSE/PESQ/STOI means were 0.0861/1.293/0.826,
0.00372/2.744/0.979 and 0.000881/3.823/0.994. Self-reference PESQ/STOI were
4.643888/~1.0. [Complete comparisons](documentation/l1_comparison.csv),
[all coded measurements](documentation/l1_metrics.csv),
[solver records](documentation/l1_solver.csv) and
[rate/quality figure](documentation/l1_rate_distortion.png) retain the detail.

S-AF and L1 used different recordings, durations and encoding conditions: a fair
compression ranking requires S-AF serialization and evaluation on the same clips.
Development data and test-clean have informed new hypotheses; future confirmation
needs fresh evaluation data. No conventional audio-codec superiority is claimed.

## Measurements

| Quantity | Definition / interpretation |
|---|---|
| L0 / support | Number of nonzero coefficients; track before and after quantization. |
| L1 | Sum of absolute coefficient values; the L1 objective, not a support guarantee. |
| NMSE | `sum((x-reconstruction)^2) / sum(x^2)`; silent inputs use absolute error. |
| SNR | `-10 log10(NMSE)` dB; the same waveform error expressed differently. |
| PESQ-WB / STOI | Reference-based speech-quality/intelligibility estimates; retain validity counts. |
| Bitrate / bits per sample | Actual serialized bits divided by duration / original sample count. |
| Runtime / RTF | Encode/decode time separately; RTF is time divided by audio duration. |
| Dual gap | Feasible objective minus a numerical lower bound, with stated normalization. |

Also tracked: SI-SDR, multiresolution STFT, log-spectral, transient and high-band
errors. ViSQOL is not implemented. Speech metrics apply to reconstructed audio.
Primary coding quality is measured **after decoding saved bytes**, without free
alignment/gain fitting. The SAC1 codec includes a 28-byte header, dictionary ID,
length/rate, binary64 quantizer step, support bitmap/indices, values and padding;
precisions are 4/6/8/12/16 bits. Raw mono 16-kHz PCM16 is a 256-kb/s reference.
Shared learned-model storage must be reported separately.

## Exhaustive L0 activity

A separate search uses the full two-second clip `librispeech:3081-166546-0005`,
64,000 impulse/DCT atoms and NMSE <= 0.01. It enumerates supports without pruning
or a time limit; each support receives an independent SVD least-squares fit.
32 workers use one numerical-library thread each. Disjoint chunks are committed
in order, with five-second checkpoints and OS locking; a restart may replay
uncommitted work but does not skip supports. Source/implementation hashes protect
resume. Progress is live, not a completion claim:

```powershell
Get-Content runs/l0_librispeech_20261029T120949726Z/progress.json
# Only after the previous worker has stopped:
.\tools\start_l0.ps1 -ResumeRun runs/l0_librispeech_20261029T120949726Z -Workers 32
```

A feasible code gives an upper support bound; exhausting smaller supports gives a
lower bound. Float64 and a boundary guard provide numerical evidence, not an exact
certificate. Completion would be a reference for this clip/dictionary/tolerance,
not every representation domain or minimum bitrate. Check `search_state.json`
for authoritative completion; a reboot requires the resume command.

## PitchFlower RVQ / Gaussian-splatting training

Prepared from the [official PitchFlower implementation](https://github.com/diegotg2000/PitchFlower)
and [GS-Codec](https://github.com/ronaluf/gs-codec), pinned in
`tools/setup_pitchflower.py`. **PitchFlower training has not been launched.**
When ready, one foreground command runs the entire resumable campaign:

```powershell
.\tools\run_pitchflower.ps1
```

It prepares a shared offline WORLD flat/shift cache, trains RVQ then GS sequentially
on the full cached train-clean-360 (104,014 files / 921 speakers / 363.6 hours),
then trains the GS predictor and calibrates 6-bit scalar centroids. Original
LibriSpeech is resampled from 16 to 24 kHz; dev-clean speakers are separate.
Crops use the released peak normalization convention, unlike the earlier
amplitude-preserving baseline; waveform gains would need separate codec accounting.
Both codecs use PitchFlower's ConvNeXt/attention/flow backbone, CFM plus commitment,
1M optimizer steps, AdamW 1e-4 and effective batch 32 (microbatch 4 x accumulation 8).
The GS predictor uses 100k steps and effective batch 16. Configuration:
`configs/study_pitchflower_librispeech.yaml`.

RVQ uses 8 x 512 entries, dimension 256. GS uses the paper's C=32, shared widths,
fixed uniform centers, 100 inner Adam steps at LR 0.4 and STE outer gradients.
Its density 80/225 becomes 25 Gaussians over 71 latent frames here; this is a
PitchFlower adaptation, not a reproduction of GS-Codec's SEANet/GAN backbone.
Commitment weights are respectively 0.25 and 20. Paper CFG dropout is implemented
as 10% per condition, correcting the inverted F0 comparison/absent content dropout
in the reference loss. Neither RVQ dropout nor training-time GS quantization is used.
The GS predictor uses the released implementation (four dilated blocks, positional
encodings and one attention decoder); scalar PTQ comes
after training. No measured bitrate or quality results are claimed yet.

Terminal bars show preprocessing and optimizer-step/loss progress. Outputs are in
`runs/pitchflower_librispeech/{rvq,gs,predictor}/`: checkpoints, TensorBoard logs,
progress JSON and final Vocos waveform examples at -6/0/+6 semitones. The root
`progress.json` mirrors the current phase. Rerun the same command after Ctrl+C
to resume; model/optimizer state is restored, but data-order/crop replay is not
guaranteed identical. Cached utterances are reused. `-Phase prepare|rvq|gs|predictor|calibrate`
selects one phase; edit resource settings before the first run. This launcher does
not stop or resize the ongoing L0 search. Preprocessing may require substantial
time and roughly 250 GB before compression. Paper WER/SIM/UTMOS benchmarks remain
a later evaluation, not metrics computed by this training recipe.

## Run and extend

```powershell
.\tools\bootstrap.ps1
.\.venv\Scripts\Activate.ps1
python run.py experiment --config configs/example.yaml
python run.py study --config configs/study_l1_synthetic.yaml
python run.py study --config configs/study_l1_librispeech.yaml
python run.py evaluate --run runs/YOUR_RUN --split clean --metric-workers 4
```

`experiment` prepares/audits data, trains, evaluates validation and reports;
`study` runs numerical experiments without training. No downloads occur without
`--download`. Use `python run.py COMMAND --help` for options. Saved checkpoints
supply the model for `evaluate`; `--data-config` selects external evaluation data.

Copy an existing YAML. Training files use `data`, `model`, `train`, `evaluation`;
`train.name` labels the run. Studies use `name`, `kind`, `parameters`, for example:

```yaml
name: my_l1_check
kind: l1
parameters:
  dataset: synthetic
  targets: [0.1, 0.01]
  workers: 4
```

`extends` reuses YAML files relative to that file; child values override parents.
Ordinary configuration paths resolve against the project root and expand `~`;
study-specific paths are resolved by the study. Use a separate manifest directory
when changing data preparation. Keep source crops/speakers within one data role.
Resolved settings, source hashes, metrics, audio and checkpoints live under `runs/`.

Code stays in flat `lrac/` modules, with `models/` and `data/` subfolders;
settings in `configs/`, helpers in `tools/`, tests in `tests/`. Add a numerical
study as `study_<topic>.py: run(output, *, ...)`, an explicit branch in `study.py`
and a small YAML. Reuse metrics/data routines; add no framework for one experiment.
New models use `models/base.py` and selection/configuration in `models/__init__.py`
and `config.py`. **Comment new code with useful docstrings and explanations of
mathematics, units and non-obvious choices.** Check meaningful numerical changes
with `python -m pytest -q`.

Maintain this README as the only explanatory Markdown guide. Generated run reports
stay in ignored `runs/`; selected numerical/figure/provenance artifacts stay in
`documentation/`. Preserve the LaTeX/PDF sources and keep paper titles/authors in
[bibliography.md](bibliography.md); that ledger includes screened and partially
reviewed work, not a claim that every proof was audited. Git history retains the
superseded Markdown discussions.
