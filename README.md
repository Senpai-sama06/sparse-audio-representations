# Sparse audio representations

Find the fewest values that reconstruct an audio sample accurately, allowing the
representation to adapt without requiring a named transform or semantic features.
Minimum support, minimum L1 norm and minimum encoded bitrate are different goals.

## What we have established

- **Theory:** minimum support is attained whenever the declared feasible finite-code
  family is nonempty. A freely sample-adapted basis can make any nonzero finite
  waveform one-sparse, but describing that basis can carry the waveform itself.
  Existence therefore does not establish useful compression or an efficient search.
  Full proofs are available in the LaTeX source at
  [documentation/sparsest_representation.tex](documentation/sparsest_representation.tex),
  alongside the original problem source at `documentation/init/init.tex`.
- **Learned baselines:** sparse nonlinear (S-NL), dense-trained nonlinear (D-NL)
  and sparse affine (S-AF) autoencoders were trained; S-AF performed best among
  those three. Targets are the input waveforms themselves (self-supervised).
- **S-AF architecture:** four strided convolution/residual stages, a 256-channel
  latent projection, global magnitude top-k, and affine convolutional synthesis.
  Training used normalized waveform MSE, 20,000 steps and sampled support budgets.
  A two-second input passes through 16,000 values before its 32,000-slot latent
  projection; full support is not a lossless reference. See the
  Model mathematics are in the LaTeX source at
  `documentation/sparse_autoencoder_models.tex`.
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
Full model scores are recorded in the project’s research results archive.

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
4.643888/~1.0. Complete comparisons, coded measurements, solver records and the
rate/quality figure are retained in the research results archive.

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
`tools/setup_pitchflower.py`. One foreground command starts or resumes the
hardware-tuned weekly campaign:

```powershell
.\tools\run_pitchflower.ps1
```

The launcher now selects `configs/study_pitchflower_hardware.yaml`: batch 64,
125k updates per codec (the same 8M crops as batch 32 x 250k), peak learning rate
1.4142e-4 and 1,000-update linear warm-up. Validation/checkpoint intervals are
halved in updates to retain their crop-based cadence. Predictor settings and
72/72/12-hour caps are inherited unchanged. These are tuning choices, not a
guarantee of equal convergence. Batch 64 matched batch 128's measured compute
throughput while using about 7.9 rather than 14.6 GiB of allocated VRAM.
The profile uses four data-loader workers and mapped NPY reads. Cold-corpus
checks still show HDD waits; their throughput is substantially below resident-GPU
throughput. Raw comparisons are recorded in
the hardware tuning record; these are short samples during
ongoing cache preparation, not full-run ETAs.
The corrected upstream pitch KEEP-mask semantics are covered by a regression
test: training drops pitch 10%, validation retains it, and the unconditional
guidance branch drops it. This objective correction uses a fresh run under
`runs/pitchflower_hardware`; older checkpoints remain preserved. Use
`-Config configs/study_pitchflower_week.yaml` to select the older weekly schedule.

It prepares a shared offline WORLD flat/shift cache and lossless memory-mapped
copies, then trains RVQ and GS sequentially
on the full cached train-clean-360 (104,014 files / 921 speakers / 363.6 hours),
then trains the GS predictor and calibrates 6-bit scalar centroids. Original
LibriSpeech is resampled from 16 to 24 kHz; dev-clean speakers are separate.
Crops use the released peak normalization convention, unlike the earlier
amplitude-preserving baseline; waveform gains would need separate codec accounting.
Both codecs use PitchFlower's ConvNeXt/attention/flow backbone, CFM plus commitment,
AdamW 1e-4 and effective batch 32 in the baseline configuration. The baseline
`configs/study_pitchflower_week.yaml` caps each codec at 250k total optimizer
updates or 72 active hours, whichever comes first. The GS predictor is capped at
100k steps or 12 hours, with effective batch 16. These caps total 156 active
hours, leaving 12 hours within a week for setup and outputs under continuous
operation. Interruptions/downtime and an unusually long batch can extend calendar
time. This is a shorter exploratory baseline, not paper-level convergence.
The original 1M-step settings remain in `configs/study_pitchflower_librispeech.yaml`.
Weekly outputs are separate in `runs/pitchflower_week`. Both codecs start fresh
because corrected pitch keep-mask semantics differ from the earlier RVQ run.
The old checkpoint is preserved for provenance; it does not seed this comparison.
On this machine that output directory is a junction to NVMe storage, keeping
checkpoint/log writes off the dataset HDD; local `storage.json` records its target.

RVQ uses 8 x 512 entries, dimension 256. GS uses the paper's C=32, shared widths,
fixed uniform centers, 100 inner Adam steps at LR 0.4 and STE outer gradients.
Its density 80/225 becomes 25 Gaussians over 71 latent frames here; this is a
PitchFlower adaptation, not a reproduction of GS-Codec's SEANet/GAN backbone.
Commitment weights are respectively 0.25 and 20. CFG dropout is implemented
as 10% per condition, respecting the reference pitch keep-mask and adding content
dropout. Neither RVQ dropout nor training-time GS quantization is used.
The GS predictor uses the released implementation (four dilated blocks, positional
encodings and one attention decoder); scalar PTQ comes
after training. No measured bitrate or quality results are claimed yet.

Terminal bars show preprocessing and optimizer-step/loss progress. Outputs are in
`runs/pitchflower_hardware/{rvq,gs,predictor}/` for the launcher's default profile
(`runs/pitchflower_week` for the baseline): checkpoints, TensorBoard logs,
progress JSON and final Vocos waveform examples at -6/0/+6 semitones. The root
`progress.json` mirrors the current phase. Rerun the same command after Ctrl+C
to resume; model/optimizer state is restored, but data-order/crop replay is not
guaranteed identical. Cached utterances are reused. `-Phase prepare|rvq|gs|predictor|calibrate`
selects one phase. This launcher does not stop or resize the ongoing L0 search.
Preprocessing may require substantial
time and roughly 250 GB before compression. Paper WER/SIM/UTMOS benchmarks remain
a later evaluation, not metrics computed by this training recipe.
Each stage's `time_budget.json` records cumulative active seconds across resumes.
`complete.json` reports actual updates and whether the step cap or time cap ended
training; compare actual update counts when a stage exhausts its time allowance.
Budget accounting is flushed every 30 seconds and at normal stop/interruption.

Execution uses SDPA attention with unchanged checkpoint keys, CUDA foreach Adam,
cuDNN autotuning and above-normal Windows training-process priority. RVQ still
updates its codebooks in groups of four, preserving the original EMA cadence.
The mapped-cache RVQ benchmark measured 0.35 s/optimizer step and 4.63 GiB
allocated GPU memory on a warm 832-utterance subset (24 measured steps, real
loader, random corpus sampling, Lightning autocast settings). This is a warm-cache
throughput check, not a full-corpus speed or completion-time guarantee.
Outer optimizers disable fused mode because the installed Lightning AMP plugin
rejects it with gradient clipping. Disposable Lightning tests verify BF16,
norm clipping and checkpoint resume for RVQ, GS and predictor training.
Exact measurements and conditions are in
the runtime benchmark record.
`tools/benchmark_pitchflower.py --priority --loader --apply` benchmarks disposable
models and explicitly migrates the stopped campaign's execution settings/source
hashes, retaining history in `execution_history/`. It never starts production
training. Dataset/objective changes and effective-batch changes are rejected;
ordinary resume continues to reject unexplained settings/source changes.

The processed corpus is on the HDD. Compressed NPZ inputs require decompressing
whole utterances to obtain 1.504-second crops; mapped NPY copies interleave both
waveforms and read just the crop's pages. Tests verify identical arrays, crops,
normalization and F0. Original NPZ files remain intact. The normal preparation
phase builds these copies once, resumes partial conversion, and skips completed
files. Allow roughly 239 GiB extra disk space and about two hours at the sampled
conversion rate. Training can read either format, so `-Phase rvq` can bypass
conversion if needed; use the normal command to prepare the faster layout first.

## Run and extend

```powershell
.\tools\bootstrap.ps1
.\.venv\Scripts\Activate.ps1
python main.py experiment --config configs/example.yaml
python main.py study --config configs/study_l1_synthetic.yaml
python main.py study --config configs/study_l1_librispeech.yaml
python main.py evaluate --run runs/YOUR_RUN --split clean --metric-workers 4
```

`experiment` prepares/audits data, trains, evaluates validation and reports;
`study` runs numerical experiments without training. No downloads occur without
`--download`. Use `python main.py COMMAND --help` for options. Saved checkpoints
supply the model for `evaluate`; `--data-config` selects external evaluation data.
For Python-defined experiments and ablation lists, edit `EXPERIMENTS` in
`main.py`; each named case can construct settings directly and call an
approach-level training or study function. If the list is empty, `main.py` uses
the existing command-line workflows.

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

Research approaches live in `approaches/`, grouped by method: `pitchflower/`,
`representations/`, `sparse_coding/`, `sparse_search/` and `interpretation/`.
Shared cross-approach helpers live in `util/`; keep domain-specific operations
with their approach until the same behavior is genuinely reused. `main.py` is
the project entry point, and `run.py` remains a compatibility entry point for
the existing commands. PitchFlower codec and predictor training entry points
live under `approaches/pitchflower/training/`.

Training outputs and model checkpoints belong under unique run-specific
directories in `runs/`, never beside source modules. The existing trainers
already create checkpoint folders per run; when defining several cases, give
each one a distinct run name or `run_dir`. Keep each case's settings, provenance
and metrics beside its checkpoints so ablations do not overwrite one another.
Generated run artifacts stay outside Git. Add a
numerical study beside the approach it evaluates; reuse shared metrics/data
helpers without adding a framework for one experiment. **Comment new code with
useful docstrings and explanations of mathematics, units and non-obvious
choices.**

Maintain this README as the only explanatory Markdown guide. Generated run reports
and numerical result artifacts stay outside Git. Preserve LaTeX source files and
keep paper titles/authors in
[bibliography.md](bibliography.md); that ledger includes screened and partially
reviewed work, not a claim that every proof was audited. Git history retains the
superseded Markdown discussions.
