# Sparse audio representations

## Shared learned-function pilot (experiment 1)

`approaches/parametric/learned.py` starts the learned-representation arm of
experiment 1. It compares an uncentered PCA/SVD basis against shared conditional
sine and SnakeBeta decoders. Both neural candidates have the same trainable
Fourier features and FiLM conditioning; the activation differs. A small encoder
maps a normalized component to a code. The shared decoder receives only the code
and coordinates; its weights are stored once. Every component's mean and RMS
gain are two additional transmitted scalars. This is not yet symbolic expression
discovery, a learned selector, learned VMD, quantization or RVQ.

The pilot has two separate data/model domains: direct 640-sample waveforms, and
fixed eight-mode VMD plus its encoded residual. VMD receives only each isolated
40 ms block. There is no overlap or stitching. Mode identities need not align
across clips because the same decoder processes every component.

Default frozen selection: 64 training speakers × 8 utterances × 8 nonoverlapping
blocks (4096 blocks), 8 validation speakers × 4 utterances × 8 blocks (256), and
8 test speakers with the same evaluation counts (256). Official train-clean-360,
dev-clean and test-clean are local inputs. Speaker disjointness is checked;
paths, utterance hashes and starts are in `data_manifest.json`. No quality-based
block selection or downloads occur. This is a small training pilot, not a full
corpus benchmark.

```powershell
./tools/train_parametric_representation.ps1 -CorpusRoot data/LibriSpeech
```

Code sizes 4/8/16/32/64 are trained jointly by cycling the active prefix. Each
neural model runs 12,000 Adam steps (batch 32), validating every 1000. The loss is
normalized component waveform MSE plus `1e-5` times mean squared code magnitude.
Checkpoints are selected by mean whole-block validation NMSE across code budgets.
Test blocks are evaluated only after checkpoint selection. Direct and VMD models
are separately trained. Per-block scalar counts are `components × (code_size+2)`;
shared decoder storage and encoder/decoder parameter counts are separate. These
are unquantized float32 codes and do not establish a compressed bitrate.

Training creates `last.ckpt` (optimizer and RNG state), `best.ckpt`, and independent
`decoder_best.pt` assets. Identical settings/output resume from checkpoints;
changed settings or source hashes require a new output. Small end-to-end smoke
tests cover data preparation, all model/domain combinations, validation/test,
asset+code decoding and HTML generation. Model tests also check masking, gain/
mean inversion, decoder identity hashes and exact scalar accounting.

The launcher returns a live dashboard (default http://127.0.0.1:8768/), a standalone
audio/plot report and an archive index (http://127.0.0.1:8769/index.html). Data
preparation is followed by actual training. Progress units combine prepared blocks,
optimizer steps and final model evaluations; ETA is provisional. Reports refresh
at validation checkpoints. Metrics name validation/test explicitly. Raw files,
logs, caches and checkpoint assets remain under the chosen output directory.

The previous two experiments have frozen progress/settings/metrics snapshots in
`runs/parametric_reports/records/`, report checksums in `experiments.json`, and
self-contained reports linked from `index.html`. Rebuild this archive with
`python tools/register_parametric_runs.py`. The multi-window pilot is stopped at
452/744; the isolated study completed 972/972. No raw results were deleted.

Decode a new block from shared weights and its transmitted code only:

```powershell
python -m approaches.parametric.learned_payload decoder_best.pt CODE.json decoded.wav
```

## Isolated 40 ms representation study

The multi-window pilot was stopped after its implementation audit. Its completed
jobs remain in the run directory. The frozen standalone listening report is
`runs/parametric_reports/preliminary_multi_window_sweep_stopped_after_audit.html`.
It labels the pilot's empty low-budget models, inadequate nonlinear convergence,
quantization/metadata costs and unvalidated PESQ outliers. It is not evidence of
the best achievable performance of each representation family.

`approaches/parametric/isolated.py` studies exactly 640 samples per testcase.
There is no overlap, stitching, quantization or RVQ. Default offsets (.96, 1.96,
2.96 seconds) are fixed before fitting; all three clips must exist in the source.
Backends receive only the selected block, without neighboring speech. Native
decomposition boundary extensions still apply. Each native residue is an ordinary
component. A direct-waveform baseline and sampled decomposition controls are kept.

```powershell
./tools/run_isolated_parametric.ps1 -SourceAudio data/testcase.wav -Output runs/parametric_isolated_40ms
```

The 972-job default library compares five decomposers plus direct fitting across
three clips and 53 candidates: degree 0–128 Chebyshev polynomials, 2–256 fixed
uniform spline knots, 1–640 fixed-prefix DCT coefficients, 1–16 sinusoidal atoms,
12 separate amplitude/phase polynomial-order pairs, and widths 1/2/4/8 for two
hidden-layer sine/SnakeBeta networks. Zero crossings/extrema are diagnostics,
never order-selection rules. All coefficients and all per-mode neural weights
are counted. Counts are per isolated block, not per second; no minimum-size
architecture is silently replaced by an empty model.

Linear spaces report SVD numerical rank and condition. Full-rank least-squares
results certify sampled MSE optimality only within that declared linear space;
rank truncation is labeled. Nonlinear fits use three starts and up to 1600
objective evaluations (analytic-Jacobian least squares) or Adam steps per start.
AM–FM initialization uses Hilbert amplitude/unwrapped phase and an FFT alternative;
amplitude is clipped nonnegative. Neural fits retain their best iterate and may
stop on a documented plateau after at least 400 steps. Histories, termination,
actual parameters and per-component errors remain in `jobs/*.json`. Local
convergence is not a global capacity certificate. The full DCT candidate is an
identity reconstruction control, not a compression result.

Whole-block and component complexity–distortion plots include raw waveform NMSE,
rectangular/Hann magnitude-spectrum error and separate edge/interior error.
PESQ/STOI are deliberately absent for these short fragments. Threshold tables use
exact dynamic programming over fitted candidates to minimize count under the
triangle-inequality bound on summed component errors. This bound is conservative;
the result is not a minimum over true summed-waveform distortion or all functions.
Thresholds are .1/.03/.01/.003/.001 NMSE. Missing/failed fits stay explicit.

`index.html` is the local report. `isolated_40ms_shareable.html` embeds all audio,
figures and linked tables/settings, and can be shared on its own. WAV bytes are
losslessly gzip-compressed and deduplicated; click **Load embedded audio** in a
current Chrome/Edge browser. The report refreshes every 12 jobs. Exact float64
JSON payloads can be independently decoded with the command below. Main metrics
use float64 parameter decoding; browser listening WAVs use float32, with their
separate `playback_nmse` recorded in the CSV:

```powershell
python -m approaches.parametric.isolated_payload MODEL.json decoded.wav
python tools/check_portable_report.py runs/parametric_isolated_40ms/isolated_40ms_shareable.html
```

The same progress dashboard is available at http://127.0.0.1:8765/ with a JSON
endpoint at `/progress.json`. Global ETA is provisional. Resume uses identical
arguments/output and requires matching sources/settings. For manual grids call
`isolated.run(...)` from `main.py`; the `library` argument accepts ordinary Python
candidate dictionaries. No corpus training or learned codebooks are involved.

## Parametric coding ablations

`approaches/parametric/ablation.py` compares EMD, VMD, EEMD, PyEMD's CEEMDAN
(including later improvements), and PySDKit SVMD, plus direct waveform fitting.
Each is paired with Chebyshev polynomials, natural cubic splines, sinusoidal
atoms, polynomial amplitude/phase models, or two-layer sine/SnakeBeta networks.
Network weights are fitted separately per mode and all are transmitted; there is
no corpus-trained decoder. Uniform allocation is the default; `--allocation energy`
is an explicitly different heuristic. Residues are parameterized within the total
budget by default (`--residual omit` is a separate ablation). Mode identities are
independent across frames. SVMD has explicit iteration/mode caps; cached sampled
modes never enter the decoder.

Install `pip install -e ".[parametric,speech-metrics]"`, then run:

```powershell
./tools/run_parametric_ablation.ps1 -Audio data/testcase.wav -Output runs/parametric_ablation
```

The hidden worker continues independently. The launcher returns its PID and the
local dashboard URL (default http://127.0.0.1:8765/). `/progress.json` exposes global
elapsed active time, completed jobs, current frame, heartbeat, failures and an ETA.
ETA is provisional: fitting families, frame lengths, and backends have different
costs. A stale heartbeat can indicate a stopped process; inspect `worker-error.log`
and `launch.json`. Rerun the same settings/output to resume completed jobs. Failed
jobs remain explicitly failed; choose a new output for corrected experiments.

Default sweep: four windows (40,20,80,160 ms), 50% overlap, five budgets
(500,1000,2000,4000,8000 model scalars/second), six backends including direct,
six fitting families, and float32/16/12/8-bit serialization. This is 744 jobs,
2,880 coded comparisons per testcase. Every overlap frame consumes budget.
Actual scalar use may fall below the target because model sizes are discrete.
Quantizer min/range scalars are counted in addition to model scalars. Actual byte
rates include JSON timing/model metadata and quantizer data; no entropy coding
is applied. Twelve-bit values are packed. These rates can exceed PCM for some
settings; the report presents measured costs without assuming compression.

Use ordinary Python arguments from `main.py`, or
`python -m approaches.parametric.ablation --help` for reduced grids, repeated
seeds, different fitting limits, and multiple local 16 kHz mono input files.
Inputs are complete utterances; the caller chooses development/test files.
There is no automatic corpus selection or hidden tuning on test results.

Outputs: resumable per-job records, decomposition caches, `.par` parameter
streams, decoded FLOAT WAVs, full-utterance PESQ/STOI/waveform/spectral metrics,
`metrics.csv`, `pareto.csv`, scatter plots, a fixed-setting heatmap and a listening
table in `index.html`. The Pareto frontier is per utterance and covers actual
bitrate/PESQ. One-testcase rankings are exploratory; multiple-speaker validation
and repeated seeds require explicit runs. Reports refresh every eight jobs.
`manifest.json` records settings, package versions and source hashes; incompatible
resumes are rejected. A fresh Python process verifies independent decoding.

Decode a saved file without the encoder's source audio or caches:

```powershell
python -m approaches.parametric.payload runs/parametric_ablation/FILE.par decoded.wav
python -m approaches.parametric.progress runs/parametric_ablation --port 8766
```

Meaningful component/stream checks:
`python -m pytest approaches/parametric/tests -q`.

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

## Parametric: bare EMD/VMD window experiment

`approaches/parametric/` compares reconstruction of a single local mono speech
testcase at several window lengths. It runs on CPU without dataset training or
downloads. Install the optional backends and speech metrics on the new machine:

```powershell
python -m pip install -e ".[parametric,speech-metrics]"
python main.py parametric --audio path/to/speech.wav --window-ms 20 40 80 160 --duration-seconds 2
```

The same experiment is callable with Python variables in `main.py`:

```python
from approaches.parametric import run

run("path/to/speech.wav", window_ms=(20, 40, 80, 160),
    overlap=0.5, vmd_modes=8, vmd_alpha=2000.0)
```

Use `python main.py parametric --help` for crop, overlap, EMD/VMD parameters and
output options. `python -m approaches.parametric` exposes the same arguments.
Default analysis is 16 kHz, with 50% overlap, all extracted EMD IMFs and eight
VMD modes. Requested windows are rounded to sample counts and actual lengths
and hops are recorded. Native amplitude is retained. Frames use reflected
context within the selected crop and normalized Hann synthesis; the original
clip length, including an odd final sample, is preserved. Modes are summed
within each frame before stitching; no mode matching across frames is assumed.

Each method/window produces two distinct conditions: **modes only**, and
**modes plus residual control**. The latter restores EMD's native residue or
VMD's raw input-minus-mode-sum residual. Near-perfect residual-control scores
are expected by construction and do not demonstrate compression. This is the
decomposition/windowing baseline: no fitted parameter vectors, quantization,
bitrate results or learned model are involved yet.

PESQ-WB is the speech-quality proxy and STOI measures intelligibility. Both
are evaluated on the full reconstructed testcase, alongside NMSE, SNR, SI-SDR
and spectral errors, without gain fitting or time alignment. A self-reference
condition supplies the metric ceiling for that clip. Missing/invalid speech
scores have explicit statuses; `--no-speech-metrics` disables them deliberately.
Use actual speech for interpreting those scores. The local HTML report includes
reference/reconstructed audio for listening comparisons.

Outputs are timestamped under `runs/parametric_*`: `index.html`, `metrics.csv`,
`summary.json`, `settings.json`, source/environment records, progress, reconstructed
FLOAT WAVs and per-frame statistics. Scoring reads back the saved WAVs. Optional
`--save-modes` adds per-frame sampled modes/residuals as NPZ files for later
investigation; those files are not a parametric bitstream. Explicit output
directories must be new. VMD initialization is deterministic; its fit-time
record is shared by the modes-only and residual-control conditions.

Backends are [PyEMD / EMD-signal](https://github.com/laszukdawid/PyEMD) and the
pinned [vmdpy 0.2 reference](https://github.com/vrcarva/vmdpy), whose maintained
continuation is in sktime. The adapter boundary allows replacing either backend.
See the [original VMD paper](https://ww3.math.ucla.edu/camreport/cam13-22.pdf)
for the bandwidth-constrained decomposition.

Run the frame-boundary, real-backend and artifact-roundtrip checks with:

```powershell
python -m pytest approaches/parametric/tests -q
```

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
`representations/`, `sparse_coding/`, `sparse_search/`, `parametric/` and `interpretation/`.
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
