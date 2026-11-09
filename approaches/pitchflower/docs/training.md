# PitchFlower training recipe

Run the commands below from the repository root. This records the original
training setup. Configuration files, corpus manifests, cached data, reference
checkouts, and checkpoints are local run assets and are not included with the
shareable report. The saved report can be viewed and refreshed without them.

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
after training. The published RVQ crop evaluation is documented in
[the method README](../README.md). It does not establish GS or predictor quality.
A measured complete codec bitrate is not available.

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
