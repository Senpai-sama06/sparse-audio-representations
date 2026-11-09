# PitchFlower

This folder owns the PitchFlower implementation, method documentation, report
builder, and published evaluation artifacts. Commands below run from the
repository root.

## Contents

| Location | Purpose |
| --- | --- |
| [model/](model/) | PitchFlower backbone, RVQ and Gaussian bottlenecks, Gaussian predictor |
| [training/](training/) | Codec and predictor training entry points and campaign orchestration |
| [data.py](data.py) | WORLD preprocessing, cached data, and crop loading |
| [docs/training.md](docs/training.md) | Training recipe, reference implementations, settings, and runtime limits |
| [reporting.py](reporting.py) | Inference evaluation and report refresh from saved assets |
| [reports/](reports/) | Published HTML reports, metric records, and verification records |

## Published RVQ evaluation

- [Standalone listening and results report](reports/pitchflower_best_testcases.html)
- [Full-precision case metrics and bitrate accounting](reports/pitchflower_best_metrics.json)
- [Verification and standalone-file checks](reports/pitchflower_best_verification.json)

The report evaluates the RVQ checkpoint at step 91,500 on 40 dev-clean crops,
one per selected speaker. Each crop is 1.504 seconds at 24 kHz. These utterances
were excluded from checkpoint selection and calibration, and were not used for
training. This is a short-crop evaluation, not full-utterance or test-clean
performance. The flow solver uses 10 steps and unchanged source pitch. Both
reconstruction paths use the pretrained `charactr/vocos-mel-24khz` Vocos vocoder.
The report does not evaluate the GS bottleneck or predictor.

| Comparison | After: PitchFlower and vocoder | Vocoder alone |
| --- | ---: | ---: |
| Mean PESQ-WB | 2.1629 | 3.4409 |
| Mean STOI | 0.9049 | 0.9716 |
| Estimated decoder-input payload (kbit/s) | 6.420 | 302.128 |
| Measured output PCM (kbit/s) | 384.000 | 384.000 |
| Measured output WAV, including header (kbit/s) | 384.234 | 384.234 |

After uses RVQ indices plus source pitch. For each crop, 71 latent frames,
8 codebooks, and 9 bits per index give 5,112 content bits. The 142 float32 pitch
values add 4,544 bits. Together this is 9,656 bits per 1.504 seconds, or
6.420 kbit/s. Pitch accounts for 47.06% of this estimate.

The vocoder control receives the full source mel spectrogram. Storing its
142 frames, 100 bands, and 32 bits per value requires 454,400 bits per crop,
equivalent to 302.128 kbit/s. Thus After uses an estimated 97.875% fewer
decoder-input bits, a 47.06:1 ratio, with lower reconstruction scores.
Higher control scores alone do not establish the better codec at a fixed rate.

These are fixed-width representation estimates. The raw float32 mel comparison
is not a tested compressed mel codec. No complete serialized codec bitstream
has been measured. Headers, packetization, byte padding, gain and other metadata
are excluded; model weights and codebooks are assumed shared at the decoder.
The decoded WAVs use the same 16-bit PCM format, so their playback bitrates are
equal. Each output WAV is 72,236 bytes for 1.504 seconds. PESQ and STOI are
computed against the source using the embedded PCM samples resampled to 16 kHz.

## Sharing and verification

Share only `reports/pitchflower_best_testcases.html` (about 21.9 MB).
It includes all 120 WAV clips, 120 spectrograms, case scores, rate comparisons,
formulas, assumptions, test scope, and checkpoint identity. Recipients can save
the file and open it in a browser without internet, Python, the repository,
models, datasets, or companion JSON files.

The verification record includes checks of all WAV headers and sample counts,
PNG checksums and decompressed pixel streams, absence of external dependencies,
and a copy placed alone in another folder. All original audio, images, and
quality scores were preserved when adding the bitrate comparison.

## Refreshing the saved report

This uses saved scores and embedded assets, with Python's standard library.
It does not run inference or require model dependencies:

```powershell
python -m approaches.pitchflower.reporting --refresh-existing approaches/pitchflower/reports/pitchflower_best_testcases.html --metrics approaches/pitchflower/reports/pitchflower_best_metrics.json --verification approaches/pitchflower/reports/pitchflower_best_verification.json
```

The refresh verifies case identities, embedded asset preservation, and agreement
between case scores and summary scores. The old command
`python tools/build_pitchflower_report.py` remains a compatibility entry point
to the same implementation.

## Running inference or training

`python -m approaches.pitchflower.reporting` runs the original fixed evaluation
recipe and writes the HTML and metrics into this folder's `reports/` directory.
It requires the original checkpoint, cached data, manifests, configuration,
CUDA, and inference dependencies. These local assets are not present in this
checkout. Refreshing or viewing the saved report does not require them.
The fixed recipe overwrites the published report names; preserve a separate
report version before using it for a new checkpoint or test group.

See [the training recipe](docs/training.md) for training and resume commands.
Repository launch/setup/benchmark helpers remain in `tools/`; their implementation
paths are listed in that recipe. Large checkpoints, raw logs, datasets, and
intermediate run artifacts remain in the configured `runs/` and data locations.
Published method reports and their supporting records belong here.
