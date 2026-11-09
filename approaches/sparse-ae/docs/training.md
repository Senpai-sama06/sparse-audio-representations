# Sparse autoencoder training details

## Implemented stage

The scripts implement the native-objective training stage in the experiment plan.
No quantizer or entropy coder is inserted. No operational bitrate is inferred
from latent counts. All models reconstruct 16-kHz mono waveforms with the same
temporal backbone. The user launches corpus training explicitly; source data is
resolved from local paths/cache or obtained through KaggleHub when absent.

## Common backbone

Four convolution/residual stages downsample by four each, for a hop of 256 samples
(16 ms). Default channels are 32, 64, 128, 128. The final feature vector at each
time step has a non-affine LayerNorm. A learned projection produces 256 latent
channels. The decoder has a 1x1 input projection, residual blocks, and four
transposed-convolution upsampling stages. It outputs the waveform directly.

Right padding reaches a multiple of 256; reconstruction is cropped to the source
length. Invalid latent time steps are excluded from selection and penalties.
Training batches contain fixed-duration crops or shorter valid utterances.
Full-utterance validation uses batch size one. All models use nonnegative codes.

Projection rows and decoder input columns are normalized to unit L2 norm after
every update. Feature LayerNorm fixes the input scale to the latent projection.
These constraints remove obvious latent scale freedoms without fixing the entire
nonlinear decoder's gain. Log latent RMS and dead/inactive channels to audit scale
and capacity. This is a shared audio architecture adaptation, not an exact replica
of a published language-model SAE experiment. Default hyperparameters are starting
points, not independently validated optimal settings.

## Losses

The common reconstruction term is:

    R = mean waveform MSE / mse_scale^2 + spectral_weight * MR-STFT

Defaults are mse_scale=0.1 and spectral_weight=0.1. The fixed MSE scale is not
per-utterance gain normalization. MR-STFT uses FFT sizes 256, 512, 1024 with a Hann
window and quarter-window hop. It averages spectral convergence and absolute
log-magnitude errors. Padding is excluded by cropping each signal to its valid
length. The waveform objective averages per-utterance means.

Native penalties are sums over channels averaged over valid latent time steps:

| Method | Code | Additional terms |
| --- | --- | --- |
| dense | ReLU(projected features) | None |
| l1 | ReLU(projected features) | sparsity_weight * mean L1 sum |
| kl | sigmoid(projected features) | sparsity_weight * sum Bernoulli KL(target || batch mean activity) |
| topk | Keep k largest nonnegative channels per time step | Dead-latent residual auxiliary objective |
| jumprelu | Keep positive activations above learned channel thresholds | sparsity_weight * mean L0 count |
| gated | Binary gate times separately scaled nonnegative magnitudes | L1 gate proxy and detached-decoder gate reconstruction |
| batchtopk | Keep k times valid-time-step-count activations across the training batch | Dead-latent auxiliary objective; frozen threshold at inference |

L1 defaults to weight 0.0001. KL defaults to weight 0.001 and target 0.05.
JumpReLU defaults to L0 weight 0.001. Gated defaults to L1 weight 0.0001.
Dense and both Top-K methods have no activity penalty by default.

The KL mean is computed per training batch. Validation batch size one computes
each complete utterance's activity statistics. The target constrains average
activity, not a number of exact nonzeros. No extra threshold or Top-K is applied
to KL, L1, or dense codes.

JumpReLU uses positive learned thresholds parameterized by their logarithm.
The rectangular-kernel backward estimator for the activation has an exact input
gradient above threshold and a surrogate threshold gradient. Its L0 term has a
threshold gradient and no direct gradient to the preactivation. Initial threshold
and bandwidth both default to 0.1 in this normalized audio backbone. They are
tunable; these differ from the language-model experiment's numerical defaults.

Gated shares projection directions between gate and magnitude paths. The
magnitude path has a learned positive channel scale and a separate bias. The
binary gate has no straight-through gradient. A second reconstruction through
the ReLU gate proxy trains the gate path, with all decoder parameters detached.
This waveform MSE auxiliary term replaces the original linear-feature-space
auxiliary error. Its default weight is 1.0. No periodic feature resampling is
implemented; inactive-channel diagnostics make that limitation visible.

For Top-K/BatchTopK, a channel is eligible for the dead-feature auxiliary objective
after 100 training updates with no activation. Up to aux_k=32 eligible channels
per time step reconstruct the detached waveform residual. The auxiliary waveform
contribution is decode(aux_code) - decode(zero), since a nonlinear decoder has
a nonzero zero-code response. Weight defaults to 0.03; set --aux-weight 0 for a
declared no-auxiliary ablation. This is an adaptation of dead-feature residual
learning, not the exact original linear-decoder auxiliary objective.

BatchTopK uses valid latent time steps as its training examples. Before each
validation epoch, it fits one threshold using the distinct calibration partition:
average minimum positive retained activation across fixed centered calibration
batches. The mean is over batches. The model uses this threshold independently
for every utterance at validation/inference. The threshold and calibration-ready
flag are included in best weights and resume state. No validation or test score
is used to choose it. Native validation uses this inference path.

## Stopping and checkpoints

See the method README for CLI examples. Lower monitored validation loss is better.
The monitored loss is either total native objective or common reconstruction loss.
The best-weight tracker saves every observed minimum. A separate significant-
improvement anchor resets patience only after improvement exceeds min_delta.
This avoids losing genuinely best weights to an early-stopping tolerance.

Stop at max epochs or the patience limit, whichever happens first. Patience zero
disables early stopping. All stopping counters and the best snapshot survive
resumption. Full checkpoints retain optimizer, loader generator, Torch CPU/CUDA,
Python and NumPy RNG states. Epoch crops depend directly on seed, epoch, and
recording ID. Cross-device or cross-version bit-exact reproduction is not claimed.

Validation NMSE and zero-mean SI-SDR are diagnostics, not checkpoint-selection
scores. PESQ-WB and STOI are optional diagnostics on the same cases. Best model
selection uses the named validation loss; unrelated metric scales are not summed.
All validation cases and unavailable metric statuses are saved at the best epoch.

## Data and reproducibility

Preparation records full source SHA-256 hashes, path, byte count, timestamp,
speaker, split, length, and sample rate. Recording duplicates and speaker overlaps
between partitions are rejected. A supplied training-hour cap selects utterances
round-robin across shuffled speakers; it approximates balance by utterance count,
not exactly equal hours per speaker. An epoch uses one crop per selected utterance.

Manifest reuse validates settings and checksums. Training validates file sizes and
timestamps; --verify-audio-hashes also rehashes non-test audio. Evaluation rehashes
the evaluated audio. Resume requires unchanged training source and manifest hashes.
Test audio is not loaded by training. Optional test metadata is included in the
preparation audit. Corpus acquisition is cache-first and uses a shared setup lock.
See remote.md for the automatic KaggleHub fallback and offline mode.

For the comparative study, keep manifests, seed, crop schedule, backbone, training
budget, stopping rule, and reconstruction loss consistent. Give each family's
method-specific settings a declared tuning budget. Do not treat equal penalty
weights or equal active fractions as equal bitrates.

## Sources

- [Classical KL formulation](https://nlp.stanford.edu/~socherr/sparseAutoencoder_2011new.pdf)
- [K-sparse autoencoders](https://arxiv.org/abs/1312.5663)
- [Top-K and dead-feature residual training](https://arxiv.org/abs/2406.04093)
- [Gated architecture and auxiliary objective](https://arxiv.org/abs/2404.16014)
- [JumpReLU gradient estimators](https://arxiv.org/abs/2407.14435)
- [BatchTopK selection and inference threshold](https://arxiv.org/abs/2412.06410)
