# Proposed sparse autoencoder rate-distortion benchmark

Status: proposed benchmark protocol. Native-objective training scripts and pipeline
checks are implemented; see training.md. No corpus training has been started.
Quantization-aware training and operational bitstream evaluation remain planned.

## Purpose

Determine which sparse encoder produces the best reconstructed speech at a measured
bitrate. Report PESQ-WB, STOI, SI-SDR, and NMSE against actual bits per second.
Also test whether explicit sparsity improves compression over an otherwise matched
dense autoencoder. A sparsity-quality frontier is a separate diagnostic.

## Scope and baseline families

Sparse autoencoders are a family of methods. Backbone architecture, sparsity rule,
quantizer, and bitstream format are separate experimental choices. Recent SAE work
on language-model activations does not establish audio-codec performance. Applying
its bottlenecks to a temporal waveform network is an audio adaptation.

| ID | Method | Method-specific training or selection |
| --- | --- | --- |
| l1 | ReLU latent with L1 activity penalty | Sweep penalty weight; count realized activity |
| kl | Sigmoid latent with Bernoulli KL activity penalty | Sweep target mean activity and penalty weight |
| topk | Nonnegative Top-K latent | Sweep retained channels per latent time step |
| jumprelu | JumpReLU latent with learned thresholds | Sweep L0 penalty; use the published gradient estimator |
| gated | Gated SAE bottleneck | Separate activation selection from magnitude; retain published auxiliary objective |
| batchtopk | BatchTopK bottleneck | Allocate activity across training examples; use a frozen, batch-independent inference rule |
| dense | Matched ReLU autoencoder without sparsity penalty | Sweep quantization; train rate-specific controls if rate tuning is used |
| dct | Fixed transform control | Sweep coefficient retention and quantization; no neural training |

The first four rows are the core families previously discussed. Gated and BatchTopK
extend coverage of established SAE methods. Generic hard thresholding is a useful
ablation; it is not a synonym for the full JumpReLU method.

The KL target is a mean activation constraint. It does not guarantee per-example
nonzeros or exact zeros. Do not add an undisclosed Top-K operation to the KL or L1
baselines. Quantization may turn small activations into zeros; report both counts.

Use nonnegative codes in the main comparison. This preserves the usual sigmoid,
ReLU, and JumpReLU formulations. Signed magnitude Top-K, as used in the older local
model, is a separately labeled extension, not the same baseline definition.

## Shared audio architecture

Proposed initial input: mono speech at 16 kHz. Train on two-second crops. Evaluate
complete utterances with a declared, identical boundary and padding policy.

Use one temporal convolution/residual encoder and one waveform decoder for the
neural comparison. Start by auditing the local four-stage backbone with hop 256
samples and latent width 256. Freeze channel widths, receptive field, latent rate,
decoder nonlinearities, and boundary policy before the comparative run. Check dense
reconstruction capacity before accepting this backbone.

The sparse mechanism acts on the 256 channels at each latent time step. Each time
step is one example for Top-K and BatchTopK activity allocation. Exclude padded
steps from all activity and loss statistics. BatchTopK inference must not depend
on unrelated utterances in an evaluation batch; follow and document its published
inference calibration protocol using development data only.

Retain the shared decoder for comparable synthesis capacity. Keep the necessary
method-specific gates, biases, constraints, gradient estimators, and auxiliary
losses. Record parameter counts and compute; claim matched backbone, not exact
parameter equality. Specify any deviation from a cited method before training.

Do not cross every sparsity method with every network architecture initially.
After the main study, a decoder-capacity ablation can compare affine and nonlinear
synthesis for all core methods on the same frozen data and rate grid.

## Data separation

Audit actual dataset paths and previous experiment manifests before selecting data.
The repository README describes historical LibriSpeech use, but no data directory
is present at the workspace root. Availability and historical exposure remain to
be verified; do not silently download data or assume an old test set is fresh.

Create immutable manifests with speaker IDs, source checksums, recording IDs,
sample rates, crop selection, and exclusions. Separate these roles:

1. Training: model weights and training normalization statistics.
2. Checkpoint validation: choose checkpoints without consulting the test set.
3. Calibration: tune quantizer scales, inference thresholds, rate operating points,
   and any fitted coding distributions. This is a held-out development partition.
4. Final test: locked recordings from speakers absent from all development roles.

Use speaker-disjoint partitions for these roles where corpus size permits.
Previously inspected test-clean results are a historical replication set. A new
confirmatory test requires genuinely unused data; a new crop of an inspected
utterance does not make it fresh. If fresh data is unavailable, label the resulting
benchmark exploratory and state its exposure history.

Pilot training uses a fixed speaker-balanced ten-hour subset, subject to the audit.
Compare all pilot families on the same subset. Choose the final training corpus
size from measured pilot resource use before comparative final runs begin.
Do not promote only a favored family to more training data.

## Common codec and bitrate measurement

The decoder must reconstruct from a saved stream and shared decoder assets alone.
It cannot access original audio, cached unquantized codes, or per-source statistics
that were not transmitted.

Start with a common scalar quantizer and a versioned lossless packing format.
Proposed precision grid: 4, 6, 8, and 12 bits. Fix the quantization mapping and
zero reconstruction level. Fit fixed scales on development calibration data only;
if adaptive scales or gains are used, transmit and count them. Freeze calibration
before final test. Record saturation, code range, and the actual quantizer mapping.

Use a common support/value syntax for sparse neural methods. If the format permits
a choice among bitmaps, indices, and dense values, make that choice an explicit
signaled format option available to every neural method, including the dense
control. Include all option flags, headers, support, values, scales, lengths, and
padding. DCT uses the same applicable quantization and support-cost rules, with
its own declared transform framing metadata.

Report bits per second as total stream bits divided by original audio duration.
Report payload and overhead separately, plus bits per sample. Report decoder and
encoder storage separately under a shared-model deployment assumption. Any
recording-specific adaptation must be included in its rate.

Add entropy coding as a second, common codec condition after fixed packing works.
It must model both support and values, or explicitly state which part it codes.
Fit probabilities without test leakage. Compare all families under each coding
condition. Do not compare one family's entropy-coded rate to another's fixed rate
as evidence about the sparsity rule alone. Estimated entropy is a diagnostic;
saved byte counts are the reported operational rate.

## Training stages

### Stage A: integrity and dense capacity

Run small end-to-end checks for shape, padding, silence handling, decoder-only
reconstruction, and exact byte accounting. Confirm Top-K counts, KL statistics,
JumpReLU threshold gradients, gating gradients, and BatchTopK inference behavior.
Check that validation does not update weights or calibration state. A failing
dense capacity control triggers a shared-backbone revision before comparison.

### Stage B: native-objective pilot

Train the six sparse families and the dense control with one seed. The DCT control
requires no training. Use a common waveform reconstruction objective:

    D_train = waveform MSE + alpha * multiresolution STFT loss

Fix the normalization and alpha on development data. Include the necessary native
sparsity and auxiliary terms for each family. Prevent latent/decoder rescaling
from trivially reducing a sparsity penalty; specify compatible weight constraints
or normalization and audit latent and decoder norms.

For Top-K, propose k = 4, 8, 16, 32, and 64 out of 256 channels. Use equal search
budgets to find penalty/target settings that cover comparable realized activity
for other methods. Match achieved rates for codec comparisons, not penalty values
or support fractions alone. Broad coverage is more important than forcing an
infeasible exact activity match for every method.

Do not rank families from one run per method. Each family must cover multiple
operating points. Charge calibration and unsuccessful tuning trials to its search
budget. Record dead channels, activity distributions, convergence, and wall time.

Measure unquantized and decoded quantized quality separately. The unquantized arm
isolates representation error. The quantized arm is the compression result.

### Stage C: quantization-aware operational benchmark

Using pilot results, freeze four common rate targets within the observed overlap.
Do not promise 1, 3, or 6 kb/s until feasibility is measured. Pick precision and
method settings using calibration data only. Train or fine-tune with quantization
in the forward path and a documented gradient approximation. Apply the same
quantization-aware opportunity to every neural family, including the dense control.

For variable-rate entropy-coded optimization, use the shared objective:

    L = D_train + lambda_R * R_est + beta * S_method + A_method

R_est includes the modeled support and value cost and is normalized consistently
per original second. S_method and A_method are the native sparsity and auxiliary
terms when needed; Top-K has an explicit support constraint. For a fixed-length
format, do not invent a differentiable rate penalty for a constant bit budget;
optimize the quantized distortion under its declared budget instead.

Choose checkpoints using decoded validation distortion subject to a predeclared
rate tolerance, or a fixed D_train + lambda_R * measured-rate objective. Specify
the rule for each run in advance. Do not select checkpoints by adding unrelated
PESQ, STOI, and SI-SDR scores without a justified scaling rule.

Run three fixed seeds for each neural family at the four final operating points.
This is 84 model-seed-rate runs for seven neural families if trained separately.
Quantizer sweeps in Stage B are evaluations, not necessarily separate trainings.
Any method requiring multiple precision-specific fine-tunes adds runs and must
receive the same declared tuning allowance. Use pilot timings to set a feasible
common budget in audio-seconds seen; record updates, batch size, and GPU-hours.
Grant justified extensions by a predeclared convergence rule for all methods.

## Evaluation and comparison

Primary outputs are separate bitrate-versus-PESQ-WB, bitrate-versus-STOI, and
bitrate-versus-SI-SDR curves from decoded saved streams. Higher scores are better.
NMSE is a supporting amplitude-sensitive distortion measure. Do not combine these
metrics into one arbitrary winner score. Use listening examples to inspect failure
modes, without selecting only successful cases.

Evaluate every selected complete utterance. Undo only declared codec delay and
padding; no per-example search for a better shift or gain. SI-SDR retains its
mathematical scale invariance. PESQ/STOI use their required preprocessing and valid
speech lengths. Retain failed, silent, or invalid cases with explicit statuses
and validity counts instead of dropping them to improve averages.

Report utterance-mean quality and speaker-macro summaries, per-seed results, and
paired speaker-level bootstrap intervals. Give total bits divided by total seconds
as the aggregate rate; also show per-utterance rate distributions. Report seed
spread separately from speaker uncertainty.

Compare at measured common rates using development-selected operating points.
Any interpolation of test curve points is descriptive, within observed overlap,
and must be labeled. Never tune a test-time threshold to hit a favored test score.
Also report encode/decode real-time factor, parameter counts, model storage,
retained coefficient counts before and after quantization, and clipping rates.

## Outputs and implementation outline

Keep the requested approaches/sparse-ae directory name. Plan an explicit Python
launcher from that directory rather than assuming a standard dotted import using
the hyphenated name. Decide import/package integration during implementation.

Proposed shared components are method adapters, training, data manifests, codec,
evaluation, and reporting. A method adapter must expose its dense representation,
sparse representation, native loss terms, and diagnostics. Decoder assets must be
exportable without optimizer or training-data access. All adapter-specific inference
state must accompany those assets. No method may bypass the common bitstream path.

Store method documentation under docs/ and final reports under reports/ in this
folder. Keep large checkpoints, streams, and manifests under runs/sparse_ae/<run_id>/.
Save source revision/hashes, settings, environment, seeds, last/resumable and best
checkpoints, calibration records, stream bytes, decoded audio, and per-case metrics.
Follow documentation/REPORTS_README.md for the self-contained HTML results report.

## Existing evidence and limits

The README documents previous convolutional sparse and dense models. Their learned
codes were unquantized, so those scores are historical representation baselines.
The existing sparse_coding codec measures saved bytes and can inform the new codec,
but its format assumes a fixed dictionary and cannot be reused blindly for neural
latents. Existing models use whole-recording magnitude Top-K; the proposed local
nonnegative Top-K is a distinct declared experimental condition.

This benchmark measures achievable operating points for declared methods and
budgets. It does not prove a Shannon rate-distortion limit or a globally optimal
sparse representation. Relative gains on language-model activations do not predict
the winner on waveform coding.

## References

- Classical KL sparse autoencoder: https://nlp.stanford.edu/~socherr/sparseAutoencoder_2011new.pdf
- K-sparse autoencoders: https://arxiv.org/abs/1312.5663
- Modern Top-K training: https://arxiv.org/abs/2406.04093
- Gated SAE: https://arxiv.org/abs/2404.16014
- JumpReLU SAE: https://arxiv.org/abs/2407.14435
- BatchTopK SAE: https://arxiv.org/abs/2412.06410
- Learned rate-distortion coding: https://arxiv.org/abs/1611.01704
- STOI: https://sps.ewi.tudelft.nl/pubs/Taal2010.pdf
- SI-SDR: https://arxiv.org/abs/1811.02508

## Next action

Supply the local corpus path, prepare shared manifests, and freeze the pilot tuning
budget. Run the dense capacity pilot before the comparative corpus campaign.
The implemented native-objective scripts cover the initial training stage;
quantization and operational rate evaluation still require implementation.
