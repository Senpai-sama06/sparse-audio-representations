# Implementation specification

The chosen convolutional sparse-autoencoder baseline is now implemented. See
the [model, training, and run guide](sparse_autoencoder.md) for the code actually
used. The architecture comparisons and earlier sketches below are research
history, rather than the implemented specification.

## Current working goal

Take an audio sample and automatically find the fewest possible values needed
to reconstruct it at the required accuracy, without being told or needing to
identify, name, or explain the representation it discovers.

This section is a **model-design discussion**, not an implemented model or an
agreed detailed architecture. The user has now chosen to begin with a sparse
autoencoder baseline; the [proposed run protocol](experiments.md) specifies the
dataset, convolutional baseline, training stages, and measurements for discussion
before that implementation. The user requested architecture analysis before choosing
the sparsity mechanism, loss, or training routine. The earlier budgeted-autoencoder
sketch below is deferred. The existing implementation is documented further below.
The [adaptive-transform review](adaptive_transforms.md) supplies the mathematical
limits and literature background.

## Architecture decisions come first

The inputs and outputs do not uniquely specify a network. We should justify its
representation family, temporal assumptions, and reconstruction mechanism before
deciding how to select coefficients. Top-k masking and a fixed latent dimension
are not accepted starting requirements.

### Decision 1: how are analysis and reconstruction related?

| Candidate family | Architectural assumption | Scientific advantage | Limitation for our goal |
|---|---|---|---|
| Independently learned encoder and decoder | The two mappings may have different structures and need not be inverses globally | Permits direct low-dimensional and deliberately lossy representations | Reconstruction failure can reflect encoder, decoder, or information already discarded; these must be diagnosed |
| Learned invertible network | A composition of bijective blocks; reconstruction uses the inverse | Gives a learned, potentially nonlinear change of coordinates with full-code reconstruction by construction in exact arithmetic | Retains input dimension before compression; constrains the representation family and may be unnecessarily restrictive for a lossy goal |
| Learned perfect-reconstruction filter bank | Local filters and a reconstruction operator satisfy reconstruction conditions | Makes time locality, parameter reuse, and boundaries explicit; useful structured comparison | Assumes a filter-bank organization and restricts which domains can be discovered |

None requires the learned coordinates to have names or human-interpretable roles.
None by itself guarantees sparse coordinates. Invertibility is not orthogonality,
statistical independence, or a claim of optimality. Numerical conditioning and
finite-precision reconstruction still need verification.

The first research distinction is whether we want to begin with an information-
preserving coordinate change and study its sparsifiability, or allow the
architecture itself to discard information immediately. Both can serve the dream,
but they answer different initial questions.

### Decision 2: how does the model process time?

This decision is separate: a CNN can implement an ordinary autoencoder or the
internal functions of an invertible network. A Transformer is also a component
choice rather than an alternative to the analysis/reconstruction relationship.

| Component choice | Assumption it introduces | Reason to consider it | Concern to resolve |
|---|---|---|---|
| Fully connected layers | Fixed input extent and position-specific interactions | Direct whole-example interaction; a useful short-signal comparison with few temporal assumptions | Dense layers scale poorly with waveform length and do not automatically reuse a pattern after translation |
| Temporal convolutions | Local operations shared over time | Parameter reuse for recurring waveform structure and support for varying lengths | The receptive field must cover the structures of interest; downsampling can discard information unless designed otherwise |
| Attention | Interactions between positions are input-dependent | Can represent long-range relationships without restricting every interaction to neighboring samples | Context, positional information, tokenization, and computation need justification; tokenization can itself impose an unwanted bottleneck |

In particular, globality is a property to demonstrate from the actual architecture,
not infer from calling something deep or using a large number of channels.
Receptive field, temporal resolution, and reconstruction boundaries are relevant
design quantities. No layer count or kernel size is selected yet.

### Why examine invertible coupling networks closely?

One candidate worth investigating is a composition of learned mixing and
invertible coupling blocks. A coupling block divides its current state into two
parts; one part determines an invertible modification of the other. Alternating
which part is modified permits interaction between them. Because the conditioning
part is available during inversion, the behavior can depend on the input without
requiring an unreported input-specific transform matrix. This is an architectural
way to make adaptation reconstructible, not a proof of sparse representation.

Real NVP supplies a concrete construction; its standard probability-modeling
objective is not automatically our sparsity objective. The functions inside the
blocks can use different temporal architectures. Learned mixing must also be
justified: permutations, local channel mixing, and whole-example mixing have
different expressive and computational consequences.

This family is provisionally attractive because it separates full representation
from later information removal. The main counterargument is that the best
approximate representation might be better served by a freely learned decoder.
That is why an unconstrained encoder/decoder remains in the comparison. We have
not decided that invertibility is a requirement of the dream or selected coupling
layers as the implementation.

An audio flow paper does not automatically provide an end-to-end audio
representation. For example, WaveGlow reconstructs/generates conditioned on a
mel-spectrogram. Such conditioning contains waveform information and could not
be a free decoder input in our experiment. Internal conditioning on recoverable
model state is different from external conditioning that must be encoded.

### Decision order

1. Compare the independently learned and invertible analysis/reconstruction
   families; decide what the first experiment must distinguish.
2. Justify temporal context and parameter sharing, then choose the components
   and the scale at which the model operates.
3. Specify latent organization and any internal conditioning or side information.
4. Derive the sparsity mechanism and reconstruction objectives for that architecture.
5. Choose layer sizes, numerical controls, training schedule, and datasets.

The selection mechanism should follow these decisions. The earlier proposal below
does not override this order.

### Architecture literature added in this discussion

Three more papers were inspected and appended to [bibliography.md](../bibliography.md),
bringing it to 46 records:

- **44. Dinh, Sohl-Dickstein, and Bengio, Density estimation using Real NVP**
  ([ICLR 2017 paper](https://arxiv.org/abs/1605.08803)): coupling construction and
  inverse properties reviewed. It motivates a learnable reversible representation;
  its density-estimation results do not establish minimum-support audio coding.
- **45. Prenger, Valle, and Catanzaro, WaveGlow: A Flow-based Generative Network
  for Speech Synthesis** ([paper](https://arxiv.org/abs/1811.00002)): flow blocks,
  temporal conditioners, and external mel conditioning reviewed. An audio
  architectural precedent, not a sparse codec we can adopt without accounting
  for its conditioning information.
- **46. Pfister and Bresler, Learning Filter Bank Sparsifying Transforms**
  ([author preprint](https://arxiv.org/abs/1803.01980)): transform/filter-bank
  correspondence and reconstruction formulation reviewed; experiments are in
  imaging. It motivates the locality/reconstruction comparison, not a theorem
  that this organization is the sparsest audio representation.

## Formulation: representation and search

Let a complete input example be a waveform `x` with `N` samples. Let `D_theta`
be a learned decoder shared across recordings, with parameters fixed during
evaluation. For that decoder, the ideal sparse inference problem is

```text
k*_theta(x, epsilon) = min over z  ||z||_0
                      subject to d(x, D_theta(z)) <= epsilon.
```

The decoder defines a potentially nonlinear, unnamed reconstruction space.
It need not synthesize audio by adding dictionary atoms: values can interact
nonlinearly.

For each k, define `R_k(theta) = {D_theta(z) : ||z||_0 <= k}`: the waveforms
reachable using at most k active values. The scientific purpose of learning the
decoder is to arrange these reachable sets so unseen audio lies close to them
even for small k. Learning the encoder is the separate computational task of
finding suitable values and positions for each input. Neither role requires
recognizing what the learned domain resembles.

A practical encoder proposes codes in that space:

```text
x -> learned encoder -> candidate values -> retain at most k values -> decoder -> x_hat
```

Encoder and decoder learn jointly from reconstruction, without pitch, speaker,
phoneme, Fourier-coordinate, or other feature labels. A nonlinear mapping can
react differently to different inputs without constructing an explicit transform
matrix per sample. Explicit sample-specific decoder adaptation remains a later
hypothesis; any such parameters belong in the representation accounting.

This is a candidate family, not a claim that its learned space is the best
possible space. The dream's global minimum remains the target; the first
experiment must state what it can and cannot establish.

## Deferred budgeted-autoencoder sketch

Use one ungrouped latent vector for the entire input example. The three functions
below need not be three separate neural networks.

| Part | Responsibility | First proposal |
|---|---|---|
| Encoder `E_phi` | Propose values from the complete waveform and requested budget | A small neural network with signed final outputs |
| Selection `H_k` | Limit the total number of retained values | Keep the k largest-magnitude values globally; zero the rest |
| Decoder `D_theta` | Reconstruct from the sparse code | A learned nonlinear network with no original-waveform input |

The forward pass is `z_k = H_k(E_phi(x, k))`, then `x_hat_k = D_theta(z_k)`.
Conditioning the encoder on k permits the values and selected positions to change
with the budget; it does not force every code to be a prefix of one longer code.
The budget is an explicit control, not additional waveform information. Compare
against a budget-independent encoder to test whether this flexibility helps.
Selection is a provisional inference rule, not the definition of a correct
representation. Large magnitude need not mean high reconstructive importance
with a nonlinear decoder. Top-k selection does not certify the best support.

For short fixed-length examples, fully connected networks are a useful first
candidate: every proposed value can depend on the complete waveform, without a
prescribed filter bank or time-frequency tiling. Compare a linear encoder/decoder
control with a small nonlinear encoder/decoder at matched input and latent
dimensions. This asks whether nonlinearity improves reconstruction economy.
Neither candidate requires orthogonality or tied encoder/decoder weights.

Initially use maximum latent size `M = N` and vary retained budget `k`. This avoids
making a narrow dense bottleneck the only possible reason for missing detail.
The implemented convolutional baseline does not fully achieve this intention:
its final slot count is `M = N`, but an earlier 128-channel layer at stride 256
contains only `N / 2` values for the two-second crops. This was identified during
the 28 October 2026 held-out evaluation. Record that hidden compression when
interpreting the baseline; final latent width alone is not a sufficient check.
It does not guarantee trained dense reconstruction: full-budget performance is
an explicit control. Layer counts, hidden widths, clip length, and initialization
remain implementation choices to settle after this discussion. A short fixed
clip is one complete testcase; its results would not establish whole-utterance
or variable-duration performance.

An independent learned support-scoring head could improve on magnitude selection
later. It requires an explicit gradient estimator or auxiliary objective: hard
selection alone will not train independent scores through its discrete indices.

## Training objective

For the first experiment, use waveform normalized squared error on non-silent
examples as the primary distortion:

```text
d(x, x_hat) = ||x - x_hat||_2^2 / ||x||_2^2

L(phi, theta) = average over training x and sampled k of
               d(x, D_theta(H_k(E_phi(x, k)))).
```

Train at a declared range of budgets, including generous budgets, rather than
only train a dense model and prune it afterwards. Log budget sampling and its
weights. Average training quality does not enforce the target for every input;
each evaluated sample must pass its own reconstruction check. Use a separate
absolute-error criterion for silence. Zero-budget reconstruction reveals what
the shared decoder produces without a sample-specific code.

The explicit budget already limits support, so the first candidate does not
need an l1 penalty to encourage small values. Actual zeros and small magnitudes
must remain distinct. Spectral and perceptual metrics remain useful observations
and can define later distortion-specific experiments; they are not silently
substituted for waveform fidelity.

With hard magnitude selection, gradients pass through retained values while
support membership changes discretely. This is not differentiable global support
optimization. Unselected coordinates may receive little learning signal.
Dense/generous-budget examples and activity diagnostics are controls, not a
guarantee against dead coordinates or optimization failure.

## Inference

Given a waveform and requested reconstruction tolerance:

1. Encode and reconstruct candidates for each declared budget.
2. Measure each candidate against that waveform.
3. Return the smallest actual nonzero count among successful candidates, along
   with retained values, coordinate identifiers, and reconstruction metadata.
4. If none meets the target, report failure within the searched budgets.

For a nonlinear decoder, pruning a predicted code need not produce monotone
reconstruction error. Inspect the declared budgets directly rather than assume
binary search is valid. Untested budgets remain untested. The mathematical
optimum over all codes with at most k nonzeros is monotone because those feasible
sets are nested; practical predictions need not attain it.

Later, refine values for each particular input while freezing the shared decoder,
and test alternative supports. These measure inference quality separately from
representation quality. They do not certify a global optimum. Optimizing an
input-specific code is distinct from fitting decoder weights on evaluation data.

## Representation accounting

The inverse receives the shared decoder and the explicit sparse representation.
Count active values over the complete input example. Also record coordinate IDs,
original length, numerical precision, and any gain, offset, timing, or adaptation
parameters needed to reconstruct. Sample-specific normalization values are not
free information if the decoder needs them.

Initially use an ordinary declared floating-point format. Scalar support count
is not encoded bit length; report index and metadata costs separately. Report
shared-model capacity as well. A result depending on increased coefficient
precision or unreported adaptation has not demonstrated the intended economy.

## Experimental questions and controls

| Question | Evidence needed |
|---|---|
| Can the dense candidate reconstruct? | Full-budget performance before blaming sparsity for missing detail |
| Does joint sparse training help? | Same architecture/data with varied-budget training versus dense training followed by pruning |
| Does nonlinearity improve economy? | Linear/nonlinear support-error frontiers, with parameter counts reported |
| Does use adapt to each input? | Per-example curves and minimum successful counts at fixed targets |
| Is inference missing better codes? | Fixed-decoder refinement, alternative supports, and small exact controls where available |
| Does economy generalize? | Unseen synthetic draws/families, then speaker-disjoint speech; no decoder fitting on evaluation samples |

Use controlled periodic, transient, mixed, and unstructured signals. Generator
controls are not latent targets. Random/noise-like signals are negative controls,
not examples assumed to admit extremely small accurate descriptions. Preserve
source/speaker separation and report per-example distributions and failures.

The uncertainties are distinct: a better code might exist in the current decoder;
a better decoder might need fewer values; and a lower bound is needed to certify
that nothing smaller exists. This first design addresses the first two empirically.
No training loss or successful pruning curve alone proves the strongest claim.

## Existing-code audit and literature

The existing `GroupedAutoencoder` is not this proposed candidate. Its budget is
applied independently at each latent time step. In an untrained shape/support
audit with `latent_width=8`, four encoder stages, `groups=1`, and `code_budget=1`,
inputs of 512/1024/2048 samples retained 2/4/8 values respectively. Output lengths
were correct. This checks counting and shape only, not reconstruction quality.
No model code was changed or model trained in this design pass.

Two papers were method-reviewed and appended to [bibliography.md](../bibliography.md),
bringing it to 43 records:

- **42. Makhzani and Frey, k-Sparse Autoencoders**
  ([2014 revision](https://arxiv.org/abs/1312.5663v2)): explicit top-k activation
  selection and reconstruction training are established precedents. Their tied
  linear model and classification experiments do not establish this proposal's
  nonlinear audio performance or minimum-support optimality.
- **43. Rajamanoharan et al., Improving Dictionary Learning with Gated Sparse
  Autoencoders** ([2024 v2](https://arxiv.org/abs/2404.16014v2)): separates selection
  from magnitude estimation and explains l1 shrinkage and auxiliary training for
  hard gates. The experiments reconstruct language-model activations with linear
  decoders; they do not directly establish raw-audio performance or universal
  sparsity. This motivates a later comparison rather than an architecture transplant.

The architecture comparison above now precedes this sketch. Its selector,
budget-conditioned encoder, and loss remain deferred rather than agreed design
decisions. No new model implementation should be inferred from this document.

## Existing baseline implementation

The remaining sections describe code already present, including earlier grouped
and dictionary experiments. They are not additional requirements of the dream
or an agreement to adopt those architectures.

For everyday use, run `python run.py experiment --config CONFIG.yaml` from the
project root. The workflow is the short function in
`lrac/experiment.py`; it calls the existing prepare, train, evaluate,
analyze, and report functions. It uses validation data for its report. See
[the contribution guide](../CONTRIBUTING.md) for naming and extension points.

## What is being tested

The representation should reconstruct unseen signals while exposing reusable
constituents. Reconstruction, representation activity, acoustic interpretation,
geometric separation, and functional separation are separate measurements.
Compact exact reconstruction of arbitrary unconstrained signals is not assumed.

## Representations

All models implement `encode(audio, lengths)` and `decode(representation)`. Audio
has shape `(batch, 1, samples)`. Each representation group has shape
`(batch, time, group_width)`. The representation stores original lengths, sampling
rate, timing/padding, and a version. The decoder has no waveform bypass.

| Family | Initial configuration | Purpose |
|---|---|---|
| DCT/random/PCA | 512 coordinates, 512-sample frames, hop 128 | Exact references and coordinate economy |
| Orthogonal dictionary | 512 learned orthonormal atoms | Learned geometry with sparse activations |
| Grouped dictionary | Four groups of 128, sparse-group inference | Complementary feature families |
| Overcomplete dictionary | 1,024 unit-norm atoms | Sparse alternatives without impossible column orthogonality |
| Convolutional dictionary | 64 kernels of length 512, hop 64 | Reusable time-local motifs |
| Autoencoder | Channels 32/64/128/128, four stride-4 stages, width 128 | Nonlinear factors and shared synthesis |

Frame analysis pads on both sides and uses square-root Hann windows; synthesis
normalizes overlap-add by the actual window weights. This preserves sample 0 and
the final sample, including inputs shorter than a frame. PCA includes its fitted
frame mean in decoding. Its moments are fitted on valid training frames only.

Dictionary training alternates sparse inference and a synthesis-dictionary update.
Orthogonal sparse-group inference uses the closed-form proximal operator.
Overcomplete and convolutional models use FISTA, a power estimate of the operator
norm, and checked backtracking. Inference defaults to 200 iterations and relative
objective tolerance 1e-5; selected comparisons should be repeated at 1,000 iterations.
Overcomplete atoms and convolutional kernels are renormalized after updates.

The nonlinear decoder mirrors its encoder using transposed convolutions. There
are no source-waveform skip connections. Inputs are right-padded to multiples of
256 and cropped back to their original length. Width sweeps use 32/64/128/256;
group count remains four.

## Losses and derivative estimation

Dictionary updates minimize valid-sample MSE. Thresholds and sparse-group proximal
terms control inferred coefficients. Adding an external coefficient penalty to
detached dictionary codes would not train the dictionary; use the proximal controls.

Neural reconstruction is MSE divided by fixed training-set signal power, plus
0.1 times a multi-resolution STFT loss. STFT windows are 256/512/1024 with quarter
hops. Each valid waveform is scored separately, excluding padded samples.

Optional nonlinear penalties are coefficient L1, group norms, and normalized
Jacobian block interactions:

    sum(i < j) ||Ji.T Jj||F^2 / (||Ji||F^2 ||Jj||F^2 + epsilon).

Training estimates traces with independent Gaussian Jacobian-vector probes. It
uses four probes per group on a 128 ms output window every fourth step, multiplying
the scheduled penalty by its interval. Exact small-model Jacobians verify the
estimator. Ratios of finite random estimates are not unbiased estimates of ratios.
Normalization handles uniform rescaling away from the epsilon regime, not all
nonlinear reparameterizations. Local separation does not establish global semantics.

Penalty warm-up occupies the first 10% of training, followed by a ramp of equal
length. Monitor reconstruction, coordinate variance, activity, and sensitivity
together. A dead decoder group cannot be counted as a useful separated factor.

## Training protocol

- Adam at 1e-3 for dictionaries; AdamW at 2e-4 and weight decay 1e-4 for neural models.
- Eight two-second crops per effective batch; accumulation supports smaller microbatches.
- Clip gradient norm at 1. Independent configurations may share the GPU when
  measured aggregate throughput improves; record contention and keep each
  experiment's batch/data/update budget unchanged for controlled comparisons.
- Screen for 2,000 updates on one hour, compare for 20,000 on ten hours, then confirm
  selected configurations on all training data. Repeat selected configurations at seeds 0/1/2.
- Screen regularizers at 0/1e-4/1e-3/1e-2 one axis at a time. Keep architecture and
  total latent width fixed within causal comparisons of regularizers.
- Checkpoints are selected by validation reconstruction. Assess reconstruction/
  structure tradeoffs as curves; do not claim meaningful structure from that selection alone.

Bounded validation selects round-robin across speakers, rather than taking a
prefix from speaker-sorted corpus files. The default 64 examples cover all 40
LibriSpeech validation speakers. Final tests are never used for checkpoint choice.

## Measurements

Measure ordinary waveform SNR, NRMSE, supplementary SI-SDR, multi-resolution STFT,
log-spectral distance, high-band error, and rapid-event error. The high band begins
at one-quarter of the sample rate. Rapid events use the upper decile of reference
sample-to-sample changes; this is a reproducible diagnostic, not a phonetic label.

The first near-accurate target is median ordinary SNR >=30 dB and fifth percentile
>=20 dB. Silence is scored separately. Gain changes must remain visible in ordinary
SNR even when SI-SDR is high. No alignment search or output clipping hides errors.

Record coefficient dimensions, scalar coordinates per second, and active
coordinates per second. These continuous-valued quantities are not bitrates or
proofs of compression. Fairness across model families also requires disclosing
parameter counts and computation, not merely matching latent dimensions.

Analysis includes seed-matched atoms (sign/permutation allowed), subspace angles,
post-training atom clusters, group interventions, acoustic proxy associations,
and known synthetic-component alignments. Large interventions may leave the
training distribution. Frozen cross-domain tests must be distinguished from
training a separate model on that domain.

## Data isolation

Official LibriSpeech train/dev/test roles remain separate. Source interval
manifests preserve nested subsets and all training speakers. Training-time
statistics and PCA never fit on validation/test data. The audit checks speaker,
source-recording, decoded-audio, and seed intersections. File identity is checked
at the full recording level so disjoint crops cannot hide shared-source leakage.

Cross-corpus speaker names are namespaced. Different corpora do not provide a
universal person-identity registry; metadata checks cannot establish that no human
ever participated in both corpora. VCTK remains an additional corpus/condition test.

## Current boundaries

The implementation supports offline full-utterance reconstruction and records its
context. It does not claim causal or bounded-lookahead streaming. Quantization,
entropy coding, supervised acoustic extraction, source-filter vocoders, and learned
dependency graphs are outside this initial representation-discovery implementation.
