# Proposed L1 sparse-coding rate-distortion experiment

Status: **proposal only**, 29 October 2026. No audio experiment or codec
implementation was run for this proposal. This records the first peer suggestion:
find a representation by L1 minimization and measure its reconstruction fidelity
and actual encoding bitrate. The next idea has not yet been incorporated.

## The question this experiment can answer

Does L1 sparse inference provide a useful **measured rate-distortion trade-off**
within a declared representation family?

This is a feasible, conventional optimization experiment. It does not require
neural training or a ground-truth sparse code. It also does not, by itself,
discover the globally sparsest domain or certify minimum nonzero count/bitrate.

Our current pipeline measures reconstruction and coefficient counts, including
coefficients per second. It does **not** yet provide a common serialized sparse
audio bitstream for those model comparisons. The earlier transform-description
bit accounting in the tiny exact search is not a complete audio codec.

## Recommended first representation

Use a fixed linear, unit-column-norm, overcomplete synthesis dictionary. A simple
starting proposal for a complete length-N waveform is

$$
\Phi=[I,\ Q],
$$

where $Q$ is an orthonormal inverse-DCT synthesis matrix. There are $2N$
candidate coefficient coordinates: waveform impulses and cosine atoms.

Reasons for this starting point:
- The identity component guarantees that every finite input is representable.
- Overcompleteness creates a genuine choice of decomposition; exact coding in
  one invertible basis alone has a unique coefficient vector.
- The atoms have unit norm, avoiding an L1 advantage from arbitrary atom scaling.
- The dictionary is shared and generated from N; no sample-specific learned
  matrix has to be transmitted.
- Products can be computed without materializing an $N\times2N$ matrix:

$$
\Phi(a,b)=a+Qb,\qquad \Phi^\top r=(r,Q^\top r).
$$

The DCT/IDCT operators permit efficient matrix-free optimization. This is a
recommended baseline family, not a claim that impulses/cosines are the right
final representation.

A secondary controlled arm could optimize coefficients in the **frozen S-AF
synthesis bank** and compare with its encoder/top-k codes. That arm answers a
different question about inference in the learned bank. Its affine offset must
be included explicitly, atom scaling handled consistently, and its known
rank bottleneck retained in the interpretation. Tight error constraints can be
infeasible there. It should not be our only baseline for the feasibility of L1
representations.

## Sparse optimization

For a nonzero original waveform $x$, propose the constrained problem

$$
\min_z\|z\|_1
\quad\text{subject to}\quad
\frac{\|x-\Phi z\|_2^2}{\|x\|_2^2}\leq\eta.
$$

For fixed linear $\Phi$, this is convex. The dictionary is not being learned
or adapted by this optimization. For a fixed affine synthesis bank, replace
$\Phi z$ by $b+\Phi z$.

A penalized path is another possible implementation:

$$
\min_z \frac12\|x-\Phi z\|_2^2+\lambda\|z\|_1.
$$

Sweeping $\lambda$ explores the residual/L1 trade-off. We must verify the actual
residual at each solution rather than assume a given $\lambda$ enforces a
particular $\eta$. Proximal algorithms are an appropriate implementation route.
Record the objective, convergence criterion, iteration limit and an optimality
residual or primal-dual gap where supported. A timeout is not a converged optimum.

L1 is a tractable surrogate, not generally identical to minimum support.
Furthermore, its continuous-valued objective does not minimize serialized bits.
The experiment is designed to measure whether it is useful despite those gaps.

No extra sparsity threshold should be silently selected after looking at results.
Any threshold/quantization operation belongs to the encoder configuration and
its reconstruction must be re-evaluated. Support-only least-squares refitting,
if later tested, is a separately labelled variant rather than the original
L1 solution.

## The actual encoding path

The measured object must be a self-contained reconstruction:

```text
original waveform -> L1 coefficients -> quantization -> serialized bytes
serialized bytes -> parse/dequantize -> shared synthesis -> reconstructed waveform
```

The decoder receives no original waveform, unquantized coefficients, per-record
scale, transform choice or normalization factor through an uncounted side input.

A simple initial codec proposal:
- Signed uniform scalar coefficient quantization, sweeping 4, 6, 8, 12 and 16 bits.
- A transmitted coefficient scale when it is selected per clip; quantization
  must use the same represented scale as the decoder.
- Drop coefficients whose transmitted quantized value is zero.
- Encode support using either a bitmap or sorted fixed-width positions,
  choosing the shorter under a declared mode flag.
- Include the original length, dictionary/codec identifier, quantizer settings,
  any scale/gain/offset information, and byte padding in the actual serialized size.
- Keep analytic dictionaries or a frozen decoder shared. Report shared model
  storage separately; report amortized model cost only under an explicit reuse
  assumption. Do not charge sample-specific adaptation as shared knowledge.

For a dictionary with M coordinates and K transmitted nonzeros, bitmap support
cost is M bits and a simple index-list payload costs
$K\lceil\log_2M\rceil$ bits, before mode/count/header/padding information.
These formulas are design checks, not substitutes for measuring the emitted file.

No per-clip waveform normalization, mean removal, resampling reversal or
post-decoding gain adjustment should restore information for free. Any
sample-specific inverse parameter must be available in the coded description.

The initial codec need not claim state-of-the-art entropy coding. A more advanced
support/value entropy coder can be studied later under the same accounting rules.

## Metric definitions

Let $N$ be the original number of samples, $f_s$ the sample rate,
$T=N/f_s$, $b$ the emitted byte string and
$\widehat x=\operatorname{Decode}(b)$. All primary quality scores use this
**post-quantization, post-serialization reconstruction**.

| Metric | Definition | Interpretation |
|---|---|---|
| NMSE | $\sum_n(x_n-\widehat x_n)^2/\sum_nx_n^2$ | Primary waveform fidelity; lower is better. |
| Ordinary SNR | $-10\log_{10}(\mathrm{NMSE})$ dB | Same waveform error expressed differently, not independent evidence. |
| Bitrate | $R=8\,\mathrm{len}(b)/T$ bits/s; divide by 1000 for kb/s | Actual description length per audio duration, including transmitted overhead. |
| Bits per input sample | $8\,\mathrm{len}(b)/N$ | Same size normalized by sample count. |
| Compression ratio to stated PCM reference | $R_{\rm PCM}/R$ | Reference precision/channels/sample rate must be stated. |
| Transmitted support | Count of nonzero quantized coefficients, plus K/T | Representation diagnostic, not itself bitrate. |
| PESQ-WB and STOI | Compare original speech with decoded speech using existing implementations | Speech-quality/intelligibility diagnostics; report validity counts. |
| Encoding and decoding time | Elapsed time separately; real-time factor = time/T | Includes optimization for encoding; report setup separately if cached. |

For exactly reconstructed non-silent inputs, NMSE is zero and SNR is mathematically
infinite; do not interpret a numerical floor as a measured finite error.
For silence or extremely low reference power, use a declared absolute MSE/RMS
criterion instead of dividing by near-zero energy. Keep reference/estimate
lengths and signal levels consistent; no alignment or gain fitting hides error.

Also reuse the existing SI-SDR, multiresolution STFT, log-spectral, transient and
high-band error diagnostics from [the metric implementation](../lrac/metrics.py).
SI-SDR does not replace ordinary waveform fidelity because its scale invariance
can conceal gain errors. Intermediate coefficient arrays are not speech and are
not scored with PESQ/STOI.

For 16-kHz mono 16-bit PCM, the uncompressed reference rate is **256 kb/s**.
That is a reference format calculation, not the measured rate of the cached
losslessly compressed FLAC files.

### Separate optimization loss from coding loss

Record both
$d_{\rm pre}=d(x,\Phi z)$ and
$d_{\rm final}=d(x,\operatorname{Decode}(b))$.
The continuous optimization's error constraint need not survive quantization.
Only the final decoded score decides whether an operational point meets a target.

Break the transmitted length down as

$$
L=L_{\rm header}+L_{\rm support}+L_{\rm values}
  +L_{\rm quantizer}+L_{\rm adaptation}+L_{\rm padding}.
$$

A term already represented in the header must not be counted twice.
Encoder-only optimization parameters need not be transmitted if reconstruction
does not use them. All actual transmitted data do count.

## Proposed small pilot

These are proposed settings to discuss before execution, not a frozen protocol:

1. Synthetic impulses, tones, mixtures and noise for solver and codec sanity checks.
2. Cached LibriSpeech dev-clean, not new test-clean measurements. Use balanced
   recordings across its 40 speakers, excluding the 200 checkpoint-selection
   utterances. Do not restrict the pilot to the voiced regions used in pitch tests.
3. For example, select three deterministic two-second crops per speaker before
   inspecting results: 120 clips / four minutes of audio.
4. Split 20 development speakers for encoder/coder calibration and 20 for the
   exploratory comparison. These are still development speakers previously used
   in other studies, not an untouched confirmation set.
5. Sweep reconstruction targets such as NMSE 0.1, 0.01 and 0.001
   (10, 20 and 30 dB ordinary SNR), together with coefficient quantization.
6. Compare L1 coding with ordinary DCT hard-threshold coding, using the same
   quantization/serialization rules and counting each method's actual dictionary
   size and metadata. Include PCM as a transparent reference.
7. Plot final reconstruction error against actual kb/s, plus PESQ/STOI versus
   kb/s. Compare rates at matched distortion and distortion at matched rate.
   Report paired speaker summaries and source-level failures.

To attribute an improvement specifically to the L1 inference algorithm,
a same-dictionary greedy-pursuit control can be added. If the intended meaning
of competitive becomes practical audio compression, add a conventional audio
codec baseline; beating DCT alone does not establish that claim.

Existing S-AF scores cannot be placed on these bitrate plots by assigning
assumed bits per coefficient. Its codes would need the same encode/decode
measurement and fresh quality scoring on the same pilot inputs.

## Feasibility conclusion

This is a possible and informative experiment with the existing data and
linear-transform tooling. Its immediate deliverable would be measured
rate-distortion curves and a reproducible, decodable sparse bitstream.

It tests the usefulness of L1 inference within a specified family. It does not
yet resolve the unknown-domain search, certify minimum L0/bitrate, or establish
that one representation is best for every recording. No neural training is
required for this first experiment.

Relevant sources reviewed:
- [Atomic Decomposition by Basis Pursuit — Chen, Donoho and Saunders](https://web.stanford.edu/group/SOL/papers/BasisPursuit-SIGEST.pdf):
  formulation/intro and selected approximation/denoising discussion reviewed;
  the available PDF is the 2001 SIAM Review republication of the 1998 article.
- [Proximal Algorithms — Parikh and Boyd](https://stanford.edu/~boyd/papers/pdf/prox_algs.pdf):
  proximal-method overview and selected Lasso/algorithm discussion reviewed.

Their titles and authors are recorded in [bibliography.md](../bibliography.md).
The codec design above is our proposal, not a method attributed to these papers.
