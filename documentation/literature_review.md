# Reconstructive feature discovery: literature and formulation gaps

Review date: 27 October 2026. This is a first critical review, not an exhaustive
systematic review or an established novelty claim.

**Clarification after this first pass:** the primary target is the sparsest
reconstructive domain for each particular input, with the transform itself
allowed to adapt. The [focused follow-up on adaptive transforms](adaptive_transforms.md)
examines that stronger objective, its degeneracies, and what can be certified.
The feature-role questions below remain background rather than prerequisites.
The first two review passes cover 41 papers; two further method reviews accompany
the [current model-design proposal](implementation.md). The reading counts in
this document describe the initial 25-paper pass only.

## What we are trying to understand

Can we learn features that describe a particular speech or audio sample
economically and jointly reconstruct it accurately? What would make one such
description objectively better than another, and what can we establish about
the structure it has learned?

A dictionary, a fixed number of slots, orthogonality, independence, and a particular
neural architecture are hypotheses, not requirements. The original
[problem statement](init/init.tex) already allows dependent, nonorthogonal features.
Speech is the main domain; controlled synthetic signals can expose assumptions
and failure modes. Success on finite synthetic families cannot establish a result
for all signals.

We should distinguish two possible discoveries:

- **Economical building blocks:** recurring patterns that reduce the description
  needed for reconstruction. They need not correspond to named human attributes.
- **Organized factors:** components with stable, distinguishable roles under
  changes to the signal. These may be acoustic or generative rather than linguistic.

Either is a legitimate investigation. Finding the first does not establish the
second. We should not silently require semantic interpretability, nor silently
claim it from a sparse representation.

## Scope and reading depth

Searches covered sparse coding and auditory efficient coding; reconstructive
representation learning; disentanglement definitions and identifiability; temporal
and weak supervision; differentiable synthesis; neural codecs and factorized
speech codecs; object slots; and adaptive representation allocation. Sources were
publisher proceedings, author/university copies of papers, and arXiv. Recent
screening included 2025 speech codecs and a 2026 identifiability paper. Examples
of search phrases were `efficient auditory coding`, `nonlinear ICA identifiability`,
`speech decomposition information bottleneck`, `factorized speech codec`, and
`variable bitrate residual vector quantization`.

The 25 records below distinguish two depths:

- **M — method/assumption review (18 papers):** abstract plus selected method,
  formulation, theorem-statement, or evaluation sections inspected. This does not
  mean every proof, appendix, implementation, or experimental claim was audited.
- **S — abstract screening (7 papers):** title/authors and abstract inspected;
  deeper methodological conclusions are deferred.

[bibliography.md](../bibliography.md) is deliberately only one title-and-authors
line per paper. It includes both depths; the distinction lives here. PDF bylines
are preferred when proceedings metadata differs. In particular, the SpeechSplit
PDF orders David Cox before Mark Hasegawa-Johnson. Its body uses SpeechFlow, while
the proceedings call the method SpeechSplit; these are one paper, not two records.

Accessible papers and extracted text are cached under `runs/literature/`, with
download sources in `sources.json`. Some screened papers have no local full text;
the Olshausen PDF was image-only in extraction. Citations here point to persistent
primary sources rather than depending on that disposable cache.

## The missing steps in our formulation

### 1. Define what a feature is allowed to be

An atom, an event with a time and amplitude, a trajectory, an entire subspace, a
speaker attribute, and a physical sound source are different objects. Each can be
called a feature, but their counts are not comparable. A single trajectory may
contain more information than hundreds of scalar coefficients.

**Question:** Is our first claim about economical signal primitives, organized
factors of variation, or both? Before comparing methods, specify the allowed
object, time scale, and what is shared between recordings. Do not choose the
number or names of feature families merely to fit an available architecture.

### 2. Distinguish reconstruction from identifying a representation

Let an encoder and decoder reconstruct through `D(E(x))`. For an invertible
latent transformation `g`, the encoder `g(E(x))` and decoder `D(g^-1(z))` produce
the same reconstruction. Yet the coordinates may have completely different
meanings. A latent sparsity penalty narrows the possibilities but is itself
coordinate-dependent; adding it is not a general proof of correct factor recovery.

The non-identifiability result in [Locatello et al. (2019)](https://proceedings.mlr.press/v97/locatello19a.html)
makes this ambiguity precise for unrestricted generative models with factorized
latent densities. It does not rule out learning when justified structural
assumptions or additional information are available.

**Question:** What observable evidence or defensible assumption distinguishes
the organization we want from equally reconstructive alternatives? Possible
answers include temporal variation, paired recordings, known transformations,
physical synthesis constraints, and limited attribute supervision. These are
different research hypotheses, not interchangeable regularizers.

### 3. Make “sparsest possible” a meaningful comparison

For a fixed dictionary and distortion tolerance, minimum nonzero coefficient
count is a precise problem. It is generally computationally difficult; see
[Natarajan (1995)](https://epubs.siam.org/doi/10.1137/S0097539792240406).
That is different from minimizing over all possible feature definitions and
decoders, where the comparison has not yet been specified.

For example, a recording can be its own single atom. A decoder can memorize a
training collection and accept one recording index. Neither demonstrates the
reusable economy we seek. Latent dimension also ignores numerical precision,
support locations, event times, and sample-specific decoder parameters.

**Question:** What information is charged, over what collection of recordings,
and at what distortion? Record at least the amount of sample-specific information
and the shared representation's capacity. If adapting the representation per
sample, account for that adaptation. Description-length reasoning is useful here
without committing the project to building a transmission codec; see
[Rissanen (1978)](https://research.ibm.com/publications/modeling-by-shortest-data-description).

### 4. Specify what “accurate” preserves

Recovering a particular waveform is different from generating another waveform
with similar audible content. A representation may preserve words and speaker
quality while losing phase, fine timing, background detail, or a particular noise
realization. Whether those losses matter is part of the question.

**Question:** Are we investigating waveform sufficiency, perceptual sufficiency,
or separate frontiers for both? Keep the metrics already discussed as separate
observations rather than treating one score as proof of the others. Reconstruction
scores alone do not measure feature discovery. A dedicated review of perceptual
metric validity remains outstanding; the present paper set does not settle it.

### 5. Separate reusable knowledge from input-specific selection

Learning common primitives, selecting among them for an input, allocating more
detail to difficult intervals, and inventing new primitives for that input are
four distinct operations. An encoder conditioned on the sample does not by itself
demonstrate the last operation.

**Question:** Which part should remain stable across recordings, and which part
should adapt? Compare reuse and sample adaptation under explicit costs rather
than treating “shared bank or new bank” as an immediate preference question.

### 6. Decide what evidence would establish feature organization

If organized roles are part of our claim, reconstruction and visually appealing
latent plots are insufficient. We need selective-response tests, sensible
recombination where valid, stability across seeds up to appropriate equivalences,
and generalization beyond training speakers and recording conditions.

**Question:** Which observable changes should affect a proposed component, and
which should leave it stable? Speech transformations are imperfect: changing
pitch or time can alter other acoustic properties. A successful augmentation test
may reflect the augmentation algorithm rather than a natural factor. Synthetic
generators provide known controls, but their controls are assumptions to vary.

### 7. Explain completeness, dependence, and residual detail

Removing “redundancy” must not remove information needed to describe interactions.
For example, the observable consequence of one attribute may depend on another.
A component need not reconstruct a meaningful standalone waveform, and not every
combination of attribute values is physically attainable.

**Question:** Can organized components remain jointly sufficient without a residual
carrying most of the description? Measure residual capacity and contribution
explicitly, including what returns when the residual is available. Dependencies
between useful components are permitted; sparse use and independent factors are
different hypotheses.

## Paper-by-paper evidence

The observations below summarize the papers. Statements about applicability and
remaining questions are our interpretation, not claims of missing work everywhere
in the literature.

### A. Learning economical signal primitives

**1. Olshausen and Field (1996) — S.**
[Emergence of simple-cell receptive field properties by learning a sparse code for natural images](https://pubmed.ncbi.nlm.nih.gov/8637596/).
Sparse coding learns visual primitives resembling receptive-field properties.
It establishes historical precedent for useful structure emerging from an
economy objective. This screening does not establish transfer to speech or
recovery of high-level generative attributes. The image-only full-text copy was
not method-reviewed in this pass.

**2. Smith and Lewicki (2006) — M.**
[Efficient auditory coding](https://www.nature.com/articles/nature04485).
Models audio with learned time-relative kernels and events carrying amplitudes
and temporal positions. Kernel shapes and lengths adapt to sound ensembles;
matching pursuit approximates inference. Precision is related to reconstruction
noise, and economy to event use. This is a close antecedent to learning features
for sparse full-waveform description. Its numerical coding improvements do not
certify the globally smallest representation or identify semantic speech factors.
Inspect the model and encoding discussion in the [paper](https://web.stanford.edu/~jlmcc/Presentations/SmithLewicki.pdf).

**3. Aharon, Elad, and Bruckstein (2006) — M.**
[K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation](https://freddy.cs.technion.ac.il/wp-content/uploads/2017/12/K-SVD-An-Algorithm-for-Designing-Overcomplete.pdf).
Alternates sparse inference with atom/coefficient updates using restricted
residuals and singular-value decomposition. Dictionary learning is genuine
learning, but within a chosen additive representation class. Atom updates do
not imply that atoms correspond to generative attributes or that the joint
optimization is globally optimal. Useful as a controlled baseline, not as the
definition of our target.

**4. Natarajan (1995) — S.**
[Sparse Approximate Solutions to Linear Systems](https://epubs.siam.org/doi/10.1137/S0097539792240406).
The abstract states NP-hardness for minimum-support approximate linear solutions
and a greedy approximation result. This concerns a specified matrix and error
criterion. It motivates separating exact conditional statements from heuristic
optimization; it is not a theorem about all possible audio representations.
Proofs and approximation constants were not audited.

**5. Rissanen (1978) — S.**
[Modeling by shortest data description](https://research.ibm.com/publications/modeling-by-shortest-data-description).
The abstract connects model structure and parameter estimation to description
length. This gives a useful accounting perspective on sample-specific features
and shared model information. It does not supply an audio feature ontology or
an immediately usable universal complexity score. Full-text methodology was
not reviewed here.

**6. Bengio, Courville, and Vincent (2013) — S.**
[Representation Learning: A Review and New Perspectives](https://arxiv.org/abs/1206.5538).
The survey frames representation learning through explanatory variation,
learning objectives, inference, and geometry. It is useful background for
separating what is learned from how a sample's representation is computed.
Only its abstract was screened; no detailed survey conclusions are used as
evidence for our proposed research gap.

### B. Defining factors and understanding identifiability

**7. Higgins et al. (2018) — M.**
[Towards a Definition of Disentangled Representations](https://arxiv.org/abs/1812.02230).
Defines disentanglement relative to a decomposition of transformations into
subgroups: different parts of a representation respond to different transformations.
The definition does not require a particular basis or dimension for each part.
This makes feature roles relational and testable, but presupposes relevant
transformation structure. It defines a target; it does not solve discovery of
unknown audio transformations or guarantee complete waveform reconstruction.

**8. Locatello et al. (2019) — M.**
[Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](https://proceedings.mlr.press/v97/locatello19a.html).
The theorem constructs entangled alternatives with the same latent distribution,
and therefore indistinguishable observations, in the stated unrestricted setting.
The empirical study also questions selection of disentangling methods without
supervision. The lesson is to disclose and investigate assumptions. It is not
a universal impossibility result for structured audio, temporal observations,
or learning with additional evidence. The theorem statement and construction's
argument were inspected; a complete proof audit was not performed.

**9. Locatello et al. (2020) — M.**
[Weakly-Supervised Disentanglement Without Compromises](https://proceedings.mlr.press/v119/locatello20a.html).
Paired observations share some generative factors while others change. This
relational evidence changes the identification problem; the practical methods
need not be told which factors changed. Theory still has assumptions on the
generative model and pair distribution. For audio, paired mixtures or controlled
recordings could provide evidence, but ordinary augmentation does not automatically
satisfy those assumptions. Results on images are not direct speech validation.

**10. Khemakhem et al. (2020) — M.**
[Variational Autoencoders and Nonlinear ICA: A Unifying Framework](https://proceedings.mlr.press/v108/khemakhem20a.html).
Uses conditionally factorized exponential-family latent distributions depending
on an observed auxiliary variable. Theorem conditions include an injective
mixing function and sufficiently varied conditional parameters. The first result
identifies sufficient statistics up to a linear transformation; stronger
conclusions require further conditions. This supplies a route from assumptions
to a model, not evidence that an unconditional speech autoencoder recovers
the right features. The useful auxiliary variation in speech must be identified.

**11. Lachapelle et al. (2022) — M.**
[Disentanglement via Mechanism Sparsity Regularization: A New Principle for Nonlinear ICA](https://proceedings.mlr.press/v177/lachapelle22a.html).
Studies sparse dependencies on past latent variables and/or observed auxiliary
variables, with conditions on variability and graph structure. A VAE-based
procedure learns latent mechanisms using binary masks; simulations test recovery.
Mechanism sparsity concerns which variables influence which others. It is not
the same as few active waveform coefficients. Temporal structure offers a
candidate source of evidence, provided the speech-specific assumptions are
defensible rather than simply imported.

**12. Gresele et al. (2021) — M.**
[Independent mechanism analysis, a new concept?](https://proceedings.neurips.cc/paper_files/paper/2021/file/edc27f139c3b4e4bb29d1cdbc45663f9-Paper.pdf).
Adds a structural principle about contributions of sources to a nonlinear
mixing function. Its geometric criterion involves Jacobian columns and excludes
important ambiguities beyond source independence alone. Statistical independence,
mechanism independence, and geometric orthogonality are distinct ideas. This
paper makes one possible assumption explicit; it does not demonstrate that
arbitrary speech attributes satisfy it or that sparse coding requires it.

**13. Simon, Frossard, and De Vleeschouwer (2026) — M.**
[Unsupervised Disentanglement Without Compromises: How Functional Orthogonality Enforces Identifiability](https://proceedings.mlr.press/v306/simon26b.html).
The formulation combines a real-analytic invertible generative mapping with
orthogonal Jacobian columns and global conditions on the latent domain. The
dependent-factor result uses a full-combination support assumption. Experiments
use constructed generative settings and normalizing flows. These conditions
differ from imposing an orthogonal waveform dictionary or a soft group penalty.
The theorem statements motivate an assumption audit; they do not justify such
constraints for our speech task. Proofs were not independently verified.

### C. Reconstructive speech decompositions: the closest prior work

**14. Hsu, Zhang, and Glass (2017) — M.**
[Unsupervised Learning of Disentangled and Interpretable Representations from Sequential Data](https://sls.csail.mit.edu/publications/2017/NIPS17_Hsu.pdf).
FHVAE separates sequence-associated and segment-associated variables through
hierarchical priors, plus a sequence-discriminative objective discouraging collapse.
It demonstrates how temporal grouping can organize speech information without
attribute labels. The organization rests on timescale assumptions; it is not
automatic discovery of every acoustic feature type. Spectral reconstruction
and tested speech tasks do not establish sample-accurate waveform sufficiency
or a smallest possible description.

**15. Qian et al. (2020) — M.**
[Unsupervised Speech Decomposition via Triple Information Bottleneck](https://proceedings.mlr.press/v119/qian20a/qian20a.pdf).
SpeechSplit prespecifies content, timbre, pitch, and rhythm. Bottlenecks,
normalized pitch input, speaker identity, and random temporal resampling encourage
their separation. The formulation assumes resampling preserves other components;
its content variable also includes residual information. “Without text labels”
therefore does not mean without attribute knowledge or structural assumptions.
This is strong prior work on reconstructive decomposition, but not discovery
of an unconstrained feature vocabulary or proof of minimum representation size.

**16. Ju et al. (2024) — M.**
[NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models](https://proceedings.mlr.press/v235/ju24b.html).
FACodec assigns content, prosody, timbre, and detail roles. Codec training uses
phoneme, normalized-F0, and speaker supervision, gradient reversal, and detail
dropout. The paper explicitly discusses information leaking into detail codes.
It is close precedent for combining structured components and high-quality
reconstruction; roles are prescribed and anchored. Codec claims must be separated
from the complete TTS system's scale and results. It does not establish that
these roles are the unique or smallest sufficient set.

**17. Luo et al. (2025), arXiv v1 — M.**
[DeCodec: Rethinking Audio Codecs as Universal Disentangled Representation Learners](https://arxiv.org/abs/2509.09201v1).
Prescribes speech/background subspaces, then semantic and residual speech codes.
Training combines orthogonality, recombination targets from constructed mixtures,
and HuBERT guidance from clean speech. Its controllable feature selection is
directly relevant prior work. Alignment has evidence beyond orthogonality alone.
The reported decomposition should not be generalized into a proof that
orthogonality implies statistical independence, nor into discovery of unknown
feature classes. This review uses the identified arXiv version, without asserting
a later conference publication status.

**18. Zhang et al. (2024) — M.**
[SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models](https://proceedings.iclr.cc/paper_files/paper/2024/file/86d1ab582afb247ccaa84bec4a7e24f7-Paper-Conference.pdf).
Uses HuBERT distillation to guide the first residual-vector-quantizer layer
toward content; later layers encode complementary acoustic information.
Reconstruction, adversarial, commitment, and distillation losses serve different
roles. This demonstrates a reconstructive hierarchy anchored by a teacher.
Residual quantization by itself does not identify named factors, and a teacher's
semantic organization is not evidence for discovering feature types without
that information.

**19. Li et al. (2025), arXiv screening — S.**
[DisCo-Speech: Controllable Zero-Shot Speech Generation with A Disentangled Speech Codec](https://arxiv.org/abs/2512.13251).
The abstract describes prescribed content/prosody/timbre branches, followed by
fusion and reconstruction, for controllable TTS. This reinforces that factorized
reconstructive codecs are an active research area. Hybrid-loss details,
supervision, leakage controls, and evaluation require full-text review before
using this as a detailed architectural comparison.

**20. Khurana et al. (2025), arXiv screening — S.**
[Factorized RVQ-GAN For Disentangled Speech Tokenization](https://arxiv.org/abs/2506.15456).
HAC's abstract specifies acoustic, phonetic, and lexical levels with HuBERT and
LaBSE distillation. This is evidence of another close hierarchical approach,
with predefined levels and external guidance. Detailed experimental and
supervision analysis remains outstanding; the abstract's disentanglement claims
are not treated as independently verified.

### D. Alternative inductive biases and adaptive allocation

**21. Engel et al. (2020) — M.**
[DDSP: Differentiable Digital Signal Processing](https://arxiv.org/abs/2001.04643).
Learns controls for differentiable synthesis components. The studied harmonic-plus-noise
model constrains the sound generator; supervised and unsupervised variants
handle pitch differently. Experiments concern monophonic musical sources.
Interpretability is partly supplied by the synthesis model, and its raw control
trajectories need not be lower-dimensional than the waveform. This is a useful
route to test physical structure, not a generic solution to unknown speech
feature discovery or exact stochastic-detail recovery.

**22. Zeghidour et al. (2021) — S.**
[SoundStream: An End-to-End Neural Audio Codec](https://research.google/pubs/soundstream-an-end-to-end-neural-audio-codec/).
The abstract describes a learned convolutional encoder/decoder and residual
vector quantizer, with joint reconstruction/adversarial training and variable-rate
training. It establishes compact learned audio reconstruction as existing work.
The screening does not establish identifiable constituent roles. Full methods
and benchmark protocols were not reviewed here.

**23. Kumar et al. (2023) — M.**
[High-Fidelity Audio Compression with Improved RVQGAN](https://arxiv.org/abs/2306.06546).
DAC improves neural audio compression through periodic inductive bias and
quantizer/training changes. Its factorized codebook separates low-dimensional
lookup from higher-dimensional embeddings; this is not a claim of semantic
factor separation. It is a strong relevant reconstruction baseline. Its
compression objective does not itself test discovery of stable feature roles,
and perceptual quality should not be equated with sample-exact recovery.

**24. Locatello et al. (2020) — M.**
[Object-Centric Learning with Slot Attention](https://proceedings.neurips.cc/paper/2020/file/8511df98c02ab60aea1b2356c013bc0f-Paper.pdf).
Iterative competitive attention creates exchangeable object slots. The image
autoencoder decodes slots with a shared decoder and combines them using masks.
Its slots can bind different objects; slot indices are not permanent attribute
names. Acoustic sources overlap differently from visible image regions, and
object instances differ from attributes of one source. This provides a useful
alternative meaning of “slot,” not an automatic implementation of our feature
containers.

**25. Chae et al. (2025), ICASSP camera-ready arXiv v3 — M.**
[Variable Bitrate Residual Vector Quantization for Audio Coding](https://arxiv.org/abs/2410.06016v3).
An importance network predicts time-varying masks controlling the number of
active residual quantizers; surrogate gradients and a rate loss enable training.
This is direct prior work on allocating representation to the particular input.
It adapts detail allocation within a specified hierarchy, rather than discovering
new feature kinds. The paper calculates rate from code use without a trained
entropy coder, a distinction worth retaining in complexity comparisons.

## What the literature does and does not resolve

| Question | Evidence already present | What remains to establish for our study |
|---|---|---|
| Can audio primitives be learned for sparse reconstruction? | Auditory efficient coding; dictionary learning | Which primitive class and cost best fit our data and accuracy regime |
| Can a learned code reconstruct audio compactly? | SoundStream; DAC | Whether its organization is the structure we seek |
| Can reconstructive speech components have useful roles? | FHVAE; SpeechSplit; FACodec; DeCodec; SpeechTokenizer | Whether roles can emerge with fewer prescribed identities, and under which evidence |
| Can representation use adapt to an input? | Sparse inference; VRVQ | Adaptation of amount versus selection or discovery of feature types |
| Can latent recovery be justified theoretically? | Conditional ICA; weak supervision; mechanism sparsity; geometric assumptions | Whether assumptions hold in our speech setting and which ambiguities remain |
| Can slots discover a composition? | Object-centric image models | What composition means for overlapping sound and attributes, and how to evaluate it |

The broad ideas “learn features,” “sparse reconstruction,” and “factorized codec”
are therefore not sufficient novelty claims. Their conjunction is also not
automatically new. The promising unresolved issue **for us** is the relation
between the evidence supplied during learning, the structure that emerges, and
its cost at a specified reconstruction accuracy.

A candidate research question, still requiring a closer novelty check, is:

> Under what observable variations and model assumptions can a reconstructive
> audio representation discover reusable components, adapt their use to each
> sample, and demonstrate measurable economy without assigning all component
> roles in advance?

If distinct feature roles become part of the claim, add the requirement that
those roles be supported by selective-response and generalization evidence.
Economy alone remains a valid narrower claim. Neither formulation assumes
orthogonality, statistical independence, a fixed dictionary, or a particular
number of components.

## Questions to settle before another architecture

| Order | Question | Concrete outcome of the discussion |
|---|---|---|
| 1 | What does a feature need to accomplish beyond reconstruction, if anything? | Distinguish economy-only and organized-factor claims |
| 2 | Which signal differences must be retained? | Separate waveform and perceptual accuracy criteria; specify recording scope |
| 3 | What costs are comparable across representations? | Account for values, indices/times, precision, shared capacity, and adaptation |
| 4 | What observations make the desired structure distinguishable? | Justify temporal, paired, physical, or supervised evidence; list failure cases |
| 5 | What would refute the feature claim? | Controlled mixing/reparameterization counterexamples and negative controls |
| 6 | Which closest existing method already addresses each part? | Matched comparison with assumptions and teacher information disclosed |
| 7 | Which minimal model expresses the surviving hypothesis? | Architecture and loss derived from that hypothesis, then training design |

Useful thought experiments before training include rotating or nonlinearly
remapping an equally reconstructive latent space, storing a sample as its own
atom, allocating everything to a residual, and changing correlated generative
controls separately. Each exposes a different ambiguity; no large model is
needed to establish that ambiguity.

For eventual tests, retain several synthetic generative families, multiple
ways of generating the same apparent property, unseen combinations, and a
path to speaker-disjoint speech. Do not assume synthetic controls are the
only valid answer or require one latent coordinate per control. Speech-specific
claims need speech evidence. This is a research sequence, not authorization
to start another architecture sweep before the formulation is discussed.

## Implications for the existing repository

The current orthonormal block dictionary is a restricted sparse-coding baseline.
Its exact support selection is conditional on that transform class and waveform
squared error; it does not solve feature discovery across representation spaces.
Its [pilot result](block_orthogonal_results.md) does not establish an absence of
useful audio structure. The earlier [experiment registry](experiments.md) remains
a record of candidate experiments, not an agreed derivation of the target model.

No model implementation or training routine is changed by this review. The next
scientific step is to settle the first questions above and deepen the closest
comparisons, rather than make the current dictionary larger or more constrained.

## Remaining literature work

This first pass maps the problem; it does not close the literature study.

1. Promote the seven abstract-screened records to deeper review where relevant;
   prioritize the recent factorized speech codecs and complexity foundations.
2. Trace backward and forward citations of auditory efficient coding,
   SpeechSplit/FHVAE, FACodec, DeCodec, and identifiable representation learning.
   Check newer versions and publication records before final comparisons.
3. Review source-filter/sinusoidal models, convolutional sparse coding and its
   recovery conditions, nonlinear source separation, and multiscale auditory
   representations in more depth. This pass does not establish their limits.
4. Review sufficiency/minimality and finite-precision complexity formulations,
   including what can be compared without building an actual codec.
5. Review speech/audio reconstruction metrics, listening-test evidence, and
   methods for assessing feature organization without known ground-truth factors.
6. For each closest method, extract data partitions, teacher pretraining,
   supervision, code availability, reconstruction regime, adaptation costs, and
   leakage tests before selecting reproduction experiments.

For future entries, append only title and authors to `bibliography.md`; put
links, reading depth, assumptions, evidence, and implications in this document.
