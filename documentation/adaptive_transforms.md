# Finding the sparsest domain for a particular audio sample

Review date: 27 October 2026. This follow-up sharpens the
[initial literature review](literature_review.md) around the user's clarified
objective. Feature semantics and disentanglement are secondary unless needed to
justify the sparsifying representation. No architecture is selected here.

## The clarified target

Given a waveform, or information derived from it, learn a mapping into a domain
that may be unknown beforehand. The mapping should adapt to the sample and
produce the fewest nonzero coefficients compatible with accurate reconstruction.
The strongest requested statement is that no other representation can be sparser.

This is stronger than outperforming DCT, learning a useful dictionary, or
minimizing average training loss. It contains three separate tasks:

1. Define which representations are comparable without hiding the sample in the
   coordinate system, decoder, numerical precision, or side inputs.
2. Find a good representation for each input within that definition.
3. Establish a lower bound that proves no allowed representation is sparser.

The domain does not have to be a named transform or a set of human-interpretable
features. It also need not be orthogonal, statistically independent, or linear.
We should not replace the minimum-support question with a semantic-factor task
or silently substitute minimum bitrate as the objective.

## A mathematical obstruction before model design

The following constructions are our elementary derivations, not empirical claims
about speech and not findings attributed to a particular paper.

### Unrestricted sample-specific linear transforms already attain one nonzero

Let `x` be any nonzero vector in `R^n`, representing a finite waveform. Choose an
orthonormal basis whose first vector is `x / ||x||_2`. Write its analysis matrix
as `Q_x`. Then

```text
Q_x x = (||x||_2, 0, ..., 0)
x = Q_x^T (||x||_2, 0, ..., 0).
```

Every nonzero waveform therefore has an exact one-sparse representation under a
suitably chosen linear transform. No invertible linear transform can produce a
zero vector from a nonzero input, so one is also the global minimum in this
unrestricted linear setting.

This example uses an orthogonal transform only to make the argument especially
strong: even perfect conditioning, energy preservation, and exact inversion do
not prevent the issue. Orthogonality is not a proposed requirement for the model.

The direction of the original waveform is contained in `Q_x`. A decoder with only
the scalar norm cannot know that direction. If it is given `Q_x`, that is additional
sample-specific information. If it can generate `Q_x` from a small description
and a fixed learned model, that could be useful economy, but the small description
must be included in the representation accounting.

### Allowing arbitrary affine transforms makes the minimum zero

For a fixed target waveform, define `T_x(y) = y - x`. This is invertible, and
`T_x(x) = 0`. The inverse recovers the sample from the transform's offset.
Thus unrestricted nonzero count can become zero while the transform contains the
entire sample. Allowing a sufficiently flexible sample-specific decoder has the
same problem.

Consequently, the literal unrestricted support objective is degenerate rather
than merely a difficult unsolved learning problem. A meaningful formulation must
say what may be adapted and what information reconstruction can use.

### Fixed support count also ignores precision

A finite quantized waveform can be packed into one sufficiently large integer.
A fixed decoder can unpack its sample values. The support count is one, but the
integer still requires essentially the original information to specify. Treating
an arbitrary-precision number as one ordinary coefficient makes support count an
unfair cross-domain measure.

This does not invalidate coefficient count within a controlled setting. It means
precision, dynamic range, and decoder conventions must be comparable. Exact real
arithmetic and finite-precision audio are distinct formulations.

## What changes if the model is shared?

Suppose one decoder is fixed for all test samples. It can receive an adaptive
code and, if necessary, an adaptive transform description. The decoder must be
able to reconstruct from those inputs without receiving the original waveform.

Saying that a network generates the transform from the waveform is not enough:
we must explain how the inverse obtains the transform. It may be known from a
shared finite family, transmitted through an index, generated from an explicit
parameter code, or inferable from the code itself. Any route is possible, but an
unaccounted waveform-conditioned decoder is a hidden information channel.

A second distinction is local versus uniform sufficiency. A decoder designed for
one known point can output it with no code. A single stable decoder required to
represent an entire full-dimensional signal class faces different limits.

For example, assume a fixed Lipschitz decoder from `R^m` to `R^n`, and codes with
at most `k < n` nonzeros. There are finitely many supports; each is at most
`k`-dimensional. Lipschitz images of these sets cannot cover an open subset of
`R^n` because their n-dimensional volume is zero. Thus exact recovery of every
waveform in an open full-dimensional class is impossible under these assumptions.
This is a dimension argument, not a claim that speech occupies an open set or
that useful approximate compression is impossible. Count additional continuous
adaptation parameters if the decoder receives them.

The role of stability and restrictions on encoder/decoder mappings is developed
in [Cohen et al., Optimal Stable Nonlinear Approximation](https://arxiv.org/abs/2009.09907).
That paper's bounds concern a model class, not a certificate of the shortest
description of one arbitrary waveform.

## The best domain can depend on the accuracy target

For a given orthonormal transform, the minimum squared error with `k` retained
coefficients is the sum of the squared discarded coefficients. This supplies a
simple example of why a single best transform need not exist for all accuracies.

Consider two coordinate systems for the same unit-energy vector, with sorted
coefficient energies:

```text
A: (0.60, 0.20, 0.20)
B: (0.50, 0.45, 0.05).
```

Both are possible: their square-root coefficient vectors have equal norms, so
orthogonal changes of coordinates can map the same vector to either one.

| Coefficients retained | Squared error in A | Squared error in B | Better domain |
|---|---|---|---|
| 1 | 0.40 | 0.50 | A |
| 2 | 0.20 | 0.05 | B |
| 3 | 0 | 0 | Tie |

At squared error tolerance `0.45`, A needs one coefficient and B needs two.
At tolerance `0.10`, A needs three and B needs two. These are squared errors,
normalized by the original unit energy.

Thus the search should generally be conditioned on `(signal, distortion target)`,
or on `(signal, coefficient budget)`. A transform minimizing an entropy score
also need not minimize support at every distortion. DeVore's
[Nonlinear approximation, section 8.1](https://people.math.sc.edu/devore/publications/nlacta.pdf)
explicitly discusses changes of best basis as the retained term count varies.

As a numerical check of the algebra, a 64-dimensional Householder construction
gave an off-axis coefficient maximum of `3.33e-16`, orthogonality error of
`1.42e-15`, and relative reconstruction error of `1.42e-16`. The two tail-energy
curves above were also checked. Results are cached in
`runs/literature/sparsity_counterexamples.json`. This is an algebraic illustration,
not a trained model or a speech benchmark.

## A formulation that preserves minimum support as the objective

Let `h` be shared knowledge fixed before evaluating a test sample. It can include
a learned model and reconstruction conventions. Let `theta` describe any allowed
sample-specific transform choice or adaptation, and let `z` be its sparse code.

For a declared admissible family `A`, define

```text
s*_A(x; epsilon, B | h) = min over theta,z  ||z||_0

subject to:
    d(x, D_h(theta, z)) <= epsilon
    L(theta | h) <= B
    (theta, z, D_h) obey the rules of A.
```

Here `d` is the chosen distortion, `B` limits adaptation information, and
`L(theta | h)` is a specified finite description length under fixed conventions.
Coefficient precision/range and support-index conventions are also declared.
For a finite transform library, `theta` may simply be a basis identifier.
An alternative non-bit-based capacity restriction is possible, but its units and
stability implications must be made explicit.

This retains the user's requested objective: fewest nonzeros. It prevents
unbounded free adaptation rather than declaring bitrate the sole goal. A useful
study can report both minimum support at each adaptation budget and total
description size. At large enough adaptation budgets, the one-sparse construction
may legitimately return; the budget exposes where its information went.

Unknown domains are compatible with this formulation. `A` could contain learned
invertible mappings, learned filters, structured nonlinear decoders, or compact
generative programs. The important restriction is on what can be chosen and
specified, not whether the resulting domain has a familiar name.

The conditioning on `h` is deliberate. A certificate with a fixed learned decoder
is conditional on that decoder. To compare different shared models, include their
allowed descriptions/capacities in the search and in its lower bounds. One could
make `h` a fixed interpreter and let `theta` describe any reconstruction program
within declared length/runtime bounds. This is a broad unknown-domain class,
although exhaustive search may be impractical. We should derive admissibility
from reconstruction access, precision, stability, and resource assumptions;
merely restricting it to a few convenient named transforms would not establish
the user's stronger claim.

Further necessary details are:

- **Scale and conditioning:** an invertible transform can shrink every coefficient
  and amplify it on inversion. Exact support is unaffected, but an l1 penalty or
  absolute numerical threshold can be fooled. A condition-number bound alone
  does not constrain uniform scaling; control scale as well.
- **Actual reconstruction:** for general nonorthogonal transforms, small
  transform-domain residual need not imply equally small waveform residual.
  Evaluate through the inverse/decoder. The two residuals coincide under an
  isometry, which is a special case rather than a requirement.
- **Accuracy versus noise:** decide whether the particular recording's noise and
  phase must be retained. Denoising the input solves a different reconstruction
  target from reproducing it.
- **Window and duration:** one nonzero per short frame is not one nonzero for the
  whole recording. Specify the units and include boundaries and adaptive timing.
- **Shared-model capacity:** a fixed model need not be charged afresh for every
  sample, but compare its size, training information, and reuse consistently.
  Sample-specific changes cannot be treated as shared training knowledge.

## What “nothing sparser exists” would require

A feasible reconstruction using `U` nonzeros proves an **upper bound** on the
unknown optimum. A lower bound `L` must rule out all better allowed choices,
including transform choices, when those are optimization variables:

```text
L <= s*_A(x; epsilon, B | h) <= U.
```

Matching integer bounds certify optimality. Equivalently, a feasible code with
`k` nonzeros plus a proof that every allowed representation with at most `k-1`
nonzeros violates the constraints certifies the claim within `A`.

| Evidence | What it establishes |
|---|---|
| Better than DCT/STFT on selected samples | Improvement over those baselines |
| Best of all tested candidates | An upper bound; the candidate search may miss a better solution |
| An exact sparse-code update | Optimality of that subproblem with other variables fixed |
| Convergence of alternating updates | The convergence statement actually proved; often a local result |
| Certified global solution in a declared family | No sparser feasible member of that family |
| Universal shortest finite description | A different objective with fundamental computability limits |

If the transform is learned and then frozen, a certificate for sparse inference
does not certify that the learned transform is the best possible one. Similarly,
minimizing an l1 surrogate or a weighted distortion-plus-sparsity loss does not
automatically solve minimum l0 at a hard error tolerance. Weighted objectives may
miss some constrained trade-offs.

### Where exact answers are already possible

For a fixed orthonormal transform and waveform squared error, sorting coefficient
energies gives exact best-k approximation. A finite list of such transforms can
be exhaustively compared at each tolerance. These are conditional oracle results.

For a structured basis tree, additive costs permit dynamic programming rather
than enumerating every basis. Coifman and Wickerhauser's best-basis method is a
central precedent. For a direct coefficient-budget version, a possible extension
is to retain a cost curve `C_v(k)` at each node:

```text
C_v(k) = min(
    best tail-energy error retaining at most k coefficients at node v,
    min over k1+k2 <= k [C_left(k1) + C_right(k2)]
).
```

With a fixed finite orthogonal tree and squared-error additivity, this recurrence
finds the best allowed pruning and coefficient allocation. This recurrence is our
formulation of a reference problem, not an attribution of that exact algorithm
to the original entropy paper. Adding basis-description budgets requires accounting
for them as well. General learned mappings or perceptual losses may destroy the
decomposition that makes this calculation exact.

For fixed nonorthogonal dictionaries, uniqueness/recovery conditions and global
optimization supply other conditional routes. For unrestricted learned domains,
we cannot transfer those certificates without proving their assumptions apply.

### The strongest information-based interpretation

If the intended claim is instead the shortest complete effective description of
the particular finite waveform, including the transform description, the closest
theoretical language is algorithmic information theory. With a fixed universal
reference interpreter, one can define a shortest program producing the waveform
or a reconstruction within the chosen distortion.

This is not coefficient-count sparsity. It also does not provide a universal
trainable oracle: exact Kolmogorov complexity is not computable for arbitrary
finite strings of unbounded length. Particular cases and restricted description
classes can still admit proofs. The claim is about the absence of a general exact
algorithm, not that every individual certification is impossible.
[Vitányi (2020)](https://homepages.cwi.nl/~paulv/papers/incomp.pdf) explains the
distinction. [Vereshchagin and Vitányi (2010)](https://arxiv.org/abs/cs/0411014)
develop rate-distortion questions for individual data; distribution-average
Shannon limits are not automatically certificates for a particular recording.

Finite-precision, bounded-runtime description classes can yield finite, decidable
searches, often enormously expensive. Both the reference description language
and resource bounds then belong to the problem definition.

## Focused literature extension

Sixteen papers were added to [bibliography.md](../bibliography.md): fifteen had
selected formulations, methods, or theorem statements examined; one was abstract
screened. This is not a full proof audit or an exhaustive novelty search. The
first review's 25 records remain intact, bringing these two passes to 41 papers.
Two later reviews accompany the [current model-design proposal](implementation.md),
bringing the bibliography to 43 entries.
The numbered entries below continue that earlier review. Author/preprint versions
were used where noted; no implementation or benchmark was reproduced.

| Paper | Reading and relevance to the clarified problem | Limit of its optimality claim |
|---|---|---|
| **26. Coifman and Wickerhauser (1992), Entropy-Based Algorithms for Best Basis Selection** — [paper](https://bpb-us-e1.wpmucdn.com/sites.gatech.edu/dist/2/436/files/2011/04/coifman92en.pdf?bid=436) | Method reviewed: tree-organized libraries, information cost, and bottom-up basis selection; includes acoustic segmentation. Direct precedent for input-adapted domains. | Best for the specified library and cost. Entropy optimality is not minimum support for every reconstruction tolerance, nor a search over every transform. |
| **27. Ramchandran and Vetterli (1993), Best wavelet packet bases in a rate-distortion sense** — [abstract](https://pubmed.ncbi.nlm.nih.gov/18296206/) | Abstract screened: dynamic-programming pruning for adaptive coding trees with operational rate-distortion optimization. | The abstract specifies the convex hull of the operational curve. Full methodology remains to be inspected; no claim of unrestricted minimum-support optimality follows. |
| **28. Ravishankar and Bresler (2013), Learning Sparsifying Transforms** — [paper](https://transformlearning.csl.illinois.edu/assets/Sai/JournalPapers/SaiTSP2013Square.pdf) | Formulation reviewed: learn a square analysis transform and sparse approximations, with constraints/regularization preventing degenerate solutions. Addresses learning the domain directly. | Works in specified transform classes and learning objectives. Transform-domain approximation and waveform reconstruction require careful comparison; it is not a universal per-sample sparsity theorem. |
| **29. Ravishankar and Bresler (2015), ℓ₀ Sparsifying Transform Learning with Efficient Optimal Updates and Convergence Guarantees** — [paper](https://arxiv.org/abs/1501.02859) | Formulations and convergence statement reviewed: exact sparse approximation in transform coordinates and closed-form transform updates; log-determinant and scale regularization. | Globally solved update steps are not a globally solved joint learning problem. The stated convergence is to a set of local minimizers; a special error-free model can attain a known lower bound. |
| **30. Ravishankar and Bresler (2016), Data-Driven Learning of a Union of Sparsifying Transforms Model for Blind Compressed Sensing** — [paper](https://arxiv.org/abs/1511.01289) | Union-of-transforms formulation reviewed: patches are clustered with transforms that better sparsify them; jointly adapts image reconstruction and the transform collection. | Evidence is primarily imaging. Partial/global subproblem and local convergence statements do not establish unrestricted global optimality of the whole adaptive model. |
| **31. Pătraşcu, Rusu, and Irofti (2024), Learning Explicitly Conditioned Sparsifying Transforms** — [paper](https://arxiv.org/abs/2403.03168v1) | Formulation and update discussion reviewed: explicit conditioning and norm constraints replace indirect control, permitting nonorthogonal transforms. | The joint problem is nonconvex; the paper does not supply an all-domain per-sample optimality certificate. Conditioning and scale are practical mathematical constraints, not evidence that orthogonality is necessary. |
| **32. DeVore (1998), Nonlinear approximation** — [paper](https://people.math.sc.edu/devore/publications/nlacta.pdf) | Selected best-n-term and adaptive basis sections reviewed, especially 8.1. Formalizes approximation over libraries and notes that the preferred basis can change with term count. | A survey of approximation within specified systems/classes. The unknown domain and permitted representation parameters still require definition. |
| **33. Cohen, Dahmen, and DeVore (2009), Compressed sensing and best k-term approximation** — [author preprint](https://people.math.sc.edu/devore/publications/CDDSensing_6.pdf) | Instance-optimality definition and introduction reviewed in the 2006 author preprint. Bounds recovery error against each vector's best-k-term error. | “Instance optimal” is relative to a reference approximation system, norm, and constant. It does not mean discovering the sparsest transform among all possible coordinate systems. |
| **34. Cohen, DeVore, Petrova, and Wojtaszczyk (2022), Optimal Stable Nonlinear Approximation** — [2020 preprint](https://arxiv.org/abs/2009.09907) | Formulation reviewed: encoder and reconstruction maps, stability restrictions, model-class widths, and entropy comparisons. Supplies a principled language for unknown nonlinear domains with stability. | Bounds concern specified classes of signals and admissible mappings. Worst-case class bounds do not certify a particular signal's global minimum support. |
| **35. Donoho and Elad (2003), Optimally sparse representation in general (nonorthogonal) dictionaries via ℓ1 minimization** — [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC153464/) | Uniqueness and spark-based arguments reviewed: sufficiently sparse exact representations are unique, with conditions relating l1 solutions to l0 solutions. | Dictionary is fixed. Conditions are sufficient, may be hard to verify, and do not prove that no other dictionary or transform offers a smaller code. Exact-representation results are not automatically noisy/perceptual guarantees. |
| **36. Vereshchagin and Vitányi (2010), Rate Distortion and Denoising of Individual Data Using Kolmogorov Complexity** — [2009 preprint](https://arxiv.org/abs/cs/0411014) | Individual-data formulation and stated results reviewed. Distinguishes per-object algorithmic rate-distortion behavior from distribution-average theory. | Provides a theoretical objective, not an efficient universal procedure for learning or certifying the shortest representation of an audio sample. Program length and coefficient count are different quantities. |
| **37. Vitányi (2020), How Incomputable Is Kolmogorov Complexity?** — [paper](https://homepages.cwi.nl/~paulv/papers/incomp.pdf) | Incomputability theorem and upper-approximation argument inspected. Explains why discovering progressively shorter descriptions does not generally reveal when the shortest has been reached. | The limitation concerns general Kolmogorov complexity with a fixed interpreter. It does not forbid exact solutions in finite restricted models or proofs for particular inputs. |
| **38. Hazimeh, Mazumder, and Saab, Sparse Regression at Scale: Branch-and-Bound rooted in First-Order Optimization** — [2021 preprint](https://arxiv.org/abs/2004.06152v2) | Mixed-integer formulation and bound-based framework reviewed for l0-plus-ridge regression. Illustrates separation of candidate upper bounds from relaxation-based lower bounds. | The certified objective is fixed-design regularized regression. Applying it to minimum support at an audio distortion constraint or to joint transform learning needs a new formulation and appropriate bounds. |
| **39. Gregor and LeCun (2010), Learning Fast Approximations of Sparse Coding** — [paper](https://icml.cc/2010/papers/449.pdf) | Sparse inference formulation and learned iterative architecture reviewed. A feed-forward learned predictor approximates codes for a specified sparse-coding problem. | Fast learned prediction is distinct from an optimality certificate. Its l1 objective and dictionary assumptions do not solve universal minimum-support transform discovery. |
| **40. Michau, Frusque, and Fink (2022), Fully learnable deep wavelet transform for unsupervised monitoring of high-frequency time series** — [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8872732/) | Learned filter/threshold architecture and reconstruction-plus-coefficient-l1 objective reviewed, using the preprint and published text. Evaluates sound tasks and gives a relevant learned-domain example. | Its wavelet-inspired family and surrogate loss do not certify global minimum l0. Denoising and downstream task results differ from complete reconstruction of the particular noisy recording. |
| **41. Ruiz Reyes et al. (2003), Adaptive wavelet-packet analysis for audio coding purposes** — [publisher text](https://www.sciencedirect.com/science/article/abs/pii/S0165168402004899) | Abstract and publisher-provided method/evaluation excerpts reviewed, not the full paper. Selects audio-dependent packet decompositions with a perceptual cost and adaptive time analysis. | Optimization is within a wavelet-packet coding scheme. Perceptual bit allocation differs from waveform minimum-support optimality; full-paper reproduction and proof inspection remain outstanding. |

The closest conceptual thread is now **adaptive approximation and transform
learning**, with global optimization and algorithmic information theory supplying
different meanings and limits of “optimal.” Factorized speech codecs remain
useful context but no longer define the central question.

## The research gap we can responsibly state

The unrestricted scalar-support objective has the explicit degeneracies above.
That is a formulation problem, not a literature gap to solve with a larger model.
Once a reconstructible, stable, fairly accounted representation class is declared,
existing literature already supplies parts of the solution: adaptive basis
selection, learned transforms, local transform selection, fast sparse inference,
and conditional optimality results.

The remaining candidate research problem is to connect these parts for our audio
setting: discover a sufficiently expressive adaptive domain while obtaining
meaningful bounds on each sample's minimum support under stated reconstruction
and adaptation constraints. This is a direction to investigate, not a demonstrated
novel contribution. Establishing novelty requires closer citation tracing and
comparison with adaptive coding, transform learning, and global optimization.

It separates into the following questions:

| Priority | Question | Why it precedes an architecture |
|---|---|---|
| 1 | What information may change per recording, and what is fixed before it? | Rules out hidden waveform-specific coordinate systems while permitting useful adaptation |
| 2 | What precision, stability, and adaptation budget make coefficient counts comparable? | Prevents an apparent improvement created only by units or unbounded coefficient information |
| 3 | Which accuracy targets and recording details are required? | The optimal domain can change with the target and with whether noise is retained |
| 4 | Which reference families permit exact or bounded per-input answers? | Supplies a way to check whether learning finds sparse solutions or merely claims to |
| 5 | Can a learned predictor approach those optima, with what gap? | Separates inference speed and learning error from limits of the representation family |
| 6 | What expressive extension improves the frontier without losing all useful bounds? | Lets model complexity grow for a reason rather than selecting a fashionable architecture |

For eventual experiments, the important evidence is a per-sample
support-versus-distortion curve with adaptation costs and, where available,
lower/upper bounds. Average reconstruction alone is insufficient for the
requested instance-specific claim. Report how often a result is certified, how
large unresolved gaps remain, and whether the chosen domain changes with the
accuracy target. If a learned selector is added, compare it to the oracle of the
same family. If the family itself is learned, make that additional uncertainty
explicit.

Synthetic signal families remain useful for controlled examples and exact small
searches. Then test on unseen speakers and recording conditions. Synthetic
generative controls need not become latent coordinates: minimum support is the
primary target. No experiment over finitely many models proves optimality over
all future representation designs.

## What this changes in the project

The first discussion should now concern admissible adaptation, reconstruction
information, and support certificates. We do not need to decide whether the
latent features correspond to familiar speech attributes before addressing that.
Likewise, no dictionary bank, orthogonality penalty, or named transform is required
by the clarified objective.

The current implementation is still a conditional baseline; it is not evidence
that the best domain has been found. This review changes documentation and the
bibliography only. The algebraic checks are research illustrations, not another
training run or a new model implementation.

The next literature tasks are full review of the screened rate-distortion paper,
forward/backward citation tracing of the closest transform-learning and
best-basis work, audio-specific comparisons including transform side information,
and methods for bounding joint transform-and-support optimization. Any eventual
architecture proposal should state exactly which of these questions it answers
and which optimality claims remain unsupported.
