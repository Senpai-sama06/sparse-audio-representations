# Searching for the sparsest representation as an optimization problem

29 October 2026. This note formulates the requested search over **both the domain
and the coefficients**. It also records a completed tiny exact optimization,
without a neural encoder, to establish that we can compute and certify solutions
in a declared family.

## The joint objective

For a nonzero sequence $x\in\mathbb R^N$, fixed shared reconstruction rules
$h$, and target normalized squared error $\eta$, define

$$
k^*(x;\eta,B)
=\min_{\theta,z}\|z\|_0
\quad\text{subject to}\quad
\|x-G_h(\theta,z)\|_2^2\leq\eta\|x\|_2^2,\qquad
\theta\in\Theta_B,\quad z\in\mathcal Z.
$$

Here:

- $\theta$ specifies a candidate domain or transform, available to the decoder.
- $z$ contains its coefficients; $\mathcal Z$ specifies their precision,
  range and maximum count.
- $\Theta_B$ is a declared family with a bounded transform description or
  another explicit, justified adaptation-capacity constraint.
- $G_h$ reconstructs using only shared rules, $\theta$, and $z$. It does
  not receive the original sequence as an additional reconstruction input.

For silence, specify an absolute error convention. The zero case in the completed
linear prototype below reconstructs exactly.

This objective retains minimum nonzero count, rather than substituting an
L1 penalty or bitrate objective. Description and precision limits make the
comparison meaningful. A transform need not have a familiar name. It can be
generated from permitted parameters or a finite program.

Searching over arbitrary unrestricted bases would immediately recover the
one-coefficient construction from the [existence proof](sparsest_representation.md).
The computational challenge of interest is the joint optimum **under declared
reconstruction and adaptation rules**. Norm preservation alone does not prevent
a sample-specific basis from storing the entire waveform.

## An exact mixed-integer formulation

Let $M$ be the maximum number of coefficient coordinates and $R_j$ their
declared magnitude bounds. Introduce binary support variables $s_j$:

$$
\begin{aligned}
\min_{\theta,z,s}\quad &\sum_{j=1}^M s_j\\
\text{subject to}\quad
&\|x-G_h(\theta,z)\|_2^2\leq\eta\|x\|_2^2,\\
&-R_js_j\leq z_j\leq R_js_j,\qquad j=1,\ldots,M,\\
&s_j\in\{0,1\},\qquad\theta\in\Theta_B,\qquad z\in\mathcal Z.
\end{aligned}
$$

At an optimum this objective equals the actual number of nonzero coefficients:
$z_j\ne0$ requires $s_j=1$, and an unnecessary $s_j=1$ with $z_j=0$
can be changed to zero without affecting reconstruction. This argument assumes
the support flags have no other decoder role. The $R_j$ are part of the
admissible problem, not arbitrary numbers chosen merely to simplify a solver.

| Representation family | Optimization structure |
|---|---|
| Fixed linear synthesis $Dz$ | Mixed-integer convex quadratic/conic reconstruction constraints with bounded codes. |
| Finite family of domains | Enumerate domains or introduce discrete selection variables, then optimize each code. |
| Generated basis with discrete operations | Joint discrete domain/support search; special structure can eliminate coefficient optimization. |
| Continuously parameterized domain and coefficients | Generally nonconvex mixed-integer nonlinear optimization; even $Dz$ is bilinear when both $D$ and $z$ vary. |

A solver needs an explicit finite mathematical representation of the allowed
family. The formulation does not make arbitrary unknown reconstruction programs
automatically accessible to one generic MINLP solver.

Spatial branch-and-bound is an established approach for factorable nonconvex
MINLPs, as described by [Vigerske and Gleixner](https://doi.org/10.1080/10556788.2017.1335312)
and the [SCIP documentation](https://www.scipopt.org/doc/html/index.php).
Fixed-design L0 regression also has dedicated bound-based methods, for example
[Hazimeh, Mazumder and Saab](https://arxiv.org/abs/2004.06152).
Those algorithms' guarantees do not automatically transfer to a new joint
transform-learning formulation. No external MINLP solver was installed or run
for the prototype below.

The [structural-cue follow-up](sparse_cues.md) tests greedy proposals, beam search,
and geometry-guided certified pruning against this oracle, then derives a direct
construction in a separate continuous-angle family.

## How a global search certifies its answer

A feasible representation with $U$ coefficients supplies an upper bound on
the optimum. Relaxations, exhaustive coverage or other valid arguments supply
a lower bound $L$. A global claim requires matching bounds, using the integer
nature of support count where appropriate.

Equivalently, solve the feasibility problem with $\sum_j s_j\leq k$:
a feasible solution at $k$, together with proven infeasibility at $k-1$,
certifies the minimum. Monotonicity makes all smaller budgets infeasible too.

A timeout, unsuccessful local search or failure to find a code is **not**
infeasibility. A numerical solver result must report its tolerances, feasible
witness, lower bound, optimality gap and termination status. Numerical optimality
to a tolerance is distinct from an exact rational certificate.

## Completed exact prototype

The first run searches a deliberately small family of **generated bases**,
specified independently of the target sequences:

- Sequence length: **8 samples**.
- Each primitive acts on a coordinate pair with rotation cosine **3/5** and
  sine **+4/5 or −4/5**.
- There are **56 primitive instructions**: 28 pairs × two signs.
- A domain is a word of zero, one or two such instructions.
- There are **1 + 56 + 56² = 3,193 descriptions**. Different descriptions can
  yield the same matrix; we do not claim 3,193 distinct bases.
- Synthesis is $Q_\theta=G_{\text{last}}\cdots G_{\text{first}}$.
  Analysis applies the inverse operations in reverse order.
- The coefficient alphabet is fixed at multiples of **1/625 in [−8,8]**:
  10,001 symbols. All comparisons use Python rational arithmetic.
- The domain-description convention uses two bits for word length and six bits
  per instruction, at most **14 bits**. These are transform-description bits
  only; coefficient and support information are separate.
- We test seven sequences, three maximum depths, and five NMSE limits:
  **0, 0.001, 0.01, 0.1 and 0.25**. This gives **105 certified optima**.

All inputs lie on the 1/25 lattice and have norm at most 8. Applying at most two
inverse rational rotations puts every exact analysis coefficient on the 1/625
lattice. Orthogonality bounds its magnitude by the input norm, so every candidate
projection belongs to the declared coefficient alphabet. The implementation
checks these facts and rejects inputs that violate them.

### Why the optimization is exact

For each candidate $Q_\theta$, calculate $a=Q_\theta^\top x$.
By orthogonality, the best error with at most $k$ retained coefficients is

$$
\rho_\theta(k)=\sum_{j=k+1}^N |a|_{(j)}^2,
$$

where magnitudes are sorted from largest to smallest. The optimal retained
values are the corresponding entries of $a$. They are admissible finite-
precision coefficients here, as checked above. Therefore this analytic step
solves the entire support/value subproblem, including every possible support.

We then exhaust **every permitted domain description** and form

$$
\rho_B(k)=\min_{\theta:\,\mathrm{depth}(\theta)\leq B}\rho_\theta(k).
$$

The first $k$ satisfying $\rho_B(k)\leq\eta\|x\|_2^2$ is globally optimal
within this family. Its reconstructed witness gives the upper bound; the
strictly larger minimum error at $k-1$ gives the matching lower bound.
There are no floating-point feasibility thresholds in this certificate.

### Initial results

Exact reconstruction minima:

| Sequence | Identity only | At most one rotation | At most two rotations |
|---|---:|---:|---:|
| Zero | 0 | 0 | 0 |
| Impulse | 1 | 1 | 1 |
| Rotated one-coefficient example | 2 | 1 | 1 |
| Rotated two-coefficient example | 4 | 3 | 2 |
| Dense rational example | 8 | 8 | 8 |
| Short harmonic cycle | 4 | 4 | 4 |
| Seeded noise-like rational example | 7 | 7 | 7 |

For the rotated two-coefficient example,

$$
x=(3/5,\ 12/25,\ 16/25,\ 2,\ 0,\ 0,\ 0,\ 0),
$$

the search finds a two-rotation domain with coefficients

$$
z=(1,\ 0,\ 0,\ 2,\ 0,\ 0,\ 0,\ 0).
$$

It reconstructs exactly. Across **all** allowed domains, the smallest squared
error achievable with at most one nonzero is

$$
\rho_2(1)=192196/390625>0.
$$

Thus a one-coefficient exact code is impossible in this family and the exact
minimum really is **two**. The generating code was not assumed to be optimal;
the search independently verified the lower bound.

The accuracy target also changes the answer: at NMSE ≤0.1 this same example
has a certified **one-coefficient** solution. The dense example, whose exact
minimum is eight, needs **six** coefficients at NMSE ≤0.01 with two rotations,
versus eight with identity alone.

The seven signals do not establish anything about speech or all audio. The
harmonic cycle's four-coefficient result also does not rule out a one-coefficient
solution in a different basis outside this family. This is a validation of the
optimization/certification mechanism, not the final admissible domain class.

## Scaling the investigation

The number of descriptions with $P$ primitives and maximum depth $d$ is

$$
\sum_{\ell=0}^d P^\ell.
$$

For the present 56 instructions, depths 2, 3 and 4 require respectively
3,193, 178,809 and 10,013,305 words before deduplication. Greater depth can also
require finer coefficient precision to retain the analytic elimination used
here; it cannot silently change between comparisons.

The next optimization experiments should:
1. Establish which admissible domain family captures our intended freedom,
   adaptation information and reconstruction access.
2. Increase dimension and domain complexity while keeping the precision rules
   explicit; measure runtime, examined/pruned candidates and optimality gaps.
3. Introduce branch-and-bound for families that outgrow enumeration, comparing
   its results with exhaustive exact answers on small instances.
4. Extend the domain parameterization and then apply the search to short audio.
   Continuous nonorthogonal or nonlinear families need appropriate bounds and
   numerical verification, not the orthogonal tail-energy shortcut.
5. Treat outputs as future training targets only with their scope and
   certification status retained.

No neural model was trained. The orthogonal family was selected because its
coefficient subproblem has an exact solution; it is not a requirement imposed
on the final representation system.

## Reproduce and inspect

```powershell
python run.py study --config configs/study_sparse_search.yaml
python -m pytest tests/test_sparse_search.py -q
```

- [Exact minimum-support results](sparse_search_results.csv).
- [Complete best-error curves for all support budgets](sparse_search_curves.csv).
- [Rational witnesses, smaller-support lower bounds and protocol](sparse_search_results.json).
- [Search implementation](../lrac/sparse_search.py) and [experiment driver](../lrac/study_sparse_search.py).

Fresh results go to the printed `runs/` directory, leaving the published tables
above intact. The old `tools/search_sparsest.py` command forwards to the same
runner. See [the codebase guide](codebase.md) for run metadata and settings.

The independent small validation test enumerates every coefficient pair in a
two-dimensional quantized example, across every permitted basis description,
and checks the analytic search against that full enumeration.

The SCIP paper was reviewed at the abstract/formulation and global-search
method level; its solver was not reproduced here. Its title and authors have
been added to [bibliography.md](../bibliography.md).
