# Structural cues and constructive alternatives to sparse-domain search

29 October 2026. The question is whether we can use the structure of the signal
and the representation operations to avoid most of the exhaustive search,
before deciding whether a neural predictor is useful.

We tested deterministic cues against the existing exact optimizer, derived a
pruning bound, constructed a case where greedy decisions fail, and examined a
separate continuous-angle construction and simple recurrence descriptions.
No neural model was trained.

## First correction: enumeration does not converge

The original optimizer exhausts a finite family. At eight samples and depth two,
it checks 3,193 rotation-word descriptions, so it terminates after that coverage.
It does not approach an optimum by gradient descent or gradual convergence.
The problem is exponential family growth with depth, not an observed failure
of an iterative convergence process.

Likewise, the absence of an obvious pattern in optimal basis matrices does not
establish that no useful cue exists. Equivalent bases, supports and operation
orders can obscure patterns. We should examine the decision geometry and
reconstructive behavior rather than require a unique matrix label.

## A cue that gives a rigorous bound

Let $\rho_k(a)$ be the squared error of the best at-most-k coefficient
approximation to a vector $a$: the sum of its discarded squared magnitudes.
Put $\rho_m(a)=0$ for $m\geq N$. Let $\kappa_\eta(a)$ be the smallest
$k$ with $\rho_k(a)\leq\eta\|a\|_2^2$.

**Lemma 1 — One rotation cannot improve required support by more than one.**
If $U$ is an orthogonal rotation acting on only two coordinates, then

$$
\rho_k(Ua)\geq\rho_{k+1}(a).
$$

**Proof.** Fix a retained support $S$ of size at most $k$ after rotation.
If it excludes the rotated pair, its captured energy is a sum of at most
$k$ original squared coordinates. If it includes the pair, its captured
energy is at most the energy of both original pair coordinates plus the
retained unaffected coordinates. That involves at most $k+1$ original
coordinates. In either case it cannot exceed the energy of the largest
$k+1$ original coordinates. Orthogonality preserves total energy; subtracting
captured energy gives the inequality. ∎

Repeating the argument, after at most $r$ remaining pair rotations,

$$
\rho_k(U_r\cdots U_1a)\geq\rho_{k+r}(a).
$$

Consequently, a valid lower bound on the final feasible support is

$$
L(a,r)=
\begin{cases}
0,&a=0\text{ or }\eta\geq1,\\
\max\{1,\kappa_\eta(a)-r\},&\text{otherwise}.
\end{cases}
$$

If the best already-found feasible code has $U$ coefficients and
$L(a,r)\geq U$, this branch cannot improve the primary objective and can be
discarded. This remains a valid relaxation bound when angles or coefficients
are further restricted. It is specific to pairwise orthogonal operations and
squared Euclidean reconstruction error, not a bound for every possible domain.

We also merge exactly equal coefficient states when reached with no more
operations than before. This is safe here because future operations and
Euclidean best-k error depend on that state, and a shallower path leaves at
least as much remaining depth. Equal coefficient states can have different
synthesis matrices, but their best-k Euclidean errors agree. This merging
argument would need to be re-established for a perceptual objective, different
history-dependent operation grammar, or other adaptation costs.

## Deterministic methods tested

Every method uses the **same** at-most-two rational-rotation family, coefficient
alphabet (multiples of 1/625 in [−8,8]), reconstruction tolerances and decoder
access as the existing exact optimizer.

| Method | Decision mechanism | Certification |
|---|---|---|
| Exhaustive | Visit all 3,193 descriptions; solve each coefficient subproblem exactly | Complete coverage |
| Ratio greedy | Construct immediate zero-producing rotations from exact coordinate ratios | Only if the witness meets the root lower bound |
| Tail-energy greedy | Try one-step actions; accept an improvement in required support or proximity to the next smaller support | Same root-bound condition |
| Beam, width 8 | Retain eight promising distinct states at each depth, allowing temporary worsening | No general guarantee; root-bound matches can certify individual results |
| Geometry-guided branch-and-bound | Use the proven support bound, best-first expansion and exact state dominance | Exhaust all potentially improving states or meet the root lower bound |

For the current inverse rotation,

$$
a'=(3a+4s b)/5,\qquad b'=(-4s a+3b)/5,\qquad s\in\{-1,1\},
$$

an exact zero requires $3b=4s a$ or $3a=-4s b$. The ratio method uses a value
hash table to construct matching pairs. This is a genuine arithmetic cue, but
it comes from the declared operation family; it is not an established acoustic
property of speech.

The greedy score first favors smaller feasible support, then smaller error
at one fewer coefficient, then current reconstruction error. It is a heuristic
cue, not a valid global lower bound.

## Benchmark and results

The suite contains **96 eight-sample diagnostic sequences**:

- Seven examples from the previous exact study.
- One deliberately constructed nonmonotone counterexample.
- Two analytic recurrence examples.
- 48 planted examples: 16 each with one, two and three generating nonzeros.
- 18 seeded dense rational sequences.
- 20 perturbed planted examples, adding coordinate noise in {−1/25,0,1/25}.

The generating code is not assumed optimal: the exhaustive optimizer determines
the reference answer. The seed is 20261029. Beam width and cues were specified
before running this suite; they were not fit to these examples.

Each sequence is tested at NMSE limits **0, 0.01 and 0.1**. There are 288
sequence/tolerance problems and 1,440 method comparisons. Every reported
fixed-family witness is verified with exact rational arithmetic.

| Method | Matches optimum: exact | Matches optimum: NMSE ≤0.01 | Matches optimum: NMSE ≤0.1 | Median candidate states, exact |
|---|---:|---:|---:|---:|
| Exhaustive | 100% | 100% | 100% | 3,193 |
| Ratio greedy | 87.5% | 49.0% | 33.3% | 2 |
| Tail-energy greedy | 87.5% | 78.1% | 91.7% | 113 |
| Beam, width 8 | 92.7% | 100% | 97.9% | 113 |
| Geometry-guided branch-and-bound | 100% | 100% | 100% | 113 |

For branch-and-bound, median candidate counts at the two approximate tolerances
are 93 and 72.5. The exact-case **mean** is 795.2 and the worst case still visits
3,193 candidates. Thus the median reduction is not a universal speed or
complexity guarantee. Candidate counts include duplicate proposals; separate
scored-state, primitive-update and pruning counts are exported.

A reference enumeration reapplies up to two operations from the original input
for each word; the new methods reuse intermediate states. That implementation
difference also reduces primitive work. The count comparison should not be
interpreted solely as the effect of one bound. Timing is diagnostic: the oracle
computes all budgets/targets together and its time is amortized across targets.

Overall hit rates also include cases where identity coordinates are already
optimal. Among the **63 exact-reconstruction cases where a better basis exists**,
ratio/tail greedy match 51, beam matches 56 and branch-and-bound matches all 63.

Post-hoc oracle agreement is distinct from a method's own certificate. For
example, at exact reconstruction, the ratio method matches 87.5% of answers
but can independently certify only 44.8% through the root bound. Branch-and-bound
certifies all 288 problems in this small experiment.

These are diagnostic synthetic results, not evidence of general audio
performance or learned acoustic features.

## A counterexample to immediate improvement

Take

$$
x=(-7/25,\ 24/25,\ 0,\ldots,0).
$$

It is produced from a one-coordinate vector by two same-sign rotations on the
first pair. Initially the best one-coefficient squared error is $49/625=0.0784$.
The first inverse rotation on the successful two-step path gives

$$
(-7/25,24/25)\longmapsto(3/5,4/5),
$$

whose best one-coefficient error is **0.36**, temporarily worse. The second
inverse rotation gives $(1,0)$, with error zero.

There is no immediate exact cancellation. No single permitted action strictly
improves the greedy support/tail score, so both greedy methods stop at two
coefficients for exact reconstruction and NMSE ≤0.01. Beam and branch-and-bound
find the true one-coefficient optimum. In the exact case, beam generates 225
states and branch-and-bound 115, versus exhaustive enumeration's 3,193.

This demonstrates why a local cue alone may be insufficient. A useful
construction or predictor may need to identify coordinated operations whose
benefit appears only after several steps.

## A direct optimal construction in a different family

There is a positive construction result if pair rotations may use **arbitrary
real angles**, with real-valued coefficients.

**Corollary 2 — Optimal support with a budget of continuous pair rotations.**
Let $q=\kappa_\eta(x)$, and allow at most $r$ arbitrary pairwise orthogonal
rotations. If $q=0$, the optimal support is zero. Otherwise it is

$$
k^*_{\mathrm{continuous}}=\max(1,q-r).
$$

**Proof.** Lemma 1 gives the lower bound. For the upper bound, retain the indices
of the largest $q$ original magnitudes. Merge $\min(r,q-1)+1$ of those
coordinates into one using successive plane rotations. For a pair $(a,b)$,
choose

$$
c=\frac{a}{\sqrt{a^2+b^2}},\qquad
s=\frac{b}{\sqrt{a^2+b^2}},
$$

so the analysis rotation maps it to $(\sqrt{a^2+b^2},0)$.
Retain the merged coordinate and the unmerged selected coordinates. Their total
energy equals the original largest-q energy, so their reconstruction error is
exactly $\rho_q(x)$, within tolerance. Their count equals the lower bound. ∎

This avoids combinatorial search: sort the coefficients and perform the
specified merges. The float64 demonstration checks its ideal error to absolute
squared-error tolerance 1e-12 across the suite; it is not an exact rational
implementation of arbitrary real angles.

**The family and information requirements have changed.** These angles are not
the previous primitives with fixed cosine 3/5 and sine ±4/5, and the coefficients need not lie on the
previous finite grid. Each merge introduces an adaptive angle plus pair indices.
If $r_{\rm used}$ merges are used, there are $q-r_{\rm used}$ retained scalar
values and $r_{\rm used}$ independent angle parameters: **q scalar values and
angle parameters in total**, before index/precision accounting.

This construction maximizes coefficient sparsity within that continuous-angle
family, but does not automatically shorten the complete representation. It
shows precisely how information can move from coefficients into the domain.
A useful compact construction must exploit structure in the domain parameters
themselves. This bookkeeping observation is not a general information-theoretic
lower bound over all possible signal descriptions.

## Cues deeper than the current rotation grammar

We also computed exact ranks of each sequence's 4×5 Hankel matrix and verified
three compact recurrence descriptions:

| Example | Hankel rank | Exact optimum in the fixed-angle rotation family |
|---|---:|---:|
| Geometric growth $x_n=2^n/25$ | 1 | 8 coefficients |
| Linear ramp $x_n=n/25$ | 2 | 6 coefficients |
| Harmonic cycle $(0,1,0,-1,\ldots)$ | 2 | 4 coefficients |
| Earlier rotated-two example | 4 | 2 coefficients |

The growth sequence is reconstructed from its initial value and ratio using
$x_{n+1}=2x_n$. The ramp uses two initial values and a shared second-difference
rule. The harmonic cycle uses two initial values and an oscillator-recurrence
parameter. All eight samples are checked after reconstruction.

These descriptions are **outside the fixed-angle optimizer's family** and are
not scored as if they used the same free side information. Their rule identifiers,
initial values, adaptive parameters, precision and stability must be accounted
for. They are verified examples of constructive structure, not a universal model
selector or a proof of shortest description. Low Hankel rank alone is not a
certificate of minimum support in another family.

The growth example is particularly informative: exhaustive search correctly
returns eight coefficients for its allowed rotations, while a simple recurrence
generates the sequence. The allowed building blocks determine which regularities
an optimizer can exploit. More search in an unsuitable family does not expose a
rule outside that family.

[Learning Fast Sparsifying Transforms — Rusu and Thompson](https://arxiv.org/abs/1611.08230)
is relevant method-level precedent for building transforms from structured
Givens-like operations. Its dictionary-learning/update guarantees are not this
experiment's global minimum-support guarantee.
[Sampling Signals With Finite Rate of Innovation — Vetterli, Marziliano and Blu](https://bigwww.epfl.ch/publications/vetterli0201.html)
is a relevant constructive-recovery direction for specific parametric signal
classes. Here that paper was abstract-screened; its full PDF could not be fetched,
and its algorithms or theorems were not reproduced.

## What this establishes and the next investigation

- There are useful deterministic cues in the current search geometry.
- A bound can preserve global optimality while avoiding many candidates.
- Greedy local improvement is insufficient; short-horizon coordination matters.
- Direct optimal constructions can exist for a specified family.
- A compact generative rule may exploit structure absent from the current family.
- None of this establishes a universal cheap construction or proves a neural
  predictor cannot learn useful decisions.

The next investigation should test **construction rules**, not start another
network training run: detect candidate periodicity/recurrence/low-rank or event
structure, infer a small parameter set, reconstruct with only that description,
and measure the actual residual under fixed precision and adaptation budgets.
Any unexplained residual remains part of the reconstruction problem. Test
noise and numerical sensitivity explicitly, then compare these constructions
with certified small searches and eventually development speech.

The original dream remains an unknown economical representation; these are
candidate mechanisms to investigate, not required semantic slots or a requirement
that all future domains be orthogonal.

## Reproduce and inspect

```powershell
python run.py study --config configs/study_sparse_cues.yaml
python -m pytest tests/test_sparse_cues.py -q
```

- [All method comparisons](sparse_cues_results.csv).
- [Group/tolerance summaries](sparse_cues_summary.csv).
- [Measured root cues and Hankel ranks](sparse_cues_features.csv).
- [Separate continuous-angle construction results](sparse_cues_continuous.csv).
- [Exact witnesses, reconstruction rules, provenance and scope](sparse_cues_results.json).
- [Core algorithms](../lrac/sparse_cues.py) and [driver](../lrac/study_sparse_cues.py).

Fresh results go to the printed `runs/` directory; the committed results and
their original provenance remain unchanged. `tools/analyze_sparse_cues.py` is
now a compatibility entry point to the same runner.

The two reviewed papers were added as one-line entries to
[bibliography.md](../bibliography.md). The bound and construction proofs above
are elementary derivations for this investigation, with no novelty claim.
