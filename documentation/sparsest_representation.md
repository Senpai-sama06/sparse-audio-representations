# Existence of a sparsest possible representation

Formal note, 29 October 2026. Read the [complete theorem-and-proof PDF](sparsest_representation.pdf)
or its [LaTeX source](sparsest_representation.tex). The PDF contains all proofs;
this page gives the principal arguments and a result map.

## What is proved?

**A sparsest representation exists whenever the declared family contains at
least one feasible finite code.** If every sequence is feasible in that family,
this conclusion applies to every sequence. Separately, every nonzero sequence
has an exact one-sparse representation if an arbitrary sequence-dependent
algebraic basis is allowed. The latter conclusion includes infinite sequences
under the stated basis-extension assumptions.

These are existence results, not proof of a unique useful domain, economical
basis specification, an efficient search algorithm, or optimality of our
autoencoder.

## Main existence theorem: an attained minimum

Let a permitted representation be $r=(\theta,z)$, where $z$ has finitely many
scalar coordinates, $\theta$ describes allowed adaptation, and $G$ is the
declared reconstruction rule. Define the feasible set

$$
\mathcal F_\varepsilon(x)
=\{(\theta,z):d(x,G(\theta,z))\leq\varepsilon,\ 
                 (\theta,z)\text{ is admissible}\}.
$$

**Lemma 1 — Integer-valued attainment.** If $S\neq\varnothing$ and
$c:S\to\mathbb N_0$, then $c$ attains a minimum on $S$.

**Proof.** The image $c(S)$ is a nonempty subset of the nonnegative integers.
By well-ordering it contains a least element $k_*$. Because $k_*$ belongs to
the image, some $s_*\in S$ has $c(s_*)=k_*$. Thus it is attained. ∎

**Theorem 2 — Sparsest feasible representation.** If
$\mathcal F_\varepsilon(x)\neq\varnothing$, there is a feasible
$(\theta_*,z_*)$ with

$$
\|z_*\|_0
=\min_{(\theta,z)\in\mathcal F_\varepsilon(x)}\|z\|_0.
$$

**Proof.** Apply Lemma 1 with $S=\mathcal F_\varepsilon(x)$ and
$c(\theta,z)=\|z\|_0$. ∎

Compactness is **not required** for this support-count result. Actual feasibility
is required: an error infimum of zero need not be an attained exact reconstruction.

## Constructive result for every nonzero finite sequence

**Lemma 5 — Orthonormal completion.** Any unit vector $u\in\mathbb F^N$, with
$\mathbb F=\mathbb R$ or $\mathbb C$, can be completed to an orthonormal basis.

**Proof.** Apply Gram–Schmidt to the spanning list
$(u,e_1,\ldots,e_N)$, discarding zero residuals. Its first output is $u$;
exactly $N$ orthonormal outputs span the space. ∎

**Corollary 6 — Exact one-sparse minimum.** Given $x\neq0$, take
$u=x/\|x\|_2$ and complete it to the columns of a unitary synthesis matrix
$Q_x$. Then

$$
Q_x^*x=\|x\|_2e_1,\qquad x=Q_x(\|x\|_2e_1).
$$

One nonzero therefore suffices. Zero nonzeros cannot reconstruct a nonzero
vector under linear synthesis, so **one is globally minimal across all linear
bases**, including all orthonormal bases. For the zero sequence, the minimum is
zero.

The basis here depends on the target sequence and contains its direction.
The inverse must know that basis; one coefficient alone does not specify the
waveform.

## Complete result map

| Result in PDF | Statement and scope |
|---|---|
| Lemma 1, Theorem 2 | Every nonempty feasible finite-code family attains its minimum support. |
| Lemma 3, Theorem 4 | Every nonzero finite or infinite sequence is one-sparse in an adapted algebraic basis; general infinite-dimensional basis extension uses Zorn's lemma/choice. |
| Lemma 5, Corollary 6 | Finite-dimensional orthonormal construction by Gram–Schmidt, without infinite-dimensional choice. |
| Corollary 7 | Over unrestricted orthonormal bases, norm-error tolerance gives minimum zero if zero is accurate enough; otherwise one. |
| Theorem 8 | Finite-energy infinite sequences admit one-sparse expansions in an adapted countable complete orthonormal Hilbert basis. |
| Proposition 9 | Unrestricted sample-specific affine translation makes the minimum zero. |
| Lemma 10, Theorem 11 | A fixed finite dictionary has an attained sparsest code whenever its synthesis range meets the error tolerance. A spanning dictionary represents every finite sequence exactly. |
| Corollary 12 | Best-k tail energies give a certified optimum over a declared finite orthonormal library. |
| Theorem 13 | The spark condition gives sufficient uniqueness within a fixed dictionary, not optimality of the dictionary itself. |
| Lemma 14, Theorem 15 | A shared Lipschitz decoder with fewer than N nonzeros cannot exactly cover an open subset of N-dimensional waveform space. |
| Example 2 | An infinite sequence can require infinitely many exact canonical coefficients while admitting a finite approximate code or a one-sparse adapted basis. |
| Theorem 16 | A finite-alphabet counting bound accounts for coefficient precision, support choices and model identities. |
| Proposition 17 | A shared finite-bit decoder cannot exactly describe every infinite binary sequence. |

The paper makes the quantifiers explicit: “for each sequence there exists a
basis” does not imply “one fixed basis makes every sequence equally sparse.”
Algebraic existence for arbitrary infinite sequences also does not provide a
finite-bit, stable reconstruction system.

## Implication for the research

The existence question has a rigorous answer under the above definitions.
The remaining task is to specify an admissible family that allows genuinely
unknown, adaptive domains while making their reconstruction information explicit,
then find and certify minimum support **within that family**.

A feasible learned code gives an upper bound on the optimum. To claim that no
sparser admissible representation exists, we also need a matching lower bound
covering all allowed transform choices and codes.

The cited papers—Donoho and Elad, Cohen and colleagues, and Rissanen—are already
recorded as one-line entries in [bibliography.md](../bibliography.md).
The elementary proofs in the note are not claimed as new mathematical results.

## Build

From the documentation directory:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error sparsest_representation.tex
```
