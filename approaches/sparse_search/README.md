# Sparse representation theory and search

[Sparsest-representation foundations](docs/sparsest_representation.tex) contains
the existence results, proofs, and limits of the declared finite-code families.
It distinguishes minimum support from minimum encoded bitrate and does not
claim a universally sparsest audio representation.

Build from `approaches/sparse_search/docs/`:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error sparsest_representation.tex
```

The theory source has no external figure or bibliography inputs. Related
deterministic search and rotation studies live in this package. Large search
records remain in their configured `runs/` directories. Published documentation
belongs in `docs/`; future published result compilations belong in `reports/`.
