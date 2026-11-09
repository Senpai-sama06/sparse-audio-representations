# Documentation and report map

Method documents and published reports live with their owning approach.
This folder retains shared reporting instructions and the shared literature library.

| Study | Method entry point |
| --- | --- |
| Parametric fitting, shared representations, and symbolic IMF discovery | [Parametric reports](../approaches/parametric/README.md) |
| PitchFlower RVQ / GS | [PitchFlower](../approaches/pitchflower/README.md) |
| Historical sparse-autoencoder baselines and adaptive STFT dictionary | [Learned representations](../approaches/representations/README.md) |
| Sparsest-representation theory and deterministic search | [Sparse search](../approaches/sparse_search/README.md) |
| Sparse-affine channel, pitch, and module interpretation | [Interpretation](../approaches/interpretation/README.md) |
| Sparse coding / L1 exports | [Sparse coding](../approaches/sparse_coding/README.md) |
| New sparse-autoencoder rate-distortion benchmark (planning) | [Sparse-AE plan](../approaches/sparse-ae/README.md) |

## Relocated files

The following files were moved from this folder. Supporting JSON records are
included even when they were ignored by Git. The adaptive-STFT standalone HTML,
its metrics and verification record, and both autoencoder result tables retain
their original bytes. Documentation links and LaTeX input paths were rebased.

| Previous path | Current location |
| --- | --- |
| `documentation/sparsest_representation.tex` | [sparsest_representation.tex](../approaches/sparse_search/docs/sparsest_representation.tex) |
| `documentation/sparse_autoencoder_models.tex` | [sparse_autoencoder_models.tex](../approaches/representations/docs/sparse_autoencoder/sparse_autoencoder_models.tex) |
| `documentation/sparse_autoencoder_models.bib` | [sparse_autoencoder_models.bib](../approaches/representations/docs/sparse_autoencoder/sparse_autoencoder_models.bib) |
| `documentation/sparse_autoencoder_results.tex` | [sparse_autoencoder_results.tex](../approaches/representations/reports/sparse_autoencoder/sparse_autoencoder_results.tex) |
| `documentation/sparse_autoencoder_test_results.tex` | [sparse_autoencoder_test_results.tex](../approaches/representations/reports/sparse_autoencoder/sparse_autoencoder_test_results.tex) |
| `documentation/adaptive_stft_dictionary_design.html` | [adaptive_stft_dictionary_design.html](../approaches/representations/docs/adaptive_stft_dictionary/adaptive_stft_dictionary_design.html) |
| `documentation/adaptive_stft_dictionary_training_run.html` | [adaptive_stft_dictionary_training_run.html](../approaches/representations/docs/adaptive_stft_dictionary/adaptive_stft_dictionary_training_run.html) |
| `documentation/adaptive_stft_dictionary_reconstruction_diagnosis.md` | [adaptive_stft_dictionary_reconstruction_diagnosis.md](../approaches/representations/docs/adaptive_stft_dictionary/adaptive_stft_dictionary_reconstruction_diagnosis.md) |
| `documentation/reports/adaptive_stft_dictionary_testcases.html` | [adaptive_stft_dictionary_testcases.html](../approaches/representations/reports/adaptive_stft_dictionary/adaptive_stft_dictionary_testcases.html) |
| `documentation/reports/adaptive_stft_dictionary_metrics.json` | [adaptive_stft_dictionary_metrics.json](../approaches/representations/reports/adaptive_stft_dictionary/adaptive_stft_dictionary_metrics.json) |
| `documentation/reports/adaptive_stft_dictionary_verification.json` | [adaptive_stft_dictionary_verification.json](../approaches/representations/reports/adaptive_stft_dictionary/adaptive_stft_dictionary_verification.json) |
| `documentation/reports/pitchflower_best_testcases.html` | [PitchFlower standalone report](../approaches/pitchflower/reports/pitchflower_best_testcases.html) |
| `documentation/reports/pitchflower_best_metrics.json` | [PitchFlower metrics](../approaches/pitchflower/reports/pitchflower_best_metrics.json) |
| `documentation/reports/pitchflower_best_verification.json` | [PitchFlower verification](../approaches/pitchflower/reports/pitchflower_best_verification.json) |

The separate `runs/parametric_reports/` archive has also been distributed into
its study-specific `report/` folders. Its complete per-file map and integrity
checks are in the [parametric relocation record](../approaches/parametric/report/relocation.json).

## Shared material that stays here

- [Report-writing instructions](REPORTS_README.md) apply across methods.
- `literature/` contains downloaded source papers used across studies, including
  priority-reading and sparsifying-transform collections. These are references,
  not generated method reports; existing literature-review links continue to use
  this shared location.

## Publication and provenance

Keep documentation under `approaches/<owner>/docs/` and published artifacts
under `approaches/<owner>/reports/` (or the explicit `report/` convention used
by parametric studies). If an approach contains several studies,
use a study-named subfolder in each. Keep the metrics and verification beside
their report, and link everything from the owning README.

Raw run products, datasets, checkpoints, and live progress records remain in
their configured run directories. Historical source/build hashes in published
results describe the original run; relocation does not rewrite those identities.

Some archived autoencoder assets named by its LaTeX report were already missing
before relocation. Their availability is documented in the
[archived-results README](../approaches/representations/reports/sparse_autoencoder/README.md).
