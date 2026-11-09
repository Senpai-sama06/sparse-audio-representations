# Archived sparse-autoencoder results

This folder contains the saved development and held-out test LaTeX tables for
the completed three-model S-NL / D-NL / S-AF study. The corresponding
[model report](../../docs/sparse_autoencoder/sparse_autoencoder_models.tex) and
[bibliography](../../docs/sparse_autoencoder/sparse_autoencoder_models.bib)
live under the method's documentation folder.

The historical report also names a test-results PNG, CSV exports, frozen-run
JSON, paired-comparison JSON, and a detailed Markdown comparison. Those files
were already absent from `documentation/` before this reorganization. They have
not been reconstructed or treated as verified evidence during relocation.

To regenerate exports from the original completed local runs, use
`tools/export_model_results.py` for development/test tables and provenance,
then `tools/compare_models.py` for the held-out comparison and figure. Both
default to this folder. These commands require the original runs and manifests;
they are not needed to read the archived tables.
