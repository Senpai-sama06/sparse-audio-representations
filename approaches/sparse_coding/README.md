# Sparse coding

This package owns the sparse-coding studies and their codec implementation.
Published method documentation belongs in `docs/`; published result exports
belong in `reports/`, with metrics and provenance beside them.

`tools/export_l1_results.py` now defaults to `approaches/sparse_coding/reports/`.
It verifies the completed L1 run's recorded artifacts before copying selected
tables and metadata. Original audio, payloads, checkpoints, and intermediate
results stay in the configured run directory.
