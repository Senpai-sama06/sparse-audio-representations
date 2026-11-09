# Sparse-affine model interpretation

This package contains channel, pitch, and module interpretation analyses of the
historical sparse-affine autoencoder. The model study and archived result tables
belong to [representations](../representations/README.md).

The repository scripts `tools/analyze_channels.py`, `tools/analyze_pitch.py`,
and `tools/analyze_modules.py` now default to
`approaches/interpretation/reports/` for published analysis outputs. Intermediate
arrays, protocol records, and progress remain with the original model run.
Limited channel probes and reduced pitch cohorts keep their run-local outputs.

Channel and pitch checks look for the frozen model record under
`approaches/representations/reports/sparse_autoencoder/sparse_autoencoder_test_runs.json`.
That historical JSON is absent from this checkout. A real frozen record and the
original model/data assets are needed for analyses that require it.
