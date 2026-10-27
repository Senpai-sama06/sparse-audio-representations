# First model: sparse orthonormal block dictionary

The current starting point uses a transparent alternating algorithm, rather than
a neural encoder or an l1 surrogate for the coefficient budget.

## Architecture

Split each mono waveform into nonoverlapping blocks of 512 samples (32 ms at
16 kHz). The learned matrix D has shape 512 x 512 with orthonormal columns.
Four blocks of 128 columns are candidate feature dictionaries. These are
candidate groupings; their acoustic meaning has not been established.

For full signal blocks X, analysis is C = X D and synthesis is A D^T (frames are
rows in the implementation). Selection retains the largest-magnitude coefficients
across the **whole recording**, not a fixed count per frame. A partial final block
uses a fixed orthonormal DCT of its actual length. Its coefficients are included
in selection and total counts, and reported separately from learned atom usage.
Batch padding contributes no coefficients and no error.

The implementation uses float64 for the dictionary and its transforms. Store
precision explicitly when interpreting scalar counts as representation complexity.
The shared dictionary contains 262,144 values; it must not be treated as free if
one later adapts a separate dictionary for each recording.

## Training

Each minibatch update performs:

1. Transform signals and select at most ceil(keep_fraction * valid_samples)
   nonzero coefficients per recording.
2. Hold those coefficients fixed and form X^T A across full blocks in the batch.
3. Compute U Sigma V^T = X^T A and replace D with U V^T.
4. Recompute sparse codes and log reconstruction error.

The fixed-code SVD update and recoding cannot increase squared error on that
batch, apart from roundoff. Different batches and validation data can get worse.
CUDA updates explicitly use the QR-based `gesvd` driver to handle the
rank-deficient cross-covariances common in small batches without first attempting
Jacobi iterations and an implicit fallback.
The outer learning problem is nonconvex; the routine is not a globally optimal
dictionary search. It uses no optimizer, learning rate, or gradient accumulation.
`batch_size` sets the update batch directly. Tail transforms remain fixed.

Training uses a fixed support budget, waveform MSE, and validation-only checkpoint
selection. `initial.pt` preserves the starting dictionary, `best.pt` is selected
by validation reconstruction, and `last.pt` contains the final update. A best
checkpoint at step zero is a valid result if learning does not improve validation.

## Run it

```powershell
python run.py experiment --config configs/smoke_block_orthogonal.yaml
python run.py experiment --config configs/synthetic_block_orthogonal.yaml
```

The first command is an eight-update execution check. The second is a 200-update
synthetic screening run with 2,048 training, 128 validation, and 128 examples per
test regime. Both use existing data audits, logging, analysis, and reports. Final
test sets are left for separate evaluation. A speech configuration is available
as `configs/librispeech_block_orthogonal_1h.yaml` after LibriSpeech preparation.

## Minimum-support inference

For a fixed orthonormal transform, discarded coefficient energy equals waveform
squared error. Sorting all coefficient energies therefore gives the minimum
number needed for a specified waveform NMSE. Evaluation saves predicted NMSE,
actual NMSE, the error with one fewer coefficient, and numerical verification.

```powershell
python run.py evaluate --run runs/YOUR_RUN --split validation --nmse-targets 0.1,0.01,0.001
python run.py evaluate --run runs/YOUR_RUN --split validation --nmse-targets 0.1,0.01,0.001 --checkpoint initial.pt
python run.py evaluate --run runs/YOUR_RUN --split validation --nmse-targets 0.1,0.01,0.001 --checkpoint last.pt
python run.py report --runs runs/YOUR_RUN --output runs/YOUR_RUN/report
```

These tolerances correspond to ordinary SNR targets of 10, 20, and 30 dB.
`--budgets` instead specifies absolute whole-recording counts for this model;
budget zero retains all coefficients. Without either flag, evaluation uses the
configured fraction. `sample_budget` takes precedence over a fraction; training
does not accept `target_nmse` as its objective. Choose a fraction or fixed budget
for fitting, then freeze the dictionary and evaluate error tolerances.

The support result is conditional on this dictionary and waveform squared error.
It does not certify the best dictionary, the minimum number of feature families,
or a minimum support under STOI/PESQ/ViSQOL constraints. Active coefficient count,
distinct learned atom count, active learned feature-dictionary count, and tail
coefficient count are reported separately. Current evaluation includes MSE, MAE,
NMSE, log NMSE, SNR, SI-SDR, magnitude/complex spectral errors, and event/band
diagnostics. Speech perceptual backends remain additional work; they are not
silently substituted by the spectral scores.

See the [first screening results](block_orthogonal_results.md) for the executed
200-update pilot, including the negative comparison against its initial DCT bank.
