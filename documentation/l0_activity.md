# Separate exhaustive L0 activity

Status: launched 29 October 2026 and running independently of the L1 study.

Solve min ||z||_0 subject to ||x - Phi z||^2 <= 0.01 ||x||^2, with the same
fixed Phi = [I, orthonormal IDCT-II] as the L1 experiment. The input is the full
two-second, 32,000-sample measurement clip `librispeech:3081-166546-0005` from
the saved L1 cohort. There are 64,000 candidate atoms. No signal shortening,
dictionary restriction, pruning, learned proposals, greedy search or time limit
is applied.

## What brute force means here

Enumerate supports in increasing cardinality, lexicographically within each size.
For every support, construct its columns and fit its continuous coefficients by
SVD least squares. This computes the minimum reconstruction error attainable on
that support; enumerating real coefficient values individually would be infinite.
Rank-deficient supports are also evaluated. No dense 32,000 x 64,000 matrix is
stored. Only the current support matrix is constructed.

The first feasible support after all smaller cardinalities have been exhausted
establishes a minimum cardinality in ideal exact arithmetic. Exhausting the other
supports of that same cardinality is unnecessary for minimum support, so the run
finishes when it finds its first clearly feasible witness.

An initial identity representation gives a feasible **upper bound of 31,912**
nonzero coefficients; it is saved in `initial_witness.npz`. This does not skip
supports. Exhausting smaller cardinalities improves the **lower bound**, while
finding a feasible representation improves the upper bound. Exhaustive search
does not immediately produce a useful tight upper bound.

## Numerical scope

The implementation uses float64 SVD least squares, not exact or interval arithmetic.
It therefore reports numerical evidence rather than a formal mathematical
certificate. Residuals within 1e-12 times source energy of the error threshold
are flagged as ambiguous. Their smallest cardinality remains unresolved in the
displayed lower bound. This guard is not a rigorous floating-point error enclosure.

The result concerns this fixed dictionary, this clip and this error tolerance.
It cannot establish optimality over every possible transform/domain or minimum
encoding bitrate. Stored coefficient values are continuous numerical values;
quantization and operational bitrate are separate questions.

## Live run and progress

Run directory:
`runs/l0_librispeech_20261029T120949726Z`

At the initial inspection, support sizes zero and one were complete, the search
was processing size two, and no ambiguous supports had been encountered. The
numerical lower bound was 2 and the feasible identity upper bound was 31,912.
Size two contains **2,047,968,000** combinations; larger sizes grow much faster.
No completion date is promised.

Read live progress from the project root:

```powershell
Get-Content runs/l0_librispeech_20261029T120949726Z/progress.json
```

The progress record contains the actual Python worker PID, current support size,
total tested count, next support, bounds, ambiguous count and elapsed active time.
The launcher may have a different PID because the virtual environment executable
starts a Python child. Logs and launch metadata are beside the run directory.

## Start and resume

```powershell
# Start another independent exhaustive activity, detached and hidden.
.\tools\start_l0.ps1

# Resume this activity only after its previous worker has stopped.
.\tools\start_l0.ps1 -ResumeRun runs/l0_librispeech_20261029T120949726Z
```

The running process has no time/support cap. On 30 October it was moved from one
CPU thread to **32 worker processes**, one numerical-library thread each, on this
24-core / 32-logical-CPU workstation. It continues
while the computer and process remain running. After termination or reboot, the
resume command continues from the saved state; automatic reboot scheduling has
not been installed.

Checkpoints are atomic JSON files written every five seconds and at cardinality
boundaries. A sudden process termination may replay work since the last saved
state. Restarting does not walk through earlier combinations to reach its cursor.
An OS file lock prevents two workers from concurrently searching the same run.
Resume verifies frozen source and implementation hashes, so preserve this code
version for long-lived runs.

## Parallel execution update

The numerical support fitter and dictionary are unchanged. `lrac/l0_parallel.py`
assigns disjoint chunks of 512 consecutive supports to workers. Up to twice the
worker count of chunks is in flight. The coordinator commits their results in
lexicographic order; only this contiguous completed prefix advances the saved
cursor and counts. Cardinalities do not overlap. A crash can replay uncommitted
chunks, but cannot skip an unfinished support or claim a premature lower bound.
Every candidate still receives an independent SVD least-squares fit.

The serial worker was stopped and its latest atomic checkpoint preserved as
`serial_state_before_parallel.json` and `serial_progress_before_parallel.json`.
The same source and search protocol were retained, and the search resumed at that
cursor. The original `l0.py` and `linear.py` source hashes were preserved. Parallel
execution also records its own implementation hash in `parallel_execution.json`.

`progress.json` now includes the coordinator PID, worker PIDs, worker count and
chunk size. These values are separate from the virtual-environment launcher PID.
To choose a different worker count after stopping the existing activity:

```powershell
.\tools\start_l0.ps1 -ResumeRun runs/l0_librispeech_20261029T120949726Z -Workers 32
```

Zero workers means all logical CPUs. For a new run, the `workers` keyword in its
study YAML selects the count (default zero); the launcher's `-Workers` option
applies to resumes. No GPU is used. Completion time remains unrestricted.

Additional tests compare serial and parallel minimum-support results and restart
counts, and independently verify chunk traversal against all small combinations.

During an approximately 66-second observation after startup, the 32 workers used
93.4% of the machine's logical CPU capacity and committed about 21,600 supports
per second. Utilization is the sum of worker CPU seconds divided by elapsed wall
seconds times 32; throughput uses committed checkpoint counters and active search
time. These measurements were taken while project checks ran concurrently, and
are observations rather than guaranteed rates at larger support sizes.

Files include `source.npy`, `search_protocol.json`, `search_state.json`,
`initial_witness.npz` and, on success, `witness.npz`. The authoritative search
completion flag is `search_state.json: status=complete`, including after a manual
resume. The generic study runner also writes `completion.json` if the original
study invocation finishes normally.

## Code and verification

- `lrac/l0.py`: support traversal, independent support fits, checkpointing and locks.
- `lrac/study_l0.py`: source selection, frozen protocol and initial witness.
- `configs/study_l0_librispeech.yaml`: declared settings.
- `tools/start_l0.ps1` / `tools/resume_l0.py`: launch and resume commands.

All 105 project tests pass. The added checks verify complete combination traversal,
the known two-coordinate minimum for an independent tiny example, and resuming
from the middle of a cardinality without skipping supports. Full speech search
remains in progress; no completed speech L0 optimum is claimed.
