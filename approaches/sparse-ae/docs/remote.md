# Parallel remote training and reconnectable monitoring

## Transfer and environment

Copy the whole sparse-ae folder, including sae/, docs/, and the entry scripts.
It is standalone: its training code does not import the rest of this repository.
Do not copy local path-bearing manifests or checkpoints as a new remote run.
Prepare remote manifests from the remote corpus/cache. Cross-system relocation
of an existing run's absolute paths is not implemented.

The commands below assume Linux, Python 3.11+, and a CUDA-compatible PyTorch
environment. Run them from the copied sparse-ae folder. Use an existing suitable
environment, or create a dedicated one:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-remote.txt
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
nvidia-smi
```

The CUDA check must return True before a GPU campaign. The requirements match the
repository's PyTorch 2.8 series. PESQ may require compiler tooling when a wheel is
unavailable. No remote environment has been configured by this change; SSH details
will be supplied separately.

## Automatic LibriSpeech setup

Each individual train_*.py script supports automatic corpus setup. You can run it
with no data argument when the usual KaggleHub cache is present:

```bash
python train_l1.py --output runs/l1_seed0 --epochs 100 --patience 8
```

Resolution order is:

1. Explicit --manifest-dir bypasses corpus discovery and reuses prepared manifests.
2. Explicit --root, or LIBRISPEECH_ROOT, selects a local corpus or archive directory.
3. A previous successful shared-cache resolution is reused if its splits exist.
4. Common local data folders and the KaggleHub dataset cache are searched.
5. KaggleHub is imported, installed if missing, and used to obtain the fallback
   dataset only if the required train/development splits were not found.

The default handle is pypiahmad/librispeech-asr-corpus/versions/1, matching the
existing repository experiment. --dataset OWNER/SLUG/versions/N overrides it.
Already-cached LibriSpeech mirrors can also be reused; data_source.json records
the actual source. Standard and nested split layouts are accepted. Cached
official per-split tar/tar.gz archives are extracted into the shared sparse-ae
cache, with safe paths and completion markers. Links/special files are rejected.

KaggleHub normally uses ~/.cache/kagglehub. If the remote cache is elsewhere, set:

```bash
export KAGGLEHUB_CACHE=/absolute/path/to/existing/kagglehub
# Alternatively point directly at the corpus:
export LIBRISPEECH_ROOT=/absolute/path/to/LibriSpeech
```

The scripts do not search every mounted filesystem. Use the explicit root or
cache environment variable for nonstandard locations. --offline prohibits
installation and downloads, while still allowing extraction of existing archives.
--cache-dir or SPARSE_AE_CACHE selects the shared setup/manifests/extraction cache;
its default is ~/.cache/sparse-ae. This is separate from the KaggleHub cache.

The KaggleHub package is installed automatically only when it is missing and
data is not already available. Torch, NumPy, SoundFile, and optional speech metric
dependencies belong in the environment created above. If Kaggle requires account
authentication, configure its supported credentials on the server. The scripts
do not ask for a token interactively in a detached session.

OS advisory locks serialize dataset acquisition, extraction, and shared manifest
preparation. Locks are released automatically if the owning process exits. A
parallel campaign prepares one manifest set before launching its model processes.
Independent trainers also reuse that shared setup, so starting seven scripts does
not intentionally download seven corpus copies. Third-party processes outside
this launcher are not coordinated by these locks.

Training defaults to train-clean-100, with dev-clean speakers divided into separate
validation and calibration groups. test-clean is optional and is never scored by
training. Use --train-split train-clean-360 for that cached split. --train-hours 10
selects a pilot; omit it to use all utterances in the requested training split.
One epoch still uses one two-second crop per selected training utterance.

## Start all seven models

```bash
python run_all.py start --output runs/ada6000 --parallel 7 --device cuda --batch-size 4 --epochs 100 --patience 8 --min-delta 0.0001
```

This returns after starting a detached supervisor. Data preparation and all training
logs persist under the campaign directory. Each model has an independent process,
optimizer, validation loss, stopping counter, and best checkpoint. All seven share
the selected CUDA device. Batch size 4 is per model. The default worker count is
0 and CPU thread count is 2 per model, limiting CPU oversubscription.

For a local corpus or prepared manifests:

```bash
python run_all.py start --output runs/ada6000 --root /datasets/LibriSpeech --parallel 7
python run_all.py start --output runs/another_campaign --manifest-dir /path/to/shared/manifests --parallel 7
```

Extra common trainer settings go after --. Architecture-specific script defaults
remain intact, including the different sparsity penalties:

```bash
python run_all.py start --output runs/pilot --parallel 7 --train-hours 10 --epochs 10 -- --speech-metrics --save-every 5
```

--methods l1 kl topk selects a subset. --parallel 3 runs at most three jobs and
queues the others. --foreground stays attached and is useful for diagnostics.
Use the detached default when you intend to disconnect.

## GPU memory and concurrency

The target RTX 6000 Ada has a stated 48 GB available for this campaign; actual
available memory is read from nvidia-smi on the server. Seven-way execution is
supported, but the final model/batch settings must be checked on that machine.
It does not imply seven times the throughput of sequential training.

By default, each job's PyTorch CUDA allocator is capped at 0.85 divided by the
number of simultaneous jobs. With seven jobs, this is about 12.1% per model,
roughly 5.8 GB of nominal 48-GB capacity. The remaining space accommodates process
contexts and other allocations. This is an allocator cap, not reserved memory or
a hard limit on all CUDA allocations. Other workloads can reduce usable memory.
--gpu-memory-fraction overrides the cap. An out-of-memory failure is recorded;
the launcher does not silently change batch sizes or the training protocol.

Start with the batch size 4 default. The status command includes GPU snapshots.
If jobs fail from concurrent memory pressure, resume unfinished jobs with reduced
concurrency, keeping their original training batch settings:

```bash
python run_all.py resume --output runs/ada6000 --parallel 3
```

The launcher starts independent processes, not distributed data-parallel replicas.
No MPS service or system-wide GPU setting is changed. launch-delay defaults to two
seconds to spread initialization peaks.

## Inspect progress after reconnecting

Activate the same environment, return to this folder, and run:

```bash
python run_all.py status --output runs/ada6000
python run_all.py status --output runs/ada6000 --watch
python run_all.py status --output runs/ada6000 --json
python run_all.py logs --output runs/ada6000
python run_all.py logs --output runs/ada6000 --method topk --lines 60
tail -f runs/ada6000/jobs/topk/train.log
```

The table shows model state, epoch, batch, current phase loss, best monitored
validation value, and last completed checkpoint epoch. It also reports process
liveness, update timestamps, errors, and a recent GPU snapshot. JSON adds checkpoint
paths, stopping counters, and per-process allocator measurements. Different model
loss definitions must not be ranked by their raw loss values.

Progress is written atomically at phase boundaries and approximately every five
seconds during training, validation, and threshold calibration. A single slow
operation can delay its heartbeat. Dataset discovery/indexing writes phase status
and setup logs; it does not have a per-file heartbeat. The supervisor records
state regularly while jobs run. Process liveness checks include process birth
identity so a reused PID is not mistaken for an old training job.

Stopping the status watcher or tail viewer does not stop training.

## Stop and resume

```bash
python run_all.py stop --output runs/ada6000
python run_all.py resume --output runs/ada6000
```

Stop requests interrupt only the supervisor's own child processes. On Linux it
first sends SIGINT, allowing the trainer to retain its last completed epoch.
After the configured stop-grace deadline, unresponsive children are terminated.
An incomplete epoch is repeated on resume. A stop during shared data setup is
handled when that setup operation returns. Already completed methods are skipped.
Failed/interrupted jobs with last.pt resume their checkpoint; jobs that failed
before a checkpoint get a new attempt directory. Prior logs and artifacts remain.
Running supervisors and still-live orphan trainers block duplicate launches.

Ordinary SSH disconnection does not terminate a detached POSIX session. A server
reboot, scheduler time limit, administrator cleanup, or a host policy that kills
all user processes can stop it. The launcher does not install an automatic boot
service. Reconnect and use resume after such a stop. Training source and data
fingerprints must match; copy all source before launching and do not edit it during
a campaign. A separately started individual trainer can use its usual --resume.

## Campaign layout

```text
runs/ada6000/
  campaign.json
  supervisor_process.json
  supervisor.log
  data_source.json
  jobs/
    topk/
      train.log
      attempt_01/
        progress.json
        best_weights.pt
        decoder_best.pt
        last.pt
        history.csv
        summary.json
    l1/ ...
    kl/ ...
    jumprelu/ ...
    gated/ ...
    batchtopk/ ...
    dense/ ...
```

These remain unquantized native-objective baselines. Remote orchestration does
not add quantization or operational bitrate measurement.

## Reference

[KaggleHub documentation](https://github.com/Kaggle/kagglehub) describes
dataset_download, versioned handles, cache reuse, cache configuration, and
authentication. The resolver avoids a network call when an appropriate existing
corpus/cache is found.
