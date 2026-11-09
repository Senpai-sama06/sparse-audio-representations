"""Detached single-GPU campaign supervisor and reconnectable status commands."""
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from . import METHODS
from .runtime import atomic_json, file_lock, process_alive, process_identity, read_json, utc_now

ENTRY_DIR = Path(__file__).resolve().parents[1]
TERMINAL_JOB_STATES = {"completed", "failed", "interrupted", "not_started"}


def gpu_snapshot():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        return dict(available=result.returncode == 0, rows=result.stdout.strip().splitlines(),
                    updated_at=utc_now())
    except (OSError, subprocess.TimeoutExpired):
        return dict(available=False, rows=[], updated_at=utc_now())


def detached_process(command, log, cwd):
    kwargs = dict(stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                  cwd=cwd, close_fds=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
        kwargs["startupinfo"] = startup
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(command, **kwargs)


def prepare_start(config):
    output = Path(config["output"]).resolve()
    with file_lock(output.parent / (output.name + ".create.lock"), timeout=0):
        return _prepare_start(config, output)


def _prepare_start(config, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Campaign output is not empty: {output}; use the resume command")
    output.mkdir(parents=True, exist_ok=True)
    config["output"] = str(output)
    jobs = {method: dict(method=method, state="queued", attempts=[], output=None, log=str(output / "jobs" / method / "train.log"))
            for method in config["methods"]}
    state = dict(format_version=1, config=config, state="queued", created_at=utc_now(),
                 updated_at=utc_now(), supervisor=None, jobs=jobs)
    atomic_json(output / "campaign.json", state)
    return output


def launch(output, foreground=False, parallel=None):
    output = Path(output).resolve()
    with file_lock(output / ".launch.lock", timeout=0):
        state = read_json(output / "campaign.json")
        if not state:
            raise ValueError(f"No campaign at {output}")
        if process_alive(state.get("supervisor")):
            raise ValueError("This campaign supervisor is already running")
        # A pending detached supervisor may not yet have opened campaign.json.
        pending = read_json(output / "supervisor_process.json", {})
        if process_alive(pending.get("process")):
            raise ValueError("A detached supervisor is already starting")
        for job in state["jobs"].values():
            for attempt in job["attempts"]:
                if process_alive(attempt.get("process")):
                    raise ValueError("An orphaned trainer is still running; inspect status before starting another supervisor")
        if parallel is not None:
            if parallel < 1:
                raise ValueError("parallel must be positive")
            old = min(state["config"]["parallel"], len(state["config"]["methods"]))
            state["config"]["parallel"] = parallel
            if parallel > old:
                state["config"]["gpu_memory_fraction"] = min(
                    state["config"]["gpu_memory_fraction"], .85 / min(parallel, len(state["config"]["methods"])))
            atomic_json(output / "campaign.json", state)
        (output / "stop.request").unlink(missing_ok=True)
        if not foreground:
            command = [sys.executable, str(ENTRY_DIR / "run_all.py"), "_supervise", "--output", str(output)]
            with open(output / "supervisor.log", "a", encoding="utf-8") as log:
                process = detached_process(command, log, Path.cwd())
            identity = process_identity(process.pid)
            atomic_json(output / "supervisor_process.json", dict(process=identity, started_at=utc_now()))
            print(f"Started detached campaign: {output}\nSupervisor PID: {process.pid}\n"
                  f"Status: python {ENTRY_DIR / 'run_all.py'} status --output {output}\n"
                  f"Log: {output / 'supervisor.log'}", flush=True)
            return 0
        atomic_json(output / "supervisor_process.json", dict(process=process_identity(), started_at=utc_now()))
    return supervise(output)


def job_command(config, method, output, checkpoint=None):
    command = [config["python"], "-u", str(ENTRY_DIR / f"train_{method}.py")]
    if checkpoint:
        command += ["--resume", str(checkpoint)]
    else:
        command += ["--manifest-dir", config["manifest_dir"], "--output", str(output),
                    "--seed", str(config["seed"]), "--batch-size", str(config["batch_size"])]
        command += config["trainer_args"]
    command += ["--epochs", str(config["epochs"]), "--patience", str(config["patience"]),
                "--min-delta", str(config["min_delta"]), "--device", config["device"],
                "--workers", str(config["workers"]), "--threads", str(config["threads"]),
                "--gpu-memory-fraction", str(config["gpu_memory_fraction"])]
    return command


def trainer_output(job, campaign):
    if job.get("output"):
        previous = Path(job["output"])
        if (previous / "last.pt").is_file():
            return previous, previous / "last.pt"
    number = len(job["attempts"]) + 1
    return campaign / "jobs" / job["method"] / f"attempt_{number:02d}", None


def terminate_owned(active, interrupt=True):
    for process, _, _, _ in active.values():
        if process.poll() is None:
            if interrupt and os.name != "nt":
                process.send_signal(signal.SIGINT)
            else:
                process.terminate()


def supervise(output):
    output = Path(output).resolve()
    with file_lock(output / ".supervisor.lock", timeout=0):
        state = read_json(output / "campaign.json")
        if not state:
            raise ValueError("Missing campaign.json")
        config = state["config"]
        state.update(supervisor=process_identity(), state="preparing_data", updated_at=utc_now(), error=None)
        atomic_json(output / "campaign.json", state)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
        active, stopping_at = {}, None
        def requested_stop(signum, frame):
            atomic_json(output / "stop.request", dict(requested_at=utc_now(), signal=signum))
        signal.signal(signal.SIGTERM, requested_stop)
        try:
            if not config.get("manifest_dir"):
                from .acquire import ensure_manifests
                manifests, source = ensure_manifests(config)
                config["manifest_dir"] = str(manifests)
                atomic_json(output / "data_source.json", source)
            else:
                config["manifest_dir"] = str(Path(config["manifest_dir"]).resolve())
            from .data import read_manifests
            read_manifests(config["manifest_dir"])
            state.update(config=config, state="running")
            queue = []
            for method, job in state["jobs"].items():
                summary = read_json(Path(job["output"]) / "summary.json", {}) if job.get("output") else {}
                if job["state"] == "completed" or (summary.get("stop_reason") in {"max_epochs", "early_stopping"}):
                    job["state"] = "completed"
                else:
                    job["state"] = "queued"
                    queue.append(method)
            last_launch, last_snapshot = -float("inf"), -float("inf")
            while queue or active:
                now = time.monotonic()
                if (output / "stop.request").exists() and stopping_at is None:
                    stopping_at = now
                    state["state"] = "stopping"
                    terminate_owned(active)
                    for method in queue:
                        state["jobs"][method]["state"] = "not_started"
                    queue.clear()
                if stopping_at is not None and now - stopping_at > config["stop_grace"]:
                    terminate_owned(active, interrupt=False)
                if (stopping_at is None and queue and len(active) < config["parallel"] and
                        now - last_launch >= config["launch_delay"]):
                    method = queue.pop(0)
                    job = state["jobs"][method]
                    destination, checkpoint = trainer_output(job, output)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    command = job_command(config, method, destination, checkpoint)
                    log = open(job["log"], "a", encoding="utf-8", buffering=1)
                    log.write(f"\n=== {utc_now()} {'RESUME' if checkpoint else 'START'} {method} ===\n")
                    log.flush()
                    environment = {**os.environ, "PYTHONUNBUFFERED": "1",
                                   "OMP_NUM_THREADS": str(config["threads"]),
                                   "MKL_NUM_THREADS": str(config["threads"])}
                    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                               stderr=subprocess.STDOUT, cwd=config["cwd"], env=environment,
                                               close_fds=True)
                    attempt = dict(started_at=utc_now(), process=process_identity(process.pid),
                                   command=command, resumed=bool(checkpoint), output=str(destination), exit_code=None)
                    job["attempts"].append(attempt)
                    job.update(output=str(destination), state="running")
                    active[method] = (process, log, attempt, destination)
                    print(f"Started {method} pid={process.pid} output={destination}", flush=True)
                    last_launch = now
                for method in list(active):
                    process, log, attempt, destination = active[method]
                    code = process.poll()
                    if code is None:
                        continue
                    log.close()
                    summary = read_json(destination / "summary.json", {})
                    progress = read_json(destination / "progress.json", {})
                    if stopping_at is not None or summary.get("stop_reason") == "interrupted":
                        result = "interrupted"
                    elif code == 0 and summary.get("stop_reason") in {"max_epochs", "early_stopping"} and progress.get("phase") == "completed":
                        result = "completed"
                    else:
                        result = "failed"
                    attempt.update(exit_code=code, ended_at=utc_now())
                    state["jobs"][method].update(state=result, exit_code=code)
                    print(f"Finished {method}: {result}, exit={code}", flush=True)
                    del active[method]
                if now - last_snapshot >= 15:
                    state["gpu"] = gpu_snapshot()
                    last_snapshot = now
                state["updated_at"] = utc_now()
                atomic_json(output / "campaign.json", state)
                if queue or active:
                    time.sleep(.5)
            if stopping_at is not None:
                state["state"] = "stopped"
            elif any(job["state"] == "failed" for job in state["jobs"].values()):
                state["state"] = "completed_with_failures"
            else:
                state["state"] = "completed"
            state["finished_at"] = utc_now()
            state["updated_at"] = utc_now()
            atomic_json(output / "campaign.json", state)
            print(f"Campaign {state['state']}: {output}", flush=True)
            return 1 if state["state"] == "completed_with_failures" else 0
        except BaseException as error:
            terminate_owned(active, interrupt=False)
            for process, log, attempt, _ in active.values():
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                log.close()
                attempt.update(exit_code=process.returncode, ended_at=utc_now())
            state.update(state="failed", error=f"{type(error).__name__}: {error}",
                         traceback=traceback.format_exc(), updated_at=utc_now())
            atomic_json(output / "campaign.json", state)
            raise


def snapshot(output):
    output = Path(output).resolve()
    state = read_json(output / "campaign.json")
    if not state:
        raise ValueError(f"No campaign at {output}")
    alive = process_alive(state.get("supervisor"))
    pending = read_json(output / "supervisor_process.json", {})
    alive = alive or process_alive(pending.get("process"))
    result = dict(output=str(output), state=state["state"], supervisor_alive=alive,
                  updated_at=state["updated_at"], error=state.get("error"), gpu=state.get("gpu"), jobs=[])
    if not alive and state["state"] in {"preparing_data", "running", "stopping"}:
        result["state"] = "supervisor_not_running"
    for method, job in state["jobs"].items():
        progress = read_json(Path(job["output"]) / "progress.json", {}) if job.get("output") else {}
        phase = progress.get("phase", job["state"]) if job["state"] == "running" else job["state"]
        if job["state"] == "running" and job["attempts"] and progress.get("process") != job["attempts"][-1].get("process"):
            phase = "starting"
        running = process_alive(job["attempts"][-1].get("process")) if job["attempts"] else False
        if job["state"] == "running" and not running:
            phase = "exited" if alive else "not_running"
        result["jobs"].append(dict(method=method, state=phase, alive=running,
                                    output=job["output"], log=job["log"], exit_code=job.get("exit_code"), **{
                                        key: progress.get(key) for key in
                                        ("epoch", "max_epochs", "batch", "batches", "completed_epochs", "current_loss",
                                         "validation_value", "best_epoch", "best_value", "bad_epochs", "patience",
                                         "gpu_allocated_mb", "gpu_peak_mb", "updated_at", "last_checkpoint", "error")}))
    return result


def print_status(state):
    print(f"Campaign: {state['output']}\nState: {state['state']} | supervisor alive: {state['supervisor_alive']}")
    print(f"{'METHOD':12} {'STATE':17} {'EPOCH':9} {'BATCH':13} {'LOSS':12} {'BEST':12} {'CHECKPOINT':10}")
    def number(value):
        return f"{value:.6g}" if isinstance(value, (float, int)) else "-"
    for job in state["jobs"]:
        epoch = f"{job.get('epoch') or 0}/{job.get('max_epochs') or '-'}"
        batch = f"{job.get('batch') or 0}/{job.get('batches') or '-'}"
        print(f"{job['method']:12} {job['state']:17} {epoch:9} {batch:13} "
              f"{number(job.get('current_loss')):12} {number(job.get('best_value')):12} "
              f"{str(job.get('completed_epochs') or 0):10}")
        if job.get("error"):
            print(f"  {job['error']}")
        if job.get("updated_at"):
            print(f"  updated {job['updated_at']} | log: {job['log']}")
    if (state.get("gpu") or {}).get("available"):
        print("GPU snapshot (index, name, total MiB, used MiB, utilization %):")
        for row in state["gpu"]["rows"]:
            print("  " + row)
    if state.get("error"):
        print("Supervisor error: " + state["error"])


def tail_log(output, method, lines=40):
    path = Path(output) / "jobs" / method / "train.log" if method else Path(output) / "supervisor.log"
    if not path.exists():
        raise ValueError(f"Log does not exist yet: {path}")
    with open(path, encoding="utf-8", errors="replace") as stream:
        print("".join(deque(stream, maxlen=lines)), end="")
