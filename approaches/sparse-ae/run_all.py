"""Start detached parallel trainers, inspect progress after SSH reconnect, or resume."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

from sae import METHODS
from sae.acquire import DEFAULT_DATASET
from sae.campaign import launch, prepare_start, print_status, snapshot, supervise, tail_log
from sae.runtime import atomic_json, process_alive, read_json, utc_now


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="Start a new campaign, detached by default")
    start.add_argument("--output", help="New campaign directory; defaults to runs/sparse_ae/campaign_TIMESTAMP")
    start.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    start.add_argument("--parallel", type=int, default=7, help="Maximum concurrent models on the same GPU")
    start.add_argument("--root")
    start.add_argument("--manifest-dir")
    start.add_argument("--dataset", default=DEFAULT_DATASET)
    start.add_argument("--cache-dir")
    start.add_argument("--offline", action="store_true")
    start.add_argument("--train-split", default="train-clean-100")
    start.add_argument("--dev-split", default="dev-clean")
    start.add_argument("--test-split", default="test-clean")
    start.add_argument("--train-hours", type=float)
    start.add_argument("--seed", type=int, default=0)
    start.add_argument("--epochs", type=int, default=100)
    start.add_argument("--patience", type=int, default=8)
    start.add_argument("--min-delta", type=float, default=.0001)
    start.add_argument("--batch-size", type=int, default=4)
    start.add_argument("--device", default="cuda")
    start.add_argument("--workers", type=int, default=0)
    start.add_argument("--threads", type=int, default=2)
    start.add_argument("--gpu-memory-fraction", type=float, help="Default: .85 / number of simultaneous jobs")
    start.add_argument("--launch-delay", type=float, default=2, help="Seconds between child launches")
    start.add_argument("--stop-grace", type=float, default=60, help="Graceful interrupt deadline before termination")
    start.add_argument("--foreground", action="store_true", help="Stay attached; use detached default for SSH")
    start.add_argument("trainer_args", nargs=argparse.REMAINDER, help="Extra trainer arguments after --")
    resume = commands.add_parser("resume", help="Resume only unfinished/failed methods in an existing campaign")
    resume.add_argument("--output", required=True)
    resume.add_argument("--parallel", type=int)
    resume.add_argument("--foreground", action="store_true")
    status = commands.add_parser("status", help="Show persisted progress without attaching to training")
    status.add_argument("--output", required=True)
    status.add_argument("--watch", action="store_true")
    status.add_argument("--interval", type=float, default=5)
    status.add_argument("--json", action="store_true")
    stop = commands.add_parser("stop", help="Request interruption; keeps last completed epoch checkpoints")
    stop.add_argument("--output", required=True)
    logs = commands.add_parser("logs", help="Show recent supervisor or method log lines")
    logs.add_argument("--output", required=True)
    logs.add_argument("--method", choices=METHODS)
    logs.add_argument("--lines", type=int, default=40)
    internal = commands.add_parser("_supervise", help="Internal detached supervisor")
    internal.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "start":
            config = vars(args).copy()
            foreground = config.pop("foreground")
            config.pop("command")
            if args.parallel < 1 or args.batch_size < 1 or args.threads < 1 or args.epochs < 1 or args.workers < 0 or args.patience < 0:
                raise ValueError("Invalid concurrency, batch size, epoch count, workers, threads, or patience")
            if (not math.isfinite(args.min_delta) or args.min_delta < 0 or
                    not math.isfinite(args.launch_delay) or args.launch_delay < 0 or
                    not math.isfinite(args.stop_grace) or args.stop_grace <= 0):
                raise ValueError("Invalid min-delta, launch-delay, or stop-grace")
            if len(set(args.methods)) != len(args.methods):
                raise ValueError("Methods must be unique")
            if args.root and args.manifest_dir:
                raise ValueError("Choose --root or --manifest-dir")
            if args.manifest_dir and args.train_hours is not None:
                raise ValueError("Prepared manifests already define training hours")
            if args.train_hours is not None and (not math.isfinite(args.train_hours) or args.train_hours <= 0):
                raise ValueError("train-hours must be finite and positive")
            if not 0 <= args.seed < 2**32:
                raise ValueError("seed must be in [0, 2**32)")
            extras = args.trainer_args[1:] if args.trainer_args[:1] == ["--"] else args.trainer_args
            reserved = {"--root", "--manifest-dir", "--dataset", "--cache-dir", "--offline", "--train-hours",
                        "--output", "--resume", "--seed", "--batch-size", "--epochs", "--max-epochs", "--patience",
                        "--min-delta", "--device", "--workers", "--threads", "--gpu-memory-fraction",
                        "--train-split", "--dev-split", "--test-split"}
            if any(token.split("=", 1)[0] in reserved for token in extras):
                raise ValueError("Pass shared data/runtime/stopping options before --")
            config["trainer_args"] = extras
            count = min(args.parallel, len(args.methods))
            fraction = args.gpu_memory_fraction if args.gpu_memory_fraction is not None else .85 / count
            if not 0 < fraction <= 1:
                raise ValueError("gpu-memory-fraction must be in (0, 1]")
            config["gpu_memory_fraction"] = fraction
            if fraction * count > .90:
                print("Note: combined CUDA allocator caps leave little room for process/context overhead.", flush=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            config["output"] = str(Path(args.output or f"runs/sparse_ae/campaign_{stamp}").resolve())
            for key in ("root", "manifest_dir", "cache_dir"):
                if config[key]:
                    config[key] = str(Path(config[key]).expanduser().resolve())
            config.update(python=sys.executable, cwd=str(Path.cwd()))
            output = prepare_start(config)
            return launch(output, foreground)
        if args.command == "resume":
            return launch(args.output, args.foreground, args.parallel)
        if args.command == "status":
            if args.interval <= 0 or not math.isfinite(args.interval):
                raise ValueError("interval must be finite and positive")
            while True:
                state = snapshot(args.output)
                if args.watch and sys.stdout.isatty():
                    print("\033[2J\033[H", end="")
                if args.json:
                    print(json.dumps(state, indent=2), flush=True)
                else:
                    print_status(state)
                if not args.watch:
                    return 0
                time.sleep(args.interval)
        if args.command == "logs":
            if args.lines < 1:
                raise ValueError("lines must be positive")
            tail_log(args.output, args.method, args.lines)
            return 0
        if args.command == "stop":
            state = snapshot(args.output)
            if not state["supervisor_alive"]:
                raise ValueError("Supervisor is not running; inspect status and resume unfinished jobs")
            atomic_json(Path(args.output) / "stop.request", dict(requested_at=utc_now()))
            print("Stop requested. Trainers retain last completed epoch checkpoints.")
            return 0
        return supervise(args.output)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
