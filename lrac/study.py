"""Run a no-training study with ordinary functions and the existing run conventions."""
from datetime import datetime, timezone
from inspect import signature
import json
from pathlib import Path
import time

from lrac.utils import environment, file_hash, source_fingerprint, write_json


def study_function(kind):
    """Explicit selection; adding a study needs no registry or plugin machinery."""
    if kind == "sparse_search":
        from lrac.study_sparse_search import run
    elif kind == "sparse_cues":
        from lrac.study_sparse_cues import run
    else:
        raise ValueError(f"Unknown study kind: {kind}")
    return run


def run_study(config, output=None):
    """Save resolved settings/provenance, execute once, and retain failure status.

    Each study implements run(output, *, ...). Keyword defaults are saved, and
    unknown parameters fail before creating a run. Output must be a new directory.
    """
    function = study_function(config.kind)
    try:
        bound = signature(function).bind(Path("."), **config.parameters)
    except TypeError as error:
        raise ValueError(f"Invalid {config.kind} parameters: {error}") from error
    bound.apply_defaults()
    parameters = {k: v for k, v in bound.arguments.items() if k != "output"}
    resolved = {**config.to_dict(), "parameters": parameters}
    json.dumps(resolved, allow_nan=False)

    created = datetime.now(timezone.utc)
    stamp = created.strftime("%Y%m%dT%H%M%S%fZ")
    destination = config.path(output) if output is not None else config.path(config.runs_dir) / f"{config.name}_{stamp}"
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_json(destination / "config.json", resolved)
    write_json(destination / "progress.json", {"status": "running", "kind": config.kind, "created_utc": created.isoformat()})
    print(f"RUN={destination}", flush=True)
    try:
        write_json(destination / "environment.json", environment())
        write_json(destination / "source.json", source_fingerprint(Path(__file__).resolve().parents[1]))
        function(destination, **parameters)
        artifacts = {
            p.relative_to(destination).as_posix(): {"bytes": p.stat().st_size, "sha256": file_hash(p)}
            for p in sorted(destination.rglob("*")) if p.is_file() and p.name != "progress.json"
        }
        completion = {"status": "complete", "created_utc": created.isoformat(),
                      "finished_utc": datetime.now(timezone.utc).isoformat(),
                      "seconds": time.perf_counter() - started, "artifacts": artifacts}
        write_json(destination / "completion.json", completion)
        write_json(destination / "progress.json", {"status": "complete", "seconds": completion["seconds"]})
    except (Exception, KeyboardInterrupt) as error:
        write_json(destination / "progress.json", {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "error": f"{type(error).__name__}: {error}", "seconds": time.perf_counter() - started,
        })
        raise
    return destination
