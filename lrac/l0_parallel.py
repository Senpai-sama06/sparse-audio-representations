"""Parallel exhaustive L0 traversal using the original, unchanged support fitter.

Workers receive disjoint contiguous chunks of one cardinality. The coordinator
commits results in lexicographic order, so the saved cursor always describes a
fully evaluated prefix. Uncommitted work may repeat after a crash; none is skipped.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import json
from math import comb
import os
from pathlib import Path
import time

import numpy as np

from lrac.l0 import _exclusive_run, implementation_hashes, next_support, support_fit
from lrac.utils import file_hash, write_json


def chunk_after(start, size, count):
    """Describe a bounded chunk without building a list of all its supports."""
    cursor, actual = start, 0
    while cursor is not None and actual < count:
        cursor = next_support(cursor, size)
        actual += 1
    return cursor, actual


def _initialize(source):
    global _waveform
    # Each process reads one small waveform once. BLAS/OpenMP threads are limited
    # to one by the launcher, avoiding workers times BLAS threads oversubscription.
    _waveform = np.load(source, allow_pickle=False)


def _evaluate(start, count, threshold, guard):
    """Fit every support in this chunk, using the same float64 SVD as serial."""
    cursor, ambiguous = start, 0
    for index in range(count):
        coefficients, error = support_fit(_waveform, cursor)
        ambiguous += int(abs(error-threshold) <= guard)
        if error <= threshold-guard:
            return dict(count=index+1, ambiguous=ambiguous, cursor=cursor,
                        coefficients=coefficients, error=error, pid=os.getpid())
        cursor = next_support(cursor, 2*len(_waveform))
    return dict(count=count, ambiguous=ambiguous, cursor=cursor, coefficients=None, pid=os.getpid())


def resume_search_parallel(folder, workers=0, chunk_size=512):
    """Continue the existing checkpoint with bounded parallel work and no time cap.

    Cardinalities never overlap. Only committed results contribute to progress or
    bounds. This permits an ordinary serial checkpoint to continue in parallel.
    """
    workers = workers or os.cpu_count() or 1
    if workers < 1 or chunk_size < 1:
        raise ValueError("Workers and chunk size must be positive")
    folder = Path(folder)
    with _exclusive_run(folder):
        protocol = json.loads((folder/"search_protocol.json").read_text())
        if implementation_hashes() != protocol["implementation_sha256"]:
            raise ValueError("Original support fitter or dictionary changed")
        if file_hash(folder/"source.npy") != protocol["source_npy_sha256"]:
            raise ValueError("Frozen source changed")
        execution = folder/"parallel_execution.json"
        source_hash = file_hash(Path(__file__))
        if execution.exists() and json.loads(execution.read_text())["implementation_sha256"] != source_hash:
            raise ValueError("Parallel traversal changed; preserve its implementation to resume")
        write_json(execution, dict(workers=workers, chunk_size=chunk_size,
                                  implementation_sha256=source_hash))
        x = np.load(folder/"source.npy", allow_pickle=False)
        size, energy = 2*len(x), float(x@x)
        threshold, guard = protocol["target_nmse"]*energy, protocol["boundary_guard_relative"]*energy
        state = json.loads((folder/"search_state.json").read_text())
        if state["status"] == "complete":
            return state
        clock, seen_workers = time.monotonic(), set()
        def save(status="running"):
            nonlocal clock
            now = time.monotonic()
            state["seconds"] += now-clock
            clock = now
            state["status"] = status
            write_json(folder/"search_state.json", state)
            write_json(folder/"progress.json", dict(status=status, process_id=os.getpid(), workers=workers,
                worker_pids=sorted(seen_workers), chunk_size=chunk_size, support_size=state["k"],
                tested_supports=state["tested_supports"], supports_at_size=str(comb(size, state["k"])),
                completed_smaller_sizes=state["k"]-1,
                numerical_lower_bound=min(state["k"], state["first_ambiguous_k"])
                    if state["first_ambiguous_k"] is not None else state["k"],
                feasible_upper_bound=state["upper_bound"], ambiguous_supports=state["ambiguous_supports"],
                seconds=state["seconds"], next_support=state["next_support"]))
        save()
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=_initialize,
                                     initargs=(str(folder/"source.npy"),)) as pool:
                while True:
                    scheduled = state["next_support"]
                    pending = deque()
                    while scheduled is not None or pending:
                        # Bounded prefetch prevents billions of Future objects.
                        while scheduled is not None and len(pending) < 2*workers:
                            after, count = chunk_after(scheduled, size, chunk_size)
                            pending.append(pool.submit(_evaluate, scheduled, count, threshold, guard))
                            scheduled = after
                        try:
                            result = pending[0].result(timeout=1)
                        except TimeoutError:
                            if time.monotonic()-clock >= 5:
                                save()
                            continue
                        pending.popleft()
                        seen_workers.add(result["pid"])
                        state["tested_supports"] += result["count"]
                        state["ambiguous_supports"] += result["ambiguous"]
                        if result["ambiguous"] and state["first_ambiguous_k"] is None:
                            state["first_ambiguous_k"] = state["k"]
                        if result["coefficients"] is not None:
                            np.savez(folder/"witness.npz", support=np.asarray(result["cursor"]),
                                     coefficients=result["coefficients"])
                            state.update(upper_bound=int(np.count_nonzero(result["coefficients"])),
                                achieved_nmse=result["error"]/energy if energy else 0, error_sq=result["error"],
                                numerical_minimum_established=state["first_ambiguous_k"] is None)
                            state["next_support"] = result["cursor"]
                            for future in pending:
                                future.cancel()
                            save("complete")
                            return state
                        state["next_support"] = result["cursor"]
                        if state["next_support"] is None:
                            # Advance before checkpointing: None marks the end of
                            # this cardinality, never a valid restart support.
                            state["k"] += 1
                            state["next_support"] = list(range(state["k"]))
                            save()
                        if time.monotonic()-clock >= 5:
                            save()
        except BaseException:
            save("interrupted")
            raise
