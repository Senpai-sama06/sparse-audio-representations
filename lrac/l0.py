"""Literal support enumeration for minimum L0 in a fixed finite dictionary.

No pruning, greedy support proposals or learned inference is used. Coefficients
are continuous: least squares solves the best reconstruction on EACH support.
Float64 decisions are numerical evidence, not exact-arithmetic certificates.
"""
from contextlib import contextmanager
import json
from math import comb
import os
from pathlib import Path
import time

import numpy as np
from scipy.linalg import lstsq

from lrac.linear import CosineDictionary
from lrac.utils import file_hash, write_json


def next_support(support, size):
    """Lexicographic successor without replaying combinations on a restart."""
    result = list(support)
    for i in range(len(result)-1, -1, -1):
        if result[i] < size-len(result)+i:
            result[i] += 1
            for j in range(i+1, len(result)):
                result[j] = result[j-1]+1
            return result
    return None


def support_fit(x, support):
    """Materialize only the selected columns, then solve their least squares.

    There are 2N atoms but no dense N x 2N dictionary is stored. Each column is
    evaluated explicitly; all support candidates still receive an independent fit.
    """
    n = len(x)
    if not support:
        return np.empty(0), float(x@x)
    matrix = np.empty((n, len(support)))
    bank = CosineDictionary(n, "dct")
    for column, atom in enumerate(support):
        unit = np.zeros(n)
        unit[atom % n] = 1
        matrix[:, column] = unit if atom < n else bank.synthesis(unit)
    coefficients = lstsq(matrix, x, lapack_driver="gelsd")[0]
    residual = x-matrix@coefficients
    return coefficients, float(residual@residual)


def implementation_hashes():
    """A restart must use the same search and dictionary definitions."""
    root = Path(__file__).resolve().parent
    return {name: file_hash(root/name) for name in ("l0.py", "linear.py")}


@contextmanager
def _exclusive_run(folder):
    """OS locks release on process exit, so crashes leave no stale-lock cleanup."""
    with (folder/"search.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def resume_search(folder):
    """Run indefinitely until minimum support is found or the process is stopped.

    Progress is saved every five seconds and at cardinality boundaries. An abrupt
    termination may replay the last checkpoint interval, but cannot skip supports.
    A regular interruption saves the exact next support before propagating it.
    """
    folder = Path(folder)
    with _exclusive_run(folder):
        protocol = json.loads((folder/"search_protocol.json").read_text())
        if implementation_hashes() != protocol["implementation_sha256"]:
            raise ValueError("Search implementation changed; preserve the original code to resume")
        if file_hash(folder/"source.npy") != protocol["source_npy_sha256"]:
            raise ValueError("Frozen source changed")
        x = np.load(folder/"source.npy", allow_pickle=False)
        size, energy = 2*len(x), float(x@x)
        threshold = protocol["target_nmse"]*energy
        guard = protocol["boundary_guard_relative"]*energy
        state = json.loads((folder/"search_state.json").read_text())
        if state["status"] == "complete":
            return state
        clock = time.monotonic()
        def save(status="running"):
            nonlocal clock
            now = time.monotonic()
            state["seconds"] += now-clock
            clock = now
            state["status"] = status
            write_json(folder/"search_state.json", state)
            write_json(folder/"progress.json", dict(status=status, process_id=os.getpid(),
                support_size=state["k"], tested_supports=state["tested_supports"],
                supports_at_size=str(comb(size, state["k"])),
                completed_smaller_sizes=state["k"]-1,
                numerical_lower_bound=min(state["k"], state["first_ambiguous_k"])
                    if state["first_ambiguous_k"] is not None else state["k"],
                feasible_upper_bound=state["upper_bound"], ambiguous_supports=state["ambiguous_supports"],
                seconds=state["seconds"], next_support=state["next_support"]))
        save()
        try:
            while True:
                support = state["next_support"]
                coefficients, error = support_fit(x, support)
                state["tested_supports"] += 1
                if abs(error-threshold) <= guard:
                    state["ambiguous_supports"] += 1
                    if state["first_ambiguous_k"] is None:
                        state["first_ambiguous_k"] = state["k"]
                if error <= threshold-guard:
                    # All smaller cardinalities have been enumerated. Save the
                    # witness before declaring completion; near-boundary earlier
                    # supports remain explicitly unresolved in the lower bound.
                    np.savez(folder/"witness.npz", support=np.asarray(support), coefficients=coefficients)
                    state.update(upper_bound=int(np.count_nonzero(coefficients)), error_sq=error,
                                 achieved_nmse=error/energy if energy else 0,
                                 numerical_minimum_established=state["first_ambiguous_k"] is None)
                    save("complete")
                    return state
                successor = next_support(support, size)
                if successor is None:
                    state["k"] += 1
                    state["next_support"] = list(range(state["k"]))
                    save()
                    print(f"L0 completed sizes below {state['k']}; tested {state['tested_supports']}", flush=True)
                else:
                    state["next_support"] = successor
                if time.monotonic()-clock >= 5:
                    save()
        except BaseException:
            save("interrupted")
            raise
