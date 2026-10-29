"""Freeze a full speech crop and launch the separate exhaustive L0 activity."""
import json
from pathlib import Path

import numpy as np

from lrac.data.l1_examples import read_example
from lrac.l0 import implementation_hashes, resume_search
from lrac.utils import file_hash, write_json


def run(output, *, cohort="runs/l1_librispeech_20261029T110402750634Z/cohort.json",
        source_id=None, target_nmse=.01, boundary_guard_relative=1e-12):
    """Use the first measurement clip by default, without shortening the waveform."""
    if not 0 < target_nmse < 1 or not 0 < boundary_guard_relative < target_nmse:
        raise ValueError("Require 0 < boundary guard < target NMSE < 1")
    root = Path(__file__).resolve().parents[1]
    cohort = Path(cohort).expanduser()
    cohort = cohort if cohort.is_absolute() else root/cohort
    records = json.loads(cohort.read_text(encoding="utf-8"))["examples"]
    row = next(r for r in records if r["id"] == source_id) if source_id else next(r for r in records if r["role"] == "measurement")
    x, digest = read_example(row, row["sample_rate"])
    output = Path(output)
    np.save(output/"source.npy", x)
    # Identity atoms provide an initial feasible upper bound, without consulting
    # L1 or skipping any support. The search still begins at the empty support.
    support = np.flatnonzero(x)
    np.savez(output/"initial_witness.npz", support=support, coefficients=x[support])
    write_json(output/"search_protocol.json", dict(dictionary="impulse_dct", samples=len(x),
        coefficient_slots=2*len(x), target_nmse=target_nmse,
        boundary_guard_relative=boundary_guard_relative, source_id=row["id"],
        source_sha256=digest, source_npy_sha256=file_hash(output/"source.npy"),
        cohort_sha256=file_hash(cohort), sample_rate=row["sample_rate"],
        implementation_sha256=implementation_hashes(),
        scope="Exhaustive support enumeration; real coefficients fitted by float64 SVD least squares; numerical bounds only"))
    write_json(output/"search_state.json", dict(status="running", k=0, next_support=[],
        tested_supports=0, upper_bound=len(support), first_ambiguous_k=None,
        ambiguous_supports=0, seconds=0.0))
    resume_search(output)
