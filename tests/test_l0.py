"""Small complete enumerations validate traversal, minimum support and restart."""
from itertools import combinations
import json

import numpy as np

from lrac.l0 import next_support, resume_search, implementation_hashes
from lrac.utils import file_hash, write_json


def test_successor_enumerates_every_support_once():
    for k in range(6):
        support = list(range(k))
        observed = []
        while support is not None:
            observed.append(tuple(support))
            support = next_support(support, 5)
        assert observed == list(combinations(range(5), k))


def test_minimum_support_and_resume_from_middle_of_cardinality(tmp_path):
    for resume in (False, True):
        folder = tmp_path/str(resume)
        folder.mkdir()
        # Neither identity nor either 2-point cosine atom spans this waveform.
        # The first two-atom support (the identity pair) reconstructs it exactly.
        np.save(folder/"source.npy", np.array([1., .2]))
        write_json(folder/"search_protocol.json", dict(target_nmse=1e-6,
            boundary_guard_relative=1e-12, implementation_sha256=implementation_hashes(),
            source_npy_sha256=file_hash(folder/"source.npy")))
        write_json(folder/"search_state.json", dict(status="interrupted", k=1 if resume else 0,
            next_support=[2] if resume else [], tested_supports=3 if resume else 0,
            upper_bound=2, first_ambiguous_k=None, ambiguous_supports=0, seconds=0.))
        state = resume_search(folder)
        assert state["upper_bound"] == state["k"] == 2
        assert state["tested_supports"] == 6
        assert state["numerical_minimum_established"]
        witness = np.load(folder/"witness.npz")
        np.testing.assert_allclose(witness["coefficients"], [1, .2])
        assert json.loads((folder/"progress.json").read_text())["numerical_lower_bound"] == 2
