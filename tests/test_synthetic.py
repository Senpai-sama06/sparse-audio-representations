import numpy as np
from lrac.data.synthetic import generate, records


def test_reproducible_components_reconstruct_signal():
    row = next(records("train", 1))
    x, parts, factors = generate(row)
    again, _, _ = generate(row)
    assert np.array_equal(x, again)
    assert np.allclose(sum(parts.values()), x)
    assert np.isfinite(x).all()


def test_combination_is_reserved_and_seeds_do_not_overlap():
    rows = {s: list(records(s, 30)) for s in ["train", "validation", "iid", "combinations", "extrapolation", "families"]}
    seeds = [r["seed"] for group in rows.values() for r in group]
    assert len(set(seeds)) == len(seeds)
    for row in rows["train"]:
        _, _, factors = generate(row)
        assert not (factors["f0"] > 220 and factors["cutoff"] > 3000)
    for row in rows["combinations"]:
        _, _, factors = generate(row)
        assert factors["f0"] >= 220 and factors["cutoff"] >= 3000
    assert {generate(r)[2]["family"] for r in rows["families"]} == {"chirp", "am_fm", "damped"}
