import numpy as np

from validot.semisynthetic import generate_pair, validate_truth


def _base():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 16))
    xy = rng.normal(size=(200, 2))
    labels = np.asarray([f"L{i % 4}" for i in range(200)])
    return x, xy, labels


def test_crop_truth_is_in_range_and_has_missing():
    pair = generate_pair(*_base(), scenario="crop_missing", seed=2, n=160, crop_fraction=0.25)
    check = validate_truth(pair)
    assert check["passed"]
    assert np.isclose(pair.truth_missing.mean(), 0.25)


def test_duplicate_cost_inputs_are_exactly_symmetric():
    pair = generate_pair(*_base(), scenario="duplicate_motif", seed=3, n=160)
    start = 160
    for offset, old in enumerate(pair.metadata["duplicate_old_indices"]):
        assert np.array_equal(pair.target_x[old], pair.target_x[start + offset])
        assert np.array_equal(pair.target_xy[old], pair.target_xy[start + offset])
