import numpy as np

from validot.p1 import arm_weights, mixed_cost


def test_registered_path_endpoints():
    ce = np.array([[1.0, 2.0]])
    cs = np.array([[3.0, 4.0]])
    np.testing.assert_allclose(mixed_cost(ce, cs, arm_weights("R", "I_EXPR", 0)), 0.5 * ce + 0.5 * cs)
    np.testing.assert_allclose(mixed_cost(ce, cs, arm_weights("R", "I_EXPR", 1)), 0.5 * cs)
    np.testing.assert_allclose(mixed_cost(ce, cs, arm_weights("R", "I_SPATIAL", 0)), 0.5 * ce + 0.5 * cs)
    np.testing.assert_allclose(mixed_cost(ce, cs, arm_weights("R", "I_SPATIAL", 1)), 0.5 * ce)
