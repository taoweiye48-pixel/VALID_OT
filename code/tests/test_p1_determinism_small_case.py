import numpy as np

from validot.p1 import P1Parameters, solve_p1


def test_p1_determinism_small_case():
    rng = np.random.default_rng(4)
    cost = rng.uniform(0.0, 2.0, size=(10, 9))
    a = np.full(10, 0.1)
    b = np.full(9, 1 / 9)
    parameters = P1Parameters("uot", 0.25, 2.0, tolerance=1e-10)
    first = solve_p1(cost, a, b, parameters)
    second = solve_p1(cost, a, b, parameters)
    np.testing.assert_array_equal(first.plan, second.plan)
