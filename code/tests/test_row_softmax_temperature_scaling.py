import numpy as np

from validot.p1 import P1Parameters, solve_p1


def test_row_softmax_temperature_scaling():
    rng = np.random.default_rng(3)
    cost = rng.uniform(0.1, 2.0, size=(8, 7))
    a = np.full(8, 1 / 8)
    b = np.full(7, 1 / 7)
    first = solve_p1(cost, a, b, P1Parameters("row_softmax", 0.25))
    second = solve_p1(0.5 * cost, a, b, P1Parameters("row_softmax", 0.125))
    np.testing.assert_allclose(first.plan, second.plan, rtol=0, atol=1e-15)
