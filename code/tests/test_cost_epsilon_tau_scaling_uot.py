import numpy as np

from validot.p1 import P1Parameters, solve_p1


def test_cost_epsilon_tau_scaling_uot():
    rng = np.random.default_rng(2)
    cost = rng.uniform(0.1, 2.0, size=(8, 7))
    a = np.full(8, 1 / 8)
    b = np.full(7, 1 / 7)
    first = solve_p1(cost, a, b, P1Parameters("uot", 0.25, 2.0, tolerance=1e-10))
    second = solve_p1(0.5 * cost, a, b, P1Parameters("uot", 0.125, 1.0, tolerance=1e-10))
    np.testing.assert_allclose(first.plan, second.plan, rtol=1e-8, atol=1e-11)
