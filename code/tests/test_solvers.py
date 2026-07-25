import numpy as np

from validot.solvers import log_sinkhorn, row_softmax


def test_balanced_sinkhorn_marginals():
    cost = np.array([[0.0, 1.0], [1.0, 0.0]])
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    result = log_sinkhorn(cost, a, b, epsilon=0.1, max_iter=1000, tol=1e-12)
    assert result.converged
    assert np.allclose(result.plan.sum(axis=1), a, atol=1e-9)
    assert np.allclose(result.plan.sum(axis=0), b, atol=1e-9)


def test_row_softmax_ignores_column_marginal():
    cost = np.array([[0.0, 1.0], [0.0, 1.0]])
    a = np.array([0.25, 0.75])
    result = row_softmax(cost, a, epsilon=0.1)
    assert np.allclose(result.plan.sum(axis=1), a)
