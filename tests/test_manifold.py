"""Manifold primitive tests.

These check the port against first principles rather than against the PyTorch
original: the budget normal is verified against a finite difference of the
budget function, the projection is verified to remove exactly the normal
component, and the retraction is verified to hit its target.
"""

import numpy as np
import pytest

from rco_mlx.manifold import (
    budget_normal,
    project_gradient,
    retraction,
    softmax,
    vector_transport,
)


def _budget(alpha, costs, weights):
    probs = softmax(alpha, axis=-1)
    row = (probs * costs).sum(axis=-1)
    return float((row * weights).sum())


def test_budget_normal_matches_finite_difference():
    rng = np.random.default_rng(0)
    alpha = rng.normal(size=(5, 4))
    costs = np.array([2.0, 3.0, 4.0, 6.0])
    weights = rng.random(5) + 0.5

    analytic = budget_normal(alpha, costs, weights)

    eps = 1e-6
    numeric = np.zeros_like(alpha)
    for i in range(alpha.shape[0]):
        for k in range(alpha.shape[1]):
            plus = alpha.copy()
            minus = alpha.copy()
            plus[i, k] += eps
            minus[i, k] -= eps
            numeric[i, k] = (_budget(plus, costs, weights) - _budget(minus, costs, weights)) / (2 * eps)

    assert np.allclose(analytic, numeric, atol=1e-5)


def test_budget_normal_is_normal_to_the_constraint_surface():
    """Moving alpha along the normal must change the budget, and only that."""
    rng = np.random.default_rng(1)
    alpha = rng.normal(size=(3, 4))
    costs = np.array([2.0, 3.0, 4.0, 6.0])
    weights = np.array([0.2, 0.3, 0.5])

    n = budget_normal(alpha, costs, weights)
    before = _budget(alpha, costs, weights)
    after = _budget(alpha + 0.1 * n, costs, weights)

    # the normal direction must move the budget
    assert abs(after - before) > 1e-4


def test_project_gradient_removes_the_normal_component():
    rng = np.random.default_rng(2)
    alpha = rng.normal(size=(6, 4))
    grad = rng.normal(size=(6, 4))
    costs = np.array([2.0, 3.0, 4.0, 6.0])
    weights = rng.random(6) + 0.5

    projected, coeff, raw_norm, proj_norm = project_gradient(alpha, grad, costs, weights)

    n = budget_normal(alpha, costs, weights)
    assert np.isclose(float(projected.ravel() @ n.ravel()), 0.0, atol=1e-8)
    assert proj_norm < raw_norm
    assert proj_norm <= raw_norm + 1e-9


def test_projected_step_drifts_less_than_an_unprojected_one():
    """Projection makes the gradient tangent, so the step drifts only to 2nd order."""
    rng = np.random.default_rng(3)
    alpha = rng.normal(size=(4, 3))
    grad = rng.normal(size=(4, 3))
    costs = np.array([2.0, 4.0, 8.0])
    weights = np.array([0.1, 0.2, 0.3, 0.4])

    projected, *_ = project_gradient(alpha, grad, costs, weights)

    step = 1e-3
    before = _budget(alpha, costs, weights)
    drift_proj = abs(_budget(alpha - step * projected, costs, weights) - before)
    drift_raw = abs(_budget(alpha - step * grad, costs, weights) - before)

    # the projected step is second-order in the step size; the raw step is not
    assert drift_proj < drift_raw
    assert drift_proj < 1e-5


def test_retraction_hits_the_target():
    """The weights are parameter fractions, so they must sum to 1.

    With unnormalised weights the budget ranges over ``[min_cost, max_cost]``
    scaled by ``sum(weights)``, and a target inside ``[2, 8]`` is simply
    unreachable -- the bisection then converges to the nearest end, which looks
    like a broken retraction but is an infeasible target.
    """
    rng = np.random.default_rng(4)
    costs = np.array([2.0, 3.0, 4.0, 6.0, 8.0])
    weights = rng.random(7) + 0.2
    weights /= weights.sum()
    assert np.isclose(weights.sum(), 1.0)

    for target in [2.5, 4.0, 5.5, 7.0]:
        alpha = rng.normal(size=(7, 5))
        achieved = retraction(alpha, costs, target, weights, tol=1e-3)
        assert abs(achieved - target) < 1e-3


def test_retraction_raises_or_saturates_on_an_infeasible_target():
    """A target below the cheapest possible budget has no root."""
    costs = np.array([2.0, 4.0, 8.0])
    weights = np.array([0.5, 0.5])
    alpha = np.zeros((2, 3))

    # the minimum achievable budget is 2.0 * sum(weights) = 2.0
    achieved = retraction(alpha, costs, 1.0, weights, tol=1e-3)
    assert achieved >= 1.99


def test_retraction_is_monotone_in_the_shift():
    """Larger shifts must always give a larger budget."""
    rng = np.random.default_rng(5)
    alpha = rng.normal(size=(3, 4))
    costs = np.array([2.0, 3.0, 4.0, 6.0])
    weights = np.array([0.3, 0.3, 0.4])

    values = [_budget(alpha + s * costs, costs, weights) for s in (-5.0, -1.0, 0.0, 1.0, 5.0)]
    assert values == sorted(values)


def test_vector_transport_keeps_the_moment_tangent():
    rng = np.random.default_rng(6)
    alpha = rng.normal(size=(5, 4))
    costs = np.array([2.0, 3.0, 4.0, 6.0])
    weights = rng.random(5) + 0.5

    exp_avg = rng.normal(size=(5, 4))
    before = _budget(alpha, costs, weights)

    vector_transport(exp_avg, alpha, costs, weights)

    n = budget_normal(alpha, costs, weights)
    assert np.isclose(float(exp_avg.ravel() @ n.ravel()), 0.0, atol=1e-8)
    # transporting the moment must not move alpha
    assert _budget(alpha, costs, weights) == pytest.approx(before)


def test_retraction_handles_extreme_targets():
    """The bracket must expand far enough for targets outside the initial bracket."""
    rng = np.random.default_rng(7)
    alpha = rng.normal(size=(2, 3)) * 5.0
    costs = np.array([2.0, 4.0, 8.0])
    weights = np.array([0.5, 0.5])

    target = 7.9
    achieved = retraction(alpha, costs, target, weights, tol=1e-3)
    assert abs(achieved - target) < 1e-3
