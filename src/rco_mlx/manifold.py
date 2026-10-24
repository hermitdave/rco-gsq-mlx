"""Budget-manifold primitives for RCO, ported from PyTorch to NumPy.

Faithful port of ``src/manifold.py`` from ISTA-DASLab/RCO (Apache-2.0).

The original mutates tensors in place and reads ``alpha.grad`` off a live
autograd graph. This port takes the gradient and the Adam moments explicitly,
so the same code drives an MLX graph, a PyTorch graph, or plain NumPy.

The constraint is the softmax-budget equality (paper Eq. 2):

    C(alpha) = sum_i w_i * sum_k p_ik * c_k = B,     p_ik = softmax(alpha_i)_k

with ``alpha`` of shape ``(n_groups, n_options)``, ``costs`` of shape
``(n_options,)`` and ``weights`` of shape ``(n_groups,)``.

The three primitives (paper Sec. 2 / Appendix A.5):

    budget_normal      normal vector dC/dalpha to the level surface.
    project_gradient   tangent projection of the gradient onto the
                       budget-preserving subspace.
    retraction         binary-search retraction back onto the surface.
    vector_transport   re-project Adam's first moment onto the rotated
                       tangent plane.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "softmax",
    "budget_normal",
    "project_gradient",
    "retraction",
    "vector_transport",
    "project_gradient_per_layer",
    "retraction_per_layer",
    "vector_transport_per_layer",
]

_EPS = 1e-12


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def budget_normal(
    alpha: np.ndarray,
    costs: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Normal vector to the budget surface at ``alpha``.

    ``dC/dalpha_ik = w_i * p_ik * (c_k - E_i[c])`` where ``E_i[c]`` is the
    expected cost under the softmax. Same shape as ``alpha``; the tangent
    space is its orthogonal complement.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    probs = softmax(alpha, axis=-1)
    expected = (probs * costs).sum(axis=-1, keepdims=True)
    normal = probs * (costs - expected)
    if weights is not None:
        normal = normal * np.asarray(weights, dtype=np.float64)[..., None]
    return normal


def project_gradient(
    alpha: np.ndarray,
    grad: np.ndarray,
    costs: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, float]:
    """Project ``grad`` onto the budget-preserving subspace.

    Subtracts the component along the budget normal. Returns
    ``(grad_projected, coeff, raw_norm, projected_norm)`` for diagnostics.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    normal = budget_normal(alpha, costs, weights)

    nf = normal.ravel()
    g = grad.ravel()
    raw_norm = float(np.linalg.norm(g))
    coeff = float((g @ nf) / (float(nf @ nf) + _EPS))
    g_proj = (g - coeff * nf).reshape(grad.shape)

    return g_proj, coeff, raw_norm, float(np.linalg.norm(g_proj))


def retraction(
    alpha: np.ndarray,
    costs: np.ndarray,
    target: float,
    weights: np.ndarray | None = None,
    max_iter: int = 60,
    tol: float = 0.05,
) -> float:
    """Binary-search retraction back onto the constraint surface.

    Adds a scalar ``shift * costs`` to ``alpha`` so the budget hits ``target``.
    **Modifies ``alpha`` in place** (matching the original). Returns the
    achieved budget.

    The shift is found by bisection: the bracket starts narrow and is doubled
    outward until it contains the root, then bisection halts as soon as the
    budget is within ``tol`` of the target.
    """
    alpha = np.asarray(alpha)
    costs = np.asarray(costs, dtype=np.float64)
    w = None if weights is None else np.asarray(weights, dtype=np.float64)

    def budget_at(shift: float) -> float:
        probs = softmax(alpha + shift * costs, axis=-1)
        row = (probs * costs).sum(axis=-1)
        return float((row * w).sum() if w is not None else row.sum())

    current = budget_at(0.0)
    if abs(current - target) < tol:
        return current

    lo, hi = -1.0, 1.0
    for _ in range(40):
        if budget_at(lo) <= target <= budget_at(hi):
            break
        if budget_at(hi) < target:
            hi *= 2.0
        if budget_at(lo) > target:
            lo *= 2.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        value = budget_at(mid)
        if abs(value - target) < tol:
            lo = hi = mid
            break
        if value > target:
            hi = mid
        else:
            lo = mid

    shift = 0.5 * (lo + hi)
    alpha += shift * costs.astype(alpha.dtype)
    return budget_at(0.0)


def vector_transport(
    exp_avg: np.ndarray,
    alpha: np.ndarray,
    costs: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Vector transport (by projection) of Adam's first moment.

    Re-projects ``exp_avg`` onto the current tangent plane. As ``alpha`` moves
    along the manifold the tangent plane rotates and the old moment acquires a
    component along the new normal; removing it keeps Adam's adaptive capacity
    from leaking into the budget direction. **Modifies ``exp_avg`` in place.**

    The second moment is a variance estimate without a direction and is left
    untouched. This is a first-order approximation to parallel transport.
    """
    exp_avg = np.asarray(exp_avg)
    alpha = np.asarray(alpha, dtype=np.float64)
    normal = budget_normal(alpha, costs, weights)

    nf = normal.ravel()
    nsq = float(nf @ nf) + _EPS
    flat = exp_avg.ravel()
    coeff = float((flat @ nf) / nsq)
    exp_avg -= (coeff * nf).reshape(exp_avg.shape).astype(exp_avg.dtype)
    return exp_avg


# ---------------------------------------------------------------------------
# Block-separable variants for MoE expert pruning (one constraint per layer)
# ---------------------------------------------------------------------------


def project_gradient_per_layer(
    alpha: np.ndarray,
    grad: np.ndarray,
    costs: np.ndarray,
    n_layers: int,
    n_experts: int,
) -> float:
    """Block-separable tangent projection.

    Each layer carries its own equality constraint, so the projection
    decomposes into independent per-layer scalar coefficients. Returns the
    mean projection coefficient across layers.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    normal = budget_normal(alpha, costs)

    n_ll = normal.reshape(n_layers, n_experts, -1)
    g_ll = grad.reshape(n_layers, n_experts, -1)
    dot = (g_ll * n_ll).sum(axis=(1, 2))
    nsq = (n_ll * n_ll).sum(axis=(1, 2)) + _EPS
    coeff = dot / nsq
    grad -= (coeff.reshape(-1, 1, 1) * n_ll).reshape(grad.shape)
    return float(coeff.mean())


def retraction_per_layer(
    alpha: np.ndarray,
    costs: np.ndarray,
    target_per_layer: float,
    n_layers: int,
    n_experts: int,
    max_iter: int = 60,
    tol: float = 1e-3,
) -> float:
    """Per-layer retraction: shift each layer's logits to hit its budget.

    Vectorized bisection across layers. Each layer carries its own bracket;
    brackets start at +-1 and are doubled outward on whichever side has not
    yet crossed the target. **Modifies ``alpha`` in place.**
    """
    alpha = np.asarray(alpha)
    costs = np.asarray(costs, dtype=np.float64)
    alpha_ll = alpha.reshape(n_layers, n_experts, -1)
    c = costs.reshape(1, 1, -1)

    def expected_at(shift: np.ndarray) -> np.ndarray:
        shifted = alpha_ll + shift.reshape(-1, 1, 1) * c
        probs = softmax(shifted, axis=-1)
        return (probs * c).sum(axis=(1, 2))

    lo = np.full(n_layers, -1.0)
    hi = np.full(n_layers, 1.0)
    for _ in range(40):
        need_lo = expected_at(lo) > target_per_layer
        need_hi = expected_at(hi) < target_per_layer
        if not need_lo.any() and not need_hi.any():
            break
        lo = np.where(need_lo, 2.0 * lo, lo)
        hi = np.where(need_hi, 2.0 * hi, hi)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        value = expected_at(mid)
        if (np.abs(value - target_per_layer) < tol).all():
            lo = hi = mid
            break
        above = value > target_per_layer
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)

    final = 0.5 * (lo + hi)
    alpha_ll += final.reshape(-1, 1, 1) * c.astype(alpha_ll.dtype)
    probs = softmax(alpha_ll, axis=-1)
    return float((probs * c).sum(axis=(1, 2)).mean())


def vector_transport_per_layer(
    exp_avg: np.ndarray,
    alpha: np.ndarray,
    costs: np.ndarray,
    n_layers: int,
    n_experts: int,
) -> np.ndarray:
    """Block-separable vector transport of Adam's first moment."""
    exp_avg = np.asarray(exp_avg)
    alpha = np.asarray(alpha, dtype=np.float64)
    normal = budget_normal(alpha, costs)

    n_ll = normal.reshape(n_layers, n_experts, -1)
    nsq = (n_ll * n_ll).sum(axis=(1, 2)) + _EPS
    m_ll = exp_avg.reshape(n_layers, n_experts, -1)
    coeff = (m_ll * n_ll).sum(axis=(1, 2)) / nsq
    exp_avg -= (coeff.reshape(-1, 1, 1) * n_ll).reshape(exp_avg.shape).astype(exp_avg.dtype)
    return exp_avg
