"""
Cubic-interpolation line search (Moré-Thuente style).

The cubic fit uses the current and previous residual norms and their approximate
directional derivatives (taken as -||R||² since the Newton direction is a descent direction for 0.5·||R||²).
"""

import typing

import numba
import numpy as np
import numpy.typing as npt


@numba.njit(cache=True)
def cubic_min(a: float, fa: float, dfa: float, b: float, fb: float, dfb: float) -> float:
    """
    Minimiser of the cubic through two points with prescribed derivatives.

    :param a: Left endpoint.
    :param fa: Function value at a.
    :param dfa: Derivative at a.
    :param b: Right endpoint.
    :param fb: Function value at b.
    :param dfb: Derivative at b.
    :return: Minimiser of the cubic, clamped to [a, b].
    """
    if abs(b - a) < 1e-14:
        return 0.5 * (a + b)

    d1 = dfa + dfb - 3.0 * (fb - fa) / (b - a)
    discriminant = d1 * d1 - dfa * dfb
    if discriminant < 0.0:
        return 0.5 * (a + b)

    d2 = np.sqrt(discriminant)
    denom = dfb - dfa + 2.0 * d2
    if abs(denom) < 1e-14:
        return 0.5 * (a + b)

    alpha = b - (b - a) * (dfb + d2 - d1) / denom
    return float(min(max(alpha, a), b))


def line_search(
    saturation_vector: npt.NDArray,
    saturation_change: npt.NDArray,
    current_residual_norm: float,
    residual_norm_fn: typing.Callable[[npt.NDArray], float],
    project_fn: typing.Callable[[npt.NDArray], npt.NDArray],
    maximum_cuts: int = 8,
    sufficient_decrease: float = 1e-4,
    min_step: float = 1e-8,
) -> tuple[npt.NDArray, float, float]:
    """
    Safeguarded cubic line search along the Newton step direction.

    Implements the Armijo sufficient-decrease condition with cubic interpolation
    fallback to find a step size that reduces the residual norm while respecting
    saturation constraints.

    :param saturation_vector: Current packed [Sw, Sg, ...] iterate.
    :param saturation_change: Full Newton step δS (already damped by maximum_saturation_change).
    :param current_residual_norm: ||R(S_k)|| at the current iterate.
        Used as the reference for the Armijo condition.
    :param residual_norm_fn: Callable that accepts a trial saturation
        vector and returns ||R||. Must handle projection internally or caller
        must project before passing.
    :param project_fn: Projects a trial vector onto the feasible simplex (Sw>=0, Sg>=0, Sw+Sg<=1).
    :param maximum_cuts: Maximum number of step-size reductions.
    :param sufficient_decrease: Armijo constant c₁. Condition:
        ||R(S+α·δS)|| < (1 - c₁·α)·||R(S)||. Use a small value (1e-4) so the
        condition is easy to satisfy initially.
    :param min_step: Minimum step size before giving up and accepting the best found.
    :return: Tuple of (projected trial saturation vector, accepted step size α, ||R|| at the accepted step).
    """
    alpha_prev = 0.0
    norm_prev = current_residual_norm
    # Approximate directional derivative: d/dα ||R||² at α=0 ≈ -2·||R_0||²
    # (Newton direction is a descent direction for 0.5·||R||²)
    dfa = -current_residual_norm * current_residual_norm

    alpha = 1.0
    best_alpha = 1.0
    best_norm = float("inf")
    best_vector = project_fn(saturation_vector + saturation_change)

    for _ in range(maximum_cuts):
        trial = project_fn(saturation_vector + alpha * saturation_change)
        norm = residual_norm_fn(trial)

        if norm < best_norm:
            best_norm = norm
            best_alpha = alpha
            best_vector = trial

        # Armijo sufficient-decrease check
        if norm < current_residual_norm * (1.0 - sufficient_decrease * alpha):
            return trial, alpha, norm

        if alpha < min_step:
            break

        # Cubic interpolation for next trial step
        # Approximate derivative at current α by finite difference
        interval = alpha - alpha_prev
        if abs(interval) > 1e-14:
            dfb = (norm * norm - norm_prev * norm_prev) / interval
            alpha_next = cubic_min(
                alpha_prev,
                norm_prev * norm_prev,
                dfa,
                alpha,
                norm * norm,
                dfb,
            )
            alpha_next = float(np.clip(alpha_next, 0.1 * alpha, 0.9 * alpha))
        else:
            # Degenerate interval. Fall back to bisection
            dfb = dfa
            alpha_next = 0.5 * alpha

        # Safety bounds: stay in (0.1·alpha, 0.9·alpha)
        alpha_prev = alpha
        norm_prev = norm
        dfa = dfb
        alpha = alpha_next

    # No sufficient decrease found so we return the best we have
    return best_vector, best_alpha, best_norm
