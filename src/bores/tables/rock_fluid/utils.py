import typing

import numba
import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator

from bores.typing import Spacing


@numba.njit(cache=True)
def make_saturation_field(
    n_points: int = 200,
    min_saturation: float = 0.0,
    max_saturation: float = 1.0,
    spacing: Spacing = "cosine",
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray:
    """
    Build a 1-D saturation field over `[min_saturation, max_saturation]`.

    :param n_points: Number of field points (>= 2).
    :param min_saturation: Physical saturation range. Must satisfy `0 ≤ min_saturation < max_saturation ≤ 1`.
    :param max_saturation: Physical saturation range. Must satisfy `0 ≤ min_saturation < max_saturation ≤ 1`.
    :param spacing: `"cosine"` (default) - Chebyshev-cosine spacing, denser at the
        endpoints. `"linspace"` - uniform spacing.
    :return: NDArray of shape `(n_points,)` with values in `[min_saturation, max_saturation]`,
    monotonically increasing.
    """
    if n_points < 2:
        raise ValueError(f"`n_points` must be >= 2, got {n_points}")
    if not (0.0 <= min_saturation < max_saturation <= 1.0):
        raise ValueError(
            f"Require 0 ≤ min_saturation < max_saturation ≤ 1, got `min_saturation={min_saturation}`, `max_saturation={max_saturation}`"
        )

    if spacing == "cosine":
        i = np.arange(n_points, dtype=dtype)
        unit: npt.NDArray = 0.5 * (1.0 - np.cos(np.pi * i / (n_points - 1)))
    elif spacing == "linspace":
        unit: npt.NDArray = np.linspace(0.0, 1.0, n_points)
    else:
        raise ValueError(f"`spacing` must be 'cosine' or 'linspace', got '{spacing}'")

    return (min_saturation + unit * (max_saturation - min_saturation)).astype(dtype)


@numba.njit(cache=True, inline="always")
def make_min_span_saturation_field(
    number_of_points: int,
    min_saturation: float,
    max_saturation: float,
    spacing: Spacing,
    minimum_span: float,
) -> npt.NDArray:
    """
    Build a saturation grid, enforcing a floor of `minimum_span` on the
    total range so the grid never collapses to a single point.

    `minimum_span` is a plain positional argument rather than keyword-only
    so that Numba can inline this function without a typing error.

    :param number_of_points: Number of points in the output grid.
    :param min_saturation: Lower bound of the saturation range.
    :param max_saturation: Upper bound of the saturation range.
    :param spacing: Grid spacing strategy (e.g. `'cosine'`, `'linspace'`).
    :param minimum_span: Minimum permitted distance between the lower and upper
        bounds. When the natural span is smaller, the upper bound is extended to
        `min_saturation + minimum_span`.
    :return: 1-D array of saturation values.
    """
    if max_saturation - min_saturation < minimum_span:
        return np.array(
            [
                min_saturation,
                max(min_saturation + minimum_span, max_saturation),
            ],
            dtype=np.float64,
        )
    return make_saturation_field(
        n_points=number_of_points,
        min_saturation=min_saturation,
        max_saturation=max_saturation,
        spacing=spacing,
        dtype=np.float64,
    )


def pchip_resample(
    source_saturations: npt.NDArray,
    source_values: npt.NDArray,
    number_of_output_points: int,
    spacing: Spacing,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    """
    Fit a PCHIP interpolant through (`source_saturations`, `source_values`),
    resample at `number_of_output_points` and return the resampled values.

    :param source_saturations: Original saturation knots, strictly increasing.
    :param source_values: Function values at each knot (kr or Pc).
    :param number_of_output_points: Number of points in the resampled grid.
    :param spacing: Grid spacing strategy for the output grid.
    :return: Two-tuple of `(resampled_saturations, resampled_values)`.
    """
    interpolant = PchipInterpolator(source_saturations, source_values)
    resampled_saturations = make_saturation_field(
        n_points=number_of_output_points,
        min_saturation=float(source_saturations[0]),
        max_saturation=float(source_saturations[-1]),
        spacing=spacing,
        dtype=np.float64,
    )
    return resampled_saturations, interpolant(resampled_saturations)


@numba.njit(cache=True)
def build_saturation_reference_field(
    number_of_base_points: int,
    saturation_lower_bound: float,
    saturation_upper_bound: float,
    spacing: Spacing,
    number_of_endpoint_extra_points: int,
    minimum_grid_span: float = 1e-6,
) -> npt.NDArray:
    """
    Build a saturation reference field with optional endpoint refinement.

    The base field spans [`saturation_lower_bound`, `saturation_upper_bound`]
    at `number_of_base_points` using `spacing`. When
    `number_of_endpoint_extra_points` > 0, extra knots are injected into the
    first and last 10 % of the range to capture the rapid variation of kr and
    Pc curves near residual saturations.

    :param number_of_base_points: Number of evenly-spaced base knots.
    :param saturation_lower_bound: Left end of the saturation range.
    :param saturation_upper_bound: Right end of the saturation range.
    :param spacing: Grid spacing strategy (e.g. `'cosine'`, `'linspace'`).
    :param number_of_endpoint_extra_points: Number of extra knots added inside
        each boundary decade. Pass `0` to disable endpoint refinement.
    :return: Sorted 1-D array of unique saturation values.
    """
    base_field = make_min_span_saturation_field(
        number_of_base_points,
        saturation_lower_bound,
        saturation_upper_bound,
        spacing,
        minimum_grid_span,
    )
    saturation_span = saturation_upper_bound - saturation_lower_bound
    if saturation_span < minimum_grid_span or number_of_endpoint_extra_points <= 0:
        return base_field

    endpoint_decade_width = 0.10 * saturation_span
    lower_endpoint_refinement = np.linspace(
        saturation_lower_bound,
        saturation_lower_bound + endpoint_decade_width,
        number_of_endpoint_extra_points + 2,
    )
    upper_endpoint_refinement = np.linspace(
        saturation_upper_bound - endpoint_decade_width,
        saturation_upper_bound,
        number_of_endpoint_extra_points + 2,
    )
    return np.unique(
        np.concatenate(
            (base_field, lower_endpoint_refinement, upper_endpoint_refinement)
        )
    )


def build_pchip_interpolant(
    reference_saturation: npt.NDArray,
    values: npt.NDArray,
    number_of_base_points: int,
    number_of_endpoint_extra_points: int,
    spacing: Spacing,
    minimum_scale_span: float = 1e-6,
) -> typing.Tuple[PchipInterpolator, PchipInterpolator]:
    """
    Build a PCHIP interpolant (and its derivative) for a two-phase kr or Pc
    curve, optionally after expanding the knot field.

    When `number_of_base_points > 0` *and* the saturation range is wide enough,
    the raw knots are first resampled to a denser field via
    `_pchip_resample_with_derivative`. This is the same field-expansion
    strategy used by `as_three_phase_relperm_table`.

    When `derivative_values` are supplied they are used as the source for
    the initial PCHIP fit (giving a C¹-consistent seed); otherwise the raw
    `values` array is used directly.

    :param reference_saturation: Monotonically increasing saturation knots.
    :param values: Function values (kr or Pc) at each knot.
    :param derivative_values: Optional pre-computed derivative at each knot.
        Used only as the source for the initial PCHIP when present.
    :param number_of_base_points: Target number of base points for field expansion.
        Pass `0` to use the raw knots without expansion.
    :param number_of_endpoint_extra_points: Extra knots in each boundary decade during
        expansion.  Ignored when `number_of_base_points == 0`.
    :param spacing: Grid spacing mode passed to
        `_build_saturation_reference_grid`.
    :param minimum_scale_span: Minimum saturation span required before field scaling is attempted.
    :return: Two-tuple `(interpolant, derivative_interpolant)` where
        `derivative_interpolant` is `interpolant.derivative()`.
    """
    saturation_field = reference_saturation
    vals = values

    span = float(saturation_field[-1]) - float(saturation_field[0])
    should_scale = (
        number_of_base_points > 0
        and len(saturation_field) < number_of_base_points
        and span > minimum_scale_span
    )
    if should_scale:
        expanded_saturation_field = build_saturation_reference_field(
            number_of_base_points=number_of_base_points,
            saturation_lower_bound=float(saturation_field[0]),
            saturation_upper_bound=float(saturation_field[-1]),
            spacing=spacing,
            number_of_endpoint_extra_points=number_of_endpoint_extra_points,
        )
        # Fit a temporary PCHIP on the raw knots to resample onto the expanded field
        source_pchip = PchipInterpolator(saturation_field, vals)
        vals = source_pchip(expanded_saturation_field)
        saturation_field = expanded_saturation_field

    interpolant = PchipInterpolator(saturation_field, vals)
    derivative_interpolant: PchipInterpolator = interpolant.derivative(1)
    return interpolant, derivative_interpolant
