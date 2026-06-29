"""
Hysteresis models for relative permeability and capillary pressure.

Implements the Killough (1976) scanning-curve model with Land's (1968) trapping
for relative permeability hysteresis, and a Killough-type scanning-curve model
for capillary pressure hysteresis.

**References**:

- Killough, J.E. (1976). "Reservoir Simulation With History-Dependent
  Saturation Functions". SPE 5106.
- Land, C.S. (1968). "Calculation of Imbibition Relative Permeability for
  Two- and Three-Phase Flow from Rock Properties". SPE 1942.
- Carlson, F.M. (1981). "Simulation of Relative Permeability Hysteresis to
  the Non-Wetting Phase". SPE 10157.
"""

import typing

import attrs
import numba
import numpy as np
import numpy.typing as npt

from bores.constants import c
from bores.errors import ValidationError
from bores.tables.rock_fluid.capillary_pressure import (
    CapillaryPressureTable,
    TwoPhaseCapillaryPressureTable,
    capillary_pressure_table,
)
from bores.tables.rock_fluid.relperm import (
    MixingRule,
    RelativePermeabilityTable,
    TwoPhaseRelPermTable,
    get_mixing_rule,
    get_mixing_rule_partial_derivatives,
    relperm_table,
    serialize_mixing_rule,
)
from bores.typing import (
    BooleanArray,
    CapillaryPressureDerivatives,
    CapillaryPressures,
    FluidPhase,
    NDimension,
    Number,
    NumberArray,
    NumberOrArray,
    RelativePermeabilities,
    RelativePermeabilityDerivatives,
)
from bores.utils import atleast_1d

__all__ = ["KilloughCapillaryPressureTable", "KilloughLandRelPermTable"]


@numba.njit(cache=True)
def _compute_land_residual_saturation_scalar(
    initial_non_wetting_saturation: Number,
    maximum_residual_saturation: Number,
    land_trapping_coefficient: Number,
    saturation_epsilon: Number = 1e-12,
) -> Number:
    """
    Compute the dynamic residual non-wetting saturation via Land's model (scalar).

    Land (1968) relates the residual saturation that will be trapped when
    imbibition begins at `initial_non_wetting_saturation` to the maximum
    possible residual saturation observed at the drainage endpoint:

        S_r = S_r_max / (1 + C * S_i)

    Higher *C* means more trapping (smaller residual for the same initial
    saturation). *C* = 0 means no dynamic trapping (residual equals
    `max_residual_saturation` regardless of initial saturation).

    :param initial_non_wetting_saturation: Non-wetting saturation at the
        drainage-imbibition reversal point.
    :param maximum_residual_saturation: Maximum residual saturation from the
        drainage endpoint (S_r_max).
    :param land_trapping_coefficient: Land trapping coefficient *C* (>= 0).
    :param saturation_epsilon: Small value to guard against division by zero.
    :return: Dynamic residual saturation (scalar).
    """
    safe_maximum_residual = max(maximum_residual_saturation, saturation_epsilon)

    if land_trapping_coefficient <= 0.0:
        return safe_maximum_residual

    non_negative_initial = max(initial_non_wetting_saturation, 0.0)
    dynamic_residual = safe_maximum_residual / (
        1.0 + land_trapping_coefficient * non_negative_initial
    )
    return min(dynamic_residual, non_negative_initial)


@numba.njit(cache=True, parallel=True)
def _compute_land_residual_saturation_array(
    initial_non_wetting_saturation: NumberArray[NDimension],
    maximum_residual_saturation: Number,
    land_trapping_coefficient: Number,
    saturation_epsilon: Number = 1e-12,
) -> NumberArray[NDimension]:
    """
    Compute the dynamic residual non-wetting saturation via Land's model (array).

    :param initial_non_wetting_saturation: Non-wetting saturations at the
        drainage-imbibition reversal points (1-D or N-D array).
    :param maximum_residual_saturation: Maximum residual saturation from the
        drainage endpoint (scalar).
    :param land_trapping_coefficient: Land trapping coefficient *C* (>= 0).
    :param saturation_epsilon: Small value to guard against division by zero.
    :return: Dynamic residual saturation array with the same shape as the input.
    """
    initial_saturation_array = atleast_1d(initial_non_wetting_saturation)
    safe_maximum_residual = max(maximum_residual_saturation, saturation_epsilon)

    result = np.empty_like(initial_saturation_array)
    for flat_index in numba.prange(initial_saturation_array.size):  # type: ignore
        non_negative_initial = max(initial_saturation_array.flat[flat_index], 0.0)
        if land_trapping_coefficient <= 0.0:
            result.flat[flat_index] = min(safe_maximum_residual, non_negative_initial)
        else:
            dynamic_residual = safe_maximum_residual / (
                1.0 + land_trapping_coefficient * non_negative_initial
            )
            result.flat[flat_index] = min(dynamic_residual, non_negative_initial)
    return typing.cast(NumberArray[NDimension], result)


def compute_land_residual_saturation(
    initial_non_wetting_saturation: NumberOrArray[NDimension],
    maximum_residual_saturation: Number,
    land_trapping_coefficient: Number,
    saturation_epsilon: Number = 1e-12,
) -> NumberOrArray[NDimension]:
    """
    Compute the dynamic residual non-wetting saturation via Land's model.

    :param initial_non_wetting_saturation: Non-wetting saturation at the
        drainage-imbibition reversal point (scalar or array).
    :param maximum_residual_saturation: Maximum residual saturation from the
        drainage endpoint (S_r_max).
    :param land_trapping_coefficient: Land trapping coefficient *C* (>= 0).
    :param saturation_epsilon: Small value to guard against division by zero.
    :return: Dynamic residual saturation matching the shape of the input.
    """
    if np.isscalar(initial_non_wetting_saturation):
        return _compute_land_residual_saturation_scalar(
            initial_non_wetting_saturation=float(initial_non_wetting_saturation),  # type: ignore
            maximum_residual_saturation=maximum_residual_saturation,
            land_trapping_coefficient=land_trapping_coefficient,
            saturation_epsilon=saturation_epsilon,
        )
    return _compute_land_residual_saturation_array(
        initial_non_wetting_saturation=np.asarray(
            initial_non_wetting_saturation, dtype=np.float64
        ),
        maximum_residual_saturation=maximum_residual_saturation,
        land_trapping_coefficient=land_trapping_coefficient,
        saturation_epsilon=saturation_epsilon,
    )


@numba.njit(cache=True)
def _compute_killough_scanning_curve_scalar(
    saturation: Number,
    drainage_curve_value: Number,
    imbibition_curve_value: Number,
    reversal_saturation: Number,
    maximum_historical_saturation: Number,
    is_imbibition: Number,
    scanning_exponent: Number = 1.0,
    numerical_epsilon: Number = 1e-12,
) -> Number:
    """
    Killough scanning-curve interpolation between primary drainage and
    imbibition curves (scalar).

    When the flow direction reverses the reservoir property (kr or Pc)
    follows a scanning curve interpolated between the two primary curves::

        value = value_drain + (value_imb - value_drain) * f(S)

    where the interpolation factor is::

        f(S) = clamp( ((S - S_rev) / (S_max - S_rev))^n , 0, 1 )

    :param saturation: Current saturation.
    :param drainage_curve_value: Value from the primary drainage curve.
    :param imbibition_curve_value: Value from the primary imbibition curve.
    :param reversal_saturation: Saturation at the last reversal point (S_rev).
    :param maximum_historical_saturation: Maximum saturation reached before the reversal (S_max).
    :param is_imbibition: 1.0 if currently on the imbibition path, 0.0 if drainage.
    :param scanning_exponent: Killough interpolation exponent *n* (1 = linear).
    :param numerical_epsilon: Numerical stability tolerance.
    :return: Interpolated scanning-curve value.
    """
    saturation_span = maximum_historical_saturation - reversal_saturation
    if abs(saturation_span) > numerical_epsilon:
        raw_interpolation_factor = (saturation - reversal_saturation) / saturation_span
    else:
        raw_interpolation_factor = 0.0

    interpolation_factor = min(
        max(raw_interpolation_factor**scanning_exponent, 0.0), 1.0
    )
    scanning_curve_value = (
        drainage_curve_value
        + (imbibition_curve_value - drainage_curve_value) * interpolation_factor
    )

    on_primary_drainage = (is_imbibition < 0.5) and (
        abs(saturation - maximum_historical_saturation) < numerical_epsilon
    )
    on_primary_imbibition = (is_imbibition >= 0.5) and (
        abs(saturation - reversal_saturation) < numerical_epsilon
    )

    if on_primary_drainage:
        return drainage_curve_value
    if on_primary_imbibition:
        return imbibition_curve_value
    return scanning_curve_value


@numba.njit(cache=True)
def _compute_killough_scanning_curve_array(
    saturation: NumberOrArray[NDimension],
    drainage_curve_value: NumberOrArray[NDimension],
    imbibition_curve_value: NumberOrArray[NDimension],
    reversal_saturation: NumberOrArray[NDimension],
    maximum_historical_saturation: NumberOrArray[NDimension],
    is_imbibition: NumberOrArray[NDimension],
    scanning_exponent: Number = 1.0,
    numerical_epsilon: Number = 1e-12,
) -> NumberArray[NDimension]:
    """
    Killough scanning-curve interpolation between primary drainage and
    imbibition curves (array).

    :param saturation: Current saturation array.
    :param drainage_curve_value: Drainage curve values at current saturations.
    :param imbibition_curve_value: Imbibition curve values at current saturations.
    :param reversal_saturation: Saturations at the last reversal points.
    :param maximum_historical_saturation: Maximum saturations reached before the reversals.
    :param is_imbibition: Per-cell imbibition flag (1.0 = imbibition, 0.0 = drainage).
    :param scanning_exponent: Killough interpolation exponent *n* (1 = linear).
    :param numerical_epsilon: Numerical stability tolerance.
    :return: Interpolated scanning-curve values with the same shape as `saturation`.
    """
    saturation_array = atleast_1d(saturation)
    drainage_value_array = atleast_1d(drainage_curve_value)
    imbibition_value_array = atleast_1d(imbibition_curve_value)
    reversal_saturation_array = atleast_1d(reversal_saturation)
    maximum_saturation_array = atleast_1d(maximum_historical_saturation)
    imbibition_flag_array = atleast_1d(is_imbibition)

    # broadcast to common shape
    (
        saturation_array,
        drainage_value_array,
        imbibition_value_array,
        reversal_saturation_array,
        maximum_saturation_array,
        imbibition_flag_array,
    ) = np.broadcast_arrays(
        saturation_array,
        drainage_value_array,
        imbibition_value_array,
        reversal_saturation_array,
        maximum_saturation_array,
        imbibition_flag_array,
    )

    result = np.empty_like(saturation_array)
    for flat_index in numba.prange(saturation_array.size):  # type: ignore
        saturation_span = (
            maximum_saturation_array.flat[flat_index]
            - reversal_saturation_array.flat[flat_index]
        )
        if abs(saturation_span) > numerical_epsilon:
            raw_interpolation_factor = (
                saturation_array.flat[flat_index]
                - reversal_saturation_array.flat[flat_index]
            ) / saturation_span
        else:
            raw_interpolation_factor = 0.0
        interpolation_factor = min(
            max(raw_interpolation_factor**scanning_exponent, 0.0), 1.0
        )
        scanning_curve_value = (
            drainage_value_array.flat[flat_index]
            + (
                imbibition_value_array.flat[flat_index]
                - drainage_value_array.flat[flat_index]
            )
            * interpolation_factor
        )

        on_primary_drainage = (imbibition_flag_array.flat[flat_index] < 0.5) and (
            abs(
                saturation_array.flat[flat_index]
                - maximum_saturation_array.flat[flat_index]
            )
            < numerical_epsilon
        )
        on_primary_imbibition = (imbibition_flag_array.flat[flat_index] >= 0.5) and (
            abs(
                saturation_array.flat[flat_index]
                - reversal_saturation_array.flat[flat_index]
            )
            < numerical_epsilon
        )

        if on_primary_drainage:
            result.flat[flat_index] = drainage_value_array.flat[flat_index]
        elif on_primary_imbibition:
            result.flat[flat_index] = imbibition_value_array.flat[flat_index]
        else:
            result.flat[flat_index] = scanning_curve_value
    return typing.cast(NumberArray[NDimension], result)


def compute_killough_scanning_curve(
    saturation: NumberOrArray[NDimension],
    drainage_curve_value: NumberOrArray[NDimension],
    imbibition_curve_value: NumberOrArray[NDimension],
    reversal_saturation: NumberOrArray[NDimension],
    maximum_historical_saturation: NumberOrArray[NDimension],
    is_imbibition: NumberOrArray[NDimension],
    scanning_exponent: Number = 1.0,
    numerical_epsilon: Number = 1e-12,
) -> NumberOrArray[NDimension]:
    """
    Killough scanning-curve interpolation between primary drainage and
    imbibition curves.

    :param saturation: Current saturation (scalar or array).
    :param drainage_curve_value: Value from the primary drainage curve.
    :param imbibition_curve_value: Value from the primary imbibition curve.
    :param reversal_saturation: Saturation at the last reversal (scalar or array).
    :param maximum_historical_saturation: Maximum saturation before the reversal (scalar or array).
    :param is_imbibition: Imbibition flag - 1.0 for imbibition, 0.0 for drainage (scalar or array).
    :param scanning_exponent: Killough interpolation exponent *n* (1 = linear).
    :param numerical_epsilon: Numerical stability tolerance.
    :return: Interpolated scanning-curve value matching the shape of the input.
    """
    is_scalar = (
        np.isscalar(saturation)
        and np.isscalar(drainage_curve_value)
        and np.isscalar(imbibition_curve_value)
        and np.isscalar(reversal_saturation)
        and np.isscalar(maximum_historical_saturation)
        and np.isscalar(is_imbibition)
    )
    if is_scalar:
        return _compute_killough_scanning_curve_scalar(
            saturation=float(saturation),  # type: ignore[arg-type]
            drainage_curve_value=float(drainage_curve_value),  # type: ignore[arg-type]
            imbibition_curve_value=float(imbibition_curve_value),  # type: ignore[arg-type]
            reversal_saturation=float(reversal_saturation),  # type: ignore[arg-type]
            maximum_historical_saturation=float(maximum_historical_saturation),  # type: ignore[arg-type]
            is_imbibition=float(is_imbibition),  # type: ignore[arg-type]
            scanning_exponent=scanning_exponent,
            numerical_epsilon=numerical_epsilon,
        )
    return _compute_killough_scanning_curve_array(
        saturation=saturation,  # type: ignore[arg-type]
        drainage_curve_value=drainage_curve_value,  # type: ignore[arg-type]
        imbibition_curve_value=imbibition_curve_value,  # type: ignore[arg-type]
        reversal_saturation=reversal_saturation,  # type: ignore[arg-type]
        maximum_historical_saturation=maximum_historical_saturation,  # type: ignore[arg-type]
        is_imbibition=is_imbibition,  # type: ignore[arg-type]
        scanning_exponent=scanning_exponent,
        numerical_epsilon=numerical_epsilon,
    )


@numba.njit(cache=True, inline="always")
def _compute_killough_scanning_curve_derivative_scalar(
    saturation: Number,
    drainage_curve_value: Number,
    imbibition_curve_value: Number,
    drainage_curve_derivative: Number,
    imbibition_curve_derivative: Number,
    reversal_saturation: Number,
    maximum_historical_saturation: Number,
    is_imbibition: Number,
    scanning_exponent: Number = 1.0,
    numerical_epsilon: Number = 1e-12,
) -> Number:
    """
    Analytical derivative of the Killough scanning-curve value with respect
    to the scanning saturation (scalar).

    The scanning curve is:

        V(S) = V_d(S) + [V_i(S) - V_d(S)] * f(S)

    By the product / chain rule:

        dV/dS = dV_d/dS + [dV_i/dS - dV_d/dS] * f
                        + [V_i - V_d] * df/dS

    :param saturation: Current saturation.
    :param drainage_curve_value: Drainage curve value at current saturation.
    :param imbibition_curve_value: Imbibition curve value at current saturation.
    :param drainage_curve_derivative: Derivative of drainage value w.r.t. saturation.
    :param imbibition_curve_derivative: Derivative of imbibition value w.r.t. saturation.
    :param reversal_saturation: Saturation at the last reversal (S_rev).
    :param maximum_historical_saturation: Maximum saturation before the reversal (S_max).
    :param is_imbibition: 1.0 if imbibition, 0.0 if drainage.
    :param scanning_exponent: Killough exponent *n*.
    :param numerical_epsilon: Numerical tolerance.
    :return: Derivative of scanning-curve value w.r.t. saturation.
    """
    saturation_span = maximum_historical_saturation - reversal_saturation

    on_primary_drainage = (is_imbibition < 0.5) and (
        abs(saturation - maximum_historical_saturation) < numerical_epsilon
    )
    on_primary_imbibition = (is_imbibition >= 0.5) and (
        abs(saturation - reversal_saturation) < numerical_epsilon
    )
    if on_primary_drainage:
        return drainage_curve_derivative
    if on_primary_imbibition:
        return imbibition_curve_derivative

    if abs(saturation_span) > numerical_epsilon:
        raw_ratio = (saturation - reversal_saturation) / saturation_span
    else:
        raw_ratio = 0.0

    clamped_ratio = min(max(raw_ratio, 0.0), 1.0)
    interpolation_factor = clamped_ratio**scanning_exponent

    in_active_range = (
        (raw_ratio > 0.0)
        and (raw_ratio < 1.0)
        and (abs(saturation_span) > numerical_epsilon)
    )
    if in_active_range:
        if abs(scanning_exponent - 1.0) < 1e-10:
            interpolation_factor_derivative = 1.0 / saturation_span
        else:
            safe_ratio = clamped_ratio if clamped_ratio > 0.0 else 1e-30
            interpolation_factor_derivative = (
                scanning_exponent
                * (safe_ratio ** (scanning_exponent - 1.0))
                / saturation_span
            )
    else:
        interpolation_factor_derivative = 0.0

    return (
        drainage_curve_derivative
        + (imbibition_curve_derivative - drainage_curve_derivative)
        * interpolation_factor
        + (imbibition_curve_value - drainage_curve_value)
        * interpolation_factor_derivative
    )


@numba.njit(cache=True)
def _compute_killough_scanning_curve_derivative_array(
    saturation: NumberOrArray[NDimension],
    drainage_curve_value: NumberOrArray[NDimension],
    imbibition_curve_value: NumberOrArray[NDimension],
    drainage_curve_derivative: NumberOrArray[NDimension],
    imbibition_curve_derivative: NumberOrArray[NDimension],
    reversal_saturation: NumberOrArray[NDimension],
    maximum_historical_saturation: NumberOrArray[NDimension],
    is_imbibition: NumberOrArray[NDimension],
    scanning_exponent: Number = 1.0,
    numerical_epsilon: Number = 1e-12,
) -> NumberArray[NDimension]:
    """
    Analytical derivative of the Killough scanning-curve value with respect
    to the scanning saturation (array).

    :param saturation: Current saturation array.
    :param drainage_curve_value: Drainage curve values at current saturations.
    :param imbibition_curve_value: Imbibition curve values at current saturations.
    :param drainage_curve_derivative: Derivatives of drainage values w.r.t. saturation.
    :param imbibition_curve_derivative: Derivatives of imbibition values w.r.t. saturation.
    :param reversal_saturation: Saturations at the last reversal points.
    :param maximum_historical_saturation: Maximum saturations before the reversals.
    :param is_imbibition: Per-cell imbibition flags (1.0 / 0.0).
    :param scanning_exponent: Killough exponent *n*.
    :param numerical_epsilon: Numerical tolerance.
    :return: Derivative array with the same shape as `saturation`.
    """
    saturation_array = atleast_1d(saturation)
    drainage_value_array = atleast_1d(drainage_curve_value)
    imbibition_value_array = atleast_1d(imbibition_curve_value)
    drainage_derivative_array = atleast_1d(drainage_curve_derivative)
    imbibition_derivative_array = atleast_1d(imbibition_curve_derivative)
    reversal_saturation_array = atleast_1d(reversal_saturation)
    maximum_saturation_array = atleast_1d(maximum_historical_saturation)
    imbibition_flag_array = atleast_1d(is_imbibition)

    (
        saturation_array,
        drainage_value_array,
        imbibition_value_array,
        drainage_derivative_array,
        imbibition_derivative_array,
        reversal_saturation_array,
        maximum_saturation_array,
        imbibition_flag_array,
    ) = np.broadcast_arrays(
        saturation_array,
        drainage_value_array,
        imbibition_value_array,
        drainage_derivative_array,
        imbibition_derivative_array,
        reversal_saturation_array,
        maximum_saturation_array,
        imbibition_flag_array,
    )

    result = np.empty_like(saturation_array)
    for flat_index in numba.prange(saturation_array.size):  # type: ignore
        result.flat[flat_index] = _compute_killough_scanning_curve_derivative_scalar(
            saturation=saturation_array.flat[flat_index],
            drainage_curve_value=drainage_value_array.flat[flat_index],
            imbibition_curve_value=imbibition_value_array.flat[flat_index],
            drainage_curve_derivative=drainage_derivative_array.flat[flat_index],
            imbibition_curve_derivative=imbibition_derivative_array.flat[flat_index],
            reversal_saturation=reversal_saturation_array.flat[flat_index],
            maximum_historical_saturation=maximum_saturation_array.flat[flat_index],
            is_imbibition=imbibition_flag_array.flat[flat_index],
            scanning_exponent=scanning_exponent,
            numerical_epsilon=numerical_epsilon,
        )
    return typing.cast(NumberArray[NDimension], result)


def compute_killough_scanning_curve_derivative(
    saturation: NumberOrArray[NDimension],
    drainage_curve_value: NumberOrArray[NDimension],
    imbibition_curve_value: NumberOrArray[NDimension],
    drainage_curve_derivative: NumberOrArray[NDimension],
    imbibition_curve_derivative: NumberOrArray[NDimension],
    reversal_saturation: NumberOrArray[NDimension],
    maximum_historical_saturation: NumberOrArray[NDimension],
    is_imbibition: NumberOrArray[NDimension],
    scanning_exponent: Number = 1.0,
    numerical_epsilon: Number = 1e-12,
) -> NumberOrArray[NDimension]:
    """
    Analytical derivative of the Killough scanning-curve value with respect
    to the scanning saturation.

    :param saturation: Current saturation (scalar or array).
    :param drainage_curve_value: Drainage curve value at current saturation.
    :param imbibition_curve_value: Imbibition curve value at current saturation.
    :param drainage_curve_derivative: Derivative of drainage value w.r.t. saturation.
    :param imbibition_curve_derivative: Derivative of imbibition value w.r.t. saturation.
    :param reversal_saturation: Saturation at the last reversal (scalar or array).
    :param maximum_historical_saturation: Maximum saturation before the reversal (scalar or array).
    :param is_imbibition: Imbibition flag (scalar or array).
    :param scanning_exponent: Killough exponent *n*.
    :param numerical_epsilon: Numerical tolerance.
    :return: Derivative of scanning-curve value w.r.t. saturation, matching the shape of the input.
    """
    is_scalar = (
        np.isscalar(saturation)
        and np.isscalar(drainage_curve_value)
        and np.isscalar(imbibition_curve_value)
        and np.isscalar(drainage_curve_derivative)
        and np.isscalar(imbibition_curve_derivative)
        and np.isscalar(reversal_saturation)
        and np.isscalar(maximum_historical_saturation)
        and np.isscalar(is_imbibition)
    )
    if is_scalar:
        return _compute_killough_scanning_curve_derivative_scalar(
            saturation=float(saturation),  # type: ignore[arg-type]
            drainage_curve_value=float(drainage_curve_value),  # type: ignore[arg-type]
            imbibition_curve_value=float(imbibition_curve_value),  # type: ignore[arg-type]
            drainage_curve_derivative=float(drainage_curve_derivative),  # type: ignore[arg-type]
            imbibition_curve_derivative=float(imbibition_curve_derivative),  # type: ignore[arg-type]
            reversal_saturation=float(reversal_saturation),  # type: ignore[arg-type]
            maximum_historical_saturation=float(maximum_historical_saturation),  # type: ignore[arg-type]
            is_imbibition=float(is_imbibition),  # type: ignore[arg-type]
            scanning_exponent=scanning_exponent,
            numerical_epsilon=numerical_epsilon,
        )
    return _compute_killough_scanning_curve_derivative_array(
        saturation=saturation,  # type: ignore[arg-type]
        drainage_curve_value=drainage_curve_value,  # type: ignore[arg-type]
        imbibition_curve_value=imbibition_curve_value,  # type: ignore[arg-type]
        drainage_curve_derivative=drainage_curve_derivative,  # type: ignore[arg-type]
        imbibition_curve_derivative=imbibition_curve_derivative,  # type: ignore[arg-type]
        reversal_saturation=reversal_saturation,  # type: ignore[arg-type]
        maximum_historical_saturation=maximum_historical_saturation,  # type: ignore[arg-type]
        is_imbibition=is_imbibition,  # type: ignore[arg-type]
        scanning_exponent=scanning_exponent,
        numerical_epsilon=numerical_epsilon,
    )


def _get_oil_water_relative_permeabilities(
    oil_water_table: typing.Union[TwoPhaseRelPermTable, RelativePermeabilityTable],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    **kwargs: typing.Any,
) -> typing.Tuple[NumberOrArray[NDimension], NumberOrArray[NDimension]]:
    """
    Return `(krw, kro_w)` from an oil-water relative permeability table.

    Dispatches saturations correctly for both `TwoPhaseRelPermTable` and
    full `RelativePermeabilityTable` instances by calling
    `table.get_oil_water_wetting_phase()` rather than inspecting any internal attribute.

    :param oil_water_table: Oil-water relative permeability table (two-phase or three-phase).
    :param water_saturation: Water saturation (scalar or array).
    :param oil_saturation: Oil saturation (scalar or array).
    :param gas_saturation: Gas saturation (scalar or array).
    :param kwargs: Additional keyword arguments forwarded to parametric tables.
    :return: Tuple of `(krw, kro_w)` - water and oil relative permeabilities
        from the oil-water sub-system.
    """
    oil_water_wetting_phase = oil_water_table.get_oil_water_wetting_phase()

    if isinstance(oil_water_table, TwoPhaseRelPermTable):
        if oil_water_wetting_phase == FluidPhase.WATER:
            wetting_saturation = water_saturation
            non_wetting_saturation = oil_saturation
        else:
            wetting_saturation = oil_saturation
            non_wetting_saturation = water_saturation

        water_relative_permeability = (
            oil_water_table.get_wetting_phase_relative_permeability(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
            if oil_water_wetting_phase == FluidPhase.WATER
            else oil_water_table.get_non_wetting_phase_relative_permeability(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
        )
        oil_relative_permeability = (
            oil_water_table.get_non_wetting_phase_relative_permeability(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
            if oil_water_wetting_phase == FluidPhase.WATER
            else oil_water_table.get_wetting_phase_relative_permeability(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
        )
        return water_relative_permeability, oil_relative_permeability

    # Full three-phase table
    result = oil_water_table.get_relative_permeabilities(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        **kwargs,
    )
    return result["water"], result["oil"]


def _get_gas_oil_relative_permeabilities(
    gas_oil_table: typing.Union[TwoPhaseRelPermTable, RelativePermeabilityTable],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    **kwargs: typing.Any,
) -> typing.Tuple[NumberOrArray[NDimension], NumberOrArray[NDimension]]:
    """
    Return `(kro_g, krg)` from a gas-oil relative permeability table.

    Dispatches saturations correctly for both `TwoPhaseRelPermTable` and
    full `RelativePermeabilityTable` instances by calling
    `table.get_gas_oil_wetting_phase()` rather than inspecting any internal attribute.

    :param gas_oil_table: Gas-oil relative permeability table (two-phase or three-phase).
    :param water_saturation: Water saturation (scalar or array).
    :param oil_saturation: Oil saturation (scalar or array).
    :param gas_saturation: Gas saturation (scalar or array).
    :param kwargs: Additional keyword arguments forwarded to parametric tables.
    :return: Tuple of `(kro_g, krg)` - oil and gas relative permeabilities
        from the gas-oil sub-system.
    """
    gas_oil_wetting_phase = gas_oil_table.get_gas_oil_wetting_phase()

    if isinstance(gas_oil_table, TwoPhaseRelPermTable):
        if gas_oil_wetting_phase == FluidPhase.OIL:
            wetting_saturation = oil_saturation
            non_wetting_saturation = gas_saturation
        else:
            wetting_saturation = gas_saturation
            non_wetting_saturation = oil_saturation

        oil_relative_permeability = (
            gas_oil_table.get_wetting_phase_relative_permeability(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
            if gas_oil_wetting_phase == FluidPhase.OIL
            else gas_oil_table.get_non_wetting_phase_relative_permeability(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
        )
        gas_relative_permeability = (
            gas_oil_table.get_non_wetting_phase_relative_permeability(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
            if gas_oil_wetting_phase == FluidPhase.OIL
            else gas_oil_table.get_wetting_phase_relative_permeability(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
        )
        return oil_relative_permeability, gas_relative_permeability

    result = gas_oil_table.get_relative_permeabilities(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        **kwargs,
    )
    return result["oil"], result["gas"]


def _get_oil_water_relative_permeability_derivatives(
    oil_water_table: typing.Union[TwoPhaseRelPermTable, RelativePermeabilityTable],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    **kwargs: typing.Any,
) -> typing.Tuple[NumberOrArray[NDimension], NumberOrArray[NDimension]]:
    """
    Return `(d_krw/d_ref, d_kro_w/d_ref)` for the oil-water table, where
    *ref* is the table's natural reference saturation (Sw in water-wet, So in
    oil-wet).

    :param oil_water_table: Oil-water relative permeability table (two-phase or three-phase).
    :param water_saturation: Water saturation (scalar or array).
    :param oil_saturation: Oil saturation (scalar or array).
    :param gas_saturation: Gas saturation (scalar or array).
    :param kwargs: Additional keyword arguments forwarded to parametric tables.
    :return: Tuple of `(d_krw/d_ref, d_kro_w/d_ref)` where *ref* is the
        reference saturation axis of the oil-water sub-system.
    """
    oil_water_wetting_phase = oil_water_table.get_oil_water_wetting_phase()

    if isinstance(oil_water_table, TwoPhaseRelPermTable):
        if oil_water_wetting_phase == FluidPhase.WATER:
            wetting_saturation = water_saturation
            non_wetting_saturation = oil_saturation
        else:
            wetting_saturation = oil_saturation
            non_wetting_saturation = water_saturation

        wetting_derivative = (
            oil_water_table.get_wetting_phase_relative_permeability_derivative(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
        )
        non_wetting_derivative = (
            oil_water_table.get_non_wetting_phase_relative_permeability_derivative(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
        )
        if oil_water_wetting_phase == FluidPhase.WATER:
            return (
                wetting_derivative,
                non_wetting_derivative,
            )  # d_krw/d_Sw, d_kro_w/d_Sw
        return non_wetting_derivative, wetting_derivative  # d_krw/d_So, d_kro_w/d_So

    derivatives = oil_water_table.get_relative_permeability_derivatives(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        **kwargs,
    )
    # The oil-water hysteresis scanning variable is always Sw
    return derivatives["dKrw_dSw"], derivatives["dKro_dSw"]


def _get_gas_oil_relative_permeability_derivatives(
    gas_oil_table: typing.Union[TwoPhaseRelPermTable, RelativePermeabilityTable],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    **kwargs: typing.Any,
) -> typing.Tuple[NumberOrArray[NDimension], NumberOrArray[NDimension]]:
    """
    Return `(d_kro_g/d_ref, d_krg/d_ref)` for the gas-oil table, where
    *ref* is the table's natural reference saturation (So in oil-wet, Sg in
    gas-wet).

    :param gas_oil_table: Gas-oil relative permeability table (two-phase or three-phase).
    :param water_saturation: Water saturation (scalar or array).
    :param oil_saturation: Oil saturation (scalar or array).
    :param gas_saturation: Gas saturation (scalar or array).
    :param kwargs: Additional keyword arguments forwarded to parametric tables.
    :return: Tuple of `(d_kro_g/d_ref, d_krg/d_ref)` where *ref* is the
        reference saturation axis of the gas-oil sub-system.
    """
    gas_oil_wetting_phase = gas_oil_table.get_gas_oil_wetting_phase()

    if isinstance(gas_oil_table, TwoPhaseRelPermTable):
        if gas_oil_wetting_phase == FluidPhase.OIL:
            wetting_saturation = oil_saturation
            non_wetting_saturation = gas_saturation
        else:
            wetting_saturation = gas_saturation
            non_wetting_saturation = oil_saturation

        wetting_derivative = (
            gas_oil_table.get_wetting_phase_relative_permeability_derivative(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
        )
        non_wetting_derivative = (
            gas_oil_table.get_non_wetting_phase_relative_permeability_derivative(
                wetting_saturation=wetting_saturation,
                non_wetting_saturation=non_wetting_saturation,
            )
        )
        if gas_oil_wetting_phase == FluidPhase.OIL:
            return (
                wetting_derivative,
                non_wetting_derivative,
            )  # d_kro_g/d_So, d_krg/d_So
        return non_wetting_derivative, wetting_derivative  # d_kro_g/d_Sg, d_krg/d_Sg

    derivatives = gas_oil_table.get_relative_permeability_derivatives(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        **kwargs,
    )
    # The gas-oil hysteresis scanning variable is always Sg
    return derivatives["dKro_dSg"], derivatives["dKrg_dSg"]


def _get_oil_water_capillary_pressure(
    oil_water_capillary_pressure_table: typing.Union[
        TwoPhaseCapillaryPressureTable, CapillaryPressureTable
    ],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    **kwargs: typing.Any,
) -> NumberOrArray[NDimension]:
    """
    Extract Pcow from an oil-water capillary pressure table, dispatching
    saturations correctly via the canonical wetting-phase API.

    :param oil_water_capillary_pressure_table: Oil-water capillary pressure table (two-phase or three-phase).
    :param water_saturation: Water saturation (scalar or array).
    :param oil_saturation: Oil saturation (scalar or array).
    :param gas_saturation: Gas saturation (scalar or array).
    :param kwargs: Additional keyword arguments forwarded to parametric tables.
    :return: Oil-water capillary pressure Pcow = Po - Pw (scalar or array).
    """
    oil_water_wetting_phase = (
        oil_water_capillary_pressure_table.get_oil_water_wetting_phase()
    )

    if isinstance(oil_water_capillary_pressure_table, TwoPhaseCapillaryPressureTable):
        if oil_water_wetting_phase == FluidPhase.WATER:
            wetting_saturation = water_saturation
            non_wetting_saturation = oil_saturation
        else:
            wetting_saturation = oil_saturation
            non_wetting_saturation = water_saturation
        return oil_water_capillary_pressure_table.get_capillary_pressure(
            wetting_saturation=wetting_saturation,
            non_wetting_saturation=non_wetting_saturation,
        )

    result = oil_water_capillary_pressure_table.get_capillary_pressures(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        **kwargs,
    )
    return result["oil_water"]


def _get_gas_oil_capillary_pressure(
    gas_oil_capillary_pressure_table: typing.Union[
        TwoPhaseCapillaryPressureTable, CapillaryPressureTable
    ],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    **kwargs: typing.Any,
) -> NumberOrArray[NDimension]:
    """
    Extract Pcgo from a gas-oil capillary pressure table, dispatching
    saturations correctly via the canonical wetting-phase API.

    :param gas_oil_capillary_pressure_table: Gas-oil capillary pressure table (two-phase or three-phase).
    :param water_saturation: Water saturation (scalar or array).
    :param oil_saturation: Oil saturation (scalar or array).
    :param gas_saturation: Gas saturation (scalar or array).
    :param kwargs: Additional keyword arguments forwarded to parametric tables.
    :return: Gas-oil capillary pressure Pcgo = Pg - Po (scalar or array).
    """
    gas_oil_wetting_phase = gas_oil_capillary_pressure_table.get_gas_oil_wetting_phase()

    if isinstance(gas_oil_capillary_pressure_table, TwoPhaseCapillaryPressureTable):
        if gas_oil_wetting_phase == FluidPhase.OIL:
            wetting_saturation = oil_saturation
            non_wetting_saturation = gas_saturation
        else:
            wetting_saturation = gas_saturation
            non_wetting_saturation = oil_saturation
        return gas_oil_capillary_pressure_table.get_capillary_pressure(
            wetting_saturation=wetting_saturation,
            non_wetting_saturation=non_wetting_saturation,
        )

    result = gas_oil_capillary_pressure_table.get_capillary_pressures(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        **kwargs,
    )
    return result["gas_oil"]


def _get_oil_water_capillary_pressure_derivative(
    oil_water_capillary_pressure_table: typing.Union[
        TwoPhaseCapillaryPressureTable, CapillaryPressureTable
    ],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    **kwargs: typing.Any,
) -> NumberOrArray[NDimension]:
    """
    Return dPcow/d(reference_sat) for the oil-water capillary pressure table.

    For `TwoPhaseCapillaryPressureTable` this is the derivative w.r.t. the
    table's own reference saturation (Sw in water-wet, So in oil-wet).  For
    three-phase tables `dPcow/dSw` is returned to match the scanning variable
    used by the hysteresis layer.

    :param oil_water_capillary_pressure_table: Oil-water capillary pressure table (two-phase or three-phase).
    :param water_saturation: Water saturation (scalar or array).
    :param oil_saturation: Oil saturation (scalar or array).
    :param gas_saturation: Gas saturation (scalar or array).
    :param kwargs: Additional keyword arguments forwarded to parametric tables.
    :return: Derivative of Pcow w.r.t. the reference saturation axis (scalar or array).
    """
    oil_water_wetting_phase = (
        oil_water_capillary_pressure_table.get_oil_water_wetting_phase()
    )

    if isinstance(oil_water_capillary_pressure_table, TwoPhaseCapillaryPressureTable):
        if oil_water_wetting_phase == FluidPhase.WATER:
            wetting_saturation = water_saturation
            non_wetting_saturation = oil_saturation
        else:
            wetting_saturation = oil_saturation
            non_wetting_saturation = water_saturation
        return oil_water_capillary_pressure_table.get_capillary_pressure_derivative(
            wetting_saturation=wetting_saturation,
            non_wetting_saturation=non_wetting_saturation,
        )

    derivatives = oil_water_capillary_pressure_table.get_capillary_pressure_derivatives(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        **kwargs,
    )
    return derivatives["dPcow_dSw"]


def _get_gas_oil_capillary_pressure_derivative(
    gas_oil_capillary_pressure_table: typing.Union[
        TwoPhaseCapillaryPressureTable, CapillaryPressureTable
    ],
    water_saturation: NumberOrArray[NDimension],
    oil_saturation: NumberOrArray[NDimension],
    gas_saturation: NumberOrArray[NDimension],
    **kwargs: typing.Any,
) -> NumberOrArray[NDimension]:
    """
    Return dPcgo/d(reference_sat) for the gas-oil capillary pressure table.

    For `TwoPhaseCapillaryPressureTable` this is the derivative w.r.t. the
    table's own reference saturation.  For three-phase tables
    `dPcgo/dSg` is returned to match the scanning variable used by the
    hysteresis layer.

    :param gas_oil_capillary_pressure_table: Gas-oil capillary pressure table (two-phase or three-phase).
    :param water_saturation: Water saturation (scalar or array).
    :param oil_saturation: Oil saturation (scalar or array).
    :param gas_saturation: Gas saturation (scalar or array).
    :param kwargs: Additional keyword arguments forwarded to parametric tables.
    :return: Derivative of Pcgo w.r.t. the reference saturation axis
        (scalar or array).
    """
    gas_oil_wetting_phase = gas_oil_capillary_pressure_table.get_gas_oil_wetting_phase()

    if isinstance(gas_oil_capillary_pressure_table, TwoPhaseCapillaryPressureTable):
        if gas_oil_wetting_phase == FluidPhase.OIL:
            wetting_saturation = oil_saturation
            non_wetting_saturation = gas_saturation
        else:
            wetting_saturation = gas_saturation
            non_wetting_saturation = oil_saturation
        return gas_oil_capillary_pressure_table.get_capillary_pressure_derivative(
            wetting_saturation=wetting_saturation,
            non_wetting_saturation=non_wetting_saturation,
        )

    derivatives = gas_oil_capillary_pressure_table.get_capillary_pressure_derivatives(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        **kwargs,
    )
    return derivatives["dPcgo_dSg"]


@relperm_table
@attrs.frozen
class KilloughLandRelPermTable(
    RelativePermeabilityTable,
    serializers={"mixing_rule": serialize_mixing_rule},
    deserializers={"mixing_rule": get_mixing_rule},
    load_exclude={"supports_vector"},
    dump_exclude={"supports_vector"},
):
    """
    Implements the Killough relative permeability hysteresis model/table with Land trapping.

    During *primary drainage* the relative permeabilities follow the
    `oil_water_drainage_table` and `gas_oil_drainage_table`.

    When the flow reverses (*imbibition*) two things happen simultaneously.

    **Land trapping**: a portion of the non-wetting phase becomes
    disconnected. The dynamic residual saturation depends on the saturation
    at the reversal point via Land's formula:

        S_r(S_i) = S_r_max / (1 + C * S_i)

    **Killough scanning curves**: between the reversal point and the
    maximum historical saturation, kr follows a scanning curve that
    interpolates between the primary drainage and imbibition bounds.

    Both two-phase (`TwoPhaseRelPermTable`) and full three-phase
    (`RelativePermeabilityTable`) backing tables are supported. Wetting
    and non-wetting phase roles are resolved through the canonical API
    (`get_oil_water_wetting_phase` / `get_gas_oil_wetting_phase`) so the
    model is wettability-agnostic.

    The hysteresis history is passed as additional keyword arguments to
    `get_relative_permeabilities` and `get_relative_permeability_derivatives`. When these arguments are
    absent the model degenerates to the primary drainage curves, which is the
    physically correct behaviour for the first drainage cycle
    (see `simulate.py`, the `enable_hysteresis` flag in `Config` controls
    whether history is tracked and passed through).
    """

    __type__ = "killough_land_relperm_model"

    oil_water_drainage_table: RelativePermeabilityTable
    """Primary drainage relative permeability table for the oil-water system."""

    gas_oil_drainage_table: RelativePermeabilityTable
    """Primary drainage relative permeability table for the gas-oil system."""

    oil_water_imbibition_table: typing.Optional[RelativePermeabilityTable] = None
    """Primary imbibition table for the oil-water system. Defaults to the drainage table."""

    gas_oil_imbibition_table: typing.Optional[RelativePermeabilityTable] = None
    """Primary imbibition table for the gas-oil system. Defaults to the drainage table."""

    land_trapping_coefficient_water: Number = 1.0
    """Land trapping coefficient *C* for the oil-water system (>= 0)."""

    land_trapping_coefficient_gas: Number = 1.0
    """Land trapping coefficient *C* for the gas-oil system (>= 0)."""

    maximum_residual_oil_saturation_water: typing.Optional[float] = None
    """
    Maximum residual oil saturation S_r_max used by Land's formula for the
    oil-water system. Required when `oil_water_imbibition_table` is set.
    """

    maximum_residual_oil_saturation_gas: typing.Optional[float] = None
    """
    Maximum residual oil saturation S_r_max for the gas-oil system. Required
    when `gas_oil_imbibition_table` is set.
    """

    maximum_residual_gas_saturation: typing.Optional[float] = None
    """
    Maximum residual gas saturation S_r_max used by Land's formula. Required
    when `gas_oil_imbibition_table` is set.
    """

    scanning_interpolation_exponent: Number = 1.0
    """Killough scanning curve interpolation exponent *n* (1 = linear)."""

    mixing_rule: typing.Union[MixingRule, str] = "eclipse_rule"
    """Three-phase oil relative permeability mixing rule."""

    supports_vector: bool = attrs.field(init=False, repr=False, default=True)

    def __attrs_post_init__(self) -> None:
        if isinstance(self.mixing_rule, str):
            object.__setattr__(self, "mixing_rule", get_mixing_rule(self.mixing_rule))

        if isinstance(self.oil_water_drainage_table, TwoPhaseRelPermTable) and {
            self.oil_water_drainage_table.wetting_phase,
            self.oil_water_drainage_table.non_wetting_phase,
        } != {FluidPhase.WATER, FluidPhase.OIL}:
            raise ValidationError(
                "`oil_water_drainage_table` must involve water and oil phases."
            )

        if isinstance(self.gas_oil_drainage_table, TwoPhaseRelPermTable) and {
            self.gas_oil_drainage_table.wetting_phase,
            self.gas_oil_drainage_table.non_wetting_phase,
        } != {FluidPhase.OIL, FluidPhase.GAS}:
            raise ValidationError(
                "`gas_oil_drainage_table` must involve oil and gas phases."
            )

        if (
            self.oil_water_imbibition_table is not None
            and isinstance(self.oil_water_imbibition_table, TwoPhaseRelPermTable)
            and {
                self.oil_water_imbibition_table.wetting_phase,
                self.oil_water_imbibition_table.non_wetting_phase,
            }
            != {FluidPhase.WATER, FluidPhase.OIL}
        ):
            raise ValidationError(
                "`oil_water_imbibition_table` must involve water and oil phases."
            )

        if (
            self.gas_oil_imbibition_table is not None
            and isinstance(self.gas_oil_imbibition_table, TwoPhaseRelPermTable)
            and {
                self.gas_oil_imbibition_table.wetting_phase,
                self.gas_oil_imbibition_table.non_wetting_phase,
            }
            != {FluidPhase.OIL, FluidPhase.GAS}
        ):
            raise ValidationError(
                "`gas_oil_imbibition_table` must involve oil and gas phases."
            )

    def get_oil_water_wetting_phase(self) -> FluidPhase:
        """
        Return the wetting phase for the oil-water sub-system.

        :return: `FluidPhase.WATER` for water-wet or `FluidPhase.OIL` for
            oil-wet systems, as reported by the drainage table.
        """
        return self.oil_water_drainage_table.get_oil_water_wetting_phase()

    def get_gas_oil_wetting_phase(self) -> FluidPhase:
        """
        Return the wetting phase for the gas-oil sub-system.

        :return: `FluidPhase.OIL` for oil-wet or `FluidPhase.GAS` for
            gas-wet systems, as reported by the drainage table.
        """
        return self.gas_oil_drainage_table.get_gas_oil_wetting_phase()

    def get_oil_relperm_endpoint(self) -> Number:
        """Oil relative permeability endpoint from the primary drainage table."""
        return self.oil_water_drainage_table.get_oil_relperm_endpoint()

    def get_water_relperm_endpoint(self) -> Number:
        """Water relative permeability endpoint from the primary drainage table."""
        return self.oil_water_drainage_table.get_water_relperm_endpoint()

    def get_gas_relperm_endpoint(self) -> Number:
        """Gas relative permeability endpoint from the primary drainage table."""
        return self.gas_oil_drainage_table.get_gas_relperm_endpoint()

    def _parse_hysteresis_kwargs(
        self,
        water_saturation: npt.NDArray,
        gas_saturation: npt.NDArray,
        max_water_saturation: typing.Optional[NumberOrArray[NDimension]],
        max_gas_saturation: typing.Optional[NumberOrArray[NDimension]],
        water_imbibition_flag: typing.Optional[
            typing.Union[bool, BooleanArray[NDimension]]
        ],
        gas_imbibition_flag: typing.Optional[
            typing.Union[bool, BooleanArray[NDimension]]
        ],
        water_reversal_saturation: typing.Optional[NumberOrArray[NDimension]],
        gas_reversal_saturation: typing.Optional[NumberOrArray[NDimension]],
    ) -> typing.Tuple[
        npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
    ]:
        """
        Parse and broadcast saturation-history arrays.

        When all history arguments are `None` the method returns arrays that
        replicate the primary-drainage state (no-hysteresis fallback).

        :param water_saturation: Broadcast-ready water saturation array.
        :param gas_saturation: Broadcast-ready gas saturation array.
        :param max_water_saturation: Historical maximum water saturation grid or `None`.
        :param max_gas_saturation: Historical maximum gas saturation grid or `None`.
        :param water_imbibition_flag: Per-cell flag: 1 = water imbibition, 0 = drainage,
            or `None`.
        :param gas_imbibition_flag: Per-cell flag: 1 = gas imbibition, 0 = drainage,
            or `None`.
        :param water_reversal_saturation: Water saturation at last oil-water reversal,
            or `None` (defaults to `max_water_saturation`).
        :param gas_reversal_saturation: Gas saturation at last gas-oil reversal, or
            `None` (defaults to `max_gas_saturation`).
        :return: Six broadcast-compatible arrays: `(sw_max, sg_max, sw_imb, sg_imb,
            sw_rev, sg_rev)`.
        """
        use_hysteresis = (
            max_water_saturation is not None
            and max_gas_saturation is not None
            and water_imbibition_flag is not None
            and gas_imbibition_flag is not None
        )
        if use_hysteresis:
            maximum_water_saturation = np.atleast_1d(max_water_saturation)  # type: ignore
            maximum_gas_saturation = np.atleast_1d(max_gas_saturation)  # type: ignore
            water_imbibition_flag = np.atleast_1d(water_imbibition_flag)  # type: ignore
            gas_imbibition_flag = np.atleast_1d(gas_imbibition_flag)  # type: ignore
            water_reversal_saturation = (
                np.atleast_1d(water_reversal_saturation)
                if water_reversal_saturation is not None
                else maximum_water_saturation.copy()
            )
            gas_reversal_saturation = (
                np.atleast_1d(gas_reversal_saturation)
                if gas_reversal_saturation is not None
                else maximum_gas_saturation.copy()
            )
        else:
            maximum_water_saturation = water_saturation.copy()
            maximum_gas_saturation = gas_saturation.copy()
            water_imbibition_flag = np.zeros_like(water_saturation)
            gas_imbibition_flag = np.zeros_like(gas_saturation)
            water_reversal_saturation = water_saturation.copy()
            gas_reversal_saturation = gas_saturation.copy()

        (
            maximum_water_saturation,
            maximum_gas_saturation,
            water_imbibition_flag,
            gas_imbibition_flag,
            water_reversal_saturation,
            gas_reversal_saturation,
        ) = np.broadcast_arrays(
            maximum_water_saturation,
            maximum_gas_saturation,
            water_imbibition_flag,  # type: ignore
            gas_imbibition_flag,  # type: ignore
            water_reversal_saturation,
            gas_reversal_saturation,
        )
        return (
            maximum_water_saturation,
            maximum_gas_saturation,
            water_imbibition_flag,
            gas_imbibition_flag,
            water_reversal_saturation,
            gas_reversal_saturation,
        )

    def get_relative_permeabilities(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        max_water_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
        max_gas_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
        water_imbibition_flag: typing.Optional[
            typing.Union[bool, BooleanArray[NDimension]]
        ] = None,
        gas_imbibition_flag: typing.Optional[
            typing.Union[bool, BooleanArray[NDimension]]
        ] = None,
        water_reversal_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
        gas_reversal_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Compute three-phase relative permeabilities with Killough/Land hysteresis.

        When the saturation-history keyword arguments are absent the method
        returns primary drainage kr values (no hysteresis).

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param max_water_saturation: Historical maximum water saturation per cell.
            Enables hysteresis when provided together with the other history args.
        :param max_gas_saturation: Historical maximum gas saturation per cell.
        :param water_imbibition_flag: Boolean / float flag per cell - 1 if water
            saturation is currently increasing (imbibition in the oil-water
            system), 0 if decreasing (drainage).
        :param gas_imbibition_flag: Boolean / float flag per cell - 1 if gas
            saturation is currently increasing, 0 if decreasing.
        :param water_reversal_saturation: Water saturation at the last oil-water
            reversal.  Defaults to `max_water_saturation` when not supplied.
        :param gas_reversal_saturation: Gas saturation at the last gas-oil reversal.
            Defaults to `max_gas_saturation` when not supplied.
        :param kwargs: Additional keyword arguments forwarded to the underlying
            backing tables (e.g. residual saturation overrides for parametric
            models).
        :return: `RelativePermeabilities` dictionary with keys `"water"`, `"oil"`, and `"gas"`.
        """
        is_scalar = (
            np.isscalar(water_saturation)
            and np.isscalar(oil_saturation)
            and np.isscalar(gas_saturation)
        )
        water_saturation = np.atleast_1d(water_saturation)
        oil_saturation = np.atleast_1d(oil_saturation)
        gas_saturation = np.atleast_1d(gas_saturation)
        water_saturation, oil_saturation, gas_saturation = np.broadcast_arrays(
            water_saturation, oil_saturation, gas_saturation
        )

        # Normalise saturations
        total_saturation = water_saturation + oil_saturation + gas_saturation
        normalisation_mask = (np.abs(total_saturation - 1.0) > c.SATURATION_EPSILON) & (
            total_saturation > 0.0
        )
        if np.any(normalisation_mask):
            water_saturation = np.where(
                normalisation_mask,
                water_saturation / total_saturation,
                water_saturation,
            )
            oil_saturation = np.where(
                normalisation_mask,
                oil_saturation / total_saturation,
                oil_saturation,
            )
            gas_saturation = np.where(
                normalisation_mask,
                gas_saturation / total_saturation,
                gas_saturation,
            )

        (
            maximum_water_saturation,
            maximum_gas_saturation,
            water_imbibition_flag,
            gas_imbibition_flag,
            water_reversal_saturation,
            gas_reversal_saturation,
        ) = self._parse_hysteresis_kwargs(
            water_saturation=water_saturation,
            gas_saturation=gas_saturation,
            max_water_saturation=max_water_saturation,
            max_gas_saturation=max_gas_saturation,
            water_imbibition_flag=water_imbibition_flag,
            gas_imbibition_flag=gas_imbibition_flag,
            water_reversal_saturation=water_reversal_saturation,
            gas_reversal_saturation=gas_reversal_saturation,
        )

        oil_water_drainage_table = self.oil_water_drainage_table
        oil_water_imbibition_table = (
            self.oil_water_imbibition_table or oil_water_drainage_table
        )
        gas_oil_drainage_table = self.gas_oil_drainage_table
        gas_oil_imbibition_table = (
            self.gas_oil_imbibition_table or gas_oil_drainage_table
        )
        use_hysteresis = (
            max_water_saturation is not None and max_gas_saturation is not None
        )

        # Oil-water system - Land trapping on oil
        oil_saturation_at_oil_water_reversal = np.maximum(
            0.0,
            1.0 - water_reversal_saturation - gas_saturation,
        )
        imbibition_oil_water_kwargs = dict(kwargs)
        if use_hysteresis and self.maximum_residual_oil_saturation_water is not None:
            dynamic_residual_oil_saturation_water = compute_land_residual_saturation(
                initial_non_wetting_saturation=oil_saturation_at_oil_water_reversal,
                maximum_residual_saturation=self.maximum_residual_oil_saturation_water,
                land_trapping_coefficient=self.land_trapping_coefficient_water,
            )
            imbibition_oil_water_kwargs["residual_oil_saturation_water"] = (
                dynamic_residual_oil_saturation_water
            )

        (
            water_relative_permeability_drainage,
            oil_relative_permeability_water_drainage,
        ) = _get_oil_water_relative_permeabilities(
            oil_water_table=oil_water_drainage_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        (
            water_relative_permeability_imbibition,
            oil_relative_permeability_water_imbibition,
        ) = _get_oil_water_relative_permeabilities(
            oil_water_table=oil_water_imbibition_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **imbibition_oil_water_kwargs,
        )

        water_relative_permeability = compute_killough_scanning_curve(
            saturation=water_saturation,
            drainage_curve_value=water_relative_permeability_drainage,
            imbibition_curve_value=water_relative_permeability_imbibition,
            reversal_saturation=water_reversal_saturation,
            maximum_historical_saturation=maximum_water_saturation,
            is_imbibition=water_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )
        oil_relative_permeability_water = compute_killough_scanning_curve(
            saturation=water_saturation,
            drainage_curve_value=oil_relative_permeability_water_drainage,
            imbibition_curve_value=oil_relative_permeability_water_imbibition,
            reversal_saturation=water_reversal_saturation,
            maximum_historical_saturation=maximum_water_saturation,
            is_imbibition=water_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )

        # Gas-oil system - Land trapping on gas and oil
        oil_saturation_at_gas_oil_reversal = np.maximum(
            0.0,
            1.0 - gas_reversal_saturation - water_saturation,
        )
        imbibition_gas_oil_kwargs = dict(kwargs)
        if use_hysteresis and self.maximum_residual_gas_saturation is not None:
            dynamic_residual_gas_saturation = compute_land_residual_saturation(
                initial_non_wetting_saturation=gas_reversal_saturation,
                maximum_residual_saturation=self.maximum_residual_gas_saturation,
                land_trapping_coefficient=self.land_trapping_coefficient_gas,
            )
            imbibition_gas_oil_kwargs["residual_gas_saturation"] = (
                dynamic_residual_gas_saturation
            )
        if use_hysteresis and self.maximum_residual_oil_saturation_gas is not None:
            dynamic_residual_oil_saturation_gas = compute_land_residual_saturation(
                initial_non_wetting_saturation=oil_saturation_at_gas_oil_reversal,
                maximum_residual_saturation=self.maximum_residual_oil_saturation_gas,
                land_trapping_coefficient=self.land_trapping_coefficient_gas,
            )
            imbibition_gas_oil_kwargs["residual_oil_saturation_gas"] = (
                dynamic_residual_oil_saturation_gas
            )

        oil_relative_permeability_gas_drainage, gas_relative_permeability_drainage = (
            _get_gas_oil_relative_permeabilities(
                gas_oil_table=gas_oil_drainage_table,
                water_saturation=water_saturation,
                oil_saturation=oil_saturation,
                gas_saturation=gas_saturation,
                **kwargs,
            )
        )
        (
            oil_relative_permeability_gas_imbibition,
            gas_relative_permeability_imbibition,
        ) = _get_gas_oil_relative_permeabilities(
            gas_oil_table=gas_oil_imbibition_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **imbibition_gas_oil_kwargs,
        )

        gas_relative_permeability = compute_killough_scanning_curve(
            saturation=gas_saturation,
            drainage_curve_value=gas_relative_permeability_drainage,
            imbibition_curve_value=gas_relative_permeability_imbibition,
            reversal_saturation=gas_reversal_saturation,
            maximum_historical_saturation=maximum_gas_saturation,
            is_imbibition=gas_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )
        oil_relative_permeability_gas = compute_killough_scanning_curve(
            saturation=gas_saturation,
            drainage_curve_value=oil_relative_permeability_gas_drainage,
            imbibition_curve_value=oil_relative_permeability_gas_imbibition,
            reversal_saturation=gas_reversal_saturation,
            maximum_historical_saturation=maximum_gas_saturation,
            is_imbibition=gas_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )
        oil_relperm_endpoint = self.get_oil_relperm_endpoint()

        # Three-phase oil via mixing rule
        three_phase_mixing_rule = typing.cast(MixingRule, self.mixing_rule)
        oil_relative_permeability = three_phase_mixing_rule(
            kro_w=oil_relative_permeability_water,
            kro_g=oil_relative_permeability_gas,
            krw=water_relative_permeability,
            krg=gas_relative_permeability,
            kr_max=oil_relperm_endpoint,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
        )

        if is_scalar:
            return RelativePermeabilities(
                water=water_relative_permeability.item(),  # type: ignore
                oil=oil_relative_permeability.item(),  # type: ignore
                gas=gas_relative_permeability.item(),  # type: ignore
            )
        return RelativePermeabilities(
            water=water_relative_permeability,
            oil=oil_relative_permeability,
            gas=gas_relative_permeability,
        )

    def get_relative_permeability_derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        max_water_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
        max_gas_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
        water_imbibition_flag: typing.Optional[
            typing.Union[bool, BooleanArray[NDimension]]
        ] = None,
        gas_imbibition_flag: typing.Optional[
            typing.Union[bool, BooleanArray[NDimension]]
        ] = None,
        water_reversal_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
        gas_reversal_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
        **kwargs: typing.Any,
    ) -> RelativePermeabilityDerivatives:
        """
        Compute partial derivatives of three-phase relative permeabilities
        with Killough/Land hysteresis.

        Returns all nine ∂kr_α/∂S_β entries. Derivatives with respect to
        the scanning variable (Sw for oil-water, Sg for gas-oil) are computed
        analytically via the chain rule through the Killough scanning-curve
        formula; all other cross-derivatives are zero (consistent with the
        assumption that each two-phase sub-system depends only on its own
        reference saturation).

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param max_water_saturation: Historical maximum water saturation per cell,
            or `None` to disable hysteresis.
        :param max_gas_saturation: Historical maximum gas saturation per cell,
            or `None` to disable hysteresis.
        :param water_imbibition_flag: Per-cell imbibition flag for the oil-water
            system (1 = imbibition, 0 = drainage), or `None`.
        :param gas_imbibition_flag: Per-cell imbibition flag for the gas-oil
            system (1 = imbibition, 0 = drainage), or `None`.
        :param water_reversal_saturation: Water saturation at the last oil-water
            reversal, or `None` (defaults to `max_water_saturation`).
        :param gas_reversal_saturation: Gas saturation at the last gas-oil reversal,
            or `None` (defaults to `max_gas_saturation`).
        :param kwargs: Additional keyword arguments forwarded to the backing tables.
        :return: `RelativePermeabilityDerivatives` dictionary containing all
            nine ∂kr/∂S entries.
        """
        is_scalar = np.isscalar(water_saturation)
        water_saturation = np.atleast_1d(water_saturation)
        oil_saturation = np.atleast_1d(oil_saturation)
        gas_saturation = np.atleast_1d(gas_saturation)
        water_saturation, oil_saturation, gas_saturation = np.broadcast_arrays(
            water_saturation, oil_saturation, gas_saturation
        )
        zeros = np.zeros_like(water_saturation)

        (
            maximum_water_saturation,
            maximum_gas_saturation,
            water_imbibition_flag,
            gas_imbibition_flag,
            water_reversal_saturation,
            gas_reversal_saturation,
        ) = self._parse_hysteresis_kwargs(
            water_saturation=water_saturation,
            gas_saturation=gas_saturation,
            max_water_saturation=max_water_saturation,
            max_gas_saturation=max_gas_saturation,
            water_imbibition_flag=water_imbibition_flag,
            gas_imbibition_flag=gas_imbibition_flag,
            water_reversal_saturation=water_reversal_saturation,
            gas_reversal_saturation=gas_reversal_saturation,
        )

        oil_water_drainage_table = self.oil_water_drainage_table
        oil_water_imbibition_table = (
            self.oil_water_imbibition_table or oil_water_drainage_table
        )
        gas_oil_drainage_table = self.gas_oil_drainage_table
        gas_oil_imbibition_table = (
            self.gas_oil_imbibition_table or gas_oil_drainage_table
        )
        use_hysteresis = (
            max_water_saturation is not None and max_gas_saturation is not None
        )

        # Build imbibition kwargs with Land trapping
        oil_saturation_at_oil_water_reversal = np.maximum(
            0.0,
            1.0 - water_reversal_saturation - gas_saturation,
        )
        imbibition_oil_water_kwargs = dict(kwargs)
        if use_hysteresis and self.maximum_residual_oil_saturation_water is not None:
            dynamic_residual_oil_saturation_water = compute_land_residual_saturation(
                initial_non_wetting_saturation=oil_saturation_at_oil_water_reversal,
                maximum_residual_saturation=self.maximum_residual_oil_saturation_water,
                land_trapping_coefficient=self.land_trapping_coefficient_water,
            )
            imbibition_oil_water_kwargs["residual_oil_saturation_water"] = (
                dynamic_residual_oil_saturation_water
            )

        oil_saturation_at_gas_oil_reversal = np.maximum(
            0.0,
            1.0 - gas_reversal_saturation - water_saturation,
        )
        imbibition_gas_oil_kwargs = dict(kwargs)
        if use_hysteresis and self.maximum_residual_gas_saturation is not None:
            dynamic_residual_gas_saturation = compute_land_residual_saturation(
                initial_non_wetting_saturation=gas_reversal_saturation,
                maximum_residual_saturation=self.maximum_residual_gas_saturation,
                land_trapping_coefficient=self.land_trapping_coefficient_gas,
            )
            imbibition_gas_oil_kwargs["residual_gas_saturation"] = (
                dynamic_residual_gas_saturation
            )
        if use_hysteresis and self.maximum_residual_oil_saturation_gas is not None:
            dynamic_residual_oil_saturation_gas = compute_land_residual_saturation(
                initial_non_wetting_saturation=oil_saturation_at_gas_oil_reversal,
                maximum_residual_saturation=self.maximum_residual_oil_saturation_gas,
                land_trapping_coefficient=self.land_trapping_coefficient_gas,
            )
            imbibition_gas_oil_kwargs["residual_oil_saturation_gas"] = (
                dynamic_residual_oil_saturation_gas
            )

        # Oil-water - values and derivatives
        (
            water_relative_permeability_drainage,
            oil_relative_permeability_water_drainage,
        ) = _get_oil_water_relative_permeabilities(
            oil_water_table=oil_water_drainage_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        (
            water_relative_permeability_imbibition,
            oil_relative_permeability_water_imbibition,
        ) = _get_oil_water_relative_permeabilities(
            oil_water_table=oil_water_imbibition_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **imbibition_oil_water_kwargs,
        )

        (
            water_relative_permeability_drainage_derivative,
            oil_relative_permeability_water_drainage_derivative,
        ) = _get_oil_water_relative_permeability_derivatives(
            oil_water_table=oil_water_drainage_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        (
            water_relative_permeability_imbibition_derivative,
            oil_relative_permeability_water_imbibition_derivative,
        ) = _get_oil_water_relative_permeability_derivatives(
            oil_water_table=oil_water_imbibition_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **imbibition_oil_water_kwargs,
        )

        water_relative_permeability_derivative = compute_killough_scanning_curve_derivative(
            saturation=water_saturation,
            drainage_curve_value=water_relative_permeability_drainage,
            imbibition_curve_value=water_relative_permeability_imbibition,
            drainage_curve_derivative=water_relative_permeability_drainage_derivative,
            imbibition_curve_derivative=water_relative_permeability_imbibition_derivative,
            reversal_saturation=water_reversal_saturation,
            maximum_historical_saturation=maximum_water_saturation,
            is_imbibition=water_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )
        oil_relative_permeability_water_derivative = compute_killough_scanning_curve_derivative(
            saturation=water_saturation,
            drainage_curve_value=oil_relative_permeability_water_drainage,
            imbibition_curve_value=oil_relative_permeability_water_imbibition,
            drainage_curve_derivative=oil_relative_permeability_water_drainage_derivative,
            imbibition_curve_derivative=oil_relative_permeability_water_imbibition_derivative,
            reversal_saturation=water_reversal_saturation,
            maximum_historical_saturation=maximum_water_saturation,
            is_imbibition=water_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )

        water_relative_permeability = compute_killough_scanning_curve(
            saturation=water_saturation,
            drainage_curve_value=water_relative_permeability_drainage,
            imbibition_curve_value=water_relative_permeability_imbibition,
            reversal_saturation=water_reversal_saturation,
            maximum_historical_saturation=maximum_water_saturation,
            is_imbibition=water_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )
        oil_relative_permeability_water = compute_killough_scanning_curve(
            saturation=water_saturation,
            drainage_curve_value=oil_relative_permeability_water_drainage,
            imbibition_curve_value=oil_relative_permeability_water_imbibition,
            reversal_saturation=water_reversal_saturation,
            maximum_historical_saturation=maximum_water_saturation,
            is_imbibition=water_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )

        # Gas-oil - values and derivatives
        oil_relative_permeability_gas_drainage, gas_relative_permeability_drainage = (
            _get_gas_oil_relative_permeabilities(
                gas_oil_table=gas_oil_drainage_table,
                water_saturation=water_saturation,
                oil_saturation=oil_saturation,
                gas_saturation=gas_saturation,
                **kwargs,
            )
        )
        (
            oil_relative_permeability_gas_imbibition,
            gas_relative_permeability_imbibition,
        ) = _get_gas_oil_relative_permeabilities(
            gas_oil_table=gas_oil_imbibition_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **imbibition_gas_oil_kwargs,
        )

        (
            oil_relative_permeability_gas_drainage_derivative,
            gas_relative_permeability_drainage_derivative,
        ) = _get_gas_oil_relative_permeability_derivatives(
            gas_oil_table=gas_oil_drainage_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        (
            oil_relative_permeability_gas_imbibition_derivative,
            gas_relative_permeability_imbibition_derivative,
        ) = _get_gas_oil_relative_permeability_derivatives(
            gas_oil_table=gas_oil_imbibition_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **imbibition_gas_oil_kwargs,
        )

        gas_relative_permeability_derivative = compute_killough_scanning_curve_derivative(
            saturation=gas_saturation,
            drainage_curve_value=gas_relative_permeability_drainage,
            imbibition_curve_value=gas_relative_permeability_imbibition,
            drainage_curve_derivative=gas_relative_permeability_drainage_derivative,
            imbibition_curve_derivative=gas_relative_permeability_imbibition_derivative,
            reversal_saturation=gas_reversal_saturation,
            maximum_historical_saturation=maximum_gas_saturation,
            is_imbibition=gas_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )
        oil_relative_permeability_gas_derivative = compute_killough_scanning_curve_derivative(
            saturation=gas_saturation,
            drainage_curve_value=oil_relative_permeability_gas_drainage,
            imbibition_curve_value=oil_relative_permeability_gas_imbibition,
            drainage_curve_derivative=oil_relative_permeability_gas_drainage_derivative,
            imbibition_curve_derivative=oil_relative_permeability_gas_imbibition_derivative,
            reversal_saturation=gas_reversal_saturation,
            maximum_historical_saturation=maximum_gas_saturation,
            is_imbibition=gas_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )

        gas_relative_permeability = compute_killough_scanning_curve(
            saturation=gas_saturation,
            drainage_curve_value=gas_relative_permeability_drainage,
            imbibition_curve_value=gas_relative_permeability_imbibition,
            reversal_saturation=gas_reversal_saturation,
            maximum_historical_saturation=maximum_gas_saturation,
            is_imbibition=gas_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )
        oil_relative_permeability_gas = compute_killough_scanning_curve(
            saturation=gas_saturation,
            drainage_curve_value=oil_relative_permeability_gas_drainage,
            imbibition_curve_value=oil_relative_permeability_gas_imbibition,
            reversal_saturation=gas_reversal_saturation,
            maximum_historical_saturation=maximum_gas_saturation,
            is_imbibition=gas_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )
        oil_relperm_endpoint = self.get_oil_relperm_endpoint()

        # Three-phase oil mixing rule - chain rule
        three_phase_mixing_rule = typing.cast(MixingRule, self.mixing_rule)
        mixing_rule_partial_derivatives = get_mixing_rule_partial_derivatives(
            rule=three_phase_mixing_rule,
            kro_w=oil_relative_permeability_water,
            kro_g=oil_relative_permeability_gas,
            krw=water_relative_permeability,
            krg=gas_relative_permeability,
            kr_max=oil_relperm_endpoint,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            epsilon=c.FINITE_DIFFERENCE_EPSILON,
        )

        oil_relative_permeability_derivative_water = (
            mixing_rule_partial_derivatives["d_kro_d_kro_w"]
            * oil_relative_permeability_water_derivative
            + mixing_rule_partial_derivatives["d_kro_d_sw_explicit"]
        )
        oil_relative_permeability_derivative_oil = mixing_rule_partial_derivatives[
            "d_kro_d_so_explicit"
        ]
        oil_relative_permeability_derivative_gas = (
            mixing_rule_partial_derivatives["d_kro_d_kro_g"]
            * oil_relative_permeability_gas_derivative
            + mixing_rule_partial_derivatives["d_kro_d_sg_explicit"]
        )

        if is_scalar:
            return RelativePermeabilityDerivatives(
                dKrw_dSw=water_relative_permeability_derivative.item(),  # type: ignore
                dKro_dSw=oil_relative_permeability_derivative_water.item(),  # type: ignore
                dKrg_dSw=0.0,
                dKrw_dSo=0.0,
                dKro_dSo=oil_relative_permeability_derivative_oil.item(),  # type: ignore
                dKrg_dSo=0.0,
                dKrw_dSg=0.0,
                dKro_dSg=oil_relative_permeability_derivative_gas.item(),  # type: ignore
                dKrg_dSg=gas_relative_permeability_derivative.item(),  # type: ignore
            )

        return RelativePermeabilityDerivatives(
            dKrw_dSw=water_relative_permeability_derivative,
            dKro_dSw=oil_relative_permeability_derivative_water,
            dKrg_dSw=zeros.copy(),
            dKrw_dSo=zeros.copy(),
            dKro_dSo=oil_relative_permeability_derivative_oil,
            dKrg_dSo=zeros.copy(),
            dKrw_dSg=zeros.copy(),
            dKro_dSg=oil_relative_permeability_derivative_gas,
            dKrg_dSg=gas_relative_permeability_derivative,
        )


@capillary_pressure_table
@attrs.frozen
class KilloughCapillaryPressureTable(
    CapillaryPressureTable,
    load_exclude={"supports_vector"},
    dump_exclude={"supports_vector"},
):
    """
    Implements the Killough capillary pressure hysteresis model/table.

    Capillary pressure hysteresis involves no trapping.  When the displacement
    direction reverses, capillary pressure traces a *scanning curve* that
    interpolates between the primary drainage and imbibition bounds.

    Both two-phase (`TwoPhaseCapillaryPressureTable`) and full three-phase
    (`CapillaryPressureTable`) backing tables are supported.  Wetting and
    non-wetting phase roles are resolved through the canonical API
    (`get_oil_water_wetting_phase` / `get_gas_oil_wetting_phase`) so the
    model is wettability-agnostic.

    The oil-water scanning curve scans over *water* saturation; the gas-oil
    scanning curve scans over *gas* saturation.

    Saturation history is passed as additional keyword arguments to
    `get_capillary_pressures` and `get_capillary_pressure_derivatives`.
    When these arguments are absent the model returns primary drainage Pc
    values.
    """

    __type__ = "killough_capillary_pressure_model"

    oil_water_drainage_table: typing.Union[
        TwoPhaseCapillaryPressureTable, CapillaryPressureTable
    ]
    """Primary drainage capillary pressure table for the oil-water system."""

    gas_oil_drainage_table: typing.Union[
        TwoPhaseCapillaryPressureTable, CapillaryPressureTable
    ]
    """Primary drainage capillary pressure table for the gas-oil system."""

    oil_water_imbibition_table: typing.Optional[
        typing.Union[TwoPhaseCapillaryPressureTable, CapillaryPressureTable]
    ] = None
    """Primary imbibition Pc table for the oil-water system. Defaults to the drainage table."""

    gas_oil_imbibition_table: typing.Optional[
        typing.Union[TwoPhaseCapillaryPressureTable, CapillaryPressureTable]
    ] = None
    """Primary imbibition Pc table for the gas-oil system. Defaults to the drainage table."""

    scanning_interpolation_exponent: Number = 1.0
    """Killough interpolation exponent *n* (1 = linear)."""

    supports_vector: bool = attrs.field(init=False, repr=False, default=True)

    def __attrs_post_init__(self) -> None:
        if isinstance(
            self.oil_water_drainage_table, TwoPhaseCapillaryPressureTable
        ) and {
            self.oil_water_drainage_table.wetting_phase,
            self.oil_water_drainage_table.non_wetting_phase,
        } != {FluidPhase.WATER, FluidPhase.OIL}:
            raise ValidationError(
                "`oil_water_drainage_table` must involve water and oil phases."
            )

        if isinstance(self.gas_oil_drainage_table, TwoPhaseCapillaryPressureTable) and {
            self.gas_oil_drainage_table.wetting_phase,
            self.gas_oil_drainage_table.non_wetting_phase,
        } != {FluidPhase.OIL, FluidPhase.GAS}:
            raise ValidationError(
                "`gas_oil_drainage_table` must involve oil and gas phases."
            )

        if (
            self.oil_water_imbibition_table is not None
            and isinstance(
                self.oil_water_imbibition_table, TwoPhaseCapillaryPressureTable
            )
            and {
                self.oil_water_imbibition_table.wetting_phase,
                self.oil_water_imbibition_table.non_wetting_phase,
            }
            != {FluidPhase.WATER, FluidPhase.OIL}
        ):
            raise ValidationError(
                "`oil_water_imbibition_table` must involve water and oil phases."
            )

        if (
            self.gas_oil_imbibition_table is not None
            and isinstance(
                self.gas_oil_imbibition_table, TwoPhaseCapillaryPressureTable
            )
            and {
                self.gas_oil_imbibition_table.wetting_phase,
                self.gas_oil_imbibition_table.non_wetting_phase,
            }
            != {FluidPhase.OIL, FluidPhase.GAS}
        ):
            raise ValidationError(
                "`gas_oil_imbibition_table` must involve oil and gas phases."
            )

    def get_oil_water_wetting_phase(self) -> FluidPhase:
        """
        Return the wetting phase for the oil-water sub-system.

        :return: `FluidPhase.WATER` for water-wet or `FluidPhase.OIL` for
            oil-wet systems, as reported by the drainage table.
        """
        return self.oil_water_drainage_table.get_oil_water_wetting_phase()

    def get_gas_oil_wetting_phase(self) -> FluidPhase:
        """
        Return the wetting phase for the gas-oil sub-system.

        :return: `FluidPhase.OIL` for oil-wet or `FluidPhase.GAS` for
            gas-wet systems, as reported by the drainage table.
        """
        return self.gas_oil_drainage_table.get_gas_oil_wetting_phase()

    def _parse_hysteresis_kwargs(
        self,
        water_saturation: npt.NDArray,
        gas_saturation: npt.NDArray,
        max_water_saturation: typing.Optional[NumberOrArray[NDimension]],
        max_gas_saturation: typing.Optional[NumberOrArray[NDimension]],
        water_imbibition_flag: typing.Optional[
            typing.Union[bool, BooleanArray[NDimension]]
        ],
        gas_imbibition_flag: typing.Optional[
            typing.Union[bool, BooleanArray[NDimension]]
        ],
        water_reversal_saturation: typing.Optional[NumberOrArray[NDimension]],
        gas_reversal_saturation: typing.Optional[NumberOrArray[NDimension]],
    ) -> typing.Tuple[
        npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
    ]:
        """
        Parse and broadcast saturation-history arrays.

        When all history arguments are `None` the method returns arrays that
        replicate the primary-drainage state (no-hysteresis fallback).

        :param water_saturation: Broadcast-ready water saturation array.
        :param gas_saturation: Broadcast-ready gas saturation array.
        :param max_water_saturation: Historical maximum water saturation or `None`.
        :param max_gas_saturation: Historical maximum gas saturation or `None`.
        :param water_imbibition_flag: Per-cell oil-water imbibition flag or `None`.
        :param gas_imbibition_flag: Per-cell gas-oil imbibition flag or `None`.
        :param water_reversal_saturation: Water saturation at last oil-water reversal,
            or `None` (defaults to `max_water_saturation`).
        :param gas_reversal_saturation: Gas saturation at last gas-oil reversal, or
            `None` (defaults to `max_gas_saturation`).
        :return: Six broadcast-compatible arrays: `(sw_max, sg_max, sw_imb, sg_imb,
            sw_rev, sg_rev)`.
        """
        use_hysteresis = (
            max_water_saturation is not None
            and max_gas_saturation is not None
            and water_imbibition_flag is not None
            and gas_imbibition_flag is not None
        )
        if use_hysteresis:
            maximum_water_saturation = np.atleast_1d(max_water_saturation)  # type: ignore
            maximum_gas_saturation = np.atleast_1d(max_gas_saturation)  # type: ignore
            water_imbibition_flag = np.atleast_1d(water_imbibition_flag)  # type: ignore
            gas_imbibition_flag = np.atleast_1d(gas_imbibition_flag)  # type: ignore
            water_reversal_saturation = (
                np.atleast_1d(water_reversal_saturation)
                if water_reversal_saturation is not None
                else maximum_water_saturation.copy()
            )
            gas_reversal_saturation = (
                np.atleast_1d(gas_reversal_saturation)
                if gas_reversal_saturation is not None
                else maximum_gas_saturation.copy()
            )
        else:
            maximum_water_saturation = water_saturation.copy()
            maximum_gas_saturation = gas_saturation.copy()
            water_imbibition_flag = np.zeros_like(water_saturation)
            gas_imbibition_flag = np.zeros_like(gas_saturation)
            water_reversal_saturation = water_saturation.copy()
            gas_reversal_saturation = gas_saturation.copy()

        (
            maximum_water_saturation,
            maximum_gas_saturation,
            water_imbibition_flag,
            gas_imbibition_flag,
            water_reversal_saturation,
            gas_reversal_saturation,
        ) = np.broadcast_arrays(
            maximum_water_saturation,
            maximum_gas_saturation,
            water_imbibition_flag,  # type: ignore
            gas_imbibition_flag,  # type: ignore
            water_reversal_saturation,
            gas_reversal_saturation,
        )
        return (
            maximum_water_saturation,
            maximum_gas_saturation,
            water_imbibition_flag,
            gas_imbibition_flag,
            water_reversal_saturation,
            gas_reversal_saturation,
        )

    def get_capillary_pressures(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        max_water_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
        max_gas_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
        water_imbibition_flag: typing.Optional[
            typing.Union[bool, BooleanArray[NDimension]]
        ] = None,
        gas_imbibition_flag: typing.Optional[
            typing.Union[bool, BooleanArray[NDimension]]
        ] = None,
        water_reversal_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
        gas_reversal_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute three-phase capillary pressures with Killough hysteresis.

        When the saturation-history keyword arguments are absent the method
        returns primary drainage Pc values.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param max_water_saturation: Historical maximum water saturation per cell.
        :param max_gas_saturation: Historical maximum gas saturation per cell.
        :param water_imbibition_flag: Per-cell oil-water imbibition flag (1 / 0),
            or `None`.
        :param gas_imbibition_flag: Per-cell gas-oil imbibition flag (1 / 0),
            or `None`.
        :param water_reversal_saturation: Water saturation at the last oil-water
            reversal, or `None` (defaults to `max_water_saturation`).
        :param gas_reversal_saturation: Gas saturation at the last gas-oil reversal,
            or `None` (defaults to `max_gas_saturation`).
        :param kwargs: Additional keyword arguments forwarded to the backing tables.
        :return: `CapillaryPressures` dictionary with keys `"oil_water"` and
            `"gas_oil"`.
        """
        is_scalar = np.isscalar(water_saturation)
        water_saturation = np.atleast_1d(water_saturation)
        oil_saturation = np.atleast_1d(oil_saturation)
        gas_saturation = np.atleast_1d(gas_saturation)
        water_saturation, oil_saturation, gas_saturation = np.broadcast_arrays(
            water_saturation, oil_saturation, gas_saturation
        )

        (
            maximum_water_saturation,
            maximum_gas_saturation,
            water_imbibition_flag,
            gas_imbibition_flag,
            water_reversal_saturation,
            gas_reversal_saturation,
        ) = self._parse_hysteresis_kwargs(
            water_saturation=water_saturation,
            gas_saturation=gas_saturation,
            max_water_saturation=max_water_saturation,
            max_gas_saturation=max_gas_saturation,
            water_imbibition_flag=water_imbibition_flag,
            gas_imbibition_flag=gas_imbibition_flag,
            water_reversal_saturation=water_reversal_saturation,
            gas_reversal_saturation=gas_reversal_saturation,
        )

        oil_water_drainage_table = self.oil_water_drainage_table
        oil_water_imbibition_table = (
            self.oil_water_imbibition_table or oil_water_drainage_table
        )
        gas_oil_drainage_table = self.gas_oil_drainage_table
        gas_oil_imbibition_table = (
            self.gas_oil_imbibition_table or gas_oil_drainage_table
        )

        # Oil-water Pc: scan over water saturation
        oil_water_capillary_pressure_drainage = _get_oil_water_capillary_pressure(
            oil_water_capillary_pressure_table=oil_water_drainage_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        oil_water_capillary_pressure_imbibition = _get_oil_water_capillary_pressure(
            oil_water_capillary_pressure_table=oil_water_imbibition_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        oil_water_capillary_pressure = compute_killough_scanning_curve(
            saturation=water_saturation,
            drainage_curve_value=oil_water_capillary_pressure_drainage,
            imbibition_curve_value=oil_water_capillary_pressure_imbibition,
            reversal_saturation=water_reversal_saturation,
            maximum_historical_saturation=maximum_water_saturation,
            is_imbibition=water_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )

        # Gas-oil Pc: scan over gas saturation
        gas_oil_capillary_pressure_drainage = _get_gas_oil_capillary_pressure(
            gas_oil_capillary_pressure_table=gas_oil_drainage_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        gas_oil_capillary_pressure_imbibition = _get_gas_oil_capillary_pressure(
            gas_oil_capillary_pressure_table=gas_oil_imbibition_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        gas_oil_capillary_pressure = compute_killough_scanning_curve(
            saturation=gas_saturation,
            drainage_curve_value=gas_oil_capillary_pressure_drainage,
            imbibition_curve_value=gas_oil_capillary_pressure_imbibition,
            reversal_saturation=gas_reversal_saturation,
            maximum_historical_saturation=maximum_gas_saturation,
            is_imbibition=gas_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )

        if is_scalar:
            return CapillaryPressures(
                oil_water=oil_water_capillary_pressure.item(),  # type: ignore
                gas_oil=gas_oil_capillary_pressure.item(),  # type: ignore
            )
        return CapillaryPressures(
            oil_water=oil_water_capillary_pressure,  # type: ignore[typeddict-item]
            gas_oil=gas_oil_capillary_pressure,  # type: ignore[typeddict-item]
        )

    def get_capillary_pressure_derivatives(
        self,
        water_saturation: NumberOrArray[NDimension],
        oil_saturation: NumberOrArray[NDimension],
        gas_saturation: NumberOrArray[NDimension],
        max_water_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
        max_gas_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
        water_imbibition_flag: typing.Optional[
            typing.Union[bool, BooleanArray[NDimension]]
        ] = None,
        gas_imbibition_flag: typing.Optional[
            typing.Union[bool, BooleanArray[NDimension]]
        ] = None,
        water_reversal_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
        gas_reversal_saturation: typing.Optional[NumberOrArray[NDimension]] = None,
        **kwargs: typing.Any,
    ) -> CapillaryPressureDerivatives:
        """
        Compute partial derivatives of capillary pressures with Killough hysteresis.

        Returns the following non-zero entries.

        - `dPcow_dSw`: dPcow/dSw - oil-water Pc scanned over Sw.
        - `dPcow_dSo`: zero - Pcow does not depend directly on So in this model.
        - `dPcgo_dSg`: dPcgo/dSg - gas-oil Pc scanned over Sg.
        - `dPcgo_dSo`: zero - Pcgo does not depend directly on So.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param max_water_saturation: Historical maximum water saturation per cell,
            or `None` to disable hysteresis.
        :param max_gas_saturation: Historical maximum gas saturation per cell,
            or `None` to disable hysteresis.
        :param water_imbibition_flag: Per-cell oil-water imbibition flag or `None`.
        :param gas_imbibition_flag: Per-cell gas-oil imbibition flag or `None`.
        :param water_reversal_saturation: Water saturation at the last oil-water
            reversal, or `None` (defaults to `max_water_saturation`).
        :param gas_reversal_saturation: Gas saturation at the last gas-oil reversal,
            or `None` (defaults to `max_gas_saturation`).
        :param kwargs: Additional keyword arguments forwarded to the backing tables.
        :return: `CapillaryPressureDerivatives` dictionary containing
            `dPcow_dSw`, `dPcow_dSo`, `dPcgo_dSg`, and `dPcgo_dSo`.
        """
        is_scalar = np.isscalar(water_saturation)
        water_saturation = np.atleast_1d(water_saturation)
        oil_saturation = np.atleast_1d(oil_saturation)
        gas_saturation = np.atleast_1d(gas_saturation)
        water_saturation, oil_saturation, gas_saturation = np.broadcast_arrays(
            water_saturation, oil_saturation, gas_saturation
        )
        zeros = np.zeros_like(water_saturation)

        (
            maximum_water_saturation,
            maximum_gas_saturation,
            water_imbibition_flag,
            gas_imbibition_flag,
            water_reversal_saturation,
            gas_reversal_saturation,
        ) = self._parse_hysteresis_kwargs(
            water_saturation=water_saturation,
            gas_saturation=gas_saturation,
            max_water_saturation=max_water_saturation,
            max_gas_saturation=max_gas_saturation,
            water_imbibition_flag=water_imbibition_flag,
            gas_imbibition_flag=gas_imbibition_flag,
            water_reversal_saturation=water_reversal_saturation,
            gas_reversal_saturation=gas_reversal_saturation,
        )

        oil_water_drainage_table = self.oil_water_drainage_table
        oil_water_imbibition_table = (
            self.oil_water_imbibition_table or oil_water_drainage_table
        )
        gas_oil_drainage_table = self.gas_oil_drainage_table
        gas_oil_imbibition_table = (
            self.gas_oil_imbibition_table or gas_oil_drainage_table
        )

        # Oil-water
        oil_water_capillary_pressure_drainage = _get_oil_water_capillary_pressure(
            oil_water_capillary_pressure_table=oil_water_drainage_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        oil_water_capillary_pressure_imbibition = _get_oil_water_capillary_pressure(
            oil_water_capillary_pressure_table=oil_water_imbibition_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        oil_water_capillary_pressure_drainage_derivative = (
            _get_oil_water_capillary_pressure_derivative(
                oil_water_capillary_pressure_table=oil_water_drainage_table,
                water_saturation=water_saturation,
                oil_saturation=oil_saturation,
                gas_saturation=gas_saturation,
                **kwargs,
            )
        )
        oil_water_capillary_pressure_imbibition_derivative = (
            _get_oil_water_capillary_pressure_derivative(
                oil_water_capillary_pressure_table=oil_water_imbibition_table,
                water_saturation=water_saturation,
                oil_saturation=oil_saturation,
                gas_saturation=gas_saturation,
                **kwargs,
            )
        )

        oil_water_capillary_pressure_derivative = compute_killough_scanning_curve_derivative(
            saturation=water_saturation,
            drainage_curve_value=oil_water_capillary_pressure_drainage,
            imbibition_curve_value=oil_water_capillary_pressure_imbibition,
            drainage_curve_derivative=oil_water_capillary_pressure_drainage_derivative,
            imbibition_curve_derivative=oil_water_capillary_pressure_imbibition_derivative,
            reversal_saturation=water_reversal_saturation,
            maximum_historical_saturation=maximum_water_saturation,
            is_imbibition=water_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )

        # Gas-oil
        gas_oil_capillary_pressure_drainage = _get_gas_oil_capillary_pressure(
            gas_oil_capillary_pressure_table=gas_oil_drainage_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        gas_oil_capillary_pressure_imbibition = _get_gas_oil_capillary_pressure(
            gas_oil_capillary_pressure_table=gas_oil_imbibition_table,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
        gas_oil_capillary_pressure_drainage_derivative = (
            _get_gas_oil_capillary_pressure_derivative(
                gas_oil_capillary_pressure_table=gas_oil_drainage_table,
                water_saturation=water_saturation,
                oil_saturation=oil_saturation,
                gas_saturation=gas_saturation,
                **kwargs,
            )
        )
        gas_oil_capillary_pressure_imbibition_derivative = (
            _get_gas_oil_capillary_pressure_derivative(
                gas_oil_capillary_pressure_table=gas_oil_imbibition_table,
                water_saturation=water_saturation,
                oil_saturation=oil_saturation,
                gas_saturation=gas_saturation,
                **kwargs,
            )
        )

        gas_oil_capillary_pressure_derivative = compute_killough_scanning_curve_derivative(
            saturation=gas_saturation,
            drainage_curve_value=gas_oil_capillary_pressure_drainage,
            imbibition_curve_value=gas_oil_capillary_pressure_imbibition,
            drainage_curve_derivative=gas_oil_capillary_pressure_drainage_derivative,
            imbibition_curve_derivative=gas_oil_capillary_pressure_imbibition_derivative,
            reversal_saturation=gas_reversal_saturation,
            maximum_historical_saturation=maximum_gas_saturation,
            is_imbibition=gas_imbibition_flag,  # type: ignore
            scanning_exponent=self.scanning_interpolation_exponent,
        )

        if is_scalar:
            return CapillaryPressureDerivatives(
                dPcow_dSw=oil_water_capillary_pressure_derivative.item(),  # type: ignore
                dPcow_dSo=0.0,
                dPcgo_dSg=gas_oil_capillary_pressure_derivative.item(),  # type: ignore
                dPcgo_dSo=0.0,
            )

        return CapillaryPressureDerivatives(
            dPcow_dSw=oil_water_capillary_pressure_derivative,
            dPcow_dSo=zeros.copy(),
            dPcgo_dSg=gas_oil_capillary_pressure_derivative,
            dPcgo_dSo=zeros.copy(),
        )
