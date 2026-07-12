"""Shared wellbore hydraulics primitives."""

import math
import typing

import attrs
import numpy as np

from bores.constants import c
from bores.errors import ValidationError
from bores.serde.base import Serializable
from bores.typing import FluidPhase, Number, NumberArray, OneDimension
from bores.wells.base import Well
from bores.wells.perforations import PerforationIndex
from bores.wells.states import ConnectionSample

__all__ = [
    "PressureDropComponents",
    "SurfaceFluidProperties",
    "compute_mixture_density",
    "compute_mixture_viscosity",
    "compute_mixture_velocity",
    "compute_friction_factor",
    "compute_segment_pressure_drop",
    "compute_static_hydrostatic_drop",
    "static_mixture_density",
    "WellboreModel",
]


@attrs.frozen(kw_only=True, slots=True)
class PressureDropComponents(Serializable):
    """One tubing segment's pressure drop, broken out by mechanism."""

    hydrostatic: Number
    """`rho * g * dz`. Positive = pressure increases going down."""

    friction: Number
    """Darcy-Weisbach term."""

    acceleration: Number
    """`rho * (v_out**2 - v_in**2) / 2`."""

    @property
    def total(self) -> Number:
        """Sum of all three components."""
        return self.hydrostatic + self.friction + self.acceleration


@attrs.frozen(kw_only=True, slots=True)
class SurfaceFluidProperties(Serializable):
    """Density and viscosity of the produced/injected stream at surface
    conditions - there's no reservoir cell at surface to pull a
    `ConnectionSample` from, so `WellboreModel.tubing_head_pressure` takes
    this instead.
    """

    density: Number
    viscosity: Number


def compute_mixture_density(
    phase_rates: typing.Mapping[FluidPhase, Number],
    phase_densities: typing.Mapping[FluidPhase, Number],
) -> Number:
    """
    No-slip rate-weighted mixture density:

    `sum(rate_p * density_p) / sum(rate_p)` across phases present in
    `phase_rates`.

    :param phase_rates: Rates per phase, consistent condition with
        `phase_densities`.
    :param phase_densities: Density per phase, same condition as
        `phase_rates`. A phase present in `phase_rates` with rate `0.0`
        doesn't need an entry here.
    :returns: Mixture density.
    :raises ValidationError: If `phase_rates` sums to zero - use
        `compute_static_hydrostatic_drop` instead.
    """
    total_rate = sum(phase_rates.values())
    if total_rate == 0.0:
        raise ValidationError(
            "`phase_rates` sums to zero; use compute_static_hydrostatic_drop "
            "for the no-flow case instead."
        )
    return (
        sum(
            rate * phase_densities[phase]
            for phase, rate in phase_rates.items()
            if rate != 0.0
        )
        / total_rate
    )


def compute_mixture_viscosity(
    phase_rates: typing.Mapping[FluidPhase, Number],
    phase_viscosities: typing.Mapping[FluidPhase, Number],
) -> Number:
    """
    No-slip rate-weighted mixture viscosity, same formula/convention as
    `compute_mixture_density`.

    :param phase_rates: Rates per phase.
    :param phase_viscosities: Viscosity per phase, same condition as
        `phase_rates`.
    :returns: Mixture viscosity.
    :raises ValidationError: If `phase_rates` sums to zero.
    """
    total_rate = sum(phase_rates.values())
    if total_rate == 0.0:
        raise ValidationError(
            "`phase_rates` sums to zero; use compute_static_hydrostatic_drop "
            "for the no-flow case instead."
        )
    return (
        sum(
            rate * phase_viscosities[phase]
            for phase, rate in phase_rates.items()
            if rate != 0.0
        )
        / total_rate
    )


def compute_mixture_velocity(
    phase_rates: typing.Mapping[FluidPhase, Number], tubing_inner_diameter: Number
) -> Number:
    """
    No-slip superficial mixture velocity:
    `sum(rate_p) / cross_sectional_area`.

    :param phase_rates: Rates per phase, reservoir condition.
    :param tubing_inner_diameter: Tubing inner diameter.
    :returns: Mixture velocity.
    :raises ValidationError: If `tubing_inner_diameter` is `None` or
        non-positive.
    """
    if tubing_inner_diameter is None or tubing_inner_diameter <= 0:
        raise ValidationError(
            "`tubing_inner_diameter` must be a positive number; got "
            f"{tubing_inner_diameter!r}."
        )
    cross_sectional_area = math.pi * (tubing_inner_diameter / 2.0) ** 2
    return sum(phase_rates.values()) / cross_sectional_area


def compute_friction_factor(
    reynolds_number: Number,
    relative_roughness: Number,
    *,
    method: typing.Literal["simplified", "colebrook"] = "simplified",
    laminar_reynolds_limit: typing.Optional[Number] = None,
    turbulent_reynolds_limit: typing.Optional[Number] = None,
    max_iterations: typing.Optional[int] = None,
    tolerance: typing.Optional[Number] = None,
) -> Number:
    """
    Darcy friction factor for a tubing segment.

    `method="simplified"` (default): `64/Re` for `Re < laminar_reynolds_limit`
    (laminar); Blasius `0.316 * Re**-0.25` for
    `laminar_reynolds_limit <= Re < turbulent_reynolds_limit`; a
    Swamee-Jain-style explicit approximation (includes `relative_roughness`)
    above that. Closed-form, no iteration.

    `method="colebrook"` (standard/rigorous): solves
    `1/sqrt(f) = -2*log10(relative_roughness/3.7 + 2.51/(Re*sqrt(f)))` via
    fixed-point iteration, bounded by `max_iterations`/`tolerance`. More
    accurate across the full turbulent range; the simplified method
    degrades at very rough pipe / low relative roughness combinations.

    :param reynolds_number: `rho * v * D / mu`.
    :param relative_roughness: `tubing_roughness / tubing_inner_diameter`,
        `0.0` for a smooth-pipe assumption.
    :param method: `"simplified"` or `"colebrook"`.
    :param laminar_reynolds_limit: Laminar/turbulent transition Reynolds
        number, `method="simplified"` only. `c.WELLBORE_LAMINAR_REYNOLDS_LIMIT`
        if `None`.
    :param turbulent_reynolds_limit: Blasius/Swamee-Jain transition
        Reynolds number, `method="simplified"` only.
        `c.WELLBORE_TURBULENT_REYNOLDS_LIMIT` if `None`.
    :param max_iterations: Fixed-point iteration cap, `method="colebrook"`
        only. `c.COLEBROOK_MAX_ITERATIONS` if `None`.
    :param tolerance: Absolute convergence tolerance on `f`,
        `method="colebrook"` only. `c.COLEBROOK_TOLERANCE` if `None`.
    :returns: Darcy friction factor (dimensionless).
    :raises ValidationError: If `reynolds_number` isn't positive, or
        `method` isn't recognized.
    """
    if reynolds_number <= 0:
        raise ValidationError(
            f"`reynolds_number` must be positive; got {reynolds_number}."
        )

    if method == "simplified":
        laminar_limit = (
            laminar_reynolds_limit
            if laminar_reynolds_limit is not None
            else c.WELLBORE_LAMINAR_REYNOLDS_LIMIT
        )
        turbulent_limit = (
            turbulent_reynolds_limit
            if turbulent_reynolds_limit is not None
            else c.WELLBORE_TURBULENT_REYNOLDS_LIMIT
        )
        if reynolds_number < laminar_limit:
            return 64.0 / reynolds_number
        if reynolds_number < turbulent_limit:
            return 0.316 * reynolds_number**-0.25
        return (
            0.25
            / (math.log10(relative_roughness / 3.7 + 5.74 / reynolds_number**0.9)) ** 2
        )

    if method == "colebrook":
        iteration_cap = (
            max_iterations if max_iterations is not None else c.COLEBROOK_MAX_ITERATIONS
        )
        abs_tolerance = tolerance if tolerance is not None else c.COLEBROOK_TOLERANCE
        friction_factor = 0.02  # initial guess
        for _ in range(iteration_cap):
            rhs = -2.0 * math.log10(
                relative_roughness / 3.7
                + 2.51 / (reynolds_number * math.sqrt(friction_factor))
            )
            updated = 1.0 / rhs**2
            if abs(updated - friction_factor) < abs_tolerance:
                friction_factor = updated
                break
            friction_factor = updated
        return friction_factor

    raise ValidationError(f"Unknown friction method {method!r}.")


def compute_segment_pressure_drop(
    *,
    length: Number,
    inclination_from_vertical: Number,
    tubing_inner_diameter: Number,
    tubing_roughness: typing.Optional[Number],
    mixture_density: Number,
    mixture_viscosity: Number,
    mixture_velocity_in: Number,
    mixture_velocity_out: Number,
    gravitational_acceleration: Number,
    friction_method: typing.Literal["simplified", "colebrook"] = "simplified",
    laminar_reynolds_limit: typing.Optional[Number] = None,
    turbulent_reynolds_limit: typing.Optional[Number] = None,
    friction_max_iterations: typing.Optional[int] = None,
    friction_tolerance: typing.Optional[Number] = None,
) -> PressureDropComponents:
    """
    Pressure drop across one tubing segment of length `length`.

    Returns the three components unsigned (magnitudes) - the caller
    combines them with whatever sign convention its integration direction
    and flow direction require. `MechanisticWellboreModel` applies
    hydrostatic/acceleration by geometric position and friction by flow
    direction (opposing it), since the two aren't the same thing once an
    injector's flow direction is considered - see
    `WellboreModel.perforation_pressures`'s `is_injector` parameter.

    :param length: Segment length (along-wellbore, not vertical depth).
    :param inclination_from_vertical: `0` = vertical, `pi/2` = horizontal.
    :param tubing_inner_diameter: Tubing inner diameter.
    :param tubing_roughness: Tubing roughness, may be `None` (smooth pipe).
    :param mixture_density: From `compute_mixture_density`, or an
        in-situ/holdup-corrected density for a slip correlation.
    :param mixture_viscosity: From `compute_mixture_viscosity`.
    :param mixture_velocity_in: Superficial velocity entering the segment.
    :param mixture_velocity_out: Superficial velocity leaving the segment.
    :param gravitational_acceleration: Passed explicitly - use
        `bores.constants.c.ACCELERATION_DUE_TO_GRAVITY_*` for the caller's
        unit system.
    :param friction_method: See `compute_friction_factor`.
    :param laminar_reynolds_limit: Forwarded to `compute_friction_factor`.
    :param turbulent_reynolds_limit: Forwarded to `compute_friction_factor`.
    :param friction_max_iterations: Forwarded to `compute_friction_factor`
        as `max_iterations`.
    :param friction_tolerance: Forwarded to `compute_friction_factor` as
        `tolerance`.
    :returns: `PressureDropComponents` for this segment.

    **Unit note, caught by testing this against FIELD-unit numbers, not
    assumed correct:** `hydrostatic = density * g * length` does not land
    in psi when `density` is `lbm/ft3` and `g` is `ft/s2` - FIELD-unit
    convention needs an additional `/ (gc * 144)`, where `gc =
    bores.constants.c.GRAVITATIONAL_CONSTANT_LBM_FT_PER_LBF_S2`. Not
    applied here: the formula is implemented exactly as specified, with
    unit-system consistency left to the caller.
    """
    hydrostatic = (
        mixture_density
        * gravitational_acceleration
        * length
        * math.cos(inclination_from_vertical)
    )

    mean_velocity = 0.5 * (mixture_velocity_in + mixture_velocity_out)
    relative_roughness = (
        tubing_roughness / tubing_inner_diameter
        if tubing_roughness is not None
        else 0.0
    )
    reynolds_number = (
        mixture_density * abs(mean_velocity) * tubing_inner_diameter / mixture_viscosity
    )
    if reynolds_number <= 0.0:
        friction = 0.0
    else:
        friction_factor = compute_friction_factor(
            reynolds_number,
            relative_roughness,
            method=friction_method,
            laminar_reynolds_limit=laminar_reynolds_limit,
            turbulent_reynolds_limit=turbulent_reynolds_limit,
            max_iterations=friction_max_iterations,
            tolerance=friction_tolerance,
        )
        friction = (
            friction_factor
            * (length / tubing_inner_diameter)
            * (mixture_density * mean_velocity**2 / 2.0)
        )

    acceleration = (
        mixture_density * (mixture_velocity_out**2 - mixture_velocity_in**2) / 2.0
    )
    return PressureDropComponents(
        hydrostatic=hydrostatic, friction=friction, acceleration=acceleration
    )


def compute_static_hydrostatic_drop(
    mixture_density: Number, length: Number, gravitational_acceleration: Number
) -> PressureDropComponents:
    """
    No-flow special case (shut-in well, or a well evaluated at zero rate)
    - hydrostatic term only; `friction`/`acceleration` are exactly zero
    rather than computed-and-happening-to-be-zero (avoids a divide-by-zero
    Reynolds number at `velocity=0`).

    :param mixture_density: Static column density.
    :param length: Vertical length.
    :param gravitational_acceleration: See `compute_segment_pressure_drop`.
    :returns: `PressureDropComponents` with `friction=0`, `acceleration=0`.
    """
    return PressureDropComponents(
        hydrostatic=mixture_density * gravitational_acceleration * length,
        friction=0.0,
        acceleration=0.0,
    )


def static_mixture_density(sample: ConnectionSample) -> Number:
    """
    Saturation-weighted density for the no-flow case.

    `compute_mixture_density` is rate-weighted and undefined at zero rate;
    the static column instead reflects whatever's in place, so this weights
    by `sample.phase_saturations` instead.

    :param sample: Reservoir sample to derive a static density from.
    :returns: Saturation-weighted density.
    :raises ValidationError: If `sample.phase_saturations` sums to zero.
    """
    total_saturation = np.sum(sample.phase_saturations.values())  # type: ignore[arg-type]
    if total_saturation == 0.0:
        raise ValidationError(
            f"ConnectionSample at cell_index={sample.cell_index} has zero "
            "total phase saturation; cannot derive a static column density."
        )
    return (
        sum(
            saturation * sample.phase_densities[phase]
            for phase, saturation in sample.phase_saturations.items()
        )
        / total_saturation
    )


class WellboreModel(typing.Protocol):
    """
    A well bore hydraulic model.

    This is the interface every wellbore hydraulics strategy implements.
    """

    def perforation_pressures(
        self,
        well: Well,
        reference_pressure: Number,
        phase_rates: typing.Mapping[FluidPhase, Number],
        perforation_indices: typing.Sequence[PerforationIndex],
        connection_samples: typing.Sequence[ConnectionSample],
        is_injector: bool,
    ) -> NumberArray[OneDimension]:
        """
        Computes flowing pressures, one per connection, same
        order as `perforation_indices`/`connection_samples`.

        :param well: Static well data (depths, tubing geometry).
        :param reference_pressure: Pressure at `well.reference_depth` (the
            BHP) to integrate from.
        :param phase_rates: Well-total phase rates driving friction/mixture
            density.
        :param perforation_indices: One `PerforationIndex` per connection
            (a `Perforation` spanning multiple cells contributes multiple
            entries here), supplying each connection's `representative_depth`
            and `cell_index`. Same order as `connection_samples`.
        :param connection_samples: One `ConnectionSample` per connection,
            same order as `perforation_indices`.
        :param is_injector: Flow direction - a producer's flow moves from
            each connection toward `reference_depth`; an injector's moves
            from `reference_depth` toward each connection. Friction
            opposes flow direction, not geometric position, so this
            changes the sign friction is applied with, independent of
            whether a given connection sits above or below
            `reference_depth`.
        :returns: Array of flowing pressures, one per connection, same
            order as `perforation_indices`/`connection_samples`.
        """
        ...

    def tubing_head_pressure(
        self,
        well: Well,
        reference_pressure: Number,
        phase_rates: typing.Mapping[FluidPhase, Number],
        surface_fluid_properties: SurfaceFluidProperties,
        is_injector: bool,
    ) -> Number:
        """
        Computes tubing head pressure.
        
        :param well: Static well data.
        :param reference_pressure: BHP at `well.reference_depth`.
        :param phase_rates: Well-total phase rates.
        :param surface_fluid_properties: Density/viscosity at surface
            conditions.
        :param is_injector: See `perforation_pressures`.
        :returns: THP, integrating from `reference_depth` up to surface
            (depth `0`).
        """
        ...
