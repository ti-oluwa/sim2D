"""Wellbore pressure-drop / hydraulics.

Depends on `wells.data`, `wells.state` (`ConnectionSample`). Implements D6.

`WellboreModel` is a `typing.Protocol` on purpose: `control_engine.py` only
ever calls `perforation_pressures`/`tubing_head_pressure` through this
interface, never checks which concrete class it has. `MechanisticWellboreModel`
is the only implementation built here (v1, no-slip homogeneous mixture,
vertical wells); a future slip-model (`HagedornBrownWellboreModel`) or
tabulated (`VFPTableWellboreModel`) implementation is a new class
satisfying the same protocol, not a change to this module or to
`control_engine.py`.
"""

import math
import typing

import attrs
import numpy as np

from bores.constants import c
from bores.errors import ValidationError
from bores.serde.base import Serializable
from bores.typing import FluidPhase, Number, NumberArray, OneDimension
from bores.wells.data import WellSpec
from bores.wells.states import ConnectionSample, as_phase_mapping

__all__ = [
    "PressureDropComponents",
    "SurfaceFluidProperties",
    "compute_mixture_density",
    "compute_mixture_viscosity",
    "compute_mixture_velocity",
    "compute_friction_factor",
    "compute_segment_pressure_drop",
    "compute_static_hydrostatic_drop",
    "WellboreModel",
    "MechanisticWellboreModel",
]

# v1 friction-correlation regime boundaries (Darcy friction factor,
# "simplified" method). Named rather than inline so a future correlation
# swap only touches these two lines.
_LAMINAR_REYNOLDS_LIMIT = 2300.0
_BLASIUS_REYNOLDS_LIMIT = 1.0e5

# Colebrook-White fixed-point iteration bounds (method="colebrook").
_COLEBROOK_MAX_ITERATIONS = 50
_COLEBROOK_TOLERANCE = 1.0e-10


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
    """Density/viscosity of the produced/injected stream at surface
    conditions, for `WellboreModel.tubing_head_pressure` - there's no
    reservoir cell at surface to pull a `ConnectionSample` from.
    """

    density: Number
    viscosity: Number


def compute_mixture_density(
    phase_rates: typing.Mapping[FluidPhase, Number],
    phase_densities: typing.Mapping[FluidPhase, Number],
) -> Number:
    """
    v1 (no-slip). Rate-weighted mixture density:
    `sum(rate_p * density_p) / sum(rate_p)` across phases present in
    `phase_rates`.

    :param phase_rates: Rates per phase, consistent condition with
        `phase_densities`.
    :param phase_densities: Density per phase, same condition as
        `phase_rates`. A phase present in `phase_rates` with rate `0.0`
        doesn't need an entry here (e.g. a 2-phase oil-water well with no
        gas PVT data) - only phases with nonzero rate are looked up.
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
    v1 (no-slip). Rate-weighted mixture viscosity, same formula/convention
    as `compute_mixture_density`.

    Not present in the original plan sketch - added because
    `compute_segment_pressure_drop` needs a `mixture_viscosity` input and
    no function computed one; same no-slip rate-weighting as density is the
    natural analogue, so this mirrors `compute_mixture_density` rather than
    inventing a different formula.

    :param phase_rates: Rates per phase.
    :param phase_viscosities: Viscosity per phase, same condition as
        `phase_rates`. Same zero-rate-phase tolerance as
        `compute_mixture_density`.
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
    v1 (no-slip). Superficial mixture velocity:
    `sum(rate_p) / cross_sectional_area`.

    :param phase_rates: Rates per phase, reservoir condition (already
        converted from surface rate * FVF by the caller - no PVT lookups
        here, per D7).
    :param tubing_inner_diameter: `WellSpec.tubing_inner_diameter`.
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
) -> Number:
    """
    Darcy friction factor for a tubing segment.

    `method="simplified"` (v1, default): `64/Re` for `Re < 2300` (laminar);
    Blasius `0.316 * Re**-0.25` for `2300 <= Re < 1e5`; a Swamee-Jain-style
    explicit approximation (includes `relative_roughness`) for `Re >= 1e5`.
    Closed-form, no iteration.

    `method="colebrook"` (standard/rigorous): solves
    `1/sqrt(f) = -2*log10(relative_roughness/3.7 + 2.51/(Re*sqrt(f)))` via
    fixed-point iteration. More accurate across the full turbulent range;
    the v1 default degrades at very rough pipe / low relative roughness
    combinations.

    :param reynolds_number: `rho * v * D / mu`.
    :param relative_roughness: `tubing_roughness / tubing_inner_diameter`,
        `0.0` for a smooth-pipe assumption.
    :param method: `"simplified"` or `"colebrook"`.
    :returns: Darcy friction factor (dimensionless).
    :raises ValidationError: If `reynolds_number` isn't positive, or
        `method` isn't recognized.
    """
    if reynolds_number <= 0:
        raise ValidationError(
            f"`reynolds_number` must be positive; got {reynolds_number}."
        )

    if method == "simplified":
        if reynolds_number < _LAMINAR_REYNOLDS_LIMIT:
            return 64.0 / reynolds_number
        if reynolds_number < _BLASIUS_REYNOLDS_LIMIT:
            return 0.316 * reynolds_number**-0.25
        return (
            0.25
            / (math.log10(relative_roughness / 3.7 + 5.74 / reynolds_number**0.9)) ** 2
        )

    if method == "colebrook":
        friction_factor = 0.02  # initial guess
        for _ in range(_COLEBROOK_MAX_ITERATIONS):
            rhs = -2.0 * math.log10(
                relative_roughness / 3.7
                + 2.51 / (reynolds_number * math.sqrt(friction_factor))
            )
            updated = 1.0 / rhs**2
            if abs(updated - friction_factor) < _COLEBROOK_TOLERANCE:
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
) -> PressureDropComponents:
    """
    Pressure drop across one tubing segment of length `length`.

    Friction's sign follows the `+dz`-going-down integration convention
    used by `MechanisticWellboreModel.perforation_pressures` (matches
    producer flow, where integrating downward from the reference pressure
    against the upward flow direction recovers frictional loss as well as
    hydrostatic head). Not signed for injector flow direction in v1 -
    flagged here rather than silently assumed correct for both.

    :param length: Segment length (along-wellbore, not vertical depth).
    :param inclination_from_vertical: `0` = vertical, `pi/2` = horizontal.
        `0.0` for every v1 segment (vertical-well limitation, D6).
    :param tubing_inner_diameter: From `WellSpec`.
    :param tubing_roughness: From `WellSpec`, may be `None` (smooth pipe).
    :param mixture_density: From `compute_mixture_density`.
    :param mixture_viscosity: From `compute_mixture_viscosity`.
    :param mixture_velocity_in: Superficial velocity entering the segment.
    :param mixture_velocity_out: Superficial velocity leaving the segment.
    :param gravitational_acceleration: Passed explicitly - use
        `bores.constants.c.ACCELERATION_DUE_TO_GRAVITY_*` for the caller's
        unit system, never a bare literal.
    :param friction_method: See `compute_friction_factor`.
    :returns: `PressureDropComponents` for this segment.

    **Unit note, caught by testing this against FIELD-unit numbers, not
    assumed correct:** `hydrostatic = density * g * length` does not land
    in psi when `density` is `lbm/ft3` and `g` is `ft/s2` - FIELD-unit
    convention needs an additional `/ (gc * 144)`, where `gc =
    bores.constants.c.GRAVITATIONAL_CONSTANT_LBM_FT_PER_LBF_S2`. Not
    applied here: the formula is implemented exactly as specified, with
    unit-system consistency left to the caller per that same design
    decision - this note exists so that responsibility is visible rather
    than discovered later.
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
            reynolds_number, relative_roughness, method=friction_method
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


def _static_mixture_density(sample: ConnectionSample) -> Number:
    """
    Saturation-weighted density for the no-flow case.

    `compute_mixture_density` is rate-weighted and undefined at zero rate;
    the static column instead reflects whatever's in place, so this weights
    by `sample.phase_saturations` instead. Not in the original plan sketch
    - needed to actually implement the no-flow branch of
    `MechanisticWellboreModel.perforation_pressures`.
    """
    total_saturation = sum(sample.phase_saturations.values())
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
    Contract every wellbore hydraulics strategy implements. v1
    (`MechanisticWellboreModel`) and any future v2/v3 satisfy this same
    signature - `control_engine.py` never branches on which one it has.
    """

    def perforation_pressures(
        self,
        well: WellSpec,
        reference_pressure: Number,
        phase_rates: typing.Mapping[FluidPhase, Number],
        connection_samples: typing.Sequence[ConnectionSample],
    ) -> NumberArray[OneDimension]:
        """
        :param well: Static well data (depths, tubing geometry).
        :param reference_pressure: Pressure at `well.reference_depth` (the
            BHP) to integrate from.
        :param phase_rates: Well-total phase rates driving friction/mixture
            density.
        :param connection_samples: One `ConnectionSample` per open
            perforation, `well.open_perforations` order.
        :returns: Array of flowing pressures, one per open perforation,
            same order as `connection_samples`.
        """
        ...

    def tubing_head_pressure(
        self,
        well: WellSpec,
        reference_pressure: Number,
        phase_rates: typing.Mapping[FluidPhase, Number],
        surface_fluid_properties: SurfaceFluidProperties,
    ) -> Number:
        """
        :param well: Static well data.
        :param reference_pressure: BHP at `well.reference_depth`.
        :param phase_rates: Well-total phase rates.
        :param surface_fluid_properties: Density/viscosity at surface
            conditions.
        :returns: THP, integrating from `reference_depth` up to surface
            (depth `0`).
        """
        ...


@attrs.frozen(kw_only=True, slots=True)
class MechanisticWellboreModel(Serializable):
    """
    v1 `WellboreModel` - segmented no-slip mixture model (D6). One segment
    per (reference_depth -> perforation) pair, single straight segment each
    (v1 assumes vertical wells, no intermediate discretization needed for a
    single-segment integral to be accurate). A future multi-segment
    (`WELSEGS`) extension subdivides each segment; it doesn't change this
    class's shape.
    """

    friction_method: typing.Literal["simplified", "colebrook"] = "simplified"
    gravitational_acceleration: Number = (
        c.ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE
    )
    """FIELD-unit default. A `METRIC`/`SI` well needs the matching
    `bores.constants.c.ACCELERATION_DUE_TO_GRAVITY_*` value supplied
    explicitly by whatever constructs this (a future `factories.py`) -
    never silently assumed for a non-FIELD well."""

    def perforation_pressures(
        self,
        well: WellSpec,
        reference_pressure: Number,
        phase_rates: typing.Mapping[FluidPhase, Number],
        connection_samples: typing.Sequence[ConnectionSample],
    ) -> NumberArray[OneDimension]:
        """See `WellboreModel.perforation_pressures`."""
        open_perforations = well.open_perforations
        if len(connection_samples) != len(open_perforations):
            raise ValidationError(
                f"len(connection_samples)={len(connection_samples)} != "
                f"len(well.open_perforations)={len(open_perforations)} for "
                f"well {well.name!r}."
            )

        total_rate = sum(phase_rates.values())
        pressures = np.empty(len(open_perforations), dtype=float)

        for i, (perforation, sample) in enumerate(
            zip(open_perforations, connection_samples)
        ):
            representative_depth = 0.5 * (
                perforation.top_depth + perforation.bottom_depth
            )
            dz = representative_depth - well.reference_depth

            if total_rate == 0.0:
                drop = compute_static_hydrostatic_drop(
                    mixture_density=_static_mixture_density(sample),
                    length=abs(dz),
                    gravitational_acceleration=self.gravitational_acceleration,
                )
            else:
                mixture_density = compute_mixture_density(
                    phase_rates, sample.phase_densities
                )
                mixture_viscosity = compute_mixture_viscosity(
                    phase_rates, sample.phase_viscosities
                )
                velocity = compute_mixture_velocity(
                    phase_rates, well.tubing_inner_diameter
                )
                drop = compute_segment_pressure_drop(
                    length=abs(dz),
                    inclination_from_vertical=0.0,
                    tubing_inner_diameter=well.tubing_inner_diameter,
                    tubing_roughness=well.tubing_roughness,
                    mixture_density=mixture_density,
                    mixture_viscosity=mixture_viscosity,
                    mixture_velocity_in=velocity,
                    mixture_velocity_out=velocity,
                    gravitational_acceleration=self.gravitational_acceleration,
                    friction_method=self.friction_method,
                )

            sign = 1.0 if dz >= 0 else -1.0
            pressures[i] = reference_pressure + sign * drop.total

        return pressures

    def tubing_head_pressure(
        self,
        well: WellSpec,
        reference_pressure: Number,
        phase_rates: typing.Mapping[FluidPhase, Number],
        surface_fluid_properties: SurfaceFluidProperties,
    ) -> Number:
        """See `WellboreModel.tubing_head_pressure`."""
        dz = 0.0 - well.reference_depth  # negative: surface is above reference_depth
        total_rate = sum(phase_rates.values())

        if total_rate == 0.0:
            drop = compute_static_hydrostatic_drop(
                mixture_density=surface_fluid_properties.density,
                length=abs(dz),
                gravitational_acceleration=self.gravitational_acceleration,
            )
        else:
            velocity = compute_mixture_velocity(phase_rates, well.tubing_inner_diameter)
            drop = compute_segment_pressure_drop(
                length=abs(dz),
                inclination_from_vertical=0.0,
                tubing_inner_diameter=well.tubing_inner_diameter,
                tubing_roughness=well.tubing_roughness,
                mixture_density=surface_fluid_properties.density,
                mixture_viscosity=surface_fluid_properties.viscosity,
                mixture_velocity_in=velocity,
                mixture_velocity_out=velocity,
                gravitational_acceleration=self.gravitational_acceleration,
                friction_method=self.friction_method,
            )

        # Integrating up (dz < 0): subtract the drop rather than add it.
        return reference_pressure - drop.total
